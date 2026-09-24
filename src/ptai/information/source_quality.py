"""
Source quality - credibility, novelty, corroboration and recency, all computed.

Why this module existed in name only
------------------------------------
Every number it returned was a constant or a proxy for something else:

    novelty = 0.6            # placeholder
    recency = 0.7            # never looked at a timestamp
    corroborated = len(other_sources) >= 2

That last one is the dangerous one. Corroboration means independent sources
saying the same thing. Counting how many other sources exist says nothing about
whether they agree - three sources on unrelated topics returned
`corroborated=True` and pushed the quality score up by 0.14. And novelty was a
flat 0.6 for a brand-new story and for the tenth copy of the same wire report,
so the module could not tell original reporting from syndication, which is the
one thing a novelty score is for.

There was also a credibility lookup that substring-matched the whole URL, so
`http://scam.example/?ref=reuters.com` scored as Reuters.

What it does now
----------------
All four dimensions are computed from actual input:

  credibility  registrable-domain tier lookup, not a URL substring match.
               Unknown domains score low with the reason recorded, rather than
               defaulting to a confident-looking 0.5.
  novelty      1 - the highest content overlap with anything already seen.
               Original reporting scores high; the tenth syndicated copy of a
               wire story scores near zero, which is correct.
  corroborated a real content-overlap test against other documents, with the
               matching sources named. Counting sources is not corroboration.
  recency      exponential decay from an actual publication timestamp. No
               timestamp means an unknown publication date, which is scored
               as stale-but-not-zero and flagged, not treated as fresh.

The weights are explicit and the inputs are returned with the score, so a
caller can see which dimension is carrying the result instead of trusting a
single number that used to be mostly constants.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

# Two-label public suffixes that must not be split. Not exhaustive - the goal
# is to avoid scoring "news.co.uk" as domain "uk", not to replicate the full
# public suffix list.
# A hostname is dot-separated labels of letters, digits and hyphens.
_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")

MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au",
    "co.nz", "co.jp", "or.jp", "ne.jp", "com.br", "com.mx", "co.za",
    "com.sg", "com.hk", "co.in", "com.tr", "com.ar",
}

# Credibility tiers. Ordered so the most specific match wins.
#
# These are editorial judgements about source TYPE, not guarantees about any
# individual story - a wire service can still be wrong. They are kept
# deliberately coarse because pretending to finer precision than "this is a
# wire service" would be the same kind of invention this rewrite removes.
CREDIBILITY_TIERS: Tuple[Tuple[str, str, float], ...] = (
    # wire services and public broadcasters: primary reporting, corrections policy
    ("wire_service", "reuters.com", 0.95),
    ("wire_service", "apnews.com", 0.95),
    ("wire_service", "afp.com", 0.92),
    ("public_broadcaster", "bbc.co.uk", 0.90),
    ("public_broadcaster", "bbc.com", 0.90),
    ("public_broadcaster", "npr.org", 0.88),
    ("public_broadcaster", "pbs.org", 0.88),
    ("financial_press", "bloomberg.com", 0.90),
    ("financial_press", "ft.com", 0.90),
    ("financial_press", "wsj.com", 0.88),
    ("financial_press", "economist.com", 0.87),
    ("newspaper_of_record", "nytimes.com", 0.85),
    ("newspaper_of_record", "washingtonpost.com", 0.83),
    ("newspaper_of_record", "theguardian.com", 0.83),
    ("government", "gov", 0.90),
    ("government", "mil", 0.88),
    ("international_body", "un.org", 0.88),
    ("international_body", "imf.org", 0.88),
    ("international_body", "worldbank.org", 0.86),
    ("regulator", "sec.gov", 0.95),
    ("regulator", "federalreserve.gov", 0.95),
    ("regulator", "ecb.europa.eu", 0.93),
    ("data_provider", "coingecko.com", 0.75),
    ("data_provider", "coinmarketcap.com", 0.72),
    ("exchange", "binance.com", 0.80),
    ("exchange", "coinbase.com", 0.80),
    ("exchange", "deribit.com", 0.78),
    ("aggregator", "google.com", 0.55),
    ("social", "twitter.com", 0.40),
    ("social", "x.com", 0.40),
    ("social", "reddit.com", 0.35),
    ("social", "t.me", 0.30),
    ("social", "discord.com", 0.25),
    ("user_generated", "medium.com", 0.35),
    ("user_generated", "substack.com", 0.35),
    ("user_generated", "blogspot.com", 0.25),
    ("user_generated", "wordpress.com", 0.25),
)

# An unknown domain is not 0.5. Half implies "average, measured"; unknown means
# unmeasured, and the caller needs to be able to tell those apart.
UNKNOWN_DOMAIN_CREDIBILITY = 0.30

# Recency: half-life in hours. News value roughly halves per day for event
# markets, so a 24h half-life is a reasonable default and is configurable.
RECENCY_HALF_LIFE_HOURS = 24.0

# No timestamp at all: the document may be fresh or years old, so it gets a
# low-but-not-zero recency and an explicit flag. Assuming 0.7 was the bug.
NO_TIMESTAMP_RECENCY = 0.30

# Content overlap above which two documents are treated as covering the same
# story. Set high: a false "corroboration" inflates trust in a single source.
CORROBORATION_SIMILARITY = 0.30

# Overlap above which a document is considered a derivative of one already seen
# rather than original reporting.
NOVELTY_DERIVATIVE_THRESHOLD = 0.45

DEFAULT_WEIGHTS = {"credibility": 0.40, "novelty": 0.20,
                   "corroboration": 0.20, "recency": 0.20}


@dataclass
class SourceDocument:
    """One retrieved document with the metadata needed to score it."""
    url: str
    content: str = ""
    published_at: Optional[datetime] = None
    title: str = ""
    # Set by the engine so a document can be compared against the rest.
    domain: str = ""
    tier: str = "unknown"


@dataclass
class SourceAssessment:
    """Scored source, with the inputs that produced each component."""
    url: str
    domain: str
    tier: str
    credibility: float
    novelty: float
    corroborated: bool
    recency: float
    overall_quality: float
    # Diagnostics: without these a caller cannot tell which dimension is
    # carrying the score, which is how constants went unnoticed.
    corroborating_sources: List[str] = field(default_factory=list)
    most_similar_source: str = ""
    max_similarity: float = 0.0
    has_timestamp: bool = False
    age_hours: Optional[float] = None
    content_words: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A source with no content or an unknown domain is not evidence."""
        return self.content_words > 0 and self.tier != "unknown"


class SourceQualityEngine:
    """
    Scores sources on four independently computed dimensions.

    Documents should be added via `add_document` as they are retrieved so that
    novelty and corroboration have a corpus to compare against. Calling
    `assess` on a lone document still works - novelty is 1.0 and corroboration
    is false - but that is the honest answer for a single source, not a
    failure to compute.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 half_life_hours: float = RECENCY_HALF_LIFE_HOURS,
                 extra_credibility: Optional[Dict[str, float]] = None,
                 now: Optional[datetime] = None):
        """
        `now` is injectable so recency is testable without waiting.
        `extra_credibility` lets a deployment add its own trusted domains
        rather than editing the tier table.
        """
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        # normalise so the overall score stays in [0, 1] whatever weights are used
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.half_life_hours = half_life_hours
        self.extra_credibility = dict(extra_credibility or {})
        self._now = now
        self.documents: List[SourceDocument] = []

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @property
    def now(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Domain and credibility
    # ------------------------------------------------------------------

    @staticmethod
    def registrable_domain(url: str) -> str:
        """
        Extract the registrable domain from a URL.

        Matches on the parsed host, never on the whole URL - the old substring
        match meant `http://scam.example/?ref=reuters.com` scored as Reuters.
        """
        if not url:
            return ""
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower().strip(".")
        if not host:
            return ""
        # A hostname with a space, or with no dot and no plausible TLD, is not
        # a domain - returning it would let free text score as a source.
        if " " in host or not _HOST_RE.match(host):
            return ""
        if host.startswith("www."):
            host = host[4:]
        labels = host.split(".")
        if len(labels) < 2:
            return host
        last_two = ".".join(labels[-2:])
        if last_two in MULTI_PART_SUFFIXES and len(labels) >= 3:
            return ".".join(labels[-3:])
        return last_two

    def credibility_for(self, url: str) -> Tuple[float, str, str]:
        """
        Credibility score, tier name, and the reason.

        An unmatched domain returns the low unknown score with the reason
        stated, rather than a 0.5 that looks like a measurement.
        """
        domain = self.registrable_domain(url)
        if not domain:
            return 0.0, "invalid", "could not parse a domain from this URL"

        for needle, cred in self.extra_credibility.items():
            if domain == needle or domain.endswith("." + needle):
                return float(cred), "configured", f"deployment-configured trust for {domain}"

        for tier, needle, cred in CREDIBILITY_TIERS:
            if domain == needle or domain.endswith("." + needle):
                return cred, tier, f"{domain} is a known {tier.replace('_', ' ')}"

        return (UNKNOWN_DOMAIN_CREDIBILITY, "unknown",
                f"{domain} is not in any credibility tier - scored low, not average")

    # ------------------------------------------------------------------
    # Content handling
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]{2,}", (text or "").lower())

    @staticmethod
    def _shingles(text: str, k: int = 5) -> set:
        """Word k-shingles. More robust than a bag of words for near-duplicates."""
        tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

    def similarity(self, a: str, b: str) -> float:
        """Jaccard similarity over word shingles, in [0, 1]."""
        sa, sb = self._shingles(a), self._shingles(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def add_document(self, url: str, content: str = "",
                     published_at: Optional[datetime] = None,
                     title: str = "") -> SourceDocument:
        """Register a document so later ones can be compared against it."""
        doc = SourceDocument(url=url, content=content or "", published_at=published_at,
                             title=title or "", domain=self.registrable_domain(url))
        doc.tier = self.credibility_for(url)[1]
        self.documents.append(doc)
        return doc

    def _others(self, url: str) -> Iterable[SourceDocument]:
        return [d for d in self.documents if d.url != url]

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def novelty_of(self, content: str, url: str = "") -> Tuple[float, str, float]:
        """
        Novelty = 1 - the highest overlap with anything already seen.

        A wire story republished ten times should score near zero on the tenth
        copy. The old flat 0.6 could not distinguish original reporting from
        syndication, which is the only thing a novelty score is for.
        """
        if not content.strip():
            return 0.0, "", 0.0
        best_score, best_url = 0.0, ""
        for doc in self._others(url):
            if not doc.content.strip():
                continue
            sim = self.similarity(content, doc.content)
            if sim > best_score:
                best_score, best_url = sim, doc.url
        if best_score >= NOVELTY_DERIVATIVE_THRESHOLD:
            return round(max(0.0, 1.0 - best_score), 4), best_url, round(best_score, 4)
        return round(max(0.0, 1.0 - best_score), 4), best_url, round(best_score, 4)

    def corroboration_of(self, content: str, url: str = ""
                         ) -> Tuple[bool, List[str], float]:
        """
        Whether independent sources cover the same content.

        Requires actual overlap, and excludes sources on the same registrable
        domain - two pages from one outlet are not two witnesses. The old
        version returned True whenever two other sources existed, whatever
        they said.
        """
        if not content.strip():
            return False, [], 0.0
        own_domain = self.registrable_domain(url)
        matches: List[str] = []
        best = 0.0
        for doc in self._others(url):
            if not doc.content.strip():
                continue
            if own_domain and doc.domain == own_domain:
                continue          # same outlet is not independent
            sim = self.similarity(content, doc.content)
            best = max(best, sim)
            if sim >= CORROBORATION_SIMILARITY:
                matches.append(doc.url)
        return bool(matches), matches, round(best, 4)

    def recency_of(self, published_at: Optional[datetime]
                   ) -> Tuple[float, bool, Optional[float]]:
        """
        Exponential decay from the publication timestamp.

        No timestamp is scored low and flagged. The old code returned 0.7 for
        every document regardless of when it was written, so a five-year-old
        blog post looked fresher than this morning's wire story.
        """
        if published_at is None:
            return NO_TIMESTAMP_RECENCY, False, None
        try:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (self.now - published_at).total_seconds() / 3600.0)
        except Exception:
            return NO_TIMESTAMP_RECENCY, False, None
        score = math.exp(-math.log(2.0) * age_hours / self.half_life_hours)
        return round(score, 4), True, round(age_hours, 3)

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self, url: str, content: str = "",
               other_sources: Optional[List[str]] = None,
               published_at: Optional[datetime] = None) -> SourceAssessment:
        """
        Score one source.

        `other_sources` is accepted for backward compatibility but is only
        used to seed the corpus when those documents have content. A bare list
        of URLs cannot corroborate anything - there is nothing to compare -
        and saying so is the point.
        """
        warnings: List[str] = []
        credibility, tier, cred_reason = self.credibility_for(url)
        if tier == "unknown":
            warnings.append(cred_reason)
        elif tier == "invalid":
            warnings.append(cred_reason)

        # Register this document so future assessments can compare against it,
        # and so a repeated call does not compare the document with itself.
        if not any(d.url == url for d in self.documents):
            self.add_document(url, content, published_at, "")

        novelty, most_similar, max_sim = self.novelty_of(content, url)
        if max_sim >= NOVELTY_DERIVATIVE_THRESHOLD:
            warnings.append(f"near-duplicate of {most_similar} "
                            f"({max_sim*100:.0f}% overlap) - likely syndication")

        corroborated, matches, best_sim = self.corroboration_of(content, url)
        if not corroborated:
            if other_sources:
                warnings.append(
                    f"{len(other_sources)} other source(s) supplied but none shares "
                    f"content above {CORROBORATION_SIMILARITY*100:.0f}% - a URL list "
                    f"is not corroboration")
            else:
                warnings.append("no independent source covers this content")

        recency, has_ts, age_hours = self.recency_of(published_at)
        if not has_ts:
            warnings.append("no publication timestamp - recency assumed stale")

        words = len(self._tokens(content))
        if words == 0:
            warnings.append("no content supplied - credibility alone is not evidence")

        corroboration_score = 1.0 if corroborated else (
            0.3 if best_sim > 0 else 0.0)

        overall = (credibility * self.weights["credibility"]
                   + novelty * self.weights["novelty"]
                   + corroboration_score * self.weights["corroboration"]
                   + recency * self.weights["recency"])

        return SourceAssessment(
            url=url, domain=self.registrable_domain(url), tier=tier,
            credibility=round(credibility, 4), novelty=round(novelty, 4),
            corroborated=corroborated, recency=round(recency, 4),
            overall_quality=round(overall, 4),
            corroborating_sources=matches, most_similar_source=most_similar,
            max_similarity=max_sim, has_timestamp=has_ts, age_hours=age_hours,
            content_words=words, warnings=warnings)

    def rank(self, urls: List[Tuple[str, str]],
             published: Optional[Dict[str, datetime]] = None) -> List[SourceAssessment]:
        """Score and rank a batch, best first. Register all documents first."""
        published = published or {}
        for url, content in urls:
            if not any(d.url == url for d in self.documents):
                self.add_document(url, content, published.get(url), "")
        scored = [self.assess(u, c, published_at=published.get(u)) for u, c in urls]
        return sorted(scored, key=lambda a: -a.overall_quality)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, object]:
        return {
            "engine": "Source Quality",
            "dimensions": {
                "credibility": ("registrable-domain tier lookup; unknown domains score "
                                f"{UNKNOWN_DOMAIN_CREDIBILITY} with a reason, not a "
                                "confident-looking 0.5"),
                "novelty": ("1 - highest shingle overlap with the corpus; a tenth "
                            "syndicated copy scores near zero"),
                "corroboration": (f"independent domains sharing >{CORROBORATION_SIMILARITY*100:.0f}% "
                                  "content; same-domain pages excluded, and a bare URL "
                                  "list cannot corroborate"),
                "recency": (f"exponential decay, {self.half_life_hours:.0f}h half-life; "
                            "a missing timestamp scores low and is flagged"),
            },
            "weights": dict(self.weights),
            "corpus_size": len(self.documents),
            "removed_placeholders": [
                "novelty was a hardcoded 0.6 for every document",
                "recency was a hardcoded 0.7 that never read a timestamp",
                "corroborated was len(other_sources) >= 2, which never compared content",
                "credibility substring-matched the whole URL, so ?ref=reuters.com scored as Reuters",
            ],
        }
