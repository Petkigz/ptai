"""
Web research - fetch real sources, score them, and separate bull from bear.

What was wrong
--------------
The researcher never researched. It accepted a `browser` argument and never
used it, issued no HTTP request, and returned `sources = ["heuristic"]`. What
it called a bull case was the question restated with the words "supporting
YES" in front of it:

    supporting_yes = f"Polls, betting odds, fundraising, endorsements
                      supporting YES for {question[:80]}"

That is a list of topics one might look into, presented in a field named
`supporting_yes` that the agent then treated as evidence. `confidence` was a
literal 0.6, or 0.65 when an LLM answered - the same number whatever was
found, including nothing.

The LLM path was worse than useless. It split a single block of free text by
character offset:

    supporting_yes = text[:200]
    supporting_no  = text[200:400]

so the "bear case" was whatever happened to start at character 200, which
could be the middle of a word in the middle of the bull argument.

What it does now
----------------
It fetches. Sources are retrieved over HTTP, their text extracted, and each
one scored by SourceQualityEngine for credibility, novelty, corroboration and
recency. The result reports which URLs were actually retrieved and which
failed, so a caller can tell a researched market from an unresearched one.

The LLM output is parsed by section marker rather than by offset. If a section
is missing, that section is reported as missing instead of being filled with
neighbouring text.

Confidence is derived from what was retrieved - how many sources, how credible,
whether both sides found support - rather than being a constant. A market with
no retrievable sources gets a low confidence and a `researched` flag of False,
which is the honest answer.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from loguru import logger

from ..betting.sports_data import BROWSER_USER_AGENT

from .source_quality import SourceAssessment, SourceQualityEngine

# A search endpoint that returns HTML and needs no key. Used only to discover
# candidate URLs; the substance comes from fetching the pages themselves.
DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"

# Sections the LLM must fill. Parsed by marker, never by character offset.
SECTION_MARKERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("supporting_yes", ("bull case", "for yes", "supporting yes", "yes case")),
    ("supporting_no", ("bear case", "against yes", "for no", "supporting no", "no case")),
    ("contradicting_both", ("contradiction", "contradicting", "both wrong")),
    ("unknown", ("unknown", "what we don't know", "don't know")),
    ("resolution_risks", ("resolution risk", "resolution", "ambigu")),
)

# How much of a fetched page to keep. Enough for evidence, small enough that a
# page cannot swamp the context.

_LABEL_RE = re.compile(
    r"^[\s\d\.\)\-\*#:]*(bull case|bear case|for yes|for no|"
    r"supporting yes|supporting no|against yes|yes case|no case|"
    r"contradiction|contradicting|both wrong|unknown|"
    r"what we don't know|don't know|resolution risks?|resolution|"
    r"ambiguity|ambiguities)[\s:\*\-]*", re.I)


_BULLET_RE = re.compile(r"^[\s\d\.\)\-\*#:]+")


def _match_marker(line: str) -> Tuple[Optional[str], str]:
    """
    Whether a line opens a labelled section, and the content after the label.

    Only a line that STARTS with a label counts. The previous approach tested
    whether the label appeared anywhere within the first 80 characters, which
    both mistook continuation lines for headers and silently dropped long
    lines that did open a section.
    """
    if not line or not line.strip():
        return None, ""
    stripped = _BULLET_RE.sub("", line.strip())
    low = stripped.lower()
    for name, aliases in SECTION_MARKERS:
        for alias in aliases:
            if low.startswith(alias):
                # strip repeatedly so "1. Bull case FOR YES: ..." does not
                # leave the "FOR YES:" fragment glued to the real answer
                tail = stripped
                for _ in range(4):
                    once = _strip_section_label(tail)
                    if once == tail:
                        break
                    tail = once
                return name, tail
    return None, ""



def _strip_section_label(text: str) -> str:
    """Remove one leading section label plus its punctuation."""
    return _LABEL_RE.sub("", text)

MAX_CHARS_PER_SOURCE = 4000


@dataclass
class FetchedSource:
    """One retrieved page with its quality score."""
    url: str
    status: int = 0
    ok: bool = False
    text: str = ""
    error: str = ""
    assessment: Optional[SourceAssessment] = None
    supports: str = "unknown"          # yes | no | both | neither | unknown
    published_at: Optional[datetime] = None

    @property
    def quality(self) -> float:
        return self.assessment.overall_quality if self.assessment else 0.0


@dataclass
class ResearchResult:
    market_id: str
    question: str
    supporting_yes: str
    supporting_no: str
    contradicting_both: str
    unknown: str
    resolution_risks: str
    final_summary: str
    sources: List[str]
    confidence: float
    time_spent_seconds: float
    # Everything below is new. Without it a caller cannot distinguish a market
    # that was actually researched from one where nothing could be fetched.
    researched: bool = False
    source_details: List[FetchedSource] = field(default_factory=list)
    sources_attempted: int = 0
    sources_retrieved: int = 0
    yes_evidence: int = 0
    no_evidence: int = 0
    avg_source_quality: float = 0.0
    used_llm: bool = False
    missing_sections: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: str = "web_fetch"


# How long to stop asking a search engine that just refused us. Long enough to
# cover a whole scan of 500-1000 markets, short enough that research comes back
# by itself when the network does.
SEARCH_COOLDOWN_SECONDS = 1800.0


class WebResearcher:
    """
    Researches a market from real sources.

    Fetching is injectable so the whole path is testable without a network:
    pass `fetch` as an async callable (url) -> (status, text).
    """

    def __init__(self, llm_router=None, browser=None, fetch: Optional[Callable] = None,
                 search_url: str = DUCKDUCKGO_HTML, max_sources: int = 6,
                 timeout: float = 12.0, source_engine: Optional[SourceQualityEngine] = None,
                 user_agent: str = BROWSER_USER_AGENT):
        """
        `browser` is retained for callers that pass one, and is now actually
        used when supplied - it is the preferred way to fetch, since a real
        browser handles the JavaScript and blocking that defeat plain HTTP.
        """
        self.llm_router = llm_router
        self.browser = browser
        self._fetch = fetch
        self.search_url = search_url
        self.max_sources = max_sources
        self.timeout = timeout
        self.source_engine = source_engine or SourceQualityEngine()
        self.user_agent = user_agent
        self.last_error: str = ""
        # Search that cannot be reached is remembered. The operator's log asked
        # DuckDuckGo once per market and waited out a 12 s connect timeout every
        # single time (and on one market a DNS failure, Errno 11001):
        #
        #   [research] fetch https://html.duckduckgo.com/... ConnectTimeout (12.0)
        #   Web research for 4910857 fetched nothing
        #   ...same line for 4910858, 4910859, 4910860, ... every 13 seconds
        #
        # Nothing about the seventh market is different from the first, so after
        # a search failure this stops asking for a while and says so once.
        self._search_unavailable_until: float = 0.0
        self._search_blocked_reason: str = ""

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    async def _fetch_one(self, url: str) -> Tuple[int, str]:
        """
        Retrieve one URL. Returns (status, text); (0, "") on failure.

        Order of preference: injected fetch (tests), a real browser if one was
        supplied, then plain HTTP. Every path degrades to (0, "") rather than
        raising, so one bad page cannot abort the research.
        """
        if self._fetch is not None:
            try:
                result = self._fetch(url)
                if asyncio.iscoroutine(result):
                    result = await result
                status, text = result
                return int(status), str(text or "")
            except Exception as e:
                self.last_error = f"fetch {url}: {type(e).__name__}"
                return 0, ""

        if self.browser is not None:
            try:
                get = getattr(self.browser, "get_page", None) or getattr(self.browser, "fetch", None)
                if get is not None:
                    result = get(url)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, tuple):
                        return int(result[0]), str(result[1] or "")
                    return 200, str(result or "")
            except Exception as e:
                logger.debug(f"[research] browser fetch failed for {url}: {e}")

        try:
            import requests
            r = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, timeout=self.timeout,
                                     headers={"User-Agent": self.user_agent}))
            if r.status_code != 200:
                return r.status_code, ""
            return 200, r.text
        except Exception as e:
            self.last_error = f"fetch {url}: {type(e).__name__}: {e}"
            logger.debug(f"[research] {self.last_error}")
            return 0, ""

    async def discover_sources(self, query: str) -> List[str]:
        """
        Find candidate URLs for a query.

        Uses a keyless HTML search endpoint. If discovery fails the caller
        still gets an honest empty list - it does not fall back to inventing
        URLs or to pretending the question itself was a source.
        """
        if time.time() < self._search_unavailable_until:
            self.last_error = self._search_blocked_reason
            return []
        status, html = await self._fetch_one(f"{self.search_url}?q={quote_plus(query)}")
        if status != 200 or not html:
            # One failure is a fact about the network, not about this market.
            self._search_unavailable_until = time.time() + SEARCH_COOLDOWN_SECONDS
            self._search_blocked_reason = (
                f"search is unreachable from this machine ({self.last_error[:120]}) - "
                f"not retrying for {int(SEARCH_COOLDOWN_SECONDS/60)} min")
            logger.warning(
                f"[research] {self._search_blocked_reason}. Markets will report "
                f"unresearched without the {int(SEARCH_COOLDOWN_SECONDS/60)}-minute "
                f"wait per market; research resumes on its own when search answers.")
            return []
        urls = re.findall(r'uddg=([^&"\']+)', html)
        if not urls:
            urls = re.findall(r'href="(https?://[^"]+)"', html)
        out: List[str] = []
        seen = set()
        from urllib.parse import unquote
        for u in urls:
            try:
                cand = unquote(u)
            except Exception:
                continue
            host = (urlparse(cand).hostname or "").lower()
            if not host or "duckduckgo" in host or "google" in host:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
            if len(out) >= self.max_sources:
                break
        return out

    # ------------------------------------------------------------------
    # Text handling
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(html: str) -> str:
        """
        Strip a page down to readable prose.

        Scripts, styles and navigation are removed first - leaving them in
        would let boilerplate and tracking code dominate the similarity
        comparison that novelty and corroboration depend on.
        """
        if not html:
            return ""
        text = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?is)<!--.*?-->", " ", text)
        text = re.sub(r"(?is)<(nav|footer|header|aside)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_CHARS_PER_SOURCE]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]

    # ------------------------------------------------------------------
    # Evidence classification
    # ------------------------------------------------------------------

    def _classify(self, text: str, question: str) -> str:
        """
        Which side of the market a page's content leans toward.

        Deliberately crude and deliberately allowed to answer "neither". A
        confident wrong classification is worse than an admitted one, because
        it would be counted as evidence on the wrong side.
        """
        if not text:
            return "unknown"
        low = text.lower()
        qlow = question.lower()
        # key nouns from the question, minus the usual framing words
        stop = {"will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "and",
                "or", "for", "is", "it", "that", "this", "what", "which", "who"}
        keys = [w for w in re.findall(r"[a-z0-9]{4,}", qlow) if w not in stop][:6]
        if not keys:
            return "neither"
        hits = sum(1 for k in keys if k in low)
        if hits < max(1, len(keys) // 3):
            return "neither"          # not about this market

        yes_hits = len(re.findall(r"\b(will|expected to|likely|set to|on track|"
                                  r"confirms?|announced|agreed|passed|won)\b", low))
        no_hits = len(re.findall(r"\b(will not|won't|unlikely|doubt|rules out|"
                                  r"denies|denied|rejected|failed|delayed|no plans)\b", low))
        if yes_hits > no_hits * 1.5:
            return "yes"
        if no_hits > yes_hits * 1.5:
            return "no"
        return "both"

    # ------------------------------------------------------------------
    # LLM section parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_sections(text: str) -> Dict[str, str]:
        """
        Split an LLM answer into its sections by marker.

        The previous version took `text[:200]` as the bull case and
        `text[200:400]` as the bear case, so the bear case was whatever
        happened to begin at character 200 - possibly mid-word, possibly
        still the bull argument.
        """
        if not text:
            return {}
        lines = text.split("\n")
        found: Dict[str, str] = {}
        current: Optional[str] = None
        buffer: List[str] = []

        def flush():
            if current is not None:
                body = " ".join(b for b in buffer if b.strip()).strip()
                if body and current not in found:
                    found[current] = body

        for line in lines:
            # A marker line must START with a section label (after any
            # numbering or bullet). Testing for the label merely appearing
            # anywhere means a continuation line like "still the bull case"
            # gets mistaken for a new header and swallows the section body.
            marker, tail = _match_marker(line)
            if marker:
                flush()
                current, buffer = marker, []
                if tail.strip():
                    buffer.append(tail)
            elif current is not None:
                buffer.append(line)
        flush()
        return found

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    async def research(self, market, max_time_seconds: int = 45,
                       candidate_urls: Optional[List[str]] = None) -> ResearchResult:
        """
        Research one market from real sources.

        `candidate_urls` lets a caller supply sources directly (for example
        from a news engine that already ran). Otherwise sources are discovered
        by search. Either way, nothing is fabricated: if no page can be
        retrieved the result says `researched=False` and explains why.
        """
        start = time.time()
        question = getattr(market, "question", str(market))
        warnings: List[str] = []

        urls = list(candidate_urls or [])
        if not urls:
            urls = await self.discover_sources(question)
            if not urls:
                if self._search_blocked_reason:
                    warnings.append(self._search_blocked_reason)
                else:
                    warnings.append(
                        "no sources could be discovered - nothing was fetched, so this "
                        "market was not researched")

        sources: List[FetchedSource] = []
        deadline = start + max_time_seconds
        for url in urls[:self.max_sources]:
            if time.time() > deadline:
                warnings.append(f"time budget exhausted after {len(sources)} source(s); "
                                f"{len(urls) - len(sources)} not attempted")
                break
            status, raw = await self._fetch_one(url)
            text = self.extract_text(raw) if status == 200 else ""
            fs = FetchedSource(url=url, status=status, ok=bool(text), text=text,
                               error="" if text else (f"HTTP {status}" if status else "unreachable"))
            if fs.ok:
                fs.supports = self._classify(text, question)
                fs.assessment = self.source_engine.assess(url, text)
            sources.append(fs)

        retrieved = [s for s in sources if s.ok]
        yes_sources = [s for s in retrieved if s.supports == "yes"]
        no_sources = [s for s in retrieved if s.supports == "no"]
        both_sources = [s for s in retrieved if s.supports in ("both", "neither")]

        if not retrieved:
            warnings.append("no page could be retrieved - confidence reflects an "
                            "unresearched market, not a market with no evidence")

        # ---- evidence sentences, attributed to the page they came from ----
        supporting_yes = self._evidence_text(yes_sources)
        supporting_no = self._evidence_text(no_sources)
        if not supporting_yes:
            supporting_yes = "no retrieved source supports YES"
        if not supporting_no:
            supporting_no = "no retrieved source supports NO"

        contradicting = self._evidence_text(both_sources[:2]) or "none found in retrieved sources"

        # ---- LLM synthesis over what was actually retrieved ----
        summary, sections, used_llm, missing = await self._llm_synthesis(
            question, retrieved, getattr(market, "description", "") or "")

        unknown = sections.get("unknown") or (
            "not established - no source addressed the open questions"
            if not retrieved else "no source discussed the remaining unknowns")
        resolution_risks = sections.get("resolution_risks") or (
            "resolution criteria were not checked against any retrieved source")

        confidence = self._confidence(retrieved, yes_sources, no_sources, used_llm)

        elapsed = time.time() - start
        if elapsed > max_time_seconds:
            warnings.append(f"research took {elapsed:.1f}s, over the {max_time_seconds}s budget")

        return ResearchResult(
            market_id=getattr(market, "id", ""),
            question=question,
            supporting_yes=supporting_yes,
            supporting_no=supporting_no,
            contradicting_both=contradicting,
            unknown=unknown,
            resolution_risks=resolution_risks,
            final_summary=summary,
            sources=[s.url for s in retrieved],
            confidence=round(confidence, 3),
            time_spent_seconds=round(elapsed, 3),
            researched=bool(retrieved),
            source_details=sources,
            sources_attempted=len(sources),
            sources_retrieved=len(retrieved),
            yes_evidence=len(yes_sources),
            no_evidence=len(no_sources),
            avg_source_quality=(round(sum(s.quality for s in retrieved) / len(retrieved), 4)
                                if retrieved else 0.0),
            used_llm=used_llm,
            missing_sections=missing,
            warnings=warnings,
            provenance=("web_fetch+llm" if used_llm else
                        "web_fetch" if retrieved else "no_sources"),
        )

    def _evidence_text(self, sources: List[FetchedSource], per_source: int = 2,
                       max_chars: int = 600) -> str:
        """
        Real sentences from real pages, attributed.

        The old fields contained the question restated with "supporting YES"
        in front of it. These contain text that was actually retrieved.
        """
        parts: List[str] = []
        for s in sources:
            host = urlparse(s.url).hostname or s.url
            for sent in self._sentences(s.text)[:per_source]:
                parts.append(f"[{host}] {sent}")
                if len(" ".join(parts)) > max_chars:
                    break
            if len(" ".join(parts)) > max_chars:
                break
        return " ".join(parts).strip()

    async def _llm_synthesis(self, question: str, sources: List[FetchedSource],
                             description: str) -> Tuple[str, Dict[str, str], bool, List[str]]:
        """
        Ask the LLM to reason over retrieved text, then parse it by marker.

        Returns (summary, sections, used_llm, missing_sections). When there is
        no router, or no sources to reason over, the synthesis is built from
        the fetched evidence alone and `used_llm` is False - the result never
        claims a synthesis that did not happen.
        """
        expected = [name for name, _ in SECTION_MARKERS]
        if self.llm_router is None or not sources:
            summary = self._plain_summary(question, sources)
            return summary, {}, False, list(expected)

        evidence = "\n".join(
            f"- [{urlparse(s.url).hostname}] {s.text[:900]}" for s in sources[:5])
        prompt = (
            f"Market: {question}\n"
            f"Description: {description[:300]}\n\n"
            f"Retrieved sources:\n{evidence}\n\n"
            "Answer in exactly these labelled sections, using only the sources above:\n"
            "1. Bull case FOR YES\n2. Bear case AGAINST YES\n"
            "3. Contradiction that makes both wrong\n"
            "4. Unknown - what we don't know\n"
            "5. Resolution risks / ambiguous language\n"
            "If the sources do not support a section, say so in that section.")

        try:
            gen = getattr(self.llm_router, "generate", None)
            if gen is None:
                return self._plain_summary(question, sources), {}, False, list(expected)
            result = gen(prompt, max_tokens=500)
            if asyncio.iscoroutine(result):
                result = await result
            text = result if isinstance(result, str) else str(result or "")
        except Exception as e:
            logger.debug(f"[research] LLM synthesis failed: {e}")
            return self._plain_summary(question, sources), {}, False, list(expected)

        sections = self.parse_sections(text)
        missing = [name for name in expected if name not in sections]
        summary = text.strip()[:900] if text.strip() else self._plain_summary(question, sources)
        return summary, sections, bool(sections), missing

    @staticmethod
    def _plain_summary(question: str, sources: List[FetchedSource]) -> str:
        """A summary built only from what was retrieved, with no LLM involved."""
        if not sources:
            return (f"No sources could be retrieved for '{question[:80]}'. "
                    "This market is unresearched, not evidence-free.")
        yes = sum(1 for s in sources if s.supports == "yes")
        no = sum(1 for s in sources if s.supports == "no")
        avg = sum(s.quality for s in sources) / len(sources)
        return (f"{len(sources)} source(s) retrieved for '{question[:80]}': "
                f"{yes} supporting YES, {no} supporting NO, "
                f"average quality {avg:.2f}. "
                f"Highest-quality source: {max(sources, key=lambda s: s.quality).url}")

    @staticmethod
    def _confidence(sources: List[FetchedSource], yes: List[FetchedSource],
                    no: List[FetchedSource], used_llm: bool) -> float:
        """
        Confidence from what was actually retrieved.

        The old value was a literal 0.6, or 0.65 if an LLM answered - the same
        number whether ten credible sources were found or none at all.
        """
        if not sources:
            return 0.1

        count = min(len(sources), 6) / 6.0
        quality = sum(s.quality for s in sources) / len(sources)
        # A market where every source leans one way is less well understood
        # than one where both sides found support - one-sided retrieval usually
        # means the search missed the other side.
        if yes and no:
            balance = 1.0
        elif yes or no:
            balance = 0.55
        else:
            balance = 0.3

        score = 0.30 * count + 0.45 * quality + 0.25 * balance
        if used_llm:
            # An LLM synthesis over real sources adds a little; it cannot add
            # confidence to sources that were never fetched.
            score = min(0.95, score + 0.05)
        return max(0.05, min(0.95, score))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        return {
            "researcher": "Web Research",
            "method": ("discover sources by search, fetch each page, extract prose, score "
                       "with SourceQualityEngine, classify which side each supports, then "
                       "synthesise with the LLM over the retrieved text"),
            "browser_used": self.browser is not None,
            "max_sources": self.max_sources,
            "removed_fabrication": [
                "the browser argument was accepted and never used - no HTTP was issued",
                "sources was the literal string list ['heuristic']",
                "bull/bear fields were the question restated, not evidence",
                "confidence was a literal 0.6, or 0.65 if an LLM answered",
                "LLM output was split by character offset: text[:200] / text[200:400]",
            ],
            "last_error": self.last_error,
        }
