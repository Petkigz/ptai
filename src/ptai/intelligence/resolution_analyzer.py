"""
Resolution Risk Analyzer - "What exactly causes this contract to resolve YES?"
Critical for prediction markets.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
from loguru import logger

from ..markets.base import Market


@dataclass
class ResolutionAnalysis:
    market_id: str
    question: str
    resolution_source: Optional[str] = None
    resolution_date: Optional[str] = None
    timezone: str = "UTC"
    exact_criteria: str = ""
    ambiguous_language: List[str] = field(default_factory=list)
    cancellation_rules: List[str] = field(default_factory=list)
    early_resolution_possible: bool = False
    oracle: Optional[str] = None
    # What names the outcome, if anything does: "polymarket", "uma", "reuters",
    # "official broadcast", ... An empty string means we found nothing that
    # decides the market, which is the genuinely risky case.
    named_resolver: str = ""
    risks: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    should_trade: bool = True
    risk_score: float = 0.0  # 0-1, higher = riskier


class ResolutionAnalyzer:
    """
    For every market:
    - Question
    - Resolution source
    - Resolution date
    - Resolution timezone
    - Exact criteria
    - Ambiguous language
    - Cancellation rules
    - Early resolution possibilities
    - Oracle/source

    If ambiguous: TRADE = FALSE no matter edge.
    """
    
    AMBIGUOUS_TERMS = [
        "at least", "at most", "approximately", "around", "roughly",
        "substantially", "significant", "significantly", "material",
        "official", "credible", "reliable", "generally", "usually",
        "might", "could", "perhaps", "possibly", "likely to be",
        "deemed", "considered", "interpreted"
    ]

    # Terms that make the QUESTION unresolvable whoever reports the outcome, or
    # however carefully: "significantly higher" has no threshold, "approximately"
    # has no bound. One of these in the question is enough to refuse the market,
    # named resolver or not.
    HARD_AMBIGUOUS_TERMS = [
        "approximately", "around", "roughly", "substantially",
        "significant", "significantly", "material",
        "deemed", "considered", "interpreted",
    ]

    VAGUE_SOURCES = [
        "consensus", "general agreement", "widely reported",
        "official sources", "credible reports"  # without naming source
    ]

    # Phrases that NAME who decides the market. Polymarket's own published rule
    # for its auto-generated markets is "resolve based on the consensus of
    # credible reporting" - a named resolver, printed on the market. The
    # analyzer used to read the word "consensus" in that sentence as vagueness
    # and block the market, so the operator's whole scan came back like this:
    #
    #   Resolution analysis 4910857: risk_score=0.85 ambiguous=True
    #   should_trade=False risks=["Vague resolution source: 'consensus'", ...]
    #   -> every Game 2 prop, every nomination market, every competition winner
    #
    # A market that states who resolves it, and how, is resolved. The gate is
    # for questions nothing can settle, and it stays closed for those.
    RESOLVER_PATTERNS = [
        r"resolution source",
        r"resolves? (?:to|as|based|according|per)",
        r"will be resolved",
        r"settled by",
        r"determined by",
        r"according to",
        r"as reported by",
        r"per the",
        r"based on",
    ]
    # Named sources we can look up ourselves: an outlet, a feed, a rulebook, or
    # the venue's own published resolution process.
    NAMED_SOURCES = [
        "polymarket", "uma", "kalshi", "associated press", "reuters",
        "bloomberg", "official broadcast", "official results", "official feed",
        "official league", "credible reporting", "news reporting",
        "box score", "espn", "bbc", "sky sports", "flashscore",
        "government", "census", "bureau", "election commission", "sec filing",
        "coinbase", "binance", "coingecko", "coinmarketcap",
    ]

    def __init__(self, llm_router=None):
        self.llm_router = llm_router

    def _find_named_resolver(self, text: str) -> str:
        """Who decides this market, found in its own wording. "" if nobody."""
        lowered = (text or "").lower()
        for source in self.NAMED_SOURCES:
            if source in lowered:
                return source
        for pattern in self.RESOLVER_PATTERNS:
            match = re.search(pattern, lowered)
            if match and re.search(r"[a-z]{3,}", lowered[match.end():match.end() + 60]):
                # "resolves based on X" / "according to X" - something follows it
                return match.group(0).strip()
        return ""

    def analyze(self, market: Market) -> ResolutionAnalysis:
        question = market.question
        description = getattr(market, 'description', '') or market.raw.get('description', '') if hasattr(market, 'raw') else ''
        full_text = f"{question} {description}".lower()
        # Ambiguity is a property of the QUESTION. The description carries the
        # venue's resolution boilerplate, and counting words in it meant a
        # market was blocked for phrasing it did not control.
        question_text = (question or "").lower()

        analysis = ResolutionAnalysis(
            market_id=market.id,
            question=market.question
        )

        # Who resolves it, from the market's own text (question or description).
        analysis.named_resolver = self._find_named_resolver(full_text)

        # Detect ambiguous language - in the question being bet on
        for term in self.AMBIGUOUS_TERMS:
            if term in question_text:
                analysis.ambiguous_language.append(term)

        # Detect vague sources. Only when nothing else names a resolver: a
        # market that says "consensus of credible reporting AND the Associated
        # Press" has named its resolver, and one that says "by general
        # agreement" with nothing behind it has not.
        for term in self.VAGUE_SOURCES:
            if term in full_text and not analysis.named_resolver:
                analysis.risks.append(f"Vague resolution source: '{term}' - "
                                      f"nothing in the market text names who "
                                      f"decides it")

        # Check for cancellation rules
        cancel_keywords = ["cancel", "postpone", "void", "refund", "if not", "unless"]
        for kw in cancel_keywords:
            if kw in full_text:
                analysis.cancellation_rules.append(f"Contains cancellation language: '{kw}'")

        # Early resolution
        if "early" in full_text and "resolv" in full_text:
            analysis.early_resolution_possible = True
            analysis.risks.append("Early resolution possible - timing risk")

        # Try to extract resolution source
        # Look for patterns like "resolves to YES if", "source: ", "according to"
        source_patterns = [
            r"resolves?.*?according to (.*?)[\.\,]",
            r"source:\s*(.*?)[\.\,]",
            r"based on (.*?)[\.\,]",
            r"as reported by (.*?)[\.\,]"
        ]
        for pattern in source_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                analysis.resolution_source = match.group(1).strip()[:200]
                break

        # Check for timezone issues
        if "et" in full_text or "pt" in full_text or "utc" in full_text:
            # Has timezone mentioned, good
            pass
        else:
            if "date" in full_text or "time" in full_text:
                analysis.risks.append("Resolution date/time mentioned but no timezone specified")

        # Calculate risk score. Capped per category, because the old total was
        # a count of paraphrase - three words in a description could exceed the
        # threshold that the underlying question never came close to.
        risk_score = 0.0
        risk_score += min(0.30, len(analysis.ambiguous_language) * 0.15)
        risk_score += len(analysis.risks) * 0.2
        risk_score += min(0.20, len(analysis.cancellation_rules) * 0.1)
        if analysis.early_resolution_possible:
            risk_score += 0.15
        if not analysis.resolution_source and not analysis.named_resolver:
            risk_score += 0.2  # No source named anywhere is risky

        analysis.risk_score = min(1.0, risk_score)
        # Blocking conditions, all of them about the question or about nobody
        # being named - not about how many descriptive words the venue used:
        #   - one unresolvable term in the question ("significant", "approximately")
        #   - two softer ambiguous words in the question itself
        #   - one soft ambiguous word in a market that names no resolver
        #   - vague-source risk with no named resolver
        hard_terms = [t for t in analysis.ambiguous_language
                      if t.lower() in self.HARD_AMBIGUOUS_TERMS]
        analysis.is_ambiguous = bool(
            hard_terms
            or len(analysis.ambiguous_language) >= 2
            or (len(analysis.ambiguous_language) >= 1 and not analysis.named_resolver)
            or (any("Vague resolution source" in r for r in analysis.risks)
                and not analysis.named_resolver)
            or analysis.risk_score >= 0.5
        )
        analysis.should_trade = not analysis.is_ambiguous and analysis.risk_score < 0.5

        if analysis.is_ambiguous:
            analysis.risks.append(f"AMBIGUOUS: risk_score {analysis.risk_score:.2f} >= 0.4 or {len(analysis.ambiguous_language)} ambiguous terms")

        logger.info(f"Resolution analysis {market.id}: risk_score={analysis.risk_score:.2f} "
                    f"resolver={analysis.named_resolver or 'NONE'} "
                    f"ambiguous={analysis.is_ambiguous} "
                    f"should_trade={analysis.should_trade} risks={analysis.risks}")

        return analysis

    def is_safe_to_trade(self, market: Market) -> tuple[bool, List[str]]:
        """Quick check if safe to trade"""
        analysis = self.analyze(market)
        return analysis.should_trade, analysis.risks + analysis.ambiguous_language
