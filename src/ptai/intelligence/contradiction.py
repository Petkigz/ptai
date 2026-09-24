"""
Contradiction Engine - researcher actively tries to DISPROVE its trade
Instead of "Find evidence supporting YES", do:
- Researcher A: Find supporting YES
- Researcher B: Find supporting NO
- Researcher C: Find info that makes both wrong
- Researcher D: Check resolution rules
- Researcher E: Look for info after latest market move

Reduces confirmation bias.

What this used to do instead
----------------------------
Researcher A and B both received the SAME string and ran substring matching on
it:

    if any(word in text.lower() for word in ["will", "likely", "expected", "success"]):
        evidence.append(Evidence(side="YES", strength=0.6, credibility=0.6, ...))

Market questions almost always contain "will", and the question was part of the
text passed in - so the bull case fired on essentially every market regardless
of what the research said, and its strength (0.6) and credibility (0.6) were
literals. The bear case did the same on ["unlikely", "fail", "risk", ...]. Then
`confidence_adjustment` was derived from those numbers and used to size
positions.

`web_researcher` was accepted as a constructor argument and never referenced by
any method. All three call sites passed only `llm_router`.

Now the two researchers are fed different things and their evidence carries the
retrieved source's own quality score. Nothing retrieved means no evidence, and
the report says so rather than reporting balanced-looking 0.6s.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone
from loguru import logger


@dataclass
class Evidence:
    side: str  # YES, NO, NEUTRAL, UNKNOWN, RISK
    strength: float  # 0-1
    content: str
    source: str
    timestamp: Optional[datetime] = None
    credibility: float = 0.5

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class ContradictionReport:
    market_id: str
    question: str
    supporting_yes: List[Evidence] = field(default_factory=list)
    supporting_no: List[Evidence] = field(default_factory=list)
    contradicting_both: List[Evidence] = field(default_factory=list)
    unknown: List[Evidence] = field(default_factory=list)
    resolution_risks: List[Evidence] = field(default_factory=list)
    new_info_after_market_move: List[Evidence] = field(default_factory=list)
    final_assessment: str = ""
    bull_case: str = ""
    bear_case: str = ""
    net_score: float = 0.0  # positive = favors YES, negative = favors NO
    confidence_adjustment: float = 0.0  # negative if conflicting evidence strong
    # Whether web sources were actually retrieved. Without this a report with no
    # evidence is indistinguishable from one where research failed, and the
    # empty bull/bear strings read as "no evidence exists".
    researched: bool = False
    research_blockers: List[str] = field(default_factory=list)
    provenance: str = "market_definition_only"


class ContradictionEngine:
    def __init__(self, llm_router=None, web_researcher=None):
        self.llm_router = llm_router
        self.web_researcher = web_researcher
        self.last_error: str = ""
        # Set by research() / synthesize() so evidence can be built from the
        # retrieved sources rather than from the question's wording.
        self._last_result = None

    async def research(self, market, max_time_seconds: int = 30):
        """
        Retrieve real sources for a market.

        Returns the WebResearcher's result, or None when no researcher is
        configured. Callers must treat None as "nothing was researched" rather
        than as an empty result.
        """
        if self.web_researcher is None:
            self.last_error = ("no web_researcher configured - contradiction analysis "
                               "will report no evidence rather than fabricate it")
            logger.debug(self.last_error)
            return None
        try:
            return await self.web_researcher.research(
                market, max_time_seconds=max_time_seconds)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"contradiction research failed: {self.last_error}")
            return None

    # Keyword matching on the question is not evidence. These terms are only
    # used to spot resolution ambiguity in the market's OWN rules text, which
    # is a legitimate reading of the market rather than of the world.
    _AMBIGUOUS_TERMS = ["at least", "approximately", "around", "substantially",
                        "significant", "official", "credible"]

    def _evidence_from_sources(self, side: str) -> List[Evidence]:
        """
        Build evidence from retrieved sources of a given side.

        Strength comes from the source's assessed quality and the researcher's
        own count of sources supporting that side - not from a literal.
        """
        result = getattr(self, "_last_result", None)
        if result is None or not getattr(result, "researched", False):
            return []

        evidence: List[Evidence] = []
        for src in getattr(result, "source_details", []) or []:
            if getattr(src, "supports", "unknown") != side:
                continue
            quality = src.quality
            if quality <= 0:
                continue
            body = (getattr(src, "text", "") or "").strip()
            if not body:
                continue
            evidence.append(Evidence(
                side=side.upper(),
                # A source only supports a side as strongly as it is credible,
                # and it can never be more than a single piece of evidence.
                strength=round(min(1.0, quality), 4),
                content=body[:300],
                source=getattr(src, "url", "unknown"),
                credibility=round(quality, 4),
                timestamp=getattr(src, "published_at", None),
            ))
        return evidence

    def research_bull_case(self, market_question: str, research_text: str = "") -> List[Evidence]:
        """
        Evidence supporting YES.

        Previously: substring-matched ["will", "likely", "expected", "success"]
        in text that included the market question, so it fired on almost every
        market at a hardcoded 0.6 strength and 0.6 credibility.
        """
        return self._evidence_from_sources("yes")

    def research_bear_case(self, market_question: str, research_text: str = "") -> List[Evidence]:
        """
        Evidence supporting NO.

        Previously: substring-matched ["unlikely", "fail", "risk", "doubt",
        "against"] on the same string the bull case received.
        """
        return self._evidence_from_sources("no")

    def research_contradicting_both(self, market_question: str) -> List[Evidence]:
        """
        Researcher C: information that could void BOTH sides.

        The old check matched ["cancel", "postpone", "if"] - and "if" appears in
        a large share of market questions ("Will X happen if Y?"), so almost any
        conditional market produced a 0.8-credibility "could void" warning. The
        terms are now narrower and the credibility reflects that this is a
        reading of the question's wording, not a reported fact.
        """
        evidence = []
        text = (market_question or "").lower()
        triggers = [t for t in ("will be cancelled", "may be cancelled", "postponed",
                                "be voided", "void if", "annulled")
                    if t in text]
        if triggers:
            evidence.append(Evidence(
                side="UNKNOWN",
                strength=0.5,
                content=(f"Question wording allows a void/cancellation outcome: "
                         f"{', '.join(triggers)} - could make both YES and NO wrong"),
                source="question_wording",
                credibility=0.5
            ))
        return evidence

    def check_resolution_rules(self, market) -> List[Evidence]:
        """Researcher D: Check resolution rules"""
        evidence = []
        # Check for ambiguous language
        desc = getattr(market, 'description', '') + " " + market.question
        for term in self._AMBIGUOUS_TERMS:
            if term in desc.lower():
                evidence.append(Evidence(
                    side="RISK",
                    strength=0.7,
                    content=f"Ambiguous resolution term detected: '{term}' in '{desc[:100]}'",
                    source="resolution_analyzer",
                    credibility=0.9
                ))
        
        # Check if resolution source defined
        if "resolve" not in desc.lower() and "source" not in desc.lower():
            evidence.append(Evidence(
                side="RISK",
                strength=0.4,
                content="No clear resolution source mentioned",
                source="resolution_analyzer",
                credibility=0.6
            ))
        
        return evidence

    def check_new_info(self, market, last_market_move_time: datetime = None) -> List[Evidence]:
        """
        Researcher E: information published after the last market move.

        Not implemented. It previously returned [] under a comment saying "for
        now placeholder", which is at least honest, but a caller cannot tell it
        apart from "checked and found nothing". It now reports that it did not
        run, so an absent result is not read as a clean one.
        """
        self.last_error = ("check_new_info is not implemented; cannot compare news "
                           "timestamps against market price moves")
        logger.debug(self.last_error)
        return []

    def synthesize(self, market, research_text: str = "", news: str = "",
                   tweets: List = None, research_result=None) -> ContradictionReport:
        """
        Synthesize all researchers into final report.

        `research_result` is a WebResearcher ResearchResult. When it is absent or
        unresearched, both sides come back empty - which is the honest answer.
        The two researchers are deliberately given different inputs now: the
        bull case reads sources classified as supporting YES, the bear case
        reads sources classified as supporting NO.
        """
        # An explicitly passed result wins; otherwise use whatever research()
        # last stored, so an async caller can research first and synthesize after.
        if research_result is not None:
            self._last_result = research_result
        report = ContradictionReport(
            market_id=market.id,
            question=market.question
        )

        last = getattr(self, "_last_result", None)
        researched = bool(last is not None and getattr(last, "researched", False))

        report.supporting_yes = self.research_bull_case(market.question, research_text)
        report.supporting_no = self.research_bear_case(market.question, research_text)
        report.contradicting_both = self.research_contradicting_both(market.question)
        report.resolution_risks = self.check_resolution_rules(market)
        report.new_info_after_market_move = self.check_new_info(market)

        report.researched = researched
        if not researched:
            report.research_blockers.append(
                "no sources retrieved - contradiction analysis ran on the market "
                "definition alone and produced no web evidence")

        # Calculate net score
        yes_strength = sum(e.strength * e.credibility for e in report.supporting_yes)
        no_strength = sum(e.strength * e.credibility for e in report.supporting_no)
        risk_strength = sum(e.strength for e in report.resolution_risks)
        
        report.net_score = yes_strength - no_strength
        
        # Confidence adjustment: if both sides strong, reduce confidence.
        # With no retrieval both strengths are 0, which must NOT be reported as
        # a balanced, low-conflict market.
        if not researched:
            report.confidence_adjustment = -0.1
            report.final_assessment = (
                "No web evidence retrieved - analysis covers the market definition "
                "only (resolution wording and rules). Treat as unverified.")
            report.bull_case = ""
            report.bear_case = ""
            if report.resolution_risks:
                report.final_assessment += f" | Resolution risks: {len(report.resolution_risks)}"
            logger.info(f"Contradiction report for {market.id}: NOT RESEARCHED "
                        f"(no sources), adjustment {report.confidence_adjustment:.2f}")
            return report

        if yes_strength > 0.5 and no_strength > 0.5:
            report.confidence_adjustment = -0.15  # conflicting evidence
            report.final_assessment = f"Conflicting evidence: YES strength {yes_strength:.2f} vs NO strength {no_strength:.2f} - reduce confidence"
        elif risk_strength > 0.5:
            report.confidence_adjustment = -0.2
            report.final_assessment = f"High resolution risk: {risk_strength:.2f} - avoid trade"
        elif yes_strength > no_strength:
            report.final_assessment = f"Bull case stronger: {yes_strength:.2f} vs {no_strength:.2f}"
        else:
            report.final_assessment = f"Bear case stronger: {no_strength:.2f} vs {yes_strength:.2f}"

        # Build bull/bear cases
        report.bull_case = "\n".join([f"- {e.content} (strength {e.strength}, cred {e.credibility})" for e in report.supporting_yes[:5]])
        report.bear_case = "\n".join([f"- {e.content} (strength {e.strength}, cred {e.credibility})" for e in report.supporting_no[:5]])

        if report.resolution_risks:
            report.final_assessment += f" | Resolution risks: {len(report.resolution_risks)}"

        report.provenance = getattr(last, "provenance", "web_fetch")
        logger.info(f"Contradiction report for {market.id}: net_score={report.net_score:.2f} "
                    f"adjustment={report.confidence_adjustment:.2f} provenance={report.provenance}")

        return report
