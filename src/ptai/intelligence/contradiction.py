"""
Contradiction Engine - researcher actively tries to DISPROVE its trade
Instead of "Find evidence supporting YES", do:
- Researcher A: Find supporting YES
- Researcher B: Find supporting NO
- Researcher C: Find info that makes both wrong
- Researcher D: Check resolution rules
- Researcher E: Look for info after latest market move

Reduces confirmation bias.
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


class ContradictionEngine:
    def __init__(self, llm_router=None, web_researcher=None):
        self.llm_router = llm_router
        self.web_researcher = web_researcher

    def research_bull_case(self, market_question: str, research_text: str = "") -> List[Evidence]:
        """Researcher A: Find evidence supporting YES"""
        # In production, would use LLM + web search with prompt "Find evidence supporting YES for: {question}"
        # For now heuristic
        evidence = []
        if research_text:
            # Simple keyword matching for demo
            if any(word in research_text.lower() for word in ["will", "likely", "expected", "success"]):
                evidence.append(Evidence(
                    side="YES",
                    strength=0.6,
                    content=f"Found supporting evidence in research: {research_text[:200]}",
                    source="web_research",
                    credibility=0.6
                ))
        return evidence

    def research_bear_case(self, market_question: str, research_text: str = "") -> List[Evidence]:
        """Researcher B: Find evidence supporting NO"""
        evidence = []
        if research_text:
            if any(word in research_text.lower() for word in ["unlikely", "fail", "risk", "doubt", "against"]):
                evidence.append(Evidence(
                    side="NO",
                    strength=0.6,
                    content=f"Found contradicting evidence: {research_text[:200]}",
                    source="web_research",
                    credibility=0.6
                ))
        return evidence

    def research_contradicting_both(self, market_question: str) -> List[Evidence]:
        """Researcher C: Find information that could make both models wrong"""
        # e.g. market might cancel, ambiguous, external event
        evidence = []
        # Check for cancellation language
        if any(word in market_question.lower() for word in ["cancel", "postpone", "if"]):
            evidence.append(Evidence(
                side="UNKNOWN",
                strength=0.5,
                content="Question contains conditional/cancellation language - could make both YES/NO wrong",
                source="resolution_analysis",
                credibility=0.8
            ))
        return evidence

    def check_resolution_rules(self, market) -> List[Evidence]:
        """Researcher D: Check resolution rules"""
        evidence = []
        # Check for ambiguous language
        desc = getattr(market, 'description', '') + " " + market.question
        ambiguous_terms = ["at least", "approximately", "around", "substantially", "significant", "official", "credible"]
        for term in ambiguous_terms:
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
        """Researcher E: Look for information published after latest market move"""
        # If market moved recently, check if new info justifies it or if it's overreaction
        evidence = []
        # Would compare news timestamps vs market price move timestamps
        # For now placeholder
        return evidence

    def synthesize(self, market, research_text: str = "", news: str = "", tweets: List = None) -> ContradictionReport:
        """Synthesize all researchers into final report"""
        report = ContradictionReport(
            market_id=market.id,
            question=market.question
        )

        report.supporting_yes = self.research_bull_case(market.question, research_text + " " + news)
        report.supporting_no = self.research_bear_case(market.question, research_text + " " + news)
        report.contradicting_both = self.research_contradicting_both(market.question)
        report.resolution_risks = self.check_resolution_rules(market)
        report.new_info_after_market_move = self.check_new_info(market)

        # Calculate net score
        yes_strength = sum(e.strength * e.credibility for e in report.supporting_yes)
        no_strength = sum(e.strength * e.credibility for e in report.supporting_no)
        risk_strength = sum(e.strength for e in report.resolution_risks)
        
        report.net_score = yes_strength - no_strength
        
        # Confidence adjustment: if both sides strong, reduce confidence
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

        logger.info(f"Contradiction report for {market.id}: net_score={report.net_score:.2f} adjustment={report.confidence_adjustment:.2f}")

        return report
