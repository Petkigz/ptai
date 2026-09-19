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

    VAGUE_SOURCES = [
        "consensus", "general agreement", "widely reported",
        "official sources", "credible reports"  # without naming source
    ]

    def __init__(self, llm_router=None):
        self.llm_router = llm_router

    def analyze(self, market: Market) -> ResolutionAnalysis:
        question = market.question
        description = getattr(market, 'description', '') or market.raw.get('description', '') if hasattr(market, 'raw') else ''
        full_text = f"{question} {description}".lower()

        analysis = ResolutionAnalysis(
            market_id=market.id,
            question=market.question
        )

        # Detect ambiguous language
        for term in self.AMBIGUOUS_TERMS:
            if term in full_text:
                analysis.ambiguous_language.append(term)

        # Detect vague sources
        for term in self.VAGUE_SOURCES:
            if term in full_text:
                analysis.risks.append(f"Vague resolution source: '{term}'")

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

        # Calculate risk score
        risk_score = 0.0
        risk_score += len(analysis.ambiguous_language) * 0.15
        risk_score += len(analysis.risks) * 0.2
        risk_score += len(analysis.cancellation_rules) * 0.1
        if analysis.early_resolution_possible:
            risk_score += 0.15
        if not analysis.resolution_source:
            risk_score += 0.2  # No clear source is risky

        analysis.risk_score = min(1.0, risk_score)
        analysis.is_ambiguous = analysis.risk_score > 0.4 or len(analysis.ambiguous_language) >= 2
        analysis.should_trade = not analysis.is_ambiguous and analysis.risk_score < 0.5

        if analysis.is_ambiguous:
            analysis.risks.append(f"AMBIGUOUS: risk_score {analysis.risk_score:.2f} >= 0.4 or {len(analysis.ambiguous_language)} ambiguous terms")

        logger.info(f"Resolution analysis {market.id}: risk_score={analysis.risk_score:.2f} ambiguous={analysis.is_ambiguous} should_trade={analysis.should_trade} risks={analysis.risks}")

        return analysis

    def is_safe_to_trade(self, market: Market) -> tuple[bool, List[str]]:
        """Quick check if safe to trade"""
        analysis = self.analyze(market)
        return analysis.should_trade, analysis.risks + analysis.ambiguous_language
