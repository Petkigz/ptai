"""
Web Researcher - uses computer, browser, terminal to research
45s per market for top 10-50
Actively tries to disprove trade (bull/bear)
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


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


class WebResearcher:
    def __init__(self, llm_router=None, browser=None):
        self.llm_router = llm_router
        self.browser = browser

    async def research(self, market, max_time_seconds: int = 45) -> ResearchResult:
        """
        Research market with bull/bear approach
        """
        import time
        start = time.time()

        # In production, would use browser, web search, LLM to research
        # For now mock with placeholder that encourages contradiction thinking
        
        supporting_yes = f"Search for evidence supporting YES for: {market.question}"
        supporting_no = f"Search for evidence supporting NO for: {market.question}"
        contradicting_both = f"Search for info that makes both YES/NO wrong for: {market.question}"
        unknown = "What don't we know?"
        resolution_risks = "Check resolution rules for ambiguity"
        
        # Would actually do web searches
        # For demo, use simple heuristic based on question
        question_lower = market.question.lower()
        if "will" in question_lower:
            supporting_yes = f"Looking for reasons {market.question} will happen"
            supporting_no = f"Looking for reasons {market.question} will NOT happen"

        elapsed = time.time() - start

        return ResearchResult(
            market_id=market.id,
            question=market.question,
            supporting_yes=supporting_yes,
            supporting_no=supporting_no,
            contradicting_both=contradicting_both,
            unknown=unknown,
            resolution_risks=resolution_risks,
            final_summary=f"Bull: {supporting_yes[:100]} | Bear: {supporting_no[:100]}",
            sources=["web_search", "browser"],
            confidence=0.6,
            time_spent_seconds=elapsed
        )
