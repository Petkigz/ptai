"""
Event Trading Strategy - news-driven, event-driven edge
Part of venue × market × strategy engine
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime, timezone

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType


@dataclass
class EventSignal:
    market_id: str
    event_type: str  # earnings, fed, election, sports, etc
    news_impact: float  # -1 to +1, how news affects prob
    recency_hours: float
    source_credibility: float
    reasoning: str
    should_trade: bool
    estimated_edge: float


class EventTradingEngine:
    """
    Event-driven strategy:
    - Earnings announcements
    - Fed decisions
    - Elections
    - Sports events
    - Economic data releases
    Uses news, timeline, and market reaction to estimate edge
    """
    def __init__(self, llm_router=None):
        self.llm_router = llm_router
        self.event_keywords = {
            "earnings": ["earnings", "eps", "revenue", "beat", "miss", "guidance"],
            "fed": ["fed", "fomc", "rate", "powell", "interest"],
            "election": ["election", "vote", "poll", "candidate", "ballot"],
            "sports": ["game", "match", "championship", "final", "win"],
            "economic": ["cpi", "inflation", "jobs", "unemployment", "gdp", "ppi"],
            "crypto": ["btc", "eth", "crypto", "bitcoin", "ethereum", "halving"],
            "weather": ["temperature", "weather", "hurricane", "storm"],
            "politics": ["trump", "biden", "senate", "congress", "bill", "law"]
        }

    def detect_event_type(self, market: Market, context: Dict = None) -> str:
        q = market.question.lower()
        context_text = (context.get("news", "") + " " + context.get("research", "")).lower() if context else ""
        combined = q + " " + context_text
        
        for event_type, keywords in self.event_keywords.items():
            for kw in keywords:
                if kw in combined:
                    return event_type
        return "general"

    def evaluate(self, market: Market, context: Dict = None) -> EventSignal:
        context = context or {}
        event_type = self.detect_event_type(market, context)
        
        news = context.get("news", "")
        news_credibility = context.get("news_credibility", 0.5)
        sentiment = context.get("sentiment", {})
        sentiment_score = sentiment.get("score", 0) if isinstance(sentiment, dict) else 0
        
        # Estimate news impact
        # If market price hasn't moved but news is strong, potential edge
        news_impact = 0.0
        if news and len(news) > 20:
            # Simple heuristic: positive sentiment -> YES prob up
            # In production, LLM would parse news impact
            news_impact = sentiment_score * 0.1  # -0.1 to +0.1
            if "beat" in news.lower() or "exceed" in news.lower() or "win" in news.lower():
                news_impact = abs(news_impact) + 0.05
            elif "miss" in news.lower() or "lose" in news.lower() or "below" in news.lower():
                news_impact = -abs(news_impact) - 0.05
        
        # Recency
        recency_hours = 24.0  # default
        if context.get("news_timestamp"):
            try:
                ts = context["news_timestamp"]
                if isinstance(ts, datetime):
                    recency_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            except:
                pass
        
        # Edge estimation: if news impact >0 but market hasn't moved, edge = impact
        # Simplified: edge = news_impact * credibility * time_decay
        time_decay = max(0.1, 1.0 - recency_hours / 72.0)  # decay over 3 days
        estimated_edge = news_impact * news_credibility * time_decay
        
        should_trade = abs(estimated_edge) > 0.08 and news_credibility > 0.6 and recency_hours < 24
        
        reasoning = (
            f"Event type {event_type} | News impact {news_impact:.3f} credibility {news_credibility:.2f} "
            f"recency {recency_hours:.1f}h decay {time_decay:.2f} | Edge {estimated_edge:.3f} | "
            f"Trade {should_trade} | News: {news[:100]}"
        )
        
        return EventSignal(
            market_id=market.id,
            event_type=event_type,
            news_impact=news_impact,
            recency_hours=recency_hours,
            source_credibility=news_credibility,
            reasoning=reasoning,
            should_trade=should_trade,
            estimated_edge=estimated_edge
        )

    def to_venue_opportunity(self, market: Market, signal: EventSignal, context: Dict = None) -> Optional[VenueOpportunity]:
        if not signal.should_trade:
            return None
        
        # Fair value = market price + edge
        fair = market.best_price + signal.estimated_edge
        fair = max(0.01, min(0.99, fair))
        
        opp = VenueOpportunity(
            market=market,
            venue_id=getattr(market, 'source', 'unknown').value if hasattr(getattr(market, 'source', ''), 'value') else str(getattr(market, 'source', 'unknown')),
            venue_type=VenueType.PREDICTION,
            side="YES" if signal.estimated_edge > 0 else "NO",
            market_price=market.best_price,
            estimated_fair=fair,
            raw_edge=signal.estimated_edge,
            effective_edge=signal.estimated_edge * signal.source_credibility,
            confidence=signal.source_credibility,
            uncertainty=1-signal.source_credibility,
            liquidity_score=min(1.0, market.liquidity / 10000),
            execution_quality=0.7,
            category=signal.event_type,
            sources=[f"event_{signal.event_type}"],
            reasoning=signal.reasoning,
            bull_case=f"News supports YES: {context.get('news', '')[:100]}" if signal.estimated_edge > 0 else "",
            bear_case=f"News supports NO: {context.get('news', '')[:100]}" if signal.estimated_edge < 0 else "",
            should_trade=signal.should_trade
        )
        opp.calculate_common_score()
        return opp
