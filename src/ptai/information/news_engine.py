"""
News Engine - processes current news, official announcements, reputable sources, timestamps
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger


@dataclass
class NewsSignal:
    market_id: str
    summary: str
    relevance: float  # 0-1
    credibility: float  # 0-1 based on source reputation
    recency: float  # 0-1, recent = higher
    sentiment: float  # -1 to 1
    sources: List[str]
    timestamp: datetime


class NewsEngine:
    def __init__(self, web_search=None, llm_router=None):
        self.web_search = web_search
        self.llm_router = llm_router
        # Reputable sources
        self.reputable_sources = {
            "reuters.com": 0.95,
            "apnews.com": 0.95,
            "bbc.com": 0.9,
            "nytimes.com": 0.85,
            "wsj.com": 0.9,
            "bloomberg.com": 0.9,
            "official": 0.95,  # official announcements
            "gov": 0.9
        }

    def assess_source_credibility(self, url: str) -> float:
        """Assess credibility based on source"""
        url_lower = url.lower()
        for domain, cred in self.reputable_sources.items():
            if domain in url_lower:
                return cred
        return 0.5  # unknown source

    def calculate_recency(self, timestamp: datetime) -> float:
        """Recency score - recent news more relevant"""
        if not timestamp:
            return 0.5
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hours_ago = (now - timestamp).total_seconds() / 3600
        if hours_ago < 1:
            return 1.0
        elif hours_ago < 24:
            return 0.8
        elif hours_ago < 24*7:
            return 0.5
        else:
            return 0.2

    async def get_news(self, market, max_articles: int = 5) -> List[NewsSignal]:
        """Get news for market"""
        signals = []
        
        # Would use web search to find news
        # For now mock
        if self.web_search:
            try:
                # Search for news about market question
                results = await self.web_search.search(market.question, limit=max_articles) if hasattr(self.web_search, 'search') else []
                for r in results[:max_articles]:
                    url = r.get("url", "")
                    credibility = self.assess_source_credibility(url)
                    # Would parse timestamp
                    signals.append(NewsSignal(
                        market_id=market.id,
                        summary=r.get("snippet", "")[:200],
                        relevance=0.7,
                        credibility=credibility,
                        recency=0.6,
                        sentiment=0.0,
                        sources=[url],
                        timestamp=datetime.now(timezone.utc)
                    ))
            except Exception as e:
                logger.warning(f"News search failed: {e}")

        return signals

    def synthesize(self, signals: List[NewsSignal]) -> Dict:
        """Synthesize news signals into summary"""
        if not signals:
            return {"summary": "No news", "sentiment": 0, "credibility": 0, "relevance": 0}
        
        # Weighted by credibility and recency
        total_weight = sum(s.credibility * s.recency for s in signals)
        if total_weight == 0:
            return {"summary": "No credible news", "sentiment": 0, "credibility": 0, "relevance": 0}
        
        weighted_sentiment = sum(s.sentiment * s.credibility * s.recency for s in signals) / total_weight
        avg_credibility = sum(s.credibility for s in signals) / len(signals)
        avg_relevance = sum(s.relevance for s in signals) / len(signals)
        
        summary = " | ".join([s.summary[:100] for s in signals[:3]])
        
        return {
            "summary": summary,
            "sentiment": weighted_sentiment,
            "credibility": avg_credibility,
            "relevance": avg_relevance,
            "count": len(signals),
            "sources": [src for s in signals for src in s.sources]
        }
