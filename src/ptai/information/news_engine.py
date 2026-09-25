"""
News Engine - processes current news, official announcements, reputable sources, timestamps
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger

from ..betting.sports_data import BROWSER_USER_AGENT


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
        """Get news for market - V9 FIX #4 real RSS + web_search, not mock
        Local-only, public info, no cloud
        """
        signals = []
        
        # Path 1: Use injected web_search if available (real search)
        if self.web_search:
            try:
                results = await self.web_search.search(market.question, limit=max_articles) if hasattr(self.web_search, 'search') else []
                for r in results[:max_articles]:
                    url = r.get("url", "")
                    credibility = self.assess_source_credibility(url)
                    ts_str = r.get("timestamp") or r.get("published")
                    ts = datetime.now(timezone.utc)
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except:
                            pass
                    signals.append(NewsSignal(
                        market_id=market.id,
                        summary=r.get("snippet", "")[:200],
                        relevance=0.7,
                        credibility=credibility,
                        recency=self.calculate_recency(ts),
                        sentiment=0.0,
                        sources=[url],
                        timestamp=ts
                    ))
            except Exception as e:
                logger.warning(f"News search failed: {e}")

        # Path 2: Real RSS feeds - local HTTP, no API keys, public info only
        if not signals:
            try:
                import requests
                # Extract keywords from market question for filtering
                q_lower = market.question.lower()
                keywords = [w for w in q_lower.split() if len(w) > 4][:5]
                
                # Try BBC, Reuters RSS - public, no auth
                rss_feeds = [
                    "https://feeds.bbci.co.uk/news/rss.xml",
                    "https://www.reutersagency.com/feed/?best-topics=tech&post_type=best",
                ]
                # For local testing, use lightweight fetch with timeout
                # Only fetch if market category suggests news relevance
                has_news_potential = any(k in q_lower for k in ["trump", "biden", "election", "fed", "cpi", "inflation", "rate", "earnings", "nfl", "nba", "btc", "bitcoin"])
                if has_news_potential:
                    for feed_url in rss_feeds[:1]:  # only 1 to save time
                        try:
                            resp = requests.get(feed_url, timeout=3, headers={"User-Agent": BROWSER_USER_AGENT})
                            if resp.status_code == 200:
                                text = resp.text[:10000]
                                # Simple keyword matching
                                relevance = 0.0
                                for kw in keywords:
                                    if kw in text.lower():
                                        relevance += 0.2
                                relevance = min(1.0, relevance)
                                if relevance > 0.1:
                                    signals.append(NewsSignal(
                                        market_id=market.id,
                                        summary=f"RSS {feed_url} mentions {keywords} relevance {relevance:.2f} for {market.question[:80]}",
                                        relevance=relevance,
                                        credibility=self.assess_source_credibility(feed_url),
                                        recency=0.8,
                                        sentiment=0.0,
                                        sources=[feed_url],
                                        timestamp=datetime.now(timezone.utc)
                                    ))
                        except Exception as e:
                            logger.debug(f"RSS fetch {feed_url} failed: {e}")
                            continue
            except Exception as e:
                logger.debug(f"RSS news failed for {market.id}: {e}")

        # Path 3: Category-based heuristic when no real news - still provides context
        if not signals:
            q_lower = market.question.lower()
            if any(k in q_lower for k in ["trump", "biden", "election"]):
                signals.append(NewsSignal(
                    market_id=market.id,
                    summary=f"Politics market {market.question[:100]} - check polls, official announcements, reputable sources Reuters/AP/BBC",
                    relevance=0.5,
                    credibility=0.6,
                    recency=0.5,
                    sentiment=0.0,
                    sources=["category_heuristic_politics"],
                    timestamp=datetime.now(timezone.utc)
                ))
            elif any(k in q_lower for k in ["fed", "cpi", "inflation"]):
                signals.append(NewsSignal(
                    market_id=market.id,
                    summary=f"Economics market {market.question[:100]} - check Fed announcements, CPI data, official gov sources",
                    relevance=0.5,
                    credibility=0.7,
                    recency=0.5,
                    sentiment=0.0,
                    sources=["category_heuristic_economics"],
                    timestamp=datetime.now(timezone.utc)
                ))

        return signals[:max_articles]

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
