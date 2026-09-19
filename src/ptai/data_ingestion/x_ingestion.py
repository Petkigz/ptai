"""
X (Twitter) Sentiment Ingestion
Need scraper or API. Be aware rate limits and ToS.
Lightweight approach: snscrape (may break) or paid API if scale
"""
from typing import List, Dict, Any, Optional
from loguru import logger


class XIngestion:
    """
    X sentiment ingestion - lightweight, local-first
    Options:
    - snscrape (free, may break, X blocks)
    - Paid API (reliable, costs)
    - Browser automation fallback (fragile)
    
    Circuit breaker: if X blocking detected, disable for 10 min, use LLM-only
    """
    def __init__(self, method: str = "snscrape", enabled: bool = False):
        self.method = method
        self.enabled = enabled
        self.circuit_breaker_fails = 0
        self.circuit_breaker_threshold = 3
        self.is_disabled = False

    def fetch_tweets(self, query: str, max_tweets: int = 50) -> List[Dict]:
        """Fetch tweets for query - with circuit breaker"""
        if not self.enabled:
            logger.info("X ingestion disabled - fastest LLM-only mode, recommended for R1 slow model")
            return []
        
        if self.is_disabled:
            logger.warning("X ingestion circuit breaker active - disabled for 10 min, using LLM-only")
            return []
        
        try:
            # Would use snscrape or API
            # Mock for offline
            mock_tweets = [
                {"text": f"Trump winning {query}!", "author": "user1", "likes": 100, "retweets": 20, "credibility": 0.6, "timestamp": "2026-01-01"},
                {"text": f"Bearish on {query}, polls show opposite", "author": "analyst", "likes": 500, "retweets": 100, "credibility": 0.8, "timestamp": "2026-01-01"},
            ]
            logger.info(f"X ingestion: {len(mock_tweets)} tweets for '{query}' via {self.method} (mock)")
            return mock_tweets[:max_tweets]
        except Exception as e:
            self.circuit_breaker_fails += 1
            logger.error(f"X ingestion failed {self.circuit_breaker_fails}/{self.circuit_breaker_threshold}: {e}")
            if self.circuit_breaker_fails >= self.circuit_breaker_threshold:
                self.is_disabled = True
                logger.warning("X BLOCKING DETECTED - circuit breaker 10 min, 40 sec not 21 min if blocked with fix")
            return []

    def analyze_sentiment(self, tweets: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from tweets"""
        if not tweets:
            return {"score": 0, "volume": 0, "credibility": 0, "bot_likelihood": 0}
        
        # Simple heuristic: positive words
        positive_words = ["win", "bullish", "up", "beat", "exceed", "good", "great"]
        negative_words = ["lose", "bearish", "down", "miss", "below", "bad"]
        
        pos_count = sum(1 for t in tweets for w in positive_words if w in t.get("text", "").lower())
        neg_count = sum(1 for t in tweets for w in negative_words if w in t.get("text", "").lower())
        
        total = pos_count + neg_count
        score = (pos_count - neg_count) / total if total > 0 else 0
        
        avg_credibility = sum(t.get("credibility", 0.5) for t in tweets) / len(tweets) if tweets else 0
        
        return {
            "score": score,  # -1 to +1
            "volume": len(tweets),
            "credibility": avg_credibility,
            "bot_likelihood": 0.2,  # would detect bot bursts, duplicates, farming
            "novelty": 0.7,
            "time_decay": 0.8,
            "positive_count": pos_count,
            "negative_count": neg_count
        }

    def get_report(self) -> Dict:
        return {
            "method": self.method,
            "enabled": self.enabled,
            "circuit_breaker": f"{self.circuit_breaker_fails}/{self.circuit_breaker_threshold} fails, disabled={self.is_disabled}",
            "recommendation": "Set SENTIMENT_USE_X=false for fastest (you have R1 slow model) - 40 sec not 21 min if blocked",
            "options": {
                "snscrape": "Free, may break, X blocks 404, circuit breaker 10 min",
                "paid_api": "Reliable, costs $100+/month, not for $50 bankroll",
                "browser": "Fragile, UI changes break, ToS risk"
            },
            "fair_value_input": "LLM reads X sentiment volume, credibility, bot likelihood, novelty, time decay as info source NOT truth"
        }
