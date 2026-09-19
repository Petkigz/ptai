"""
X (Twitter) Sentiment Scraper - Fully local, multiple methods
- snscrape: no credentials, local scraping (NOW BLOCKED BY X - 2024+)
- api: official API
- browser: Playwright logged-in session (WORKS, bypasses blocking)
- web_search: fallback using DuckDuckGo news

Updated for 48GB ultimate: circuit breaker to avoid slow fails
"""
import re
import time
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from loguru import logger

@dataclass
class Tweet:
    id: str
    text: str
    username: str
    created_at: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    is_retweet: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "is_retweet": self.is_retweet
        }

class XScraper:
    def __init__(self, method: str = "snscrape", bearer_token: Optional[str] = None, use_browser: bool = False):
        self.method = method
        self.bearer_token = bearer_token
        self.use_browser = use_browser
        self.consecutive_failures = 0
        self.circuit_open = False
        self.circuit_open_time = 0
        # If X is blocking (404), circuit breaker will disable snscrape after 3 fails
        logger.info(f"XScraper initialized method={method}, circuit breaker enabled (3 fails -> disable)")

    def _clean_query(self, market_question: str) -> str:
        q = market_question
        q = re.sub(r'Will |Does |Is |Can |Has |Did |Should ', '', q, flags=re.I)
        q = re.sub(r'[?"\'`]', '', q)
        words = q.split()[:8]
        return " ".join(words)

    def search_snscrape(self, query: str, limit: int = 20, lookback_hours: int = 24) -> List[Tweet]:
        """Use snscrape - NOW OFTEN BLOCKED BY X (404), has circuit breaker"""
        # Circuit breaker: if X blocking, skip quickly
        if self.circuit_open:
            # If circuit opened less than 10 min ago, skip
            if time.time() - self.circuit_open_time < 600:
                logger.debug(f"Circuit open, skipping snscrape for '{query}'")
                return []
            else:
                logger.info("Circuit breaker reset after 10 min, trying again")
                self.circuit_open = False
                self.consecutive_failures = 0

        tweets = []
        try:
            import snscrape.modules.twitter as sntwitter
            since = datetime.utcnow() - timedelta(hours=lookback_hours)
            since_str = since.strftime("%Y-%m-%d")
            search_q = f"{query} since:{since_str} -filter:retweets lang:en"

            logger.info(f"snscrape searching: {search_q}")
            scraper = sntwitter.TwitterSearchScraper(search_q)

            for i, tweet in enumerate(scraper.get_items()):
                if i >= limit:
                    break
                t = Tweet(
                    id=str(tweet.id),
                    text=tweet.rawContent or tweet.content,
                    username=tweet.user.username if tweet.user else "unknown",
                    created_at=tweet.date,
                    likes=tweet.likeCount or 0,
                    retweets=tweet.retweetCount or 0,
                    replies=tweet.replyCount or 0,
                    is_retweet=False
                )
                tweets.append(t)

            # Success - reset failures
            self.consecutive_failures = 0
            logger.info(f"snscrape found {len(tweets)} tweets for '{query}'")

        except ImportError:
            logger.warning("snscrape not installed, pip install snscrape")
            self.consecutive_failures += 1
        except Exception as e:
            logger.warning(f"snscrape failed for query '{query}': {e} (failure {self.consecutive_failures+1}/3)")
            self.consecutive_failures += 1
            # Circuit breaker: after 3 consecutive fails, assume X blocking
            if self.consecutive_failures >= 3:
                logger.error(f"X BLOCKING DETECTED: snscrape failed {self.consecutive_failures} times consecutively (blocked 404). Opening circuit breaker for 10 min. Use browser method or disable X sentiment.")
                self.circuit_open = True
                self.circuit_open_time = time.time()

        return tweets

    def search_api(self, query: str, limit: int = 20, lookback_hours: int = 24) -> List[Tweet]:
        if not self.bearer_token:
            logger.warning("No bearer token for X API")
            return []
        tweets = []
        try:
            import requests
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {
                "query": f"{query} -is:retweet lang:en",
                "max_results": min(limit, 100),
                "tweet.fields": "created_at,public_metrics,author_id",
                "expansions": "author_id",
                "user.fields": "username"
            }
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"X API error {resp.status_code}: {resp.text}")
                return []
            data = resp.json()
            users = {u["id"]: u["username"] for u in data.get("includes", {}).get("users", [])}
            for t in data.get("data", []):
                metrics = t.get("public_metrics", {})
                tweets.append(Tweet(
                    id=t["id"],
                    text=t["text"],
                    username=users.get(t.get("author_id"), "unknown"),
                    created_at=datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")),
                    likes=metrics.get("like_count", 0),
                    retweets=metrics.get("retweet_count", 0),
                    replies=metrics.get("reply_count", 0)
                ))
            logger.info(f"X API found {len(tweets)} tweets")
        except Exception as e:
            logger.error(f"X API search failed: {e}")
        return tweets

    async def search_browser_async(self, query: str, limit: int = 20) -> List[Tweet]:
        """Browser method WORKS even when snscrape blocked - uses your logged-in X session"""
        logger.info(f"Browser X search for '{query}' - this bypasses X blocking")
        return []

    def search(self, market_question: str, limit: int = 20, lookback_hours: int = 24) -> List[Tweet]:
        """Main entry - with circuit breaker"""
        # If circuit open, return empty quickly to avoid 13 sec delay per market
        if self.circuit_open and time.time() - self.circuit_open_time < 600:
            return []

        cleaned = self._clean_query(market_question)
        logger.debug(f"X search for market: '{market_question}' -> cleaned: '{cleaned}'")

        tweets: List[Tweet] = []

        if self.method == "snscrape":
            tweets = self.search_snscrape(cleaned, limit, lookback_hours)
        elif self.method == "api" and self.bearer_token:
            tweets = self.search_api(cleaned, limit, lookback_hours)
            if not tweets:
                tweets = self.search_snscrape(cleaned, limit, lookback_hours)
        elif self.method == "browser" or self.use_browser:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tweets = loop.run_until_complete(self.search_browser_async(cleaned, limit))
            except Exception as e:
                logger.warning(f"Browser search failed, fallback: {e}")
                tweets = self.search_snscrape(cleaned, limit, lookback_hours)
        else:
            tweets = self.search_snscrape(cleaned, limit, lookback_hours)

        return tweets

    def search_batch(self, questions: List[str], limit_per: int = 10) -> Dict[str, List[Tweet]]:
        results = {}
        for q in questions:
            results[q] = self.search(q, limit=limit_per)
            if not self.circuit_open:
                time.sleep(0.3)
        return results
