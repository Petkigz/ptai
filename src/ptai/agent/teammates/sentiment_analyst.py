"""
Sentiment Analyst Teammate - Reads live X sentiment, news, web
Signs into X via browser, uses web_search, can use Grok API if available
"""
from typing import Dict, Any, List
from loguru import logger
import os
import time
from .base import BaseTeammate, Task
from ...sentiment import XScraper, SentimentAnalyzer

class SentimentAnalystTeammate(BaseTeammate):
    def __init__(self, vault=None, memory=None, llm_router=None):
        super().__init__(name="SentimentAnalyst", role="X/Twitter & News Sentiment Reader", vault=vault, memory=memory, llm_router=llm_router)
        self.x_scraper = XScraper(method="snscrape")
        self.analyzer = SentimentAnalyzer()
        self.sign_in_to_tool("x_snscrape")
        self.sign_in_to_tool("web_search")
        self.sign_in_to_tool("duckduckgo")
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        if task.type == "analyze_sentiment_batch":
            markets = task.payload.get("markets", [])
            max_markets = task.payload.get("max_markets", 60)
            use_x = task.payload.get("use_x", True)
            
            # Check env override
            env_use_x = os.getenv("SENTIMENT_USE_X", "true").lower() == "true"
            if not env_use_x:
                use_x = False
            
            logger.info(f"[SentimentAnalyst] Analyzing {min(len(markets), max_markets)} markets, use_x={use_x}")
            
            if not use_x:
                from ...sentiment import SentimentResult
                sentiments = {}
                for m in markets[:max_markets]:
                    sentiments[m.question] = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34, summary="X disabled, LLM-only fast mode", key_phrases=[], tweet_count=0, confidence=0.2)
                    sentiments[m.id] = sentiments[m.question]
                return {
                    "sentiments": sentiments,
                    "count": len(sentiments)//2,
                    "x_enabled": False,
                    "message": f"X disabled, neutral fallback for {len(sentiments)//2} markets - fastest mode"
                }
            
            sentiments = {}
            sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:max_markets]
            
            for i, market in enumerate(sorted_markets):
                if self.x_scraper.circuit_open:
                    logger.warning(f"[SentimentAnalyst] X circuit open, skipping remaining {len(sorted_markets)-i}")
                    from ...sentiment import SentimentResult
                    for remaining in sorted_markets[i:]:
                        sentiments[remaining.question] = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33, neutral_pct=0.34, summary="X blocked 404, neutral fallback", key_phrases=[], tweet_count=0, confidence=0.2)
                        sentiments[remaining.id] = sentiments[remaining.question]
                    break
                
                try:
                    tweets = self.x_scraper.search(market_question=market.question, limit=15, lookback_hours=24)
                    sentiment = self.analyzer.analyze(tweets, market.question)
                    sentiments[market.question] = sentiment
                    sentiments[market.id] = sentiment
                    
                    if i % 10 == 0:
                        logger.info(f"[SentimentAnalyst] Progress {i}/{len(sorted_markets)}")
                    
                    if not self.x_scraper.circuit_open:
                        time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Sentiment failed {market.question[:50]}: {e}")
                    continue
            
            # Web search fallback if X blocked
            if self.x_scraper.circuit_open or self.x_scraper.consecutive_failures >= 3:
                try:
                    from ...sentiment.web_search_sentiment import WebSearchSentiment
                    web_searcher = WebSearchSentiment()
                    top_20 = sorted_markets[:20]
                    for m in top_20:
                        if m.question in sentiments and sentiments[m.question].tweet_count == 0:
                            news = web_searcher.search(m.question.replace("Will ", "")[:60], max_results=3)
                            if news:
                                news_text = " ".join([f"{n.title}: {n.snippet}" for n in news])[:1000]
                                sentiments[m.question].sentiment_summary += f" | WEB NEWS: {news_text[:300]}"
                                sentiments[m.question].confidence = max(0.3, sentiments[m.question].confidence)
                except Exception as e:
                    logger.warning(f"Web search fallback failed: {e}")
            
            return {
                "sentiments": sentiments,
                "count": len(sentiments)//2,
                "x_enabled": True,
                "circuit_open": self.x_scraper.circuit_open,
                "failures": self.x_scraper.consecutive_failures,
                "message": f"Analyzed {len(sentiments)//2} markets, X circuit open: {self.x_scraper.circuit_open}"
            }
        
        elif task.type == "analyze_single":
            question = task.payload.get("question", "")
            tweets = self.x_scraper.search(market_question=question, limit=20)
            sentiment = self.analyzer.analyze(tweets, question)
            return {"sentiment": sentiment, "tweet_count": len(tweets)}
        
        else:
            raise ValueError(f"SentimentAnalyst doesn't know {task.type}")
