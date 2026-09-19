"""
News Ingestion - Fair Value Inputs
For news, use RSS feeds or local news API. LLM reads to assess probabilities
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime, timezone


class NewsIngestion:
    """
    News ingestion for fair value engine
    RSS feeds or local news API, no cloud
    """
    def __init__(self):
        self.rss_feeds = [
            "https://feeds.reuters.com/reuters/topNews",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PTAI/1.0 Local News Ingestion"})

    def fetch_news_rss(self, max_articles: int = 10) -> List[Dict[str, Any]]:
        """Fetch news via RSS - local-first, no API costs"""
        articles = []
        # Mock for offline - in production would parse RSS
        # For demo, return mock reputable sources
        mock_news = [
            {"title": "Fed holds rates steady, signals future cuts", "source": "Reuters", "credibility": 0.95, "timestamp": datetime.now(timezone.utc), "summary": "Federal Reserve holds rates, Powell signals cuts if inflation cools"},
            {"title": "Unemployment drops to 3.8%, beats expectations", "source": "Bloomberg", "credibility": 0.9, "timestamp": datetime.now(timezone.utc), "summary": "Jobs report beats, unemployment down"},
            {"title": "Trump leads polls in key swing states", "source": "AP", "credibility": 0.85, "timestamp": datetime.now(timezone.utc), "summary": "Polling data shows Trump lead"},
            {"title": "Bitcoin surges past $50k on ETF inflows", "source": "CoinDesk", "credibility": 0.8, "timestamp": datetime.now(timezone.utc), "summary": "BTC up on ETF demand"},
        ]
        articles.extend(mock_news[:max_articles])
        logger.info(f"News ingestion: {len(articles)} articles via RSS (mock for offline)")
        return articles

    def fetch_news_for_market(self, market, max_articles: int = 3) -> List[Dict]:
        """Fetch news relevant to market question"""
        # Simple keyword matching
        question = market.question.lower()
        all_news = self.fetch_news_rss(max_articles=10)
        relevant = []
        for article in all_news:
            # Check if any keyword from question in title
            keywords = question.split()[:5]  # first 5 words
            for kw in keywords:
                if len(kw) > 3 and kw in article["title"].lower():
                    relevant.append(article)
                    break
        return relevant[:max_articles]

    def get_report(self) -> Dict:
        return {
            "method": "RSS feeds + local news API",
            "feeds": self.rss_feeds,
            "advantages": ["No API costs", "Local-first", "Reputable sources", "Recency"],
            "fair_value_input": "LLM reads headlines to assess probabilities",
            "mock_note": "Offline mock for demo, real would parse RSS XML"
        }
