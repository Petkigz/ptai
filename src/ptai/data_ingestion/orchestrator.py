"""
Data Ingestion Orchestrator - The Senses
Gathers all necessary data, API-First critical for reliability

From blueprint:
1. Polymarket Market Data: official Python SDK
2. X Sentiment: scraper or API, rate limits, ToS
3. Fair Value Inputs: news RSS, LLM assesses probabilities

Local-first, modular, runs on PC without cloud
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from ..markets.base import Market
from .polymarket_ingestion import PolymarketIngestion
from .news_ingestion import NewsIngestion
from .x_ingestion import XIngestion


class DataIngestionOrchestrator:
    """
    Orchestrates all data ingestion - API-First
    """
    def __init__(self, use_x: bool = False):
        self.polymarket = PolymarketIngestion()
        self.news = NewsIngestion()
        self.x = XIngestion(enabled=use_x)
        self.use_x = use_x

    def fetch_all_for_market(self, market: Market) -> Dict[str, Any]:
        """
        Fetch all data for single market - for fair value engine
        """
        data = {
            "market": market,
            "orderbook": self.polymarket.fetch_orderbook_api_first(market),
            "news": self.news.fetch_news_for_market(market, max_articles=3),
            "x_tweets": self.x.fetch_tweets(query=market.question[:50], max_tweets=20) if self.use_x else [],
            "x_sentiment": {},
            "timestamp": market.raw.get("timestamp") if isinstance(market.raw, dict) else None
        }
        
        if data["x_tweets"]:
            data["x_sentiment"] = self.x.analyze_sentiment(data["x_tweets"])
        
        logger.info(f"Ingestion for {market.id}: orderbook spread {data['orderbook'].get('spread')}, news {len(data['news'])}, x {len(data['x_tweets'])}")
        return data

    def fetch_markets_bulk(self, target_count: int = 500) -> List[Market]:
        """Fetch bulk markets API-First"""
        return self.polymarket.fetch_markets_api_first(target_count=target_count)

    def get_full_report(self) -> Dict:
        return {
            "orchestrator": "Data Ingestion - The Senses",
            "polymarket": self.polymarket.get_ingestion_report(),
            "news": self.news.get_report(),
            "x": self.x.get_report(),
            "api_first": "Critical for reliability - SDK not scraping web UI",
            "local_first": "Runs on PC without cloud, Ollama local LLM, no API costs",
            "modular": "Each ingestion module independent, can disable X for fastest"
        }
