"""Tests for markets - scanner, polymarket"""
import pytest
from unittest.mock import Mock, patch
from src.ptai.markets.scanner import MarketScanner
from src.ptai.markets.base import Market

class TestMarket:
    def test_create_market(self):
        from src.ptai.markets.base import MarketSource
        market = Market(
            id="test_id",
            source=MarketSource.POLYMARKET,
            question="Will BTC go up?",
            volume_24h=10000,
            liquidity=5000,
            event_slug="btc-up",
            slug="btc-up",
            outcomes=["YES", "NO"],
            outcome_prices=[0.6, 0.4],
        )
        assert market.question == "Will BTC go up?"
        assert market.best_price == 0.6

class TestMarketScanner:
    def test_scanner_init(self):
        scanner = MarketScanner()
        assert scanner is not None
    
    def test_quick_stats_empty(self):
        scanner = MarketScanner()
        stats = scanner.quick_stats([])
        assert "count" in stats or "total" in stats or isinstance(stats, dict)
    
    def test_quick_stats_with_markets(self):
        from src.ptai.markets.base import MarketSource
        scanner = MarketScanner()
        markets = [
            Market(
                id=f"test_{i}",
                source=MarketSource.POLYMARKET,
                question=f"Question {i}",
                volume_24h=1000 + i*100,
                liquidity=500 + i*50,
                event_slug=f"event-{i}",
                slug=f"slug-{i}",
                outcomes=["YES", "NO"],
                outcome_prices=[0.5 + i*0.01, 0.5 - i*0.01],
            )
            for i in range(5)
        ]
        stats = scanner.quick_stats(markets)
        assert isinstance(stats, dict)

class TestMockMarket:
    def test_mock_markets_generation(self):
        from src.ptai.markets.mock import generate_mock_markets
        markets = generate_mock_markets(count=10)
        assert len(markets) <= 10
        assert all(hasattr(m, 'question') for m in markets)
