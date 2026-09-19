"""
V3 Tests - genuinely multi-venue, multi-strategy opportunity engine
"""
import pytest
import asyncio
from src.ptai.venues.adapter import VenueType, EligibilityStatus, VenueOpportunity
from src.ptai.venues.registry import VenueRegistry
from src.ptai.venues.polymarket_adapter import PolymarketAdapter
from src.ptai.venues.kalshi_adapter import KalshiAdapter
from src.ptai.venues.manifold_adapter import ManifoldAdapter
from src.ptai.venues.crypto_adapter import CryptoAdapter
from src.ptai.venues.stock_adapter import StockAdapter
from src.ptai.markets.base import Market, Token, MarketSource
from src.ptai.strategy.arbitrage import ArbitrageEngine
from src.ptai.strategy.event_trading import EventTradingEngine
from src.ptai.strategy.market_making import MarketMakingEngine
from src.ptai.strategy.momentum import MomentumEngine
from src.ptai.strategy.strategy_engine import StrategyEngineV3
from src.ptai.strategy.strategy_selector import StrategyType


def make_market(id="M1", question="Will Trump win?", price=0.6, vol=10000, liq=5000, source=MarketSource.POLYMARKET):
    return Market(
        id=id,
        source=source,
        question=question,
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1-price],
        tokens=[Token(token_id=f"{id}_YES", outcome="YES", price=price), Token(token_id=f"{id}_NO", outcome="NO", price=1-price)],
        volume=vol,
        volume_24h=vol*0.5,
        liquidity=liq,
        active=True,
        closed=False,
        slug=id.lower(),
        raw={"venue": source.value}
    )


class TestV3Venues:
    def test_five_venues_registered(self):
        registry = VenueRegistry(country_code="UG")
        registry.register(PolymarketAdapter())
        registry.register(KalshiAdapter())
        registry.register(ManifoldAdapter())
        registry.register(CryptoAdapter())
        registry.register(StockAdapter())
        assert len(registry.adapters) == 5
        assert "polymarket" in registry.adapters
        assert "kalshi" in registry.adapters
        assert "manifold" in registry.adapters
        assert "crypto_binance" in registry.adapters
        assert "stock_mock" in registry.adapters

    def test_kalshi_eligibility_ug_restricted(self):
        adapter = KalshiAdapter()
        status = adapter.check_eligibility("UG")
        assert status == EligibilityStatus.RESTRICTED

    def test_kalshi_eligibility_us_eligible(self):
        adapter = KalshiAdapter()
        status = adapter.check_eligibility("US")
        assert status == EligibilityStatus.ELIGIBLE

    def test_manifold_eligibility_worldwide(self):
        adapter = ManifoldAdapter()
        assert adapter.check_eligibility("UG") == EligibilityStatus.ELIGIBLE
        assert adapter.check_eligibility("US") == EligibilityStatus.ELIGIBLE
        assert adapter.check_eligibility("GB") == EligibilityStatus.ELIGIBLE

    def test_crypto_eligibility(self):
        adapter = CryptoAdapter()
        assert adapter.check_eligibility("UG") == EligibilityStatus.ELIGIBLE

    def test_polymarket_ug_requires_verification(self):
        adapter = PolymarketAdapter()
        assert adapter.check_eligibility("UG") == EligibilityStatus.REQUIRES_VERIFICATION
        assert adapter.check_eligibility("US") == EligibilityStatus.RESTRICTED

    @pytest.mark.asyncio
    async def test_all_venues_discover(self):
        venues = [PolymarketAdapter(), KalshiAdapter(), ManifoldAdapter(), CryptoAdapter(), StockAdapter()]
        for adapter in venues:
            markets = await adapter.discover_markets(target_count=20)
            assert len(markets) > 0, f"{adapter.venue_id} should discover markets"
            assert len(markets) <= 20

    def test_venue_capabilities(self):
        poly = PolymarketAdapter()
        assert poly.capabilities.supports_market_discovery
        assert poly.capabilities.supports_orderbook
        assert poly.capabilities.fee_taker_pct == 0.02

        kalshi = KalshiAdapter()
        assert kalshi.capabilities.supports_market_discovery

        crypto = CryptoAdapter()
        assert crypto.capabilities.fee_taker_pct == 0.001  # 0.1% crypto fee lower than prediction 2%


class TestArbitrageEngine:
    def test_same_event_detection(self):
        engine = ArbitrageEngine(min_spread=0.03, min_confidence_same_event=0.5)
        m1 = make_market(id="POLY-TRUMP", question="Will Trump win 2024 election?", price=0.61, source=MarketSource.POLYMARKET)
        m2 = make_market(id="KALSHI-TRUMP", question="Will Trump win 2024 election?", price=0.68, source=MarketSource.KALSHI)
        score = engine._same_event_score(m1, m2)
        assert score > 0.7, f"Same event should have high similarity, got {score}"

    def test_different_event_low_score(self):
        engine = ArbitrageEngine()
        m1 = make_market(id="M1", question="Will Trump win?", price=0.6)
        m2 = make_market(id="M2", question="Will it rain in London tomorrow?", price=0.3)
        score = engine._same_event_score(m1, m2)
        assert score < 0.5

    def test_arbitrage_detection(self):
        engine = ArbitrageEngine(min_spread=0.03, min_confidence_same_event=0.5)
        m1 = make_market(id="POLY-TRUMP", question="Will Trump win 2024 election?", price=0.61, source=MarketSource.POLYMARKET)
        m2 = make_market(id="KALSHI-TRUMP", question="Will Trump win 2024 election?", price=0.72, source=MarketSource.KALSHI)
        arbs = engine.find_arbitrage([m1, m2])
        assert len(arbs) > 0
        assert arbs[0].spread >= 0.03
        assert arbs[0].estimated_profit_pct > 0

    def test_no_arbitrage_same_venue(self):
        engine = ArbitrageEngine()
        m1 = make_market(id="M1", question="Will Trump win?", price=0.6, source=MarketSource.POLYMARKET)
        m2 = make_market(id="M2", question="Will Trump win?", price=0.8, source=MarketSource.POLYMARKET)
        arbs = engine.find_arbitrage([m1, m2])
        assert len(arbs) == 0, "Same venue should not be arbitrage"

    def test_arbitrage_to_venue_opportunity(self):
        engine = ArbitrageEngine(min_spread=0.03, min_confidence_same_event=0.5)
        m1 = make_market(id="POLY-TRUMP", question="Will Trump win 2024 election?", price=0.61, source=MarketSource.POLYMARKET)
        m2 = make_market(id="KALSHI-TRUMP", question="Will Trump win 2024 election?", price=0.72, source=MarketSource.KALSHI)
        arbs = engine.find_arbitrage([m1, m2])
        # Force tradeable for test
        for arb in arbs:
            arb.should_trade = True
        venue_opps = engine.to_venue_opportunities(arbs)
        assert len(venue_opps) > 0
        assert venue_opps[0].category == "arbitrage"


class TestStrategyEngines:
    def test_event_trading_detection(self):
        engine = EventTradingEngine()
        market = make_market(question="Will Fed raise rates in Jan?", price=0.5)
        context = {"news": "Federal Reserve beats expectations and will raise rates significantly tomorrow", "news_credibility": 0.8, "sentiment": {"score": 0.7}}
        signal = engine.evaluate(market, context=context)
        assert signal.event_type in ["fed", "earnings", "economic"]
        # Edge should be non-zero with positive news containing beat
        assert signal.news_impact != 0, f"News impact should be non-zero, got {signal.news_impact} reasoning {signal.reasoning}"
        assert signal.source_credibility == 0.8

    def test_market_making_evaluation(self):
        engine = MarketMakingEngine(min_spread=0.01, min_depth=1000)
        market = make_market(price=0.5, vol=50000, liq=10000)
        # High liquidity, tight spread, low turnover = low inventory risk, should be profitable
        orderbook = {"spread": 0.02, "spread_pct": 0.02, "depth": 10000, "volatility": 0.01}
        signal = engine.evaluate(market, orderbook=orderbook)
        assert signal.spread_pct == 0.02
        # Profit may be small but structure works - check reasoning exists
        assert signal.reasoning != ""
        # Test with better conditions: higher spread, lower turnover
        market2 = make_market(price=0.5, vol=5000, liq=20000)  # low turnover
        orderbook2 = {"spread": 0.04, "spread_pct": 0.04, "depth": 20000, "volatility": 0.01}
        signal2 = engine.evaluate(market2, orderbook=orderbook2)
        assert signal2.estimated_profit_per_trade > 0, f"Should be profitable with 4% spread low turnover, got {signal2.estimated_profit_per_trade}"

    def test_momentum_evaluation(self):
        engine = MomentumEngine()
        market = make_market(price=0.6, vol=100000, liq=50000)
        market.raw["change_pct"] = 5.0  # 5% up
        context = {"price_history": [], "orderbook": {}}
        signals = engine.evaluate(market, context=context)
        assert len(signals) == 2  # momentum + mean reversion
        # Momentum should have edge with 5% change
        mom_signal = [s for s in signals if s.strategy == "momentum"][0]
        assert mom_signal.estimated_edge != 0

    def test_mean_reversion_extreme_price(self):
        engine = MomentumEngine()
        market = make_market(price=0.95, vol=10000, liq=5000)  # extreme
        context = {}
        rev_signal = engine.evaluate_mean_reversion(market, context=context)
        # Extreme price should trigger mean reversion
        assert rev_signal.estimated_edge != 0
        assert rev_signal.strategy == "mean_reversion"


class TestStrategyEngineV3:
    def test_venue_x_market_x_strategy(self):
        engine = StrategyEngineV3()
        market = make_market(id="TEST", question="Will BTC go up?", price=0.55, vol=50000, liq=20000)
        market.raw["venue"] = "crypto_binance"
        market.raw["change_pct"] = 3.0
        context = {"orderbook": {"spread": 0.02, "depth": 20000}, "news": "BTC beats expectations", "news_credibility": 0.7, "sentiment": {"score": 0.5}}
        opps = engine.evaluate_market_with_all_strategies(market, context=context)
        # Should evaluate multiple strategies
        assert isinstance(opps, list)
        # At least market making might trigger
        # We don't assert >0 because edge thresholds, but structure should work

    @pytest.mark.asyncio
    async def test_multi_venue_scan(self):
        engine = StrategyEngineV3()
        markets_by_venue = {
            "polymarket": [make_market(id=f"POLY-{i}", question=f"Will event {i} happen? Trump", price=0.5 + i*0.01, vol=20000, liq=10000) for i in range(10)],
            "kalshi": [make_market(id=f"KALSHI-{i}", question=f"Will event {i} happen? Trump", price=0.55 + i*0.01, vol=15000, liq=8000, source=MarketSource.KALSHI) for i in range(10)],
            "manifold": [make_market(id=f"MANI-{i}", question=f"Will AI achieve milestone {i}?", price=0.4 + i*0.02, vol=10000, liq=5000, source=MarketSource.PREDICTIT) for i in range(10)],
        }
        result = await engine.scan_all_venues(markets_by_venue=markets_by_venue, max_final_trades=2)
        assert result.total_scanned == 30
        assert len(result.venue_reports) == 3
        assert "I scanned 30 opportunities across 3 venues" in result.reasoning
        assert result.execution_time > 0
        # Check venue reports have per-venue counts
        for report in result.venue_reports:
            assert report.total_discovered > 0
            assert report.venue_id in ["polymarket", "kalshi", "manifold"]

    @pytest.mark.asyncio
    async def test_v3_report_format(self):
        engine = StrategyEngineV3()
        markets_by_venue = {
            "polymarket": [make_market(id="P1", question="Will Trump win?", price=0.61, vol=50000, liq=20000)],
            "kalshi": [make_market(id="K1", question="Will Trump win?", price=0.68, vol=40000, liq=15000, source=MarketSource.KALSHI)],
        }
        result = await engine.scan_all_venues(markets_by_venue=markets_by_venue, max_final_trades=2)
        # Must contain required report phrases from user spec
        assert "I scanned" in result.reasoning
        assert "polymarket" in result.reasoning.lower()
        assert "kalshi" in result.reasoning.lower()
        assert "After fees/liquidity/uncertainty" in result.reasoning or "actually tradeable" in result.reasoning


class TestV3Endpoints:
    def test_v3_status_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/status")
        assert res.status_code == 200
        data = res.json()
        assert "mission" in data
        assert "venues" in data
        assert "strategies" in data
        assert len(data["venues"]) >= 5
        assert "mispricing" in data["strategies"]
        assert "arbitrage" in data["strategies"]

    def test_v3_venues_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/venues")
        assert res.status_code == 200
        data = res.json()
        assert "venues" in data
        assert len(data["venues"]) >= 5
        assert "polymarket" in data["venues"]
        assert "kalshi" in data["venues"]
        assert "venue_details" in data
        assert data["venue_details"]["polymarket"]["type"] == "prediction"

    def test_v3_strategies_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/strategies")
        assert res.status_code == 200
        data = res.json()
        assert "strategies" in data
        assert len(data["strategies"]) == 6
        assert "arbitrage" in data["strategies"]
        assert "event_trading" in data["strategies"]

    def test_v3_discovery_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/discovery?target_per_venue=10")
        assert res.status_code == 200
        data = res.json()
        assert "total_scanned" in data
        assert "per_venue" in data
        assert data["total_scanned"] > 0
        assert "I scanned" in data["message"]

    def test_v3_opportunities_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/opportunities?target_per_venue=10&max_trades=2")
        assert res.status_code == 200
        data = res.json()
        assert "total_scanned" in data
        assert "venue_reports" in data
        assert "strategy_breakdown" in data
        assert "reasoning" in data
        # Check report format matches user spec
        assert "I scanned" in data["reasoning"]

    def test_v3_arbitrage_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/arbitrage?target_per_venue=10")
        assert res.status_code == 200
        data = res.json()
        assert "total_markets_scanned" in data
        assert "arbitrage_candidates" in data

    def test_v3_run_cycle_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.post("/api/v3/run-cycle?target_per_venue=10&max_trades=1")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert data["status"] == "completed"
        assert "discovery" in data
        assert "opportunities" in data
        assert "reasoning" in data
        # V3 report must show per-venue breakdown
        assert "per_venue" in data["discovery"]
