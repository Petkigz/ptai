"""Tests for PTAI v2 - market-agnostic autonomous engine"""
import tempfile
from pathlib import Path

class TestMarketAdapter:
    def test_adapter_interface(self):
        from src.ptai.venues.adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
        assert VenueType.PREDICTION == "prediction"
        assert EligibilityStatus.ELIGIBLE == "eligible"
        
        cap = AdapterCapability(supports_trading=True, fee_taker_pct=0.02)
        assert cap.fee_taker_pct == 0.02

    def test_venue_opportunity_scoring(self):
        from src.ptai.venues.adapter import VenueOpportunity, VenueType
        from src.ptai.markets.base import Market, MarketSource
        
        market = Market(
            id="test1",
            source=MarketSource.POLYMARKET,
            question="Will BTC go up?",
            outcome_prices=[0.6, 0.4],
            outcomes=["YES", "NO"],
            volume_24h=10000,
            liquidity=5000
        )
        
        opp = VenueOpportunity(
            market=market,
            venue_id="polymarket",
            venue_type=VenueType.PREDICTION,
            side="YES",
            market_price=0.6,
            estimated_fair=0.73,
            raw_edge=0.13,
            effective_edge=0.10,
            confidence=0.8,
            uncertainty=0.05,
            liquidity_score=0.8,
            execution_quality=0.9,
            fees_pct=0.01,
            spread_pct=0.02,
            slippage_pct=0.01
        )
        
        score = opp.calculate_common_score()
        assert score > 0
        assert opp.score == score

    def test_polymarket_adapter_eligibility(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        from src.ptai.venues.adapter import EligibilityStatus
        
        adapter = PolymarketAdapter()
        # UG should be requires_verification or eligible, NOT restricted
        status = adapter.check_eligibility("UG")
        assert status != EligibilityStatus.RESTRICTED
        
        # US should be restricted
        status_us = adapter.check_eligibility("US")
        assert status_us == EligibilityStatus.RESTRICTED

    def test_venue_registry(self):
        from src.ptai.venues.registry import VenueRegistry
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        
        registry = VenueRegistry(country_code="UG")
        adapter = PolymarketAdapter()
        registry.register(adapter)
        
        assert "polymarket" in registry.adapters
        eligibility = registry.check_all_eligibility()
        assert "polymarket" in eligibility
        
        # Test ranking
        from src.ptai.venues.adapter import VenueOpportunity, VenueType
        from src.ptai.markets.base import Market, MarketSource
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Test", outcome_prices=[0.5,0.5], outcomes=["YES","NO"])
        opp = VenueOpportunity(market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION, side="YES", market_price=0.5, estimated_fair=0.7, raw_edge=0.2, effective_edge=0.15, confidence=0.8, uncertainty=0.05, liquidity_score=0.8, execution_quality=0.8, fees_pct=0.01, spread_pct=0.01, slippage_pct=0.01)
        opp.calculate_common_score()
        ranked = registry.rank_opportunities([opp])
        assert len(ranked) == 1


class TestIntelligence:
    def test_base_rate_model(self):
        from src.ptai.intelligence.forecaster import BaseRateModel
        from src.ptai.markets.base import Market, MarketSource
        
        model = BaseRateModel()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Will Trump win?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"], volume_24h=50000, liquidity=10000)
        forecast = model.forecast(market, category="politics")
        assert 0 < forecast.probability < 1
        assert forecast.model_name == "base_rate"

    def test_ensemble_forecaster(self):
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        from src.ptai.markets.base import Market, MarketSource
        
        forecaster = EnsembleForecaster()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Test?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"])
        
        # Test with context
        result = forecaster.forecast_market(market, context={"category": "politics", "news": "Positive news", "sentiment": {"score": 0.2}})
        assert result.market_id == "1"
        assert 0 < result.fair_probability < 1
        assert hasattr(result, 'conservative_fair')

    def test_calibration_engine(self):
        from src.ptai.intelligence.calibration import CalibrationEngine
        
        engine = CalibrationEngine()
        fid = engine.record_forecast(market_id="test1", question="Will BTC up?", forecast_prob=0.7, confidence=0.8, market_price=0.6, category="crypto")
        assert fid is not None
        
        engine.record_resolution(fid, actual_outcome=1.0)
        brier = engine.calculate_brier_score()
        assert 0 <= brier <= 1
        
        stats = engine.get_stats()
        assert "total_forecasts" in stats
        assert "brier_score" in stats

    def test_uncertainty_engine(self):
        from src.ptai.intelligence.uncertainty import UncertaintyEngine
        from src.ptai.intelligence.forecaster import ModelForecast
        
        engine = UncertaintyEngine()
        forecasts = [
            ModelForecast(model_name="base_rate", probability=0.68, confidence=0.6, uncertainty=0.15, reasoning="test", sources=[]),
            ModelForecast(model_name="news", probability=0.75, confidence=0.7, uncertainty=0.1, reasoning="test", sources=[]),
        ]
        
        uncertainty = engine.calculate_uncertainty(forecasts, context={"spread": 0.02, "sources": ["a","b","c"]})
        assert 0 <= uncertainty <= 0.3
        
        conservative = engine.conservative_estimate(probability=0.71, uncertainty=0.08)
        assert conservative < 0.71  # Should be more conservative
        assert abs(conservative - 0.63) < 0.01  # 71% - 8% = 63%
        
        effective_edge, cons_fair, reasoning = engine.effective_edge(market_price=0.6, fair_prob=0.71, uncertainty=0.08, fees=0.008, spread=0.02, slippage=0.01)
        assert isinstance(effective_edge, float)
        assert "Raw edge" in reasoning

    def test_contradiction_engine(self):
        from src.ptai.intelligence.contradiction import ContradictionEngine
        from src.ptai.markets.base import Market, MarketSource
        
        engine = ContradictionEngine()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Will X happen?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"], volume_24h=10000, liquidity=5000)
        report = engine.synthesize(market, research_text="Will likely succeed but risk of failure", news="Positive")
        assert report.market_id == "1"
        assert hasattr(report, 'bull_case')
        assert hasattr(report, 'bear_case')
        assert hasattr(report, 'net_score')

    def test_resolution_analyzer(self):
        from src.ptai.intelligence.resolution_analyzer import ResolutionAnalyzer
        from src.ptai.markets.base import Market, MarketSource
        
        analyzer = ResolutionAnalyzer()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Will BTC be above $100k?", description="Resolves YES if BTC >= $100k on Coinbase at 12pm ET Dec 31", outcome_prices=[0.6,0.4], outcomes=["YES","NO"])
        analysis = analyzer.analyze(market)
        assert analysis.market_id == "1"
        assert hasattr(analysis, 'should_trade')
        assert hasattr(analysis, 'risk_score')
        
        # Ambiguous market should be flagged
        ambiguous = Market(id="2", source=MarketSource.POLYMARKET, question="Will there be significant improvement?", description="Resolves based on credible reports", outcome_prices=[0.5,0.5], outcomes=["YES","NO"])
        analysis2 = analyzer.analyze(ambiguous)
        assert analysis2.risk_score > 0
        assert len(analysis2.ambiguous_language) > 0 or len(analysis2.risks) > 0


class TestStrategy:
    def test_edge_calculator(self):
        from src.ptai.strategy.edge import EdgeCalculator
        from src.ptai.markets.base import Market, MarketSource
        
        calc = EdgeCalculator()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Test?", outcome_prices=[0.61,0.39], outcomes=["YES","NO"], volume_24h=10000, liquidity=5000)
        
        effective = calc.calculate(market=market, fair_prob=0.73, uncertainty=0.05, orderbook={"spread": 0.02}, amount_usd=3.0)
        assert effective.raw_edge == 0.73 - 0.61
        assert effective.effective_edge < effective.raw_edge  # After deductions
        assert "Raw" in effective.reasoning
        assert hasattr(effective, 'should_trade')

    def test_fair_value_engine(self):
        from src.ptai.strategy.fair_value import FairValueEngine
        from src.ptai.markets.base import Market, MarketSource
        
        engine = FairValueEngine()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Will BTC go up? Resolves based on Coinbase price at 12pm ET", outcome_prices=[0.6,0.4], outcomes=["YES","NO"], volume_24h=10000, liquidity=5000)
        
        result = engine.estimate(market, context={"news": "BTC up", "category": "crypto", "orderbook": {"spread": 0.02}})
        assert result.market_id == "1"
        assert 0 < result.fair_value < 1
        assert hasattr(result, 'should_trade')
        assert hasattr(result, 'reasoning')

    def test_opportunity_engine(self):
        from src.ptai.strategy.opportunity import OpportunityEngine
        from src.ptai.markets.base import Market, MarketSource
        
        engine = OpportunityEngine()
        markets = [
            Market(id=f"{i}", source=MarketSource.POLYMARKET, question=f"Market {i}?", outcome_prices=[0.5 + i*0.01, 0.5 - i*0.01], outcomes=["YES","NO"], volume_24h=1000 + i*100, liquidity=500 + i*50, active=True, closed=False)
            for i in range(20)
        ]
        
        filtered = engine.cheap_filters(markets)
        assert len(filtered) <= len(markets)
        
        liq_filtered = engine.liquidity_filter(filtered)
        assert len(liq_filtered) <= len(filtered)
        
        fast_filtered = engine.fast_model_screen(liq_filtered)
        assert len(fast_filtered) <= 50


class TestRiskV2:
    def test_exposure_manager(self):
        from src.ptai.risk.exposure import ExposureManager
        
        manager = ExposureManager(bankroll=100.0)
        can_open, reason = manager.can_open(market_id="test1", amount_usd=5.0, category="politics", correlation_group="trump")
        assert can_open is True
        
        manager.add_position(market_id="test1", amount_usd=5.0, category="politics", correlation_group="trump")
        exposure = manager.get_exposure()
        assert exposure.total_exposure_usd == 5.0
        
        # Try to exceed cap
        can_open2, reason2 = manager.can_open(market_id="test1", amount_usd=10.0, category="politics")
        # Should fail if exceeds 6% cap (6% of 100 = 6, existing 5 + 10 = 15 >6)
        assert can_open2 is False
        assert "cap" in reason2.lower()

    def test_correlation_engine(self):
        from src.ptai.risk.correlation import CorrelationEngine
        from src.ptai.markets.base import Market, MarketSource
        
        engine = CorrelationEngine()
        market_a = Market(id="1", source=MarketSource.POLYMARKET, question="Will Trump win election?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"])
        market_b = Market(id="2", source=MarketSource.POLYMARKET, question="Will Republican win?", outcome_prices=[0.55,0.45], outcomes=["YES","NO"])
        market_c = Market(id="3", source=MarketSource.POLYMARKET, question="Will BTC go up?", outcome_prices=[0.5,0.5], outcomes=["YES","NO"])
        
        corr_ab = engine.calculate_correlation(market_a, market_b)
        corr_ac = engine.calculate_correlation(market_a, market_c)
        
        assert corr_ab > corr_ac  # Trump and Republican should correlate more than Trump and BTC
        assert corr_ab > 0.3

    def test_kill_switch(self):
        from src.ptai.risk.kill_switch import KillSwitch, KillLevel
        
        with tempfile.TemporaryDirectory() as tmpdir:
            kill = KillSwitch(data_dir=tmpdir)
            assert kill.current_level == KillLevel.NORMAL
            assert kill.can_trade() is True
            
            kill.trigger(KillLevel.NO_NEW_TRADES, "Daily loss 15%", {"daily_loss": 0.15})
            assert kill.current_level == KillLevel.NO_NEW_TRADES
            assert kill.can_trade() is False
            assert kill.should_cancel_orders() is False
            
            kill.trigger(KillLevel.CANCEL_ORDERS, "Execution mismatch", {"mismatch": 0.2})
            assert kill.should_cancel_orders() is True
            
            status = kill.get_status_report()
            assert "current_level" in status
            assert "can_trade" in status
            
            kill.reset("manual reset for test")
            assert kill.current_level == KillLevel.NORMAL
            assert kill.can_trade() is True

    def test_limits_engine(self):
        from src.ptai.risk.limits import LimitsEngine
        
        engine = LimitsEngine(bankroll=100.0)
        
        # Valid proposal
        proposal = {
            "market_id": "test1",
            "fair_probability": 0.73,
            "market_probability": 0.61,
            "edge": 0.12,
            "confidence": 0.8,
            "side": "YES",
            "trade": True,
            "max_price": 0.63
        }
        
        allowed, reason, adjusted = engine.validate_proposal(proposal)
        assert allowed is True
        assert "max_spend_usd" in adjusted
        assert adjusted["max_spend_usd"] <= 100 * 0.06 + 0.01  # 6% cap
        
        # Invalid - low edge
        proposal_low_edge = {**proposal, "edge": 0.02}
        allowed2, reason2, _ = engine.validate_proposal(proposal_low_edge)
        assert allowed2 is False
        assert "Edge" in reason2

    def test_drawdown_manager(self):
        from src.ptai.risk.drawdown import DrawdownManager
        
        manager = DrawdownManager(initial_bankroll=100.0)
        manager.update_bankroll(90.0, pnl=-10.0)
        
        state = manager.get_state()
        assert state.current_bankroll == 90.0
        assert state.drawdown_pct == 0.1
        
        triggers = manager.check_triggers()
        assert "should_pause" in triggers


class TestExecutionV2:
    def test_order_manager(self):
        from src.ptai.execution.order_manager import OrderManager
        
        manager = OrderManager()
        allowed, reason, order = manager.create_order(
            market_id="test1",
            token_id="token1",
            side="YES",
            max_price=0.65,
            max_spend_usd=5.0
        )
        assert allowed is True
        assert order is not None
        assert order.market_id == "test1"
        
        # Try to exceed absolute max
        allowed2, reason2, order2 = manager.create_order(
            market_id="test2",
            token_id="token2",
            side="YES",
            max_price=0.65,
            max_spend_usd=5000.0  # > $1000 absolute max
        )
        assert allowed2 is False
        assert "absolute max" in reason2

    def test_execution_guard(self):
        from src.ptai.execution.execution_guard import ExecutionGuard
        
        guard = ExecutionGuard(bankroll=100.0)
        
        proposal = {"market_id": "test1", "side": "YES"}
        risk_approved = {"market_id": "test1", "max_spend_usd": 5.0, "max_price": 0.65}
        
        result = guard.validate(proposal, risk_approved)
        assert result.allowed is True
        assert result.max_spend_usd == 5.0
        
        # Try insane amount from LLM - should be blocked by risk_approved, not proposal
        # Guard only checks risk_approved, so if risk says 5, guard allows 5 even if proposal says 50000
        # That's the point - LLM cannot override
        proposal_insane = {"market_id": "test1", "side": "YES", "amount": 50000}
        result2 = guard.validate(proposal_insane, risk_approved)
        assert result2.allowed is True
        assert result2.max_spend_usd == 5.0  # Still 5, not 50000
        
        # But if risk_approved itself is insane, guard blocks
        risk_insane = {"market_id": "test1", "max_spend_usd": 50000, "max_price": 0.65}
        result3 = guard.validate(proposal, risk_insane)
        assert result3.allowed is False

    def test_reconciliation(self):
        from src.ptai.execution.reconciliation import ReconciliationEngine
        from src.ptai.execution.order_manager import Order, OrderStatus
        from datetime import datetime, timezone
        
        engine = ReconciliationEngine()
        order = Order(
            id="test1",
            market_id="market1",
            token_id="token1",
            side="YES",
            max_price=0.65,
            max_spend_usd=5.0,
            amount=7.5,
            avg_price=0.65,
            status=OrderStatus.FILLED
        )
        
        before = {"balance": 100.0}
        after = {"balance": 95.0}
        
        result = engine.reconcile_order(order, before, after)
        assert result.market_id == "market1"
        assert hasattr(result, 'mismatch')


class TestLearning:
    def test_trade_outcome_tracker(self):
        from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
        
        tracker = TradeOutcomeTracker()
        tracker.record_trade(
            trade_id="test1",
            market_id="market1",
            venue_id="polymarket",
            strategy="mispricing",
            category="politics",
            forecast_prob=0.7,
            market_price=0.6,
            edge=0.1,
            side="YES",
            amount_usd=5.0
        )
        
        tracker.record_resolution(trade_id="test1", actual_outcome=1.0, pnl=2.0)
        
        venue_perf = tracker.get_venue_performance()
        assert "polymarket" in venue_perf
        assert venue_perf["polymarket"]["total"] == 1
        assert venue_perf["polymarket"]["win_rate"] == 1.0
        
        concentration = tracker.should_concentrate_on()
        assert "recommendation" in concentration

    def test_calibration_db(self):
        from src.ptai.learning.calibration_db import CalibrationDB
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db = CalibrationDB(db_path=str(Path(tmpdir) / "calibration.json"))
            fid = db.record_forecast(market_id="test1", question="Test?", forecast_prob=0.7, confidence=0.8, market_price=0.6, category="test")
            assert fid is not None
            
            report = db.get_report()
            assert "total_forecasts" in report
            assert "brier_score" in report


class TestInformation:
    def test_x_engine(self):
        from src.ptai.information.x_engine import XEngine
        from src.ptai.markets.base import Market, MarketSource
        import asyncio
        
        engine = XEngine()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Will BTC go up?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"])
        
        # Test credibility analysis
        tweets = [
            {"text": "BTC will go up!", "timestamp": "2024-01-01T00:00:00Z"},
            {"text": "BTC will go up!", "timestamp": "2024-01-01T00:00:00Z"},  # duplicate
            {"text": "BTC up!", "timestamp": "2024-01-01T00:00:00Z"},
        ]
        
        analysis = engine.analyze_tweets(tweets)
        assert "credibility" in analysis
        assert "bot_likelihood" in analysis
        assert analysis["unique_ratio"] < 1.0  # duplicates
        
        # Test full signal
        async def test_signal():
            signal = await engine.get_signal(market, news="BTC positive", web_research="BTC up")
            assert signal.market_id == "1"
            assert hasattr(signal, 'credibility')
            assert hasattr(signal, 'should_use')
        
        asyncio.run(test_signal())

    def test_market_normalizer(self):
        from src.ptai.markets.market_normalizer import MarketNormalizer
        from src.ptai.markets.base import MarketSource
        
        normalizer = MarketNormalizer()
        
        # Test polymarket normalization
        raw_event = {
            "id": "event1",
            "slug": "test-event",
            "title": "Test Event",
            "description": "Test",
            "volume": 10000,
            "volume24hr": 5000,
            "markets": [
                {
                    "id": "market1",
                    "question": "Will BTC go up?",
                    "outcomes": '["YES", "NO"]',
                    "outcomePrices": '["0.6", "0.4"]',
                    "clobTokenIds": '["token1", "token2"]',
                    "volume": 10000,
                    "volume24hr": 5000,
                    "liquidity": 2000,
                    "slug": "btc-up",
                    "conditionId": "cond1"
                }
            ]
        }
        
        markets = normalizer.normalize_polymarket(raw_event)
        assert len(markets) == 1
        assert markets[0].id == "market1"
        assert markets[0].question == "Will BTC go up?"

    def test_orderbook_analyzer(self):
        from src.ptai.markets.orderbook import OrderbookAnalyzer
        from src.ptai.markets.base import Market, MarketSource
        
        analyzer = OrderbookAnalyzer()
        market = Market(id="1", source=MarketSource.POLYMARKET, question="Test?", outcome_prices=[0.6,0.4], outcomes=["YES","NO"], volume_24h=10000, liquidity=5000)
        
        orderbook = {"bid": 0.59, "ask": 0.61, "spread": 0.02, "bid_size": 1000, "ask_size": 1000}
        trades = [{"price": 0.6, "size": 100}, {"price": 0.61, "size": 500}]  # second is large
        
        snapshot = analyzer.analyze(market, raw_orderbook=orderbook, recent_trades=trades)
        assert snapshot.market_id == "1"
        assert abs(snapshot.spread - 0.02) < 0.001
        assert abs(snapshot.imbalance - 0) < 0.001  # equal sizes
        
        is_liquid, reason = analyzer.is_liquid(snapshot, min_liquidity=500, max_spread=0.08)
        assert is_liquid is True
        
        risks = analyzer.detect_manipulation(snapshot)
        assert isinstance(risks, list)


class TestV2Endpoints:
    def test_v2_status_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        
        client = TestClient(app)
        response = client.get("/api/v2/status")
        assert response.status_code == 200
        data = response.json()
        assert "mission" in data
        assert "DO NOTHING is successful" in data["mission"]
        assert "bankroll" in data
        assert "kill_switch" in data
        assert "exposure" in data
        assert "calibration" in data

    def test_v2_venues_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        
        client = TestClient(app)
        response = client.get("/api/v2/venues")
        assert response.status_code == 200
        data = response.json()
        assert "venues" in data
        assert "eligibility" in data
        # Check Uganda not restricted
        if "polymarket" in data["eligibility"]:
            assert data["eligibility"]["polymarket"] != "restricted"

    def test_v2_risk_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        
        client = TestClient(app)
        response = client.get("/api/v2/risk")
        assert response.status_code == 200
        data = response.json()
        assert "kill_switch" in data
        assert "exposure" in data
        assert "limits" in data
        assert data["kill_switch"]["current_level"] == 0
        assert data["kill_switch"]["can_trade"] is True

    def test_v2_opportunities_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        
        client = TestClient(app)
        response = client.get("/api/v2/opportunities")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "after_cheap" in data
        assert "DO NOTHING is successful" in data["message"] or "DO NOTHING" in data["message"] or "pipeline" in data["message"].lower()

    def test_v2_calibration_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        
        client = TestClient(app)
        response = client.get("/api/v2/calibration")
        assert response.status_code == 200
        data = response.json()
        assert "brier_score" in data or "total_forecasts" in data
