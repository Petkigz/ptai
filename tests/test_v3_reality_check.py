"""
Tests for $50 challenge reality check: fees, gas, sustainability, circuit breaker, validation, ingestion
"""
import pytest
from src.ptai.markets.fees import FeeEngine, PolymarketFeeModel, KalshiFeeModel, CryptoFeeModel, StockFeeModel
from src.ptai.execution.gas import GasModel
from src.ptai.risk.sustainability import SustainabilityCalculator
from src.ptai.risk.circuit_breaker import CircuitBreaker, Position
from src.ptai.strategy.validation import FairValueValidator
from src.ptai.strategy.edge import EdgeCalculator
from src.ptai.markets.base import Market, MarketSource
from src.ptai.data_ingestion.polymarket_ingestion import PolymarketIngestion
from src.ptai.data_ingestion.news_ingestion import NewsIngestion
from src.ptai.data_ingestion.x_ingestion import XIngestion
from src.ptai.data_ingestion.orchestrator import DataIngestionOrchestrator
from datetime import datetime, timezone


class TestFeeModels:
    def test_polymarket_fee_formula_at_0_50(self):
        model = PolymarketFeeModel()
        result = model.calculate_fee(amount_usd=3.0, price=0.50)
        # Fee = 0.06 * C * p * (1-p), C=6, p=0.5 => 0.06*6*0.25=0.09
        assert abs(result.fee_usd - 0.09) < 0.001
        assert abs(result.fee_pct - 0.03) < 0.001  # 3%
        assert "0.06" in result.fee_formula

    def test_polymarket_fee_at_0_61(self):
        model = PolymarketFeeModel()
        result = model.calculate_fee(amount_usd=3.0, price=0.61)
        # fee = 0.06 * 3 * (1-0.61) = 0.06*3*0.39=0.0702
        assert abs(result.fee_usd - 0.0702) < 0.001
        assert abs(result.fee_pct - 0.0234) < 0.001

    def test_polymarket_conservative_tier(self):
        model = PolymarketFeeModel()
        result = model.calculate_fee_conservative(amount_usd=3.0, price=0.61, category="politics")
        # max of formula 2.34% vs tier 2% => 2.34%
        assert result.fee_pct >= 0.02

    def test_kalshi_fee_huge_for_small_bankroll(self):
        model = KalshiFeeModel()
        result = model.calculate_fee(amount_usd=3.0, price=0.50)
        # $0.07 per contract, 6 contracts = $0.42 but capped 7%
        assert result.fee_pct <= 0.07
        assert result.fee_usd <= 3.0 * 0.07

    def test_crypto_fee_lowest(self):
        model = CryptoFeeModel()
        result = model.calculate_fee(amount_usd=3.0, price=0.5)
        assert result.fee_pct == 0.001  # 0.1%
        assert result.fee_usd == 0.003

    def test_fee_engine_unified(self):
        engine = FeeEngine()
        poly = engine.calculate(venue_id="polymarket", amount_usd=3.0, price=0.5, category="politics")
        assert poly.fee_pct > 0
        kalshi = engine.calculate(venue_id="kalshi", amount_usd=3.0, price=0.5)
        assert kalshi.fee_pct > 0
        crypto = engine.calculate(venue_id="crypto_binance", amount_usd=3.0, price=0.5)
        assert crypto.fee_pct == 0.001

    def test_sustainability_report(self):
        engine = FeeEngine()
        report = engine.get_sustainability_report(bankroll=50.0)
        assert report["bankroll"] == 50.0
        assert report["max_position_6pct"] == 3.0
        assert "fees" in report
        assert "sustainability" in report
        # 30% monthly required
        assert "30%" in str(report["sustainability"]) or "30" in str(report["sustainability"]["required_monthly_return"])


class TestGasModel:
    def test_gas_polymarket(self):
        model = GasModel()
        result = model.calculate_gas(operation="place_order", amount_usd=3.0, venue_id="polymarket")
        assert result.gas_usd >= 0.05  # conservative min
        assert result.chain == "polygon"
        assert result.gas_pct_of_position > 0

    def test_gas_non_polygon_zero(self):
        model = GasModel()
        result = model.calculate_gas(operation="place_order", amount_usd=3.0, venue_id="kalshi")
        assert result.gas_usd == 0.0
        assert "off-chain" in result.chain or "centralized" in result.chain

    def test_gas_report(self):
        model = GasModel()
        report = model.get_gas_report(bankroll=50.0)
        assert report["bankroll"] == 50.0
        assert report["position_6pct"] == 3.0
        assert "operations" in report
        assert "place_order" in report["operations"]
        assert "reality_check" in report
        # gas 1.6-1.7% depending on rounding
        assert "1." in report["reality_check"] and "%" in report["reality_check"]

    def test_gas_significant_pct(self):
        model = GasModel()
        result = model.calculate_gas(operation="place_order", amount_usd=3.0, venue_id="polymarket")
        # $0.05 gas on $3 position = 1.6%
        assert result.gas_pct_of_position >= 0.01  # at least 1%


class TestSustainabilityCalculator:
    def test_sustainability_math(self):
        calc = SustainabilityCalculator(monthly_cost=15.0, fee_pct=0.02, gas_usd=0.05)
        report = calc.get_detailed_report(bankroll=50.0)
        assert report["bankroll"] == 50.0
        assert "cases" in report
        assert "roadmap" in report
        assert "red_flags" in report
        assert "critical_rules" in report
        assert "math" in report
        assert "20-40%" in report["math"]["required_return"] or "30%" in str(report)

    def test_scenarios(self):
        calc = SustainabilityCalculator()
        report = calc.get_detailed_report(bankroll=50.0)
        assert "scenarios" in report
        assert len(report["scenarios"]) > 0
        for s in report["scenarios"]:
            assert s["edge_after_costs"] >= 0
            assert s["win_rate"] >= 0.5

    def test_trades_needed(self):
        calc = SustainabilityCalculator()
        result = calc.calculate(bankroll=50.0, avg_edge_after_costs=0.10, win_rate=0.55)
        # At 10% edge, $0.30 profit per trade, need $0.50 daily = 1.6 trades
        assert abs(result.trades_needed_daily - 1.666) < 0.5

    def test_roadmap_phases(self):
        calc = SustainabilityCalculator()
        report = calc.get_detailed_report(bankroll=50.0)
        # roadmap is in cases or as string
        assert "roadmap" in report
        # Check phases mentioned in roadmap string or cases
        roadmap_str = str(report["roadmap"]) + str(report["cases"])
        assert "Read-Only" in roadmap_str or "read_only" in str(report).lower() or "Phase 1" in roadmap_str


class TestCircuitBreaker:
    def test_daily_loss_limit(self):
        cb = CircuitBreaker(daily_loss_limit=-5.0, max_open_positions=3, data_dir="/tmp/test_cb1")
        cb.state.daily_pnl = 0
        cb.state.is_halted = False
        cb.state.daily_trades = 0
        cb.state.consecutive_losses = 0
        cb.record_trade(pnl=-2, position_id="p1")
        assert cb.state.daily_pnl == -2
        assert cb.can_trade()
        cb.record_trade(pnl=-2, position_id="p2")
        assert cb.can_trade()
        cb.record_trade(pnl=-2, position_id="p3")
        assert not cb.can_trade()
        assert cb.state.is_halted

    def test_max_positions(self):
        cb = CircuitBreaker(daily_loss_limit=-5.0, max_open_positions=3, data_dir="/tmp/test_cb2")
        cb.state.open_positions = []
        cb.state.is_halted = False
        cb.state.daily_pnl = 0
        # Add 3 positions
        for i in range(3):
            pos = Position(market_id=f"M{i}", entry_price=0.5, current_price=0.5, amount_usd=3.0, side="YES", entry_time=datetime.now(timezone.utc))
            cb.add_position(pos)
        ok, reason = cb.check_max_positions()
        assert not ok
        assert "18%" in reason or "3" in reason

    def test_thin_market(self):
        cb = CircuitBreaker(data_dir="/tmp/test_cb3")
        ok, reason = cb.check_thin_market(volume_24h=5000, liquidity=500)
        assert not ok
        assert "Thin market" in reason
        ok, reason = cb.check_thin_market(volume_24h=15000, liquidity=5000)
        assert ok

    def test_cut_losses(self):
        cb = CircuitBreaker(stop_loss_pct=0.35, take_profit_pct=0.50, data_dir="/tmp/test_cb4")
        pos = Position(market_id="M1", entry_price=0.6, current_price=0.3, amount_usd=3.0, side="YES", entry_time=datetime.now(timezone.utc))
        eval_result = cb.evaluate_position(pos)
        assert eval_result["action"] == "CUT_LOSS"

    def test_take_profit(self):
        cb = CircuitBreaker(stop_loss_pct=0.35, take_profit_pct=0.50, data_dir="/tmp/test_cb5")
        pos = Position(market_id="M1", entry_price=0.4, current_price=0.7, amount_usd=3.0, side="YES", entry_time=datetime.now(timezone.utc))
        eval_result = cb.evaluate_position(pos)
        assert eval_result["action"] == "TAKE_PROFIT_HALF"

    def test_consecutive_losses_halt(self):
        cb = CircuitBreaker(daily_loss_limit=-100.0, max_open_positions=10, data_dir="/tmp/test_cb6")
        cb.state.daily_pnl = 0
        cb.state.is_halted = False
        cb.state.consecutive_losses = 0
        for i in range(5):
            cb.record_trade(pnl=-1, position_id=f"p{i}")
        assert cb.state.is_halted
        assert "Consecutive" in cb.state.halt_reason


class TestValidation:
    def _make_market(self, price=0.6):
        m = Market(id="M1", source=MarketSource.POLYMARKET, question="Will Trump win?", volume=10000, liquidity=5000, raw={"change_pct": 0})
        m.outcome_prices = [price, 1-price]
        return m

    def test_heuristic_mean_reversion(self):
        validator = FairValueValidator()
        # Extreme high price should mean revert down
        m_high = self._make_market(price=0.90)
        heur = validator.heuristic_estimate(market=m_high)
        assert heur < 0.90  # should revert down
        # Extreme low price should revert up
        m_low = self._make_market(price=0.10)
        heur2 = validator.heuristic_estimate(market=m_low)
        assert heur2 > 0.10

    def test_validation_hallucination_detection(self):
        validator = FairValueValidator(max_disagreement=0.25)
        m = self._make_market(price=0.85)
        result = validator.validate(market=m, llm_prob=0.90)
        assert result.disagreement >= 0
        assert result.heuristic_prob >= 0
        # If disagreement large, hallucination suspected
        if result.disagreement > 0.25:
            assert result.is_hallucination

    def test_validation_confidence_adjustment(self):
        validator = FairValueValidator()
        m = self._make_market(price=0.60)
        result_small = validator.validate(market=m, llm_prob=0.65)
        # Small disagreement => confidence boost +0.05
        assert result_small.confidence_adjustment == 0.05

        m2 = self._make_market(price=0.60)
        result_large = validator.validate(market=m2, llm_prob=0.95)
        # Large disagreement => confidence penalty -0.30
        if result_large.disagreement > 0.30:
            assert result_large.confidence_adjustment == -0.30

    def test_ensemble_weighted(self):
        validator = FairValueValidator()
        m = self._make_market(price=0.60)
        result = validator.validate(market=m, llm_prob=0.70)
        assert result.ensemble_prob >= 0
        # Check reasoning contains LLM weight
        assert "LLM weight" in result.reasoning or "weight" in result.reasoning.lower()

    def test_prompt_template(self):
        validator = FairValueValidator()
        template = validator.get_prompt_template()
        assert "Market Question" in template or "question" in template.lower()
        assert "price" in template.lower()
        assert "fair_probability" in template.lower() or "fair" in template.lower()
        assert "JSON" in template


class TestEdgeWithFeesGas:
    def test_edge_includes_fees_and_gas(self):
        calc = EdgeCalculator()
        market = Market(
            id="M1", source=MarketSource.POLYMARKET, question="Will Trump win election?",
            volume=50000, liquidity=10000, raw={}
        )
        market.outcome_prices = [0.60, 0.40]
        opp = calc.calculate(market=market, fair_prob=0.70, uncertainty=0.1, amount_usd=3.0)
        # Should include fee and gas deductions
        assert opp.fees > 0
        assert "fee" in opp.reasoning.lower()
        assert "gas" in opp.reasoning.lower()
        assert "$50 math" in opp.reasoning

    def test_edge_10pct_raw_not_enough(self):
        calc = EdgeCalculator()
        market = Market(
            id="M1", source=MarketSource.POLYMARKET, question="Will Trump win election?",
            volume=50000, liquidity=10000, raw={}
        )
        market.outcome_prices = [0.60, 0.40]
        # 10% raw edge 0.60->0.70, but fees 2.4%+gas 1.7%+spread 2%+uncertainty 5% = 11.1% total, effective -1.1%
        opp = calc.calculate(market=market, fair_prob=0.70, uncertainty=0.1, amount_usd=3.0)
        assert abs(opp.raw_edge - 0.10) < 0.001
        assert opp.effective_edge < 0.08  # not enough to trade
        assert not opp.should_trade

    def test_edge_20pct_raw_enough(self):
        calc = EdgeCalculator()
        market = Market(
            id="M1", source=MarketSource.POLYMARKET, question="Will Trump win election?",
            volume=50000, liquidity=10000, raw={}
        )
        market.outcome_prices = [0.60, 0.40]
        # 20% raw edge 0.60->0.80, fees 2.4%+gas 1.7%+spread 2%+unc 5% = 11.1% total, effective 8.9%
        opp = calc.calculate(market=market, fair_prob=0.80, uncertainty=0.1, amount_usd=3.0)
        assert abs(opp.raw_edge - 0.20) < 0.001
        assert opp.effective_edge >= 0.08
        assert opp.should_trade


class TestDataIngestion:
    def test_polymarket_ingestion_api_first(self):
        ing = PolymarketIngestion()
        report = ing.get_ingestion_report()
        assert "API-First" in report["method"]
        assert "official" in report["sdk"].lower() or "SDK" in report["sdk"]
        assert report["reliability"] == "High - API first, not scraping web UI"

    def test_news_ingestion(self):
        ing = NewsIngestion()
        articles = ing.fetch_news_rss(max_articles=5)
        assert len(articles) > 0
        assert "title" in articles[0]
        assert "credibility" in articles[0]
        report = ing.get_report()
        assert "RSS" in report["method"]

    def test_x_ingestion_disabled_fastest(self):
        ing = XIngestion(enabled=False)
        tweets = ing.fetch_tweets(query="Trump", max_tweets=10)
        assert tweets == []
        report = ing.get_report()
        assert not report["enabled"]
        assert "40 sec not 21 min" in report["recommendation"] or "fastest" in report["recommendation"]

    def test_x_ingestion_circuit_breaker(self):
        ing = XIngestion(enabled=True)
        ing.circuit_breaker_fails = 3
        ing.is_disabled = True
        tweets = ing.fetch_tweets(query="test", max_tweets=10)
        assert tweets == []
        report = ing.get_report()
        assert "circuit_breaker" in report

    def test_orchestrator(self):
        orch = DataIngestionOrchestrator(use_x=False)
        report = orch.get_full_report()
        assert "orchestrator" in report
        assert "polymarket" in report
        assert "news" in report
        assert "x" in report
        assert "api_first" in report
        # Check reliability message
        assert "reliability" in str(report).lower() or "SDK" in str(report)


class TestV3NewEndpoints:
    def test_v3_fees_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/fees?bankroll=50")
        assert res.status_code == 200
        data = res.json()
        assert "bankroll" in data
        assert data["bankroll"] == 50.0
        assert "fees" in data
        assert "sustainability" in data
        assert "fee_comparison" in data

    def test_v3_gas_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/gas?bankroll=50")
        assert res.status_code == 200
        data = res.json()
        assert "bankroll" in data
        assert "operations" in data
        assert "reality_check" in data
        assert "total_cost_per_trade" in data

    def test_v3_sustainability_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/sustainability?bankroll=50")
        assert res.status_code == 200
        data = res.json()
        assert "bankroll" in data
        assert "cases" in data
        assert "red_flags" in data
        assert "critical_rules" in data

    def test_v3_circuit_breaker_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/circuit-breaker")
        assert res.status_code == 200
        data = res.json()
        assert "daily_pnl" in data
        assert "can_trade" in data
        assert "critical_rules" in data

    def test_v3_validation_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/validation")
        assert res.status_code == 200
        data = res.json()
        assert "prompt_template" in data
        assert "method" in data
        assert "max_disagreement" in data

    def test_v3_ingestion_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/ingestion")
        assert res.status_code == 200
        data = res.json()
        assert "polymarket" in data
        assert "news" in data
        assert "x" in data

    def test_v3_roadmap_endpoint(self):
        from fastapi.testclient import TestClient
        from src.ptai.dashboard import app
        client = TestClient(app)
        res = client.get("/api/v3/roadmap")
        assert res.status_code == 200
        data = res.json()
        assert "phase_1_read_only" in data
        assert "phase_2_paper_trading" in data
        assert "phase_3_live_tiny" in data
        assert "phase_4_scale" in data
        assert "reality_check" in data
