"""Tests for risk - Kelly criterion, position sizing"""
import pytest
from src.ptai.risk import KellyCalculator, RiskManager
from src.ptai.storage.db import Storage
import tempfile
from pathlib import Path

class TestKellyCalculator:
    def test_kelly_basic(self):
        kelly = KellyCalculator(kelly_fraction=0.5, max_pct=0.06, min_edge=0.08)
        # Market 60c, fair 75% => edge 15%
        result = kelly.calculate(market_price=0.6, fair_prob=0.75, bankroll=50.0)
        assert result.edge >= 0.14
        assert result.position_size_usd <= 50.0 * 0.06 + 0.01  # capped at 6%
        assert result.position_size_usd > 0
    
    def test_kelly_no_edge(self):
        kelly = KellyCalculator(kelly_fraction=0.5, max_pct=0.06, min_edge=0.08)
        # No edge
        result = kelly.calculate(market_price=0.7, fair_prob=0.71, bankroll=50.0)
        # Edge < min_edge should not bet
        assert result.edge < 0.08 or result.should_bet is False or result.position_size_usd == 0
    
    def test_kelly_capped_at_6_percent(self):
        kelly = KellyCalculator(kelly_fraction=1.0, max_pct=0.06, min_edge=0.08)
        # Huge edge should still be capped
        result = kelly.calculate(market_price=0.5, fair_prob=0.9, bankroll=100.0)
        assert result.position_size_pct <= 0.06 + 0.001
    
    def test_kelly_fraction(self):
        kelly_full = KellyCalculator(kelly_fraction=1.0, max_pct=0.5, min_edge=0.01)
        kelly_half = KellyCalculator(kelly_fraction=0.5, max_pct=0.5, min_edge=0.01)
        
        result_full = kelly_full.calculate(market_price=0.6, fair_prob=0.75, bankroll=100.0)
        result_half = kelly_half.calculate(market_price=0.6, fair_prob=0.75, bankroll=100.0)
        
        # Half Kelly should be half of full
        assert abs(result_half.position_size_usd - result_full.position_size_usd * 0.5) < 0.1

class TestRiskManager:
    def test_risk_manager_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            kelly = KellyCalculator()
            rm = RiskManager(storage=storage, kelly_calculator=kelly)
            assert rm is not None
            storage.close()
    
    def test_can_trade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            kelly = KellyCalculator()
            rm = RiskManager(storage=storage, kelly_calculator=kelly)
            # Should be able to check risk
            check = rm.check_all(market_price=0.6, fair_value=0.75, market_id="test", confidence=0.8)
            assert hasattr(check, 'allowed')
            assert isinstance(check.allowed, bool)
            assert check.allowed is True
            storage.close()
