"""Tests for backtest engine"""
from src.ptai.backtest.engine import BacktestEngine

class TestBacktestEngine:
    def test_init(self):
        engine = BacktestEngine()
        assert engine is not None
    
    def test_run_backtest(self):
        engine = BacktestEngine()
        config = {
            "name": "Test Strategy",
            "bankroll": 50.0,
            "min_edge": 0.08,
            "max_pos_pct": 0.06,
            "kelly_fraction": 0.5
        }
        result = engine.run(strategy_config=config, days=7)
        assert result.initial_bankroll == 50.0
        assert result.final_bankroll > 0
        assert result.total_trades >= 0
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'equity_curve')
    
    def test_backtest_with_different_edge(self):
        engine = BacktestEngine()
        config_low = {"name": "Low Edge", "bankroll": 50.0, "min_edge": 0.05, "max_pos_pct": 0.06, "kelly_fraction": 0.5}
        config_high = {"name": "High Edge", "bankroll": 50.0, "min_edge": 0.15, "max_pos_pct": 0.06, "kelly_fraction": 0.5}
        
        result_low = engine.run(strategy_config=config_low, days=7)
        result_high = engine.run(strategy_config=config_high, days=7)
        
        # Low edge should have more trades than high edge
        assert result_low.total_trades >= result_high.total_trades
