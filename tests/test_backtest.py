"""
Backtest tests.

The V9 gate refuses synthetic markets, which is correct - but it meant these
tests could not run at all, because nothing supplied real data. They now build
a dataset through HistoricalDataProvider with an injected fetch, so the real
parse-and-validate path executes and the engine is fed genuinely resolved
markets rather than random ones.
"""
import pytest
from src.ptai.backtest import BacktestEngine, HistoricalDataProvider

# Shaped like the real Gamma payload: outcomes/outcomePrices are JSON-encoded
# strings, closed markets settle at 1/0, and lastTradePrice is the entry price.
def _resolved(markets):
    rows = []
    for i, (question, entry, outcome, vol) in enumerate(markets):
        rows.append({
            "id": str(i + 1), "question": question, "closed": True,
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1","0"]' if outcome == 1 else '["0","1"]',
            "lastTradePrice": str(entry),
            "volume": str(vol), "liquidity": str(vol // 2),
            "endDate": f"2026-0{(i % 9) + 1}-01T00:00:00Z",
        })
    return rows


def _provider(markets):
    return HistoricalDataProvider(fetch=lambda url, params: _resolved(markets),
                                  cache_dir="/tmp/ptai_backtest_cache")


# A deterministic book of resolved markets. Entry prices vary so that a
# strategy with a real signal finds trades, and outcomes are mixed so the
# result is not degenerate.
BOOK = [
    ("Market A resolved YES", 0.62, 1, 500_000),
    ("Market B resolved NO", 0.41, 0, 250_000),
    ("Market C resolved YES", 0.55, 1, 180_000),
    ("Market D resolved NO", 0.48, 0, 320_000),
    ("Market E resolved YES", 0.70, 1, 410_000),
    ("Market F resolved NO", 0.33, 0, 275_000),
]
# Fair values recorded "while the markets were live" - the only signal mode
# that produces a production-grade backtest. Deliberately spread across the
# edge thresholds so a higher min_edge takes strictly fewer trades.
FAIR = {"1": 0.78, "2": 0.47, "3": 0.72, "4": 0.52, "5": 0.86, "6": 0.41}


def _dataset():
    return _provider(BOOK).build_dataset(markets=None, recorded_fair_values=FAIR,
                                         limit=6, days_back=90)


def _dataset_from_book():
    provider = _provider(BOOK)
    markets = provider.fetch_resolved_markets(use_cache=False)
    return provider.build_dataset(markets=markets, recorded_fair_values=FAIR)


"""Tests for backtest engine"""

class TestBacktestEngine:
    def test_init(self):
        engine = BacktestEngine()
        assert engine is not None
    
    def test_run_backtest(self):
        """
        Runs on real resolved markets.

        Before HistoricalDataProvider existed this raised ValueError on every
        call: the V9 gate correctly refused synthetic data and nothing supplied
        real data, so the command was unreachable.
        """
        engine = BacktestEngine()
        config = {
            "name": "Test Strategy",
            "bankroll": 50.0,
            "min_edge": 0.08,
            "max_pos_pct": 0.06,
            "kelly_fraction": 0.5
        }
        result = engine.run(strategy_config=config, dataset=_dataset_from_book(), days=7)
        assert result.initial_bankroll == 50.0
        assert result.final_bankroll > 0
        assert result.total_trades >= 0
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'equity_curve')
        # Real data, not synthetic - that is the whole point of the gate.
        assert result.is_synthetic is False
        assert result.data_mode == "live"

    def test_gate_still_refuses_to_run_on_nothing(self):
        """The gate must not be weakened by adding a data source."""
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="real historical data"):
            engine.run(strategy_config={"bankroll": 50.0}, days=7)

    def test_same_data_and_seed_reproduce_exactly(self):
        """
        A backtest must be reproducible.

        The V10 cost model perturbs execution price with a random walk; using
        the module-level RNG made identical inputs give different PnL, so
        comparing two configs compared their noise.
        """
        engine = BacktestEngine()
        cfg = {"name": "Repro", "bankroll": 50.0, "min_edge": 0.05}
        a = engine.run(strategy_config=cfg, dataset=_dataset_from_book(), seed=7)
        b = engine.run(strategy_config=cfg, dataset=_dataset_from_book(), seed=7)
        assert a.total_trades == b.total_trades
        assert a.final_bankroll == pytest.approx(b.final_bankroll, abs=1e-9)

    def test_baseline_run_trades_nothing(self):
        """
        With no signal, fair value equals the market price, so edge is zero.

        This is the honest baseline. A dataset that produced profitable trades
        from an invented edge would be worse than no backtest at all.
        """
        provider = _provider(BOOK)
        dataset = provider.build_dataset(markets=provider.fetch_resolved_markets(use_cache=False))
        assert dataset.is_baseline is True
        assert all(r["edge"] == 0.0 for r in dataset.rows)

        engine = BacktestEngine()
        result = engine.run(strategy_config={"bankroll": 50.0, "min_edge": 0.08},
                            dataset=dataset, days=7)
        assert result.total_trades == 0
        assert result.is_production_grade is False
        assert any("BASELINE" in w for w in result.warnings)
    
    def test_backtest_with_different_edge(self):
        engine = BacktestEngine()
        config_low = {"name": "Low Edge", "bankroll": 50.0, "min_edge": 0.05, "max_pos_pct": 0.06, "kelly_fraction": 0.5}
        config_high = {"name": "High Edge", "bankroll": 50.0, "min_edge": 0.15, "max_pos_pct": 0.06, "kelly_fraction": 0.5}
        
        data = _dataset_from_book()
        result_low = engine.run(strategy_config=config_low, dataset=data, days=7)
        result_high = engine.run(strategy_config=config_high, dataset=data, days=7)

        # A stricter edge threshold must take strictly fewer trades. The book
        # has edges +0.16/+0.06/+0.17/+0.04/+0.16/+0.08, so 0.05 passes five
        # and 0.15 passes three.
        assert result_low.total_trades == 5
        assert result_high.total_trades == 3
        assert result_low.total_trades > result_high.total_trades
