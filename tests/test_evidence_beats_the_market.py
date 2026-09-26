"""
Evidence has to beat the market, and evidence expires.

Borrowed from the sibling project (`avt-bot`), whose deployment protocol is the
discipline this file exists to hold PTAI to:

  * a model is only deployed if it beats the best simple NULL, out of sample,
    with a bootstrap interval that clears zero;
  * the entries it would have taken must show a hit rate the odds cannot explain
    (a binomial tail against the prices paid); and
  * a judgment made on one record is not trusted on a record that has since
    grown past it.

PTAI's gate had none of the three. `forecast_skill` was `max(0, 1 - brier*2)` -
a rescaled Brier score with no benchmark, which in a market priced at 0.50
rewards a constant forecast of 0.50 with 0.5 "skill"; and `max_brier <= 0.25` is
passed by that same constant. The market is the opponent, so the market price is
the null, and until the forecast beats it there is no edge to deploy.

These tests use the real storage, the real outcome log and the real gate. The
only thing faked is the market's outcomes, which is what a test is for.
"""

import sqlite3

import pytest

from src.ptai.learning.evidence import (
    FORECAST_BEHIND_PRICE,
    FORECAST_BEATS_PRICE,
    INSUFFICIENT,
    NO_EVIDENCE,
    UNMEASURED,
    bootstrap_ci,
    drift,
    market_skill,
    poisson_binomial_tail,
)
from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker,
    qualification_stats_from_outcomes,
)
from src.ptai.storage.db import Storage
from src.ptai.venues.qualification import VenueQualificationEngine

VENUE = "polymarket"


def _rows(n, *, edge, seed=11, price=None, forecast=None):
    """
    `n` resolved trades: a market price, a forecast that may be better than it,
    and an outcome drawn from the FORECAST (so the record says what it says).

    Outcomes are drawn from the forecast on purpose: a test that drew them from
    the market price and then asserted the forecast beat it would be asserting
    its own arithmetic. Here the claim under test is genuinely true of the data,
    and the question is whether the gate can tell.
    """
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p_market = price if price is not None else round(rng.uniform(0.2, 0.8), 4)
        p_model = forecast if forecast is not None else min(0.99, p_market + edge)
        won = 1.0 if rng.random() < p_model else 0.0
        out.append({"forecast_prob": p_model, "yes_price": p_market,
                    "actual_outcome": won, "side": "YES"})
    return out


def _dead_rows(n, *, forecast=0.85, price=0.5, wins_every=3):
    """
    `n` deterministic trades the agent was WRONG about: it claimed 0.95 on a
    market priced 0.50 and won one time in three.

    Deterministic on purpose. A drifting record built from random draws can flip
    its verdict with the seed, and a drift test that fails on one seed and passes
    on another trains you to re-run until it passes - which is exactly how a
    safety check becomes decoration.
    """
    return [{"forecast_prob": forecast, "yes_price": price,
             "actual_outcome": (1.0 if (i + 1) % wins_every == 0 else 0.0),
             "side": "YES"}
            for i in range(n)]


# ---------------------------------------------------------------- the statistics

class TestTheForecastMeasuredAgainstThePrice:
    def test_a_forecast_that_beats_the_price_is_reported_as_such(self):
        evidence = market_skill(_rows(300, edge=0.15))
        assert evidence.verdict == FORECAST_BEATS_PRICE
        assert evidence.forecast_beats_price
        assert evidence.beats_market, evidence.reason
        assert evidence.ci_low > 0, evidence.reason
        assert evidence.model_brier < evidence.market_brier

    def test_a_forecast_as_good_as_the_price_proves_nothing(self):
        """
        The whole point: equal to the price is ZERO evidence, not "no worse".

        This is the record a gate that only reads Brier would wave through - the
        forecast is perfectly calibrated, its Brier is decent, and there is not
        one cent of edge in it.
        """
        rows = []
        import random
        rng = random.Random(5)
        for _ in range(300):
            price = round(rng.uniform(0.2, 0.8), 4)
            rows.append({"forecast_prob": price, "yes_price": price,
                         "actual_outcome": 1.0 if rng.random() < price else 0.0,
                         "side": "YES"})
        evidence = market_skill(rows)
        assert evidence.verdict == NO_EVIDENCE, evidence.reason
        assert not evidence.beats_market
        assert evidence.improvement == pytest.approx(0.0, abs=1e-12)

    def test_a_forecast_worse_than_the_price_is_caught_as_drift(self):
        rows = _rows(200, edge=0.0, seed=3)
        # Replace the forecasts with a deliberately worse view than the price.
        broken = [{"forecast_prob": 0.5, "yes_price": r["yes_price"],
                   "actual_outcome": r["actual_outcome"], "side": "YES"}
                  for r in rows]
        evidence = market_skill(broken)
        assert evidence.verdict == FORECAST_BEHIND_PRICE
        assert evidence.ci_high < 0, evidence.reason
        assert evidence.behind_market

    def test_too_little_evidence_is_said_so(self):
        evidence = market_skill(_rows(12, edge=0.3))
        assert evidence.verdict == INSUFFICIENT
        assert not evidence.beats_market

    def test_no_evidence_at_all_is_not_a_pass(self):
        assert market_skill([]).verdict == UNMEASURED
        # ...and rows with no price are not evidence about a price.
        assert market_skill([{"forecast_prob": 0.7, "actual_outcome": 1.0}]) \
            .verdict == UNMEASURED

    def test_better_probabilities_are_not_enough_on_their_own(self):
        """
        The trap the two-part claim exists for.

        The agent's probabilities here are better than the prices - it is right
        about the world - but it keeps buying sides priced at 0.90 that pay off
        0.75 of the time. Forecast better, entries losing. A gate that granted
        live capital on the Brier comparison alone would be buying that.
        """
        rows = [{"forecast_prob": 0.75, "yes_price": 0.90,
                 "actual_outcome": (1.0 if i % 4 else 0.0), "side": "YES"}
                for i in range(120)]
        evidence = market_skill(rows)
        assert evidence.forecast_beats_price, evidence.reason
        assert not evidence.entries_clear_the_odds, evidence.reason
        assert not evidence.beats_market, (
            "better probabilities with losing entries must not count as an edge")

    def test_the_entry_test_reads_the_side_that_was_actually_bought(self):
        """
        A NO trade wins when YES does not, and it paid `1 - yes_price`.

        Counting its win against the YES price would score a 0.90 NO buy as a
        0.10 coin, and the agent would be credited with an edge it was never
        offered. Here every trade buys NO at 0.90 (the YES price is 0.10, so the
        NO token costs 0.90) and NO wins 0.75 of the time - losing money.
        """
        rows = [{"forecast_prob": 0.10, "yes_price": 0.10,
                 "actual_outcome": (1.0 if i % 4 == 0 else 0.0), "side": "NO"}
                for i in range(120)]
        evidence = market_skill(rows)
        assert evidence.expected_hits == pytest.approx(120 * 0.9, rel=0.01), (
            "the entry test priced the NO side off the YES price")

    def test_the_bootstrap_is_deterministic(self):
        """Same trades, same interval: a verdict that flips is a bug."""
        values = [0.01, -0.02, 0.03, 0.0, 0.05, -0.01, 0.02, 0.04]
        first = bootstrap_ci(values)
        second = bootstrap_ci(values)
        assert first == second
        assert first[0] <= first[1]

    def test_the_entry_tail_is_exact_not_a_coin_flip_approximation(self):
        # Each trade has its own break-even price, so the null is Poisson-binomial.
        assert poisson_binomial_tail(0, [0.5] * 10) == pytest.approx(1.0)
        assert poisson_binomial_tail(10, [0.5] * 10) == pytest.approx(0.5 ** 10)
        # 3 coins at p=1.0 always win: 3 or more is certain.
        assert poisson_binomial_tail(3, [1.0, 1.0, 1.0]) == pytest.approx(1.0)
        # ...and 4 or more is impossible, not a crash.
        assert poisson_binomial_tail(4, [1.0, 1.0, 1.0]) == 0.0


# ---------------------------------------------------------------- through the log

def _storage(tmp_path):
    return Storage(db_path=str(tmp_path / "evidence.db"))


def _record(storage, tracker, rows):
    """Write the rows through the real outcome log, the way the loop does."""
    for i, row in enumerate(rows):
        tid = storage.log_trade({
            "market_id": f"{VENUE}-{i}", "venue_id": VENUE, "side": row["side"],
            "position_size_usd": 3.0, "market_price": row["yes_price"],
            "fair_price": row["forecast_prob"], "edge": 0.1, "confidence": 0.7,
            "strategy": "value", "execution_mode": "paper", "status": "paper",
        })
        won = row["actual_outcome"] > 0.5
        pnl = 1.5 if won else -3.0
        storage.resolve_trade(tid, outcome=row["actual_outcome"], pnl=pnl)
        tracker.record_trade(
            trade_id=str(tid), market_id=f"{VENUE}-{i}", venue_id=VENUE,
            strategy="value", category="politics",
            forecast_prob=row["forecast_prob"], market_price=row["yes_price"],
            yes_price=row["yes_price"], edge=0.1, side=row["side"],
            amount_usd=3.0, fees_usd=0.06, slippage_bps=10.0,
            execution_quality=0.98, data_mode="live", execution_mode="paper",
            expected_net_ev=0.2, expected_net_ev_pct=0.07,
            book_source="ladder", executable_net_ev=pnl,
            executable_net_ev_pct=pnl / 3.0, fill_price_vs_modelled=0.0)
        tracker.record_resolution(str(tid), actual_outcome=row["actual_outcome"],
                                  pnl=pnl)


class TestTheGateReadsTheComparison:
    def test_the_outcome_log_carries_the_null(self, tmp_path):
        storage = _storage(tmp_path)
        tracker = TradeOutcomeTracker(storage=storage)
        _record(storage, tracker, _rows(300, edge=0.15))
        stats = qualification_stats_from_outcomes(storage, VENUE)
        assert stats["market_skill_samples"] == 300
        assert stats["market_skill_verdict"] == FORECAST_BEATS_PRICE
        assert stats["market_skill_beats_price"] is True
        assert stats["market_skill_coverage"] == pytest.approx(1.0)
        storage.close()

    def test_a_venue_whose_forecast_never_beat_the_price_is_refused(self, tmp_path):
        """
        The record that used to walk straight through the gate: 150 trades, 60%
        wins, its forecast exactly the price. Every absolute bar can be met, and
        there is no edge in any of it.
        """
        storage = _storage(tmp_path)
        tracker = TradeOutcomeTracker(storage=storage)
        import random
        rng = random.Random(9)
        rows = []
        for _ in range(150):
            price = round(rng.uniform(0.4, 0.6), 4)
            rows.append({"forecast_prob": price, "yes_price": price,
                         "actual_outcome": 1.0 if rng.random() < price else 0.0,
                         "side": "YES"})
        _record(storage, tracker, rows)
        stats = qualification_stats_from_outcomes(storage, VENUE)
        result = VenueQualificationEngine(
            data_dir=str(tmp_path)).evaluate_qualification(VENUE, stats)
        assert result.checks["beats_the_price"] is False
        assert result.is_qualified is False
        assert "beat the price" in result.reasoning or \
               "no measurable edge" in result.reasoning
        # The reason names the comparison, so the operator is told what failed
        # rather than which of twenty thresholds.
        assert "AGAINST THE PRICE" in result.reasoning
        storage.close()

    def test_a_venue_with_no_prices_recorded_fails_closed(self, tmp_path):
        """Legacy rows carry no price, so there is nothing to compare."""
        storage = _storage(tmp_path)
        with sqlite3.connect(storage.db_path) as conn:
            pass
        stats = qualification_stats_from_outcomes(storage, VENUE)
        result = VenueQualificationEngine(
            data_dir=str(tmp_path)).evaluate_qualification(VENUE, stats)
        assert result.checks["beats_the_price"] is False
        assert result.market_skill_verdict == UNMEASURED
        assert result.market_skill_samples == 0
        storage.close()

    def test_a_venue_that_beats_the_price_is_on_the_drift_check_only(self, tmp_path):
        storage = _storage(tmp_path)
        tracker = TradeOutcomeTracker(storage=storage)
        _record(storage, tracker, _rows(150, edge=0.2))
        stats = qualification_stats_from_outcomes(storage, VENUE)
        result = VenueQualificationEngine(
            data_dir=str(tmp_path)).evaluate_qualification(VENUE, stats)
        assert result.checks["beats_the_price"] is True, result.reasoning[-400:]
        assert result.checks["not_drifting"] is True, result.reasoning[-400:]
        storage.close()


# ---------------------------------------------------------------- expiry

class TestEvidenceExpires:
    def test_drift_over_the_recent_window_is_visible(self):
        """
        An edge that dies must not hide inside the average it used to earn.

        Nine good trades for every bad one, then the last sixty all behind the
        price: the all-time comparison is still strongly positive, and the
        recent window is not.
        """
        rows = _rows(400, edge=0.30, seed=21) + _dead_rows(60)
        assert market_skill(rows).verdict == FORECAST_BEATS_PRICE
        recent = drift(rows, window=60)
        assert recent.verdict == FORECAST_BEHIND_PRICE, recent.reason
        assert recent.ci_high < 0

    def test_a_drifting_venue_loses_its_qualification(self, tmp_path):
        storage = _storage(tmp_path)
        tracker = TradeOutcomeTracker(storage=storage)
        _record(storage, tracker, _rows(400, edge=0.30, seed=21) + _dead_rows(60))
        stats = qualification_stats_from_outcomes(storage, VENUE)
        assert stats["market_skill_verdict"] == FORECAST_BEATS_PRICE
        assert stats["recent_market_skill_verdict"] == FORECAST_BEHIND_PRICE
        result = VenueQualificationEngine(
            data_dir=str(tmp_path)).evaluate_qualification(VENUE, stats)
        assert result.checks["not_drifting"] is False
        assert result.is_qualified is False, (
            "a venue drifting behind the price kept its live qualification")
        storage.close()

    def test_a_judgment_does_not_cover_a_record_that_has_outgrown_it(self, tmp_path):
        """
        Staleness is measured from the moment the venue PASSED.

        Re-running the same bars over the same record is not new evidence, so a
        verdict earned at 100 trades and asked about at 400 has been judged on a
        quarter of what the venue has done - and the honest reading is "judge it
        again", not "it passed once".
        """
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        stats = qualification_stats_from_outcomes(None, VENUE)
        engine.evaluate_qualification(VENUE, stats)
        result = engine.qualifications[VENUE]
        assert result.is_qualified is False

        # A verdict that DID pass, stamped at 100 trades.
        result.is_qualified = True
        result.qualified_on_trades = 100
        engine.qualifications[VENUE] = result

        fresh = engine.staleness(VENUE, 150)
        assert fresh["stale"] is False
        assert fresh["new_trades"] == 50
        assert engine.is_qualified(VENUE, settled_rows=150) is True

        old = engine.staleness(VENUE, 400)
        assert old["stale"] is True
        assert old["judged_on"] == 100
        assert "400" in old["reason"] or "300" in old["reason"]
        assert engine.is_qualified(VENUE, settled_rows=400) is False, (
            "a verdict from 300 trades ago authorised live capital")
        # ...and without a count the stored verdict is returned as stored.
        assert engine.is_qualified(VENUE) is True

    def test_the_stamp_survives_a_restart(self, tmp_path):
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        stats = qualification_stats_from_outcomes(None, VENUE)
        result = engine.evaluate_qualification(VENUE, stats)
        result.is_qualified = True
        result.qualified_on_trades = 120
        result.market_skill = 0.03
        result.market_skill_verdict = FORECAST_BEATS_PRICE
        result.market_skill_reason = "forecast beats the price"
        engine._save()

        reloaded = VenueQualificationEngine(data_dir=str(tmp_path))
        stored = reloaded.qualifications[VENUE]
        assert stored.qualified_on_trades == 120, (
            "the file lost the evidence the staleness check reads")
        assert stored.market_skill_verdict == FORECAST_BEATS_PRICE
        assert reloaded.staleness(VENUE, 400)["stale"] is True
