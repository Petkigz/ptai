"""
Is the live qualification gate fed by what actually happened?

It was not. The gate read two inputs, and neither was ever written in production:

  * `adapter.performance_stats` - a dict initialised to {"total_paper_trades": 0}
    that no code path in the repository ever increments.
  * `data/venue_qualification.json` - loaded at startup and written ONLY by
    evaluate_qualification, which the trading loop never called. The only callers
    were tests.

Two consequences, and they fail in opposite directions:

  * PERMANENTLY CLOSED: sample size is always zero, so no venue can ever satisfy
    the 100-trade requirement. Live capital is unreachable no matter how well the
    agent trades.
  * FAIL-OPEN: whatever a test, a demo or an older version last wrote to that
    file is read back as evidence. A leftover entry reading "120 trades, brier
    0.18, profit factor 1.5" passes three of the four statistical gates. The gate
    would then be deciding on numbers that were never measured from that venue.

These tests pin the fix: the numbers come from recorded outcomes, in the
database, and nowhere else.
"""

from __future__ import annotations

import json

import pytest

from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker,
    qualification_stats_from_outcomes,
)
from src.ptai.storage.db import Storage
from src.ptai.venues.qualification import VenueQualificationEngine

VENUE = "polymarket"


@pytest.fixture()
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "outcomes.db"))


def _record(storage, tracker, venue, n, *, win_prob, forecast, win_pnl, loss_pnl,
            edge=0.15, expected_ev_pct=0.14, ev_every=1):
    """
    n resolved trades with a deterministic win/loss pattern.

    Deterministic on purpose: a flaky gate test is worse than no gate test,
    because it trains you to re-run until it passes.
    """
    wins = loses = 0
    for i in range(n):
        # Spread the wins evenly rather than in a block, so consecutive losses
        # stay realistic and the drawdown number means something.
        won = (i % 10) < round(win_prob * 10)
        tid = storage.log_trade({
            "market_id": f"{venue}-{i}", "venue_id": venue, "side": "YES",
            "position_size_usd": 3.0, "market_price": 0.5,
            "fair_price": forecast, "edge": edge, "confidence": 0.7,
            "strategy": "value",
            # Paper execution against live market data - the realistic paper
            # run, and the evidence the qualification gate is supposed to weigh.
            "execution_mode": "paper", "status": "paper",
        })
        pnl = win_pnl if won else loss_pnl
        storage.resolve_trade(tid, outcome=1.0 if won else 0.0, pnl=pnl)
        # Costs and execution quality, passed the way the loop passes them from
        # a real fill. The gate fails closed on an unmeasured execution quality,
        # which is asserted separately below.
        tracker.record_trade(
            trade_id=str(tid), market_id=f"{venue}-{i}", venue_id=venue,
            strategy="value", category="politics", forecast_prob=forecast,
            market_price=0.5, edge=edge, side="YES", amount_usd=3.0,
            fees_usd=0.06, slippage_bps=12.0, execution_quality=0.976,
            # The expected net EV per trade, recorded BEFORE the trade was taken
            # - the prediction the gate is supposed to weigh, and the coverage
            # that says how much of the record it came from. The loop writes
            # these on every position it opens, so a fixture that omits them is
            # simulating a system that never ran.
            expected_net_ev=(expected_ev_pct * 3.0
                             if expected_ev_pct is not None and i % ev_every == 0
                             else None),
            expected_net_ev_pct=(expected_ev_pct
                                 if expected_ev_pct is not None and i % ev_every == 0
                                 else None),
            data_mode="live", execution_mode="paper",
            # The book the paper run walked. Passed the way the loop passes it
            # from the fill: without it the gate treats every one of these
            # trades as unpriced evidence, which is asserted separately below.
            book_source="ladder")
        tracker.record_resolution(str(tid), actual_outcome=1.0 if won else 0.0,
                                  pnl=pnl)
        if won:
            wins += 1
        else:
            loses += 1
    return wins, loses


# ----------------------------------------------------------------------
# the gate's inputs come from the outcome log
# ----------------------------------------------------------------------

def test_no_outcomes_means_no_evidence(storage):
    """An unmeasured venue fails closed with numbers that fail every check."""
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["total_paper_trades"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["brier_score"] == 1.0, "an absent forecast cannot look skilful"
    assert stats["profit_factor"] == 0.0
    assert stats["source"] == "no recorded outcomes"


def test_evidence_priced_against_an_assumed_book_does_not_qualify(storage):
    """
    The point of the whole coverage check.

    Identical trades, identical P&L, identical calibration - the only
    difference is that the fills were walked against an assumed book instead of
    a real one. A venue must not be able to qualify on the simulator's own
    arithmetic.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=2.0, loss_pnl=-1.0)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["real_evidence_coverage"] == 1.0, "the ladder walk is evidence"

    # Now the same venue, with every fill labelled as an assumed book.
    storage.conn.execute("UPDATE trade_outcomes SET fill_is_real = 0, "
                         "book_source = 'assumed_default'")
    storage.conn.commit()
    simulated = qualification_stats_from_outcomes(storage, VENUE)
    assert simulated["real_evidence_coverage"] == 0.0

    result = VenueQualificationEngine().evaluate_qualification(VENUE, simulated)
    assert not result.is_qualified, (
        "a venue was qualified on fills priced against a book that never "
        "existed")
    assert result.real_evidence_coverage == 0.0


def test_the_ev_model_is_told_when_it_was_wrong(storage):
    """
    The chain the seventh report asks for ends here: predicted EV, realised
    return, and the error between them.

    A venue predicted at +14% a trade and realising -2% has an EV model that is
    wrong by 16 points, and no threshold on chart-reading would say so.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=0.3, loss_pnl=-1.0, expected_ev_pct=0.14)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["ev_bias_samples"] == 150
    assert stats["predicted_net_ev_pct"] == pytest.approx(0.14)
    assert stats["ev_bias"] < -0.10, (
        f"the model claimed +14% and got {stats['realised_net_ev_pct']:.3f}")
    assert stats["ev_abs_error"] > 0

    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    assert not result.is_qualified
    assert "EV bias" in result.reasoning


def test_the_sample_size_comes_from_the_outcome_log(storage):
    """The exact bug: this used to read zero however many trades had resolved."""
    tracker = TradeOutcomeTracker(storage=storage)
    trades = storage.log_trade({"market_id": "S0", "venue_id": VENUE,
                                "side": "YES", "position_size_usd": 3.0,
                                "market_price": 0.5, "fair_price": 0.7,
                                "edge": 0.15, "strategy": "value",
                                "execution_mode": "paper", "status": "paper"})
    storage.resolve_trade(trades, outcome=1.0, pnl=1.2)
    tracker.record_trade(trade_id=str(trades), market_id="S0", venue_id=VENUE,
                         strategy="value", forecast_prob=0.7, market_price=0.5,
                         edge=0.15, side="YES", amount_usd=3.0,
                         data_mode="live", execution_mode="paper")
    tracker.record_resolution(str(trades), actual_outcome=1.0, pnl=1.2)

    stats = qualification_stats_from_outcomes(storage, VENUE)
    # The sample size is the whole resolved record; this one trade is paper.
    assert stats["total_resolved_trades"] == 1
    assert stats["total_paper_trades"] == 1
    assert stats["live_trades"] == 0
    assert stats["win_rate"] == 1.0
    assert stats["net_pnl"] == pytest.approx(1.2)


def test_the_statistics_are_computed_not_guessed(storage):
    """
    Every number the gate checks must be derived from the records.

    120 wins at a 75% forecast and 30 losses: the Brier score, the loss and the
    profit factor all have closed-form answers, so this asserts arithmetic rather
    than agreement with itself.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=1.2, loss_pnl=-1.8)
    stats = qualification_stats_from_outcomes(storage, VENUE)

    assert stats["total_paper_trades"] == 150
    assert stats["win_rate"] == pytest.approx(0.8)
    # Brier = 0.8*(0.75-1)^2 + 0.2*(0.75-0)^2 = 0.05 + 0.1125
    assert stats["brier_score"] == pytest.approx(0.1625, abs=1e-6)
    assert stats["net_pnl"] == pytest.approx(120 * 1.2 - 30 * 1.8, abs=1e-6)
    assert stats["profit_factor"] == pytest.approx(144 / 54, abs=1e-6)
    # The agent claimed 75% and delivered 80%, so calibration is close.
    assert stats["calibration_ece"] < 0.15
    assert stats["forecast_skill"] == pytest.approx(1 - 2 * 0.1625, abs=1e-6)


def test_a_losing_venue_scores_as_a_losing_venue(storage):
    """Profit factor below 1 and negative net P&L must be reported as such."""
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.4, forecast=0.6,
            win_pnl=1.0, loss_pnl=-2.0, expected_ev_pct=-0.05)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["win_rate"] == pytest.approx(0.4)
    assert stats["net_pnl"] < 0
    assert stats["profit_factor"] < 1.0


# ----------------------------------------------------------------------
# the gate reads the outcome log, not a file
# ----------------------------------------------------------------------

def _engine(tmp_path):
    return VenueQualificationEngine(data_dir=str(tmp_path))


def test_a_stale_file_is_not_evidence(storage, tmp_path):
    """
    The fail-open path, closed.

    A leftover file claims 120 trades, a Brier score of 0.18 and a profit factor
    of 1.5 - three of the four statistical gates passed on paper that never
    happened. It is produced here the way a test run leaves one behind: by
    writing a real qualification result to disk. With no recorded outcomes, the
    gate must say so instead of reading it back as evidence.
    """
    fabricated = {
        "total_resolved_trades": 120, "win_rate": 0.6, "avg_edge": 0.05,
        "brier_score": 0.18, "log_loss": 0.4, "calibration_ece": 0.05,
        "forecast_skill": 0.7, "profit_paper": 20.0, "profit_live": 0.0,
        "net_pnl": 20.0, "expected_value": 0.05, "profit_factor": 1.5,
        "drawdown_max": 0.05, "fees_total": 0.0, "slippage_total": 0.0,
        "execution_quality_avg": 0.8,
    }
    _engine(tmp_path).evaluate_qualification(VENUE, fabricated)
    assert (tmp_path / "venue_qualification.json").exists(), (
        "the fixture must actually leave a file behind")

    # A fresh engine - as after a restart - loads that file and holds it.
    engine = _engine(tmp_path)
    assert VENUE in engine.qualifications
    assert engine.qualifications[VENUE].total_resolved_trades == 120

    # The outcome log says nothing happened. The file must not win that argument.
    stats = qualification_stats_from_outcomes(storage, VENUE)
    result = engine.evaluate_qualification(VENUE, stats)
    assert result.total_resolved_trades == 0
    assert result.total_paper_trades == 0
    assert result.is_qualified is False
    assert result.brier_score == 1.0, "the stale Brier score must be overwritten"
    assert result.net_pnl == 0.0
    assert result.profit_factor == 0.0


def test_the_cycle_refreshes_the_gate_from_the_outcome_log(tmp_path, monkeypatch):
    """
    The connection the loop was missing: qualification re-evaluated from the
    outcomes every cycle, so a venue's record decides whether it can hold money.
    """
    from src.ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    agent.storage = Storage(db_path=str(tmp_path / "cycle.db"))
    engine = _engine(tmp_path)
    agent.qualification_engine = engine
    agent.capability_engine.qualification_engine = engine
    agent.trade_outcome_tracker.storage = agent.storage

    # Nothing recorded yet: the gate knows nothing.
    assert agent._refresh_qualifications() > 0
    before = engine.qualifications.get(VENUE)
    assert before is None or before.total_paper_trades == 0

    # A good, well-calibrated record.
    tracker = TradeOutcomeTracker(storage=agent.storage)
    _record(agent.storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=1.2, loss_pnl=-1.8)

    agent._refresh_qualifications()
    after = engine.qualifications.get(VENUE)
    assert after.total_paper_trades == 150, (
        "the gate still reads zero trades with 150 resolved - the sample size "
        "is not coming from the outcome log"
    )
    assert after.win_rate == pytest.approx(0.8)
    assert after.brier_score == pytest.approx(0.1625, abs=1e-6)


def test_a_good_record_can_actually_qualify(storage, tmp_path):
    """
    The other half: the gate must be OPENABLE by evidence.

    A permanently closed gate is not a safety feature, it is a system that can
    never trade. This is the same record the API would produce for a venue that
    has done well over 150 trades.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=1.2, loss_pnl=-1.8)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    result = _engine(tmp_path).evaluate_qualification(VENUE, stats)
    failed = [k for k, ok in {
        "min_trades": result.total_paper_trades >= 100,
        "min_win_rate": result.win_rate >= 0.55,
        "max_brier": result.brier_score <= 0.25,
        "min_profit_factor": result.profit_factor >= 1.1,
        "min_net_pnl": result.net_pnl > 0,
    }.items() if not ok]
    assert not failed, f"a genuinely good record failed {failed}"
    assert result.is_qualified is True, (
        f"this record should qualify; reasoning: {result.reasoning[-300:]}"
    )


def test_an_ev_measured_on_a_sliver_of_the_record_does_not_qualify(storage, tmp_path):
    """
    A value is not evidence, and this is the check that says so.

    Ten recorded expected net EVs out of 150 resolved trades: the number itself
    is healthy (+14%), every other criterion is met, and the gate must still
    refuse - because "EV > 1% per trade" would otherwise be a statement about ten
    trades dressed as a statement about the venue.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=1.2, loss_pnl=-1.8, ev_every=15)

    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["expected_value"] > 0, "the value it has IS good"
    assert stats["expected_value_coverage"] == pytest.approx(10 / 150, abs=1e-6)

    result = _engine(tmp_path).evaluate_qualification(VENUE, stats)
    assert result.is_qualified is False, (
        "a venue was qualified on expected values recorded for 10 of its 150 "
        "trades")
    assert "min_ev_coverage" in result.reasoning or "10%" in result.reasoning


def test_an_unmeasured_venue_cannot_qualify_on_an_unlabelled_number(storage, tmp_path):
    """
    Fail closed on the third branch, not just the second.

    With no recorded expected net EV per trade the gate falls back to the mean
    edge and LABELS it, which keeps the console honest - but a labelled
    substitution is still not a measurement. Nothing here records an EV, so the
    venue cannot qualify however good the rest of the record looks.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.8, forecast=0.75,
            win_pnl=1.2, loss_pnl=-1.8, expected_ev_pct=None)

    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["expected_value_samples"] == 0
    assert stats["expected_value"] is None, (
        "nothing was predicted, which is not the same as predicting zero")

    result = _engine(tmp_path).evaluate_qualification(VENUE, stats)
    assert result.is_qualified is False
    assert result.ev_source == "avg_edge fallback (no recorded expected net EV)"
    assert result.ev_coverage == 0.0
    assert "avg_edge fallback" in result.reasoning, (
        "the operator must be told the number they are reading is a substitute")


def test_a_coverage_claim_with_no_samples_is_refused(storage, tmp_path):
    """
    Two fields describe one measurement, and they have to agree.

    A producer that asserts full coverage while reporting zero recorded samples
    is either a bug or a bluff. Both are refused: the gate requires samples AND
    coverage, so neither field can carry the other's claim.
    """
    stats = dict(qualification_stats_from_outcomes(storage, VENUE or "polymarket"))
    stats.update({
        "total_resolved_trades": 120, "win_rate": 0.6, "avg_edge": 0.05,
        "brier_score": 0.18, "log_loss": 0.4, "calibration_ece": 0.05,
        "forecast_skill": 0.7, "profit_paper": 20.0, "net_pnl": 20.0,
        "expected_value": 0.05, "profit_factor": 1.5, "drawdown_max": 0.05,
        "fees_total": 0.0, "slippage_total": 0.0, "execution_quality_avg": 0.8,
        "cost_coverage": 1.0, "execution_quality_coverage": 1.0,
        "expected_value_samples": 0, "expected_value_coverage": 1.0,
    })
    result = _engine(tmp_path).evaluate_qualification("polymarket", stats)
    failed = {k: v for k, v in result.requirements.items() if k == "min_ev_coverage"}
    assert result.is_qualified is False, (
        f"a venue qualified on a coverage claim with no samples: {failed}")


def test_a_mediocre_record_is_refused_for_measurable_reasons(storage, tmp_path):
    """
    And the gate must still say no. A record that is merely adequate - a thin
    edge, a drawdown - is refused, and the refusal names the reason rather than
    hiding behind a missing counter.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _record(storage, tracker, VENUE, 150, win_prob=0.70, forecast=0.70,
            win_pnl=1.35, loss_pnl=-3.0)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    result = _engine(tmp_path).evaluate_qualification(VENUE, stats)
    assert result.total_paper_trades == 150, "it was measured"
    assert result.is_qualified is False, "but it is not good enough"
    # The reasoning must show the numbers it decided on.
    assert "150" in result.reasoning
