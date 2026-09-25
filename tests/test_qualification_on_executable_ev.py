"""
Qualification judges the EV the fills actually had, not the one the model hoped.

The seventh report's third item: qualification must rest on REALISTIC EXECUTABLE
NET EV plus realized execution performance, not on a prediction.

`expected_net_ev_pct` is the prediction - computed at the price the book showed
when the opportunity was found. That is a model of execution. A venue whose
orders consistently fill three cents worse than modelled passes every
prediction-based EV threshold ever written while losing those three cents on
every trade, and nothing in the record could see it because the prediction and
the fill were never compared.

Two numbers fix that, and they answer different questions:

  * `executable_net_ev_pct` - the same EV recomputed at the price the order
    ACTUALLY paid and the fees it actually incurred. This is what the trade was
    worth, and it is what the gate now judges.
  * `fill_price_vs_modelled` - where the fill landed against the price the
    decision was made at. Positive means paid more than modelled.

There is deliberately no "execution gap" in return terms: the model's net EV
also deducts an uncertainty penalty and an execution-loss penalty, which are
conservatism rather than cash and whose size depends on how uncertain the agent
felt. Subtracting the two would produce a number that moves when confidence
moves, and nobody could say what it measured.
"""

from __future__ import annotations

import pytest

from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker,
    qualification_stats_from_outcomes,
)
from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.storage.db import Storage
from src.ptai.strategy.expected_ev import executable_net_ev, gross_ev_at_price
from src.ptai.venues.adapter import VenueOpportunity, VenueType
from src.ptai.venues.qualification import VenueQualificationEngine

VENUE = "polymarket"


def _market(market_id: str = "M-1", price: float = 0.60) -> Market:
    return Market(
        id=market_id, source=MarketSource.POLYMARKET,
        question=f"Will {market_id} happen?", outcomes=["YES", "NO"],
        outcome_prices=[price, 1 - price],
        tokens=[Token(token_id=market_id, outcome="YES", price=price)],
        volume=200_000.0, volume_24h=100_000.0, liquidity=50_000.0,
        raw={"orderbook": {"bids": [{"price": "0.59", "size": "5000"}],
                           "asks": [{"price": "0.60", "size": "5000"}],
                           "spread": 0.01, "is_real": True}},
    )


def _opp(side: str = "YES", price: float = 0.60, fair: float = 0.70):
    return VenueOpportunity(
        market=_market(price=price), venue_id=VENUE,
        venue_type=VenueType.PREDICTION, side=side, market_price=price,
        estimated_fair=fair, raw_edge=0.10, effective_edge=0.10,
        confidence=0.8, should_trade=True,
    )


# ----------------------------------------------------------------------
# the arithmetic
# ----------------------------------------------------------------------

def test_the_gross_formula_matches_the_engine_it_was_factored_out_of():
    """20c of edge at 0.60 on $3: 3/0.6 = 5 shares, 0.10 of edge = $0.50."""
    assert gross_ev_at_price("YES", fair_prob=0.70, entry_price=0.60,
                             amount_usd=3.0) == pytest.approx(0.50)


def test_an_executable_ev_at_the_modelled_price_is_the_modelled_ev():
    """
    The two numbers must agree when nothing went wrong.

    If the fill got exactly the modelled price and the venue charged nothing,
    the executable EV IS the model's EV. A refactor that broke that agreement
    would manufacture an execution gap out of algebra.
    """
    from src.ptai.strategy.expected_ev import ExpectedNetEVEngine

    opp = _opp()
    modelled = ExpectedNetEVEngine().calculate(opp, 3.0, opp.market.raw["orderbook"])
    executable = executable_net_ev(opp, filled_usd=3.0, filled_price=0.60,
                                   fees_usd=0.0, gas_usd=0.0,
                                   modelled_net_ev_pct=modelled.net_ev_pct,
                                   modelled_price=0.60)
    assert executable.gross_ev_usd == pytest.approx(modelled.gross_ev_usd)
    assert executable.price_paid_vs_modelled == pytest.approx(0.0)


def test_a_worse_fill_lowers_the_executable_ev_and_the_gap_records_it():
    """
    Filled at 0.63 instead of 0.60: three cents of the edge gone, on every
    share, and the gap says exactly how much.
    """
    opp = _opp()
    good = executable_net_ev(opp, 3.0, 0.60, fees_usd=0.0,
                             modelled_net_ev_pct=0.10, modelled_price=0.60)
    worse = executable_net_ev(opp, 3.0, 0.63, fees_usd=0.0,
                              modelled_net_ev_pct=0.10, modelled_price=0.60)

    assert worse.net_ev_pct < good.net_ev_pct
    assert worse.price_paid_vs_modelled == pytest.approx(0.05, abs=1e-9), (
        "paying 0.63 for something the decision priced at 0.60 is 5% more")
    # ...and the price delta is in price units, so it cannot be confused with a
    # return. The EV consequence of it is net_ev_pct, which is what the gate
    # judges.
    assert worse.price_paid_vs_modelled == pytest.approx(
        (0.63 - 0.60) / 0.60)


def test_slippage_is_not_charged_twice():
    """
    The achieved price already walked the ladder, so the fill price contains the
    slippage. Deducting a slippage term as well would charge the same cost twice
    - which this project has already done once, on the 8% hunt threshold.
    """
    opp = _opp()
    # Filled three ticks worse, and the venue charged a fee. The deduction is
    # the fee and only the fee.
    result = executable_net_ev(opp, 3.0, 0.63, fees_usd=0.06, gas_usd=0.0,
                               modelled_net_ev_pct=0.10, modelled_price=0.60)
    assert result.fees_usd == pytest.approx(0.06)
    assert result.net_ev_usd == pytest.approx(result.gross_ev_usd - 0.06)


def test_a_no_position_is_priced_at_the_no_token_price():
    """
    A NO position is a BUY of the NO token, and the venue reports the NO price.

    Passing the YES price here turns a losing NO trade into a winning one - the
    unit error this project has made before, and the reason the gross formula
    takes a token price rather than a market price.
    """
    # YES at 0.60 means the NO token costs 0.40. Fair YES 0.70 -> fair NO 0.30:
    # buying NO at 0.40 with a 30% chance is a LOSING trade.
    opp = _opp(side="NO", price=0.60, fair=0.70)
    result = executable_net_ev(opp, 3.0, 0.40, fees_usd=0.0)
    assert result.gross_ev_usd < 0, (
        "a NO bought at 0.40 with a 30% chance of paying out was priced as a "
        "winner, which means the YES price was used")

    # ...and the same NO bought cheap is a winning trade.
    cheap = executable_net_ev(opp, 3.0, 0.25, fees_usd=0.0)
    assert cheap.gross_ev_usd > 0


def test_an_unpriceable_fill_returns_none_not_zero():
    """No capital, or a price off the tick grid end, is not a break-even trade."""
    opp = _opp()
    assert executable_net_ev(opp, filled_usd=0.0, filled_price=0.60) is None
    assert executable_net_ev(opp, filled_usd=3.0, filled_price=0.0) is None
    assert executable_net_ev(opp, filled_usd=3.0, filled_price=1.0) is None


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------

@pytest.fixture()
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "executable.db"))


def _record(storage, tracker, i, *, fill_price=0.60, modelled_pct=0.10,
            outcome=1.0, pnl=0.10, forecast=0.70, fee=0.0):
    trade_id = storage.log_trade({
        "market_id": f"M{i}", "venue_id": VENUE, "side": "YES",
        "position_size_usd": 3.0, "market_price": 0.60, "fair_price": 0.70,
        "edge": 0.10, "strategy": "value",
        "execution_mode": "paper", "status": "paper",
    })
    storage.resolve_trade(str(trade_id), outcome=outcome, pnl=pnl)
    # `modelled_price` is the price the DECISION was made at - the market price
    # the opportunity was found at - so the fill can be compared with it.
    executable = executable_net_ev(_opp(fair=forecast), 3.0, fill_price,
                                   fees_usd=fee,
                                   modelled_net_ev_pct=modelled_pct,
                                   modelled_price=0.60)
    tracker.record_trade(
        trade_id=str(trade_id), market_id=f"M{i}", venue_id=VENUE,
        strategy="value", category="politics", forecast_prob=forecast,
        market_price=0.60, edge=0.10, side="YES", amount_usd=3.0,
        fees_usd=0.06, slippage_bps=10.0, execution_quality=0.9,
        execution_mode="paper", book_source="ladder",
        expected_net_ev=modelled_pct * 3.0, expected_net_ev_pct=modelled_pct,
        executable_net_ev=executable.net_ev_usd,
        executable_net_ev_pct=executable.net_ev_pct,
        fill_price_vs_modelled=executable.price_paid_vs_modelled,
    )
    tracker.record_resolution(str(trade_id), actual_outcome=outcome, pnl=pnl)
    return trade_id


def test_the_stats_carry_the_executable_figures(storage):
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(10):
        _record(storage, tracker, i)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["executable_value_samples"] == 10
    assert stats["executable_value_coverage"] == 1.0
    # The executable EV is NOT the modelled EV, deliberately: the engine also
    # deducts an uncertainty penalty and an execution-loss penalty, which are
    # conservatism rather than cash. Here they are the difference between
    # +16.7% of gross outcome and +10% of modelled return - which is exactly why
    # a "gap" between the two would not have measured execution.
    assert stats["executable_value"] == pytest.approx(0.5 / 3.0, abs=0.01)
    assert stats["fill_price_vs_modelled"] == pytest.approx(0.0)
    assert stats["price_paid_samples"] == 10


def _healthy_stats():
    """
    A qualification-stats dict where every bar except the executable EV passes.

    Copied from `test_qualification_is_fed_by_outcomes.py`'s fixtures rather than
    invented: a forecast skill of 0.82, a Brier of 0.09 and a drawdown of 0.0 are
    what a genuinely well-behaved venue looks like, so a refusal here can only
    come from the field under test.
    """
    return {
        "total_resolved_trades": 150, "win_rate": 1.0, "avg_edge": 0.10,
        "net_pnl": 15.0, "profit_paper": 15.0, "profit_live": 0.0,
        "profit_factor": 3.0, "brier_score": 0.09, "log_loss": 0.357,
        "calibration_ece": 0.05, "forecast_skill": 0.82,
        "drawdown_max": 0.0, "execution_quality_avg": 0.9,
        "expected_value": 0.07, "ev_coverage": 1.0, "ev_source": "recorded",
        "ev_bias": 0.0, "ev_bias_samples": 150,
        "real_evidence_coverage": 1.0, "real_evidence_samples": 150,
        "executable_value": 0.02, "executable_value_samples": 150,
        "executable_value_coverage": 1.0,
        "fill_price_vs_modelled": 0.0, "price_paid_samples": 150,
        "price_paid_coverage": 1.0, "cost_coverage": 1.0,
        "fees_total": 9.0, "slippage_total": 0.45,
    }


def _healthy_record(storage, tracker, *, fill_price=0.60, fee=0.0,
                    modelled_pct=0.07, forecast=0.85, win_every=20,
                    wins_per_group=17, win_pnl=0.30):
    """
    A record that passes EVERY gate except the one under test.

    Each of the checks below is individually sufficient to refuse a bad venue,
    which is exactly why a test that only asserts `not is_qualified` proves
    nothing about the check it names - the first version of these tests passed
    with the check deleted. So the fixtures are built to isolate one failure:
    everything the gate looks at is healthy, and the single defect is the one
    being tested.
    """
    for i in range(150):
        won = (i % win_every) < wins_per_group
        _record(storage, tracker, i, fill_price=fill_price, fee=fee,
                modelled_pct=modelled_pct, forecast=forecast,
                outcome=1.0 if won else 0.0,
                pnl=win_pnl if won else -win_pnl)


def test_the_executable_ev_bar_refuses_a_record_that_misses_it(storage):
    """
    The bar itself, judged on a record that is healthy in every other field.

    This one is deliberately assembled by hand rather than grown from a fixture,
    because of what the check IS. A venue whose fills really destroyed the edge
    would show it in its realised P&L too - the two are measured on the same
    trades, so no honest fixture can have a healthy realised record and a
    negative executable one. What the bar adds is a floor that is evaluated at
    the fill instead of from the prediction, and this is the only way to see it
    refuse on its own.
    """
    engine = VenueQualificationEngine()
    stats = _healthy_stats()
    stats["executable_value"] = 0.0
    stats["executable_value_samples"] = 150
    stats["executable_value_coverage"] = 1.0

    result = engine.evaluate_qualification(VENUE, stats)
    assert result.checks["min_executable_ev"] is False, (
        "an executed record worth 0% was passed by the executable EV bar")
    assert not result.is_qualified, (
        f"a venue whose fills were worth nothing was qualified: {result.reasoning}")
    assert result.checks["min_executable_ev"] is False, (
        "the refusal must come from this check, or the test is vacuous")
    assert "EXECUTABLE net EV" in result.reasoning

    # ...and the bar is a bar, not a wall: the same record at 2% passes it.
    stats["executable_value"] = 0.02
    assert engine.evaluate_qualification(VENUE, stats).checks["min_executable_ev"]


def test_only_the_fill_price_fails_when_the_fills_are_worse_than_modelled(storage):
    """
    A big enough edge that the trade is still worth having at the worse price -
    so the EXECUTABLE EV passed - but the venue systematically fills 3% above
    the price every decision was made at, which is not a venue this system can
    model.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _healthy_record(storage, tracker, fill_price=0.618)
    stats = qualification_stats_from_outcomes(storage, VENUE)

    assert stats["executable_value"] > 0.01, (
        "the edge must survive the worse fill, or the other check catches it")
    assert stats["fill_price_vs_modelled"] == pytest.approx(0.03, abs=0.002)

    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    assert not result.is_qualified, (
        "a venue that fills 3% worse than modelled was qualified anyway: "
        f"{result.reasoning}")
    assert result.checks["fills_at_modelled_price"] is False, (
        "the venue was refused, but not by the fill-price check - which means "
        "this test would still pass with that check deleted")
    assert "fills at the modelled price" in result.reasoning


def test_a_venue_measured_at_its_fills_can_qualify(storage):
    """
    The same record with no execution loss qualifies - the check is not a wall.

    Every number here has to line up or the fixture is testing the wrong gate: a
    forecast of 0.75 against a 75% win rate, and a realised return that equals
    the prediction, so the EV-bias and calibration checks pass on their merits
    rather than by luck.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(150):
        won = (i % 4) != 3                      # 75% wins, spread evenly
        _record(storage, tracker, i, fill_price=0.60, modelled_pct=0.05,
                forecast=0.75, outcome=1.0 if won else 0.0,
                pnl=0.30 if won else -0.30)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    # 150 trades do not divide into quarters, so the win count is one trade off
    # 75% - a rounding artefact of the pattern, not a model error, and the gate's
    # bar is 1% so this stays well inside it.
    assert stats["ev_bias"] == pytest.approx(0.0, abs=2e-3), (
        "the fixture's prediction must match what it realises")
    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    # At the fill: $3 buys 5 shares at 0.60, worth $1 each, with the agent's own
    # 75% forecast on them - 0.75 x $2.00 won minus 0.25 x $3.00 lost = $0.75,
    # which is 25% of the stake. Stated as arithmetic rather than copied from
    # the function under test.
    assert result.executable_value == pytest.approx(0.25, abs=0.01)
    assert result.fill_price_vs_modelled == pytest.approx(0.0)
    assert result.is_qualified, (
        f"a cleanly executed record was refused: {result.reasoning}")


def test_the_gate_fails_closed_when_nothing_was_repriced_at_its_fill(storage):
    """
    A record whose EV was only ever predicted has not shown that its edge
    survives execution. Unmeasured is a failure, not a pass - the same rule the
    cost and EV coverage checks already follow.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(150):
        _record(storage, tracker, i)
    # Strip the executable measurement, as an older system would have left it.
    # ONLY the executable EV is stripped. The fill prices stay recorded, so the
    # price check still has its evidence - otherwise this test would pass on the
    # other check failing and would prove nothing about this one.
    storage.conn.execute("UPDATE trade_outcomes SET executable_net_ev = NULL, "
                         "executable_net_ev_pct = NULL")
    storage.conn.commit()
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["executable_value_samples"] == 0
    assert stats["executable_value"] is None
    assert stats["fill_price_vs_modelled"] is not None, (
        "the price evidence is still there; only the EV is unmeasured")

    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    assert not result.is_qualified
    assert "unmeasured" in result.reasoning


def test_execution_and_forecasting_are_judged_separately(storage):
    """
    Two different questions, two different numbers.

    Here the fills land exactly where modelled and the forecasts are wrong
    (every trade loses). The FILLS check passes - nothing went wrong in
    execution - and the bias check is what refuses this venue.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(150):
        _record(storage, tracker, i, fill_price=0.60, modelled_pct=0.14,
                outcome=0.0, pnl=-0.30)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["fill_price_vs_modelled"] == pytest.approx(0.0), (
        "execution was clean, so the fills landed where they were modelled")
    assert stats["ev_bias"] < -0.05, "the model was wrong and that is the bias"

    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    assert not result.is_qualified
    assert "EV bias" in result.reasoning


# ----------------------------------------------------------------------
# the loop feeds it
# ----------------------------------------------------------------------

def test_a_real_cycle_records_the_executable_ev_on_the_outcome(tmp_path,
                                                               monkeypatch):
    """
    Fed, not merely called: the loop has to pass the fill's price and cost into
    the outcome row, or the gate above would simply refuse every venue forever.
    """
    from tests.test_full_cycle_from_discovery_to_allocation import (
        _cycle, _inject_opportunity, _simulating_place_order,
        build_agent, VENUE,
    )

    agent, adapter = build_agent(tmp_path, dry_run=True)
    adapter.place_order = _simulating_place_order(adapter)
    _inject_opportunity(agent, adapter._market(), monkeypatch, edge=0.15)
    try:
        _cycle(agent)
        rows = agent.storage.conn.execute(
            "SELECT expected_net_ev_pct, executable_net_ev_pct, "
            "fill_price_vs_modelled "
            "FROM trade_outcomes").fetchall()
        assert rows, "the cycle recorded no outcome at all"
        priced = [r for r in rows if r["executable_net_ev_pct"] is not None]
        assert priced, (
            "the outcome carries no executable EV, so qualification can never "
            f"judge what the fills were worth: {[dict(r) for r in rows]}")
        row = priced[0]
        assert row["fill_price_vs_modelled"] is not None
    finally:
        agent.storage.close()
