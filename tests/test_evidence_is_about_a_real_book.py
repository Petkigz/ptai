"""
The evidence about a venue has to be about a market that existed.

The seventh report's second item is not that the paper engine is too simple -
it is that the QUALITY of the evidence is what gets trusted, and the number of
thresholds is beside the point. Two things were wrong with the quality of it:

  * a fill walked against an assumed default spread was indistinguishable in
    the outcome log from a fill walked down a live ladder, so a venue could be
    "observed" 150 times without ever being observed once;
  * the cost of a trade was understated at the venue that matters most -
    `MultiVenueExecutor` charged Polymarket $0.05 of gas per order even though
    Polymarket relays its orders and the operator pays no gas to trade.

And the chain the report asks for - predicted EV, actual fill, realised P&L,
prediction error - could not be walked because nothing ever compared the
prediction with the result.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pytest

from src.ptai.execution.multi_venue_executor import (
    MultiVenueExecutor,
    _order_gas_usd,
)
from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker,
    qualification_stats_from_outcomes,
)
from src.ptai.storage.db import Storage
from src.ptai.venues.qualification import VenueQualificationEngine

VENUE = "polymarket"


@pytest.fixture()
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "evidence.db"))


def _trade(storage, tracker, i: int, *, book_source=None, execution_mode="paper",
           expected_ev_pct=0.05, pnl=0.10, amount=3.0):
    tid = storage.log_trade({
        "market_id": f"M{i}", "venue_id": VENUE, "side": "YES",
        "position_size_usd": amount, "market_price": 0.5, "fair_price": 0.56,
        "edge": 0.06, "strategy": "value",
        "execution_mode": execution_mode, "status": execution_mode,
    })
    storage.resolve_trade(tid, outcome=1.0, pnl=pnl)
    extra = {} if book_source is None else {"book_source": book_source}
    tracker.record_trade(
        trade_id=str(tid), market_id=f"M{i}", venue_id=VENUE, strategy="value",
        category="politics", forecast_prob=0.56, market_price=0.5, edge=0.06,
        side="YES", amount_usd=amount, fees_usd=0.06, slippage_bps=10.0,
        execution_quality=0.9, execution_mode=execution_mode,
        expected_net_ev=(expected_ev_pct * amount
                         if expected_ev_pct is not None else None),
        expected_net_ev_pct=expected_ev_pct,
        **extra,
    )
    # The outcome row is what the gate reads, and it carries `actual_outcome`
    # only once the trade is RESOLVED through the tracker. Settling the trade
    # row alone leaves the outcome unresolved and the gate sees no evidence.
    tracker.record_resolution(str(tid), actual_outcome=1.0, pnl=pnl)
    return tid


# ----------------------------------------------------------------------
# gas: the mechanism, not a constant
# ----------------------------------------------------------------------

class _Caps:
    def __init__(self, order_gas_usd):
        self.order_gas_usd = order_gas_usd


class _Adapter:
    def __init__(self, venue_id: str, order_gas_usd: Optional[float]):
        self.venue_id = venue_id
        self.capabilities = _Caps(order_gas_usd)


def test_a_relayed_venue_pays_no_gas_in_the_executor():
    """The same $0.05 that was removed from the EV engine was still here."""
    assert _order_gas_usd(_Adapter("polymarket", 0.0), "polymarket", 3.0) == 0.0


def test_a_venue_that_declares_gas_is_charged_what_it_declared():
    assert _order_gas_usd(_Adapter("afx_dex", 0.02), "afx_dex", 3.0) == 0.02


def test_an_undeclared_venue_is_estimated_not_assumed_free():
    """
    Polymarket with no declaration: the cost model answers, not a literal in
    the executor. Its estimate is conservative and its reasoning says where the
    number came from.
    """
    from src.ptai.execution.gas import GasModel
    gas = _order_gas_usd(_Adapter("polymarket", None), "polymarket", 3.0)
    assert gas == pytest.approx(
        GasModel().calculate_gas(operation="place_order", amount_usd=3.0,
                                 venue_id="polymarket").gas_usd)
    assert gas > 0, "an undeclared on-chain venue was costed at zero gas"


def test_the_executor_charges_the_venue_its_own_gas_end_to_end():
    """
    The literal lived in `execute_single`, so test the path that ran it.
    """
    from tests.test_arbitrage_pair_sizing import _market

    class _Venue(_Adapter):
        dry_run = False
        can_place_real_orders = True

        def calculate_fees(self, market, amount_usd):
            return 0.30

        def estimate_slippage(self, market, amount_usd):
            return 0.0

        async def place_order(self, opportunity, max_spend_usd, max_price):
            return {"status": "matched", "filled_usd": max_spend_usd,
                    "price": max_price, "order_id": "o1"}

    class _Registry:
        def __init__(self):
            self.adapters = {"polymarket": _Venue("polymarket", 0.0)}
            self.get_adapter_for_venue_id = self.adapters.get

    from src.ptai.venues.adapter import VenueOpportunity, VenueType

    opp = VenueOpportunity(
        market=_market("G-1", 0.50), venue_id="polymarket",
        venue_type=VenueType.PREDICTION, side="YES", market_price=0.50,
        estimated_fair=0.60, raw_edge=0.10, effective_edge=0.10,
        confidence=0.8, should_trade=True)

    ex = MultiVenueExecutor(registry=_Registry(), bankroll=50.0)
    ex.rate_limits = {}
    ex.check_rate_limit = lambda venue_id: True
    result = asyncio.run(
        ex.execute_single(opp, max_spend_usd=3.0, max_price=0.52))
    assert result.gas_usd == 0.0, (
        f"Polymarket was charged ${result.gas_usd:.4f} of gas on a relayed order")
    assert result.fees_usd == pytest.approx(0.30), "the venue's own fee stands"


# ----------------------------------------------------------------------
# where the evidence came from
# ----------------------------------------------------------------------

def test_an_assumed_book_is_not_evidence(storage):
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1, book_source="assumed_default")
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["total_resolved_trades"] == 1
    assert stats["real_evidence_samples"] == 0
    assert stats["real_evidence_coverage"] == 0.0


def test_a_real_ladder_is_evidence(storage):
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1, book_source="ladder")
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["real_evidence_samples"] == 1
    assert stats["real_evidence_coverage"] == 1.0


def test_an_unlabelled_fill_is_not_evidence(storage):
    """Fail closed: a row that predates the measurement is not a real one."""
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1)          # no book_source at all
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["real_evidence_coverage"] == 0.0


def test_a_live_fill_is_its_own_evidence(storage):
    """
    Real money crossed a real book. There is no simulation to mistrust, so a
    live fill does not need the caller to remember a label.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1, execution_mode="live")
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["real_evidence_coverage"] == 1.0


def test_a_caller_can_declare_its_own_evidence_fabricated(storage):
    """An explicit False overrides even a live label - the caller knows best."""
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1, execution_mode="live")
    storage.conn.execute("UPDATE trade_outcomes SET fill_is_real = 0")
    storage.conn.commit()
    assert qualification_stats_from_outcomes(
        storage, VENUE)["real_evidence_coverage"] == 0.0


def test_coverage_is_a_fraction_not_a_flag(storage):
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(4):
        _trade(storage, tracker, i, book_source="orderbook" if i < 3 else "unknown")
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["real_evidence_samples"] == 3
    assert stats["real_evidence_coverage"] == pytest.approx(0.75)


# ----------------------------------------------------------------------
# predicted EV against realised return
# ----------------------------------------------------------------------

def test_the_model_is_told_when_it_was_wrong(storage):
    tracker = TradeOutcomeTracker(storage=storage)
    # Predicted +5% a trade, realised +3.33% (0.10 on a 3.00 stake).
    for i in range(10):
        _trade(storage, tracker, i, book_source="ladder",
               expected_ev_pct=0.05, pnl=0.10)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["predicted_net_ev_pct"] == pytest.approx(0.05)
    assert stats["realised_net_ev_pct"] == pytest.approx(0.10 / 3.0)
    assert stats["ev_bias"] == pytest.approx(0.10 / 3.0 - 0.05)
    assert stats["ev_bias_samples"] == 10


def test_a_model_that_promises_and_does_not_deliver_is_refused(storage):
    """
    Predicted +14%, realised -2%: this is not bad luck, it is a broken EV
    model, and the gate must not accept the prediction as its evidence.
    """
    tracker = TradeOutcomeTracker(storage=storage)
    for i in range(150):
        _trade(storage, tracker, i, book_source="ladder",
               expected_ev_pct=0.14, pnl=-0.06)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["ev_bias"] < -0.15
    result = VenueQualificationEngine().evaluate_qualification(VENUE, stats)
    assert not result.is_qualified
    assert result.ev_bias == pytest.approx(stats["ev_bias"])


def test_no_prediction_means_no_bias_measured(storage):
    """None, not 0.0: "never compared" is not "perfectly calibrated"."""
    tracker = TradeOutcomeTracker(storage=storage)
    _trade(storage, tracker, 1, book_source="ladder", expected_ev_pct=None)
    stats = qualification_stats_from_outcomes(storage, VENUE)
    assert stats["ev_bias"] is None
    assert stats["ev_bias_samples"] == 0
    assert qualification_stats_from_outcomes(
        storage, "nobody")["ev_bias"] is None


# ----------------------------------------------------------------------
# the loop writes the label
# ----------------------------------------------------------------------

def test_the_loop_records_which_book_the_fill_walked():
    """
    Source-level, because the value comes from a fill that only exists at
    runtime: if the keyword is not at the call site, no row will ever carry it
    and the coverage check above would simply refuse every venue forever.
    """
    from pathlib import Path
    src = Path("src/ptai/agent/v3_loop.py").read_text()
    # The open-position call site is the LAST one; the earlier one resolves a
    # delayed fill and has its own fields.
    idx = src.rindex("self.trade_outcome_tracker.record_trade(")
    # The window has to cover the whole call, comments included.
    call = src[idx:src.index("\n                    )", idx)]
    assert "book_source=" in call, "the fill's book never reaches the outcome row"
    assert "venue_fill" in call, "a live fill must be labelled as its own evidence"
    assert "gas_usd=" in call, "the gas charged must be recorded with the trade"
