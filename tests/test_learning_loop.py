"""
The learning chain: execution -> settlement -> outcome -> calibration.

Before this, the chain was open at the resolution end and the agent could not
learn from any of its own results. Specifically:

  * `CalibrationEngine.record_forecast` had two production callers;
    `record_resolution` had ZERO, anywhere in `src/ptai/`.
  * `is_degrading()` is guarded by `resolved >= 50`, so with nothing ever
    resolved it always returned False, and
    `kill_switch.check_calibration_collapse(...)` - the mechanism meant to stop
    an agent whose probabilities had stopped working - could never fire.
  * There was no `calibration` table. Every forecast insert raised "no such
    table", was swallowed into a warning, and the forecast lived only in an
    in-memory list on an object rebuilt each cycle.
  * `trades.resolved` was written as 0 and never updated, and V3 never wrote to
    the trades table at all, so `get_performance_summary()` computed win rate
    over `WHERE resolved=1` - an empty set forever. The bankroll could only go
    down when a position was opened and never up when one was won.
  * `except: pass` wrapped the trade-outcome recording.

These tests execute the chain end to end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.ptai.execution.settlement import SettlementEngine, compute_pnl
from src.ptai.learning.calibration_db import CalibrationDB
from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
from src.ptai.storage.db import Storage
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueType,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

class _Venue(MarketAdapter):
    """A venue that answers settlement, with a controllable answer."""

    def __init__(self, outcome=None, settled=True, is_real=True, source="test_api"):
        super().__init__(venue_id="testvenue", venue_type=VenueType.PREDICTION)
        self.capabilities = AdapterCapability(supports_trading=True)
        self._outcome, self._settled, self._is_real, self._source = (
            outcome, settled, is_real, source)
        self.asked = []

    def check_eligibility(self, country_code="UG"):
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count=100, **kwargs):
        return []

    async def get_orderbook(self, market):
        return {}

    async def get_portfolio(self):
        return {}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        return {}

    async def get_settlement(self, market_id):
        self.asked.append(market_id)
        if not self._is_real:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": self._source, "reason": "cannot report settlement"}
        return {"settled": self._settled, "outcome": self._outcome,
                "is_real": True, "source": self._source, "reason": "settled"}


class _Registry:
    def __init__(self, adapter):
        self.adapters = {"testvenue": adapter}


@pytest.fixture()
def chain(tmp_path):
    """A storage + calibration + settlement stack on a throwaway database."""
    storage = Storage(db_path=str(tmp_path / "t.db"))
    cal = CalibrationDB(db_path=str(tmp_path / "cal.json"), storage=storage)
    tracker = TradeOutcomeTracker(storage=storage)
    venue = _Venue(outcome=1.0)
    engine = SettlementEngine(
        venue_registry=_Registry(venue), storage=storage,
        calibration_engine=cal, trade_outcome_tracker=tracker,
    )
    storage.set_bankroll(50.0)
    yield storage, cal, tracker, engine, venue
    storage.close()


def _open_position(storage, cal, market_id="MK-1", side="YES", price=0.60,
                   stake=3.0, forecast=0.75, venue_id="testvenue"):
    trade_id = storage.log_trade({
        "market_id": market_id, "market_question": "Will X?", "side": side,
        "market_price": price, "fair_value": forecast, "edge": 0.15,
        "position_size_usd": stake, "confidence": 0.8, "status": "open",
        "venue_id": venue_id,
    })
    forecast_id = cal.record_forecast(
        market_id, "Will X?", forecast, 0.8, price, "politics",
        venue_id=venue_id, trade_id=trade_id)
    return trade_id, forecast_id


# --------------------------------------------------------------------------
# the chain
# --------------------------------------------------------------------------

class TestSettlementClosesTheChain:
    def test_winning_position_settles_and_pays(self, chain):
        storage, cal, _, engine, _ = chain
        trade_id, _ = _open_position(storage, cal)

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 1
        # YES at 0.60 with $3 -> 5 shares -> $5 payout -> +$2
        assert report.realised_pnl_usd == pytest.approx(2.0)
        assert storage.get_bankroll() == pytest.approx(52.0)

    def test_losing_position_settles_and_costs_the_stake(self, chain):
        storage, cal, _, engine, venue = chain
        venue._outcome = 0.0
        _open_position(storage, cal)

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 1
        assert report.realised_pnl_usd == pytest.approx(-3.0)
        assert storage.get_bankroll() == pytest.approx(47.0)

    def test_resolution_reaches_the_calibration_engine(self, chain):
        """
        The single most important assertion in this file: record_resolution now
        has a production path that reaches it.
        """
        storage, cal, _, engine, _ = chain
        _open_position(storage, cal)
        assert cal.resolved_count() == 0

        asyncio.run(engine.settle_pending())

        assert cal.resolved_count() == 1
        # forecast 0.75, outcome 1.0 -> (0.75-1)^2 = 0.0625
        assert cal.calculate_brier_score() == pytest.approx(0.0625)

    def test_trade_row_is_closed_and_win_rate_becomes_possible(self, chain):
        storage, cal, _, engine, _ = chain
        _open_position(storage, cal)

        assert storage.count_open_positions() == 1
        asyncio.run(engine.settle_pending())

        assert storage.count_open_positions() == 0
        assert storage.get_unresolved_trades() == []
        summary = storage.get_performance_summary()
        assert summary["win_rate"] == pytest.approx(100.0), (
            "win rate is computed over resolved trades; with settlements now "
            "landing it must stop being permanently zero"
        )

    def test_settlement_is_idempotent(self, chain):
        """A second pass must not pay the same position twice."""
        storage, cal, _, engine, _ = chain
        _open_position(storage, cal)

        first = asyncio.run(engine.settle_pending())
        bankroll_after_first = storage.get_bankroll()
        second = asyncio.run(engine.settle_pending())

        assert first.settled == 1
        assert second.settled == 0
        assert storage.get_bankroll() == pytest.approx(bankroll_after_first)

    def test_learning_gap_is_reported_not_swallowed(self, chain):
        """
        The `except: pass` was removed. A failure to record must surface.
        """
        import inspect
        from src.ptai.agent import v3_loop

        source = inspect.getsource(v3_loop.TradingAgentV3.run_cycle)
        assert "except:\n                            pass" not in source, (
                "a bare except: pass around learning recording is back")
        assert "learning_problems" in source, (
            "recording failures must be collected and reported on the result")


class TestKillSwitchCanActuallyFire:
    """
    `is_degrading()` needs 50 resolved forecasts. With nothing ever resolved it
    returned False forever, so the calibration-collapse kill switch was
    unreachable code. Prove it is reachable now.
    """

    def test_degradation_is_reachable_after_settlement(self, chain):
        storage, cal, _, engine, venue = chain

        # A model that says 0.90 and is wrong every time.
        for i in range(55):
            _open_position(storage, cal, market_id=f"MK-{i}", forecast=0.90)

        assert cal.is_degrading() is False, "should not fire before resolution"
        venue._outcome = 0.0
        asyncio.run(engine.settle_pending(max_markets=100))

        assert cal.resolved_count() == 55
        assert cal.calculate_brier_score() > 0.3
        assert cal.is_degrading() is True, (
            "with 55 resolved and badly wrong forecasts, degradation must be "
            "detectable - before this fix it structurally could not be"
        )

    def test_kill_switch_halts_trading_on_collapse(self, chain, tmp_path):
        from src.ptai.risk.kill_switch import KillSwitch

        storage, cal, _, engine, venue = chain
        for i in range(55):
            _open_position(storage, cal, market_id=f"MK-{i}", forecast=0.90)
        venue._outcome = 0.0
        asyncio.run(engine.settle_pending(max_markets=100))

        ks = KillSwitch(data_dir=str(tmp_path / "ks"))
        assert ks.can_trade() is True
        if cal.is_degrading():
            ks.check_calibration_collapse(cal.calculate_brier_score())
        assert ks.can_trade() is False, (
            "a collapsed calibration must stop new trades"
        )

    def test_good_calibration_does_not_trigger(self, chain):
        """The check must discriminate, not just always fire."""
        storage, cal, _, engine, venue = chain
        for i in range(55):
            # Forecast 0.90, and mostly right: Brier ~0.09
            venue._outcome = 1.0 if i % 10 else 0.0
            _open_position(storage, cal, market_id=f"MK-{i}", forecast=0.90)
            asyncio.run(engine.settle_pending(max_markets=100))

        assert cal.calculate_brier_score() < 0.3
        assert cal.is_degrading() is False


class TestUnreadableSettlementInventsNothing:
    """
    An outcome that cannot be read is not an outcome. Guessing one writes a
    permanent, wrong calibration point and the agent then adjusts its
    probabilities with invented data.
    """

    def test_unsupported_venue_records_nothing(self, chain):
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)
        venue._is_real = False

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 0
        assert report.unreadable == 1
        assert cal.resolved_count() == 0
        assert storage.get_bankroll() == pytest.approx(50.0), (
            "an unreadable settlement must not move the bankroll"
        )
        assert storage.count_open_positions() == 1, "the position stays open"

    def test_unresolved_market_is_not_settled(self, chain):
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)
        venue._settled = False

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 0
        assert report.unresolved == 1
        assert cal.resolved_count() == 0

    def test_ambiguous_outcome_is_refused(self, chain):
        """
        Closed, but not a clean 0/1. Recording 0.98 as "YES won" would be a
        permanent wrong calibration point.
        """
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)
        venue._outcome = None

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 0
        assert report.ambiguous == 1
        assert cal.resolved_count() == 0

    def test_unsupported_venue_is_not_reasked_every_cycle(self, chain):
        """
        A venue that structurally cannot report settlement is skipped for the
        rest of the run. It is not the same as a transient failure: a network
        error must be retried next cycle, or a resolved market would be missed
        forever because of one bad response.
        """
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)
        venue._is_real = False
        venue._source = "unsupported"

        asyncio.run(engine.settle_pending())
        first_asks = len(venue.asked)
        assert first_asks == 1
        asyncio.run(engine.settle_pending())

        assert len(venue.asked) == first_asks, (
            "a venue that cannot report settlement should be skipped for the "
            "rest of the run, not queried every cycle"
        )
        assert "testvenue" in engine.get_report()["unsupported_venues_this_run"]

    def test_transient_failure_is_retried_next_cycle(self, chain):
        """
        The opposite case: a lookup that failed for a network reason must be
        retried, not written off.
        """
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)
        venue._is_real = False
        venue._source = "gamma_unavailable"

        asyncio.run(engine.settle_pending())
        asyncio.run(engine.settle_pending())

        assert len(venue.asked) == 2, (
            "a transient lookup failure must be retried on the next cycle"
        )

    def test_a_retry_that_succeeds_still_settles(self, chain):
        """A market that was unreadable once must not be lost forever."""
        storage, cal, _, engine, venue = chain
        _open_position(storage, cal)

        venue._is_real = False
        venue._source = "gamma_unavailable"
        first = asyncio.run(engine.settle_pending())
        assert first.settled == 0

        venue._is_real = True
        venue._outcome = 1.0
        second = asyncio.run(engine.settle_pending())
        assert second.settled == 1
        assert cal.resolved_count() == 1

    def test_no_venue_associated_records_nothing(self, chain):
        storage, cal, _, engine, _ = chain
        cal.record_forecast("ORPHAN", "Q?", 0.6, 0.7, 0.5, "default")

        report = asyncio.run(engine.settle_pending())

        assert report.settled == 0
        assert report.unreadable == 1
        assert cal.resolved_count() == 0

    def test_gamma_style_ambiguity_is_rejected(self):
        """
        PolymarketAdapter.get_settlement must not round 0.98/0.02 to a win.
        """
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        adapter = PolymarketAdapter()
        adapter.client.get_market_resolution = lambda mid: {
            "closed": True, "outcomePrices": '["0.98", "0.02"]'}

        verdict = asyncio.run(adapter.get_settlement("123"))

        assert verdict["settled"] is True
        assert verdict["outcome"] is None, (
            "a 0.98 settlement is not a 1.0 settlement"
        )
        assert "not a clean 0/1 pair" in verdict["reason"]

    @pytest.mark.parametrize("prices,expected", [
        ('["1", "0"]', 1.0),
        ('["0", "1"]', 0.0),
    ])
    def test_gamma_clean_settlement_is_read(self, prices, expected):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        adapter = PolymarketAdapter()
        adapter.client.get_market_resolution = lambda mid: {
            "closed": True, "outcomePrices": prices}

        verdict = asyncio.run(adapter.get_settlement("123"))
        assert verdict["outcome"] == expected
        assert verdict["is_real"] is True

    def test_open_market_is_not_settled(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        adapter = PolymarketAdapter()
        adapter.client.get_market_resolution = lambda mid: {
            "closed": False, "outcomePrices": '["0.6", "0.4"]'}

        verdict = asyncio.run(adapter.get_settlement("123"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None

    def test_lookup_failure_is_not_a_settlement(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        adapter = PolymarketAdapter()
        adapter.client.get_market_resolution = lambda mid: None

        verdict = asyncio.run(adapter.get_settlement("123"))
        assert verdict["settled"] is False
        assert verdict["is_real"] is False
        assert verdict["source"] == "gamma_unavailable"

    def test_default_adapter_settlement_does_not_guess(self):
        from src.ptai.venues.adapter import UnimplementedVenueAdapter

        class _Stub(UnimplementedVenueAdapter):
            def __init__(self):
                super().__init__(venue_id="stubz", venue_type=VenueType.PREDICTION,
                                 note="nothing")

        verdict = asyncio.run(_Stub().get_settlement("MKT"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is False


class TestPnlMath:
    @pytest.mark.parametrize("side,price,stake,outcome,expected", [
        ("YES", 0.60, 3.0, 1.0, 2.0),      # 5 shares pay 5, cost 3
        ("YES", 0.60, 3.0, 0.0, -3.0),
        ("NO", 0.60, 3.0, 0.0, 4.5),       # NO costs 0.40 -> 7.5 shares
        ("NO", 0.60, 3.0, 1.0, -3.0),
        ("YES", 0.50, 1.0, 1.0, 1.0),      # even money
        ("BUY", 0.25, 2.0, 1.0, 6.0),      # 8 shares
    ])
    def test_known_cases(self, side, price, stake, outcome, expected):
        assert compute_pnl(side, price, stake, outcome) == pytest.approx(expected)

    @pytest.mark.parametrize("side,price,stake,outcome", [
        ("YES", 0.60, 3.0, None),     # not settled
        ("YES", 0.60, 3.0, 0.5),      # ambiguous
        ("YES", 0.0, 3.0, 1.0),       # impossible price
        ("YES", 1.0, 3.0, 1.0),       # impossible price
        ("YES", 0.60, 0.0, 1.0),      # no stake
        ("YES", "bad", 3.0, 1.0),     # unparseable
    ])
    def test_refuses_to_invent_a_number(self, side, price, stake, outcome):
        assert compute_pnl(side, price, stake, outcome) is None


class TestPersistence:
    """
    Recording that does not survive the process is not recording.
    """

    def test_calibration_table_exists(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "s.db"))
        try:
            tables = {r[0] for r in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            storage.close()
        assert "calibration" in tables, (
            "record_forecast inserts into `calibration`; without the table every "
            "forecast failed into a warning"
        )

    def test_orders_table_exists(self, tmp_path):
        """
        OrderManager.create_order inserts into `orders`. It did not exist, so no
        order was ever persisted.
        """
        storage = Storage(db_path=str(tmp_path / "s.db"))
        try:
            tables = {r[0] for r in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            storage.close()
        assert "orders" in tables

    def test_forecasts_survive_a_restart(self, tmp_path):
        db = str(tmp_path / "r.db")
        storage = Storage(db_path=db)
        cal = CalibrationDB(db_path=str(tmp_path / "r.json"), storage=storage)
        _open_position(storage, cal)
        storage.close()

        storage2 = Storage(db_path=db)
        cal2 = CalibrationDB(db_path=str(tmp_path / "r.json"), storage=storage2)
        try:
            assert len(cal2.points) == 1, "the forecast did not survive"
            assert cal2.pending_forecasts()[0].market_id == "MK-1"
        finally:
            storage2.close()

    def test_venue_and_trade_link_survive_a_restart(self, tmp_path):
        """
        Settlement routes by venue, so the link has to persist.
        """
        db = str(tmp_path / "l.db")
        storage = Storage(db_path=db)
        cal = CalibrationDB(db_path=str(tmp_path / "l.json"), storage=storage)
        trade_id, forecast_id = _open_position(storage, cal)
        storage.close()

        storage2 = Storage(db_path=db)
        cal2 = CalibrationDB(db_path=str(tmp_path / "l.json"), storage=storage2)
        try:
            point = cal2.pending_forecasts()[0]
            assert point.venue_id == "testvenue"
            assert point.trade_id == trade_id
        finally:
            storage2.close()

    def test_points_are_not_double_counted_across_stores(self, tmp_path):
        """
        Both the SQL table and the JSON file can hold the same forecast. Loading
        both without dedupe would double-count every point in the Brier score.
        """
        db = str(tmp_path / "d.db")
        storage = Storage(db_path=db)
        cal = CalibrationDB(db_path=str(tmp_path / "d.json"), storage=storage)
        cal.record_forecast("MK-1", "Q?", 0.7, 0.8, 0.55, "sports")
        cal.save()
        storage.close()

        storage2 = Storage(db_path=db)
        cal2 = CalibrationDB(db_path=str(tmp_path / "d.json"), storage=storage2)
        try:
            assert len(cal2.points) == 1, (
                f"forecast loaded {len(cal2.points)} times from two stores"
            )
        finally:
            storage2.close()

    def test_market_price_survives_the_json_round_trip(self, tmp_path):
        """
        `save()` omitted market_price, so reload reset every entry price to the
        forecast probability.
        """
        storage = Storage(db_path=str(tmp_path / "mp.db"))
        cal = CalibrationDB(db_path=str(tmp_path / "mp.json"), storage=storage)
        cal.record_forecast("MK-1", "Q?", 0.70, 0.8, 0.55, "sports")
        cal.save()
        storage.close()

        storage2 = Storage(db_path=str(tmp_path / "mp.db"))
        cal2 = CalibrationDB(db_path=str(tmp_path / "mp.json"), storage=storage2)
        try:
            assert cal2.points[0].market_price == pytest.approx(0.55)
        finally:
            storage2.close()

    def test_a_failed_write_does_not_leave_a_phantom_forecast(self, tmp_path):
        """
        If persistence fails, the in-memory view must not keep a point the
        database does not have - the two would diverge silently.
        """
        storage = Storage(db_path=str(tmp_path / "f.db"))
        cal = CalibrationDB(db_path=str(tmp_path / "f.json"), storage=storage)

        # Drop the table behind the engine's back to force a write failure.
        storage.conn.execute("DROP TABLE calibration")
        storage.conn.commit()

        with pytest.raises(RuntimeError):
            cal.record_forecast("MK-1", "Q?", 0.7, 0.8, 0.55, "sports")

        assert not cal.has_forecast_for("MK-1"), (
            "the in-memory point survived a failed write"
        )
        storage.close()

    def test_resolve_trade_will_not_pay_twice(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "t2.db"))
        storage.set_bankroll(50.0)
        tid = storage.log_trade({"market_id": "M", "side": "YES",
                                 "market_price": 0.5, "position_size_usd": 2.0,
                                 "status": "open"})
        assert storage.resolve_trade(tid, 1.0, 2.0) is True
        assert storage.get_bankroll() == pytest.approx(52.0)
        assert storage.resolve_trade(tid, 1.0, 2.0) is False
        assert storage.get_bankroll() == pytest.approx(52.0), (
            "a repeated settlement paid out twice"
        )
        storage.close()

    def test_venue_id_column_migrates_onto_an_existing_database(self, tmp_path):
        """
        CREATE TABLE IF NOT EXISTS does nothing to an existing database, so an
        older data/ptai.db would keep its original columns and every insert
        naming venue_id would fail.
        """
        db = str(tmp_path / "old.db")
        storage = Storage(db_path=db)
        storage.conn.execute("ALTER TABLE trades DROP COLUMN venue_id")
        storage.conn.commit()
        storage.close()

        storage2 = Storage(db_path=db)
        try:
            cols = {r[1] for r in storage2.conn.execute(
                "PRAGMA table_info(trades)").fetchall()}
            assert "venue_id" in cols, "the migration did not run"
        finally:
            storage2.close()


class TestSettlementIsWiredIntoTheRuntime:
    def test_v3_owns_a_settlement_engine(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(dry_run=True)
        assert hasattr(agent, "settlement_engine")
        assert agent.settlement_engine.calibration_engine is agent.calibration_engine
        assert agent.settlement_engine.storage is agent.storage

    def test_run_cycle_settles_before_sizing(self):
        """
        Settlement moves the bankroll, and the bankroll sizes the next trade, so
        settling after discovery would size against a stale figure.
        """
        import inspect
        from src.ptai.agent import v3_loop

        source = inspect.getsource(v3_loop.TradingAgentV3.run_cycle)
        settle_at = source.index("settle_pending()")
        qualify_at = source.index("evaluate_all_venues")
        assert settle_at < qualify_at, (
            "settlement must run before the qualification and sizing work"
        )

    def test_cycle_result_reports_settlement(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(dry_run=True)

        async def _empty(*args, **kwargs):
            return {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(agent, "discover_all_venues", _empty)
            result = asyncio.run(agent.run_cycle())

        assert "settlement" in result, (
            "the cycle result must report what settled, or the operator cannot "
            "tell whether the learning chain is running"
        )
