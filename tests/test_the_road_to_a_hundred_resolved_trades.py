"""
The road to 100 resolved trades: the number that unlocks live capital.

The operator's ask was exact:

    "i need to get Polymarket paper record to 100 resolved trades so live
     unlocks"

Four things have to be true for that number to move, and each is tested here:

1. A paper trade has to be ABLE to settle. The settlement engine reads the
   venue's own answer for the market and closes the position - paper rows
   included - and the count of resolved paper trades moves with it.
2. The lane has to be able to place MORE THAN ONE trade per cycle. One was the
   ceiling while the lane only had to prove it could trade at all; the record
   needs volume, so `PTAI_PAPER_TRADES_PER_CYCLE` (default 3, ceiling 5) governs
   it and every other part of the lane's bar is unchanged.
3. The lane has to PREFER markets that settle SOON. A trade that resolves in six
   hours teaches the record the same day; one that resolves in three months
   teaches it next quarter.
4. The count alone is not the gate. Win rate, Brier and profit factor can each
   refuse a record that has the sample size, so the progress panel shows all of
   them and the note names the one that is actually binding.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.ptai.execution.settlement import SettlementEngine
from src.ptai.storage.db import Storage
from src.ptai.strategy.strategy_engine import (
    PAPER_TRADES_PER_CYCLE_CEILING,
    PAPER_TRADES_PER_CYCLE_DEFAULT,
    paper_trades_per_cycle_default,
    resolution_days,
)
from src.ptai.markets.base import Market, MarketSource
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueType,
)
from src.ptai.execution.position_ledger import PositionLedgerBuilder
from src.ptai.venues.qualification import (
    LIVE_TARGETS,
    paper_record_progress,
    paper_slot_report,
)


class _SettlingVenue(MarketAdapter):
    """A venue that can report a resolution, once the market is closed."""

    def __init__(self, venue_id="polymarket", outcome=1.0, settled=True):
        super().__init__(venue_id=venue_id, venue_type=VenueType.PREDICTION)
        self.capabilities = AdapterCapability(
            implementation_status="live", supports_market_discovery=True)
        self.label = venue_id.title()
        self._outcome = outcome
        self._settled = settled

    def check_eligibility(self, country_code="UG"):
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count=100, **kwargs):
        return []

    async def get_orderbook(self, market):
        return {}

    async def get_portfolio(self):
        return {"venue_id": self.venue_id, "available": True}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        return {"status": "dry_run"}

    async def get_settlement(self, market_id):
        if not self._settled:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "gamma_markets", "reason": "market is not closed yet"}
        return {"settled": True, "outcome": self._outcome, "is_real": True,
                "source": "gamma_markets", "reason": "market is closed"}


class _Registry:
    def __init__(self, adapter):
        self.adapters = {adapter.venue_id: adapter}

    def get_adapter_for_venue_id(self, venue_id):
        return self.adapters.get(venue_id)


def _log_paper_trade(storage, market_id="pm1", stake=1.0, price=0.42,
                     side="YES", question="Will event happen?",
                     venue_id="polymarket"):
    """A paper position, written the way the executor writes one."""
    storage.log_trade({
        "market_id": market_id, "market_question": question, "side": side,
        "market_price": price, "fair_value": 0.5, "edge": 0.05,
        "kelly_fraction": 0.02, "position_size_usd": stake, "confidence": 0.7,
        "status": "paper", "venue_id": venue_id, "execution_mode": "paper",
        "token_price_at_entry": price, "fees_usd": 0.0,
    })
    row = storage.conn.execute(
        "SELECT id FROM trades WHERE market_id = ? ORDER BY id DESC LIMIT 1",
        (market_id,)).fetchone()
    return int(row["id"])


# ---------------------------------------------------------------------------
# 1. a paper trade settles, and the record counts it
# ---------------------------------------------------------------------------

class TestAPaperTradeSettlesAndCounts:
    def test_settlement_closes_a_paper_position_and_stamps_the_time(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "paper.db"))
        try:
            trade_id = _log_paper_trade(storage)
            before = paper_record_progress(storage, "polymarket")
            assert before["resolved"] == 0
            assert before["open_trades"] == 1

            engine = SettlementEngine(
                storage=storage, venue_registry=_Registry(_SettlingVenue()))
            report = asyncio.run(engine.settle_pending())

            assert report.paper_settled == 1
            assert report.paper_pnl_usd > 0, "a winning YES at 0.42 pays out"
            row = storage.conn.execute(
                "SELECT resolved, resolved_at, pnl FROM trades WHERE id = ?",
                (trade_id,)).fetchone()
            assert row["resolved"] == 1
            assert row["resolved_at"], (
                "without a resolution time the RATE of the record cannot be "
                "measured, only its level")

            after = paper_record_progress(storage, "polymarket")
            assert after["resolved"] == 1
            assert after["wins"] == 1
            assert after["open_trades"] == 0
            assert after["remaining"] == LIVE_TARGETS["min_trades"] - 1
            assert after["rate_per_day"] is not None
            assert after["median_hold_hours"] is not None
        finally:
            storage.close()

    def test_an_unresolved_market_leaves_the_position_open(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "open.db"))
        try:
            _log_paper_trade(storage)
            engine = SettlementEngine(
                storage=storage,
                venue_registry=_Registry(_SettlingVenue(settled=False)))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 0
            assert paper_record_progress(storage, "polymarket")["resolved"] == 0
        finally:
            storage.close()

    def test_the_count_alone_does_not_unlock_live(self, tmp_path):
        """
        A hundred resolved trades that lose money must not be described as ready.
        The gate is four numbers, not one.
        """
        storage = Storage(db_path=str(tmp_path / "bad.db"))
        try:
            now = datetime.now(timezone.utc).isoformat()
            for i in range(LIVE_TARGETS["min_trades"]):
                storage.conn.execute(
                    "INSERT INTO trades (timestamp, market_id, side, market_price,"
                    " position_size_usd, resolved, pnl, status, venue_id,"
                    " execution_mode, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (now, f"pm{i}", "YES", 0.5, 1.0, 1, -1.0, "settled",
                     "polymarket", "paper", now))
            storage.conn.commit()
            progress = paper_record_progress(storage, "polymarket")
            assert progress["resolved"] == LIVE_TARGETS["min_trades"]
            assert progress["gates"]["resolved_trades"]["pass"] is True
            assert progress["ready"] is False
            assert progress["gates"]["win_rate"]["pass"] is False
            assert "win rate" in progress["note"]
        finally:
            storage.close()

    def test_a_good_record_at_a_hundred_unlocks(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "good.db"))
        try:
            now = datetime.now(timezone.utc).isoformat()
            for i in range(LIVE_TARGETS["min_trades"]):
                storage.conn.execute(
                    "INSERT INTO trades (timestamp, market_id, side, market_price,"
                    " position_size_usd, resolved, pnl, status, venue_id,"
                    " execution_mode, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (now, f"pm{i}", "YES", 0.5, 1.0, 1,
                     0.5 if i % 10 else -1.0, "settled", "polymarket", "paper", now))
                storage.conn.execute(
                    "INSERT INTO trade_outcomes (trade_id, market_id, venue_id,"
                    " actual_outcome, pnl, brier_score, was_correct, recorded_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (f"t{i}", f"pm{i}", "polymarket", 1.0, 0.5, 0.09, 1, now))
            storage.conn.commit()
            progress = paper_record_progress(storage, "polymarket")
            assert progress["ready"] is True
            assert progress["win_rate"] == 0.9
            assert progress["profit_factor"] > 1.1
            assert progress["brier"] == 0.09
            assert "unlocked" in progress["note"]
        finally:
            storage.close()

    def test_the_targets_match_the_gate_that_enforces_them(self):
        """The panel cannot promise a bar the qualification engine does not use."""
        from src.ptai.venues.qualification import VenueQualificationEngine

        engine = VenueQualificationEngine(data_dir="./data")
        assert engine.requirements["min_trades"] == LIVE_TARGETS["min_trades"]
        assert engine.requirements["min_win_rate"] == LIVE_TARGETS["min_win_rate"]
        assert engine.requirements["max_brier"] == LIVE_TARGETS["max_brier"]
        assert (engine.requirements["min_profit_factor"]
                == LIVE_TARGETS["min_profit_factor"])


# ---------------------------------------------------------------------------
# 2. more than one paper trade per cycle
# ---------------------------------------------------------------------------

class TestTheLaneCanBuildTheRecord:
    def test_the_default_is_three_and_the_setting_is_bounded(self, monkeypatch):
        monkeypatch.delenv("PTAI_PAPER_TRADES_PER_CYCLE", raising=False)
        assert PAPER_TRADES_PER_CYCLE_DEFAULT == 3
        assert paper_trades_per_cycle_default() == 3
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "5")
        assert paper_trades_per_cycle_default() == 5
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "99")
        assert paper_trades_per_cycle_default() == PAPER_TRADES_PER_CYCLE_CEILING
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "lots")
        assert paper_trades_per_cycle_default() == PAPER_TRADES_PER_CYCLE_DEFAULT

    def test_the_lane_takes_the_configured_number_soonest_first(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_PAPER_TRADES_PER_CYCLE", "2")

        class _Opp:
            def __init__(self, market_id, days, score):
                self.market = type("M", (), {
                    "id": market_id,
                    "end_date": (None if days is None else
                                 datetime.now(timezone.utc) + timedelta(days=days)),
                })()
                self.score = score

        now = datetime.now(timezone.utc)
        pool = [_Opp("far", 90, 0.99), _Opp("soon", 0.2, 0.60),
                _Opp("unknown", None, 0.95)]
        # The ordering the cycle uses, reproduced here so the test fails if the
        # preference is removed rather than if a score moves.
        def urgency(opp):
            days = resolution_days(opp.market)
            bucket = 0 if days is None else (3 if days <= 1 else
                                             (2 if days <= 7 else 1))
            return (bucket, -(days if days is not None else 1e9), opp.score)

        ordered = sorted(pool, key=urgency, reverse=True)[:2]
        assert [o.market.id for o in ordered] == ["soon", "far"], (
            "a market that settles in hours must be chosen before one that "
            "settles in three months")
        # ...and an undated market is not treated as if it settled tomorrow.
        assert resolution_days(pool[2].market) is None

    def test_the_screen_gives_a_bounded_bonus_for_a_near_date(self, tmp_path,
                                                              monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
        monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.markets.base import Market, MarketSource

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        try:
            book = {"validated": True, "spread": 0.02, "depth": 5000.0,
                    "bids": [[0.41, 1000]], "asks": [[0.43, 1000]], "is_real": True}

            def market(days):
                return Market(
                    id="pm1", question="Will it happen?",
                    source=MarketSource.POLYMARKET, venue_id="polymarket",
                    liquidity=15000.0, volume_24h=15000.0,
                    active=True, closed=False,
                    end_date=(None if days is None else
                              datetime.now(timezone.utc) + timedelta(days=days)))

            soon, why_soon = agent._screen_score(market(0.2), book)
            far, _ = agent._screen_score(market(90), book)
            undated, _ = agent._screen_score(market(None), book)
            assert soon > far, "the same book and volume, sooner, must win the slot"
            assert undated == far, "no end date is not a reason to rank above"
            assert "resolves in" in why_soon
            # Bounded: the bonus can never outweigh a missing book.
            bookless = dict(book)
            bookless["validated"] = False
            refused, _ = agent._screen_score(market(0.2), bookless)
            assert refused == -1.0
        finally:
            agent.storage.close()


# ---------------------------------------------------------------------------
# 3. the panel answers the operator's question
# ---------------------------------------------------------------------------

@pytest.fixture()
def console_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    return importlib.import_module("src.ptai.ui.console")


@pytest.fixture()
def console_client(console_app):
    return TestClient(console_app.app)


class TestThePanelAnswersHowFarAlongItIs:
    def test_the_payload_has_the_record_and_its_gates(self, console_app,
                                                      console_client):
        body = console_client.get("/api/console/venue").json()
        pr = body["paper_record"]
        assert pr["venue_id"] == "polymarket"
        assert (pr["gates"]["resolved_trades"]["need"]
                == LIVE_TARGETS["min_trades"])
        for gate in ("resolved_trades", "win_rate", "brier", "profit_factor"):
            assert gate in pr["gates"], f"{gate} must be shown, not just the count"
        assert pr["note"]

    def test_a_settled_trade_moves_the_console_number(self, console_app,
                                                      console_client, tmp_path):
        storage = Storage(db_path=str(tmp_path / "console.db"))
        try:
            _log_paper_trade(storage)
            engine = SettlementEngine(
                storage=storage, venue_registry=_Registry(_SettlingVenue()))
            asyncio.run(engine.settle_pending())
        finally:
            storage.close()
        body = console_client.get("/api/console/venue").json()
        assert body["paper_record"]["resolved"] == 1
        assert "1 of 100" in body["paper_record"]["note"]


# ---------------------------------------------------------------------------
# 4. the SLOTS - the thing that was stopping the count
# ---------------------------------------------------------------------------
#
# The record stopped moving because of SLOTS, not signals. `ExposureManager`
# carries the position-count cap for REAL capital (6), and the round read that
# same constant for its free-slot arithmetic - so once six paper positions were
# waiting to settle, every later round opened nothing while the record those
# positions exist to build stood still. A paper account has its own budget, and
# the two things that must never happen are covered below: the budget must not
# leak into live capital, and a slot must never be handed to a venue that cannot
# report a resolution (that position would hold it for good).


class _VenueWithNoOutcome(MarketAdapter):
    """Reads markets, fills in paper - and cannot say how any of them ended."""

    def __init__(self, venue_id="predictit"):
        super().__init__(venue_id=venue_id, venue_type=VenueType.PREDICTION)
        self.capabilities = AdapterCapability(
            implementation_status="live", supports_market_discovery=True)
        self.label = venue_id.title()

    def check_eligibility(self, country_code="UG"):
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count=100, **kwargs):
        return []

    async def get_orderbook(self, market):
        return {}

    async def get_portfolio(self):
        return {"venue_id": self.venue_id, "available": True}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        return {"status": "dry_run"}


def _agent(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
    monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
    from src.ptai.agent.v3_loop import TradingAgentV3

    return TradingAgentV3(country_code="UG", dry_run=True)


def _ledger(storage):
    return PositionLedgerBuilder(storage=storage).build()


class TestThePaperAccountHasItsOwnSlots:
    def test_the_paper_budget_is_not_the_live_risk_limit(self, monkeypatch,
                                                         tmp_path):
        """
        Six is the risk limit for real capital. A paper account risking nothing
        is not bound by it - and the percentage ceilings (6% a position, 15% a
        category, 20% correlated, 50% total) are unchanged and still bind.
        """
        agent = _agent(monkeypatch, tmp_path)
        try:
            budget = agent.paper_slot_budget()
            assert budget["max"] == 25, (
                "the 50% ceiling expressed in $1 slices of the $50 paper account")
            assert budget["risk_limit"] == int(
                agent.limits_engine.limits.max_open_positions)
            assert budget["max"] > budget["risk_limit"]
            assert "paper" in budget["source"]
            assert "percentage ceilings are unchanged" in budget["note"]

            # The budget is a setting, not a magic number.
            monkeypatch.setattr(agent.settings, "paper_max_open_positions", 3,
                                raising=False)
            assert agent.paper_slot_budget()["max"] == 3

            # ...and it can never loosen REAL capital: in live mode the risk
            # limit governs, whatever the paper setting says.
            agent.dry_run = False
            live = agent.paper_slot_budget()
            assert live["max"] == live["risk_limit"] == int(
                agent.limits_engine.limits.max_open_positions)
            assert "live" in live["source"]
        finally:
            agent.storage.close()

    def test_seven_open_positions_no_longer_fill_the_book(self, monkeypatch,
                                                          tmp_path):
        """
        The regression itself. Seven paper positions is 14% of a $50 paper
        account - three of the ceilings would still allow it - but under the old
        cap of six, `can_open` refused the seventh because the book was "full".
        """
        agent = _agent(monkeypatch, tmp_path)
        try:
            for i in range(7):
                _log_paper_trade(agent.storage, market_id=f"SLOT-{i}", stake=1.0,
                                 price=0.5)
            seed = agent._seed_exposure_manager(_ledger(agent.storage))

            assert seed["positions"] == 7
            assert seed["max_open_positions"] == 25
            assert seed["risk_limit"] == 6, "the live limit is still reported"
            assert seed["free_slots"] == 18
            assert "paper" in seed["slot_source"]
            assert seed["slot_note"]
            assert agent.exposure_manager.max_open_positions == 25

            allowed, reason = agent.exposure_manager.can_open(
                market_id="SLOT-NEW", amount_usd=1.0, category="politics",
                correlation_group="politics", venue="polymarket")
            assert allowed is True, (
                f"the seventh position was refused as a full book: {reason}")
        finally:
            agent.storage.close()

    def test_a_full_budget_is_still_a_full_book(self, monkeypatch, tmp_path):
        """Widening the count for paper is not the same as removing it."""
        agent = _agent(monkeypatch, tmp_path)
        try:
            for i in range(25):
                _log_paper_trade(agent.storage, market_id=f"FULL-{i}", stake=1.0,
                                 price=0.5)
            seed = agent._seed_exposure_manager(_ledger(agent.storage))
            assert seed["free_slots"] == 0
            allowed, reason = agent.exposure_manager.can_open(
                market_id="ONE-MORE", amount_usd=1.0, category="politics",
                correlation_group="politics", venue="polymarket")
            assert allowed is False and "Max open positions" in reason
        finally:
            agent.storage.close()


# ---------------------------------------------------------------------------
# 5. a slot is never handed to a venue that cannot close the position
# ---------------------------------------------------------------------------

class TestTheTwoPanelsCountTheSameThing:
    def test_an_open_live_position_is_not_an_open_paper_trade(self, tmp_path):
        """
        `paper_record_progress` counted every unresolved row for the venue -
        live mode included - while `paper_slot_report`, printed beside it on the
        same page, has always filtered by mode. The same account could therefore
        report a paper trade still waiting to settle next to a slot line that
        said nothing was in use.
        """
        storage = Storage(db_path=str(tmp_path / "modes.db"))
        try:
            _log_paper_trade(storage, market_id="paper-1")
            now = datetime.now(timezone.utc).isoformat()
            storage.conn.execute(
                "INSERT INTO trades (timestamp, market_id, side, market_price,"
                " position_size_usd, resolved, status, venue_id, execution_mode)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (now, "live-1", "YES", 0.5, 3.0, 0, "open", "polymarket", "live"))
            storage.conn.commit()

            progress = paper_record_progress(storage, "polymarket")
            slots = paper_slot_report(storage, 25)
            assert progress["open_trades"] == 1, (
                "the live row is not a paper trade waiting to settle")
            assert progress["open_trades"] == slots["open"]
        finally:
            storage.close()


class TestAVenueThatCannotCloseIsRefusedBeforeAnythingOpens:
    def test_settlement_capability_is_a_fact_about_the_adapter(self, monkeypatch,
                                                               tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            agent.venue_registry = _Registry(_VenueWithNoOutcome("predictit"))
            ok, why = agent.settlement_capable("predictit")
            assert ok is False
            assert "cannot report a settlement" in why
            assert "never be closed or learned from" in why

            agent.venue_registry = _Registry(_SettlingVenue("polymarket"))
            assert agent.settlement_capable("polymarket")[0] is True

            # An empty or unknown venue is refused, never assumed settleable.
            assert agent.settlement_capable("")[0] is False
            ok, why = agent.settlement_capable("nowhere")
            assert ok is False and "no adapter is registered" in why
        finally:
            agent.storage.close()

    def test_a_round_refuses_the_position_and_still_runs(self, tmp_path,
                                                         monkeypatch):
        """
        The whole round on a venue whose adapter cannot report a resolution: the
        opportunity reaches the decision, the gate refuses it, nothing is
        written, and the round still researches and reports.
        """
        from tests.test_full_cycle_from_discovery_to_allocation import (
            VENUE,
            _force_qualified,
            _inject_opportunity,
            build_agent,
        )

        agent, adapter = build_agent(tmp_path, dry_run=True)
        try:
            _force_qualified(agent, monkeypatch)
            _inject_opportunity(agent, adapter._market(), monkeypatch)
            # The stub CAN settle - which is what makes the refusal below about
            # the gate rather than about the fixture. Removing the override is
            # exactly the state PredictIt's adapter is in.
            assert agent.settlement_capable(VENUE)[0] is True
            monkeypatch.setattr(type(adapter), "get_settlement",
                                MarketAdapter.get_settlement)

            result = asyncio.run(agent.run_round(target_per_venue=5))

            refusals = result["settlement_refusals"]
            assert refusals, "the opportunity never reached the gate"
            assert refusals[0]["venue_id"] == VENUE
            assert "cannot report a settlement" in refusals[0]["reason"]
            assert result["execution"] == [], (
                "a refused opportunity must not be executed")
            assert result["round"]["positions_opened"] == 0
            assert agent.storage.get_performance_summary()["total_trades"] == 0
            assert result["position_slots"]["max_open_positions"] > 0
        finally:
            agent.storage.close()


class TestAPositionThatCanNeverCloseIsNamedNotCounted:
    def test_the_settlement_report_carries_the_stuck_count(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "stuck.db"))
        try:
            _log_paper_trade(storage, market_id="pi1", venue_id="predictit")
            engine = SettlementEngine(
                storage=storage,
                venue_registry=_Registry(_VenueWithNoOutcome("predictit")))
            report = asyncio.run(engine.settle_pending())

            assert report.stuck == 1
            assert report.to_dict()["stuck"] == 1
            assert report.items and report.items[0].source == "unsupported"
            # Recorded in storage, so the panel can read it in another process.
            assert json.loads(
                storage.get_state("settlement_unsupported_venues")) == ["predictit"]
            # No invented outcome, and no resolved trade: the position stays open.
            paper = storage.get_paper_performance()
            assert paper["settled_trades"] == 0, (
                "an unsupported venue must never be resolved")
            assert storage.get_open_positions(), "the position must still be open"
        finally:
            storage.close()

    def test_a_settleable_venue_is_never_recorded_as_stuck(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "fine.db"))
        try:
            _log_paper_trade(storage, market_id="pm1", venue_id="polymarket")
            engine = SettlementEngine(
                storage=storage, venue_registry=_Registry(_SettlingVenue()))
            report = asyncio.run(engine.settle_pending())
            assert report.stuck == 0
            assert report.to_dict()["stuck"] == 0
            assert storage.get_state("settlement_unsupported_venues") in (None, "")
        finally:
            storage.close()

    def test_the_slot_report_names_which_slots_are_stuck(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "slots.db"))
        try:
            _log_paper_trade(storage, market_id="pi1", venue_id="predictit")
            _log_paper_trade(storage, market_id="pi2", venue_id="predictit")
            _log_paper_trade(storage, market_id="pm1", venue_id="polymarket")
            storage.set_state("settlement_unsupported_venues", '["predictit"]')

            slots = paper_slot_report(storage, 25)
            assert slots["open"] == 3
            assert slots["free"] == 22
            assert slots["full"] is False
            assert slots["by_venue"] == {"predictit": 2, "polymarket": 1}
            assert slots["stuck_open"] == 2
            assert slots["stuck_venues"] == [{"venue_id": "predictit", "open": 2}]
            assert "cannot report a resolution" in slots["note"]
            assert "predictit (2)" in slots["note"]
        finally:
            storage.close()

    def test_a_full_book_says_the_record_grows_when_they_settle(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "full.db"))
        try:
            for i in range(25):
                _log_paper_trade(storage, market_id=f"pm{i}", venue_id="polymarket")
            slots = paper_slot_report(storage, 25)
            assert slots["full"] is True and slots["free"] == 0
            assert "every slot is held" in slots["note"]
        finally:
            storage.close()


class TestThePanelSaysWhyTheCountWouldNotMove:
    def test_the_venue_payload_carries_the_position_slots(self, console_app,
                                                          console_client):
        body = console_client.get("/api/console/venue").json()
        slots = body["paper_slots"]
        assert slots["max"] == 25
        assert slots["open"] == 0 and slots["free"] == 25
        assert slots["full"] is False
        assert "0 of 25 paper position slot(s) in use" in slots["note"]

    def test_a_stuck_slot_is_named_on_the_panel(self, console_app,
                                                console_client, tmp_path):
        """
        A count that is not moving has two very different explanations - a full
        book, and a slot that can never free itself. Both are on the page.
        """
        storage = Storage(db_path=str(tmp_path / "console.db"))
        try:
            _log_paper_trade(storage, market_id="pi1", venue_id="predictit")
            storage.set_state("settlement_unsupported_venues", '["predictit"]')
        finally:
            storage.close()

        body = console_client.get("/api/console/venue").json()
        slots = body["paper_slots"]
        assert slots["stuck_open"] == 1
        assert slots["free"] == 24
        assert "never close" in slots["note"] and "predictit (1)" in slots["note"]
        # and the panel renders it rather than leaving it in the payload.
        html = console_client.get("/").text
        assert "Position slots." in html
        assert "paper_slots" in html
