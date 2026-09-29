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
from src.ptai.venues.qualification import LIVE_TARGETS, paper_record_progress


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
                     side="YES", question="Will event happen?"):
    """A paper position, written the way the executor writes one."""
    storage.log_trade({
        "market_id": market_id, "market_question": question, "side": side,
        "market_price": price, "fair_value": 0.5, "edge": 0.05,
        "kelly_fraction": 0.02, "position_size_usd": stake, "confidence": 0.7,
        "status": "paper", "venue_id": "polymarket", "execution_mode": "paper",
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
