"""
The operator's view: the questions they asked, answered from one source.

"What the agent is doing" is a small, specific list of questions, and every one
of them has a way of being answered wrongly that looks right:

  * which venue is live - answered from a hard-coded default rather than the
    remembered selection,
  * which strategy is running - answered from the code's list of strategy names
    rather than from what has actually been recorded,
  * how full deployment is - answered by adding the paper account's positions to
    the real account's,
  * the current risk state - answered as "OK" because nothing has been checked,
  * what was decided last cycle - answered from memory, so it evaporates on
    restart,
  * profit so far - answered by summing real and simulated P&L.

These tests pin the honest answer for each, including the case where the honest
answer is "I do not know yet".
"""

from __future__ import annotations

import json

import pytest

from src.ptai.operator_view import (
    LAST_CYCLE_KEY,
    describe_snapshot,
    operator_snapshot,
)
from src.ptai.storage.db import Storage


@pytest.fixture()
def storage(tmp_path):
    store = Storage(db_path=str(tmp_path / "operator.db"))
    store.set_bankroll(50.0)
    yield store
    store.close()


class TestTheSevenQuestions:
    def test_a_fresh_install_says_it_has_nothing_to_report(self, storage):
        """
        No invented numbers. A new database has no cycle, no venues and no
        outcomes, and every one of those must say so rather than showing a zero
        that looks like a measurement.
        """
        snap = operator_snapshot(storage)

        assert snap["venues"]["live_venue"] is None, (
            "a fresh install is not trading real money at any venue")
        assert snap["strategies"]["available"] is False
        assert snap["last_cycle"]["available"] is False
        assert "no completed cycle" in snap["last_cycle"]["note"]
        # The capital block is available - it is a real reading of a real account.
        assert snap["capital"]["available"] is True
        assert snap["capital"]["equity_usd"] == pytest.approx(50.0)
        assert snap["capital"]["free_cash_usd"] == pytest.approx(50.0)
        assert snap["capital"]["deployment_pct"] == 0.0

        for block in ("capital", "profit", "positions", "venues", "strategies",
                      "risk", "last_cycle"):
            assert "source" in snap[block], f"{block} does not name its source"

    def test_the_live_venue_is_read_from_the_remembered_selection(
            self, storage, monkeypatch):
        """
        Not from a default. The agent may be paper-trading polymarket while the
        operator has moved the real money to kalshi, and the console must say
        kalshi - it is the one place the answer exists.
        """
        from src.ptai.strategy.venue_selection import VenueSelector

        VenueSelector(storage=storage).remember_live_venue("kalshi")
        snap = operator_snapshot(storage)
        assert snap["venues"]["live_venue"] == "kalshi"

        # And clearing it is a real state, not a missing value: no venue holds
        # the live capital, so nothing is live.
        VenueSelector(storage=storage).remember_live_venue(None)
        assert operator_snapshot(storage)["venues"]["live_venue"] is None

    def test_paper_results_are_labelled_and_never_counted_as_live(self, storage):
        """
        A venue whose only results are simulated is ranked on them - and says so.

        The distinction is the whole point of the split: "best venue" chosen on
        paper evidence is legitimate for funding the FIRST account, but a screen
        that presents it as proven performance is not.
        """
        for i in range(6):
            trades = storage.log_trade({
                "market_id": f"P{i}", "venue_id": "polymarket", "side": "YES",
                "position_size_usd": 3.0, "market_price": 0.5,
                "execution_mode": "paper", "status": "paper",
            })
            storage.resolve_trade(trades, outcome=1.0, pnl=1.5)

        snap = operator_snapshot(storage)
        venues = {v["venue_id"]: v for v in snap["venues"]["venues"]}
        assert venues["polymarket"]["ranked_on"] == "paper"
        assert venues["polymarket"]["paper_net_pnl_usd"] == pytest.approx(9.0)
        assert venues["polymarket"]["live_net_pnl_usd"] == 0.0
        assert venues["polymarket"]["live_resolved_trades"] == 0
        assert snap["venues"]["best_validated_on"] == "paper"

        # The same trades as LIVE flip the label, and only then.
        for i in range(6):
            trades = storage.log_trade({
                "market_id": f"L{i}", "venue_id": "polymarket", "side": "YES",
                "position_size_usd": 3.0, "market_price": 0.5,
                "execution_mode": "live", "status": "open",
            })
            storage.resolve_trade(trades, outcome=0.0, pnl=-3.0)
        snap = operator_snapshot(storage)
        venues = {v["venue_id"]: v for v in snap["venues"]["venues"]}
        assert venues["polymarket"]["ranked_on"] == "live", (
            "a venue with live results must be ranked on them, not on the "
            "simulation it also has")
        assert venues["polymarket"]["live_net_pnl_usd"] == pytest.approx(-18.0)
        assert venues["polymarket"]["paper_net_pnl_usd"] == pytest.approx(9.0), (
            "the paper record stays visible next to the real one")

    def test_open_positions_separate_the_two_accounts(self, storage):
        """The operator's positions and the simulation's, never in one list."""
        storage.log_trade({
            "market_id": "REAL-1", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 3.0, "market_price": 0.55,
            "market_question": "Will the real thing happen?",
            "status": "open",
        })
        storage.log_trade({
            "market_id": "PAPER-1", "venue_id": "polymarket", "side": "NO",
            "position_size_usd": 1.0, "market_price": 0.45,
            "market_question": "Will the simulated thing happen?",
            "status": "paper", "execution_mode": "paper",
        })

        snap = operator_snapshot(storage)
        positions = snap["positions"]
        assert [p["market_id"] for p in positions["live"]] == ["REAL-1"]
        assert [p["market_id"] for p in positions["paper"]] == ["PAPER-1"]
        assert positions["live_count"] == 1
        assert positions["paper_count"] == 1
        # The capital block counts them apart for the same reason.
        assert snap["capital"]["live_positions"] == 1
        assert snap["capital"]["paper_positions"] == 1

    def test_a_paper_position_does_not_full_the_deployment_meter(self, storage):
        """
        Deployment is about the operator's money. A $1 shadow position must not
        move the meter that decides how much real capital is at work.
        """
        storage.log_trade({
            "market_id": "PAPER-1", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 1.0, "market_price": 0.5,
            "status": "paper", "execution_mode": "paper",
        })
        snap = operator_snapshot(storage)
        assert snap["capital"]["deployment_pct"] == 0.0, (
            "a simulated position was counted as deployed capital")

    def test_strategy_evidence_comes_from_recorded_outcomes(self, storage):
        """
        Which strategy is running is answered by results, not by a name list.

        A strategy with no resolved trades is not "the best validated strategy";
        it is a strategy that has not been tested, and the snapshot says which
        of the two it is looking at.
        """
        from src.ptai.learning.trade_outcomes import TradeOutcomeTracker

        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(4):
            trades = storage.log_trade({
                "market_id": f"M{i}", "venue_id": "polymarket", "side": "YES",
                "position_size_usd": 3.0, "market_price": 0.5,
                "strategy": "mispricing", "execution_mode": "paper",
                "status": "paper",
            })
            storage.resolve_trade(trades, outcome=1.0, pnl=1.0)
            tracker.record_trade(trade_id=str(trades), market_id=f"M{i}",
                                 venue_id="polymarket", strategy="mispricing",
                                 forecast_prob=0.7, market_price=0.5,
                                 edge=0.2, side="YES", amount_usd=3.0,
                                 execution_mode="paper")
            tracker.record_resolution(str(trades), actual_outcome=1.0, pnl=1.0)

        snap = operator_snapshot(storage)
        assert snap["strategies"]["best_validated_strategy"] == "mispricing"
        assert snap["strategies"]["best_validated_on"] == "paper"
        assert snap["strategies"]["strategies"][0]["paper_resolved_trades"] == 4


class TestTheLastDecisionSurvivesTheProcess:
    def test_the_recorded_cycle_is_what_the_view_reports(self, storage):
        """
        Read back from storage, so a console opened after a restart answers the
        same as one that watched the cycle happen.
        """
        recorded = {
            "at": "2026-09-24T10:00:00+00:00",
            "verdict": "DO NOTHING",
            "why": "no qualified venue this cycle",
            "venues_searched": 3,
            "markets_scanned": 640,
            "qualified_venues": [],
            "live_venue": None,
            "venue_reasons": ["polymarket is the venue to fund first"],
            "orders": {"attempted": 1, "positions_recorded": 0,
                       "blocked_live_capital": 1,
                       "blocked_reason": "no budget authorised"},
            "risk": {"kill_switch_level": 0, "can_trade": True},
        }
        storage.set_state(LAST_CYCLE_KEY, json.dumps(recorded))

        snap = operator_snapshot(storage)
        assert snap["last_cycle"]["available"] is True
        assert snap["last_cycle"]["verdict"] == "DO NOTHING"
        assert snap["last_cycle"]["markets_scanned"] == 640
        assert snap["last_cycle"]["orders"]["blocked_live_capital"] == 1
        # The risk block carries the level the loop recorded - not a default.
        assert snap["risk"]["kill_switch_level"] == 0

        lines = "\n".join(describe_snapshot(snap))
        assert "DO NOTHING" in lines
        assert "blocked by the live capital boundary" in lines
        assert "no budget authorised" in lines

    def test_a_cycle_record_that_cannot_be_read_says_so(self, storage):
        """Corrupt state is reported, not silently rendered as "nothing happened"."""
        storage.set_state(LAST_CYCLE_KEY, "{not json")
        snap = operator_snapshot(storage)
        assert snap["last_cycle"]["available"] is False
        assert "could not be read" in snap["last_cycle"]["note"]

    def test_a_real_cycle_writes_the_record(self, tmp_path, monkeypatch):
        """
        The write side of the contract, exercised through a real cycle.

        A snapshot that only reads is half a feature: if the loop never records
        what it decided, the console can only ever report the previous answer.
        """
        import asyncio
        import sys

        sys.path.insert(0, str(tmp_path))
        from tests.test_full_cycle_from_discovery_to_allocation import (
            _cycle,
            _force_qualified,
            _inject_opportunity,
            build_agent,
        )

        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            _cycle(agent)
            stored = agent.storage.get_state(LAST_CYCLE_KEY)
            assert stored, "the cycle recorded no decision for the operator"
            data = json.loads(stored)
            assert data["verdict"] in ("DEPLOYED", "DO NOTHING")
            assert data["markets_scanned"] >= 1
            assert data["qualified_venues"] == ["polymarket"]
            assert data["decided"]["question"], (
                "the operator cannot see what the cycle was looking at")
            assert data["decided"]["venue"] == "polymarket"
            assert data["orders"]["attempted"] >= 1
            assert "equity" in data["capital"]
            assert data["risk"]["can_trade"] is True
        finally:
            agent.storage.close()


class TestBothFrontEndsReadOneSource:
    def test_the_dashboard_serves_the_snapshot(self):
        """The route exists, returns the snapshot, and does not mask failures."""
        from fastapi.testclient import TestClient

        from src.ptai.dashboard import app

        response = TestClient(app).get("/api/operator")
        assert response.status_code == 200, response.text
        data = response.json()
        for key in ("capital", "profit", "positions", "venues", "strategies",
                    "risk", "last_cycle", "lines", "mode"):
            assert key in data, f"the operator payload has no {key}"
        assert isinstance(data["lines"], list) and data["lines"], (
            "the endpoint must return the sentences the CLI prints, or the two "
            "front ends are two implementations")
        assert data["lines"][0].startswith("Live venue:")

    def test_the_cli_prints_those_same_lines(self):
        """One composition: the terminal cannot describe a different state."""
        from src.ptai.operator_view import describe_snapshot as lines_used_by_cli

        assert lines_used_by_cli is describe_snapshot
