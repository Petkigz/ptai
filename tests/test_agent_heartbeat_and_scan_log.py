"""
The console must be able to tell a live agent from a dead one.

The dashboard's System Health panel decides "Agent: Running" from evidence
in the database, not from faith:

  * a market_scans row - written by the agent after EVERY cycle, a cycle
    the kill switch blocked included (alive, refusing), and
  * the agent.heartbeat state - written at cycle start, so a first cycle
    that outlasts the liveness window (a slow local LLM makes that
    realistic) does not read as "Not running - run run_ptai.bat".

Before this wiring, the V3 loop - the loop `ptai run` actually executes -
wrote neither, so a perfectly healthy agent showed "Agent: Not running"
and "Last Scan: Never" forever, and the operator's first question ("is it
working?") had no honest answer to give.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.storage.db import Storage

from tests.test_full_cycle_from_discovery_to_allocation import (
    _cycle,
    _force_qualified,
    _inject_opportunity,
    build_agent,
)


def _scans(storage):
    rows = storage.conn.execute(
        "SELECT * FROM market_scans ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def _heartbeat(storage):
    raw = storage.get_state("agent.heartbeat")
    return json.loads(raw) if raw else None


def _health(db_file, monkeypatch):
    """
    Run the real /api/system/health route against a database file.

    The route closes the connection it is given, so each call gets a fresh
    connection to the same file - exactly how the dashboard's own
    get_storage() behaves per request.
    """
    import src.ptai.dashboard as dashboard
    monkeypatch.setattr(dashboard, "get_storage",
                        lambda: Storage(db_path=str(db_file)))
    from fastapi.testclient import TestClient
    response = TestClient(dashboard.app).get("/api/system/health")
    assert response.status_code == 200
    return response.json()


class TestV3CycleRecordsItself:
    def test_a_completed_cycle_writes_the_scan_row(self, tmp_path, monkeypatch):
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)

        _cycle(agent)

        rows = _scans(agent.storage)
        assert len(rows) == 1, "the cycle did not record its scan"
        row = rows[0]
        assert row["markets_scanned"] >= 1
        assert row["opportunities_found"] >= 1
        assert row["execution_time_seconds"] >= 0
        assert row["bankroll"] == pytest.approx(50.0)

    def test_the_cycle_start_leaves_a_heartbeat(self, tmp_path, monkeypatch):
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)

        _cycle(agent)

        hb = _heartbeat(agent.storage)
        assert hb is not None, "no heartbeat was written at cycle start"
        assert hb["status"] == "cycle"
        # The heartbeat must be a readable, recent timestamp.
        at = datetime.fromisoformat(hb["at"].replace("Z", "+00:00"))
        assert (datetime.now(timezone.utc) - at) < timedelta(minutes=5)

    def test_a_blocked_cycle_still_proves_the_agent_is_alive(
            self, tmp_path, monkeypatch):
        """
        Kill switch holding the cycle is NOT the process dying. The console
        must still see a fresh record, or a risk-protected agent looks dead
        and the operator restarts something that was doing exactly what it
        should have done.
        """
        agent, _ = build_agent(tmp_path)
        agent.kill_switch.can_trade = lambda: False
        agent.kill_switch.current_level = 2

        result = _cycle(agent)

        assert result["status"] == "blocked"
        rows = _scans(agent.storage)
        assert len(rows) == 1, "a blocked cycle left no evidence of life"
        assert rows[0]["markets_scanned"] == 0
        assert _heartbeat(agent.storage) is not None

    def test_a_no_markets_cycle_records_a_scan_row(
            self, tmp_path, monkeypatch):
        """
        A fresh deployment: no venue has credentials, so every cycle
        discovers nothing and takes the no_markets path. That path must
        record its scan too, or the console reports "Not running" and
        "Last Scan: Never" for an agent that is honestly scanning on
        every interval - the exact panel the operator saw.
        """
        agent, _ = build_agent(tmp_path)

        async def no_venues(*args, **kwargs):
            return {}

        monkeypatch.setattr(agent, "discover_all_venues", no_venues)
        result = _cycle(agent)

        assert result["status"] == "no_markets"
        rows = _scans(agent.storage)
        assert len(rows) == 1, "a no-markets cycle left no evidence of life"
        assert rows[0]["markets_scanned"] == 0
        assert _heartbeat(agent.storage) is not None

    def test_the_continuous_loop_heartbeats_while_blocked(
            self, tmp_path, monkeypatch):
        """
        While the kill switch holds, run_cycle never runs - so the loop
        itself must keep the heartbeat fresh, or the "blocked" state is
        indistinguishable from "dead" on the dashboard.
        """
        agent, _ = build_agent(tmp_path)
        agent.kill_switch.can_trade = lambda: False
        agent.kill_switch.current_level = 3

        class _StopLoop(Exception):
            pass

        async def fake_sleep(_seconds):
            raise _StopLoop("stop-the-loop")

        monkeypatch.setattr("src.ptai.agent.v3_loop.asyncio.sleep",
                            fake_sleep)
        with pytest.raises(_StopLoop):
            asyncio.run(agent.run_continuous(interval_minutes=10))

        hb = _heartbeat(agent.storage)
        assert hb is not None
        assert hb["status"] == "kill_switch_L3"


class TestDashboardLiveness:
    def test_a_fresh_heartbeat_reads_as_running_with_no_scan_yet(
            self, tmp_path, monkeypatch):
        """
        The first cycle in progress: the process is alive but has not
        completed a scan. This is the state the operator actually saw
        ("Not running - run run_ptai.bat" with a running agent) - it must
        read as running, in progress.
        """
        store = Storage(db_path=str(tmp_path / "health.db"))
        store.set_state(
            "agent.heartbeat",
            json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                        "status": "cycle"}))
        store.close()
        monkeypatch.delenv("INTERVAL_MIN", raising=False)

        data = _health(tmp_path / "health.db", monkeypatch)

        assert data["agent_running"] is True
        assert data["last_scan"] is None  # honest: nothing completed yet
        assert data["agent_heartbeat_ago_seconds"] is not None

    def test_a_stale_heartbeat_with_no_scan_reads_as_not_running(
            self, tmp_path, monkeypatch):
        store = Storage(db_path=str(tmp_path / "health.db"))
        stale = datetime.now(timezone.utc) - timedelta(hours=2)
        store.set_state(
            "agent.heartbeat",
            json.dumps({"at": stale.isoformat(), "status": "cycle"}))
        store.close()
        monkeypatch.delenv("INTERVAL_MIN", raising=False)

        data = _health(tmp_path / "health.db", monkeypatch)

        assert data["agent_running"] is False

    def test_a_recent_scan_row_reads_as_running(self, tmp_path, monkeypatch):
        store = Storage(db_path=str(tmp_path / "health.db"))
        store.log_scan(1234, 5, 0.04, 30.0, 50.0)
        store.close()
        monkeypatch.delenv("INTERVAL_MIN", raising=False)

        data = _health(tmp_path / "health.db", monkeypatch)

        assert data["agent_running"] is True
        assert data["last_scan"]["markets_scanned"] == 1234
        assert data["last_scan"]["opportunities_found"] == 5

    def test_the_liveness_window_follows_the_interval(self, tmp_path,
                                                      monkeypatch):
        """
        A healthy agent scans once per interval. With a 10-minute interval a
        scan 30 minutes old is stale, but with a 60-minute interval it is
        mid-cycle - the same evidence must be judged by the configured
        interval, not by a hard-coded 15 minutes.
        """
        store = Storage(db_path=str(tmp_path / "health.db"))
        # A scan 30 minutes in the past.
        store.log_scan(100, 2, 0.03, 20.0, 50.0)
        now = datetime.now(timezone.utc)
        store.conn.execute(
            "UPDATE market_scans SET timestamp=? "
            "WHERE timestamp=?",
            ((now - timedelta(minutes=30)).isoformat(),
             _scans(store)[0]["timestamp"]))
        store.conn.commit()  # close() rolls back an uncommitted UPDATE
        store.close()
        monkeypatch.delenv("INTERVAL_MIN", raising=False)

        data = _health(tmp_path / "health.db", monkeypatch)
        assert data["agent_running"] is False, (
            "a 30-minute-old scan must be stale on a 10-minute interval")

        monkeypatch.setenv("INTERVAL_MIN", "60")
        data = _health(tmp_path / "health.db", monkeypatch)
        assert data["agent_running"] is True, (
            "the same scan is mid-cycle on a 60-minute interval")

    def test_the_agent_row_says_first_cycle_not_nan(
            self, tmp_path, monkeypatch):
        """
        The UI string for a running agent with no completed scan must not
        render "NaN min ago".
        """
        store = Storage(db_path=str(tmp_path / "health.db"))
        store.set_state(
            "agent.heartbeat",
            json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                        "status": "cycle"}))
        store.close()
        monkeypatch.delenv("INTERVAL_MIN", raising=False)

        data = _health(tmp_path / "health.db", monkeypatch)

        assert data["agent_running"] is True
        assert data["last_scan_ago_seconds"] is None
        # Guard the JS contract: the page renders the in-progress branch
        # precisely when last_scan_ago_seconds is null AND running is true.
        source = open("src/ptai/dashboard.py").read()
        assert "first cycle in progress" in source
        assert "last_scan_ago_seconds != null" in source
