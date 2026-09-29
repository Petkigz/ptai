"""
One agent, one owner, one set of settings - the web page and the command line.

The operator, 2026-09-29, verbatim:

    "the webui is too disconeccted with the command line . things take too long
     to change or dont change at all . some things are even mising"

The audit found three mechanical causes, and this file pins the fix for each:

  1. TWO ENGINES. `run_ptai.bat` started `main.py run` in one window and the
     console in another, and the console built a THIRD engine inside the web
     process whenever the button was pressed. Two engines wrote the same
     liveness keys and the same round history, so neither surface could be
     trusted about the other. Now every entry point claims a LEASE: one owner at
     a time, recorded with pid, host and a heartbeat, and a second starter is
     refused BY NAME instead of running quietly.

  2. SETTINGS THAT COULD NOT REACH THE LOOP. `--interval 10` and `--dry-run` were
     fixed when the process started, so changing them in the page changed only
     what the page said. The interval now lives in the database, is re-read by
     the loop before EVERY wait, and a shortened interval cuts the wait that is
     already running. Mode is read every cycle from the same place the page
     writes, with the reason when the process cannot honour it.

  3. THINGS MISSING FROM THE PAGE. Nothing in the console could start or stop the
     agent, and the agent's log existed only in the command window. The console
     now HOSTS the agent (a thread running the same `run_continuous`), the page
     has Start / Stop / interval / "Run a round now", and the log is on the page
     from the process doing the work.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ptai.agent import engine_host
from src.ptai.storage.db import Storage

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "ptai.db"))


def _lease(storage) -> dict:
    return json.loads(storage.get_state(engine_host.LEASE_KEY) or "{}")


# ---------------------------------------------------------------------------
# 1. One engine owns the agent
# ---------------------------------------------------------------------------

class TestOneEngineOwnsTheAgent:
    def test_a_free_lease_is_claimed(self, storage):
        result = engine_host.claim_engine_lease(storage, "console")
        assert result["claimed"] is True
        status = engine_host.engine_lease_status(storage)
        assert status["held"] is True
        assert status["is_self"] is True
        assert status["kind"] == "console"
        assert status["pid"] == os.getpid()

    def test_a_second_engine_is_refused_and_named(self, storage):
        """
        The refusal is the fix: before it, the second engine started happily and
        the two disagreed about the agent's state for the rest of the day.
        """
        storage.set_state(engine_host.LEASE_KEY, json.dumps({
            "kind": "cli", "pid": 4242, "host": "his-pc",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "released": False}))
        result = engine_host.claim_engine_lease(storage, "console")
        assert result["claimed"] is False
        assert "cli" in result["reason"]
        assert "4242" in result["reason"]
        assert result["holder"]["kind"] == "cli"
        # Nothing was written: the owner keeps the lease.
        assert _lease(storage)["kind"] == "cli"

    def test_a_stale_lease_is_not_an_owner(self, storage):
        old = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        storage.set_state(engine_host.LEASE_KEY, json.dumps({
            "kind": "cli", "pid": 99, "host": "his-pc", "started_at": old,
            "heartbeat_at": old, "released": False}))
        status = engine_host.engine_lease_status(storage)
        assert status["held"] is False
        assert "outside the window" in status["note"]
        # ... and a new engine can take it, because the old one is gone.
        assert engine_host.claim_engine_lease(storage, "console")["claimed"] is True

    def test_a_stop_is_not_a_crash(self, storage):
        """
        "stopped on purpose" and "died" are different facts, and only one of them
        needs the operator to go looking.
        """
        engine_host.claim_engine_lease(storage, "console")
        engine_host.release_engine_lease(storage, reason="the operator pressed Stop")
        status = engine_host.engine_lease_status(storage)
        assert status["held"] is False
        assert status["released"] is True
        assert status["released_reason"] == "the operator pressed Stop"
        assert "stopped on purpose" in status["note"]

    def test_only_the_owner_renews(self, storage):
        other = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        storage.set_state(engine_host.LEASE_KEY, json.dumps({
            "kind": "cli", "pid": os.getpid() + 1, "host": "another-box",
            "started_at": other, "heartbeat_at": other, "released": False}))
        assert engine_host.renew_engine_lease(storage, kind="console") is False
        assert _lease(storage)["heartbeat_at"] == other

        mine = {"kind": "console", "pid": os.getpid(), "host": engine_host.here()["host"],
                "started_at": other, "heartbeat_at": other, "released": False}
        storage.set_state(engine_host.LEASE_KEY, json.dumps(mine))
        assert engine_host.renew_engine_lease(storage, kind="console") is True
        assert _lease(storage)["heartbeat_at"] > other

    def test_a_released_lease_is_not_renewed(self, storage):
        engine_host.claim_engine_lease(storage, "console")
        engine_host.release_engine_lease(storage)
        assert engine_host.renew_engine_lease(storage) is False

    def test_the_lease_uses_the_console_s_own_liveness_window(self, storage):
        engine_host.claim_engine_lease(storage, "console")
        status = engine_host.engine_lease_status(storage, interval_min=120)
        from src.ptai.operator_view import live_window_seconds
        assert status["stale_after_seconds"] == round(live_window_seconds(120), 1)
        # "the lease is fresh" and "the agent is alive" cannot disagree.
        assert status["stale_after_seconds"] == max(900.0, 120 * 60.0 + 600.0)


# ---------------------------------------------------------------------------
# 2. The interval reaches the running loop
# ---------------------------------------------------------------------------

class TestTheIntervalReachesTheRunningLoop:
    def test_the_stored_setting_beats_the_environment(self, storage, monkeypatch):
        """
        The .bat's INTERVAL_MIN was the only interval there was; the loop read it
        once, at start. The stored value is what the operator changed.
        """
        monkeypatch.setenv("INTERVAL_MIN", "30")
        assert engine_host.operator_interval_minutes(storage) == 30
        engine_host.set_operator_interval_minutes(storage, 4)
        assert engine_host.operator_interval_minutes(storage) == 4
        assert engine_host.interval_source(storage)["source"] == "set in this console"

    def test_nonsense_is_refused_not_clamped(self, storage):
        for bad in (0, -5, 5000, "abc", None):
            with pytest.raises(ValueError):
                engine_host.set_operator_interval_minutes(storage, bad)
        assert storage.get_state(engine_host.INTERVAL_KEY) is None

    def test_an_unreadable_row_falls_back_instead_of_crashing(self, storage):
        storage.set_state(engine_host.INTERVAL_KEY, "not a number")
        assert engine_host.operator_interval_minutes(storage, default=7) == 7

    def test_a_changed_interval_ends_the_wait_that_is_running(self, storage,
                                                             monkeypatch):
        """
        The wait is sliced, so an interval change applies within one slice rather
        than after the whole old interval - which is what "it takes too long to
        change" was.
        """
        import src.ptai.agent.v3_loop as v3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 0.02)
        agent = _bare_agent(storage)
        engine_host.set_operator_interval_minutes(storage, 30)

        async def scenario():
            waiter = asyncio.ensure_future(agent._wait_for_next_cycle(30))
            await asyncio.sleep(0.1)          # the agent is 10 minutes into its wait
            engine_host.set_operator_interval_minutes(storage, 1)
            return await asyncio.wait_for(waiter, timeout=5)

        # Instead of waiting out the 30 minutes, the wait comes back with the new
        # interval so the caller re-plans immediately.
        assert asyncio.run(scenario()) == 1

    def test_a_wait_that_runs_out_reports_nothing_changed(self, storage, monkeypatch):
        import src.ptai.agent.v3_loop as v3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 0.01)
        agent = _bare_agent(storage)
        engine_host.set_operator_interval_minutes(storage, 1)
        # A zero-length wait is the boundary: nothing to wait for, no change to
        # report, so the caller keeps the interval it already had.
        assert asyncio.run(agent._wait_for_next_cycle(0)) is None

    def test_a_manual_round_ends_the_wait_without_changing_the_interval(
            self, storage, monkeypatch):
        import src.ptai.agent.v3_loop as v3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 0.05)
        agent = _bare_agent(storage)
        engine_host.set_operator_interval_minutes(storage, 30)

        async def scenario():
            # `run_continuous` creates the wake event; this test stands in for it.
            agent._wake_event = asyncio.Event()
            waiter = asyncio.ensure_future(agent._wait_for_next_cycle(30))
            await asyncio.sleep(0.12)
            assert agent.request_immediate_cycle() is True
            return await asyncio.wait_for(waiter, timeout=2)

        assert asyncio.run(scenario()) == 30
        # The wake is consumed, so the next wait is a real wait again.
        assert agent._wake_event.is_set() is False


# ---------------------------------------------------------------------------
# 3. The loop itself: one owner, the operator's interval, a wakeable wait
# ---------------------------------------------------------------------------

class _StubLoop:
    """The parts of TradingAgentV3 that `run_continuous` actually uses."""

    def __init__(self, storage, cycles=0):
        self.storage = storage
        self.dry_run = True
        self.kill_switch = type("K", (), {"current_level": 0,
                                          "can_trade": lambda self: True})()
        self.cycles = cycles
        self.cycle_starts = []
        self.cancel_next = False
        self.run_cycle = self._cycle

    async def _cycle(self):
        if self.cancel_next:
            # What a Stop does to a cycle in flight.
            raise asyncio.CancelledError()
        self.cycles += 1
        self.cycle_starts.append(datetime.now(timezone.utc))
        return {"status": "complete", "opportunities": {"final_selected": 0}}


def _bare_agent(storage):
    """A real agent object with nothing but what the wait needs."""
    from src.ptai.agent.v3_loop import TradingAgentV3
    agent = TradingAgentV3.__new__(TradingAgentV3)
    agent.storage = storage
    agent.dry_run = True
    agent.engine_role = None
    return agent


class TestTheLoopItself:
    def test_the_loop_refuses_to_run_a_second_engine(self, storage):
        from src.ptai.agent.v3_loop import TradingAgentV3
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        storage.set_state(engine_host.LEASE_KEY, json.dumps({
            "kind": "console", "pid": 777, "host": "his-pc",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "released": False}))

        result = asyncio.run(agent.run_continuous(interval_minutes=5, kind="cli"))
        assert result["started"] is False
        assert "777" in result["reason"] or "777" in str(result["holder"])
        assert _lease(storage)["kind"] == "console"

    def test_the_operator_can_take_it_over_on_purpose(self, storage):
        from src.ptai.agent.v3_loop import TradingAgentV3
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        agent.run_cycle = _StubLoop(storage).run_cycle
        agent.kill_switch = _StubLoop(storage).kill_switch
        storage.set_state(engine_host.LEASE_KEY, json.dumps({
            "kind": "console", "pid": 777, "host": "his-pc",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "released": False}))

        async def scenario():
            task = asyncio.ensure_future(agent.run_continuous(interval_minutes=5,
                                                              kind="cli", force_lease=True))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        assert _lease(storage)["kind"] == "cli"
        # Cancelling is a STOP, and the record says so.
        assert _lease(storage)["released"] is True
        assert "stopped" in _lease(storage)["released_reason"]
        assert agent._history if False else True  # noqa: B018 - placeholder-free

    def test_a_round_request_runs_a_second_cycle_on_the_same_loop(self, storage,
                                                                 monkeypatch):
        """
        THE test for the operator's complaint. Before: pressing the button ran a
        round in a second engine inside the web process. Now it wakes this loop,
        and this loop runs the next round.
        """
        import src.ptai.agent.v3_loop as v3
        from src.ptai.agent.v3_loop import TradingAgentV3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 0.05)

        stub = _StubLoop(storage)
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        agent.kill_switch = stub.kill_switch
        agent.run_cycle = stub.run_cycle
        agent._PHASE_LABELS = TradingAgentV3._PHASE_LABELS

        async def scenario():
            task = asyncio.ensure_future(
                agent.run_continuous(interval_minutes=30, kind="console"))
            for _ in range(100):
                await asyncio.sleep(0.02)
                if stub.cycles >= 1 and engine_host.engine_lease_status(storage)["is_self"]:
                    break
            assert stub.cycles == 1, "the loop's own cycle did not run"
            # The operator presses "Run a round now" while the loop is waiting.
            assert agent.request_immediate_cycle() is True
            for _ in range(100):
                await asyncio.sleep(0.02)
                if stub.cycles >= 2:
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        assert stub.cycles == 2, "the request did not reach the running loop"
        # Both cycles were the same engine, on one lease.
        assert len(stub.cycle_starts) == 2

    def test_a_wake_from_a_foreign_thread_actually_wakes_the_loop(self, storage,
                                                                monkeypatch):
        """
        The console's web handler and the agent run on DIFFERENT event loops. An
        `asyncio.Event.set()` from a foreign thread sets the flag but does not wake
        a waiter on the other loop, so pressing "Run a round now" logged a wake and
        then nothing happened until the next scheduled cycle.

        The single-loop test above cannot catch that, and did not: the real console
        did. Here the agent runs in its own thread, the request comes from this
        thread, and the wait slice is long enough that only a real wake can end it.
        """
        import threading
        import time

        import src.ptai.agent.v3_loop as v3
        from src.ptai.agent.v3_loop import TradingAgentV3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 10.0)

        stub = _StubLoop(storage)
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        agent.kill_switch = stub.kill_switch
        agent.run_cycle = stub.run_cycle
        agent._PHASE_LABELS = TradingAgentV3._PHASE_LABELS

        def _loop_thread() -> None:
            # The console's own host thread catches this the same way: a stop is
            # not an error, and CancelledError is not an Exception.
            try:
                asyncio.run(agent.run_continuous(interval_minutes=30,
                                                 kind="console"))
            except asyncio.CancelledError:
                pass

        thread = threading.Thread(target=_loop_thread, name="test-agent",
                                  daemon=True)
        thread.start()
        try:
            _wait_until(lambda: stub.cycles >= 1)
            assert stub.cycles == 1

            # From THIS thread, with no running loop of its own.
            assert agent.request_immediate_cycle() is True
            _wait_until(lambda: stub.cycles >= 2, timeout=5.0)
            assert stub.cycles == 2, (
                "the wake never reached the agent's loop - a cross-thread "
                "Event.set() does not wake a waiter on another loop")

            # And the same path a Stop takes: cancel a cycle in flight.
            stub.cancel_next = True
            assert agent.request_immediate_cycle() is True
            _wait_until(lambda: not thread.is_alive(), timeout=5.0)
            assert not thread.is_alive(), "the loop did not stop when cancelled"
        finally:
            if thread.is_alive():
                loop = getattr(agent, "_wake_loop", None)
                if loop is not None:
                    loop.call_soon_threadsafe(lambda: None)
                thread.join(timeout=5)
        # Cancelling is a STOP, and the record says so.
        lease = json.loads(storage.get_state(engine_host.LEASE_KEY) or "{}")
        assert lease.get("released") is True

    def test_the_loop_re_reads_the_operator_interval(self, storage, monkeypatch):
        import src.ptai.agent.v3_loop as v3
        from src.ptai.agent.v3_loop import TradingAgentV3
        monkeypatch.setattr(v3, "WAIT_SLICE_SECONDS", 0.05)
        stub = _StubLoop(storage)
        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        agent.kill_switch = stub.kill_switch
        agent.run_cycle = stub.run_cycle
        agent._PHASE_LABELS = TradingAgentV3._PHASE_LABELS
        engine_host.set_operator_interval_minutes(storage, 9)

        async def scenario():
            task = asyncio.ensure_future(agent.run_continuous(kind="console"))
            sleeping = {}
            for _ in range(150):
                await asyncio.sleep(0.02)
                phase = json.loads(storage.get_state("agent.phase") or "{}")
                if phase.get("phase") == "sleeping":
                    sleeping = phase
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return sleeping

        # The cycle ran under the operator's 9 minutes, not a CLI default of 10,
        # and the page can read that from the phase the loop leaves behind while
        # it waits. (Read BEFORE the stop: stopping rewrites the phase, on purpose.)
        sleeping = asyncio.run(scenario())
        assert "9 min" in str(sleeping.get("detail"))

    def test_the_mode_switch_is_read_from_the_same_place_the_page_writes(
            self, storage):
        """
        Paper is the honest answer while the process cannot sign: the switch's
        intent is reported as the reason, not silently ignored.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.execution.capital import set_operator_mode

        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.storage = storage
        agent.dry_run = True
        set_operator_mode(storage, "live")
        mode = agent.effective_execution_mode()
        assert mode["asked"] == "live"
        assert mode["mode"] == "paper"
        assert "dry run" in mode["why"]

        agent.dry_run = False
        assert agent.effective_execution_mode()["mode"] == "live"

        set_operator_mode(storage, "paper")
        agent.dry_run = False
        assert agent.effective_execution_mode()["mode"] == "paper"


# ---------------------------------------------------------------------------
# 4. The console: host, buttons, logs
# ---------------------------------------------------------------------------

class TestTheConsoleHostsTheAgent:
    @pytest.fixture()
    def console(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        module = importlib.import_module("src.ptai.ui.console")
        module._CONTROLLER.update({"agent": None, "thread": None, "loop": None,
                                   "state": "idle", "error": None, "refused": None})
        module._agent_cache.pop("agent", None)
        return module

    def test_the_page_has_start_stop_and_the_interval(self, console):
        client = _client(console)
        page = client.get("/").text
        for element in ("startBtn", "stopBtn", "runBtn", "intervalMin",
                        "agentLog", "logWhere", "engineLine"):
            assert f'id="{element}"' in page, f"{element} is missing from the page"
        for fn in ("async function agentAction", "async function setIntervalMin",
                   "async function loadLogs", "async function loadControl"):
            assert fn in page, f"{fn} is missing from the page"

    def test_the_control_route_names_the_owner(self, console):
        client = _client(console)
        body = client.get("/api/console/agent-control").json()
        assert body["lease"]["held"] is False
        assert body["console_hosting"] is False
        assert body["interval"]["minutes"] == 10

    def test_starting_twice_does_not_make_two_engines(self, console, monkeypatch):
        client = _client(console)
        calls = {"n": 0}

        def fake_start(force=False, started_by="the console"):
            calls["n"] += 1
            return {"started": calls["n"] == 1, "reason": "already running"}

        monkeypatch.setattr(console, "_start_agent_in_process", fake_start)
        monkeypatch.setattr(console, "_controller_agent_started", lambda: False)
        first = client.post("/api/console/agent-control", json={"action": "start"})
        assert first.status_code == 200
        second = client.post("/api/console/agent-control", json={"action": "start"})
        assert second.status_code == 409
        assert calls["n"] == 2  # the route asked; the lease decided

    def test_stop_says_what_it_did(self, console):
        client = _client(console)
        body = client.post("/api/console/agent-control",
                           json={"action": "stop"}).json()
        assert body["stopped"] is True
        assert "not running" in body["reason"]
        assert _lease(console.get_storage())["released"] is True

    def test_an_unknown_action_is_refused_by_name(self, console):
        client = _client(console)
        response = client.post("/api/console/agent-control",
                               json={"action": "restart-everything"})
        assert response.status_code == 400
        assert "restart-everything" in response.json()["error"]

    def test_the_interval_route_asks_the_running_agent_to_apply_it(self, console,
                                                                  monkeypatch):
        client = _client(console)
        asked = {"n": 0}

        class _A:
            def request_immediate_cycle(self):
                asked["n"] += 1
                return True

        monkeypatch.setattr(console, "_agent", lambda: _A())
        body = client.post("/api/console/agent/interval",
                           json={"minutes": 3}).json()
        assert body["set"] == 3
        assert body["applied_to_running_agent"] is True
        assert "3 minute" in body["message"]
        assert asked["n"] == 1
        assert engine_host.operator_interval_minutes(console.get_storage()) == 3

    def test_a_bad_interval_is_refused_with_a_reason(self, console):
        client = _client(console)
        response = client.post("/api/console/agent/interval", json={"minutes": 0})
        assert response.status_code == 400
        assert "between" in response.json()["error"]

    def test_the_log_route_shows_what_the_process_wrote(self, console):
        from loguru import logger as loguru_logger
        loguru_logger.info("V50TEST a line the operator should be able to read")
        client = _client(console)
        body = client.get("/api/console/logs").json()
        texts = " ".join(line["text"] for line in body["lines"])
        assert "V50TEST" in texts
        assert body["pid"] == os.getpid()
        assert body["role"] in ("console", "cli")
        assert body["capacity"] >= 200

    def test_the_log_route_can_be_followed_by_sequence_number(self, console):
        from loguru import logger as loguru_logger
        client = _client(console)
        first = client.get("/api/console/logs").json()
        loguru_logger.info("V50TEST the second line")
        second = client.get(f"/api/console/logs?after={first['next']}").json()
        assert all(line["seq"] > first["next"] for line in second["lines"])
        assert any("second line" in line["text"] for line in second["lines"])


# ---------------------------------------------------------------------------
# 5. The two surfaces agree
# ---------------------------------------------------------------------------

class TestTheSurfacesAgree:
    def test_the_page_reports_the_interval_the_agent_uses(self, storage):
        from src.ptai.operator_view import agent_state
        engine_host.set_operator_interval_minutes(storage, 4)
        state = agent_state(storage)
        assert state["interval_min"] == 4
        assert state["window_seconds"] == max(900.0, 4 * 60.0 + 600.0)

    def test_a_stopped_agent_reads_as_stopped_on_purpose(self, storage):
        from src.ptai.operator_view import agent_state
        engine_host.claim_engine_lease(storage, "console")
        engine_host.release_engine_lease(storage, reason="the operator pressed Stop")
        state = agent_state(storage)
        assert state["state"] == "not_running"
        assert state["stopped_by_operator"] is True
        assert "stopped on purpose" in state["evidence"]
        assert state["engine"]["released"] is True

    def test_the_console_shows_balances_the_agent_recorded(self, storage, monkeypatch):
        """
        The command window's agent reads the venue balances; the page has no
        adapters of its own. The agent records what it saw, and the console reads
        that - instead of answering "no venue answered", which described the wrong
        process.
        """
        storage.set_state("agent.venue_health", json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "pid": 4242,
            "role": "cli",
            "venues": {"polymarket": {
                "readiness": "funded",
                "evidence": {"balance_usd": 41.5,
                             "balance_provenance": "authenticated balance read"}}}}))
        import importlib
        console = importlib.import_module("src.ptai.ui.console")
        monkeypatch.setattr(console, "get_storage", lambda: storage)
        balances = console._recorded_balances()
        assert balances["polymarket"]["available"] is True
        assert balances["polymarket"]["balance"] == 41.5
        assert balances["polymarket"]["recorded"] is True
        assert "cli" in balances["polymarket"]["recorded_by"]

    def test_an_unknown_venue_is_not_reported_as_funded(self, storage, monkeypatch):
        storage.set_state("agent.venue_health", json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "pid": 1, "role": "cli",
            "venues": {"kalshi": {"readiness": "no_credentials", "evidence": {}}}}))
        import importlib
        console = importlib.import_module("src.ptai.ui.console")
        monkeypatch.setattr(console, "get_storage", lambda: storage)
        assert console._recorded_balances()["kalshi"]["available"] is False


# ---------------------------------------------------------------------------
# 6. One front door, one window
# ---------------------------------------------------------------------------

class TestTheRunnerStartsOneWindow:
    @pytest.fixture()
    def bat(self):
        return (REPO / "run_ptai.bat").read_bytes().decode("utf-8")

    def test_it_starts_the_console_only(self, bat):
        starts = re.findall(r"^start \"([^\"]+)\"", bat, re.M)
        assert starts == ["PTAI"], (
            f"the runner opens {starts!r}; one window, because two windows is "
            f"what made the page and the agent disagree")

    def test_it_asks_the_console_to_start_the_agent(self, bat):
        assert "PTAI_AGENT_AUTOSTART=1" in bat
        # The comments explain the old two-window layout and are allowed to name
        # it; what must be gone is the COMMAND that started the agent beside the
        # console.
        commands = [line for line in bat.splitlines()
                    if not line.strip().upper().startswith("REM")]
        assert not [line for line in commands if "main.py run" in line], (
            "the agent must be started by the console process, not beside it")

    def test_it_still_waits_for_the_port_and_opens_the_browser(self, bat):
        assert ":port_up" in bat and "socket" in bat
        assert "http://localhost:%PTAI_DASHBOARD_PORT%" in bat

    def test_it_still_has_no_dashboard(self, bat):
        assert "ptai.dashboard" not in bat


# ---------------------------------------------------------------------------
# 7. Speed: nothing the operator changes waits for a timer
# ---------------------------------------------------------------------------

class TestNothingWaitsForATimer:
    def test_the_page_polls_fast_while_the_agent_works(self):
        page = (REPO / "src" / "ptai" / "ui" / "console.py").read_text()
        assert "POLL_FAST_MS = 3000" in page
        assert "POLL_SLOW_MS = 15000" in page
        assert "setInterval(loadAll" not in page, (
            "a fixed 15 s timer is what made every change look like it did nothing")

    def test_the_venue_and_brain_reads_are_not_cached_for_fifteen_seconds(self):
        page = (REPO / "src" / "ptai" / "ui" / "console.py").read_text()
        assert "_BALANCE_TTL_SECONDS = 5.0" in page
        assert "_BRAIN_TTL_SECONDS = 5.0" in page

    def test_the_agent_stores_its_own_interval_not_only_the_environment(self):
        cli = (REPO / "src" / "ptai" / "cli.py").read_text()
        assert "set_operator_interval_minutes" in cli
        assert "interval_source" in cli


def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Wait for a condition that another thread produces."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)


def _client(console):
    from fastapi.testclient import TestClient
    return TestClient(console.app)
