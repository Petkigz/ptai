"""
One agent, one front end.

The operator's complaint was about the SHAPE of the product, not a missing
feature: eighteen tabs of tools read as a toolbox with a trading bot somewhere
inside it, and the screen that was supposed to answer "is it earning?" answered
with scan counts and model names instead.

The fix has two halves, and this file holds both together:

  * the SYSTEM publishes what it is doing while it works (`agent.phase`) and
    computes, from what it knows, the one thing standing between it and earning
    (`blockers`) - in `operator_view`, where the CLI reads the same payload; and
  * the UI is ONE screen - the console - which reads that snapshot instead of
    computing its own version of the truth, and is the only thing the runner
    starts.

Two rules the tests below are written to enforce:

  1. A screen may not compute a figure the snapshot already owns. Two readers of
     the same state is how a dashboard ends up disagreeing with the agent.
  2. "Alive" and "earning" are different claims, and neither is allowed to
     stand in for the other: a stopped agent says stopped, and a running one
     with nothing settled says exactly that.
"""

import importlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ptai.operator_view import (
    agent_state,
    build_blockers,
    describe_snapshot,
    live_window_seconds,
    operator_snapshot,
)
from src.ptai.storage.db import Storage

REPO = Path(__file__).resolve().parents[1]



def _quoted_strings_spanning_lines(script: str):
    """
    Find single- or double-quoted literals that contain a raw line break.

    A quote opened in code and never closed on its line is a syntax error in the
    browser, and the browser reports it by doing nothing at all. Template
    literals are allowed to span lines and may contain apostrophes, so they are
    tracked with a stack (including nested ``${...}`` interpolations) and only
    quotes opened in CODE are judged.
    """
    offenders, i, n = [], 0, len(script)
    stack = []
    while i < n:
        ch = script[i]
        nxt = script[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = script.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "/" and nxt == "*":
            j = script.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if ch == "`":
            stack.pop() if stack and stack[-1] == "template" else stack.append("template")
            i += 1
            continue
        if stack and stack[-1] == "template":
            if ch == "\\":
                i += 2
                continue
            if ch == "$" and nxt == "{":
                stack.append("code")
                i += 2
                continue
            i += 1
            continue
        if ch == "}" and stack and stack[-1] == "code":
            stack.pop()
            i += 1
            continue
        if ch in ("'", '"'):
            quote, j = ch, i + 1
            while j < n:
                c = script[j]
                if c == "\\":
                    j += 2
                    continue
                if c == quote:
                    break
                if c == "\n":
                    offenders.append(script[i:i + 70])
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return offenders


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def _heartbeat(storage, status: str = "cycle", seconds_ago: float = 1.0,
               level: int = 0) -> None:
    storage.set_state("agent.heartbeat", json.dumps({
        "at": _iso(seconds_ago), "status": status, "kill_switch_level": level,
    }))


def _phase(storage, phase: str, detail: str, seconds_ago: float = 1.0,
           next_cycle_at: str = None) -> None:
    storage.set_state("agent.phase", json.dumps({
        "at": _iso(seconds_ago), "phase": phase, "label": phase,
        "detail": detail, "next_cycle_at": next_cycle_at,
    }))


def _scan(storage, seconds_ago: float = 5.0, markets: int = 214) -> None:
    storage.log_scan(markets_scanned=markets, opportunities_found=3,
                     avg_edge=0.05, execution_time=4.2, bankroll=50.0)
    # Age the row deliberately, with an explicit commit: an uncommitted UPDATE
    # is rolled back when the connection closes, and the test then measures
    # "just now" while believing it measured "an hour ago".
    storage.conn.execute(
        "UPDATE market_scans SET timestamp = ? WHERE id = "
        "(SELECT MAX(id) FROM market_scans)", (_iso(seconds_ago),))
    storage.conn.commit()


# ---------------------------------------------------------------------------
# 1. Alive, working, blocked or dead - four different things
# ---------------------------------------------------------------------------

class TestTheStateIsReadFromWhatTheAgentWrites:
    def test_a_fresh_database_says_the_agent_has_never_run(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "fresh.db"))
        try:
            state = agent_state(storage)
            assert state["running"] is False
            assert state["state"] == "not_running"
            assert "have ever been recorded" in state["evidence"]

            snapshot = operator_snapshot(storage)
            assert snapshot["blockers"][0]["id"] == "agent_not_running"
            assert snapshot["headline"].startswith("Stopped")
            assert "run_ptai.bat" in snapshot["next_action"]
        finally:
            storage.close()

    def test_a_heartbeat_makes_a_first_cycle_visible(self, tmp_path):
        """
        The first cycle can outlast the window (a slow local model makes that
        normal). A cycle in progress must not read as a dead agent - that is the
        false negative this screen kept showing the operator.
        """
        storage = Storage(db_path=str(tmp_path / "beat.db"))
        try:
            _heartbeat(storage, "cycle", seconds_ago=3)
            state = agent_state(storage)
            assert state["running"] is True
            assert state["state"] == "working"
            assert "still in progress" in state["evidence"]
        finally:
            storage.close()

    def test_the_phase_says_what_it_is_doing_right_now(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "phase.db"))
        try:
            _heartbeat(storage)
            _phase(storage, "evaluating",
                   "214 market(s) from 3 venue(s); pricing them against fees, "
                   "depth and uncertainty")
            state = agent_state(storage)
            assert state["phase"] == "evaluating"
            assert state["doing"].startswith("214 market(s)")
            # The same sentence reaches the CLI, because both read one snapshot.
            snapshot = operator_snapshot(storage)
            lines = "\n".join(describe_snapshot(snapshot))
            assert "214 market(s) from 3 venue(s)" in lines
        finally:
            storage.close()

    def test_a_stale_sign_of_life_is_not_a_live_agent(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "stale.db"))
        try:
            _heartbeat(storage, "cycle", seconds_ago=2 * 3600)
            _scan(storage, seconds_ago=2 * 3600)
            state = agent_state(storage)
            assert state["running"] is False
            assert "outside the" in state["evidence"]
        finally:
            storage.close()

    def test_the_liveness_window_follows_the_interval(self, tmp_path, monkeypatch):
        """
        A 30-minute-old cycle is dead at a 10-minute interval and perfectly
        healthy at an hourly one. The window was a hard-coded 15 minutes, so
        raising the interval made a working agent report itself as dead.
        """
        storage = Storage(db_path=str(tmp_path / "window.db"))
        try:
            _scan(storage, seconds_ago=30 * 60)
            monkeypatch.setenv("INTERVAL_MIN", "10")
            assert agent_state(storage)["running"] is False
            monkeypatch.setenv("INTERVAL_MIN", "60")
            state = agent_state(storage)
            assert state["running"] is True
            assert state["interval_min"] == 60
            assert live_window_seconds(60) > 30 * 60
        finally:
            storage.close()

    def test_a_blocked_agent_is_alive_and_says_so(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "blocked.db"))
        try:
            _heartbeat(storage, "kill_switch_L2", seconds_ago=4, level=2)
            state = agent_state(storage)
            assert state["running"] is True
            assert state["state"] == "blocked"
            assert state["blocked_by_kill_switch"] is True
            blockers = build_blockers(operator_snapshot(storage))
            assert blockers[0]["id"] == "kill_switch"
            assert "bypassed" in blockers[0]["clear"]
        finally:
            storage.close()

    def test_a_running_agent_with_nothing_settled_does_not_claim_earnings(
            self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "honest.db"))
        try:
            _heartbeat(storage)
            _scan(storage, seconds_ago=60)
            headline = operator_snapshot(storage)["headline"]
            assert "nothing has settled" in headline
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# 2. The loop publishes the phase while it works
# ---------------------------------------------------------------------------

class TestTheLoopPublishesItsOwnPhase:
    def test_a_completed_cycle_leaves_the_phase_it_ended_in(self, tmp_path,
                                                            monkeypatch):
        from tests.test_full_cycle_from_discovery_to_allocation import (
            _cycle, _force_qualified, _inject_opportunity, build_agent)

        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        _cycle(agent)

        raw = agent.storage.get_state("agent.phase")
        assert raw, "the cycle finished without saying what it was doing"
        phase = json.loads(raw)
        assert phase["phase"] == "cycle_complete"
        assert "market(s) scanned" in phase["detail"]

        # And the console's reader turns that into the sentence on the screen.
        assert agent_state(agent.storage)["doing"] == phase["detail"]

    def test_a_cycle_that_finds_nothing_still_leaves_a_phase(self, tmp_path,
                                                             monkeypatch):
        from tests.test_full_cycle_from_discovery_to_allocation import build_agent

        agent, _adapter = build_agent(tmp_path)
        # No stub venue registered: discovery finds nothing, which is the
        # common case on a fresh install.
        from src.ptai.venues.registry import VenueRegistry
        agent.venue_registry = VenueRegistry(country_code="UG")
        agent.multi_venue_executor.registry = agent.venue_registry
        agent.account_health_engine.venue_registry = agent.venue_registry
        agent.capability_engine.venue_registry = agent.venue_registry
        agent.settlement_engine.venue_registry = agent.venue_registry

        result = __import__("asyncio").run(
            agent.run_cycle(target_per_venue=1, max_trades=1))
        assert result["status"] == "no_markets"
        phase = json.loads(agent.storage.get_state("agent.phase"))
        assert phase["phase"] == "no_markets"
        # An unavailable venue is not a quiet one, and the screen says so.
        assert "unavailable" in phase["detail"]


# ---------------------------------------------------------------------------
# 3. The console reads the snapshot, and computes no figures of its own
# ---------------------------------------------------------------------------

@pytest.fixture()
def console_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    module = importlib.import_module("src.ptai.ui.console")
    return module


@pytest.fixture()
def console_client(console_app):
    return TestClient(console_app.app)


class TestTheConsoleIsOneScreenAboutOneAgent:
    def test_the_front_page_reports_the_agents_own_state(self, console_client,
                                                         console_app, monkeypatch,
                                                         tmp_path):
        body = console_client.get("/api/console/agent").json()
        assert body["agent"]["state"] == "not_running"
        assert body["blockers"][0]["id"] == "agent_not_running"
        assert body["next_action"]

        # The figures on the screen ARE the snapshot's figures - not a second
        # computation that could drift from what `ptai status` prints.
        storage = Storage(db_path=str(tmp_path / "console.db"))
        try:
            snapshot = operator_snapshot(storage)
        finally:
            storage.close()
        assert body["headline"] == snapshot["headline"]
        assert body["capital"] == snapshot["capital"]
        assert body["profit"] == snapshot["profit"]
        assert body["agent"]["state"] == snapshot["agent"]["state"]

    def test_a_dead_brain_is_the_first_thing_named(self, console_client,
                                                   console_app, monkeypatch):
        monkeypatch.setattr(console_app, "_brain_status", lambda force=False: {
            "available": True, "connected": False, "models": [],
            "active_model": None, "error": "Connection refused",
            "host": "http://localhost:1234", "env_model": "local-model",
            "pinned": False,
        })
        body = console_client.get("/api/console/agent").json()
        ids = [b["id"] for b in body["blockers"]]
        assert "brain_offline" in ids
        # Behind a dead process (nothing runs) but ahead of everything else: a
        # live agent that cannot think is still not earning.
        assert ids[0] == "agent_not_running"
        assert ids[1] == "brain_offline"
        clear = [b["clear"] for b in body["blockers"] if b["id"] == "brain_offline"][0]
        assert "LM Studio" in clear

    def test_a_slow_model_is_a_warning_with_a_fix(self, console_client,
                                                  console_app, monkeypatch):
        monkeypatch.setattr(console_app, "_brain_status", lambda force=False: {
            "available": True, "connected": True, "models": ["deepseek-r1"],
            "active_model": "deepseek-r1", "is_r1": True, "error": None,
            "env_model": "local-model", "pinned": False,
        })
        body = console_client.get("/api/console/agent").json()
        slow = [b for b in body["blockers"] if b["id"] == "brain_slow"]
        assert slow and slow[0]["severity"] == "warning"
        assert "Pin a fast model" in slow[0]["clear"]

    def test_pinning_refuses_a_model_the_server_does_not_have(self, console_client,
                                                              console_app,
                                                              monkeypatch):
        written = {}
        import src.ptai.dashboard as dashboard
        monkeypatch.setattr(dashboard, "write_env_file",
                            lambda updates: written.update(updates))
        monkeypatch.setattr(console_app, "_brain_status", lambda force=False: {
            "available": True, "connected": True, "models": ["fast-model"],
            "active_model": "fast-model", "is_r1": False, "error": None,
            "env_model": "local-model", "pinned": False,
        })
        response = console_client.post("/api/console/brain",
                                       json={"model": "not-loaded"})
        assert response.status_code == 409
        assert written == {}, "an unloaded model must not reach .env"

    def test_pinning_writes_the_model_the_agent_will_call(self, console_client,
                                                          console_app, monkeypatch):
        written = {}
        import src.ptai.dashboard as dashboard
        monkeypatch.setattr(dashboard, "write_env_file",
                            lambda updates: written.update(updates))
        monkeypatch.setattr(console_app, "_brain_status", lambda force=False: {
            "available": True, "connected": True, "models": ["fast-model"],
            "active_model": "fast-model", "is_r1": False, "error": None,
            "env_model": "local-model", "pinned": False,
        })
        response = console_client.post("/api/console/brain",
                                       json={"model": "fast-model"})
        assert response.status_code == 200
        assert written == {"LM_STUDIO_MODEL": "fast-model"}
        assert "next" in response.json()["note"].lower()

    def test_the_status_steps_come_from_the_snapshot(self, console_client):
        body = console_client.get("/api/console/status").json()
        steps = {s["step"]: s for s in body["steps"]}
        assert {"agent", "capital", "data", "qualification", "evidence"} <= set(steps)
        assert steps["agent"]["ok"] is False
        assert steps["qualification"]["ok"] is False
        assert "combination(s) have enough evidence" in steps["qualification"]["detail"]
        # The wording the older console test pins: a venue that was asked and
        # did not answer is not a venue with a zero balance.
        assert ("reported a balance" in steps["data"]["detail"]
                or "no venue was reachable" in steps["data"]["detail"])

    def test_the_page_is_the_agent_not_the_toolbox(self, console_client):
        page = console_client.get("/").text
        for tab in ("agent", "money", "venue", "orders", "activity", "setup"):
            assert f'data-tab="{tab}"' in page
        for abandoned in ("teammates", "bots", "routines", "projects", "users",
                          "vault", "approvals", "backtest"):
            assert f'data-tab="{abandoned}"' not in page, (
                f"the console is showing the {abandoned} toolbox again")
        assert "What stands in the way" in page
        assert "run_ptai.bat" in page
        # The diagnostics lab is still reachable, by name, from Setup.
        assert "ptai.dashboard" in page

    def test_every_element_the_script_touches_exists(self, console_client):
        """
        A renamed id used to kill every handler on the page silently, because an
        inline script that throws still returns HTTP 200.
        """
        page = console_client.get("/").text
        ids = set(re.findall(r'id="([^"]+)"', page))
        script = max(re.findall(r"<script>(.*?)</script>", page, re.S), key=len)
        refs = set(re.findall(r"\$\('([^']+)'\)", script))
        assert refs - ids == set()

    def test_the_page_script_parses(self, console_client):
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed")
        page = console_client.get("/").text
        block = max(re.findall(r"<script>(.*?)</script>", page, re.S), key=len)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(block)
            path = fh.name
        try:
            result = subprocess.run([node, "--check", path],
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (
                f"the console script does not parse: {result.stderr[:600]}")
        finally:
            import os
            os.unlink(path)

    def test_no_string_literal_spans_a_newline(self, console_client):
        """
        The failure mode this page has already had: an escape written for Python
        (`\\n`) inside a JavaScript string is a REAL line break by the time it
        reaches the browser, and one of those silently kills every handler on
        the page while the HTTP response still says 200.
        """
        page = console_client.get("/").text
        script = max(re.findall(r"<script>(.*?)</script>", page, re.S), key=len)
        offenders = _quoted_strings_spanning_lines(script)
        assert offenders == [], f"unterminated string literal: {offenders[:1]}"


# ---------------------------------------------------------------------------
# 4. One front door
# ---------------------------------------------------------------------------

class TestTheRunnerStartsOneFrontEnd:
    @pytest.fixture()
    def bat(self):
        return (REPO / "run_ptai.bat").read_bytes()

    def test_it_starts_the_console(self, bat):
        assert b"ptai.ui.console" in bat

    def test_it_does_not_open_the_old_toolbox_with_the_agent(self, bat):
        """
        Starting eighteen tabs of tools next to the trading loop is what made
        the product read as a pile of loose parts. The lab is started on
        purpose, from Setup, when something needs diagnosing.
        """
        assert b"ptai.dashboard" not in bat

    def test_it_waits_for_the_port_it_opened(self, bat):
        text = bat.decode("utf-8", errors="replace")
        assert ":port_up" in text and "socket" in text
