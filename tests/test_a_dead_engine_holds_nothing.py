"""
A dead engine holds nothing, the page says who is running, and the paper purse
is never empty.

The operator's report, in their words:

    "the logs are not showing whats happening anymor , if i press stop nothing
     happens , refresh does nothing too . the button for start the agenbt is
     always greyed out . paper mode has zero balance available but it supposed
     to operate on capitaal so it need to have at least 20 or 50 dollars"

All four complaints have ONE cause, and it is not a rendering bug. A lease is
written by the engine that claims the agent. Freshness was decided by heartbeat
age alone, so when the engine was killed - the common case: the window was
closed, the console was restarted, the process crashed - the lease stayed
"held" for the whole stale window. Nothing ever asked whether the process named
in the lease still existed. The consequences are exactly what was reported:

  * Start:  greyed out, and refused with "already running in the console engine
            (pid 7444)" about a pid that no longer exists;
  * Stop:   refused with "the agent is running outside this console (pid 7444);
            close that window to stop it" - a window that does not exist;
  * Refresh: re-reads the same lease and redraws the same lie, so it looks like
            the button does nothing;
  * logs:   the panel showed this process's ring buffer and never said that no
            engine was running at all, so a stopped agent and a working one
            looked the same.

The paper complaint is separate and simpler: a purse at zero is a paper account
that cannot size a trade, and paper mode is supposed to operate on capital.

What this file enforces:

  1. Liveness of the HOLDER is part of the answer, not just the age of the
     heartbeat: a pid that is gone means the engine is gone, immediately.
     Unknown liveness (another host) is still treated as possibly alive,
     because refusing to start a second engine is the safe error.
  2. Start and Stop act on that truth: Start takes an abandoned lease, Stop
     reports an abandoned lease as cleaned up, and neither is allowed to invent
     a holder. A LIVE holder in another process is still refused, by name.
  3. The page can say whose log it is showing and whether an agent is running.
  4. The paper account always has a purse, and it is at least the operator's
     floor, without ever double-counting capital that is already committed.
"""

import importlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ptai.agent import engine_host
from src.ptai.execution.capital import (
    PAPER_PURSE_MIN_USD,
    ensure_paper_purse,
    paper_purse_state,
)
from src.ptai.storage.db import Storage

REPO = Path(__file__).resolve().parents[1]
CONSOLE_SOURCE = (REPO / "src" / "ptai" / "ui" / "console.py").read_text()
LOOP_SOURCE = (REPO / "src" / "ptai" / "agent" / "v3_loop.py").read_text()
POLYMARKET_ADAPTER_SOURCE = (
    REPO / "src" / "ptai" / "venues" / "polymarket_adapter.py").read_text()


def _lease(storage, *, pid, kind="console", age_seconds=1.0, host=None,
           released=False):
    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(seconds=age_seconds)).isoformat()
    storage.set_state(engine_host.LEASE_KEY, json.dumps({
        "kind": kind, "pid": pid, "host": host or engine_host.here()["host"],
        "started_at": stamp, "heartbeat_at": stamp,
        "released": released, "released_reason": None, "released_at": None,
    }))


@pytest.fixture()
def live_other_process():
    """
    A live process that is NOT the one running the test.

    `is_self` is decided by pid+host, so a lease claimed by the test process
    itself is "ours" and would exercise the wrong branch. A refusal has to be
    about someone else's window.
    """
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(120)"])
    try:
        time.sleep(0.2)
        yield proc.pid
    finally:
        proc.kill()
        proc.wait()


def _dead_pid() -> int:
    """
    A pid that is certainly not running, on this or any machine.

    Deliberately not a hard-coded 999999-style constant: a busy machine can own
    any low number. This asks the OS for one and hands it straight back.
    """
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.fixture()
def storage(tmp_path):
    st = Storage(db_path=str(tmp_path / "v54.db"))
    try:
        yield st
    finally:
        st.close()


# ---------------------------------------------------------------------------
# 1. A pid that is gone is not a holder - and a window is not a guess
# ---------------------------------------------------------------------------

class TestTheLeaseKnowsIfItsProcessExists:
    def test_this_process_is_alive_and_a_finished_one_is_not(self):
        assert engine_host.pid_alive(os.getpid()) is True
        assert engine_host.pid_alive(_dead_pid()) is False
        # "Cannot tell" is its own answer, and it is not False: refusing to
        # start a second engine is the safe error, so unknown must never read
        # as gone.
        assert engine_host.pid_alive(None) is None
        assert engine_host.pid_alive("not-a-pid") is None
        assert engine_host.pid_alive(0) is None

    def test_a_dead_holder_is_not_a_holder_even_with_a_fresh_heartbeat(self, storage):
        # THE REPRODUCED BUG: killed mid-run, so the heartbeat is seconds old.
        # Age alone said "held"; the process was already gone.
        _lease(storage, pid=_dead_pid(), age_seconds=2.0)
        status = engine_host.engine_lease_status(storage, interval_min=10)
        assert status["stale"] is False, "a 2s-old heartbeat is not stale"
        assert status["holder_alive"] is False
        assert status["abandoned"] is True
        assert status["held"] is False, "nothing is running, so nothing is held"
        assert str(status["pid"]) in status["note"]
        assert "GONE" in status["note"] or "gone" in status["note"]

    def test_a_live_holder_is_still_a_holder(self, storage, live_other_process):
        _lease(storage, pid=live_other_process, kind="cli", age_seconds=1.0)
        status = engine_host.engine_lease_status(storage, interval_min=10)
        assert status["holder_alive"] is True
        assert status["abandoned"] is False
        assert status["held"] is True
        assert status["is_self"] is False, "this test process did not claim it"

    def test_another_host_is_unknown_and_keeps_its_leash(self, storage):
        # Liveness is not observable across machines. Treating "cannot tell" as
        # dead would let two engines run on one database - the exact bug this
        # whole mechanism exists to prevent.
        _lease(storage, pid=4242, host="some-other-box", age_seconds=1.0)
        status = engine_host.engine_lease_status(storage, interval_min=10)
        assert status["holder_alive"] is None
        assert status["abandoned"] is False
        assert status["held"] is True

    def test_a_stale_live_engine_is_not_abandoned_it_is_just_stale(self, storage):
        # The old rule still stands and was not replaced: a process that exists
        # but stopped showing life is a different failure, and it must not be
        # reported as "its process no longer exists".
        _lease(storage, pid=os.getpid(), age_seconds=4000.0)
        status = engine_host.engine_lease_status(storage, interval_min=10)
        assert status["stale"] is True
        assert status["abandoned"] is False
        assert status["held"] is False
        assert "GONE" not in status["note"]

    def test_the_stale_window_was_not_shortened_to_fake_this(self, storage):
        # A short window would "fix" the symptom by declaring live engines dead
        # every few minutes. The freshness window is the interval-derived one.
        _lease(storage, pid=_dead_pid(), age_seconds=1.0)
        status = engine_host.engine_lease_status(storage, interval_min=10)
        assert status["stale_after_seconds"] >= 600.0


# ---------------------------------------------------------------------------
# 2. Start and Stop act on the truth
# ---------------------------------------------------------------------------

@pytest.fixture()
def console(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console-v54.db"))
    module = importlib.import_module("src.ptai.ui.console")
    monkeypatch.setattr(module, "_CONTROLLER", {}, raising=False)
    return module


@pytest.fixture()
def client(console):
    return TestClient(console.app)


class TestStartAndStopAnswerThePress:
    def test_stop_cleans_up_an_abandoned_lease_instead_of_refusing(self, client, console):
        st = console.get_storage()
        dead = _dead_pid()
        _lease(st, pid=dead)
        body = client.post("/api/console/agent-control",
                           json={"action": "stop"}).json()
        assert body["stopped"] is True, body
        assert body.get("abandoned") is True
        assert str(dead) in str(body.get("reason"))
        after = engine_host.engine_lease_status(st, interval_min=10)
        assert after["held"] is False
        assert after["abandoned"] is False
        assert after["released"] is True

    def test_start_takes_an_abandoned_lease_and_starts(self, client, console):
        st = console.get_storage()
        _lease(st, pid=_dead_pid())
        before = engine_host.engine_lease_status(st, interval_min=10)
        assert before["abandoned"] is True, "precondition: the holder is gone"
        body = client.post("/api/console/agent-control",
                           json={"action": "start"}).json()
        assert body.get("started") is True, body
        assert st.get_state("agent.engine_lease")
        after = engine_host.engine_lease_status(st, interval_min=10)
        assert after["held"] is True
        assert after["is_self"] is True
        # ...and leave nothing running for the next test in this file.
        assert client.post("/api/console/agent-control",
                           json={"action": "stop"}).json()["stopped"] is True

    def test_stop_still_refuses_a_live_engine_in_another_process(
            self, client, console, live_other_process):
        # The fix must not become "Stop always says stopped".
        st = console.get_storage()
        _lease(st, pid=live_other_process, kind="cli")
        body = client.post("/api/console/agent-control",
                           json={"action": "stop"}).json()
        assert body["stopped"] is False
        assert "cli" in str(body["reason"])
        assert str(live_other_process) in str(body["reason"])
        assert body.get("holder", {}).get("pid") == live_other_process
        assert engine_host.engine_lease_status(st, interval_min=10)["held"] is True

    def test_start_still_refuses_a_live_engine_in_another_process(
            self, client, console, live_other_process):
        st = console.get_storage()
        _lease(st, pid=live_other_process, kind="cli")
        body = client.post("/api/console/agent-control",
                           json={"action": "start"}).json()
        assert body.get("started") is False
        assert "already running" in str(body.get("reason"))
        assert str(live_other_process) in str(body.get("reason"))

    def test_the_page_is_told_a_dead_holder_owns_nothing(self, client, console):
        st = console.get_storage()
        _lease(st, pid=_dead_pid())
        lease = client.get("/api/console/agent").json()["engine"]["lease"]
        assert lease["held"] is False
        assert lease["abandoned"] is True
        assert lease["holder_alive"] is False
        # The page's own arithmetic: Start is enabled unless something that
        # exists holds it.
        assert not (lease["held"] and lease["holder_alive"] is not False)


# ---------------------------------------------------------------------------
# 3. The page says whose log it is, and whether anything is running
# ---------------------------------------------------------------------------

class TestThePageCannotHideAStoppedEngine:
    def test_start_is_not_greyed_by_a_lease_whose_process_is_gone(self):
        # The exact expression the page uses. `lease.held` is already False for
        # an abandoned lease; `holder_alive !== false` keeps it honest if an
        # older payload says otherwise.
        i = CONSOLE_SOURCE.index("const held = !!lease.held")
        line = CONSOLE_SOURCE[i:i + 200]
        assert "lease.holder_alive !== false" in line
        i = CONSOLE_SOURCE.index("start.disabled =")
        assert "held" in CONSOLE_SOURCE[i:i + 120]

    def test_the_abandoned_note_is_actually_rendered(self):
        # Found while writing this: the note was pushed into `bits` AFTER
        # `el.innerHTML = bits.join(...)`, so it never reached the page. A note
        # that is built and not shown is the same defect as no note.
        fn = CONSOLE_SOURCE.index("function paintEngine(")
        end = CONSOLE_SOURCE.index("function agentAction(")
        body = CONSOLE_SOURCE[fn:end]
        push = body.index("bits.push('<span class=\"neg\">the engine that claimed")
        render = body.index("el.innerHTML = bits.join")
        assert push < render, "the abandoned note is rendered, not discarded"

    def test_every_press_leaves_an_answer_on_the_page(self):
        # "if i press stop nothing happens" is a press answered with silence.
        assert "async function agentAction(" in CONSOLE_SOURCE
        i = CONSOLE_SOURCE.index("async function agentAction(")
        body = CONSOLE_SOURCE[i:i + 2600]
        assert "$('engineBox')" in body
        assert "insertAdjacentHTML" in body
        assert 'id="engineBox"' in CONSOLE_SOURCE

    def test_the_log_panel_says_whose_log_it_is(self):
        i = CONSOLE_SOURCE.index("async function loadLogs(")
        body = CONSOLE_SOURCE[i:i + 2600]
        assert "logWhere" in body
        assert "no engine is running the agent" in body
        assert "the agent is running in this console" in body
        assert "another process (pid " in body
        # ...and it names the process the lines came from, so "these lines are
        # from THIS console" is never left implicit.
        assert "buffered by the " in body
        assert "body.pid" in body

    def test_the_stop_button_is_live_when_a_lease_must_be_cleared(self):
        i = CONSOLE_SOURCE.index("stop.disabled =")
        assert "held" in CONSOLE_SOURCE[i:i + 120]
        assert "'Stop (another process)'" in CONSOLE_SOURCE

    def test_the_refusal_names_the_process_that_owns_the_agent(self):
        assert "the agent is running in ANOTHER process" in CONSOLE_SOURCE
        assert "'the agent is already running in the " in CONSOLE_SOURCE \
            or "already running in the" in CONSOLE_SOURCE


# ---------------------------------------------------------------------------
# 4. The paper account always has a purse
# ---------------------------------------------------------------------------

class TestThePaperPurseIsNeverEmpty:
    def test_the_floor_is_the_operators_number(self):
        # "it need to have at least 20 or 50 dollars"
        assert 20.0 <= PAPER_PURSE_MIN_USD <= 50.0

    def test_an_empty_purse_with_nothing_open_is_reseeded(self, storage):
        storage.set_paper_bankroll(0.0)
        assert storage.get_paper_bankroll() == 0.0
        result = ensure_paper_purse(storage, at="the test")
        assert result["topped_up"] is True
        assert result["balance_usd"] >= PAPER_PURSE_MIN_USD
        assert storage.get_paper_bankroll() >= PAPER_PURSE_MIN_USD
        assert "paper money" in result["reason"]
        # It is a deposit, not a profit: the equity curve must not read a
        # re-seed as a winning round.
        assert "history" in result["reason"]

    def test_the_reseed_is_recorded_in_the_bankroll_history(self, storage):
        storage.set_paper_bankroll(0.0)
        ensure_paper_purse(storage, at="the test")
        rows = storage.conn.execute(
            "SELECT bankroll FROM bankroll_history WHERE account_mode = 'paper' "
            "ORDER BY id DESC LIMIT 3").fetchall()
        assert rows, "the top-up left no trace in the history"
        assert float(rows[0]["bankroll"]) >= PAPER_PURSE_MIN_USD

    def test_the_reseed_respects_the_configured_bankroll(self, storage):
        storage.set_bankroll(50.0)
        storage.set_paper_bankroll(0.0)
        result = ensure_paper_purse(storage, at="the test")
        assert result["balance_usd"] == pytest.approx(50.0, abs=0.01)

    def test_a_working_purse_is_not_rewritten(self, storage):
        storage.set_paper_bankroll(37.25)
        result = ensure_paper_purse(storage, at="the test")
        assert result["topped_up"] is False
        assert storage.get_paper_bankroll() == pytest.approx(37.25, abs=0.01)

    def test_capital_already_in_positions_is_never_topped_up(self, storage):
        # The account's own money is already in the equity figure. Adding to it
        # would double-count, so a purse at zero BECAUSE it is committed must
        # stay at zero - and say why.
        storage.set_paper_bankroll(0.0)
        storage.log_trade({
            "market_id": "m1", "venue_id": "polymarket",
            "market_question": "does a committed paper position block a top-up?",
            "side": "YES", "position_size_usd": 50.0, "market_price": 0.5,
            "token_price_at_entry": 0.5, "execution_mode": "paper",
            "status": "paper",
        })
        state = paper_purse_state(storage)
        assert state["open_positions"] >= 1
        result = ensure_paper_purse(storage, at="the test")
        assert result["topped_up"] is False
        assert storage.get_paper_bankroll() == 0.0

    def test_reading_the_purse_writes_nothing(self, storage):
        storage.set_paper_bankroll(0.0)
        before = storage.get_paper_bankroll()
        state = paper_purse_state(storage)
        assert state["empty"] is True
        assert "re-seeded" in state["reason"]
        assert storage.get_paper_bankroll() == before, "a read must not write"

    def test_the_agent_funds_the_purse_at_the_cycle_start(self):
        assert "ensure_paper_purse" in LOOP_SOURCE
        i = LOOP_SOURCE.index("ensure_paper_purse")
        assert "cycle" in LOOP_SOURCE[i:i + 400]
        # ...and the purse travels with the cycle report, so the page shows the
        # same number the agent sized against.
        assert '"paper_purse"' in LOOP_SOURCE
        assert '"local_model": self._local_model_status()' in LOOP_SOURCE

    def test_pressing_start_funds_the_purse_immediately(self, client, console):
        st = console.get_storage()
        st.set_paper_bankroll(0.0)
        body = client.post("/api/console/agent-control",
                           json={"action": "start"}).json()
        assert body.get("started") is True, body
        assert st.get_paper_bankroll() >= PAPER_PURSE_MIN_USD
        client.post("/api/console/agent-control", json={"action": "stop"})

    def test_the_capital_block_carries_the_purse_and_its_reason(self, client, console):
        st = console.get_storage()
        st.set_paper_bankroll(0.0)
        purse = client.get("/api/console/agent").json()["capital"]["paper_purse"]
        assert purse["available"] is True
        assert purse["empty"] is True
        assert purse["min_usd"] == PAPER_PURSE_MIN_USD
        assert purse["reason"], "a zero with no reason is the reported defect"

    def test_the_status_page_shows_the_purse_in_paper_mode(self, client):
        body = client.get("/api/console/status").json()
        if body.get("mode") != "paper":
            pytest.skip("this console is not in paper mode")
        purse = body["capital"].get("paper_purse") or {}
        assert purse.get("available") is True
        assert "balance_usd" in purse

    def test_paper_mode_shows_the_paper_account_not_the_live_plan(self, client):
        # `total_available_usd` is about FUNDED LIVE VENUES. In paper mode no
        # venue is funded because none needs to be, so it reads $0.00 - and the
        # operator read that, correctly, as "paper mode has zero balance".
        body = client.get("/api/console/status").json()
        if body.get("mode") != "paper":
            pytest.skip("this console is not in paper mode")
        acct = body["capital"].get("paper_account") or {}
        assert acct.get("equity_usd") is not None
        assert acct.get("free_cash_usd") is not None
        assert "STATE.mode === 'paper'" in CONSOLE_SOURCE
        assert "paper account '" in CONSOLE_SOURCE or "paper account " in CONSOLE_SOURCE

    def test_the_venue_account_is_read_from_the_database_the_agent_writes(self):
        # Found by watching a status call construct `./data/ptai.db` while the
        # console was configured for another database: PTAI_DB was ignored, so
        # the venue reported the balance of a file the agent never wrote.
        assert 'Storage(db_path="./data/ptai.db")' not in POLYMARKET_ADAPTER_SOURCE
        assert "storage = Storage()" in POLYMARKET_ADAPTER_SOURCE

    def test_the_header_can_never_show_a_bare_zero(self):
        # "$0.00 available" in paper mode was read, correctly, as an account
        # that cannot trade. The header names the purse beside it.
        assert "paper purse" in CONSOLE_SOURCE
        assert "paper_purse" in CONSOLE_SOURCE
        assert "re-seeded when the agent starts" in CONSOLE_SOURCE
