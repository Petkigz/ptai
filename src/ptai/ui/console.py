"""
PTAI Console - the control panel for the live and paper system.

What this screen is FOR, in one sentence: to show the operator what the agent has
done with their money, and to let them change the two things that decide how much
is at risk - the mode, and the budget per venue.

Design rules, each one a reaction to a specific way this goes wrong:

  * LIVE and PAPER are never ambiguous. The mode is a banner in the header, not a
    setting buried in a form. An operator should never have to wonder which one
    is running.
  * Paper and live money are shown SEPARATELY. A paper position must not consume
    live budget, and a live position must not hide behind paper ones.
  * RESERVED capital is its own row. A resting order has committed cash without
    buying anything, so it is not a position and it is not available either.
  * Nothing is shown as verified unless it was verified. A balance that was not
    read is "unread", not zero. A venue with no probe is "unproven", not ready.
  * The funding panel explains how to put money in. See execution/capital.py for
    why the answer is "deposit at the venue" and not "send the agent money".
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from ..execution.capital import (
    CapitalLedger,
    FUNDING_ROUTES,
    UNFUNDABLE_SMALL,
    authorised_budgets,
    operator_mode,
    plan_for_budget,
    set_authorised_budget,
    set_operator_mode,
)
from ..execution.round import load_rounds, round_history_summary
from ..agent import engine_host
from ..storage.db import Storage
from . import live_log

# The console process now runs the agent (see the host further down), so the page
# can show the lines the command-line window used to be the only source of.
live_log.install()
from ..strategy.venue_selection import MIN_SAMPLE_FOR_EVIDENCE, VenueSelector
from ..validation.rule_bench import validation_block
from ..venues.inventory import load_inventory
from ..venues import credentials as credential_store
from ..venues import preferences as venue_switches

app = FastAPI(title="PTAI Console", version="console-1")


@app.on_event("startup")
async def _startup_autostart_agent() -> None:
    """
    When the runner starts this console, the AGENT starts here too.

    `run_ptai.bat` used to open two windows - the agent, and the console - and
    that is the split the operator reported: the page could not change the agent
    (different process), and the agent's `--interval 10` could not be changed from
    the page (a CLI flag, fixed at start). `run_ptai.bat` now opens ONE window and
    sets `PTAI_AGENT_AUTOSTART=1`, so "run PTAI" still means "the agent is
    working", with the loop, the settings and the log all in this process.

    Autostart is skipped, with the reason on the page, if another engine already
    owns the agent: two engines on one database is precisely the bug this fixes.
    """
    if os.environ.get("PTAI_AGENT_AUTOSTART", "").strip() not in ("1", "true", "yes"):
        return
    storage = get_storage()
    lease = engine_host.engine_lease_status(storage)
    if lease.get("held"):
        logger.warning(
            f"Not autostarting the agent here: {lease.get('note')}. The page will "
            f"name that engine; use its Stop button to hand the agent over.")
        _CONTROLLER["refused"] = {
            "reason": ("autostart skipped: another engine already owns the agent"),
            "holder": lease, "at": datetime.now(timezone.utc).isoformat(),
        }
        return
    result = _start_agent_in_process(started_by="run_ptai.bat")
    if result.get("started"):
        logger.info("The runner asked for the agent and the console started it "
                    "here; Start/Stop and the interval control the loop")
    else:
        logger.warning(f"Autostart did not start the agent: {result.get('reason')}")


# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------

class ConsoleState:
    """
    The operator's choices, persisted.

    Mode and budgets live in the database, not in a module global, because a
    restart must not silently revert a live account to paper - or worse, a paper
    account to live.
    """

    # Mode and budget are read and written through execution.capital, which is
    # the one definition of both, shared with the trading loop. This class no
    # longer owns a key namespace: a second copy of these key names is how the
    # screen ends up showing $50 authorised while the cycle sizes against $0.

    def __init__(self, storage: Storage):
        self.storage = storage

    @property
    def mode(self) -> str:
        return operator_mode(self.storage)

    @mode.setter
    def mode(self, value: str) -> None:
        set_operator_mode(self.storage, value)

    def budgets(self) -> Dict[str, float]:
        return authorised_budgets(self.storage)


_STORAGE: Dict[str, Any] = {"instance": None, "path": ""}


def get_storage() -> Storage:
    """
    ONE Storage per console process, shared by every request.

    It used to build a new one per HTTP request, and the operator's log shows
    what that looks like from the outside - the page asks for eight panels every
    fifteen seconds:

        2026-09-25 21:51:58.956 | INFO | ptai.storage.db:__init__:232 - Storage initialized at data\ptai.db
        ...the same line 34 times in the next three seconds, per browser tab...

    Each construction re-opens the database, re-runs schema setup and prints a
    line, so the log was mostly about the console talking to itself and the real
    agent messages were lost in it.
    """
    import os
    path = os.getenv("PTAI_DB", "./data/ptai.db")
    if _STORAGE["instance"] is None or _STORAGE["path"] != path:
        _STORAGE["instance"] = Storage(db_path=path)
        _STORAGE["path"] = path
    else:
        # Something may have closed the handle anyway (a helper that closes what
        # it was handed). A closed handle must not turn into a console that
        # answers 500 for every panel until it is restarted, so the connection is
        # checked and rebuilt if needed.
        try:
            _STORAGE["instance"].conn.execute("SELECT 1")
        except Exception:
            _STORAGE["instance"] = Storage(db_path=path)
    return _STORAGE["instance"]


# ----------------------------------------------------------------------
# capital
# ----------------------------------------------------------------------

# A short cache. The console polls every 15 seconds and asks every venue for a
# balance; without this a browser tab left open becomes a sustained load on every
# venue's API, which is how an account gets rate limited for no reason.
_BALANCE_CACHE: Dict[str, Any] = {"at": 0.0, "values": {}}
# Five seconds, not twenty. The page polls the balance because the operator is
# watching the money move during a round; twenty seconds of stale numbers is how
# a deposit or a fill can look like it did nothing. The venues themselves are
# protected by the agent's own rate limiting and the push of this cache.
_BALANCE_TTL_SECONDS = 5.0


def _recorded_balances() -> Dict[str, Dict[str, Any]]:
    """
    The venue health the owning agent last wrote down, for a console that is not
    the process running it.

    Only funded rungs count as available, from the same evidence the agent sizes
    against - the alternative is a page that either invents a balance or claims
    none exists while the agent is trading with one.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        raw = get_storage().get_state("agent.venue_health")
        if not raw:
            return out
        payload = json.loads(raw)
        venues = payload.get("venues") or {}
        at = payload.get("at")
        pid = payload.get("pid")
        role = payload.get("role")
        funded_rungs = {"funded", "trade_permitted"}
        for venue_id, health in venues.items():
            if not isinstance(health, dict):
                continue
            evidence = health.get("evidence") or {}
            reached = str(health.get("readiness") or "")
            out[str(venue_id)] = {
                "available": reached in funded_rungs,
                "balance": float(evidence.get("balance_usd") or 0.0),
                "source": str(evidence.get("balance_provenance") or ""),
                "recorded": True,
                "recorded_at": at,
                "recorded_by": f"the {role} engine (pid {pid})",
                "readiness": reached,
            }
    except Exception as e:  # noqa: BLE001 - the caller falls back to live reads
        logger.debug(f"Could not read the recorded venue health: "
                     f"{type(e).__name__}: {e}")
    return out


async def _venue_balances(agent=None, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Ask each venue what the balance is. Never guesses.

    "Available: False" means the venue did not answer, and downstream that is
    treated as NOT FUNDED. The alternative - defaulting to whatever the operator
    typed into the budget box - is an agent that believes it has money because
    someone filled in a form.
    """
    import time as _time

    now = _time.monotonic()
    if not force and _BALANCE_CACHE["values"] and \
            now - _BALANCE_CACHE["at"] < _BALANCE_TTL_SECONDS:
        return dict(_BALANCE_CACHE["values"])

    balances: Dict[str, Dict[str, Any]] = {}
    registry = getattr(agent, "venue_registry", None)
    adapters = getattr(registry, "adapters", {}) if registry is not None else {}
    if not adapters:
        # The agent is not in THIS process: it is the command window's engine (or
        # nothing is running). Ask the record the agent keeps - the venue health it
        # read in its last cycle - instead of answering "no venue answered", which
        # described the wrong process rather than the agent.
        recorded = _recorded_balances()
        if recorded:
            _BALANCE_CACHE["at"] = _time.monotonic()
            _BALANCE_CACHE["values"] = dict(recorded)
            return recorded
    for venue_id, adapter in adapters.items():
        getter = getattr(adapter, "get_portfolio", None)
        if getter is None:
            continue
        try:
            result = getter()
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            balances[venue_id] = {"available": False, "balance": 0.0,
                                  "reason": f"{type(e).__name__}: {e}"}
            continue
        if isinstance(result, dict):
            balances[venue_id] = {
                "available": bool(result.get("available")),
                "balance": float(result.get("balance") or 0.0),
                "source": result.get("source", ""),
            }
    _BALANCE_CACHE["at"] = _time.monotonic()
    _BALANCE_CACHE["values"] = dict(balances)
    return balances


def _build_plan(state: ConsoleState, storage: Storage,
                balances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    budgets = state.budgets()
    ledger = CapitalLedger(storage=storage)
    plan = ledger.build(mode=state.mode, budgets=budgets,
                        venue_labels={v: r["label"] for v, r in FUNDING_ROUTES.items()},
                        balances=balances)
    return plan.to_dict()


@app.get("/api/console/capital")
async def api_capital() -> JSONResponse:
    storage = get_storage()
    state = ConsoleState(storage)
    balances = await _venue_balances(_agent())
    plan = _build_plan(state, storage, balances)
    plan["balances_read"] = {k: bool(v.get("available")) for k, v in balances.items()}
    plan["session"] = {"mode": state.mode, "budgets": state.budgets()}
    # The hard limits, from the same guard the order path consults. A limit
    # shown here that the dispatcher does not enforce would be worse than no
    # panel at all.
    try:
        from ..risk.money_guard import guard_snapshot

        plan["limits"] = guard_snapshot(
            storage,
            live_bankroll_usd=sum((state.budgets() or {}).values()),
            paper_bankroll_usd=_paper_bankroll(storage))
    except Exception as e:  # noqa: BLE001
        plan["limits"] = {"available": False,
                          "reason": f"{type(e).__name__}: {e}"}
    return JSONResponse(plan)


@app.post("/api/console/mode")
async def api_set_mode(request: Request) -> JSONResponse:
    """
    Switch between paper and live.

    Live mode is REFUSED unless something can actually be traded live: a funded,
    authorised venue. Letting the switch flip into live while nothing is
    verified would present an armed system that cannot fire, and worse, one the
    operator believes is trading.

    Live mode with no signer is refused for a second reason: without a key the
    agent cannot place, cancel or redeem anything, so "live" would mean watching
    a system that does nothing.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json()
    mode = str(body.get("mode", "")).lower()
    if mode not in ("paper", "live"):
        return JSONResponse(status_code=400, content={"error": "mode must be paper or live"})

    if mode == "live":
        balances = await _venue_balances(_agent())
        plan = _build_plan(state, storage, balances)
        blockers = []
        if not plan["live_venues"]:
            if not plan["accounts"]:
                blockers.append("no venue has a budget set")
            for account in plan["accounts"]:
                if not account["balance_is_real"]:
                    blockers.append(
                        f"{account['venue_id']}: the venue balance could not be read "
                        f"(no credentials, or the venue did not answer)")
                elif account["budget_usd"] <= 0:
                    blockers.append(f"{account['venue_id']}: no budget authorised")
        if blockers:
            return JSONResponse(status_code=409, content={
                "error": "live mode refused: nothing is ready to trade live",
                "blockers": blockers,
                "note": ("Paper mode needs none of this and is where the strategy "
                         "has to prove itself first. Staying in paper is not a "
                         "downgrade."),
                "plan": plan,
            })

    state.mode = mode
    return JSONResponse({"mode": mode,
                         "note": ("no order can be sent in paper mode"
                                  if mode == "paper" else
                                  "live mode: capital can be deployed where the "
                                  "venue is funded and authorised")})


@app.post("/api/console/budget")
async def api_set_budget(request: Request) -> JSONResponse:
    """
    Record how much of an account the agent may use.

    This moves no money. It is a permission, not a transfer - the money is
    already at the venue. Setting it above the venue balance is refused.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json()
    venue = str(body.get("venue", "")).lower()
    if venue not in FUNDING_ROUTES:
        return JSONResponse(status_code=400, content={
            "error": f"{venue} has no funding route; only "
                     f"{sorted(FUNDING_ROUTES)} can be funded"})
    try:
        amount = float(body.get("amount", 0))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "amount must be a number"})
    if amount < 0:
        return JSONResponse(status_code=400, content={"error": "amount cannot be negative"})

    balances = await _venue_balances(_agent())
    reported = balances.get(venue) or {}
    if amount > 0 and not reported.get("available"):
        return JSONResponse(status_code=409, content={
            "error": (f"cannot authorise capital at {venue}: the venue's balance "
                      f"was not read, so there is no evidence the account is funded"),
            "reason": reported.get("reason", "venue did not answer"),
            "note": ("Deposit at the venue first, then authorize a budget here. "
                     "A budget is permission to use money that already exists - "
                     "it is not a way to put money in."),
        })
    if amount > float(reported.get("balance") or 0.0) + 1e-9:
        return JSONResponse(status_code=409, content={
            "error": (f"cannot authorise ${amount:.2f}: {venue} reports "
                      f"${float(reported.get('balance') or 0.0):.2f}"),
            "note": "Authorise no more than the account holds.",
        })

    set_authorised_budget(storage, venue, amount)
    plan = _build_plan(state, storage, balances)
    return JSONResponse({"budget": amount, "venue": venue, "plan": plan})


@app.get("/api/console/funding")
async def api_funding(total: float = 50.0) -> JSONResponse:
    """How money gets in, and what this amount will actually buy."""
    return JSONResponse({
        "routes": FUNDING_ROUTES,
        "unfundable": UNFUNDABLE_SMALL,
        "plan": plan_for_budget(total, mode="live"),
        "principle": (
            "There is no account to send money to. The agent trades the venue "
            "account you fund, so capital goes in at the venue - and the agent's "
            "budget is a limit on that account, not a transfer."
        ),
    })


# ----------------------------------------------------------------------
# the running system
# ----------------------------------------------------------------------

_agent_cache: Dict[str, Any] = {}


def _agent():
    """
    The engine THIS process runs, if any.

    Since the console hosts the agent (below), this is normally set and the
    page's balances, rounds and log come from the same engine that is trading -
    not from a copy built for a button press.
    """
    return _agent_cache.get("agent")


# ----------------------------------------------------------------------
# the host: ONE engine, started and stopped from this page
# ----------------------------------------------------------------------
#
# The operator's 2026-09-29 report - "the webui is too disconnected with the
# command line ... things take too long to change or dont change at all. some
# things are even mising" - traced to this process not owning the agent at all.
# `run_ptai.bat` started the agent in a command window, the browser started a
# SECOND engine on every button press, and the two wrote the same keys. Settings
# changed in the page could not reach the loop in the other window, and the
# window's output was invisible here.
#
# So: the browser can start, stop, wake and reconfigure THE agent. When it starts
# one, the loop runs here, in a thread, under the same `run_continuous` the CLI
# uses - one implementation, one lease, one set of keys. Whoever holds the lease
# (this console or the command window) is named on the page.

_CONTROLLER: Dict[str, Any] = {
    "agent": None, "thread": None, "loop": None, "started_at": None,
    "role": None, "state": "idle", "error": None, "started_by": None,
    "refused": None,
}


def _newest_log_file() -> Optional[Dict[str, Any]]:
    """
    The newest file in `logs/`, for the operator who wants the whole history.

    The in-memory buffer is what the page scrolls; the file is what survives a
    restart, and saying which is which prevents the "my log is missing lines"
    confusion that comes from a ring buffer quietly dropping old ones.
    """
    try:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "logs")
        if not os.path.isdir(root):
            return None
        files = [os.path.join(root, f) for f in os.listdir(root)
                 if f.endswith(".log")]
        if not files:
            return None
        newest = max(files, key=os.path.getmtime)
        return {
            "path": newest,
            "name": os.path.basename(newest),
            "size_kb": round(os.path.getsize(newest) / 1024.0, 1),
            "modified_at": datetime.fromtimestamp(
                os.path.getmtime(newest), timezone.utc).isoformat(),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _controller_agent_started() -> bool:
    thread = _CONTROLLER.get("thread")
    return bool(thread is not None and thread.is_alive())


def _controller_status(storage=None) -> Dict[str, Any]:
    """
    The agent's lifecycle as one answer: who owns it, whether this process is
    running it, and what the owner says it is doing.
    """
    storage = storage or get_storage()
    try:
        interval = engine_host.interval_source(storage)
    except Exception as e:  # noqa: BLE001
        interval = {"minutes": None,
                    "source": f"unreadable: {type(e).__name__}: {e}"}
    try:
        lease = engine_host.engine_lease_status(
            storage, interval_min=interval.get("minutes"))
    except Exception as e:  # noqa: BLE001
        lease = {"held": False,
                 "note": f"the lease could not be read: {type(e).__name__}: {e}"}
    here_running = _controller_agent_started()
    return {
        "interval": interval,
        "lease": lease,
        "owned_by_this_console": here_running,
        "console_hosting": here_running,
        "agent_state": _CONTROLLER.get("state"),
        "agent_started_at": _CONTROLLER.get("started_at"),
        "agent_started_by": _CONTROLLER.get("started_by"),
        "agent_error": _CONTROLLER.get("error"),
        "refused": _CONTROLLER.get("refused"),
        "cycle_running": bool(_CONTROLLER.get("cycle_running")),
        "runs_elsewhere": bool(lease.get("held") and not here_running),
        "python": sys.executable,
    }


def _start_agent_in_process(force: bool = False,
                            started_by: str = "the console") -> Dict[str, Any]:
    """
    Start THE agent loop in this process, in its own thread and event loop.

    Thread, not subprocess: the loop needs the adapters, the storage and the
    loguru sink that this process already has, and a thread keeps one window and
    one process for the operator to look at. `run_continuous` claims the lease
    itself, so a second engine anywhere is refused with the holder's name and pid.
    """
    if _controller_agent_started():
        return {"started": False, "already_running": True,
                "reason": "the agent is already running in this console"}

    storage = get_storage()
    interval = engine_host.interval_source(storage)
    claim = engine_host.claim_engine_lease(
        storage, "console", interval_min=interval.get("minutes"), force=force)
    if not claim.get("claimed"):
        holder = claim.get("holder") or {}
        _CONTROLLER["refused"] = {
            "reason": claim.get("reason"), "holder": holder,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        return {"started": False, "reason": claim.get("reason"), "holder": holder}

    try:
        from ..agent.v3_loop import TradingAgentV3
        agent = TradingAgentV3(dry_run=True)
    except Exception as e:  # noqa: BLE001
        engine_host.release_engine_lease(storage,
                                         reason="the console could not start")
        _CONTROLLER["error"] = f"{type(e).__name__}: {e}"
        return {"started": False,
                "reason": f"could not build the engine: {type(e).__name__}: {e}"}

    # A console-started agent is a PAPER engine, checked rather than assumed: the
    # page must never be the way a real order reaches a venue by accident.
    if getattr(agent, "dry_run", None) is not True:
        engine_host.release_engine_lease(storage,
                                         reason="refused: not a paper engine")
        return {"started": False,
                "reason": ("refusing to start: the engine built here was not in "
                           "dry run, so a button could have sent a real order")}

    _CONTROLLER.update({"agent": agent, "loop": None, "started_at": None,
                        "error": None, "refused": None, "started_by": started_by,
                        "role": "console", "state": "starting"})
    _agent_cache["agent"] = agent

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _CONTROLLER["loop"] = loop
        _CONTROLLER["started_at"] = datetime.now(timezone.utc).isoformat()
        _CONTROLLER["state"] = "running"
        logger.info(f"The console is now running the agent (pid {os.getpid()}, "
                    f"every {interval.get('minutes')} min)")
        try:
            loop.run_until_complete(agent.run_continuous(kind="console"))
        except asyncio.CancelledError:
            # CancelledError is a BaseException, so an `except Exception` here
            # would let a stop surface as an unhandled thread traceback - and a
            # deliberate Stop is not an error.
            logger.info("The console-hosted agent was stopped on purpose")
        except Exception as e:  # noqa: BLE001 - the page must be able to report it
            logger.error(f"The console-hosted agent stopped: {type(e).__name__}: {e}")
            _CONTROLLER["error"] = f"{type(e).__name__}: {e}"
        finally:
            _CONTROLLER["state"] = "stopped"
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            # The engine object goes with the loop. Keeping it would leave the
            # next Start with adapters bound to a closed event loop.
            if _agent_cache.get("agent") is agent:
                _agent_cache.pop("agent", None)
            _CONTROLLER["agent"] = None
            _CONTROLLER["loop"] = None
            logger.info("The console-hosted agent has stopped")

    thread = threading.Thread(target=_run, name="ptai-agent", daemon=True)
    _CONTROLLER["thread"] = thread
    thread.start()
    return {"started": True, "role": "console", "pid": os.getpid(),
            "interval": interval.get("minutes")}


def _stop_agent(timeout: float = 20.0) -> Dict[str, Any]:
    """
    Stop the agent and SAY it was stopped.

    An operator stop must be distinguishable from a crash: the lease is released
    with a reason, so the page reads "stopped on purpose at 10:31" instead of
    decaying into "no sign of life", which is what a dead agent looks like.
    """
    storage = get_storage()
    agent = _CONTROLLER.get("agent")
    loop = _CONTROLLER.get("loop")
    if agent is None or loop is None:
        # Not ours: either nobody is running, or the command window owns it. In
        # the second case the lease still names the holder, which the page shows.
        lease = engine_host.engine_lease_status(storage)
        if lease.get("held") and not lease.get("is_self"):
            return {"stopped": False,
                    "reason": (f"the agent is running outside this console "
                               f"({lease.get('kind')} engine, pid {lease.get('pid')} "
                               f"on {lease.get('host')}); close that window to stop it")}
        engine_host.release_engine_lease(storage,
                                         reason="the operator pressed Stop")
        return {"stopped": True,
                "reason": "the agent was not running; the record is clean now"}

    def _cancel() -> None:
        try:
            for task in asyncio.all_tasks(loop):
                loop.call_soon_threadsafe(task.cancel)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not cancel the agent's tasks: {type(e).__name__}: {e}")

    _cancel()
    thread = _CONTROLLER.get("thread")
    if thread is not None:
        thread.join(timeout=timeout)
    stopped = not (thread is not None and thread.is_alive())
    _CONTROLLER["state"] = "stopped" if stopped else "stopping"
    if stopped:
        engine_host.release_engine_lease(storage, reason="the operator pressed Stop")
    else:
        # The cancellation was requested but a call inside the cycle has not
        # returned yet, so say exactly that on the page. Leaving the phase saying
        # "working" would claim the agent is still trading after the operator told
        # it to stop, which is the one thing a Stop button must never do.
        try:
            storage.set_state("agent.phase", json.dumps({
                "phase": "stopping", "label": "stopping",
                "detail": ("you pressed Stop; a call inside the current cycle has "
                           "not returned yet, so the agent stops as soon as it "
                           "does - nothing new starts in the meantime"),
                "at": datetime.now(timezone.utc).isoformat()}))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not write the stopping phase: {e}")
    return {"stopped": stopped, "pid": os.getpid(),
            "reason": ("stopped" if stopped else
                       f"asked to stop; still finishing a long call after "
                       f"{timeout:.0f}s")}


# A short cache for the local model check, for the same reason the balances
# have one: the console polls, and every poll must not become a fresh HTTP
# request to LM Studio. The model list changes when the operator changes it,
# not between two refreshes.
_BRAIN_CACHE: Dict[str, Any] = {"at": 0.0, "value": {}}
# Five seconds, for the same reason the balances have one: the page now polls
# faster while the agent is working, and a setup panel that lags a change looks
# like a panel that ignored it.
_BRAIN_TTL_SECONDS = 5.0


def _brain_status(force: bool = False) -> Dict[str, Any]:
    """
    The local model the agent reasons with.

    Read through the SAME detector the dashboard uses, so the two screens cannot
    disagree about which model is loaded, which one the agent will call, and
    whether it is an R1-style model that would make a cycle take hours.

    Deliberately NOT part of the operator snapshot: the snapshot is storage
    facts that the CLI prints, and it must not start requiring LM Studio to be
    running in order to describe the account.
    """
    import time as _time

    now = _time.monotonic()
    if not force and _BRAIN_CACHE["value"] and \
            now - _BRAIN_CACHE["at"] < _BRAIN_TTL_SECONDS:
        return dict(_BRAIN_CACHE["value"])
    try:
        from ..dashboard import check_lm_studio, read_env_file
        env = read_env_file()
        value = check_lm_studio(env.get("LM_STUDIO_HOST", "http://localhost:1234"),
                                configured_model=env.get("LM_STUDIO_MODEL"))
        value["env_model"] = env.get("LM_STUDIO_MODEL", "local-model")
        value["pinned"] = bool(value["env_model"] and
                               value["env_model"] not in ("local-model", "", "auto"))
        try:
            value["timeout_seconds"] = float(
                env.get("LLM_TIMEOUT_SECONDS") or 180.0)
        except (TypeError, ValueError):
            value["timeout_seconds"] = 180.0
        value["available"] = True
    except Exception as e:
        value = {"available": False, "connected": False, "models": [],
                 "active_model": None, "error": f"{type(e).__name__}: {e}"}
    _BRAIN_CACHE["at"] = _time.monotonic()
    _BRAIN_CACHE["value"] = dict(value)
    return value


@app.get("/api/console/agent")
async def api_agent() -> JSONResponse:
    """
    The agent, in one payload: what it is, what it is doing, what the money is
    doing, and the one thing to do next.

    This is the front page's data. It is the operator snapshot - the same
    payload `ptai status` prints - plus the two things only a local console can
    read: the model on this machine, and (through the copy above) nothing else.
    A blocker list is not duplicated here: the snapshot computes it, and this
    route only adds the one blocker the snapshot cannot see, which is a local
    model that is not answering.
    """
    from ..operator_view import operator_snapshot

    # The storage is shared by every request in this process, so it is NOT
    # closed here: closing it left the whole console answering 500s with
    # "Cannot operate on a closed database" until it was restarted.
    storage = get_storage()
    snapshot = operator_snapshot(storage)

    blockers = list(snapshot.get("blockers") or [])
    brain = _brain_status()
    if brain.get("available") and not brain.get("connected"):
        entry = {
            "id": "brain_offline", "severity": "critical",
            "what": "The agent's local model server is not answering, so it "
                    "cannot reason about fair value.",
            "evidence": f"{brain.get('host', 'http://localhost:1234')}: "
                        f"{brain.get('error') or 'no response'}",
            "clear": "Start LM Studio and load a model with the server on "
                     "(Developer -> Start Server), then refresh this page.",
        }
        # Right behind a dead process: a live agent that cannot think is still
        # not earning.
        at = 1 if blockers and blockers[0].get("id") == "agent_not_running" else 0
        blockers.insert(at, entry)
    elif brain.get("available") and brain.get("is_r1"):
        blockers.append({
            "id": "brain_slow", "severity": "warning",
            "what": "The model the agent will call is an R1-style reasoning "
                    "model, so a cycle takes minutes per market.",
            "evidence": f"active model: {brain.get('active_model')}",
            "clear": "Pin a fast model below, or in Setup -> Brain. The agent "
                     "uses it from its next start.",
        })

    return JSONResponse({
        "generated_at": snapshot.get("generated_at"),
        "mode": snapshot.get("mode"),
        "headline": snapshot.get("headline"),
        "agent": snapshot.get("agent"),
        "capital": snapshot.get("capital"),
        "profit": snapshot.get("profit"),
        "positions": snapshot.get("positions"),
        "venues": snapshot.get("venues"),
        "strategies": snapshot.get("strategies"),
        "risk": snapshot.get("risk"),
        "last_cycle": snapshot.get("last_cycle"),
        "blockers": blockers,
        "next_action": (blockers[0].get("clear") if blockers else None),
        "brain": brain,
        # Which engine is running this agent, and what it will do next. The page
        # paints the pill and the Start/Stop buttons from here, so the buttons
        # cannot disagree with the agent they claim to control.
        "engine": _controller_status(storage),
    })


@app.post("/api/console/brain")
async def api_pin_brain(request: Request) -> JSONResponse:
    """
    Pin the model the agent will call, from the screen.

    The operator asked not to edit .env for this, and they should not have to:
    the model the agent calls is a product decision, not a config file. Only a
    model the server actually reports may be pinned - pinning a name that is not
    loaded would make the agent fall back to "first loaded model" while the
    screen claimed otherwise, which is the exact confusion this endpoint exists
    to end.
    """
    body = await request.json() if await request.body() else {}
    from ..dashboard import write_env_file

    # The agent's wait for ONE model call. Editable here for the same reason the
    # model is: it is a product setting, not a config file. A call that takes
    # nine minutes is what made a 10-minute cycle unable to finish, and the
    # operator should be able to bound it without opening .env.
    if body.get("timeout_seconds") is not None:
        try:
            seconds = float(body["timeout_seconds"])
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={
                "error": "the model call limit has to be a number of seconds"})
        if not (10 <= seconds <= 3600):
            return JSONResponse(status_code=400, content={
                "error": "the model call limit has to be between 10 and 3600 "
                         "seconds",
                "note": "Below 10s no local model finishes a forecast; above an "
                        "hour it can hold a cycle past its interval."})
        write_env_file({"LLM_TIMEOUT_SECONDS": str(int(seconds))})
        _BRAIN_CACHE["at"] = 0.0
        return JSONResponse({
            "timeout_seconds": int(seconds),
            "note": (f"The agent will wait at most {int(seconds)}s for one "
                     f"market's forecast from its next start. A slower model then "
                     f"produces no forecast for that market - which it says in the "
                     f"log - instead of stalling the cycle."),
        })

    model = str(body.get("model") or "").strip()
    if not model:
        return JSONResponse(status_code=400,
                            content={"error": "no model was named"})
    status = _brain_status(force=True)
    if not status.get("connected"):
        return JSONResponse(status_code=409, content={
            "error": "LM Studio is not answering, so there is nothing to pin.",
            "reason": status.get("error"),
            "note": "Start the server in LM Studio, refresh, and pin again.",
        })
    loaded = list(status.get("models") or [])
    if model not in loaded:
        return JSONResponse(status_code=409, content={
            "error": f"{model} is not one of the models this server has loaded.",
            "models": loaded,
            "note": "Pin one of the models in the list, so the agent calls what "
                    "this screen says it calls.",
        })
    write_env_file({"LM_STUDIO_MODEL": model})
    _BRAIN_CACHE["at"] = 0.0
    return JSONResponse({
        "pinned": model,
        "note": ("Pinned. The agent calls exactly this model from its next "
                 "start; a cycle already in flight is using the old one."),
    })


@app.get("/api/console/status")
async def api_status() -> JSONResponse:
    """
    Where the whole loop stands, in the order it runs.

    Each step reports its own state rather than a single green tick, because
    "configured" is not "ready to trade" and the difference is exactly the step
    that is missing. The steps are read from the operator snapshot - the payload
    the CLI prints - so the console cannot describe the agent differently from
    the agent's own report.
    """
    from ..operator_view import operator_snapshot

    storage = get_storage()
    state = ConsoleState(storage)
    agent = _agent()
    control = _controller_status(storage)
    lease = control.get("lease") or {}
    out: Dict[str, Any] = {
        "mode": state.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # "running" is a claim about the AGENT, and the lease is what knows: the
        # loop may be in the command window, in this console, or nowhere. Reporting
        # `agent is not None` made a console that hosts nothing say the agent was
        # dead while it was working one window away.
        "engine": {
            "running": bool(lease.get("held")),
            "owned_by_this_console": bool(control.get("console_hosting")),
            "kind": lease.get("kind"),
            "pid": lease.get("pid"),
            "host": lease.get("host"),
            "note": lease.get("note"),
            "released": bool(lease.get("released")),
            "released_reason": lease.get("released_reason"),
            "interval_min": (control.get("interval") or {}).get("minutes"),
            "interval_source": (control.get("interval") or {}).get("source"),
        },
        "steps": [],
        "storage": {},
    }
    balances = await _venue_balances(agent)
    plan = _build_plan(state, storage, balances)
    try:
        snapshot = operator_snapshot(storage)
    except Exception as e:
        snapshot = {}
        out["snapshot_error"] = f"{type(e).__name__}: {e}"
    answering = sum(1 for b in balances.values() if b.get("available"))

    agt = snapshot.get("agent") or {}
    profit = snapshot.get("profit") or {}
    venues = snapshot.get("venues") or {}
    matrix = venues.get("matrix") or {}
    paper = profit.get("paper") or {}
    out["storage"] = {
        "bankroll": profit.get("bankroll_usd"),
        "total_trades": profit.get("total_trades"),
        "resolved_trades": profit.get("live_resolved_trades"),
        "win_rate": profit.get("win_rate_pct"),
        "open_positions": (snapshot.get("positions") or {}).get("live_count"),
    }
    out["steps"] = [
        {"step": "agent", "label": "Agent running",
         "ok": bool(agt.get("running")),
         "detail": (f"{agt.get('state', 'unknown').replace('_', ' ')} - "
                    f"{agt.get('evidence')}" if agt.get("evidence")
                    else "the agent has not been seen in this database yet")},
        {"step": "capital", "label": ("Paper bankroll" if state.mode == "paper"
                                       else "Capital authorised"),
         "ok": bool(plan["live_venues"]) or state.mode == "paper",
         "detail": ((f"${plan['total_available_usd']:.2f} available across "
                     f"{len(plan['live_venues'])} venue(s)")
                    if plan["live_venues"] else
                    (f"paper mode: ${(_paper_bankroll(storage) or 0.0):.2f} "
                     f"simulated bankroll, no real capital deployed and none "
                     f"needed to keep working")
                    if state.mode == "paper" else
                    "no funded, authorised venue")},
        {"step": "data", "label": "Market data",
         "ok": answering > 0,
         "detail": ((f"{answering} of {len(balances)} venue(s) reported a balance"
                     if balances else "no venue was reachable from this process")
                    + ("" if answering else
                       " - asked, but none answered. A venue that does not answer "
                       "is not a venue with a zero balance."))},
        {"step": "qualification", "label": "Venue qualified",
         "ok": bool(venues.get("best_validated_venue")),
         "detail": (f"{venues.get('best_validated_venue')} passed the gate on "
                    f"real resolved trades"
                    if venues.get("best_validated_venue") else
                    (f"{int(matrix.get('cells_with_enough_evidence') or 0)} of "
                     f"{int(matrix.get('cells') or 0)} venue x strategy x market "
                     f"combination(s) have enough evidence; the gate needs real "
                     f"fills and resolved outcomes, not a score. Nothing is "
                     f"qualified on a fresh install, and that is the gate "
                     f"working."))},
        {"step": "evidence", "label": "Paper evidence",
         "ok": bool(int(paper.get("settled_trades") or 0)),
         "detail": (f"{int(paper.get('settled_trades') or 0)} settled simulated "
                    f"trade(s), ${float(paper.get('net_pnl') or 0.0):+.2f}"
                    if int(paper.get("settled_trades") or 0)
                    else "no simulated trade has settled yet")},
    ]
    out["capital"] = plan
    return JSONResponse(out)


@app.get("/api/console/results")
async def api_results(limit: int = 50) -> JSONResponse:
    """
    What the system has actually achieved.

    These are the eight figures the operator is owed: equity, realised P&L, free
    capital, reserved capital, net return, drawdown, the best validated strategy
    and venue, and the risk state. A figure that cannot be computed is reported
    as null with a reason, never as a zero that reads like a result.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    out: Dict[str, Any] = {"mode": state.mode, "figures": {}, "unavailable": {}}

    try:
        perf = storage.get_performance_summary()
    except Exception as e:
        perf = {}
        out["unavailable"]["performance"] = f"{type(e).__name__}: {e}"

    balances = await _venue_balances(_agent())
    plan = _build_plan(state, storage, balances)

    def figure(key: str, value: Any, reason: str = ""):
        if value is None:
            out["unavailable"][key] = reason or "not computed"
        else:
            out["figures"][key] = value

    figure("equity_usd", perf.get("bankroll"), "no bankroll recorded")
    figure("realised_pnl_usd", perf.get("net_pnl"),
           "no resolved trades, so no realised P&L exists yet")
    figure("free_capital_usd", plan["total_available_usd"])
    figure("reserved_capital_usd", plan["total_reserved_usd"])
    figure("in_positions_usd", plan["total_in_positions_usd"])
    figure("open_positions", storage.count_open_positions())
    figure("win_rate", perf.get("win_rate"), "needs resolved trades")
    figure("total_trades", perf.get("total_trades"))
    figure("resolved_trades", perf.get("resolved_trades"))

    # Return and drawdown need a resolved history. On a fresh account they are
    # not zero, they are undefined - and presenting 0.0% as a return is a claim
    # about performance that has not happened.
    out["unavailable"].setdefault(
        "net_return_30d_pct",
        "undefined until there is a resolved-trade history spanning 30 days")
    out["unavailable"].setdefault(
        "max_drawdown_pct",
        "undefined until there is an equity curve to draw down")
    out["unavailable"].setdefault(
        "best_validated_strategy",
        "no strategy has a validated out-of-sample sample yet")
    out["unavailable"].setdefault(
        "best_validated_venue",
        "no venue has passed the 100-trade qualification gate yet")

    out["risk_state"] = {
        "mode": state.mode,
        "live_deployable": plan["is_deployable"],
        "venues_live": plan["live_venues"],
        "warnings": plan["warnings"],
    }
    try:
        trades = storage.get_recent_trades(limit=limit)
        for trade in trades:
            for key in ("pnl", "position_size_usd", "market_price", "edge"):
                if isinstance(trade.get(key), (int, float)):
                    trade[key] = round(trade[key], 4)
        out["recent_trades"] = trades
    except Exception as e:
        out["recent_trades"] = []
        out["unavailable"]["recent_trades"] = f"{type(e).__name__}: {e}"
    return JSONResponse(out)


@app.get("/api/console/orders")
async def api_orders() -> JSONResponse:
    """Working orders and what they are holding. Live against the venue."""
    storage = get_storage()
    agent = _agent()
    local = storage.get_open_orders()
    out: Dict[str, Any] = {
        "local_open_orders": len(local),
        "local_reserved_usd": round(storage.resting_capital_usd(), 4),
        "orders": local,
        "venue_view": None,
        "note": ("A working order has bought nothing, so it is not a position. "
                 "It has committed cash, so it is not available either."),
    }
    registry = getattr(agent, "venue_registry", None)
    adapters = getattr(registry, "adapters", {}) if registry is not None else {}
    adapter = adapters.get("polymarket")
    getter = getattr(adapter, "get_open_orders", None)
    if getter is not None:
        try:
            result = getter()
            if asyncio.iscoroutine(result):
                result = await result
            out["venue_view"] = result
        except Exception as e:
            out["venue_view"] = {"available": False,
                                 "reason": f"{type(e).__name__}: {e}"}
    else:
        out["venue_view"] = {"available": False,
                             "reason": "no adapter with order reads is loaded"}
    return JSONResponse(out)


def _paper_bankroll(storage) -> Optional[float]:
    """The simulated bankroll, or None. Never a stand-in number."""
    try:
        value = storage.get_paper_performance().get("bankroll")
        return round(float(value), 2) if value is not None else None
    except Exception:
        return None


@app.get("/api/console/venue")
async def api_venue() -> JSONResponse:
    """
    Which venue the agent is using right now, and why.

    The agent holds live capital at exactly one venue, because money cannot be
    moved between venues by the agent - a withdrawal and a deposit are the
    operator's actions. Everything else is scanned and paper-traded for free.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    agent = _agent()
    balances = await _venue_balances(agent)
    plan = _build_plan(state, storage, balances)

    qualified = list(getattr(agent, "_last_qualified_venue_ids", []) or [])
    labels = {v: r["label"] for v, r in FUNDING_ROUTES.items()}
    adapters = getattr(getattr(agent, "venue_registry", None), "adapters", {}) or {}
    selector = VenueSelector(storage=storage, funding_routes=FUNDING_ROUTES)
    # The venue list, in order of how much this process knows:
    #
    #   1. the running agent's registry (same process as the console),
    #   2. the inventory the agent recorded (the normal case - the runner starts
    #      the agent and the console as SEPARATE processes, so this console has
    #      no registry of its own),
    #   3. the fundable venues plus anything that has ever traded.
    #
    # Step 3 is what used to be the only fallback, and it is why the panel showed
    # two venues out of nineteen: an operator could not tell whether the other
    # seventeen were broken, missing, or merely not listed.
    inventory = load_inventory(storage)
    inv_venues = inventory.get("venues") or {}
    labels.update({vid: row["label"] for vid, row in inv_venues.items()
                   if row.get("label")})
    if adapters:
        registered = sorted(adapters)
    elif inv_venues:
        registered = sorted(inv_venues)
    else:
        registered = sorted(set(labels) | set(selector.known_venues()))
    assessments = selector.assess(
        registered,
        accounts=plan.get("accounts", []),
        qualified_ids=qualified,
        labels=labels,
    )
    selection = selector.select(assessments, total_budget_usd=plan.get("total_budget_usd") or 0.0)

    return JSONResponse({
        "mode": state.mode,
        "selection": selection.to_dict(),
        "assessments": [a.to_dict() for a in assessments],
        "ranking_basis": selection.ranking_basis,
        "min_sample_for_evidence": MIN_SAMPLE_FOR_EVIDENCE,
        "one_live_venue_cap": True,
        "balances_read": sum(1 for b in balances.values() if b.get("available")),
        "venues_asked": len(balances),
        # Distinguishes "every venue stayed silent" from "nothing was asked".
        # They read the same on a dashboard and mean opposite things.
        "engine_running": agent is not None,
        # What the simulation is running on. In paper mode the money is imaginary
        # and the bankroll is the one the operator is watching, so the panel can
        # say what is standing in for capital instead of leaving a blank.
        "paper_bankroll_usd": _paper_bankroll(storage),
        # Every venue PTAI knows about and what it can do, so "can I run this
        # one too?" has an answer on the page rather than only in the code.
        "inventory": inventory,
        # What out-of-sample validation says about the rules that are choosing
        # these trades. It can refuse a rule; it can never qualify a venue.
        # ...together with the bench: what the agent DOES about a refused
        # rule. A verdict the operator cannot see acted on reads as decoration.
        "validation": validation_block(storage),
    })


def _data_dir() -> str:
    """The data folder this console's database lives in - where the vault is."""
    try:
        from pathlib import Path as _Path

        return str(_Path(get_storage().db_path).parent)
    except Exception:  # noqa: BLE001
        return "./data"


@app.get("/api/console/logins")
async def api_logins() -> JSONResponse:
    """
    Every login PTAI can use, what each one unlocks, and where it comes from.

    The operator's ask was plain - "my logins need to be saved somewhere so I
    don't always have to log in when the system is running automatically" - and
    before this there was no way to put a credential into the product at all.
    Masked values only; a secret never leaves this process in the clear.
    """
    return JSONResponse(credential_store.describe_all(_data_dir()))


@app.post("/api/console/logins")
async def api_save_login(request: Request) -> JSONResponse:
    """Save one login. A blank required field is refused, not stored."""
    body = await request.json() if await request.body() else {}
    tool = str(body.get("tool") or "").strip()
    if not tool:
        return JSONResponse(status_code=400,
                            content={"error": "no tool was named"})
    result = credential_store.save(tool, body.get("fields") or {}, _data_dir())
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return JSONResponse(result)


@app.post("/api/console/logins/forget")
async def api_forget_login(request: Request) -> JSONResponse:
    body = await request.json() if await request.body() else {}
    tool = str(body.get("tool") or "").strip()
    result = credential_store.forget(tool, _data_dir())
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)


def _known_venue_rows(storage, inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Every venue PTAI knows about, without needing the agent to have started.

    Same order of knowledge as the venue panel above: the running agent's
    registry, then the recorded inventory, then the fundable venues plus
    anything that has ever traded. The switch list used ONLY the recorded
    inventory, so on a fresh install - which is exactly when an operator wants
    to switch off the eleven venues with no client written yet - it was empty
    and read as "there is nothing to switch". The detailed row (what it can do,
    whether it needs a login) still comes from the inventory when it exists;
    without it the row says so rather than guessing.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    inv_venues = inventory.get("venues") or {}
    for venue_id, row in inv_venues.items():
        rows[venue_id] = dict(row)

    agent = _agent()
    adapters = getattr(getattr(agent, "venue_registry", None), "adapters", {}) or {}
    if adapters:
        known = list(adapters)
    elif inv_venues:
        known = list(inv_venues)
    else:
        selector = VenueSelector(storage=storage, funding_routes=FUNDING_ROUTES)
        known = sorted(set(FUNDING_ROUTES) | set(selector.known_venues()))

    for venue_id in known:
        if venue_id in rows:
            continue
        route = FUNDING_ROUTES.get(venue_id) or {}
        rows[venue_id] = {
            "venue_id": venue_id,
            "label": route.get("label") or venue_id,
            "use": None,
            "what_it_needs": None,
            "detail_known": False,
            "reads_live_markets_now": None,
            "can_hold_real_money": None,
            "needs_credentials": None,
        }
    for row in rows.values():
        row.setdefault("detail_known", True)
    return [rows[vid] for vid in sorted(rows)]


@app.get("/api/console/venues/enabled")
async def api_venue_switches() -> JSONResponse:
    """
    Which venues the agent may use, and what each one can do.

    Nineteen adapters were scanned whether the operator wanted them or not, with
    no switch anywhere. This is that switch: an explicit choice per venue, saved,
    and read by the agent on its next cycle.
    """
    storage = get_storage()
    inventory = load_inventory(storage)
    saved = venue_switches.load(storage)
    disabled = set(saved["disabled"])
    rows = [{**row, "enabled": row["venue_id"] not in disabled}
            for row in _known_venue_rows(storage, inventory)]
    return JSONResponse({
        "venues": rows, "disabled": sorted(disabled),
        "updated_at": saved.get("updated_at", ""),
        "note": ("A venue switched off is not asked for markets and not traded. "
                 "Nothing else changes: a venue switched on still has to be "
                 "funded, qualified and inside the loss limits to touch real "
                 "money."),
    })


@app.post("/api/console/venues/enabled")
async def api_set_venue_switch(request: Request) -> JSONResponse:
    body = await request.json() if await request.body() else {}
    # `venue` and `id` are accepted as well as `venue_id`: a caller that guessed
    # the wrong one used to get "no venue named" and no way to tell which name
    # was expected.
    venue_id = str(body.get("venue_id") or body.get("venue")
                   or body.get("id") or "").strip()
    if not venue_id:
        return JSONResponse(status_code=400, content={
            "error": "no venue named - send {'venue_id': '<id>', 'enabled': true|false}"})
    enabled = bool(body.get("enabled", True))
    result = venue_switches.set_enabled(get_storage(), venue_id, enabled)
    if not result.get("ok"):
        return JSONResponse(status_code=500, content=result)
    return JSONResponse({
        **result,
        "venue_id": venue_id, "enabled": enabled,
        "takes_effect": ("the agent reads this at the start of its next cycle - "
                         "no restart needed"),
    })


@app.get("/api/console/forecast")
async def api_forecast() -> JSONResponse:
    """
    Why the fair value is what it is, component by component.

    The LLM said 65%, the ensemble said 58%, and nothing in the console could
    say what happened in between or which inputs were even present. This is that
    answer: the chain for each priced market, the cheap-screen decision that
    decided which markets got model time, and whether the base-rate component had
    real counted frequencies behind it.
    """
    storage = get_storage()
    evidence: Dict[str, Any] = {}
    try:
        raw = storage.get_state("intelligence.last_forecast_evidence")
        if raw:
            evidence = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"available": False,
                             "reason": f"could not read the last cycle's "
                                       f"evidence: {type(e).__name__}: {e}"})
    if not evidence:
        return JSONResponse({
            "available": False,
            "reason": ("no cycle has stored forecast evidence yet - run one cycle "
                       "and this fills in"),
        })
    return JSONResponse({"available": True, **evidence})


@app.get("/api/console/sports")
async def api_sports() -> JSONResponse:
    """
    The sports lane: what it priced, what it placed, what it settled.

    The lane could price a full match card and had nowhere to put the bets, so
    "3 executable" was where the story ended. This is the rest of it.
    """
    storage = get_storage()
    try:
        from ..betting.positions import SportsBook
        from ..betting.ratings import RatingsBook

        book = SportsBook(storage=storage)
        return JSONResponse({
            "available": True,
            "summary": book.summary(),
            "open": book.open_bets()[:20],
            "recent": book.recent(limit=20),
            "ratings": RatingsBook(storage=storage).snapshot(),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"available": False,
                             "reason": f"{type(e).__name__}: {e}"})


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """
    Anything at all, as something `json.dumps` can send.

    The run-cycle reply carries the cycle's own result under "detail", and a
    cycle result contains live objects - `CombinatorialGroup`, `Market`, source
    and mode enums. They are not JSON-serialisable, so `JSONResponse` raised
    inside the response and the operator got HTTP 500 for a round that HAD run:
    the page showed an error box, never received the round's bankroll, and the
    agent pill was left showing whatever it showed before the button was
    pressed. Found while reproducing the 2026-09-28 report, on a cycle that
    found 6 markets and built arbitrage groups.

    A reply that describes work that happened must not be able to fail to send,
    so this is total: unknown objects fall back to their text, and non-finite
    floats become null (a NaN would break the page's own JSON.parse instead).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if _depth >= 6:
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(getattr(value, "value", None) or str(value), _depth + 1)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth + 1) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe(dataclasses.asdict(value), _depth + 1)
        except Exception:  # noqa: BLE001 - fall through to text, never raise
            return str(value)
    for attr in ("to_dict", "model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _json_safe(method(), _depth + 1)
            except Exception:  # noqa: BLE001
                continue
    return str(value)


def _round_history(agent: Any, limit: int = 10) -> Dict[str, Any]:
    """
    The completed rounds, from the engine when it can speak and from the record
    when it cannot.

    Both paths read the same store, so the console can never show a different
    history from the one the agent scores itself against.
    """
    if agent is not None:
        try:
            history = agent.round_history(limit=limit)
            if isinstance(history, dict) and history.get("rounds") is not None:
                return history
        except Exception as e:  # noqa: BLE001 - a report must not break the page
            logger.warning(f"Round history unavailable from the engine: "
                           f"{type(e).__name__}: {e}")
    rounds = load_rounds(get_storage(), limit=limit)
    return {"rounds": rounds, "summary": round_history_summary(rounds)}


@app.get("/api/console/rounds")
async def api_rounds(limit: int = 10) -> JSONResponse:
    """
    Completed rounds, newest first, with the score across them.

    The operator asked for a bankroll at the END of a run. A run that returns
    only what it *did* leaves that question unanswered until somebody reads the
    database, so this is the route the console polls - and the run-cycle reply
    carries the round it just completed, from the same record.
    """
    try:
        wanted = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        wanted = 10
    history = _round_history(_agent(), wanted)
    return JSONResponse({
        "rounds": history.get("rounds", []),
        "summary": history.get("summary", {}),
    })


@app.get("/api/console/agent-control")
async def api_agent_control() -> JSONResponse:
    """
    Who owns the agent, and can this page start, stop or wake it.

    The page's buttons read their own truth from here. Before this, nothing in
    the console could start or stop the agent at all: its lifecycle was a
    console window, and the web page could only ask a second engine of its own
    to do work, which is where the operator's "disconnected" came from.
    """
    return JSONResponse(_json_safe(_controller_status()))


@app.post("/api/console/agent-control")
async def api_agent_control_action(request: Request) -> JSONResponse:
    """start / stop / run-now, from the page."""
    body = await request.json() if await request.body() else {}
    action = str(body.get("action") or "").strip().lower()
    interval_min = body.get("interval_min")
    if action == "start":
        if interval_min not in (None, ""):
            try:
                engine_host.set_operator_interval_minutes(get_storage(), interval_min)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
        result = _start_agent_in_process(started_by="the console")
        status = 200 if result.get("started") else 409
        return JSONResponse(status_code=status,
                            content=_json_safe({**result,
                                                "control": _controller_status()}))
    if action == "stop":
        result = _stop_agent()
        return JSONResponse(status_code=200 if result.get("stopped") else 409,
                            content=_json_safe({**result,
                                                "control": _controller_status()}))
    if action in ("run-now", "run_now", "round"):
        agent = _agent()
        if agent is not None and _controller_agent_started():
            woke = agent.request_immediate_cycle()
            return JSONResponse(status_code=200, content=_json_safe({
                "status": "requested" if woke else "no-loop",
                "woke": woke,
                "message": ("the agent will start its next cycle now, in the "
                            "engine that is already running"
                            if woke else
                            "the agent is running here but its loop is not "
                            "waiting; it will pick the request up shortly"),
                "control": _controller_status(),
            }))
        control = _controller_status()
        lease = control.get("lease") or {}
        if lease.get("held"):
            return JSONResponse(status_code=409, content=_json_safe({
                "error": (f"the agent is running elsewhere: the {lease.get('kind')} "
                          f"engine, pid {lease.get('pid')} on {lease.get('host')} "
                          f"(alive {lease.get('age_seconds')}s ago). It is doing the "
                          f"work - there is no need to start a second one. To run "
                          f"rounds from this page, stop that engine and press Start."),
                "control": control,
            }))
        # Nobody owns the agent: starting it IS the round the operator asked for.
        result = _start_agent_in_process(started_by="Run a round")
        return JSONResponse(status_code=200 if result.get("started") else 409,
                            content=_json_safe({**result,
                                                "status": "started",
                                                "control": _controller_status()}))
    return JSONResponse(status_code=400, content={
        "error": f"unknown action {action!r}; use start, stop or run-now"})


@app.post("/api/console/agent/interval")
async def api_agent_interval(request: Request) -> JSONResponse:
    """
    Set the cycle interval for the RUNNING agent.

    It used to be a CLI flag (`--interval 10` in the .bat), fixed at process
    start, which is why a change here "did not change anything" until a restart.
    The loop re-reads this value before every wait, so a new interval applies to
    the next wait - and a shortened interval interrupts the wait that is already
    in progress (within a minute) instead of being ignored until it expires.
    """
    storage = get_storage()
    body = await request.json() if await request.body() else {}
    raw = body.get("minutes")
    if raw in (None, ""):
        return JSONResponse(status_code=400, content={"error": "minutes is required"})
    try:
        value = engine_host.set_operator_interval_minutes(storage, raw)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    agent = _agent()
    applied = None
    if agent is not None:
        applied = agent.request_immediate_cycle()
    control = _controller_status(storage)
    return JSONResponse(_json_safe({
        "interval": control["interval"],
        "set": value,
        "applied_to_running_agent": bool(applied),
        "message": (f"the agent now runs every {value} minute(s); the wait in "
                    f"progress was cut short so it starts applying now"),
        "control": control,
    }))


@app.get("/api/console/logs")
async def api_logs(after: int = 0, limit: int = 200) -> JSONResponse:
    """
    The agent's log, where the agent runs.

    The operator had to keep a second window open to read what the agent was
    doing - and when the page ran a cycle of its own, that output went nowhere he
    could see. This is the console process's own loguru buffer: the same lines, in
    the browser, including the process and pid in the page's own header so it is
    unambiguous WHICH engine is talking.
    """
    payload = live_log.tail(after=after, limit=limit)
    payload.update({
        "pid": os.getpid(),
        "role": (_CONTROLLER.get("role") or "console"),
        "log_file": _newest_log_file(),
    })
    return JSONResponse(_json_safe(payload))


@app.post("/api/console/run-cycle")
async def api_run_cycle(request: Request) -> JSONResponse:
    """
    "Run a round" - on THE agent, not on a copy of it.

    This route used to build a second engine inside the web process whenever the
    agent was not already here, then run a full round in it. That engine wrote to
    the same liveness keys and round history as the loop in the command window, so
    two engines appeared as one and neither could see the other - the operator's
    "disconnected", "takes too long", "some things are missing".

    Now the button means what it says:

      * the agent runs HERE  -> wake the one loop; the round starts in the engine
        that owns the round history, the phase keys and the log;
      * the agent runs ELSEWHERE (the command window) -> refuse, and name the
        process that is already doing the work. A second engine is never the
        answer;
      * nobody runs it -> start the agent here and let it take the first cycle.

    A live round is refused unless a funded, authorised venue is ready, exactly as
    before: a first-time operator must not open a real position from a preview
    button.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json() if await request.body() else {}
    requested_mode = str(body.get("mode") or state.mode).lower()

    if requested_mode == "live":
        balances = await _venue_balances(_agent())
        plan = _build_plan(state, storage, balances)
        if not plan["is_deployable"]:
            return JSONResponse(status_code=409, content={
                "error": "live cycle refused: no funded, authorised venue is ready",
                "warnings": plan["warnings"],
            })

    control = _controller_status(storage)
    lease = control.get("lease") or {}

    if _controller_agent_started():
        if _CONTROLLER.get("cycle_running"):
            return JSONResponse(status_code=202, content=_json_safe({
                "status": "running",
                "message": "a round is already running; the page will show it as "
                           "it finishes",
                "control": control,
            }))
        agent = _agent()
        woke = bool(agent is not None and agent.request_immediate_cycle())
        logger.info("The operator asked for a round; the running agent was woken"
                    if woke else
                    "The operator asked for a round; the agent is between cycles")
        return JSONResponse(status_code=202, content=_json_safe({
            "status": "requested",
            "message": ("the agent is running and will start its next round now - "
                        "watch the log and the round card; a full round takes real "
                        "time" if woke else
                        "the agent is running and will start a round immediately"),
            "control": control,
        }))

    if lease.get("held"):
        return JSONResponse(status_code=409, content=_json_safe({
            "error": (f"the agent is already running in the {lease.get('kind')} "
                      f"engine (pid {lease.get('pid')} on {lease.get('host')}, alive "
                      f"{lease.get('age_seconds')}s ago). It is doing the work; a "
                      f"second engine on the same database is exactly what made the "
                      f"page disagree with the command line. Press Stop there (or "
                      f"close that window) and this button will start the agent here."),
            "control": control,
        }))

    result = _start_agent_in_process(started_by="Run a round")
    if not result.get("started"):
        return JSONResponse(status_code=409, content=_json_safe({
            "error": result.get("reason"), "control": _controller_status()}))
    logger.info("The operator asked for a round; the console started the agent here")
    return JSONResponse(status_code=200, content=_json_safe({
        "status": "started",
        "message": ("the agent was not running, so it was started here and is "
                    "taking a round now - the round card and the log will fill in"),
        "control": _controller_status(),
    }))


@app.get("/", response_class=HTMLResponse)
async def console() -> str:
    return CONSOLE_HTML


CONSOLE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PTAI Console</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121821; --panel2:#0f151d; --line:#243040;
  --text:#e6edf6; --dim:#8b9bb0; --dimmer:#5a6a7e;
  --green:#2ecc71; --green-dim:#16351f;
  --red:#ef4444; --red-dim:#3a1616;
  --amber:#f5a524; --amber-dim:#3a2c10;
  --blue:#4c8dff; --paper:#a78bfa;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--line);
  padding:14px 22px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{font-weight:700;font-size:17px;letter-spacing:.4px}
.mode{display:flex;align-items:center;gap:9px;padding:6px 14px;border-radius:999px;font-weight:700;
  font-size:13px;letter-spacing:.9px}
.mode.paper{background:rgba(167,139,250,.13);color:var(--paper);border:1px solid rgba(167,139,250,.4)}
.mode.live{background:var(--red-dim);color:#ff8080;border:1px solid rgba(239,68,68,.55)}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor}
.dot.live{animation:pulse 1.4s infinite}
@keyframes pulse{50%{opacity:.25}}
.spacer{flex:1}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;
  padding:8px 15px;font-size:13px;cursor:pointer;font-weight:500}
button:hover{border-color:var(--blue);color:#fff}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
button.primary:hover{opacity:.9}
button.danger{background:var(--red-dim);border-color:rgba(239,68,68,.5);color:#ff9b9b}
button:disabled{opacity:.45;cursor:not-allowed}
main{max-width:1240px;margin:0 auto;padding:22px}
.grid{display:grid;gap:16px}
.cols-4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.cols-2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:17px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:1.1px;color:var(--dim);
  margin-bottom:14px;font-weight:600}
.kpi .v{font-size:26px;font-weight:700;letter-spacing:-.5px}
.kpi .k{font-size:12px;color:var(--dim);margin-bottom:6px}
.kpi .sub{font-size:11.5px;color:var(--dimmer);margin-top:5px}
.pos{color:var(--green)} .neg{color:var(--red)} .warn{color:var(--amber)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.7px;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid rgba(36,48,64,.5)}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
.pill.ok{background:var(--green-dim);color:var(--green)}
.pill.no{background:var(--red-dim);color:#ff9b9b}
.pill.wait{background:var(--amber-dim);color:var(--amber)}
.pill.dim{background:#1b2430;color:var(--dim)}
.empty{color:var(--dimmer);font-size:13px;padding:12px 0}
.step{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)}
.step:last-child{border-bottom:none}
.step .mark{width:20px;height:20px;border-radius:50%;flex:0 0 20px;display:grid;place-items:center;
  font-size:11px;font-weight:700;margin-top:1px}
.step .mark.ok{background:var(--green-dim);color:var(--green)}
.step .mark.no{background:var(--amber-dim);color:var(--amber)}
.step .lbl{font-weight:600;font-size:13.5px}
.step .det{color:var(--dim);font-size:12.5px;margin-top:2px}
.modebox{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.modeopt{flex:1;min-width:150px;border:1.5px solid var(--line);border-radius:10px;padding:13px;
  cursor:pointer;background:var(--panel2)}
.modeopt.sel-paper{border-color:var(--paper);background:rgba(167,139,250,.09)}
.modeopt.sel-live{border-color:var(--red);background:rgba(239,68,68,.09)}
.modeopt .t{font-weight:700;font-size:14px;margin-bottom:4px}
.modeopt .d{font-size:12px;color:var(--dim);line-height:1.4}
input{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:7px;
  padding:9px 11px;font-size:13px;width:100%}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
.note{font-size:12.5px;color:var(--dim);line-height:1.55}
.warnbox{background:var(--amber-dim);border:1px solid rgba(245,165,36,.35);border-radius:9px;
  padding:11px 13px;font-size:12.5px;color:#f7c46c;margin-bottom:12px}
.errbox{background:var(--red-dim);border:1px solid rgba(239,68,68,.4);border-radius:9px;
  padding:11px 13px;font-size:12.5px;color:#ff9b9b;margin-bottom:12px}
ol{margin:9px 0 0 18px} ol li{margin-bottom:7px;font-size:13px;line-height:1.5}
details{margin-top:10px} summary{cursor:pointer;color:var(--blue);font-size:13px}
.bar{height:7px;border-radius:4px;background:var(--panel2);overflow:hidden;margin-top:9px;display:flex}
.bar i{display:block;height:100%}
.tabs{display:flex;gap:6px;margin-top:11px;flex-wrap:wrap}
.tab{padding:7px 15px;border-radius:8px;font-size:13px;cursor:pointer;color:var(--dim);
  border:1px solid transparent}
.tab.on{background:var(--panel);border-color:var(--line);color:var(--text);font-weight:600}
.hero-pill{display:inline-block;font-size:12px;font-weight:700;letter-spacing:1.2px;
  padding:4px 11px;border-radius:999px;background:var(--panel2);color:var(--dim);
  border:1px solid var(--line);margin-bottom:9px}
.hero-pill.ok{background:var(--green-dim);color:var(--green);border-color:rgba(46,204,113,.4)}
.hero-pill.wait{background:var(--amber-dim);color:var(--amber);border-color:rgba(245,165,36,.35)}
.hero-pill.no{background:var(--red-dim);color:#ff9b9b;border-color:rgba(239,68,68,.4)}
.headline{font-size:19px;font-weight:650;line-height:1.35;letter-spacing:-.2px}
.blocker{padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)}
.blocker:last-child{border-bottom:none}
.blocker .clear{color:var(--blue);font-size:12.5px;line-height:1.5}
code{background:var(--panel2);padding:1.5px 6px;border-radius:5px;font-size:12.5px}
.foot{color:var(--dimmer);font-size:11.5px;text-align:center;padding:26px 0 12px}
.section{margin-top:26px}
.section-head{padding:16px 0 2px;border-top:1px solid var(--line)}
.section-head h1{font-size:15px;font-weight:700;letter-spacing:.2px}
.section-head p{color:var(--dim);font-size:12.5px;margin-top:3px}
section[id]{scroll-margin-top:132px}
</style>
</head>
<body>
<header>
  <div class="brand">PTAI <span style="color:var(--dim);font-weight:500">&middot; one agent, trading your money</span></div>
  <div id="modeBadge" class="mode paper"><span class="dot"></span><span id="modeText">PAPER</span></div>
  <div id="agentPill" class="pill dim">checking&hellip;</div>
  <div class="mono" style="font-size:12.5px;color:var(--dim)" id="hdrCapital"></div>
  <div class="spacer"></div>
  <button onclick="loadAll()">Refresh</button>
  <button id="startBtn" onclick="agentAction('start')">Start the agent</button>
  <button id="stopBtn" onclick="agentAction('stop')">Stop</button>
  <button id="runBtn" class="primary" onclick="runCycle()">Run a round now</button>
  <!--
    Which process is running the agent, and every setting that reaches it. The
    operator's report was that the page and the command line were two systems;
    this line exists so they are never in doubt about which one is doing the work.
  -->
  <div id="engineLine" class="mono"
       style="flex-basis:100%;font-size:12px;color:var(--dim);margin-top:6px">
    reading which engine owns the agent&hellip;
  </div>
  <!--
    One page, jump links. The sections are all on this page and all visible;
    these only scroll to them, so nothing can be hidden from the operator by a
    navigation state they forgot they set. The active one is highlighted as the
    page scrolls.
  -->
  <nav class="tabs" style="flex-basis:100%">
    <div class="tab on" data-tab="agent" onclick="goTo('agent')">Agent</div>
    <div class="tab" data-tab="money" onclick="goTo('money')">Money</div>
    <div class="tab" data-tab="venue" onclick="goTo('venue')">Venue</div>
    <div class="tab" data-tab="orders" onclick="goTo('orders')">Orders</div>
    <div class="tab" data-tab="activity" onclick="goTo('activity')">Activity</div>
    <div class="tab" data-tab="setup" onclick="goTo('setup')">Setup</div>
  </nav>
</header>

<main>
  <!-- AGENT: the state of the one agent, in the order the operator asks -->
  <section id="tab-agent">
    <div class="section-head">
      <h1>Agent</h1>
      <p>Is it running, what is it doing right now, and what is the money doing.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:1;min-width:280px">
          <div id="agentState" class="hero-pill">reading the agent&hellip;</div>
          <div id="agentHeadline" class="headline">&nbsp;</div>
          <div id="agentDoing" class="note" style="margin-top:9px"></div>
        </div>
        <div id="agentVitals" class="note mono"
             style="font-size:12px;min-width:210px;text-align:right"></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>The round &mdash; what the bankroll did</h2>
      <div class="note" style="margin-bottom:9px">
        A round researches the market, picks the predictions worth betting,
        researches those further to choose which ones to bet, bets them with the
        paper currency, and ends with a figure: the account before it and the
        account after it. A round with nothing priced reports no number rather
        than a flat one.
      </div>
      <div id="round"></div>
    </div>

    <!--
      The loop's own settings, in one row, applied to the RUNNING agent. The
      interval used to live in the .bat (`--interval 10`) and was fixed for the
      life of the process, so changing it here could not have any effect.
    -->
    <div class="card" style="margin-top:16px">
      <h2>How often it works</h2>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <label class="note" for="intervalMin">A cycle every</label>
        <input id="intervalMin" type="number" min="1" max="1440" step="1"
               style="width:90px" placeholder="10">
        <span class="note">minutes</span>
        <button onclick="setIntervalMin()">Apply to the running agent</button>
        <span class="note" id="intervalNote">
          A new interval is picked up before the next wait - and a shortened one
          cuts the wait that is already running.
        </span>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>The agent's own log</h2>
      <div class="note" style="margin-bottom:9px">
        The same lines the command-line window shows, from the process that is
        running the agent. <span id="logWhere"></span>
      </div>
      <pre id="agentLog" class="mono"
           style="max-height:340px;overflow:auto;font-size:11.5px;line-height:1.45;
                  background:#0b0f13;border:1px solid var(--line);border-radius:8px;
                  padding:10px;white-space:pre-wrap"></pre>
    </div>

    <div class="grid cols-4" id="agentKpis" style="margin-top:16px"></div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>What stands in the way</h2>
        <div id="blockers"></div>
      </div>
      <div class="card">
        <h2>Last cycle</h2>
        <div id="lastCycle"></div>
      </div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>The loop, step by step</h2>
        <div id="steps"></div>
      </div>
      <div class="card">
        <h2>Brain &mdash; the model it reasons with</h2>
        <div id="brainBox"></div>
      </div>
    </div>
  </section>

  <!-- MONEY -->
  <section id="tab-money" class="section">
    <div class="section-head">
      <h1>Money</h1>
      <p>Where it is, what it will buy, and how it gets in. A budget is
        permission to use money already in a venue account.</p>
    </div>
    <div class="grid cols-2" style="margin-top:14px">
      <div class="card">
        <h2>Where the money is</h2>
        <div id="accounts"></div>
      </div>
      <div class="card">
        <h2>Setting a budget</h2>
        <div class="warnbox">
          A budget moves no money. It is permission to use money that is already
          in the account - there is nothing to send to the agent, and the agent
          has no account of its own.
        </div>
        <label>Venue</label>
        <select id="budgetVenue" onchange="loadFunding()"
          style="background:var(--panel2);border:1px solid var(--line);color:var(--text);
                 border-radius:7px;padding:9px 11px;width:100%;margin-bottom:12px">
          <option value="polymarket">Polymarket</option>
          <option value="kalshi">Kalshi</option>
        </select>
        <label>Amount the agent may use (USD)</label>
        <input id="budgetAmount" type="number" min="0" step="1" placeholder="50">
        <div style="margin-top:12px;display:flex;gap:9px">
          <button class="primary" onclick="saveBudget()">Authorise budget</button>
          <button onclick="clearBudget()">Set to 0</button>
        </div>
        <div id="budgetMsg" style="margin-top:13px"></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Loss limits &mdash; what stops it</h2>
      <div id="limits"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>How capital gets in</h2>
      <div id="funding"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>What a limited budget will not buy</h2>
      <div id="unfundable"></div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Mode</h2>
        <div class="modebox">
          <div class="modeopt" id="modePaper" onclick="setMode('paper')">
            <div class="t">Paper</div>
            <div class="d">Simulates the whole system against the real orderbook.
              No order is sent, no money moves. Needs no capital and no
              credentials.</div>
          </div>
          <div class="modeopt" id="modeLive" onclick="setMode('live')">
            <div class="t">Live</div>
            <div class="d">Places real orders with real money on venues that are
              funded and authorised. Refused unless a venue is actually ready.</div>
          </div>
        </div>
        <div id="modeMsg"></div>
        <div class="note" style="margin-top:12px">
          The switch is refused, not merely warned about, when nothing is ready:
          an armed system that cannot fire reads as progress when it is not.
        </div>
      </div>
      <div class="card">
        <h2>Not yet measurable</h2>
        <div id="unavailable" class="note"></div>
      </div>
    </div>
  </section>

  <!-- VENUE -->
  <section id="tab-venue" class="section">
    <div class="section-head">
      <h1>Venue</h1>
      <p>One venue holds live capital at a time, named with the reason. Every
        other venue is still scanned and paper-traded.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Which venue the agent is using</h2>
      <div id="venueNow"></div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Why this one</h2>
        <div id="venueWhy" class="note"></div>
      </div>
      <div class="card">
        <h2>Ranking</h2>
        <div id="venueRank"></div>
        <div class="note" style="margin-top:10px">
          Ranked on realised net P&amp;L per resolved trade, with the sample size
          shown next to it. Below the evidence floor a venue has no score at all -
          an unmeasured venue is not a zero, and it is not a winner either.
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Why the fair value is what it is</h2>
      <div class="note" style="margin-bottom:11px">
        <p>Every component of the forecast, with the probability it wanted, the
          confidence it claimed and the weight it actually got. The chain runs
          market price &rarr; the LLM's own answer &rarr; the weighted ensemble
          &rarr; calibrated &rarr; conservative.</p>
      </div>
      <div id="forecast"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Sports bets &mdash; priced, placed, settled</h2>
      <div id="sports"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Out-of-sample validation</h2>
      <div id="validation"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Every venue PTAI knows about</h2>
      <div id="venueAll"></div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Moving to another venue</h2>
        <div id="venueSwitch"></div>
      </div>
      <div class="card">
        <h2>What runs without asking</h2>
        <div id="venueAutonomy"></div>
      </div>
    </div>
  </section>

  <!-- ORDERS -->
  <section id="tab-orders" class="section">
    <div class="section-head">
      <h1>Orders</h1>
      <p>What is working, what it is holding, and whether the venue agrees.</p>
    </div>
    <div class="grid cols-4" id="orderKpis" style="margin-top:14px"></div>
    <div class="card" style="margin-top:16px">
      <h2>Working orders</h2>
      <div id="orders"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>What the venue says</h2>
      <div id="venueView" class="note"></div>
    </div>
  </section>

  <!-- ACTIVITY -->
  <section id="tab-activity" class="section">
    <div class="section-head">
      <h1>Activity</h1>
      <p>Every trade the agent has taken, and the full result of the last cycle.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Recent trades</h2>
      <div id="trades"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Last cycle result</h2>
      <div id="cycleOut" class="note">No cycle run from this page yet.</div>
    </div>
  </section>

  <!-- SETUP -->
  <section id="tab-setup" class="section">
    <div class="section-head">
      <h1>Setup</h1>
      <p>How it runs, the model it thinks with, and where the diagnostics live.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>How it runs</h2>
      <div class="note">
        <p style="margin-bottom:9px">One runner starts everything:
          <code>run_ptai.bat</code>. It opens the agent window (the loop that
          trades), this console, and nothing else. The agent cycles every
          <span id="setupInterval">10</span> minutes on its own; closing its
          window stops the trading, and closing this page changes nothing.</p>
        <p style="margin-bottom:9px">There is no approval step and no button you
          must press between cycles. The buttons here exist to look and to test,
          not to keep it alive: <b>Run one cycle</b> runs a single paper cycle in
          this process, which is why it refuses to run live.</p>
        <p>Everything the agent decides is written to the database as it goes, so
          this screen can be closed and reopened without losing the story.</p>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Brain &mdash; pin the model the agent calls</h2>
      <div id="brainSetup"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Logins &mdash; saved here, used by the agent</h2>
      <div class="note" style="margin-bottom:11px">
        <p style="margin-bottom:9px">Save a login once and the agent uses it on
          every cycle after that, including logins you add while it is already
          running: the next cycle picks them up without a restart. Values are
          encrypted and kept in <code id="vaultPath">data/vault.json</code> on
          this machine; nothing is sent anywhere.</p>
        <p>Anything already in <code>.env</code> keeps working. Where both exist,
          the value saved here wins, and each field says which source is in
          use.</p>
      </div>
      <div id="logins"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Venues the agent may use</h2>
      <div class="note" style="margin-bottom:11px">
        <p style="margin-bottom:9px">Every venue PTAI knows about, with a switch.
          What each one can actually do is on the Venue tab; this is only whether
          the agent should use it. Switching one off takes effect on the next
          cycle, and nothing else changes &mdash; a venue switched on still has to
          be funded, qualified and inside the loss limits before it can touch real
          money.</p>
      </div>
      <div id="venueSwitches"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Advanced tools (diagnostics)</h2>
      <div class="note">
        <p style="margin-bottom:9px">This console is deliberately one screen
          about one agent. The older diagnostic dashboard &mdash; raw logs, the
          V2/V3 internals, the scan history, backtests, wallet linking, the API
          config &mdash; is still on disk and is not part of the daily loop. It
          is a second process, started on purpose:</p>
        <p class="mono" style="margin-bottom:9px">&#62; set PYTHONPATH=src<br>
           &#62; set PTAI_DASHBOARD_PORT=8020<br>
           &#62; python -m ptai.dashboard</p>
        <p>Use it when something needs diagnosing. Nothing on the Agent tab
          depends on it.</p>
      </div>
    </div>
  </section>

  <div class="foot">
    Local only. No cloud. The agent trades the accounts you fund and never holds
    your seed phrase.
  </div>
</main>

<script>
let STATE = {mode:'paper'};
const $ = id => document.getElementById(id);
const money = v => (v===null||v===undefined) ? '&mdash;' : '$' + Number(v).toFixed(2);
const pct = v => (v===null||v===undefined) ? '&mdash;' : Number(v).toFixed(1) + '%';

const esc = v => String(v===null||v===undefined?'':v)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// The sections, in the order they are stacked on the page. Every one is loaded
// when the page opens and refreshed on the timer: "always visible" and "only
// loaded when you click" are the same bug in different clothes.
const SECTIONS = ['agent','money','venue','orders','activity','setup'];

function goTo(name){
  const el = $('tab-'+name);
  if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on', t.dataset.tab===name));
}

// Which section is on screen, so the link the operator is looking at is the one
// highlighted. Read from scroll position rather than a state variable, because
// the page can be scrolled without ever pressing a link.
function spyScroll(){
  const line = 150;
  let current = SECTIONS[0];
  SECTIONS.forEach(n=>{
    const el = $('tab-'+n);
    if(el && el.getBoundingClientRect().top <= line) current = n;
  });
  document.querySelectorAll('.tab').forEach(t=>
    t.classList.toggle('on', t.dataset.tab===current));
}
window.addEventListener('scroll', spyScroll, {passive:true});

async function api(path, opts){
  const r = await fetch(path, Object.assign({headers:{'Content-Type':'application/json'}}, opts||{}));
  let body = null;
  try { body = await r.json(); } catch(e){ body = {}; }
  return {ok:r.ok, status:r.status, body};
}

function setBadge(mode){
  STATE.mode = mode;
  const b = $('modeBadge');
  b.className = 'mode ' + mode;
  $('modeText').textContent = mode.toUpperCase();
  $('modeBadge').querySelector('.dot').className = 'dot' + (mode==='live'?' live':'');
  $('modePaper').className = 'modeopt' + (mode==='paper'?' sel-paper':'');
  $('modeLive').className = 'modeopt' + (mode==='live'?' sel-live':'');
}

// ---- overview ----
async function loadStatus(){
  const {body} = await api('/api/console/status');
  setBadge(body.mode || 'paper');

  const cap = body.capital || {};
  const st = body.storage || {};
  $('hdrCapital').textContent = money(cap.total_available_usd) + ' available \u00b7 '
    + money(cap.total_reserved_usd) + ' reserved';

  // The KPI row lives on the Agent tab, fed by /api/console/agent - which reads
  // the same snapshot these steps do. Rendering it twice is how two screens end
  // up disagreeing about the same number.
  $('steps').innerHTML = (body.steps||[]).map(s=>`
    <div class="step">
      <div class="mark ${s.ok?'ok':'no'}">${s.ok?'\u2713':'\u2022'}</div>
      <div><div class="lbl">${s.label}</div><div class="det">${s.detail}</div></div>
    </div>`).join('');

  const un = body.unavailable || {};
  // Capital has its own availability; the rest is what "not measurable" means.
  const keys = Object.keys(un).filter(k=>k!=='performance');
  $('unavailable').innerHTML = keys.length
    ? '<ul style="margin-left:16px">' + keys.map(k=>
        `<li class="mono" style="margin-bottom:5px">${k}: <span style="color:var(--dim)">${un[k]}</span></li>`
      ).join('') + '</ul>'
    : '<span style="color:var(--green)">Everything on this page is measured from recorded data.</span>';
}

async function setMode(mode){
  $('modeMsg').innerHTML = '';
  const {ok, status, body} = await api('/api/console/mode', {method:'POST', body:JSON.stringify({mode})});
  if(!ok){
    const bs = (body.blockers||[]).map(b=>`<li>${b}</li>`).join('');
    $('modeMsg').innerHTML = `<div class="errbox"><b>${body.error||'refused'}</b>`
      + (bs?`<ul style="margin-left:16px;margin-top:7px">${bs}</ul>`:'')
      + `<div style="margin-top:8px">${body.note||''}</div></div>`;
    return;
  }
  setBadge(body.mode);
  $('modeMsg').innerHTML = `<div class="note">${body.note||''}</div>`;
  loadStatus(); loadAgent();
}

// ---- capital ----
async function loadVenue(){
  const {body} = await api('/api/console/venue');
  const sel = body.selection || {};
  const live = sel.live_venue;
  const candidate = sel.candidate;
  const byId = {};
  (body.assessments||[]).forEach(a=>{ byId[a.venue_id]=a; });

  // ---- which venue, stated plainly ----
  //
  // The MODE leads, and "live" is subordinate to it. Paper mode is a complete
  // way of working - real markets, real order books, real settlement dates,
  // simulated capital - and it used to be announced as though no venue were
  // live yet, which read as a missing piece rather than the chosen mode. Worse,
  // with a venue still selected from an earlier live run, the panel claimed to
  // be trading live while the mode was paper and no real money was moving.
  const mode = (body.mode || 'paper');
  const liveLabel = live ? ((byId[live]||{}).label || live) : null;
  const candLabel = candidate ? ((byId[candidate]||{}).label || candidate) : null;
  let headline;
  if(mode === 'paper'){
    headline = `<div class="pill ok" style="font-size:13px">PAPER &mdash; REAL MARKETS,
      SIMULATED MONEY</div>`;
  } else if(liveLabel){
    headline = `<div class="pill ok" style="font-size:13px">TRADING LIVE ON
      ${esc(liveLabel).toUpperCase()}</div>`;
  } else {
    headline = `<div class="pill wait" style="font-size:13px">LIVE MODE
      &mdash; NOTHING FUNDED YET</div>`;
  }
  const paperNote = mode === 'paper' ? `
    <div class="note" style="margin-top:10px">
      Everything except the money is real: markets, order books, prices and
      settlement times are read from the venues live, and every order is filled
      through the same code path live mode uses. A ${money(body.paper_bankroll_usd)}
      paper bankroll stands in for capital, so the agent can work the whole loop
      without anything at risk. Nothing is missing here${
        candLabel?` &mdash; ${esc(candLabel)} is first in line when you fund one`:''}.
    </div>` : '';
  $('venueNow').innerHTML = headline + paperNote + `
    <div class="note" style="margin-top:10px">${esc(sel.verdict||'')}</div>
    <table style="margin-top:12px">
      <tr><td style="color:var(--dim);width:190px">${mode==='paper'
              ?'Real capital deployed':'Venue holding live capital'}</td>
          <td class="mono">${liveLabel?esc(liveLabel):(mode==='paper'
              ?'<span>none &mdash; paper mode</span>':'<span class="warn">none</span>')}</td></tr>
      <tr><td style="color:var(--dim)">${mode==='paper'
              ?'Paper bankroll (simulated)':'Authorised budget'}</td>
          <td class="mono">${mode==='paper'&&body.paper_bankroll_usd!=null
              ?money(body.paper_bankroll_usd):'&mdash;'}</td></tr>
      <tr><td style="color:var(--dim)">Next venue to fund</td>
          <td class="mono">${candLabel?esc(candLabel):'&mdash;'}</td></tr>
      <tr><td style="color:var(--dim)">How many may hold capital</td>
          <td class="mono">1 &mdash; the agent cannot move money between venues</td></tr>
      <tr><td style="color:var(--dim)">Balances actually read</td>
          <td class="mono">${ body.engine_running
              ? `${body.balances_read||0} of ${body.venues_asked||0} venue(s) answered`
              : 'no engine is running in this process yet, so none was asked' }</td></tr>
    </table>
    <div class="note" style="margin-top:10px">
      Every other venue is still scanned and paper-traded. It is only the money
      that sits in one place.
    </div>`;

  // ---- why ----
  $('venueWhy').innerHTML = (sel.reasons||[]).length
    ? '<ul style="margin:0;padding-left:18px">' + sel.reasons.map(r=>
        `<li style="margin-bottom:7px">${esc(r)}</li>`).join('') + '</ul>'
    : '<span class="warn">No reason was recorded for the current selection.</span>';

  // ---- ranking, with the sample size next to the score ----
  const rows = (body.assessments||[]);
  $('venueRank').innerHTML = `<table>
    <tr><th>Venue</th><th>Role</th><th>Resolved</th><th>Net P&amp;L / trade</th>
        <th>Total net P&amp;L</th></tr>` +
    rows.map(a=>{
      const noEv = !a.has_evidence;
      const pnl = noEv ? '<span style="color:var(--dim)">no score &mdash; too few trades</span>'
                       : `<span class="${a.pnl_per_trade>=0?'pos':'neg'} mono">${money(a.pnl_per_trade)}</span>`;
      const tot = noEv ? '&mdash;'
                       : `<span class="mono ${a.net_pnl_usd>=0?'pos':'neg'}">${money(a.net_pnl_usd)}</span>`;
      // The role column says what each venue is FOR. In paper mode nothing is
      // deploying real money, so a venue the selection marks "live" must not be
      // badged live here - the same contradiction the headline was fixed for,
      // one table down.
      const roleTxt = a.role==='live'
                    ? (mode==='paper'
                       ? '<span class="pill wait">funded &mdash; paper mode</span>'
                       : '<span class="pill ok">live</span>')
                    : a.role==='paper' ? '<span class="pill dim">paper</span>'
                    : '<span class="pill wait">unavailable</span>';
      return `<tr>
        <td><b>${esc(a.label)}</b>${a.qualified?' <span class="pill ok">qualified</span>':''}</td>
        <td>${roleTxt}</td>
        <td class="mono">${a.resolved_trades}${noEv?` <span style="color:var(--dim)">/ ${body.min_sample_for_evidence} to score</span>`:''}</td>
        <td>${pnl}</td><td>${tot}</td>
      </tr>`;
    }).join('') + '</table>';

  // ---- out-of-sample validation ----
  //
  // A verdict on the rules, not on the money. Every number here was produced by
  // testing a rule on folds it did not choose itself, and the holdout column is
  // the one that says whether it survived. It can REFUSE a rule and can never
  // qualify a venue - the panel says so rather than letting a green word imply
  // otherwise.
  const val = body.validation || {};
  const scopes = Object.entries(val.scopes || {});
  const benchNames = Object.keys(val.benched || {});
  const benchHtml = benchNames.length
    ? `<div class="warn" style="margin-top:9px">Benched from real money:
        <b>${esc(benchNames.join(', '))}</b> - an entry only these rules carry still runs,
        in paper, and the trade records the refusal. It changes nothing about which trades
        are considered: the entry gates are untouched.</div>` : '';
  if(!scopes.length){
    $('validation').innerHTML = `<div class="note">${esc(val.reason
      || 'Out-of-sample validation has not been run yet. Run: python main.py validate')}</div>`
      + benchHtml;
  } else {
    $('validation').innerHTML = scopes.map(([name, rep]) => {
      const good = rep.verdict === 'confirmed_economic';
      const bad = rep.verdict === 'unconfirmed' || rep.verdict === 'confirmed_uneconomic';
      const tone = good ? 'ok' : bad ? 'neg' : 'wait';
      const rows = (rep.results||[]).filter(r=>r.name !== 'all');
      return `<div style="padding:10px 0;border-bottom:1px solid rgba(36,48,64,.5)">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <b>${esc(name)}</b><span class="pill ${tone}">${esc(rep.verdict||'')}</span></div>
        <div class="note" style="margin-top:6px">${esc(rep.summary||'')}</div>
        <div class="note" style="margin-top:6px">${rep.rows||0} settled trade(s) in
          ${rep.folds||0} fold(s) &middot; base rate
          ${((rep.base_rate||0)*100).toFixed(1)}% &middot; holdout
          ${((rep.holdout_base_rate||0)*100).toFixed(1)}%</div>
        <table style="margin-top:8px"><tr><th>Rule</th><th>Entries</th><th>Hit rate</th>
          <th>Lift</th><th>p</th><th>Holdout</th><th>Break-even</th></tr>` +
        rows.map(r=>{
          const chk = v => v===null||v===undefined ? '&mdash;' : (v*100).toFixed(1)+'%';
          return `<tr><td>${esc(r.name)}${r.confirmed?' <span class="pill ok">held</span>':''}${
              r.significant&&!r.confirmed?' <span class="pill neg">failed holdout</span>':''}</td>
            <td class="mono">${r.entries}</td><td class="mono">${chk(r.hit_rate)}</td>
            <td class="mono ${(r.lift||0)>=0?'pos':'neg'}">${
              r.lift===null||r.lift===undefined?'&mdash;':((r.lift>=0?'+':'')+(r.lift*100).toFixed(1)+'%')}</td>
            <td class="mono">${r.p_value===null||r.p_value===undefined?'&mdash;':r.p_value.toFixed(4)}</td>
            <td class="mono">${r.holdout_entries||0} @ ${chk(r.holdout_hit_rate)}</td>
            <td class="mono">${chk(r.break_even)}</td></tr>`;
        }).join('') + '</table></div>';
    }).join('') + benchHtml + `
      <div class="note" style="margin-top:11px">Tested on consecutive folds, corrected for testing
        several rules at once (Holm-Bonferroni), and judged against the break-even the entries
        actually paid. This can refuse a rule. It can never qualify a venue: only settled
        money does that.</div>`;
  }

  // ---- every venue, and what it can do ----
  //
  // The ranking above only shows venues that could hold money or already have
  // evidence. That left an operator unable to tell whether the other adapters
  // existed, were broken, or were silently dropped - so this lists the whole
  // registry with one line each: can it be read now, can PTAI fill there, can
  // it ever hold real money, and what would it need.
  const inv = body.inventory || {};
  const invRows = Object.values(inv.venues || {});
  if(!invRows.length){
    $('venueAll').innerHTML = `<div class="note">${esc(inv.reason
      || 'The agent has not recorded its venue list yet. Start it once - every venue it knows about, and what each one can do, appears here.')}</div>`;
  } else {
    // The class names come from the venue inventory module, so the page cannot
    // drift from the classifier that produced them.
    const USE_REAL = 'real_money', USE_PAPER = 'paper_only',
          USE_LOGIN = 'needs_login', USE_SCAN = 'scanner', USE_NONE = 'no_client';
    const usePill = u => u===USE_REAL ? '<span class="pill ok">can hold real money</span>'
      : u===USE_PAPER ? '<span class="pill dim">paper + live data</span>'
      : u===USE_LOGIN ? '<span class="pill wait">needs a login</span>'
      : u===USE_SCAN ? '<span class="pill dim">scanner only</span>'
      : '<span class="pill wait">no client built</span>';
    const c = inv.counts || {};
    $('venueAll').innerHTML = `
      <div class="note">${c.registered||invRows.length} venue(s) registered by the agent:
        <b>${c.readable_now||0}</b> readable right now with no account,
        <b>${c.paper_tradable||0}</b> being paper-traded,
        <b>${c.can_place_real_orders||0}</b> armed to place a real order
        (only ${Object.values(inv.venues||{}).filter(v=>v.real_order_path)
                .map(v=>esc(v.label)).join(', ')||'none'} has a submission path at all),
        <b>${c.no_client||0}</b> with no client written.</div>
      <table style="margin-top:10px">
        <tr><th>Venue</th><th>What PTAI can do with it</th><th>Runs today</th>
            <th>What it needs</th></tr>` +
      invRows.map(v=>`<tr>
        <td><b>${esc(v.label)}</b><br><span style="color:var(--dim);font-size:11.5px">${esc(v.venue_id)}${
            v.venue_type?` &middot; ${esc(v.venue_type)}`:''}</span></td>
        <td>${usePill(v.use)}<div class="note" style="margin-top:5px">${esc(v.why||'')}</div></td>
        <td class="mono ${v.can_run_today?'pos':'neg'}">${v.can_run_today?'yes':'no'}</td>
        <td><span class="note">${esc(v.what_it_needs||'')}</span>${
            v.fundable && v.minimum_deposit_usd!=null
              ? `<div class="note" style="margin-top:4px">funding: ${esc(v.currency||'')},
                 minimum ${money(v.minimum_deposit_usd)}</div>`
              : (v.reason_unfundable
                 ? `<div class="note" style="margin-top:4px">${esc(v.reason_unfundable)}</div>`:'')}</td>
      </tr>`).join('') + '</table>';
  }

  // ---- switching ----
  const sp = sel.switch_plan || {};
  const stepList = (sp.steps||[]).map(x=>`<li style="margin-bottom:5px">${esc(x)}</li>`).join('');
  $('venueSwitch').innerHTML = sp.reason
    ? `<div class="note">${esc(sp.reason)}</div>` +
      (stepList ? `<ol style="margin:10px 0 0;padding-left:18px">${stepList}</ol>` : '')
    : '<span class="warn">No switch plan was recorded.</span>';

  // ---- autonomy ----
  const au = sel.autonomy || {};
  const act = (au.operator_only_actions||[]).map(x=>
    `<li style="margin-bottom:6px"><b>${esc(x.action)}</b> <span style="color:var(--dim)">
     &mdash; ${esc(x.how_often)}</span><br><span class="note">${esc(x.why)}</span></li>`).join('');
  $('venueAutonomy').innerHTML = `
    <table>
      <tr><td style="color:var(--dim);width:170px">Trades without approval</td>
          <td class="mono ${au.trades_without_approval?'pos':'neg'}">${
            au.trades_without_approval?'YES':'no'}</td></tr>
      <tr><td style="color:var(--dim)">Approval per trade</td>
          <td class="mono">${au.per_trade_approval?'required':'none'}</td></tr>
      <tr><td style="color:var(--dim)">Cycle</td>
          <td class="mono">every ${au.cycle_minutes||10} minutes, indefinitely</td></tr>
    </table>
    <div style="margin-top:12px"><b style="font-size:12.5px">Done on its own</b>
      <ul style="margin:7px 0 0;padding-left:18px">${
        (au.agent_does_autonomously||[]).map(x=>
          `<li style="margin-bottom:4px">${esc(x)}</li>`).join('')}</ul></div>
    <div style="margin-top:12px"><b style="font-size:12.5px">Never needs</b>
      <ul style="margin:7px 0 0;padding-left:18px">${
        (au.what_it_never_needs||[]).map(x=>
          `<li style="margin-bottom:4px">${esc(x)}</li>`).join('')}</ul></div>
    <div style="margin-top:12px"><b style="font-size:12.5px">Your actions only</b>
      <ol style="margin:7px 0 0;padding-left:18px">${act}</ol></div>
    ${au.honest_limit?`<div class="errbox" style="margin-top:12px;margin-bottom:0">${
      esc(au.honest_limit)}</div>`:''}`;
}

async function loadCapital(){
  const {body} = await api('/api/console/capital');
  // ---- the hard limits ----
  //
  // Read from the guard the dispatcher consults, not recomputed. The limits are
  // measured against settled outcomes in the trade log, so a restart cannot
  // reset them and this panel cannot disagree with the order path.
  const lim = body.limits || {};
  const laneRow = (name, row) => {
    if(!row) return '';
    const bar = (used, label, limit) => `
      <div style="margin-top:7px">
        <div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--dim)">
          <span>${label}</span><span class="mono">${limit==null?'&mdash;':money(limit)}</span></div>
        <div style="height:7px;background:rgba(36,48,64,.8);border-radius:4px;margin-top:4px;overflow:hidden">
          <div style="height:100%;width:${Math.min(100,(used||0)*100).toFixed(1)}%;
               background:${(used||0)>=0.8?'var(--neg, #e0555f)':(used||0)>=0.5?'#d8a13a':'var(--pos, #3fbf7f)'}"></div>
        </div></div>`;
    return `<div style="padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="text-transform:capitalize">${esc(name)}</b>
        ${row.halted ? '<span class="pill neg">stopped</span>'
                     : `<span class="mono" style="color:var(--dim)">today ${money(row.daily_pnl_usd)}</span>`}
      </div>
      ${row.halted ? `<div class="errbox" style="margin-top:8px">${esc(row.halt_reason)}</div>` : ''}
      ${bar(row.daily_used_pct, 'Daily loss limit', row.daily_loss_limit_usd)}
      ${bar(row.session_used_pct, 'Session loss limit', row.session_loss_limit_usd)}
      <div class="note" style="margin-top:7px">Capital this applies to: ${money(row.bankroll_usd)}
        &middot; committed and unsettled: ${money(row.open_risk_usd)}
        &middot; ${(row.loss_pct_of_bankroll*100).toFixed(1)}% of capital lost this session</div>
    </div>`;
  };
  const unreadable = lim.available === false || (!lim.live && !lim.paper);
  $('limits').innerHTML = unreadable
    ? `<div class="note">${esc(lim.reason || 'The limits could not be read, so nothing '
        + 'may risk real money until they can be.')}</div>`
    : laneRow('live (real money)', lim.live) + laneRow('paper (simulated)', lim.paper) + `
      <div class="note" style="margin-top:11px">Session ${(lim.session_loss_pct*100).toFixed(0)}% and daily
        ${(lim.daily_loss_pct*100).toFixed(0)}% of the capital in use. Measured from settled
        outcomes in the trade log, so a restart cannot reset them; they lift when the day rolls
        or when you authorise a budget or switch mode.
        ${lim.live_proven
          ? 'Live sizing is at the full per-order cap.'
          : `Live sizing is held at the ${(lim.micro_stake_pct*100).toFixed(0)}% micro cap until
             ${lim.live_proving_trades} live trades at a venue have settled.`}</div>`;

  const accts = body.accounts || [];
  if(!accts.length){
    $('accounts').innerHTML = `<div class="empty">No venue has a budget yet.
      Set one below, and run in paper mode in the meantime - it needs no capital.</div>`;
    return;
  }
  $('accounts').innerHTML = accts.map(a=>{
    const deployed = a.deposited_usd>0 ? Math.min(100, a.deployment_pct) : 0;
    const posPart = a.deposited_usd>0 ? (a.in_positions_usd/a.deposited_usd*100) : 0;
    const resPart = a.deposited_usd>0 ? (a.reserved_usd/a.deposited_usd*100) : 0;
    return `<div style="padding:13px 0;border-bottom:1px solid rgba(36,48,64,.5)">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <b style="font-size:14.5px">${a.venue_label}</b>
        <span class="pill ${a.balance_is_real?'ok':'wait'}">${
          a.balance_is_real ? 'balance read' : 'balance unread'}</span>
        <span class="pill ${a.can_deploy_live?'ok':'dim'}">${
          a.can_deploy_live ? 'live enabled' : 'no live capital'}</span>
        <span class="spacer" style="flex:1"></span>
        <span class="mono" style="color:var(--dim);font-size:12.5px">
          authorised ${money(a.budget_usd)}</span>
      </div>
      <div class="bar" title="committed vs deposited">
        <i style="width:${posPart}%;background:var(--blue)"></i>
        <i style="width:${resPart}%;background:var(--amber)"></i>
      </div>
      <div class="grid cols-4" style="margin-top:11px;gap:10px">
        <div><div class="k" style="font-size:11px;color:var(--dim)">Deposited</div>
             <div class="mono">${money(a.deposited_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">In positions</div>
             <div class="mono">${money(a.in_positions_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">Reserved</div>
             <div class="mono warn">${money(a.reserved_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">Available</div>
             <div class="mono pos">${money(a.available_usd)}</div></div>
      </div>
      ${(a.notes||[]).length?`<div class="note" style="margin-top:8px">${a.notes.join(' &middot; ')}</div>`:''}
      ${(a.warnings||[]).length?`<div class="errbox" style="margin-top:9px;margin-bottom:0">${
        a.warnings.join('<br>')}</div>`:''}
      <div class="note" style="margin-top:6px">${deployed.toFixed(0)}% of deposited capital is committed.</div>
    </div>`;
  }).join('');
}

async function loadFunding(){
  const venue = $('budgetVenue').value;
  const total = parseFloat($('budgetAmount').value) || 0;
  const {body} = await api('/api/console/funding?total=' + encodeURIComponent(total));
  const r = body.routes[venue];
  if(!r){ $('funding').innerHTML = '<div class="empty">No route recorded.</div>'; return; }
  $('funding').innerHTML = `
    <div class="note" style="margin-bottom:12px">${body.principle}</div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">
      <div><div style="font-size:11px;color:var(--dim)">Currency</div>
        <div style="font-size:13px">${r.currency}</div></div>
      <div><div style="font-size:11px;color:var(--dim)">Minimum deposit</div>
        <div class="mono">${money(r.minimum_deposit_usd)}</div></div>
      <div><div style="font-size:11px;color:var(--dim)">Smallest practical</div>
        <div class="mono">${money(r.smallest_practical_usd)}</div></div>
    </div>
    <ol>${r.deposit_steps.map(s=>`<li>${s}</li>`).join('')}</ol>
    <div class="note" style="margin-top:12px"><b>Withdrawing.</b> ${r.withdraw_note}</div>
    <div class="note" style="margin-top:7px"><b>Fees.</b> ${r.fees_note}</div>
    <div class="note" style="margin-top:7px"><b>The key.</b> ${r.agent_key_explanation}</div>
    ${(body.plan.warnings||[]).length
      ? `<div class="warnbox" style="margin-top:13px;margin-bottom:0">${
          body.plan.warnings.join('<br>')}</div>`
      : `<div class="note" style="margin-top:13px;color:var(--green)">${
          (body.plan.notes||[]).join(' ')}</div>`}`;

  $('unfundable').innerHTML = Object.entries(body.unfundable||{}).map(([k,v])=>
    `<div style="padding:8px 0;border-bottom:1px solid rgba(36,48,64,.4)">
       <span class="mono" style="font-size:12.5px">${k}</span>
       <span style="color:var(--dim);font-size:12.5px"> &mdash; ${v}</span></div>`).join('')
    + `<div class="note" style="margin-top:11px">These are still scanned for
       opportunities and still run in paper mode. They cost nothing there; they
       just cannot hold your $50.</div>`;
}

async function saveBudget(){
  const venue = $('budgetVenue').value;
  const amount = parseFloat($('budgetAmount').value);
  if(isNaN(amount) || amount < 0){
    $('budgetMsg').innerHTML = '<div class="errbox">Enter an amount of 0 or more.</div>';
    return;
  }
  const {ok, body} = await api('/api/console/budget', {method:'POST',
    body:JSON.stringify({venue, amount})});
  if(!ok){
    $('budgetMsg').innerHTML = `<div class="errbox"><b>${body.error}</b>`
      + (body.reason?`<div style="margin-top:7px">${body.reason}</div>`:'')
      + (body.note?`<div style="margin-top:7px">${body.note}</div>`:'') + `</div>`;
    return;
  }
  $('budgetMsg').innerHTML = `<div class="note pos">Authorised ${money(amount)} at ${venue}.
    Nothing was moved: the money is already in that account.</div>`;
  loadCapital(); loadStatus();
}

function clearBudget(){ $('budgetAmount').value = 0; saveBudget(); }

// ---- orders ----
async function loadOrders(){
  const {body} = await api('/api/console/orders');
  $('orderKpis').innerHTML = [
    ['Working orders', body.local_open_orders, 'not yet filled or cancelled'],
    ['Cash reserved', money(body.local_reserved_usd), 'locked behind those orders'],
  ].map(([k,v,s])=>`<div class="card kpi"><div class="k">${k}</div>
     <div class="v">${v}</div><div class="sub">${s}</div></div>`).join('');

  const rows = body.orders || [];
  $('orders').innerHTML = rows.length ? `<table><thead><tr>
      <th>Order</th><th>Market</th><th>Side</th><th>Limit</th>
      <th>Requested</th><th>Matched</th><th>Status</th></tr></thead><tbody>`
    + rows.map(o=>`<tr>
        <td class="mono" style="font-size:12px">${o.order_id||'&mdash;'}</td>
        <td class="mono" style="font-size:12px">${o.market_id||''}</td>
        <td>${o.side||''}</td>
        <td class="mono">${o.limit_price!=null?Number(o.limit_price).toFixed(3):'&mdash;'}</td>
        <td class="mono">${money(o.requested_usd)}</td>
        <td class="mono">${money(o.matched_usd)}</td>
        <td><span class="pill wait">${o.status||''}</span></td></tr>`).join('')
    + `</tbody></table>`
    : `<div class="empty">No working orders. ${body.note}</div>`;

  const view = body.venue_view;
  $('venueView').innerHTML = (view && view.available)
    ? `<div class="mono" style="font-size:12.5px">${JSON.stringify(view, null, 2)}</div>`
    : `<span class="warn">The venue's own order list was not read:</span>
       ${(view&&view.reason)||'no adapter'} &mdash; local orders stay reserved
       until the venue answers.`;
}

// ---- activity ----
async function loadResults(){
  const {body} = await api('/api/console/results');
  const trades = body.recent_trades || [];
  $('trades').innerHTML = trades.length ? `<table><thead><tr>
      <th>Market</th><th>Side</th><th>Size</th><th>Entry</th>
      <th>P&amp;L</th><th>Status</th></tr></thead><tbody>`
    + trades.map(t=>{
        const pnl = t.pnl;
        const cls = (pnl==null) ? '' : (pnl>=0?'pos':'neg');
        return `<tr>
          <td class="mono" style="font-size:12px">${(t.market_id||'').slice(0,22)}</td>
          <td>${t.side||''}</td>
          <td class="mono">${money(t.position_size_usd)}</td>
          <td class="mono">${t.market_price!=null?Number(t.market_price).toFixed(3):'&mdash;'}</td>
          <td class="mono ${cls}">${pnl==null?'&mdash;':(pnl>=0?'+':'')+Number(pnl).toFixed(2)}</td>
          <td><span class="pill ${t.status==='paper'?'dim':'ok'}">${t.status||''}</span></td>
        </tr>`;}).join('') + `</tbody></table>`
    : `<div class="empty">No trades recorded yet. Paper mode fills are recorded the
       same way live ones are, so this table is where the strategy earns the right
       to real money.</div>`;
}

// ---- the round ----
//
// What a completed run is FOR: the paper account at both ends, the net in
// dollars, and whether the book is up or down overall. Rendered from the record
// on every poll so the last round is on the page before the operator presses
// anything, then refreshed from the run's own reply the moment one finishes.
function roundFigure(net){
  if(net==null || net==='') return '<span class="warn">no figure</span>';
  const v = Number(net);
  return `<span class="mono ${v>=0?'pos':'neg'}" style="font-weight:700">`
    + `${v>=0?'+':''}$${Math.abs(v).toFixed(2)}</span>`;
}

function showRound(round, summary, rows, justRan){
  const r = round || (rows||[])[0] || null;
  const s = summary || {};
  if(!r){
    $('round').innerHTML = `<div class="note">No round has finished yet. Run one
      and it will end with a bankroll figure here &mdash; positive or negative.</div>`;
    return;
  }
  const net = r.net_usd;
  const verdict = r.verdict || 'unknown';
  const pill = verdict==='up' ? 'ok' : (verdict==='down' ? 'no' : 'dim');
  const head = net==null
    ? `<div style="font-size:15px"><b>Round ${r.number||''}</b> &mdash;
        <span class="warn">no figure</span>
        <span class="note">${esc((r.warnings||[])[0]
            || 'nothing could be priced this round, so the account value is unknown rather than unchanged')}</span></div>`
    : `<div style="font-size:15px"><b>Round ${r.number||''}</b>
        <span class="pill ${pill}">${esc(verdict)}</span>
        <span class="mono">${r.equity_start==null?'?':'$'+Number(r.equity_start).toFixed(2)}
          &rarr; ${r.equity_end==null?'?':'$'+Number(r.equity_end).toFixed(2)}
          = ${roundFigure(net)}</span>
        ${justRan?'<span class="note">just finished</span>':''}</div>`;
  const detail = `<div class="note" style="margin-top:5px">
      ${r.positions_opened||0} opened &middot; ${r.positions_settled||0} settled &middot;
      ${r.positions_held||0} held
      ${r.open_positions_unmarked?`&middot; ${r.open_positions_unmarked} position(s)
        carried at cost (no current price)`:''}
      &middot; discovered ${r.markets_discovered||0}, screened ${r.markets_screened||0},
      researched ${r.markets_researched||0}, priced ${r.markets_priced||0}
      ${r.staked_usd?`&middot; $${Number(r.staked_usd).toFixed(2)} staked`:''}
      ${r.duration_seconds?`&middot; took ${Math.round(r.duration_seconds)}s`:''}</div>`;
  const realised = (r.realised_pnl||r.unrealised_pnl)
    ? `<div class="note" style="margin-top:5px">of the move:
        ${roundFigure(r.realised_pnl)} realised (settled and in the bankroll) &middot;
        ${roundFigure(r.unrealised_pnl)} marked (the open book at current prices,
        not a settlement)</div>`
    : '';
  const score = s.scored
    ? `<div class="note" style="margin-top:9px">Across ${s.scored} scored round(s):
        <b class="mono ${Number(s.net_usd)>=0?'pos':'neg'}">${Number(s.net_usd)>=0?'+':''}$${Math.abs(Number(s.net_usd)).toFixed(2)}</b>
        &middot; ${s.up||0} up, ${s.down||0} down, ${s.flat||0} flat
        &middot; best ${roundFigure(s.best_round)}, worst ${roundFigure(s.worst_round)}</div>`
    : `<div class="note" style="margin-top:9px">${esc(s.note
        || 'no completed round has produced a figure yet')}</div>`;
  const history = (rows||[]).length > 1
    ? `<table style="margin-top:11px"><thead><tr><th>#</th><th>Account</th>
        <th>Before</th><th>After</th><th>Net</th><th>Opened / settled / held</th>
        <th>Took</th></tr></thead><tbody>`
      + rows.slice(0,8).map(x=>`<tr>
          <td class="mono">${x.number||''}</td>
          <td><span class="pill ${x.account==='live'?'no':'dim'}">${esc(x.account||x.mode||'paper')}</span></td>
          <td class="mono">${x.equity_start==null?'&mdash;':'$'+Number(x.equity_start).toFixed(2)}</td>
          <td class="mono">${x.equity_end==null?'&mdash;':'$'+Number(x.equity_end).toFixed(2)}</td>
          <td>${roundFigure(x.net_usd)}</td>
          <td class="mono">${x.positions_opened||0} / ${x.positions_settled||0} / ${x.positions_held||0}</td>
          <td class="mono">${x.duration_seconds!=null?Math.round(x.duration_seconds)+'s':'&mdash;'}</td>
        </tr>`).join('') + `</tbody></table>`
    : '';
  $('round').innerHTML = head + detail + realised + score + history;
}

async function loadRounds(){
  const {body} = await api('/api/console/rounds');
  showRound((body.rounds||[])[0] || null, body.summary||{}, body.rounds||[], false);
}

// ---- the agent's lifecycle, from the page ----
//
// start / stop / run-now all act on THE agent. If the command window owns the
// loop, these say so and name the process rather than starting a second engine
// that would write to the same keys.
let ENGINE = {control: null, logSeq: 0};

const STATE_WORD_ENGINE = {running:'the agent is running', starting:'the agent is starting',
                           stopping:'the agent is stopping', stopped:'the agent is stopped',
                           idle:'the agent is not running'};

async function loadControl(){
  const {ok, body} = await api('/api/console/agent-control');
  if(!ok || !body || !body.lease){ return; }
  ENGINE.control = body;
  paintEngine(body);
  const box = $('intervalMin');
  if(box && document.activeElement !== box && body.interval && body.interval.minutes){
    box.value = body.interval.minutes;
  }
}

function paintEngine(body){
  const el = $('engineLine');
  if(!el) return;
  const lease = body.lease || {};
  const iv = body.interval || {};
  const bits = [];
  if(body.console_hosting){
    bits.push('<b>this console is running the agent</b> (pid ' +
      esc(String(body.lease.pid || '')) + ')');
  } else if(lease.held){
    bits.push('<b>the agent runs in the ' + esc(String(lease.kind||'another')) +
      ' engine</b> (pid ' + esc(String(lease.pid||'?')) + ' on ' +
      esc(String(lease.host||'?')) + ', alive ' +
      esc(ageText(lease.age_seconds)) + ' ago)');
  } else if(lease.released){
    bits.push('the agent is <b>stopped</b>: ' + esc(String(lease.released_reason||'')) +
      ' at ' + esc(String(lease.released_at||'').slice(11,16)) + ' UTC');
  } else {
    bits.push('no engine is running the agent' +
      (lease.note ? ' (' + esc(String(lease.note)) + ')' : ''));
  }
  bits.push('cycle every <b>' + esc(String(iv.minutes==null?'?':iv.minutes)) +
    ' min</b> (' + esc(String(iv.source||'')) + ')');
  if(body.agent_error){
    bits.push('<span class="neg">the last engine error: ' +
      esc(String(body.agent_error)) + '</span>');
  }
  el.innerHTML = bits.join(' &middot; ');
  const start = $('startBtn'), stop = $('stopBtn'), run = $('runBtn');
  if(start && stop && run){
    const busy = body.agent_state==='starting';
    start.disabled = body.console_hosting || busy || lease.held;
    stop.disabled = !body.console_hosting && !lease.held;
    run.disabled = false;
    start.textContent = body.console_hosting ? 'Agent is running here' : 'Start the agent';
  }
}

async function agentAction(what){
  const btn = what==='start' ? $('startBtn') : $('stopBtn');
  const was = btn.textContent;
  btn.disabled = true; btn.textContent = what==='start' ? 'Starting\u2026' : 'Stopping\u2026';
  const {ok, body} = await api('/api/console/agent-control', {method:'POST',
    body:JSON.stringify({action:what})});
  btn.textContent = was;
  const box = $('cycleOut');
  const text = ok ? (what==='start' ? 'The agent is starting here; its first cycle ' +
        'begins immediately and the round card and log will fill in.'
      : 'The agent was stopped on purpose, and the record says so.')
    : (body.error || body.reason || 'that did not work');
  if(box) box.innerHTML = '<div class="' + (ok?'note':'errbox') + '"><b>' +
    esc(String(text)) + '</b>' + (!ok && body.holder && body.holder.pid
      ? '<div class="note" style="margin-top:5px">held by the ' +
        esc(String(body.holder.kind||'')) + ' engine, pid ' +
        esc(String(body.holder.pid)) + ' on ' + esc(String(body.holder.host||'')) +
        ', alive ' + esc(ageText(body.holder.age_seconds)) + ' ago</div>' : '') + '</div>';
  await loadAll(); await loadControl();
}

async function setIntervalMin(){
  const box = $('intervalMin');
  const minutes = box ? box.value : '';
  const {ok, body} = await api('/api/console/agent/interval', {method:'POST',
    body:JSON.stringify({minutes:minutes})});
  const note = $('intervalNote');
  if(note){
    note.innerHTML = ok
      ? esc(String(body.message||'saved')) +
        (body.applied_to_running_agent
          ? ' <b>The running agent was woken; its next wait uses the new interval.</b>'
          : ' The agent is not running here, so the next start uses it.')
      : '<span class="neg">' + esc(String(body.error||'could not set the interval')) + '</span>';
  }
  loadControl();
}

// ---- the log, as the process writes it ----
async function loadLogs(){
  // A hidden tab polls nothing. The log timer is the fastest one on the page, and
  // it must respect the same rule the panels do.
  if(document.visibilityState === 'hidden') return;
  const {ok, body} = await api('/api/console/logs?after=' + ENGINE.logSeq + '&limit=200');
  if(!ok || !body || !body.lines){ return; }
  const pre = $('agentLog');
  if(!pre) return;
  const where = $('logWhere');
  if(where){
    const file = body.log_file && body.log_file.name
      ? ' The durable log file is <b>' + esc(body.log_file.name) + '</b> (' +
        esc(String(body.log_file.size_kb)) + ' kB).' : '';
    where.innerHTML = 'Showing the last ' + esc(String(body.buffered)) + ' line(s) ' +
      'buffered by the ' + esc(String(body.role)) + ' engine, pid ' +
      esc(String(body.pid)) + '.' + file;
  }
  const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 24;
  const fresh = body.lines.filter(l=>l.seq > ENGINE.logSeq);
  if(ENGINE.logSeq === 0){
    pre.textContent = body.lines.map(l=>l.time + ' | ' + l.level.padEnd(7) + ' | ' + l.text).join('\n');
  } else if(fresh.length){
    pre.textContent += (pre.textContent ? '\n' : '') +
      fresh.map(l=>l.time + ' | ' + l.level.padEnd(7) + ' | ' + l.text).join('\n');
  }
  ENGINE.logSeq = body.next || ENGINE.logSeq;
  // Trim what the DOM holds, not just the buffer: an operator who leaves the page
  // open for a day must not end up with a 100k-line <pre>.
  const lines = pre.textContent.split('\n');
  if(lines.length > 600){ pre.textContent = lines.slice(lines.length-600).join('\n'); }
  if(stick) pre.scrollTop = pre.scrollHeight;
}

async function runCycle(){
  const btn = $('runBtn');
  btn.disabled = true; btn.textContent = 'Asking the agent\u2026';
  const {ok, body} = await api('/api/console/run-cycle', {method:'POST',
    body:JSON.stringify({mode:STATE.mode})});
  btn.disabled = false; btn.textContent = 'Run a round now';
  if(!ok){
    $('cycleOut').innerHTML = `<div class="errbox"><b>${esc(String(body.error||'cycle refused'))}</b>`
      + ((body.warnings||[]).length?`<ul style="margin-left:16px;margin-top:7px">${
          body.warnings.map(w=>`<li>${esc(String(w))}</li>`).join('')}</ul>`:'') + '</div>';
    loadControl();
    return;
  }
  // "requested" or "started": the round runs in the engine that OWNS the loop, so
  // the page waits for its own log and round card to fill in rather than pretending
  // the button did the work in this browser request.
  if(body.status === 'requested' || body.status === 'started' ||
     body.status === 'running'){
    $('cycleOut').innerHTML = `<div class="note"><b>${esc(String(body.message||''))}</b></div>`;
    loadControl(); loadLogs(); loadAgent();
    return;
  }
  const sp = body.sports || {};
  const spPlaced = sp.bets_placed || 0;
  const spNote = spPlaced || (sp.settlement||{}).settled
    ? `<div class="note" style="margin-top:7px">Sports: ${spPlaced} bet(s)
        placed${(sp.settlement&&sp.settlement.settled)?`, ${sp.settlement.settled} settled`:''}
        &middot; ${sp.opportunities||0} market(s) priced of ${sp.market_types_available||0} types,
        ${sp.executable||0} executable
        ${(sp.ratings&&sp.ratings.rated_teams)?`&middot; ${sp.ratings.rated_teams} team(s) rated for the models`:''}</div>`
    : `<div class="note" style="margin-top:7px">Sports: nothing placed this cycle.
        ${(sp.blockers||[]).length
          ? 'Why: '+(sp.blockers||[]).map(b=>esc(b)).join(' &middot; ')
          : 'The lane prices fixtures but only bets where it has its own view and a price to beat, and it only settles on a result a feed reported.'}</div>`;
  const lm = body.local_model || {};
  $('cycleOut').innerHTML = `
    ${lm.describe ? `<div class="note">Local model: ${esc(lm.describe)}
        ${lm.used
          ? `&middot; ${lm.answered||0} of ${lm.calls||0} call(s) answered`
          : `&middot; <b>NOT USED</b>: ${esc(lm.not_used_reason||'no reason recorded')}`}</div>` : ''}
    ${body.deep_analysis && body.deep_analysis.considered
      ? `<div class="note">Model time: ${((body.deep_analysis.shortlist)||[]).length}
          of ${body.deep_analysis.considered} market(s) deep-analysed
          (limit ${body.deep_analysis.limit}); ${body.deep_analysis.screened_out||0}
          priced on their measured book only.</div>` : ''}
    <div><b>${body.status||''}</b> &mdash; ${body.executed||0} position(s) recorded,
      ${body.orders_tracked||0} order(s) tracked for reconciliation.</div>
    <div class="note" style="margin-top:7px">
      settlement: ${JSON.stringify(body.settlement||null)} &middot;
      reconciliation: ${JSON.stringify(body.reconciliation||null)} &middot;
      redemption: ${JSON.stringify(body.redemption||null)}
    </div>` + spNote +
    ((sp.refusals||[]).length ? `<div class="note" style="margin-top:5px">Why no
      sports bet: ${esc(sp.refusals.map(r=>`${r.outcome||''}: ${(r.reasons||[])[0]||''}`).join(' · '))}</div>` : '');
  // The round is the answer, so it goes on the page from the run's OWN reply -
  // not from a later poll, which would leave the button looking like it only
  // reported activity.
  const rh = body.rounds || {};
  showRound(body.round || (rh.rounds||[])[0] || null, rh.summary || {},
            rh.rounds || [], true);
  // The cycle just wrote its own phase, heartbeat and scan row: show them, so
  // the front page cannot lag behind a cycle the operator just ran by hand.
  loadStatus(); loadAgent();
}

// ---- logins: saved here, used by the agent ----
//
// One row per login, one input per field, and a Save that means saved: the
// agent reads the vault at the start of every cycle, so what is typed here is in
// force on the next one. Each field says whether its value is the saved one or
// the environment's, because "it is set" is not the same as "it is the one the
// agent uses".
async function loadLogins(){
  const {body} = await api('/api/console/logins');
  const tools = Object.entries(body.tools || {});
  if(!tools.length){
    $('logins').innerHTML = `<div class="note">No logins are defined in this build.</div>`;
    return;
  }
  $('vaultPath').textContent = body.vault || 'data/vault.json';
  $('logins').innerHTML = tools.map(([name, t]) => {
    const rows = (t.fields||[]).map(f => `
      <tr>
        <td style="max-width:210px">${esc(f.label)}${f.required?'':' <span class="pill dim">optional</span>'}</td>
        <td><input type="text" id="login-${esc(t.name)}-${esc(f.name)}"
             value="${f.source==='environment'?'':esc(f.shown||'')}"
             placeholder="${f.set?(f.source==='environment'?'set in .env':'already saved'):(esc(f.example)||'')}"
             style="width:100%"></td>
        <td class="mono">${f.set
            ? `<span class="pill ${f.source==='saved'?'ok':'wait'}">${f.source==='saved'?'saved':'from .env'}</span>`
            : '<span class="pill no">not set</span>'}</td>
      </tr>
      <tr><td></td><td colspan="2" class="note" style="padding-top:0">${esc(f.hint||'')}</td></tr>`).join('');
    return `<div style="padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>${esc(t.label)}</b>
        <span class="pill ${t.configured?'ok':'wait'}">${t.configured?'configured':'needs a login'}</span>
      </div>
      <div class="note" style="margin-top:5px">Unlocks: ${esc(t.unlocks)}</div>
      ${t.then?`<div class="note" style="margin-top:3px">Then: ${esc(t.then)}</div>`:''}
      <table style="margin-top:8px"><tbody>${rows}</tbody></table>
      <div style="margin-top:8px" class="row">
        <button class="primary" onclick="saveLogin('${esc(t.name)}', ${JSON.stringify(
            (t.fields||[]).map(f=>f.name)).replace(/"/g,'&quot;')})">Save login</button>
        ${t.configured?`<button class="danger" onclick="forgetLogin('${esc(t.name)}')">Forget</button>`:''}
        <span id="loginMsg-${esc(t.name)}" class="note"></span>
      </div>
    </div>`;
  }).join('') + `<div class="note" style="margin-top:11px">${esc(body.note||'')}</div>`;
}

async function saveLogin(tool, fieldNames){
  const fields = {};
  fieldNames.forEach(n => {
    const el = $('login-'+tool+'-'+n);
    if(el && el.value) fields[n] = el.value;
  });
  const {ok, body} = await api('/api/console/logins', {method:'POST',
    body:JSON.stringify({tool, fields})});
  const msg = $('loginMsg-'+tool);
  if(!ok){
    msg.innerHTML = `<span class="neg">${esc(body.error||'could not save')}</span>`
      + ((body.problems||[]).length?` ${esc(body.problems.join('; '))}`:'');
    return;
  }
  msg.innerHTML = `<span class="pos">Saved (${esc((body.saved||[]).join(', '))}). The agent uses it on its next cycle.</span>`;
  loadLogins();
}

async function forgetLogin(tool){
  const {body} = await api('/api/console/logins/forget', {method:'POST',
    body:JSON.stringify({tool})});
  const msg = $('loginMsg-'+tool);
  if(msg) msg.innerHTML = `<span class="warn">${esc(body.error||'Forgotten. .env values, if any, still apply.')}</span>`;
  loadLogins();
}

// ---- venue switches ----
//
// The operator's answer to "can I stop it scanning the eleven venues with no
// client written yet". Off means off: not asked for markets, not traded, and the
// agent logs it as the operator's choice rather than a failure.
async function loadVenueSwitches(){
  const {body} = await api('/api/console/venues/enabled');
  const rows = body.venues || [];
  if(!rows.length){
    $('venueSwitches').innerHTML = `<div class="note">The agent has not recorded
      its venue inventory yet. Run one cycle and this list fills in.</div>`;
    return;
  }
  $('venueSwitches').innerHTML = `<table><thead><tr>
      <th>Venue</th><th>What it can do</th><th>Use it</th></tr></thead><tbody>`
    + rows.map(v => `<tr>
        <td><b>${esc(v.label||v.venue_id)}</b><div class="note">${esc(v.venue_id)}</div></td>
        <td class="note">${v.detail_known===false
              ? 'The agent has not been through this venue yet, so its live status is unknown. Run one cycle and this row fills in.'
              : esc(v.use||'')+(v.reads_live_markets_now?' <span class="pill ok">reads now</span>':' <span class="pill dim">no live read</span>')+(v.can_hold_real_money?'':' <span class="pill dim">paper only</span>')+(v.needs_credentials?' <span class="pill wait">needs login</span>':'')}
            <div class="note">${esc(v.what_it_needs||'')}</div></td>
        <td><button class="${v.enabled?'':'primary'}"
             onclick="setVenue('${esc(v.venue_id)}', ${v.enabled?'false':'true'})">
             ${v.enabled?'Switched on &mdash; turn off':'Switch on'}</button></td>
      </tr>`).join('') + `</tbody></table>
      <div class="note" style="margin-top:11px">${esc(body.note||'')}</div>`;
}

async function setVenue(venue_id, enabled){
  const {ok, body} = await api('/api/console/venues/enabled', {method:'POST',
    body:JSON.stringify({venue_id, enabled})});
  if(!ok){ alert(body.error||'could not save the switch'); return; }
  loadVenueSwitches();
}

// ---- why the fair value ----
//
// The chain, term by term. "The LLM says 65% and the ensemble says 58%" was
// unanswerable in this console; each component now shows its own number, the
// confidence it claimed, and the weight it got - including the components that
// contributed NOTHING, with the reason (no data, blocked, not in the shortlist).
async function loadForecast(){
  const {body} = await api('/api/console/forecast');
  if(!body.available){
    $('forecast').innerHTML = `<div class="note">${esc(body.reason||'unavailable')}</div>`;
    return;
  }
  const screen = body.deep_analysis || {};
  const br = body.base_rates || {};
  const rows = body.pricing || [];

  const lm = body.local_model || {};
  const modelLine = lm.describe
    ? `<div class="note">Local model: <b>${esc(lm.model || lm.describe)}</b>
        ${lm.used
          ? `&middot; ${lm.answered||0} of ${lm.calls||0} call(s) answered in
             ${(lm.seconds||0).toFixed ? (lm.seconds||0).toFixed(1) : lm.seconds}s`
          : `&middot; <b>NOT USED</b>: ${esc(lm.not_used_reason||'no reason recorded')}`}</div>`
    : '';
  const screenLine = screen.considered
    ? `<div class="note">Model time went to <b>${(screen.shortlist||[]).length}</b>
        of ${screen.considered} market(s) this cycle (limit ${screen.limit||'?'},
        books read in ${screen.seconds||0}s). ${screen.screened_out||0} were priced
        on their measured book only &mdash; not on model opinion.</div>`
    : `<div class="note">${esc(screen.criteria||'no screen has run yet')}</div>`;

  const cats = Object.entries(br.categories||{}).filter(e => e[1].usable);
  const baseLine = br.available
    ? `<div class="note">Base rates: real frequencies counted from
        ${(br.markets_read||0)} closed market(s)
        (${cats.map(e => `${esc(e[0])} ${(e[1].rate*100).toFixed(0)}% n=${e[1].n}`).join(', ')}).</div>`
    : `<div class="note warn">Base rates: ${esc(br.reason||'no dataset')}</div>`;

  const trace = rows.length ? rows.map(r => {
    const comps = (r.components||[]).map(c => `
      <tr><td class="mono">${esc(c.model)}${c.model_id
            ? `<span style="opacity:.6"> ${esc(c.model_id)}</span>` : ''}</td>
          <td class="mono">${Number(c.probability).toFixed(3)}</td>
          <td class="mono">${Number(c.confidence).toFixed(2)}</td>
          <td class="mono">${c.contributes
              ? Math.round(100*Number(c.weight_share||0))+'%'
              : '<span class="warn">0%</span>'}</td>
          <td class="note">${esc(c.note || (c.reasoning||'').slice(0,90))}</td></tr>`).join('');
    return `<div class="card" style="margin-bottom:11px">
      <div><b>${esc(r.market_id)}</b> <span class="note">${esc(r.question||'')}</span></div>
      <div class="mono note" style="margin:7px 0">${esc(r.explain||'')}</div>
      <table><thead><tr><th>Component</th><th>Probability</th><th>Confidence</th>
        <th>Weight</th><th>Why</th></tr></thead><tbody>${comps}</tbody></table>
    </div>`;
  }).join('') : `<div class="note">No market reached the forecast stage this
      cycle, so there is no chain to show. That is a statement about discovery,
      not about the models.</div>`;

  $('forecast').innerHTML = modelLine + screenLine + baseLine
    + `<div class="note" style="margin:7px 0 11px">${esc(screen.criteria||'')}</div>`
    + trace;
}

// ---- sports bets ----
//
// The lane used to end at "3 executable". This is the rest of the story: what was
// placed, at what price, and how it settled - with the ratings the models are
// actually using, and how many results each one is built on.
async function loadSports(){
  const {body} = await api('/api/console/sports');
  if(!body.available){
    $('sports').innerHTML = `<div class="note">${esc(body.reason||'unavailable')}</div>`;
    return;
  }
  const s = body.summary || {};
  const r = body.ratings || {};
  const kpis = [
    ['Staked', money(s.staked_usd), 'across every bet placed'],
    ['Open', s.open, 'waiting on a result'],
    ['Won / lost', `${s.won} / ${s.lost}`, 'settled on the final score'],
    ['Sports P&L', (s.pnl_usd>=0?'+':'')+Number(s.pnl_usd||0).toFixed(2),
     'paper money, through the same ledger'],
  ];
  const rows = (body.recent || []);
  const settled = body.last_settlement || s.last_settlement || {};
  $('sports').innerHTML = [
    `<div class="row" style="gap:12px;margin-bottom:11px">` + kpis.map(([k,v,sub])=>
      `<div class="card kpi" style="flex:1"><div class="k">${k}</div>
       <div class="v">${typeof v==='number'?v:esc(v)}</div>
       <div class="sub">${esc(sub)}</div></div>`).join('') + `</div>`,
    rows.length ? `<table><thead><tr><th>Bet</th><th>Market</th><th>Odds</th>
        <th>Stake</th><th>Status</th><th>P&amp;L</th></tr></thead><tbody>` +
      rows.map(b=>`<tr>
        <td>${esc(b.outcome)} <span class="note">${esc(b.book||'')}</span></td>
        <td class="note">${esc(b.league||'')} ${esc(b.market_type||'')}${b.line!=null?' '+b.line:''}</td>
        <td class="mono">${Number(b.odds||0).toFixed(2)}</td>
        <td class="mono">${money(b.stake_usd)}</td>
        <td><span class="pill ${b.status==='won'?'ok':b.status==='lost'?'no':'dim'}">${esc(b.status)}</span></td>
        <td class="mono ${(b.pnl_usd||0)>=0?'pos':'neg'}">${b.pnl_usd==null?'&mdash;':((b.pnl_usd>=0?'+':'')+Number(b.pnl_usd).toFixed(2))}</td>
      </tr>`).join('') + `</tbody></table>` :
      `<div class="empty">No sports bet has been placed yet. The lane needs a
        model view of a fixture and a price to beat: it builds ratings from
        finished results, and it will not claim an edge without them.</div>`,
    `<div class="note" style="margin-top:10px">Ratings: ${r.rated_teams||0} team(s)
       rated (${esc(r.min_matches||3)}+ results each) across
       ${Object.keys(r.leagues||{}).length} league(s). ${esc(r.how||'')}</div>`,
    settled.settled!=null?`<div class="note" style="margin-top:5px">Last settlement:
       ${settled.settled} bet(s) closed, ${settled.open||0} still open &mdash;
       ${esc(settled.note||'')}</div>`:'',
    `<div class="note" style="margin-top:5px">Settleable today: h2h, totals and
       (half/whole-line) handicaps &mdash; the markets whose result a score can
       decide. Corners, cards, both-teams-to-score and quarter-line handicaps are
       refused at placement with the reason on them, because accepting a bet whose
       result cannot be read is not risk, it is bookkeeping.</div>`,
  ].join('');
}

// ---- the agent: the front page, and the only question that matters ----

const SEVERITY_PILL = {critical:'no', loss:'no', next_action:'wait',
                       next_step:'wait', gate:'wait', warning:'wait',
                       waiting:'dim', ok:'ok'};
const STATE_PILL = {running:'ok', working:'wait', blocked:'wait',
                    not_running:'no', unknown:'dim'};
const STATE_WORD = {running:'RUNNING', working:'WORKING', blocked:'BLOCKED',
                    not_running:'STOPPED', unknown:'UNKNOWN'};

function ageText(seconds){
  if(seconds===null || seconds===undefined) return 'never';
  seconds = Math.max(0, Number(seconds));
  if(seconds < 90) return Math.round(seconds) + 's';
  if(seconds < 5400) return Math.round(seconds/60) + ' min';
  return (seconds/3600).toFixed(1) + 'h';
}

async function loadAgent(){
  const {body} = await api('/api/console/agent');
  if(!body || body.headline===undefined){
    $('agentHeadline').textContent = 'The agent state could not be read.';
    return;
  }
  const agt = body.agent || {};
  const cap = body.capital || {};
  const prof = body.profit || {};
  const paper = prof.paper || {};
  const risk = body.risk || {};
  const ven = body.venues || {};
  const st = body.strategies || {};

  STATE.mode = body.mode || STATE.mode;
  setBadge(STATE.mode);

  // header pill
  const pill = $('agentPill');
  pill.className = 'pill ' + (STATE_PILL[agt.state] || 'dim');
  pill.textContent = STATE_WORD[agt.state] || String(agt.state||'unknown').toUpperCase();

  // hero
  const heroPill = $('agentState');
  heroPill.className = 'hero-pill ' + (STATE_PILL[agt.state] || 'dim');
  heroPill.textContent = (STATE_WORD[agt.state] || 'UNKNOWN')
    + (agt.state==='running' && agt.last_scan_ago_seconds!=null
       ? ' \u00b7 LAST CYCLE ' + ageText(agt.last_scan_ago_seconds) + ' AGO'
       : agt.evidence ? ' \u00b7 ' + ageText(agt.phase_seconds || agt.heartbeat_ago_seconds || agt.last_scan_ago_seconds) + ' AGO' : '');
  $('agentHeadline').textContent = body.headline || '';
  $('agentDoing').innerHTML = agt.running
    ? '<b>Right now:</b> ' + esc(agt.doing || 'between cycles')
      + (agt.next_cycle_at ? ' \u00b7 next cycle ' + esc(String(agt.next_cycle_at).slice(11,16)) + ' UTC' : '')
    : '<b>Nothing is running.</b> ' + esc(agt.evidence || 'No agent process has left a mark in this database.')
      + ' Start <code>run_ptai.bat</code> on the machine that trades.';

  // vitals, in the operator's words
  const validated = ven.best_validated_venue
    ? esc(ven.best_validated_venue) + ' (' + esc(ven.best_validated_on||'') + ')'
    : 'none yet';
  const bestStrat = st.best_validated_strategy ? esc(st.best_validated_strategy) : 'none yet';
  $('agentVitals').innerHTML =
      'cycle every ' + esc(String(agt.interval_min||10)) + ' min<br>'
    + 'liveness window ' + Math.round((agt.window_seconds||900)/60) + ' min<br>'
    + 'heartbeat: ' + esc(agt.heartbeat_status || '\u2014') + ' ' + ageText(agt.heartbeat_ago_seconds) + '<br>'
    + 'last completed cycle: ' + ageText(agt.last_scan_ago_seconds) + '<br>'
    + 'validated venue: ' + validated + '<br>'
    + 'validated strategy: ' + bestStrat + '<br>'
    + 'risk: ' + (risk.trading_halted ? '<span class="neg">halted</span>'
        : 'kill switch ' + esc(String(risk.kill_switch_level==null?'none':risk.kill_switch_level)));

  // the money
  const pnl = cap.realised_pnl_usd;
  const pnlCls = (typeof pnl==='number') ? (pnl>=0?'pos':'neg') : '';
  const kpis = [
    ['Equity', money(cap.equity_usd), (cap.account==='paper'?'paper account':(cap.account||'')+' account')],
    ['Realised P&amp;L', money(pnl), (prof.live_resolved_trades||0) + ' resolved live trade(s), class:pnlCls'],
    ['Free capital', money(cap.free_cash_usd), 'not committed anywhere'],
    ['Reserved', money(cap.reserved_capital_usd), 'locked behind working orders'],
    ['Deployed', (cap.deployment_pct==null?'\u2014':Number(cap.deployment_pct).toFixed(1)+'%'),
      money(cap.deployed_usd) + ' working'],
    ['30-day net return', (prof.return_30d_pct==null?'\u2014':Number(prof.return_30d_pct).toFixed(2)+'%'),
      (prof.return_30d_pct==null?'undefined until a resolved history exists':'on the live account')],
    ['Max drawdown', (prof.max_drawdown_pct==null?'\u2014':Number(prof.max_drawdown_pct).toFixed(2)+'%'),
      (prof.max_drawdown_pct==null?'no equity curve to draw down yet':'worst peak-to-trough')],
    ['Paper (kept apart)', money(paper.net_pnl),
      (paper.settled_trades||0) + ' settled simulated trade(s)'],
  ];
  $('agentKpis').innerHTML = kpis.map(([k,v,s])=>{
    const cls = (s||'').indexOf('class:pnlCls')>=0 ? ' '+pnlCls : '';
    return `<div class="card kpi"><div class="k">${k}</div>
      <div class="v${cls}">${v}</div>
      <div class="sub">${(s||'').replace(' class:pnlCls','')}</div></div>`;
  }).join('');

  // what stands in the way - the reason this screen exists
  $('blockers').innerHTML = (body.blockers||[]).length
    ? body.blockers.map(b=>`
      <div class="blocker">
        <div><span class="pill ${SEVERITY_PILL[b.severity]||'dim'}">${esc(String(b.severity||'').replace('_',' '))}</span>
             <b style="margin-left:8px">${esc(b.what)}</b></div>
        <div class="note" style="margin-top:5px">${esc(b.evidence||'')}</div>
        <div class="clear" style="margin-top:6px">&#8594; ${esc(b.clear)}</div>
      </div>`).join('')
    : '<div class="empty">No blockers were computed.</div>';

  // last cycle
  const lc = body.last_cycle || {};
  if(!lc.available){
    $('lastCycle').innerHTML = `<div class="empty">${esc(lc.note||'No completed cycle is recorded.')}</div>`;
  } else {
    const dec = lc.decided || {};
    const orders = lc.orders || {};
    $('lastCycle').innerHTML = `
      <div><span class="pill ${lc.verdict==='DEPLOYED'?'ok':'dim'}">${esc(lc.verdict||'')}</span>
        <span class="note" style="margin-left:9px">${esc(String(lc.at||'').replace('T',' ').slice(0,19))} UTC
        &middot; took ${esc(String(lc.cycle_seconds||'?'))}s</span></div>
      <table style="margin-top:10px">
        <tr><td style="color:var(--dim)">Markets scanned</td>
            <td class="mono">${esc(String(lc.markets_scanned||0))} across ${esc(String(lc.venues_searched||0))} venue(s)</td></tr>
        <tr><td style="color:var(--dim)">Candidates</td>
            <td class="mono">${esc(String(lc.candidates||0))}</td></tr>
        <tr><td style="color:var(--dim)">Positions recorded</td>
            <td class="mono">${esc(String(orders.positions_recorded||0))}</td></tr>
        <tr><td style="color:var(--dim)">Blocked by capital boundary</td>
            <td class="mono">${esc(String(orders.blocked_live_capital||0))}${orders.blocked_reason?` <span style="color:var(--dim)">&mdash; ${esc(orders.blocked_reason)}</span>`:''}</td></tr>
        <tr><td style="color:var(--dim)">Best it found</td>
            <td class="mono">${dec.venue?esc(dec.venue)+' &middot; '+esc(dec.strategy||''):'&mdash;'}</td></tr>
        <tr><td style="color:var(--dim)">Edge on it</td>
            <td class="mono">${dec.edge!=null?esc(String(dec.edge)):'&mdash;'}</td></tr>
      </table>
      ${lc.why?`<div class="note" style="margin-top:10px"><b>Why:</b> ${esc(lc.why)}</div>`:''}`;
  }

  renderBrain($('brainBox'), body.brain||{}, false);
  window.__BRAIN = body.brain || {};
  window.__INTERVAL = agt.interval_min || 10;
}

function renderBrain(el, b, withPicker){
  if(!el) return;
  const active = b.active_model;
  const pinNote = b.pinned
    ? 'pinned in .env &mdash; this is the model the agent calls'
    : 'auto: whichever model LM Studio lists first';
  const head = b.connected
    ? `<div class="pill ok">CONNECTED</div>
       <span class="note" style="margin-left:9px">${esc(String(b.models.length))} model(s) loaded${b.latency_ms!=null?' &middot; '+esc(String(b.latency_ms))+' ms':''}</span>`
    : `<div class="pill no">NO ANSWER</div>
       <span class="note" style="margin-left:9px">${esc(b.host||'http://localhost:1234')}: ${esc(b.error||'not reachable')}</span>`;
  const table = `
    <table style="margin-top:11px">
      <tr><td style="color:var(--dim);width:170px">Agent will use</td>
          <td class="mono ${active?'pos':''}">${active?esc(active):'nothing - no model is loaded'}</td></tr>
      <tr><td style="color:var(--dim)">How it was chosen</td>
          <td class="note">${esc(b.model_reason || pinNote)}</td></tr>
      <tr><td style="color:var(--dim)">Speed</td>
          <td class="mono">${b.is_r1?'<span class="neg">SLOW - minutes per market (R1-style)</span>'
              :(b.connected?'<span class="pos">full speed - no long thinking phase</span>':'&mdash;')}</td></tr>
      <tr><td style="color:var(--dim)">One call waits at most</td>
          <td class="mono">${b.timeout_seconds?Number(b.timeout_seconds).toFixed(0)+'s':'&mdash;'}
            <span class="note"> - a slower model then gives no forecast for that market instead of holding the cycle</span></td></tr>
    </table>`;
  const picker = (withPicker && b.connected && b.models.length) ? `
    <div style="margin-top:13px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="modelSelect" style="max-width:340px;width:auto">${
        b.models.map(m=>`<option value="${esc(m)}"${m===active?' selected':''}>${esc(m)}</option>`).join('')}</select>
      <button class="primary" onclick="pinModel()">Pin this model</button>
      <label style="margin:0 0 0 8px;color:var(--dim);font-size:12px">Model call limit (s)</label>
      <input id="timeoutSeconds" type="number" min="10" max="3600" step="10"
             value="${b.timeout_seconds?Number(b.timeout_seconds).toFixed(0):180}"
             style="width:90px">
      <button onclick="saveTimeout()">Save</button>
    </div>
    <div id="pinMsg" class="note" style="margin-top:9px"></div>
    <div class="note" style="margin-top:9px">Pinning writes <code>LM_STUDIO_MODEL</code> to
      <code>.env</code> so the agent stops taking whichever model is listed first.
      The running agent picks it up on its <b>next start</b>.</div>`
    : (withPicker ? `<div class="note" style="margin-top:11px">Start LM Studio and load a
        model with the local server on, then refresh, to pin one from here.</div>` : '');
  el.innerHTML = head + table + picker;
}

async function pinModel(){
  const sel = $('modelSelect');
  if(!sel){ return; }
  const {ok, body} = await api('/api/console/brain', {method:'POST',
    body:JSON.stringify({model: sel.value})});
  $('pinMsg').innerHTML = ok
    ? `<span class="pos">${esc(body.note||'pinned')}</span>`
    : `<span class="neg">${esc(body.error||'could not pin')}</span> ${esc(body.note||'')}`;
  if(ok){ loadBrainSetup(); loadAgent(); }
}

async function saveTimeout(){
  const raw = $('timeoutSeconds').value;
  const {ok, body} = await api('/api/console/brain', {method:'POST',
    body:JSON.stringify({timeout_seconds: Number(raw)})});
  $('pinMsg').innerHTML = ok
    ? `<span class="pos">${esc(body.note||'saved')}</span>`
    : `<span class="neg">${esc(body.error||'could not save')}</span> ${esc(body.note||'')}`;
  if(ok) loadBrainSetup();
}

async function loadBrainSetup(){
  const {body} = await api('/api/console/agent');
  window.__BRAIN = body.brain || {};
  renderBrain($('brainSetup'), body.brain||{}, true);
  const iv = $('setupInterval');
  if(iv && body.agent) iv.textContent = String(body.agent.interval_min||10);
}

let _loading = false;
async function loadAll(){
  // One round at a time. Eight endpoints per round taking longer than the timer
  // is how a page ends up with several rounds in flight at once.
  if(_loading) return;
  // Only the tab the operator is looking at polls. Every open console tab used
  // to ask for all eight panels every fifteen seconds; five tabs left open meant
  // forty requests a round, all of them logging.
  if(document.visibilityState === 'hidden') return;
  _loading = true;
  try{
    await Promise.all([
      loadAgent(), loadStatus(), loadBrainSetup(), loadControl(),
      loadVenue(), loadCapital(), loadFunding(), loadOrders(), loadResults(),
      loadLogins(), loadVenueSwitches(), loadSports(), loadForecast(),
      loadRounds(),
    ]);
    spyScroll();
  } finally {
    _loading = false;
  }
}

// Everything, on open and on the timer. The panels are read-only views of the
// database; the only writable thing on the page is the budget box, and its
// input is never re-rendered by a refresh.
loadAll();
// Polling follows the work. A running agent's round card, log and pill change by
// the second, so a fixed 15-second timer made every action look like it "took too
// long to change". When nothing is running there is nothing to watch, so the page
// backs off and leaves the machine alone.
const POLL_FAST_MS = 3000, POLL_SLOW_MS = 15000, LOG_MS = 2500;
let _pollMs = POLL_SLOW_MS;

function agentLooksBusy(){
  const c = ENGINE.control || {};
  return !!(c.console_hosting || (c.lease && c.lease.held) || c.cycle_running ||
            (c.agent_state && c.agent_state !== 'stopped' && c.agent_state !== 'idle'));
}

async function pollLoop(){
  const started = Date.now();
  try{
    if(document.visibilityState === 'hidden'){ await loadLogs(); return; }
    await loadLogs();
    await loadControl();
    _pollMs = agentLooksBusy() ? POLL_FAST_MS : POLL_SLOW_MS;
    if(_pollMs === POLL_FAST_MS){ await loadAll(); }
  }catch(e){
    _pollMs = POLL_SLOW_MS;
  }
  // Keep the cadence even when a poll itself is slow: eight endpoints must not
  // turn a 3-second watch into a 9-second one.
  const spent = Date.now() - started;
  setTimeout(pollLoop, Math.max(500, _pollMs - spent));
}

setInterval(loadLogs, LOG_MS);
setTimeout(pollLoop, 600);
// Coming back to a backgrounded tab refreshes immediately instead of waiting
// out the rest of the interval.
document.addEventListener('visibilitychange', () => { if(document.visibilityState === 'visible') loadAll(); });
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8101) -> None:
    """
    Serve the console - the one screen the product has.

    The port comes from the environment first, because the runner sets it in one
    place (`PTAI_DASHBOARD_PORT`, the name already in the operator's .bat and in
    the docs) and every process the runner starts has to agree on it. PTAI is
    local-only either way: this binds on the machine that runs it.
    """
    import os
    chosen = (os.environ.get("PTAI_CONSOLE_PORT")
              or os.environ.get("PTAI_DASHBOARD_PORT") or port)
    try:
        port = int(chosen)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring unusable port {chosen!r}; using {port}")
    import uvicorn
    logger.info(f"PTAI Console on http://localhost:{port}")
    logger.info("Paper mode is the default and needs no capital or credentials.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
