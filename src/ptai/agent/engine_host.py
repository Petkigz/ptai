"""
Who is running the agent - one owner at a time - and the settings it re-reads.

The operator's 2026-09-29 report: "the webui is too disconnected with the command
line. things take too long to change or dont change at all." The cause was not
cosmetic. `run_ptai.bat` started TWO processes:

    start "PTAI Agent (paper)"  cmd /k "python main.py run --bankroll 50 --interval 10"
    start "PTAI Console"        cmd /k "python -m ptai.ui.console"

and the console's "Run one round" button built a THIRD engine inside the web
process. Three consequences, all of them the operator's complaint:

  * two engines wrote the same liveness keys (`agent.phase`, `agent.heartbeat`)
    and the same round history, so the screen could attribute one engine's work
    to the other, and neither knew the other existed;
  * `--interval 10` and `--dry-run` were fixed at process start, so the mode
    switch and every interval change in the web page did nothing to the agent
    that was actually running - they took effect only after a restart, or never;
  * nothing in the page could start or stop the agent at all. Its lifecycle was
    a console window.

This module is the two facts that fix that, shared by every entry point:

  * THE LEASE. One engine owns the agent at a time, recorded in the database
    with its pid, host and a heartbeat. A second starter is refused by name and
    pid rather than silently running a rival engine. A lease the operator stops
    on purpose is marked RELEASED, so the page can say "stopped by the operator
    at 10:31" instead of the much worse "no sign of life".
  * THE INTERVAL. The operator's chosen cycle interval lives in the database,
    where the running loop re-reads it, instead of in an environment variable
    that only a new process would ever see.

`execute/capital.py` already owns the mode and the budgets for exactly this
reason; this is the same rule for the loop itself.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

# The one key. `agent.*` is the namespace the loop writes and the console reads.
LEASE_KEY = "agent.engine_lease"
INTERVAL_KEY = "operator.interval_min"

DEFAULT_INTERVAL_MIN = 10
MIN_INTERVAL_MIN = 1
MAX_INTERVAL_MIN = 24 * 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(stamp: Optional[str]) -> Optional[float]:
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds()


def here() -> Dict[str, Any]:
    """This process, as the lease names it."""
    return {"pid": os.getpid(), "host": socket.gethostname()}


def _env_interval_minutes() -> int:
    try:
        return max(MIN_INTERVAL_MIN, int(os.environ.get("INTERVAL_MIN")
                                         or DEFAULT_INTERVAL_MIN))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_MIN


# ---------------------------------------------------------------------------
# the interval the loop actually uses
# ---------------------------------------------------------------------------

def operator_interval_minutes(storage, default: Optional[int] = None) -> int:
    """
    The interval for the NEXT wait: the operator's setting when there is one,
    the environment otherwise.

    Read fresh at every cycle by `run_continuous`, which is what makes a change
    in the page take effect without restarting anything. It used to be a
    constructor/CLI argument, so `--interval 10` was the value forever.
    """
    if storage is not None:
        try:
            raw = storage.get_state(INTERVAL_KEY)
            if raw:
                return max(MIN_INTERVAL_MIN, min(MAX_INTERVAL_MIN, int(float(raw))))
        except (TypeError, ValueError):
            logger.warning(f"Ignoring unusable stored interval {raw!r}")
        except Exception as e:  # noqa: BLE001 - an unreadable setting is the env's
            logger.debug(f"Could not read the operator interval: "
                         f"{type(e).__name__}: {e}")
    if default is not None:
        try:
            return max(MIN_INTERVAL_MIN, min(MAX_INTERVAL_MIN, int(default)))
        except (TypeError, ValueError):
            pass
    return _env_interval_minutes()


def set_operator_interval_minutes(storage, minutes: int) -> int:
    """Store the operator's interval. Refuses nonsense instead of clamping it silently."""
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        raise ValueError("the interval must be a whole number of minutes")
    if not (MIN_INTERVAL_MIN <= value <= MAX_INTERVAL_MIN):
        raise ValueError(f"the interval must be between {MIN_INTERVAL_MIN} and "
                         f"{MAX_INTERVAL_MIN} minutes")
    storage.set_state(INTERVAL_KEY, str(value))
    logger.info(f"Operator interval set to {value} minute(s) - the agent uses it "
                f"for its next wait")
    return value


def interval_source(storage) -> Dict[str, Any]:
    """Where the current interval comes from, for the page to show honestly."""
    stored = None
    if storage is not None:
        try:
            raw = storage.get_state(INTERVAL_KEY)
            stored = int(float(raw)) if raw else None
        except Exception:  # noqa: BLE001
            stored = None
    env = _env_interval_minutes()
    return {
        "minutes": operator_interval_minutes(storage),
        "source": "set in this console" if stored else "the environment (INTERVAL_MIN)",
        "environment_minutes": env,
    }


# ---------------------------------------------------------------------------
# the lease: one engine at a time
# ---------------------------------------------------------------------------

def engine_lease_status(storage, interval_min: Optional[int] = None,
                        stale_after_seconds: Optional[float] = None) -> Dict[str, Any]:
    """
    Who owns the agent, and is that claim still alive.

    Freshness uses the SAME window the console uses to decide the agent is
    alive, so "the lease is fresh" and "the agent is running" cannot disagree.
    The lease is renewed on every phase write (which happens for every market
    the cycle touches) and at every cycle start, so a long cycle does not look
    like an abandoned lease.
    """
    block: Dict[str, Any] = {
        "held": False, "is_self": False, "kind": None, "pid": None, "host": None,
        "started_at": None, "heartbeat_at": None, "age_seconds": None,
        "released": False, "released_reason": None, "released_at": None,
        "stale_after_seconds": None, "stale": True, "note": None,
    }
    if storage is None:
        block["note"] = "no storage to read"
        return block

    raw = None
    try:
        raw = storage.get_state(LEASE_KEY)
    except Exception as e:  # noqa: BLE001
        block["note"] = f"the lease could not be read: {type(e).__name__}: {e}"
        return block
    if not raw:
        block["note"] = "no engine has claimed the agent yet"
        return block
    try:
        lease = json.loads(raw)
    except (TypeError, ValueError):
        block["note"] = "the lease row is unreadable; treating the agent as unowned"
        return block

    if stale_after_seconds is None:
        try:
            from ..operator_view import live_window_seconds
            stale_after_seconds = live_window_seconds(interval_min)
        except Exception:  # noqa: BLE001
            stale_after_seconds = 900.0
    block["stale_after_seconds"] = round(float(stale_after_seconds), 1)

    block.update({
        "kind": lease.get("kind"), "pid": lease.get("pid"),
        "host": lease.get("host"), "started_at": lease.get("started_at"),
        "heartbeat_at": lease.get("heartbeat_at") or lease.get("started_at"),
        "released": bool(lease.get("released")),
        "released_reason": lease.get("released_reason"),
        "released_at": lease.get("released_at"),
    })
    block["age_seconds"] = _age_seconds(block["heartbeat_at"])
    me = here()
    block["is_self"] = bool(block["pid"] == me["pid"]
                            and block["host"] == me["host"])
    block["stale"] = bool(block["age_seconds"] is None
                          or block["age_seconds"] > block["stale_after_seconds"])

    if block["released"]:
        block["held"] = False
        block["note"] = (f"the agent was stopped on purpose at "
                         f"{block['released_at']} ({block['released_reason']})")
    elif block["stale"]:
        block["held"] = False
        block["note"] = (f"an engine ({block['kind']}, pid {block['pid']} on "
                         f"{block['host']}) last showed life "
                         f"{block['age_seconds']:.0f}s ago, outside the window - "
                         f"treating it as gone")
    else:
        block["held"] = True
        block["note"] = (f"owned by the {block['kind']} engine, pid {block['pid']} "
                         f"on {block['host']}, alive "
                         f"{block['age_seconds']:.0f}s ago")
    return block


def claim_engine_lease(storage, kind: str, interval_min: Optional[int] = None,
                       force: bool = False) -> Dict[str, Any]:
    """
    Take ownership, or refuse and name the engine that has it.

    `force=True` is the operator overriding a lease that still looks fresh - it
    is offered on the page, after the evidence, rather than being automatic: a
    silent takeover would put two engines on one database again, which is the
    bug this exists to prevent.
    """
    status = engine_lease_status(storage, interval_min=interval_min)
    if status["held"] and not status["is_self"] and not force:
        return {"claimed": False, "holder": status,
                "reason": (f"the agent is already running in the {status['kind']} "
                           f"engine (pid {status['pid']} on {status['host']})")}
    mine = here()
    lease = {
        "kind": kind, "pid": mine["pid"], "host": mine["host"],
        "started_at": _now(), "heartbeat_at": _now(),
        "released": False, "released_reason": None, "released_at": None,
    }
    if force and status["held"] and not status["is_self"]:
        lease["took_over_from"] = {"kind": status["kind"], "pid": status["pid"],
                                   "host": status["host"],
                                   "last_seen_seconds_ago": status["age_seconds"]}
        logger.warning(
            f"Taking the agent from the {status['kind']} engine (pid "
            f"{status['pid']} on {status['host']}, last seen "
            f"{status['age_seconds']:.0f}s ago) at the operator's request")
    storage.set_state(LEASE_KEY, json.dumps(lease))
    return {"claimed": True, "lease": lease, "holder": status}


def renew_engine_lease(storage, kind: Optional[str] = None) -> bool:
    """
    Keep the claim alive. Called from the loop's phase writes, so a cycle that
    runs for minutes keeps renewing without anyone remembering to.

    Only the owning process renews: a second engine's stray phase write must not
    keep the first one's lease warm.
    """
    if storage is None:
        return False
    try:
        raw = storage.get_state(LEASE_KEY)
        if not raw:
            return False
        lease = json.loads(raw)
        mine = here()
        if lease.get("pid") != mine["pid"] or lease.get("host") != mine["host"]:
            return False
        if lease.get("released"):
            return False
        if kind and lease.get("kind") != kind:
            lease["kind"] = kind
        lease["heartbeat_at"] = _now()
        storage.set_state(LEASE_KEY, json.dumps(lease))
        return True
    except Exception as e:  # noqa: BLE001 - never break a cycle over the lease
        logger.debug(f"Could not renew the engine lease: {type(e).__name__}: {e}")
        return False


def release_engine_lease(storage, reason: str = "stopped by the operator") -> None:
    """
    Give the agent back, saying it was deliberate.

    The difference between "stopped" and "died" is the whole reason this is
    recorded: a released lease keeps the page honest for as long as the operator
    leaves it, instead of decaying into "no sign of life in 20 minutes".
    """
    if storage is None:
        return
    try:
        raw = storage.get_state(LEASE_KEY)
        lease = json.loads(raw) if raw else {}
        mine = here()
        mine_now = bool(lease.get("pid") == mine["pid"]
                        and lease.get("host") == mine["host"])
        if lease and not mine_now and not lease.get("released"):
            # Another process owns it; it releases its own lease when it stops.
            return
        lease.update({"released": True, "released_reason": reason,
                      "released_at": _now(), "heartbeat_at": _now()})
        storage.set_state(LEASE_KEY, json.dumps(lease))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not release the engine lease: {type(e).__name__}: {e}")
