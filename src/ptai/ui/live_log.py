"""
The agent's log, on the page.

The operator's 2026-09-29 report - "the webui is too disconnected with the
command line" - was partly literal: the only place to READ the agent was the
"PTAI Agent (paper)" console window, and the only place to CONTROL it was the
browser. Two windows, two half-views of one machine.

This is the bridge. It keeps the most recent log records in memory in the
console process (which is now the process that runs the agent), so the page can
show the same lines the command line shows - including the ones that matter most
to a diagnosis: a refused order, a venue that did not answer, a model that timed
out.

Deliberately a ring buffer, not a file: this is a view of the running process,
it costs nothing to keep, and it cannot grow without bound. `logs/` on disk stays
the durable record.
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import timezone
from types import ModuleType
from typing import Any, Dict, List

from loguru import logger

# How many lines the page can scroll back through. The console asks for 200 at a
# time; the rest is for an operator who leaves the page open and comes back.
MAX_LINES = 800

# The buffer lives in ONE process-wide place, not in this module's globals.
#
# Reason: the same file can be imported twice in one process - as
# `ptai.ui.live_log` (how `python -m ptai.ui.console` imports it) and as
# `src.ptai.ui.live_log` (how a test that puts `src` on the path sees it). Two
# module objects would mean two buffers, two sinks, and a page that shows half the
# log - the tests caught exactly that. A module registered under a fixed name in
# `sys.modules` is the one object both import paths can agree on.
_STATE_NAME = "_ptai_live_log_state"


def _state() -> ModuleType:
    state = sys.modules.get(_STATE_NAME)
    if state is None:
        state = ModuleType(_STATE_NAME)
        state.buffer = deque(maxlen=MAX_LINES)
        state.seq = 0
        state.installed = False
        sys.modules[_STATE_NAME] = state
    return state


def _sink(message) -> None:
    """One loguru record, flattened into something the page can render."""
    try:
        state = _state()
        record = message.record
        state.seq += 1
        state.buffer.append({
            "seq": state.seq,
            "at": record["time"].astimezone(timezone.utc).isoformat(),
            "time": record["time"].strftime("%H:%M:%S"),
            "level": record["level"].name,
            "where": f"{record['name']}",
            "text": str(message).strip(),
        })
    except Exception:  # noqa: BLE001 - a log sink must never raise into the log
        pass


def _sink_is_attached() -> bool:
    """
    Is our sink still in loguru's handler list?

    `logger.remove()` anywhere in the process takes every handler with it,
    including this one, and a page that shows an empty log with no explanation is
    the kind of "missing" thing the operator reported. Re-adding it is cheap; the
    duplicated-lines risk is handled by clearing the buffer when we re-attach.
    """
    try:
        for handler in logger._core.handlers.values():  # type: ignore[attr-defined]
            if getattr(handler, "_ptai_live_log", False):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _add_sink(level: str) -> None:
    """Attach the buffer to loguru and mark the handler as ours."""
    state = _state()
    sink_id = logger.add(_sink, level=level,
                         format="{time:HH:mm:ss} | {level: <7} | {name} - {message}",
                         enqueue=False, backtrace=False, diagnose=False)
    try:
        handler = logger._core.handlers.get(sink_id)  # type: ignore[attr-defined]
        if handler is not None:
            handler._ptai_live_log = True
    except Exception:  # noqa: BLE001
        pass
    state.installed = True


def install(level: str = "INFO") -> bool:
    """
    Attach the buffer to loguru once per process.

    Idempotent on purpose: the console module is imported and reloaded by tests,
    and a second sink would double every line in the buffer.
    """
    state = _state()
    if state.installed and _sink_is_attached():
        return False
    if state.installed:
        # Our sink is gone: something called `logger.remove()`, which is what
        # `import ptai.cli` does at import time. Re-attach and KEEP the buffer.
        #
        # It used to clear the buffer here, on the theory that a re-attach
        # could duplicate lines. It cannot: `_add_sink` is only reached when no
        # handler is flagged as ours, so there is never a second ring sink.
        # What clearing did do was delete the lines the operator was reading -
        # a page that silently loses its log the moment the CLI is imported in
        # the same process is exactly the "things are missing" complaint this
        # module exists to answer.
        logger.debug(
            f"Log ring re-attaching after the sink was removed; keeping "
            f"{len(state.buffer)} buffered line(s)")
    try:
        _add_sink(level)
        return True
    except Exception:  # noqa: BLE001
        return False


def tail(after: int = 0, limit: int = 200) -> Dict[str, Any]:
    """
    Lines newer than `after`, oldest first, so the page can append what is new.

    `after=0` returns the most recent `limit` lines: the page's first paint wants
    the tail, not the beginning of the buffer.

    A READ also re-attaches the sink when it is gone, so the page can never end
    up showing an empty log because some library - `ptai.cli` does it on import -
    called `logger.remove()` in this process.
    """
    install()
    try:
        wanted = max(1, min(int(limit), MAX_LINES))
    except (TypeError, ValueError):
        wanted = 200
    try:
        since = int(after)
    except (TypeError, ValueError):
        since = 0
    state = _state()
    lines: List[Dict[str, Any]] = list(state.buffer)
    if since > 0:
        fresh = [line for line in lines if line["seq"] > since]
    else:
        fresh = lines[-wanted:]
    return {
        "lines": fresh[-wanted:],
        "next": state.seq,
        "buffered": len(state.buffer),
        "capacity": MAX_LINES,
        "note": ("the console process's own log buffer; the durable log is the "
                 "file in logs/"),
    }
