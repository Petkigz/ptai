"""
The operator's view of the agent, in one place.

The operator asked a specific list of questions, and a system that answers them
differently in the CLI, in the dashboard and in a status file is three systems.
This module composes ONE snapshot from the same sources the loop uses, and both
front ends read it:

  * which strategy is running, and which venue is live (and why),
  * which positions are open,
  * how much of the capital is deployed,
  * the current risk state,
  * what the agent decided last cycle,
  * the profit so far - real, and the simulation kept separate,
  * the figures the operator asked PTAI to expose: equity, realised P&L, free
    capital, reserved capital, 30-day net return, max drawdown, best validated
    strategy, best validated venue, risk state, capital deployment.

Two rules it follows, because breaking either has already cost this project a
round of rework:

  1. It NEVER reports a figure it cannot source. Every block carries the name of
     where it came from, and a block with nothing to read says so rather than
     showing a zero that looks like a measurement.
  2. Live and paper are never summed. A paper bankroll is not the operator's
     money, and a paper win is not evidence about a venue - so the simulation is
     reported next to the real account, labelled, never inside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# The key the loop writes its own decision under. One key, one meaning, so the
# console cannot disagree with the cycle about what happened.
LAST_CYCLE_KEY = "operator.last_cycle"


def _storage_or_none(storage):
    if storage is not None:
        return storage
    try:
        from .storage.db import Storage
        return Storage()
    except Exception as e:
        logger.warning(f"Operator view has no storage to read: {e}")
        return None


def _connectivity(storage) -> Dict[str, Any]:
    """What the loop decided last cycle, from storage - not from memory."""
    block: Dict[str, Any] = {
        "available": False,
        "source": f"state key {LAST_CYCLE_KEY}",
        "note": ("no completed cycle is recorded yet - the agent has not run, or "
                 "it has not finished one since this database was created"),
    }
    if storage is None:
        return block
    raw = storage.get_state(LAST_CYCLE_KEY)
    if not raw:
        return block
    try:
        data = json.loads(raw)
    except Exception as e:
        block["note"] = f"the recorded cycle could not be read: {e}"
        return block
    if isinstance(data, dict):
        block["available"] = True
        block.pop("note", None)
        block.update(data)
    return block


def _capital(storage) -> Dict[str, Any]:
    """
    Equity, free cash, reserved capital, deployment, realised P&L.

    Built through the same ledger builder the loop and the CLI use, so the three
    cannot drift. A ledger that cannot be built reports why instead of zeros.
    """
    if storage is None:
        return {"available": False, "source": "position_ledger", "reason": "no storage"}
    try:
        from .execution.position_ledger import PositionLedgerBuilder
        ledger = PositionLedgerBuilder(storage=storage).build()
    except Exception as e:
        logger.warning(f"Operator view could not build the capital ledger: {e}")
        return {"available": False, "source": "position_ledger",
                "reason": f"{type(e).__name__}: {e}"}

    # The account the operator is running. In paper, the account on screen is
    # the PAPER account - its bankroll, its positions, its free cash - because
    # that is the capital the operator is watching grow. Showing the live
    # account's $50 flat while the paper account does the work would hide the
    # very trial the operator is running.
    mode = _operator_mode(storage)
    live_positions = int(getattr(ledger, "live_position_count", 0) or 0)
    paper_positions = int(getattr(ledger, "paper_position_count", 0) or 0)
    if mode == "paper":
        equity = float(getattr(ledger, "paper_equity", 0.0) or 0.0)
        free = float(getattr(ledger, "paper_free_cash", 0.0) or 0.0)
        reserved = float(getattr(ledger, "paper_position_cost", 0.0) or 0.0)
        position_value = float(getattr(ledger, "paper_position_value", 0.0) or 0.0)
        resting = float(getattr(ledger, "paper_resting_order_cost", 0.0) or 0.0)
        committed = position_value + resting
        try:
            row = storage.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades "
                "WHERE resolved = 1 AND COALESCE(execution_mode,'live')='paper'"
            ).fetchone()
            realised = float(row["total"] or 0.0) if row else 0.0
        except Exception:
            realised = 0.0
        return {
            "available": True,
            "source": "position_ledger",
            "account": "paper",
            "equity_usd": round(equity, 2),
            "free_cash_usd": round(free, 2),
            "reserved_capital_usd": round(reserved, 2),
            "open_position_value_usd": round(position_value, 2),
            "resting_order_cost_usd": round(resting, 2),
            "realised_pnl_usd": round(realised, 2),
            "unrealised_pnl_usd": 0.0,
            "deployment_pct": (round(committed / equity * 100, 1) if equity else 0.0),
            "deployed_usd": round(committed, 2),
            # The live account is still reported by its counts - the
            # operator's paper trial must not make real positions invisible.
            "live_positions": live_positions,
            "paper_positions": paper_positions,
            "warnings": list(getattr(ledger, "warnings", []) or []),
        }

    equity = float(getattr(ledger, "equity", 0.0) or 0.0)
    committed = (float(getattr(ledger, "open_position_value", 0.0) or 0.0)
                 + float(getattr(ledger, "resting_order_cost", 0.0) or 0.0))
    return {
        "available": True,
        "source": "position_ledger",
        "account": "live",
        "equity_usd": round(equity, 2),
        "free_cash_usd": round(float(getattr(ledger, "free_cash", 0.0) or 0.0), 2),
        "reserved_capital_usd": round(
            float(getattr(ledger, "reserved_capital", 0.0) or 0.0), 2),
        "open_position_value_usd": round(
            float(getattr(ledger, "open_position_value", 0.0) or 0.0), 2),
        "realised_pnl_usd": round(
            float(getattr(ledger, "realised_pnl", 0.0) or 0.0), 2),
        "unrealised_pnl_usd": round(
            float(getattr(ledger, "unrealised_pnl", 0.0) or 0.0), 2),
        # How full the account is, as the operator means it: the share of equity
        # that is working - positions plus capital resting in open orders -
        # against the share sitting idle. Equity of 0 reports 0, not a division
        # by zero dressed up as a percentage.
        "deployment_pct": (round(committed / equity * 100, 1) if equity else 0.0),
        "deployed_usd": round(committed, 2),
        "live_positions": int(getattr(ledger, "live_position_count", 0) or 0),
        "paper_positions": int(getattr(ledger, "paper_position_count", 0) or 0),
        "warnings": list(getattr(ledger, "warnings", []) or []),
    }


def _profit(storage) -> Dict[str, Any]:
    """The real account's profit, with the simulation beside it, never inside."""
    if storage is None:
        return {"available": False, "source": "storage", "reason": "no storage"}
    try:
        perf = storage.get_performance_summary()
    except Exception as e:
        return {"available": False, "source": "storage",
                "reason": f"{type(e).__name__}: {e}"}
    history = perf.get("history") or []
    return_30d = None
    max_drawdown_pct = None
    if len(history) >= 2:
        # The LIVE curve only - the reader filters paper out, so this cannot mix
        # the simulation into the operator's return.
        points: List[tuple] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for row in history:
            try:
                value = float(row.get("amount"))
                stamp = row.get("timestamp")
                when = (datetime.fromisoformat(str(stamp))
                        if stamp else None)
            except (TypeError, ValueError):
                continue
            if when is not None and when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            points.append((when or cutoff, value))
        if len(points) >= 2:
            first, last = points[0][1], points[-1][1]
            if first > 0:
                return_30d = round((last - first) / first * 100, 2)
            peak = points[0][1]
            worst = 0.0
            for _, value in points:
                peak = max(peak, value)
                if peak > 0:
                    worst = min(worst, (value - peak) / peak)
            max_drawdown_pct = round(worst * 100, 2)
    return {
        "available": True,
        "source": "storage.get_performance_summary (LIVE only)",
        "initial_usd": round(float(perf.get("initial_bankroll", 0.0) or 0.0), 2),
        "bankroll_usd": round(float(perf.get("bankroll", 0.0) or 0.0), 2),
        "net_pnl_usd": round(float(perf.get("total_pnl", 0.0) or 0.0), 2),
        "net_pnl_pct": round(float(perf.get("total_pnl_pct", 0.0) or 0.0), 2),
        "return_30d_pct": return_30d,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": round(float(perf.get("win_rate", 0.0) or 0.0), 1),
        # RESOLVED, not "taken". This field was reading the trades-taken
        # counter, so a live trade still open was reported as a resolved one -
        # and "0 resolved trades" then turned into "1" the moment an order was
        # filled, before its market had answered anything.
        "live_resolved_trades": int(perf.get("resolved_trades", 0) or 0),
        "open_positions": int(perf.get("open_positions", 0) or 0),
        # Two counters a console needs and can source from the same summary, so
        # it does not have to call the storage layer a second time and risk
        # reporting a different number from the CLI.
        "total_trades": int(perf.get("total_trades", 0) or 0),
        "resolved_trades": int(perf.get("resolved_trades", 0) or 0),
        "paper": perf.get("paper") or {},
        "series_points": len(history),
    }


def _positions(storage) -> Dict[str, Any]:
    """What is actually open, live and paper counted apart."""
    if storage is None:
        return {"available": False, "source": "storage", "reason": "no storage"}
    try:
        rows = storage.get_open_positions()
    except Exception as e:
        return {"available": False, "source": "storage",
                "reason": f"{type(e).__name__}: {e}"}
    live: List[Dict[str, Any]] = []
    paper: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mode = str(item.get("execution_mode") or "").lower()
        is_paper = mode == "paper" or (
            not mode and str(item.get("status") or "").lower() == "paper")
        (paper if is_paper else live).append({
            "market_id": item.get("market_id"),
            "question": (item.get("market_question") or "")[:120],
            "venue": item.get("venue_id"),
            "side": item.get("side"),
            "size_usd": round(float(item.get("position_size_usd") or 0.0), 2),
            "entry_price": item.get("market_price"),
            "mode": "paper" if is_paper else "live",
        })
    return {
        "available": True,
        "source": "storage.get_open_positions",
        "live": live,
        "paper": paper,
        "live_count": len(live),
        "paper_count": len(paper),
    }


def _venue_evidence(storage) -> Dict[str, Any]:
    """
    Which venue is live, which venue the evidence says is best, and on what.

    The live venue comes from the one remembered selection; the ranking comes
    from the same per-venue figures the selector ranks on. Live figures are used
    when a venue has live results and paper figures otherwise, and which one it
    was is named - a venue ranked on simulation must not read as a proven one.
    """
    block: Dict[str, Any] = {"available": False, "source": "venue_selection"}
    if storage is None:
        return block
    try:
        from .strategy.venue_selection import VenueSelector
        selector = VenueSelector(storage=storage)
        stats = selector.venue_evidence()
        # The cell-level view of the same record, so "which venue" comes with
        # "at what" - a venue can be good at a five-minute sports market and
        # hopeless at a thirty-day political one, and a venue total cannot say
        # which of those the money would be going into.
        matrix = selector.evidence_matrix()
        cells = selector.best_cells()
    except Exception as e:
        block["reason"] = f"{type(e).__name__}: {e}"
        return block

    # The comparison that decides whether there is an edge at all: the agent's
    # forecasts against the price it had to beat. Absolute scores (Brier, win
    # rate) cannot answer it, so this is read from the qualification engine's own
    # verdicts rather than recomputed here - two computations of the same thing
    # is how a screen and a gate come to disagree.
    try:
        from .venues.qualification import VenueQualificationEngine
        # The engine keeps its verdicts beside the database it judged, so the
        # screen reads the same file the gate writes. Built from the path, not
        # from whatever the attribute happens to be - `db_path` has always been
        # a Path, but a string here used to silently fall back to `./data` and
        # report an empty record as "no comparison has been made".
        db_path = getattr(storage, "db_path", None)
        try:
            data_dir = str(Path(db_path).parent)
        except TypeError:
            data_dir = "./data"
        engine = VenueQualificationEngine(data_dir=data_dir)
        block["market_skill"] = {
            venue_id: {
                "verdict": qual.market_skill_verdict,
                "samples": qual.market_skill_samples,
                # The mean improvement in Brier units per trade - read from
                # the midpoint of the interval's inputs when the engine stored
                # them, and None when it stored nothing.
                "improvement": getattr(qual, "market_improvement", None),
                "ci_low": qual.market_skill_ci_low,
                "ci_high": qual.market_skill_ci_high,
                "p_value": qual.market_skill_p_value,
                # From the persisted verdict, falling back to the in-process
                # checks dict - a record read off disk carries the boolean, and
                # an empty checks dict must not read as "failed".
                "beats_price": bool(
                    qual.beats_the_price if qual.beats_the_price is not None
                    else qual.checks.get("beats_the_price")),
                "drifting": (qual.recent_market_skill_verdict
                             == "forecast_behind_price"),
                "reason": qual.market_skill_reason,
            }
            for venue_id, qual in engine.qualifications.items()
        }
    except Exception as e:
        block["market_skill_reason"] = f"{type(e).__name__}: {e}"

    block["matrix"] = {
        "source": "trade_outcomes, grouped by venue x strategy x market type x execution mode",
        "cells": len(matrix or {}),
        "cells_with_enough_evidence": len(cells or []),
        "min_cell_samples": VenueSelector.MIN_CELL_SAMPLES,
        "strongest_cells": [
            {"cell": cell["key"], "resolved": cell["resolved"],
             "net_pnl_per_trade": round(cell["net_pnl_per_trade"], 4),
             "real_evidence_coverage": round(cell["real_evidence_coverage"], 3)}
            for cell in (cells or [])[:3]
        ],
    }

    ranked: List[Dict[str, Any]] = []
    for venue_id, row in (stats or {}).items():
        live_resolved = int(row.get("live_resolved", 0) or 0)
        paper_resolved = int(row.get("paper_resolved", 0) or 0)
        if live_resolved:
            pnl, evidence = float(row.get("live_net_pnl", 0.0)), "live"
        elif paper_resolved:
            pnl, evidence = float(row.get("paper_net_pnl", 0.0)), "paper"
        else:
            pnl, evidence = 0.0, "none"
        ranked.append({
            "venue_id": venue_id,
            "ranked_on": evidence,
            "ranked_pnl_usd": round(pnl, 2),
            "live_resolved_trades": live_resolved,
            "live_net_pnl_usd": round(float(row.get("live_net_pnl", 0.0) or 0.0), 2),
            "paper_resolved_trades": paper_resolved,
            "paper_net_pnl_usd": round(float(row.get("paper_net_pnl", 0.0) or 0.0), 2),
            "open_live": int(row.get("open_live", 0) or 0),
            "open_paper": int(row.get("open_paper", 0) or 0),
        })
    ranked.sort(key=lambda r: (r["ranked_on"] != "live", -r["ranked_pnl_usd"]))
    block.update({
        "available": True,
        "live_venue": selector.remembered_live_venue(),
        "best_validated_venue": (ranked[0]["venue_id"] if ranked else None),
        "best_validated_on": (ranked[0]["ranked_on"] if ranked else "none"),
        "venues": ranked,
    })
    return block


def _strategy_evidence(storage) -> Dict[str, Any]:
    """
    Which strategy is running, and what the record says about strategies.

    Read from the recorded outcomes, split by execution mode, so a strategy that
    only ever ran in the simulation is labelled as such. Nothing here is
    inferred from the code's list of strategy names.
    """
    block: Dict[str, Any] = {"available": False, "source": "trade_outcomes"}
    if storage is None:
        return block
    try:
        rows = storage.conn.execute(
            """
            SELECT COALESCE(strategy, '') AS strategy,
                   SUM(CASE WHEN COALESCE(execution_mode,'') = 'live'
                            THEN 1 ELSE 0 END) AS live_n,
                   SUM(CASE WHEN COALESCE(execution_mode,'') = 'live'
                            THEN pnl ELSE 0 END) AS live_pnl,
                   SUM(CASE WHEN COALESCE(execution_mode,'') = 'paper'
                            THEN 1 ELSE 0 END) AS paper_n,
                   SUM(CASE WHEN COALESCE(execution_mode,'') = 'paper'
                            THEN pnl ELSE 0 END) AS paper_pnl
            FROM trade_outcomes
            WHERE actual_outcome IS NOT NULL
            GROUP BY COALESCE(strategy, '')
            """
        ).fetchall()
    except Exception as e:
        block["reason"] = f"{type(e).__name__}: {e}"
        return block

    strategies = []
    for row in rows:
        live_n = int(row["live_n"] or 0)
        paper_n = int(row["paper_n"] or 0)
        if not (live_n or paper_n):
            continue
        strategies.append({
            "strategy": row["strategy"] or "unlabelled",
            "live_resolved_trades": live_n,
            "live_net_pnl_usd": round(float(row["live_pnl"] or 0.0), 2),
            "paper_resolved_trades": paper_n,
            "paper_net_pnl_usd": round(float(row["paper_pnl"] or 0.0), 2),
            "evidence": "live" if live_n else "paper",
        })
    strategies.sort(key=lambda s: (s["evidence"] != "live",
                                   -s["live_net_pnl_usd"],
                                   -s["paper_net_pnl_usd"]))
    block.update({
        "available": bool(strategies),
        "best_validated_strategy": (strategies[0]["strategy"] if strategies else None),
        "best_validated_on": (strategies[0]["evidence"] if strategies else "none"),
        "strategies": strategies,
    })
    if not strategies:
        block["reason"] = "no resolved outcomes recorded yet"
    return block


def _risk(storage, last_cycle: Dict[str, Any]) -> Dict[str, Any]:
    """The current risk state: the kill switch and the self-preservation check."""
    block: Dict[str, Any] = {"available": False, "source": "self_preservation"}
    if storage is None:
        return block
    try:
        sp = storage.check_self_preservation()
    except Exception as e:
        block["reason"] = f"{type(e).__name__}: {e}"
        return block
    block.update({
        "available": True,
        "unprofitable_streak": int(sp.get("unprofitable_streak", 0) or 0),
        "trading_halted": bool(sp.get("should_shutdown")),
        "halt_reason": sp.get("shutdown_reason"),
        "days_active": int(sp.get("days_active", 0) or 0),
        "operator_budget_usd": round(float(sp.get("required_profit", 0.0) or 0.0), 2),
        "budget_covered": bool(sp.get("is_profitable_enough")),
        "kill_switch_level": (last_cycle.get("risk") or {}).get("kill_switch_level"),
        "can_trade": ((last_cycle.get("risk") or {}).get("can_trade")),
    })
    return block


# ----------------------------------------------------------------------
# Is it alive, and what is it doing right now
# ----------------------------------------------------------------------

# The key the loop writes while it is WORKING, not after it finishes. "What did
# it decide last cycle" and "what is it doing this second" are different
# questions, and a console that can only answer the first one shows a blank
# screen for the four minutes the agent spends scanning.
PHASE_KEY = "agent.phase"
HEARTBEAT_KEY = "agent.heartbeat"


def _interval_minutes() -> int:
    """How often the agent cycles, from the same env the runner sets."""
    import os
    try:
        return max(1, int(os.environ.get("INTERVAL_MIN") or 10))
    except (TypeError, ValueError):
        return 10


def live_window_seconds(interval_min: Optional[int] = None) -> float:
    """
    How long a heartbeat or a finished scan counts as proof the agent is alive.

    One interval plus a margin. The window used to be a hard-coded 15 minutes
    while the agent cycles every 10 and its first cycle can take longer than
    either, so a working agent spent part of every cycle reported as dead - and
    was reported as dead outright whenever the interval was turned up.
    """
    minutes = interval_min if interval_min else _interval_minutes()
    return max(900.0, float(minutes) * 60.0 + 600.0)


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


def _human_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "never"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f}h"


def agent_state(storage, interval_min: Optional[int] = None) -> Dict[str, Any]:
    """
    Is the agent alive, what is it doing, and when is it due to act again.

    THE ONE DEFINITION, because there were two and they disagreed. The
    dashboard decided liveness from scan rows (which the V3 loop never wrote),
    so a running agent read as "Not running"; the old console decided it from an
    engine object held inside the web process, so an agent running in its own
    window read as dead too. Both were reading evidence the agent does not
    produce. This reads the evidence the ACTIVE loop actually writes:

      * `agent.phase`     - written while a cycle works, so "evaluating 214
                            markets" is visible before the cycle ends;
      * `agent.heartbeat` - written at cycle start and while the kill switch
                            holds the loop, so a slow first cycle is not a death;
      * `market_scans`    - written when a cycle COMPLETES, and on cycles that
                            found nothing or were blocked.

    A reader that cannot see the agent says so; it never guesses "healthy".
    """
    block: Dict[str, Any] = {
        "available": False,
        "source": "state keys agent.phase / agent.heartbeat + market_scans",
        "interval_min": interval_min or _interval_minutes(),
        "window_seconds": round(live_window_seconds(interval_min), 1),
        "running": False,
        "state": "unknown",
        "evidence": None,
        "phase": None,
        "phase_label": None,
        "phase_detail": None,
        "phase_seconds": None,
        "last_scan": None,
        "last_scan_ago_seconds": None,
        "heartbeat_status": None,
        "heartbeat_ago_seconds": None,
        "kill_switch_level": None,
        "next_cycle_at": None,
    }
    if storage is None:
        block["reason"] = "no storage to read"
        return block
    block["available"] = True

    # 1. the last COMPLETED cycle
    last_scan = None
    try:
        row = storage.conn.execute(
            "SELECT * FROM market_scans ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        last_scan = dict(row) if row else None
    except Exception as e:
        block["scan_reason"] = f"market_scans unreadable: {type(e).__name__}: {e}"
    block["last_scan"] = last_scan
    scan_ago = _age_seconds((last_scan or {}).get("timestamp"))
    block["last_scan_ago_seconds"] = scan_ago

    # 2. the live marks - read before anything can close the connection
    heartbeat: Dict[str, Any] = {}
    try:
        raw = storage.get_state(HEARTBEAT_KEY)
        if raw:
            heartbeat = json.loads(raw)
    except Exception as e:
        block["heartbeat_reason"] = f"{type(e).__name__}: {e}"
    block["heartbeat_status"] = heartbeat.get("status")
    block["heartbeat_ago_seconds"] = _age_seconds(heartbeat.get("at"))
    block["kill_switch_level"] = heartbeat.get("kill_switch_level")

    phase: Dict[str, Any] = {}
    try:
        raw = storage.get_state(PHASE_KEY)
        if raw:
            phase = json.loads(raw)
    except Exception as e:
        block["phase_reason"] = f"{type(e).__name__}: {e}"
    if phase:
        block["phase"] = phase.get("phase")
        block["phase_label"] = phase.get("label") or phase.get("phase")
        block["phase_detail"] = phase.get("detail")
        block["phase_seconds"] = _age_seconds(phase.get("at"))
        block["next_cycle_at"] = phase.get("next_cycle_at")

    # 3. the verdict, from the two kinds of evidence
    window = block["window_seconds"]
    beat_ago = block["heartbeat_ago_seconds"]
    fresh_scan = scan_ago is not None and scan_ago < window
    fresh_beat = beat_ago is not None and beat_ago < window
    block["running"] = bool(fresh_scan or fresh_beat)
    status = str(block["heartbeat_status"] or "")
    block["blocked_by_kill_switch"] = status.startswith("kill_switch")
    if not block["running"]:
        block["state"] = "not_running"
        if scan_ago is None and beat_ago is None:
            block["evidence"] = ("no completed cycle and no heartbeat have ever "
                                 "been recorded in this database")
        else:
            freshest = min([a for a in (scan_ago, beat_ago) if a is not None])
            block["evidence"] = (
                f"the last sign of life was {_human_age(freshest)} ago, outside "
                f"the {window / 60:.0f} min window for a {block['interval_min']} "
                f"min interval")
    elif block["blocked_by_kill_switch"]:
        block["state"] = "blocked"
        block["evidence"] = (f"alive and refusing to trade: {status} "
                             f"({_human_age(beat_ago)} ago)")
    elif fresh_scan:
        block["state"] = "running"
        block["evidence"] = f"last completed cycle {_human_age(scan_ago)} ago"
    else:
        # Alive, mid-cycle, nothing finished yet. Not the same as "running and
        # idle", and the difference is exactly what the operator is watching for.
        block["state"] = "working"
        block["evidence"] = (f"alive ({status or 'heartbeat'} "
                             f"{_human_age(beat_ago)} ago), first completed "
                             f"cycle still in progress")
    block["doing"] = (block["phase_detail"] or block["phase_label"]
                      or ("waiting between cycles" if block["state"] == "running"
                          else None))
    return block


def _below_the_gate_evidence(venues: Dict[str, Any],
                            matrix: Dict[str, Any]) -> str:
    """
    Why the gate is shut, in the terms that actually decide it.

    The gates are not one bar - they are many - but the one that has never been
    measured is the one worth naming first: whether the forecasts ever beat the
    PRICE. `forecast_skill` is a rescaled Brier score and `win rate` is a count;
    neither says whether the agent's probabilities were better than the market's,
    and until the market is beaten there is no edge to deploy. When the venues
    have reported that comparison, this says what it found; when they have not,
    it says that instead.
    """
    comparisons = venues.get("market_skill") or {}
    # Only venues that actually measured something can be compared, called
    # beaten, or have an interval quoted for them. A fresh record has 19 venues
    # at 0 samples each - and "best is polymarket, improvement +0.0000 (95% CI
    # [+0.0000, +0.0000])" reads as a measured tie when nothing was measured.
    measured = {name: row for name, row in comparisons.items()
                if int(row.get("samples") or 0) > 0
                and row.get("improvement") is not None
                and row.get("ci_low") is not None}
    if measured:
        beats = [name for name, row in measured.items() if row.get("beats_price")]
        if beats:
            return (f"{', '.join(beats)} has beaten the price on the recorded "
                    f"evidence; the remaining bars (drawdown, cost coverage, "
                    f"execution) decide whether real capital follows")
        closest = max(measured.items(),
                      key=lambda kv: kv[1].get("improvement") or float("-inf"))
        row = closest[1]
        return (f"none of {len(measured)} venue(s) with paired evidence has shown "
                f"its forecasts beating the price yet; best is {closest[0]} on "
                f"{int(row.get('samples') or 0)} paired trade(s), improvement "
                f"{float(row['improvement']):+.4f} per trade "
                f"(95% CI [{float(row['ci_low']):+.4f}, "
                f"{float(row['ci_high']):+.4f}])")
    if comparisons:
        # Some rows are on file, just not enough of them to compute anything:
        # saying "never recorded a row" about a venue with 7 of them would be
        # the same class of overstatement in the other direction.
        best_row = max(((name, int(row.get("samples") or 0))
                        for name, row in comparisons.items()),
                       key=lambda kv: kv[1])
        if best_row[1] > 0:
            return (f"the comparison that decides the gate - the forecast "
                    f"against the price it had to beat - is not yet measurable: "
                    f"{best_row[0]} has {best_row[1]} paired trade(s), below the "
                    f"sample the gate requires. It needs resolved trades, not a "
                    f"score.")
        return (f"none of {len(comparisons)} venues has recorded a paired "
                f"forecast / price / outcome row yet, so the comparison that "
                f"decides the gate - the forecast against the price it had to "
                f"beat - has never been made. It needs resolved trades, not a "
                f"score.")
    return (f"{int(matrix.get('cells_with_enough_evidence') or 0)} combination(s) "
            f"have enough evidence of {int(matrix.get('cells') or 0)} seen; the "
            f"gate needs real fills and resolved outcomes, not a score")


def build_blockers(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    What stands between this agent and earning, in the order to fix it.

    Pure: it reads the snapshot and nothing else, so the console, the CLI and a
    test all get the same list from the same state. Every entry is a refusal the
    system itself would make, paired with the action that clears it - the
    console is not allowed to invent advice, and it is not allowed to imply the
    agent is earning before anything has settled.

    Ordered by how completely each one stops the mission, not by severity label:
    a dead process outranks an unfunded venue, and an unfunded venue outranks a
    venue that simply has not accumulated evidence yet.
    """
    blockers: List[Dict[str, Any]] = []
    agent = snapshot.get("agent") or {}
    capital = snapshot.get("capital") or {}
    venues = snapshot.get("venues") or {}
    risk = snapshot.get("risk") or {}
    profit = snapshot.get("profit") or {}
    matrix = venues.get("matrix") or {}

    def add(blocker_id, severity, what, evidence, clear):
        blockers.append({"id": blocker_id, "severity": severity, "what": what,
                         "evidence": evidence, "clear": clear})

    # 1. Nothing happens at all.
    if agent.get("available") and not agent.get("running"):
        add("agent_not_running", "critical",
            "The agent is not running, so nothing is being scanned or traded.",
            agent.get("evidence") or "no evidence of a live agent",
            "Start it with run_ptai.bat on the machine that runs the agent. "
            "For a single paper cycle, use Run one cycle.")

    # 2. Alive but refusing.
    if agent.get("running") and agent.get("blocked_by_kill_switch"):
        add("kill_switch", "critical",
            "The kill switch is holding the agent, so it is alive and not trading.",
            f"heartbeat {agent.get('heartbeat_status')}, level "
            f"{agent.get('kill_switch_level')}",
            "Read the risk state on this screen: the switch lifts on its own "
            "when the condition that raised it clears. It is not meant to be "
            "bypassed.")
    elif risk.get("trading_halted"):
        add("trading_halted", "critical",
            "The self-preservation check has halted trading.",
            risk.get("halt_reason") or "halted by the self-preservation check",
            "This is a stop, not a bug: review what the resolved trades did "
            "before restarting it.")

    # 3. Real money is not deployed.
    #
    # In PAPER mode that is not a blocker - it is the mode. Everything the agent
    # does is still real work: the markets, the order books, the prices, the
    # events and the settlement dates all come from the venues, and only the money
    # is simulated. Reporting "no venue holds live capital: the agent can only
    # paper-trade" as something standing in the way made an operator's deliberate
    # choice look like a missing piece, so the entry says what paper mode is, what
    # it is doing, and how to leave it when the evidence justifies it.
    mode = str(snapshot.get("mode") or "unknown").lower()
    if not venues.get("live_venue"):
        candidate = (snapshot.get("last_cycle") or {}).get("venue_to_fund")
        paper = snapshot.get("profit") or {}
        paper_block = paper.get("paper") or {}
        paper_bankroll = paper_block.get("bankroll")
        if paper_bankroll is None:
            paper_bankroll = (snapshot.get("capital") or {}).get("equity_usd")
        if mode == "paper":
            mirrored = candidate or venues.get("best_validated_venue") or "every venue scanned"
            add("paper_mode", "ok",
                "Paper mode: real markets, real order books, simulated money. "
                "Nothing is deployed, and nothing is supposed to be yet.",
                (f"simulating {mirrored}"
                 + (f" with a ${float(paper_bankroll):.2f} paper bankroll"
                    if isinstance(paper_bankroll, (int, float)) else "")
                 + "; every price, book and settlement time is read from the "
                   "venue, only the capital is imaginary"),
                "When the paper evidence convinces you, switch to live and fund the "
                "venue the agent ranks first (Money -> How capital gets in). The "
                "agent does not need it to keep working.")
        else:
            add("no_live_venue", "next_step",
                "No venue holds live capital, so no real money can be deployed.",
                (f"live venue: none" +
                 (f"; the agent ranked {candidate} first for funding"
                  if candidate else "; no venue is ranked for funding yet")),
                "Fund the venue the agent ranks first (Money -> How capital gets in) "
                "and authorise a budget for it.")

    # 4. No validated venue, so live capital would be a guess.
    if not venues.get("best_validated_venue"):
        add("no_validated_venue", "gate",
            "No venue has passed the qualification gate, so the agent will not "
            "put real money on any of them.",
            _below_the_gate_evidence(venues, matrix),
            "Keep it running. Every paper cycle records an outcome, and the "
            "evidence is what eventually opens the gate.")

    # 5. Nothing has settled, so there is no result to report yet.
    paper = profit.get("paper") or {}
    settled = int(profit.get("live_resolved_trades") or 0)
    if not settled and not int(paper.get("settled_trades") or 0):
        add("nothing_settled", "waiting",
            "No position has settled yet, so there is no result either way.",
            "resolved trades: 0 live, 0 paper",
            "Nothing to do. A market that has not resolved has not answered.")

    # 6. There are results, and they are negative. Say so.
    net = profit.get("net_pnl_usd")
    if settled and isinstance(net, (int, float)) and net < 0:
        add("losing", "loss",
            "The resolved live trades are net negative so far.",
            f"realised P&L ${net:+.2f} over {settled} resolved live trade(s)",
            "Do not add capital to a losing account. Leave it running and read "
            "the calibration: the numbers it must beat are on this screen.")

    if not blockers:
        add("none", "ok",
            "Nothing is blocking the agent.",
            "a live agent, a funded venue and a passing gate",
            "Nothing to do. It runs on its own.")
    return blockers


def _headline(snapshot: Dict[str, Any]) -> str:
    """
    The whole state of the agent in one honest sentence.

    Written as a sentence rather than a colour because the operator runs this
    unattended: "Running" on its own has already been read as "earning", and it
    does not mean that. This says what is true, in the same words in the CLI and
    on the console.
    """
    agent = snapshot.get("agent") or {}
    capital = snapshot.get("capital") or {}
    profit = snapshot.get("profit") or {}
    state = agent.get("state")
    if state == "unknown":
        return "The agent state could not be read."
    if state == "not_running":
        return f"Stopped - {agent.get('evidence') or 'nothing is running'}."
    deployed = capital.get("deployment_pct")
    settled = int(profit.get("live_resolved_trades") or 0)
    paper = profit.get("paper") or {}
    paper_settled = int(paper.get("settled_trades") or 0)
    if state == "blocked":
        return f"Running but refusing to trade - {agent.get('evidence')}."
    work = agent.get("doing")
    where = f", {work}" if work else ""
    if settled:
        return (f"Running{where} - ${float(profit.get('net_pnl_usd') or 0.0):+.2f} "
                f"realised on {settled} resolved live trade(s).")
    if capital.get("account") == "paper" and paper_settled:
        return (f"Running{where} - paper account "
                f"${float(paper.get('net_pnl') or 0.0):+.2f} on {paper_settled} "
                f"settled simulated trade(s); no real money is at stake yet.")
    deployed_note = (f", {float(deployed):.0f}% deployed"
                     if isinstance(deployed, (int, float)) else "")
    return (f"Running{where}{deployed_note} - nothing has settled yet, so there "
            f"is no money result to report.")


def operator_snapshot(storage=None) -> Dict[str, Any]:
    """
    Everything the operator asked to be able to see, in one payload.

    `storage` is injectable so the caller can pass the same database the loop is
    using; omitted, it opens the default one, which is what a console process
    wants. Each block names its own source, so a number on the screen can always
    be traced to the thing that produced it.
    """
    storage = _storage_or_none(storage)
    last_cycle = _connectivity(storage)
    capital = _capital(storage)
    profit = _profit(storage)
    positions = _positions(storage)
    venues = _venue_evidence(storage)
    strategies = _strategy_evidence(storage)
    risk = _risk(storage, last_cycle)
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": _operator_mode(storage),
        "capital": capital,
        "profit": profit,
        "positions": positions,
        "venues": venues,
        "strategies": strategies,
        "risk": risk,
        "last_cycle": last_cycle,
        # The two blocks that answer "is it working, and what should I do":
        # both derived from the blocks above, so they cannot disagree with them.
        "agent": agent_state(storage),
    }
    snapshot["blockers"] = build_blockers(snapshot)
    snapshot["next_action"] = snapshot["blockers"][0]["clear"]
    snapshot["headline"] = _headline(snapshot)
    return snapshot


def _operator_mode(storage) -> str:
    if storage is None:
        return "unknown"
    try:
        from .execution.capital import operator_mode
        return operator_mode(storage)
    except Exception as e:
        logger.debug(f"Operator view could not read the operator mode: {e}")
        return "unknown"


def describe_snapshot(snapshot: Dict[str, Any]) -> List[str]:
    """
    The snapshot as lines a console can print, in the order the questions were
    asked. Kept here rather than in the CLI so the dashboard can print the same
    sentences and the two cannot describe the same state differently.
    """
    lines: List[str] = []
    venues = snapshot.get("venues") or {}
    live = venues.get("live_venue")
    if live:
        reasons = (snapshot.get("last_cycle") or {}).get("venue_reasons") or []
        lines.append(f"Live venue: {live}" + (f" - {reasons[0]}" if reasons else ""))
    else:
        lines.append(
            f"Live venue: none - nothing is trading real money. Venue to fund "
            f"first: {venues.get('best_validated_venue') or 'undetermined'}")
    strategies = snapshot.get("strategies") or {}
    if strategies.get("best_validated_strategy"):
        lines.append(
            f"Strategy: {strategies['best_validated_strategy']} "
            f"(led on {strategies.get('best_validated_on')} results)")
    else:
        lines.append(f"Strategy: running the scan, no validated strategy yet "
                     f"({strategies.get('reason', 'no outcomes')})")
    capital = snapshot.get("capital") or {}
    if capital.get("available"):
        lines.append(
            f"Deployment: ${capital.get('deployed_usd', 0.0):.2f} of "
            f"${capital.get('equity_usd', 0.0):.2f} equity "
            f"({capital.get('deployment_pct', 0.0):.1f}%) - free "
            f"${capital.get('free_cash_usd', 0.0):.2f}, reserved "
            f"${capital.get('reserved_capital_usd', 0.0):.2f}")
    profit = snapshot.get("profit") or {}
    if profit.get("available"):
        lines.append(
            f"Profit: ${profit.get('net_pnl_usd', 0.0):+.2f} "
            f"({profit.get('net_pnl_pct', 0.0):+.1f}%) on "
            f"${profit.get('initial_usd', 0.0):.2f} - "
            f"{profit.get('live_resolved_trades', 0)} resolved live trades; "
            f"paper kept separate")
    risk = snapshot.get("risk") or {}
    if risk.get("available"):
        lines.append(
            f"Risk: {'HALTED - ' + str(risk.get('halt_reason')) if risk.get('trading_halted') else 'trading'}"
            f", kill switch level {risk.get('kill_switch_level')}"
            f", {risk.get('unprofitable_streak', 0)} consecutive unprofitable days")
    matrix = venues.get("matrix") or {}
    if matrix.get("cells"):
        strongest = matrix.get("strongest_cells") or []
        if strongest:
            top = strongest[0]
            lines.append(
                f"Evidence: {matrix.get('cells_with_enough_evidence', 0)} of "
                f"{matrix['cells']} venue x strategy x market type x execution "
                f"mode cells have {matrix.get('min_cell_samples', 5)}+ resolved "
                f"trades; the strongest is {top['cell']} "
                f"(${top['net_pnl_per_trade']:+.3f} a trade over "
                f"{top['resolved']} trades, "
                f"{top['real_evidence_coverage'] * 100:.0f}% priced against a "
                f"real book)")
        else:
            lines.append(
                f"Evidence: {matrix['cells']} venue x strategy x market type x "
                f"execution mode cell(s), none yet with "
                f"{matrix.get('min_cell_samples', 5)}+ resolved trades - no slice "
                f"of the record is thick enough to allocate on")
    cycle = snapshot.get("last_cycle") or {}
    if cycle.get("available"):
        orders = cycle.get("orders") or {}
        lines.append(
            f"Last cycle {str(cycle.get('at'))[:19]}: {cycle.get('verdict')} - "
            f"{cycle.get('markets_scanned')} markets across "
            f"{cycle.get('venues_searched')} venues, "
            f"{orders.get('positions_recorded', 0)} position(s) recorded"
            + (f", {orders.get('blocked_live_capital')} blocked by the live "
               f"capital boundary ({orders.get('blocked_reason')})"
               if orders.get("blocked_live_capital") else ""))
        decided = cycle.get("decided") or {}
        efficiency = decided.get("capital_efficiency")
        if isinstance(efficiency, dict):
            ratio = efficiency.get("ev_per_capital_time_risk")
            if ratio is not None:
                assumed = efficiency.get("stressed_net_ev_usd") != \
                    efficiency.get("net_ev_usd")
                lines.append(
                    f"  capital efficiency: ${float(ratio) * 100:.4f} of net EV "
                    f"per $100 per day locked up per unit of execution risk "
                    f"({efficiency.get('capital_days_usd', 0.0):.1f} capital-days"
                    + (", ranked on the STRESSED EV because a cost was assumed"
                       if assumed else ", on measured costs")
                    + ") - this is the number it ranks on")
        if cycle.get("why"):
            lines.append(f"  why: {cycle['why']}")
    else:
        lines.append(f"Last cycle: {cycle.get('note')}")

    # Appended, never prepended: the first line of this list is the live-venue
    # answer the CLI panel leads with, and the dashboard's operator payload is
    # asserted against it.
    agent = snapshot.get("agent") or {}
    if agent.get("available"):
        work = agent.get("doing")
        lines.append(
            f"Agent: {str(agent.get('state') or 'unknown').replace('_', ' ')}"
            + (f" - {agent['evidence']}" if agent.get("evidence") else "")
            + (f" - now: {work}" if work else "")
            + (f" - next cycle {str(agent.get('next_cycle_at'))[:19]}"
               if agent.get("next_cycle_at") else ""))
    if snapshot.get("headline"):
        lines.append(f"Verdict: {snapshot['headline']}")
    for blocker in (snapshot.get("blockers") or [])[:2]:
        if blocker.get("id") == "none":
            continue
        lines.append(f"Next: {blocker['clear']}")
    return lines
