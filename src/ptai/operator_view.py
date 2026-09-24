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

    equity = float(getattr(ledger, "equity", 0.0) or 0.0)
    committed = (float(getattr(ledger, "open_position_value", 0.0) or 0.0)
                 + float(getattr(ledger, "resting_order_cost", 0.0) or 0.0))
    return {
        "available": True,
        "source": "position_ledger",
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
        "live_resolved_trades": int(perf.get("total_trades", 0) or 0),
        "open_positions": int(perf.get("open_positions", 0) or 0),
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
    except Exception as e:
        block["reason"] = f"{type(e).__name__}: {e}"
        return block

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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": _operator_mode(storage),
        "capital": capital,
        "profit": profit,
        "positions": positions,
        "venues": venues,
        "strategies": strategies,
        "risk": risk,
        "last_cycle": last_cycle,
    }


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
        if cycle.get("why"):
            lines.append(f"  why: {cycle['why']}")
    else:
        lines.append(f"Last cycle: {cycle.get('note')}")
    return lines
