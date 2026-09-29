"""
A round: the unit of work the operator actually asked for.

"For a round to be complete it's supposed to do research on the market, find the
suitable predictions, research more about them on which ones to bet on, and then
bet on them using the fake currency in the paper mode and produce results. A
complete run should take some time and when it's done it's supposed to produce
results in form of either a bankroll with a negative or a positive."

So a round has a beginning, an end, and a NUMBER - the account value on both
sides of it. Without that, "Run one cycle" produced activity and no answer: the
paper bankroll sat at $50.00 through every round because open positions were
carried at cost and nothing settles for days.

What a round result is, exactly
------------------------------
    equity_end - equity_start = realised P&L + the change in what the open
                                positions are worth at the prices now on screen

Both halves are reported separately and never merged:

  * REALISED is money that came back when a market resolved. It is in the
    bankroll and it is never taken away.
  * MARKED is the book at current prices. It moves back and forth and it is NOT
    a settlement - a round that ends +$1.20 marked has not earned $1.20 yet.

A round where nothing could be priced says so and reports no number, rather
than reporting a flat $50.00 that reads like a break-even result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

ROUNDS_KEY = "agent.rounds"
MARKS_KEY = "agent.position_marks"
MAX_ROUNDS_KEPT = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def money(value: float) -> str:
    """
    A signed dollar figure, with the sign where it is read: -$1.20, not $-1.20.

    The operator reads these lines to answer one question - did the round make
    money - and `$-1.20` reads as a formatting fault rather than as a loss.
    """
    return f"{'-' if float(value) < 0 else '+'}${abs(float(value)):.2f}"


@dataclass
class RoundReport:
    """One completed round, with the bankroll at each end of it."""

    number: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    mode: str = "paper"
    # The account the round is scored on. Paper runs score the paper account;
    # a live round scores the real one. Never a mix of the two.
    account: str = "paper"
    equity_start: Optional[float] = None
    equity_end: Optional[float] = None
    cash_start: Optional[float] = None
    cash_end: Optional[float] = None
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    # What the round did, in the operator's terms.
    markets_discovered: int = 0
    markets_screened: int = 0
    markets_researched: int = 0
    markets_priced: int = 0
    candidates: int = 0
    positions_opened: int = 0
    positions_settled: int = 0
    staked_usd: float = 0.0
    research_sources: int = 0
    positions_held: int = 0
    open_positions_marked: int = 0
    open_positions_unmarked: int = 0
    # WHAT THE ROUND TRADED, and how close it came when it traded nothing. A
    # round that reports only a bankroll number cannot answer "what did it do"
    # or "how far off was it" - the two questions the operator asked.
    trades: List[Dict[str, Any]] = field(default_factory=list)
    closest_call: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def net_usd(self) -> Optional[float]:
        """The round's result: what the account is worth now minus before."""
        if self.equity_start is None or self.equity_end is None:
            return None
        return round(self.equity_end - self.equity_start, 4)

    @property
    def verdict(self) -> str:
        """up / down / flat / unknown - the one word the operator asked for."""
        net = self.net_usd
        if net is None:
            return "unknown"
        if net > 0.005:
            return "up"
        if net < -0.005:
            return "down"
        return "flat"

    def to_dict(self) -> Dict[str, Any]:
        net = self.net_usd
        return {
            "number": self.number,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 1),
            "mode": self.mode,
            "account": self.account,
            "equity_start": (None if self.equity_start is None
                             else round(self.equity_start, 2)),
            "equity_end": (None if self.equity_end is None
                           else round(self.equity_end, 2)),
            "cash_start": (None if self.cash_start is None
                           else round(self.cash_start, 2)),
            "cash_end": (None if self.cash_end is None
                         else round(self.cash_end, 2)),
            "net_usd": net,
            "verdict": self.verdict,
            "realised_pnl": round(self.realised_pnl, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "markets_discovered": self.markets_discovered,
            "markets_screened": self.markets_screened,
            "markets_researched": self.markets_researched,
            "markets_priced": self.markets_priced,
            "candidates": self.candidates,
            "positions_opened": self.positions_opened,
            "positions_settled": self.positions_settled,
            "positions_held": self.positions_held,
            "staked_usd": round(self.staked_usd, 2),
            "research_sources": self.research_sources,
            "open_positions_marked": self.open_positions_marked,
            "open_positions_unmarked": self.open_positions_unmarked,
            "trades": list(self.trades),
            "closest_call": dict(self.closest_call),
            "notes": self.notes,
            "warnings": self.warnings,
        }

    def headline(self) -> str:
        """One sentence for the log and the console."""
        if self.net_usd is None:
            return (f"Round {self.number}: no result - nothing could be priced "
                    f"this round, so the account value is unknown rather than "
                    f"unchanged")
        net = money(self.net_usd)
        if self.verdict == "flat":
            net = "$0.00"
        parts = [f"Round {self.number} ({self.duration_seconds:.0f}s): "
                 f"{self.account} account ${self.equity_start:,.2f} -> "
                 f"${self.equity_end:,.2f} = {net}"]
        parts.append(f"{self.positions_opened} opened, "
                     f"{self.positions_settled} settled, "
                     f"{self.positions_held} held")
        if self.trades:
            parts.append(f"{len(self.trades)} trade(s) this round")
        if abs(self.unrealised_pnl) >= 0.005:
            parts.append(f"of which {money(self.unrealised_pnl)} is the book at "
                         f"current prices (marked, not settled)")
        if self.realised_pnl:
            parts.append(f"{money(self.realised_pnl)} realised")
        return " | ".join(parts)


def blank_report(mode: str = "paper") -> RoundReport:
    return RoundReport(mode=mode, started_at=_now())


def load_rounds(storage: Any, limit: int = 20) -> List[Dict[str, Any]]:
    """Completed rounds, newest first. An unreadable store is an empty list."""
    if storage is None:
        return []
    try:
        raw = storage.get_state(ROUNDS_KEY)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not read completed rounds: {type(e).__name__}: {e}")
        return []
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.error(f"Completed-rounds record is unreadable ({e}); reporting "
                     f"none rather than inventing a history")
        return []
    if not isinstance(rows, list):
        return []
    return list(reversed(rows))[:limit]


def record_round(storage: Any, report: RoundReport) -> bool:
    """
    Append a completed round to the record.

    Kept as a history rather than a single latest row, because the operator's
    question is "is this thing making or losing money", and one round cannot
    answer it. Capped so the KV row cannot grow without limit.
    """
    if storage is None:
        return False
    try:
        raw = storage.get_state(ROUNDS_KEY)
        rows = json.loads(raw) if raw else []
        if not isinstance(rows, list):
            rows = []
    except Exception:  # noqa: BLE001
        rows = []
    rows.append(report.to_dict())
    rows = rows[-MAX_ROUNDS_KEPT:]
    try:
        storage.set_state(ROUNDS_KEY, json.dumps(rows))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not store the completed round: {type(e).__name__}: {e}")
        return False


def round_history_summary(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The running score across completed rounds: is the account up or down."""
    scored = [r for r in (rounds or []) if r.get("net_usd") is not None]
    if not scored:
        return {"rounds": len(rounds or []), "scored": 0, "net_usd": None,
                "up": 0, "down": 0, "flat": 0,
                "note": ("no completed round has produced a result yet - a round "
                         "with nothing priced reports no number rather than "
                         "a flat one")}
    net = sum(float(r["net_usd"]) for r in scored)
    return {
        "rounds": len(rounds or []),
        "scored": len(scored),
        "net_usd": round(net, 2),
        "up": sum(1 for r in scored if r.get("verdict") == "up"),
        "down": sum(1 for r in scored if r.get("verdict") == "down"),
        "flat": sum(1 for r in scored if r.get("verdict") == "flat"),
        "best_round": max(scored, key=lambda r: float(r["net_usd"]))["net_usd"],
        "worst_round": min(scored, key=lambda r: float(r["net_usd"]))["net_usd"],
        "last_round": scored[0],
    }
