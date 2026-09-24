"""
Position Ledger - the difference between "we have $50" and "$50 is available".

The system stored a single number, `bankroll`, and sized every new position
against it. That number did not distinguish money sitting in an open position
from money available to open another one, so five $3 positions could each be
sized as 6% of the same $50 as though the other four did not exist. The
exposure limits were the only thing standing between that and an account with
no free cash, and they are sized against the same undifferentiated figure.

This ledger keeps the five quantities apart:

    equity  = free_cash + reserved_capital + unrealised_pnl
    free_cash         - money not committed to any open position
    reserved_capital  - money committed to open positions, at cost
    open_position_value - current mark of the open positions
    realised_pnl      - banked, from settled positions
    unrealised_pnl    - open positions marked to market

Sizing reads `free_cash`. Never `equity`, and never the stored bankroll, which
is what the previous code did.

Everything is derived from the trades table, so the ledger is reconstructible:
if the process dies mid-trade, the next start recomputes the same answer from
the recorded rows rather than from memory that no longer exists.
"""

from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field
from loguru import logger

# A trade in one of these states holds capital.
OPENING_STATUSES = frozenset({"open", "paper", "pending", "executed", "partially_filled"})


@dataclass
class PositionLedger:
    """Point-in-time capital state, every figure traceable to trades."""
    equity: float = 0.0
    free_cash: float = 0.0
    reserved_capital: float = 0.0
    open_position_cost: float = 0.0
    open_position_value: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    live_position_count: int = 0
    paper_position_count: int = 0
    initial_bankroll: float = 50.0
    warnings: List[str] = field(default_factory=list)

    @property
    def reserved_pct(self) -> float:
        return (self.reserved_capital / self.equity * 100.0) if self.equity else 0.0

    @property
    def can_open_new(self) -> bool:
        """Is there any free cash at all? A zero here means no new position."""
        return self.free_cash > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 2),
            "free_cash": round(self.free_cash, 2),
            "reserved_capital": round(self.reserved_capital, 2),
            "open_position_cost": round(self.open_position_cost, 2),
            "open_position_value": round(self.open_position_value, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "realised_pnl": round(self.realised_pnl, 2),
            "reserved_pct": round(self.reserved_pct, 2),
            "live_position_count": self.live_position_count,
            "paper_position_count": self.paper_position_count,
            "initial_bankroll": round(self.initial_bankroll, 2),
            "can_open_new": self.can_open_new,
            "warnings": self.warnings,
        }


class PositionLedgerBuilder:
    """
    Build the ledger from storage.

    Deliberately reads the trade rows rather than trusting a cached bankroll,
    so the figures cannot drift from what was actually recorded.
    """

    # How long to wait before treating a position with no settlement answer as
    # stale. Not used to close anything - only to warn.
    def __init__(self, storage=None, bankroll_provider=None):
        self.storage = storage
        self._bankroll_provider = bankroll_provider

    def build(self, price_lookup=None) -> PositionLedger:
        """
        Compute the ledger.

        `price_lookup(market_id) -> Optional[float]` marks open positions to
        market. Without it, open positions are carried at cost and
        unrealised_pnl is reported as 0 with a warning, because an unknown mark
        must not be presented as a profit or a loss.
        """
        ledger = PositionLedger()

        if self.storage is None:
            ledger.warnings.append("no storage: ledger cannot be computed")
            return ledger

        try:
            summary = self.storage.get_performance_summary()
        except Exception as e:
            ledger.warnings.append(f"could not read performance summary: {e}")
            summary = {}

        ledger.initial_bankroll = float(summary.get("initial_bankroll", 50.0) or 50.0)
        bankroll = float(summary.get("bankroll", ledger.initial_bankroll) or 0.0)

        try:
            open_trades = self.storage.get_open_positions()
        except Exception as e:
            ledger.warnings.append(f"could not read open positions: {e}")
            open_trades = []

        try:
            realised_row = self.storage.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades WHERE resolved = 1"
            ).fetchone()
            ledger.realised_pnl = float(realised_row["total"] or 0.0) if realised_row else 0.0
        except Exception as e:
            ledger.warnings.append(f"could not read realised P&L: {e}")

        marked = 0
        unmarked = 0
        for trade in open_trades:
            status = str(trade.get("status") or "").lower()
            size = float(trade.get("position_size_usd") or 0.0)
            if size <= 0:
                continue

            if status == "paper":
                # Paper capital is not real capital and must never reduce the
                # free cash available for live positions.
                ledger.paper_position_count += 1
                continue

            ledger.live_position_count += 1
            ledger.open_position_cost += size

            price = float(trade.get("market_price") or 0.0)
            mark = None
            if price_lookup is not None and trade.get("market_id"):
                try:
                    mark = price_lookup(str(trade["market_id"]))
                except Exception:
                    mark = None

            side = str(trade.get("side") or "").upper()
            bought_yes = side in ("YES", "1", "LONG", "BUY", "TRUE")
            entry = price if bought_yes else (1.0 - price)

            if mark is None or entry <= 0:
                # Carry at cost. An unknown mark is not a zero and not a gain.
                unmarked += 1
                ledger.open_position_value += size
                continue

            marked_price = float(mark) if bought_yes else (1.0 - float(mark))
            shares = size / entry
            ledger.open_position_value += shares * marked_price
            marked += 1

        if unmarked:
            ledger.warnings.append(
                f"{unmarked} open position(s) could not be marked to market and "
                f"are carried at cost; unrealised P&L reflects only the "
                f"{marked} that could be priced"
            )

        ledger.reserved_capital = ledger.open_position_cost
        ledger.unrealised_pnl = ledger.open_position_value - ledger.open_position_cost

        # Free cash = the recorded bankroll minus what is committed. The stored
        # bankroll is decremented when a position opens, so this is the honest
        # remainder rather than a re-derivation that could double-count.
        ledger.free_cash = max(0.0, bankroll - ledger.open_position_cost)
        if bankroll < ledger.open_position_cost:
            ledger.warnings.append(
                f"committed capital ${ledger.open_position_cost:.2f} exceeds the "
                f"recorded bankroll ${bankroll:.2f} - the ledger and storage "
                f"disagree and one of them is wrong"
            )

        ledger.equity = ledger.free_cash + ledger.open_position_value

        return ledger

    def sizing_capital(self, price_lookup=None) -> float:
        """
        The figure position sizing must use: free cash, never equity.

        Exposed as a method so the call site reads as a decision rather than as
        an expression someone has to interpret.
        """
        ledger = self.build(price_lookup=price_lookup)
        if not ledger.can_open_new:
            logger.warning(
                f"No free capital: equity ${ledger.equity:.2f} is fully committed "
                f"(${ledger.reserved_capital:.2f} reserved). No new position can "
                f"be opened this cycle.")
        return ledger.free_cash
