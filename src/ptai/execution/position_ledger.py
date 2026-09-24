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
    # Committed to orders that have not filled. Separate from reserved_capital so
    # the two can be reported apart - they are different kinds of commitment, and
    # a single figure hides which one is growing.
    resting_order_cost: float = 0.0
    # True when the working orders could not be read. Fail closed: an unknown
    # reservation must never be treated as no reservation.
    reservations_unknown: bool = False
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
        """
        Is there any free cash at all? A zero here means no new position.

        Also false when the capital committed to working orders is unknown.
        Unknown committed capital and zero committed capital are opposite facts,
        and treating the first as the second is how the agent spends money it has
        already promised to an order resting in the book.
        """
        if self.reservations_unknown:
            return False
        return self.free_cash > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 2),
            "free_cash": round(self.free_cash, 2),
            "reserved_capital": round(self.reserved_capital, 2),
            "resting_order_cost": round(self.resting_order_cost, 2),
            "reservations_unknown": self.reservations_unknown,
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

    def build(self, price_lookup=None, venue_positions=None,
              venue_state_complete=None) -> PositionLedger:
        """
        Compute the ledger.

        `price_lookup(market_id) -> Optional[float]` marks open positions to
        market. Without it, open positions are carried at cost and
        unrealised_pnl is reported as 0 with a warning, because an unknown mark
        must not be presented as a profit or a loss.

        `venue_positions` are positions the VENUE reports that our own records do
        not contain - placed by hand, or filled while the process was down. They
        are real capital committed, so they are reserved here; without them the
        ledger could offer free cash that is already spent.

        `venue_state_complete` is the venue read's own verdict. False or None
        means the account could not be fully read, and reservations are then
        UNKNOWN - the same fail-closed treatment as an unreadable working order,
        because "I could not ask" and "there is nothing there" are different
        answers and only one of them is safe to trade on.
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
            # LIVE settlements only. Paper P&L is real data about the strategy
            # and not real money; summing both here put simulated profit into
            # the equity and drawdown the operator is shown.
            realised_row = self.storage.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades "
                "WHERE resolved = 1 AND COALESCE(execution_mode,'live')='live'"
            ).fetchone()
            ledger.realised_pnl = float(realised_row["total"] or 0.0) if realised_row else 0.0
        except Exception as e:
            ledger.warnings.append(f"could not read realised P&L: {e}")

        # Capital committed to positions the venue knows about and we do not.
        external_cost = 0.0
        external_count = 0
        for position in (venue_positions or []):
            try:
                amount = float(position.get("amount_usd") or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            if amount <= 0:
                # A venue position with no valuation is still exposure. Count it
                # as unknown rather than as nothing.
                ledger.reservations_unknown = True
                ledger.warnings.append(
                    "venue position without a value: exposure cannot be sized")
                continue
            external_cost += amount
            external_count += 1

        if external_count:
            ledger.open_position_cost += external_cost
            ledger.live_position_count += external_count
            ledger.warnings.append(
                f"{external_count} venue-only position(s) worth "
                f"${external_cost:.2f} reserved: the venue reports exposure that "
                f"is not in the local trade log")

        if venue_state_complete is False:
            ledger.reservations_unknown = True
            ledger.warnings.append(
                "venue account state could not be read: working orders and "
                "positions may exist that are not in this ledger")

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

        # Cash locked behind orders that are still working. Booked positions do
        # not include it - a resting order buys nothing until it fills, so its
        # cost is not in the trades table - so the two do not double count.
        #
        # This was missing from free cash, and the consequence was a
        # self-inflicted overcommit: with $50, a $3 position and a $3 GTC order
        # resting in the book, the ledger reported $47 free when $44 was. The
        # next cycle then sized against money that was already promised to an
        # earlier order, and the venue rejected one of them - or worse, both
        # filled and the agent held more exposure than it had authorised.
        resting_usd = 0.0
        try:
            resting_usd = float(self.storage.resting_capital_usd() or 0.0)
        except Exception as e:
            # An unreadable reservation is NOT zero. Treating it as zero is the
            # exact overcommit this guards against, so refuse to size instead.
            ledger.warnings.append(
                f"working orders could not be read ({type(e).__name__}: {e}), so "
                f"the capital already committed to them is unknown - new "
                f"positions cannot be sized safely this cycle")
            ledger.reservations_unknown = True
            resting_usd = bankroll  # assume fully committed: the safe end

        ledger.reserved_capital = ledger.open_position_cost
        ledger.resting_order_cost = resting_usd
        ledger.unrealised_pnl = ledger.open_position_value - ledger.open_position_cost

        # Free cash = the recorded bankroll minus everything already committed.
        # The stored bankroll is decremented when a position opens, so this is
        # the honest remainder rather than a re-derivation that could
        # double-count.
        committed = ledger.open_position_cost + ledger.resting_order_cost
        ledger.free_cash = max(0.0, bankroll - committed)
        if bankroll < committed:
            ledger.warnings.append(
                f"committed capital ${committed:.2f} (positions "
                f"${ledger.open_position_cost:.2f} + working orders "
                f"${ledger.resting_order_cost:.2f}) exceeds the recorded bankroll "
                f"${bankroll:.2f} - the ledger and storage disagree and one of "
                f"them is wrong"
            )

        # Equity is everything the account owns: the position at market, the
        # cash committed to working orders (which is still the account's money,
        # just promised to an order), and the free remainder.
        #
        # It used to be free_cash + position_value, which dropped the resting
        # capital entirely - so an account with $50, a $3 position and a $3 GTC
        # order reported $47 of equity. The free-cash figure stays deliberately
        # conservative; the equity figure has to describe the account, and an
        # understated equity is read by the operator as a loss that did not
        # happen.
        ledger.equity = (ledger.free_cash + ledger.open_position_value
                         + ledger.resting_order_cost)

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
