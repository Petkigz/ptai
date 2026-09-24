"""
Paper broker - simulate what an order would have done, against the real book.

Paper mode is only worth having if it can lose. The version this replaces could
not: in paper mode the adapter returned "dry_run" with no size, and the loop then
recorded a position at the REQUESTED size and the REQUESTED price. Every simulated
trade filled completely, instantly, at the price the agent wanted, with no fees
and no depth limit. A system like that reports a beautiful equity curve and has
learned nothing, because it never models the three things that actually decide
whether a strategy works:

  * the book has finite size at each price, so a large order walks up the ladder;
  * a limit away from the market does not fill, it RESTS - and may never fill;
  * the venue charges a fee on the notional.

So this module prices an order the way the venue would:

    a BUY consumes asks from the best price upward,
    stopping when the money runs out, the depth runs out, or the limit is hit.

Everything it returns is labelled. A fill simulated against a real orderbook is
`is_real=True`; one simulated because no book could be read is `is_real=False`
and carries a warning, because a simulated fill from a guessed book is a
guess wearing a number.

Two things this model is deliberately conservative about:

  * A RESTING order only fills when the market actually crosses its limit in a
    later snapshot. It is never filled on submission, and there is no queue
    model, so a resting fill is an OPTIMISTIC upper bound - flagged as such.
  * Nothing here can manufacture liquidity. If the book has $4 at the touch and
    the order is $20, the fill is $4, and the rest is reported unfilled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..markets.mechanics import MarketMechanics

# Venue taker fee on the notional. Polymarket runs a maker/taker schedule that
# arrives per market in the venue's own payload, so this is only the fallback.
DEFAULT_TAKER_FEE_RATE = 0.0
# Gas is a real per-transaction cost on Polygon, but order placement at the CLOB
# is signed off-chain and gasless; only deposit/withdraw touch the chain.
DEFAULT_GAS_USD = 0.0


@dataclass
class BookLevel:
    price: float
    size: float  # shares

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class PaperFill:
    """
    What an order would have done. Zeros mean nothing happened.

    `is_real` refers to the INPUTS (was the book real?), never to the fill: no
    fill is real, because no order was sent. `is_marketable` says whether the
    order would have traded at all, and `would_rest` says it would have sat in
    the book instead.
    """

    side: str
    requested_usd: float
    limit_price: Optional[float] = None
    filled_shares: float = 0.0
    filled_usd: float = 0.0
    avg_price: float = 0.0
    best_price: float = 0.0       # the touch the order started from
    worst_price: float = 0.0      # the deepest level it reached
    levels_consumed: int = 0
    unfilled_usd: float = 0.0
    unfilled_shares: float = 0.0
    is_marketable: bool = False
    would_rest: bool = False
    fee_usd: float = 0.0
    fee_rate: float = 0.0
    slippage_usd: float = 0.0
    slippage_bps: float = 0.0
    depth_available_usd: float = 0.0
    book_source: str = "unknown"
    is_real: bool = False
    optimistic: bool = False
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def is_fill(self) -> bool:
        return self.filled_shares > 0.0

    @property
    def is_partial(self) -> bool:
        return self.is_fill and self.unfilled_usd > 1e-9

    @property
    def total_cost_usd(self) -> float:
        return self.filled_usd + self.fee_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "requested_usd": round(self.requested_usd, 6),
            "limit_price": self.limit_price,
            "filled_shares": round(self.filled_shares, 4),
            "filled_usd": round(self.filled_usd, 6),
            "avg_price": round(self.avg_price, 6),
            "best_price": round(self.best_price, 6),
            "worst_price": round(self.worst_price, 6),
            "levels_consumed": self.levels_consumed,
            "unfilled_usd": round(self.unfilled_usd, 6),
            "unfilled_shares": round(self.unfilled_shares, 4),
            "is_marketable": self.is_marketable,
            "would_rest": self.would_rest,
            "fee_usd": round(self.fee_usd, 6),
            "fee_rate": self.fee_rate,
            "slippage_usd": round(self.slippage_usd, 6),
            "slippage_bps": round(self.slippage_bps, 1),
            "depth_available_usd": round(self.depth_available_usd, 6),
            "book_source": self.book_source,
            "is_real": self.is_real,
            "optimistic": self.optimistic,
            "filled": self.is_fill,
            "partial": self.is_partial,
            "reason": self.reason,
            "warnings": self.warnings,
        }


def _num(value: Any) -> Optional[float]:
    """A number from a venue payload field. The CLOB sends strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


class PaperBroker:
    """
    Fill simulator. One instance per market's mechanics, or one shared and given
    mechanics per call.
    """

    def __init__(self, mechanics: Optional[MarketMechanics] = None,
                 taker_fee_rate: Optional[float] = None,
                 gas_usd: float = DEFAULT_GAS_USD):
        self.mechanics = mechanics
        self.taker_fee_rate = (DEFAULT_TAKER_FEE_RATE if taker_fee_rate is None
                               else float(taker_fee_rate))
        self.gas_usd = float(gas_usd)

    # ------------------------------------------------------------------
    # reading a book
    # ------------------------------------------------------------------

    @staticmethod
    def levels_from_raw(raw: Any, side: str) -> List[BookLevel]:
        """
        A sorted ladder from a venue book payload.

        The CLOB returns {"bids": [...], "asks": [...]} with string prices and
        sizes, and the two sides arrive in different orders. Nothing is assumed
        about that: both sides are parsed, filtered and sorted here, so an
        ascending-ish bid list cannot be read backwards.
        """
        if not isinstance(raw, dict):
            return []
        key = "asks" if side.upper() in ("BUY", "YES") else "bids"
        other = "bids" if key == "asks" else "asks"
        rows = raw.get(key)
        if rows is None:
            rows = raw.get(other)
        if not isinstance(rows, list):
            return []

        levels: List[BookLevel] = []
        for row in rows:
            price = size = None
            if isinstance(row, dict):
                price = _num(row.get("price"))
                size = _num(row.get("size")) if row.get("size") is not None \
                    else _num(row.get("quantity"))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                price, size = _num(row[0]), _num(row[1])
            if price is None or size is None:
                continue
            if not (0.0 < price < 1.0) or size <= 0:
                continue
            levels.append(BookLevel(price=price, size=size))

        # BUY takes the cheapest offers first; SELL takes the highest bids first.
        levels.sort(key=lambda level: level.price, reverse=(key == "bids"))
        return levels

    @staticmethod
    def best(levels: List[BookLevel]) -> Optional[BookLevel]:
        return levels[0] if levels else None

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def simulate(self, book: Any, side: str, amount_usd: float,
                 limit_price: Optional[float] = None,
                 mechanics: Optional[MarketMechanics] = None,
                 book_source: str = "unknown",
                 order_type: str = "GTC") -> PaperFill:
        """
        Price an order against a book. Never sends anything.

        A limit that the market does not already cross fills nothing and rests.
        That is the honest answer and the one a naive simulator gets wrong: it is
        the difference between a strategy that works and a strategy that assumes
        every limit it posts gets filled.
        """
        mech = mechanics or self.mechanics or MarketMechanics.assumed(
            "paper broker had no mechanics for this market")
        side_const = str(side).upper()
        fill = PaperFill(side=side_const, requested_usd=float(amount_usd or 0.0),
                         book_source=book_source, fee_rate=self.taker_fee_rate)

        if not mech.is_real:
            fill.warnings.append(
                "mechanics were assumed, not read from the venue, so the tick "
                "and minimum size may not be the venue's")
        if book_source in ("assumed_default", "unknown", "unavailable"):
            fill.is_real = False
            fill.warnings.append(
                "no real orderbook: the fill is simulated against an assumed "
                "book and is not evidence about this market")
        else:
            fill.is_real = True

        if fill.requested_usd <= 0:
            fill.reason = "requested amount is zero"
            return fill

        if limit_price is not None:
            limit_price = mech.round_price(float(limit_price), side_const)
            fill.limit_price = limit_price

        levels = self.levels_from_raw(book, side_const)
        if not levels:
            fill.reason = ("no orderbook levels on the side this order would "
                           "consume, so nothing could fill")
            fill.would_rest = True
            return fill

        best = levels[0]
        fill.best_price = best.price

        # Marketable? A BUY can only trade at or below its limit.
        if limit_price is not None:
            if side_const in ("BUY", "YES"):
                fill.is_marketable = limit_price >= best.price
            else:
                fill.is_marketable = limit_price <= best.price
        else:
            fill.is_marketable = True

        fill.depth_available_usd = round(
            sum(level.notional for level in levels
                if limit_price is None or self._touchable(level, limit_price, side_const)),
            6)

        if not fill.is_marketable:
            fill.would_rest = True
            fill.unfilled_usd = fill.requested_usd
            fill.reason = (
                f"limit {limit_price} does not cross the market ({best.price}); "
                f"the order would rest in the book and fill only if the price "
                f"comes to it"
            )
            return fill

        # Walk the ladder.
        remaining = fill.requested_usd
        taken_shares = 0.0
        taken_usd = 0.0
        worst = best.price
        for level in levels:
            if remaining <= 1e-9:
                break
            if limit_price is not None and not self._touchable(level, limit_price, side_const):
                break
            spend = min(remaining, level.notional)
            shares = spend / level.price if level.price > 0 else 0.0
            if shares <= 0:
                continue
            taken_shares += shares
            taken_usd += shares * level.price
            remaining -= shares * level.price
            worst = level.price
            fill.levels_consumed += 1

        fill.filled_shares = taken_shares
        fill.filled_usd = taken_usd
        fill.worst_price = worst
        fill.unfilled_usd = max(0.0, fill.requested_usd - taken_usd)

        if taken_shares <= 0:
            fill.reason = "no level was touchable within the limit"
            fill.would_rest = True
            fill.unfilled_usd = fill.requested_usd
            return fill

        fill.avg_price = taken_usd / taken_shares
        fill.fee_usd = round(taken_usd * self.taker_fee_rate, 6)

        # Slippage is measured against the touch, which is what the strategy
        # thought it was paying. A positive number is worse than the touch.
        fill.slippage_usd = taken_usd - (taken_shares * best.price)
        fill.slippage_bps = ((fill.avg_price - best.price) / best.price * 10_000
                             if best.price > 0 else 0.0)

        # The venue rejects an order below its own minimum, so the simulator
        # must too - otherwise paper mode accepts trades that cannot exist.
        if taken_shares + 1e-9 < mech.min_order_size:
            fill.warnings.append(
                f"filled size {taken_shares:.4f} is below the venue minimum "
                f"{mech.min_order_size}; a real order of this size would be "
                f"rejected")
            fill.reason = (fill.reason + " | " if fill.reason else "") + (
                f"fill size below the venue minimum {mech.min_order_size}")

        if fill.is_partial:
            if order_type.upper() in ("GTC", "GTD"):
                fill.would_rest = True
                fill.reason = (fill.reason + " | " if fill.reason else "") + (
                    f"only ${taken_usd:.4f} of ${fill.requested_usd:.4f} was "
                    f"available; the remainder would rest in the book")
                fill.unfilled_shares = (fill.unfilled_usd / fill.avg_price
                                        if fill.avg_price > 0 else 0.0)
            else:
                fill.reason = (fill.reason + " | " if fill.reason else "") + (
                    f"{order_type}: unfilled ${fill.unfilled_usd:.4f} cancelled "
                    f"rather than rested")
                fill.unfilled_usd = 0.0

        if not fill.reason:
            fill.reason = (f"filled {taken_shares:.4f} shares for ${taken_usd:.4f} "
                           f"across {fill.levels_consumed} level(s); "
                           f"slippage {fill.slippage_bps:.0f}bps vs the touch")
        return fill

    @staticmethod
    def _touchable(level: BookLevel, limit_price: float, side: str) -> bool:
        if side.upper() in ("BUY", "YES"):
            return level.price <= limit_price + 1e-12
        return level.price >= limit_price - 1e-12

    def simulate_resting(self, order_limit_price: float, side: str,
                         remaining_usd: float, book: Any,
                         book_source: str = "unknown",
                         mechanics: Optional[MarketMechanics] = None) -> PaperFill:
        """
        Would a resting order fill against this new snapshot?

        A resting BUY at 0.40 only fills if the market is offering at 0.40 or
        below - that is, someone is willing to sell at the price we are bidding.
        The snapshot has to CROSS the limit, not merely touch it, and the fill is
        capped at what is actually available.

        There is no queue model here, and that makes any resting fill optimistic:
        in a real book, an order at the touch is behind everyone who got there
        first. The result says so rather than pretending otherwise.
        """
        mech = mechanics or self.mechanics or MarketMechanics.assumed(
            "paper broker had no mechanics for this market")
        side_const = str(side).upper()
        levels = self.levels_from_raw(book, side_const)
        fill = PaperFill(side=side_const, requested_usd=float(remaining_usd or 0.0),
                         limit_price=float(order_limit_price), book_source=book_source,
                         fee_rate=self.taker_fee_rate)

        if not levels:
            fill.reason = "no book to check the resting order against"
            fill.would_rest = True
            return fill

        best = levels[0]
        fill.best_price = best.price
        crosses = (best.price <= order_limit_price + 1e-12
                   if side_const in ("BUY", "YES")
                   else best.price >= order_limit_price - 1e-12)
        if not crosses:
            fill.would_rest = True
            fill.unfilled_usd = fill.requested_usd
            fill.reason = (f"market at {best.price} has not reached the resting "
                           f"limit {order_limit_price}; the order stays in the book")
            return fill

        # It crosses. Fill up to the available size, at the limit price: a
        # passive order is filled at its own price, not at a better one.
        available_usd = sum(level.notional for level in levels
                            if self._touchable(level, order_limit_price, side_const))
        spend = min(fill.requested_usd, available_usd)
        shares = spend / order_limit_price if order_limit_price > 0 else 0.0
        fill.filled_shares = mech.round_size(shares)
        fill.filled_usd = fill.filled_shares * order_limit_price
        fill.avg_price = order_limit_price
        fill.worst_price = order_limit_price
        fill.unfilled_usd = max(0.0, fill.requested_usd - fill.filled_usd)
        fill.is_marketable = True
        fill.would_rest = fill.unfilled_usd > 1e-9
        fill.fee_usd = round(fill.filled_usd * self.taker_fee_rate, 6)
        fill.is_real = book_source not in ("assumed_default", "unknown", "unavailable")
        fill.optimistic = True
        fill.warnings.append(
            "resting fills have no queue model, so this is an upper bound: a "
            "real order at the touch may sit behind others and never fill")
        fill.reason = (f"market crossed to {best.price}; filled "
                       f"{fill.filled_shares:.4f} shares at the resting limit "
                       f"{order_limit_price}")
        if fill.unfilled_usd > 1e-9:
            fill.reason += f"; ${fill.unfilled_usd:.4f} stays resting"
        return fill

    # ------------------------------------------------------------------
    # a book, or a labelled stand-in
    # ------------------------------------------------------------------

    @staticmethod
    def book_for_market(context: Dict[str, Any], market_id: str) -> Tuple[Any, str]:
        """
        The real book for a market, or a labelled empty one.

        Returns (book, source). "assumed_default" means no book was available -
        a caller must not treat a fill priced from it as evidence.
        """
        book = None
        if isinstance(context, dict):
            raw = context.get("orderbook") or context.get("books")
            if isinstance(raw, dict):
                book = raw.get(market_id) if market_id in raw else raw
        if isinstance(book, dict) and (book.get("bids") or book.get("asks")):
            return book, "orderbook"
        return None, "assumed_default"


def simulate_paper_fill(book: Any, side: str, amount_usd: float,
                        limit_price: Optional[float] = None,
                        mechanics: Optional[MarketMechanics] = None,
                        fee_rate: Optional[float] = None,
                        book_source: str = "orderbook") -> PaperFill:
    """Convenience wrapper for a one-off simulation."""
    broker = PaperBroker(mechanics=mechanics, taker_fee_rate=fee_rate)
    return broker.simulate(book, side, amount_usd, limit_price=limit_price,
                           mechanics=mechanics, book_source=book_source)
