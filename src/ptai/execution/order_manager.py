"""
Order Manager - order lifecycle and reconciliation.

This class existed, wrote rows into the `orders` table, and was never called by
the trading loop. Two references in `v3_loop.py`: the import and the
constructor. So an order could be submitted, rest in the book, fill in pieces
over hours, and the agent would know none of it. The one thing the executor
deliberately refuses to guess - "is this order still out there?" - had no answer.

The rule this module implements:

    UNLESS AN ORDER IS CONFIRMED FINAL, IT STILL EXISTS.

Every submitted order is recorded under the VENUE's order id, and every cycle
each order that is not final is re-read from the venue. Three outcomes matter,
and each was previously unrepresentable:

  * it filled MORE than last time -> the position grows, at the size-weighted
    average price actually paid;
  * it is gone from the venue and unfilled -> it was cancelled or expired, and
    the capital it was holding is released;
  * the venue cannot be asked -> the order stays open and is reported as
    unreconciled. It is never silently assumed dead, because assuming that is
    how an agent ends up with exposure it does not believe it has.

An unconfirmed send - a request whose response was lost - is treated as a live
order until the venue is asked. That is the conservative direction: the order
may be resting, and the only way to find out is to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

# Statuses a venue may report that mean "still working".
WORKING_STATUSES = ("pending", "submitted", "live", "delayed", "matched", "partial", "open")
# Statuses that mean "finished, one way or another".
FINAL_STATUSES = ("filled", "cancelled", "canceled", "rejected", "failed",
                  "expired", "unmatched", "not_cancelled", "abandoned")


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    # The send's fate is unknown: it may be resting at the venue.
    UNCONFIRMED = "unconfirmed_send"


@dataclass
class Order:
    id: str
    market_id: str
    token_id: Optional[str]
    side: str
    max_price: float
    max_spend_usd: float
    amount: float = 0.0
    avg_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    venue_id: str = "polymarket"
    raw_response: Dict = field(default_factory=dict)
    guard_checks: List[str] = field(default_factory=list)
    # Reconciliation state.
    size_matched: float = 0.0
    original_size: float = 0.0
    matched_usd: float = 0.0
    trade_id: Optional[int] = None


@dataclass
class ReconciliationReport:
    """What the venue said about our orders this cycle."""

    checked: int = 0
    # Positions created from a fill on an order that was resting.
    opened: int = 0
    filled_more: int = 0
    now_complete: int = 0
    released: int = 0
    unreconciled: int = 0
    grew_usd: float = 0.0
    released_usd: float = 0.0
    items: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def changed(self) -> bool:
        return bool(self.filled_more or self.now_complete or self.released)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "opened": self.opened,
            "filled_more": self.filled_more,
            "now_complete": self.now_complete,
            "released": self.released,
            "unreconciled": self.unreconciled,
            "grew_usd": round(self.grew_usd, 6),
            "released_usd": round(self.released_usd, 6),
            "changed": self.changed,
            "items": self.items,
        }


class OrderManager:
    """
    Deterministic order manager and reconciler.

    The guard half is unchanged in spirit: the execution layer refuses an order
    the operator's own limits forbid, regardless of what any model proposes. The
    new half is that an order, once sent, is tracked until the venue says it is
    finished.
    """

    def __init__(self, storage=None):
        self.storage = storage
        # Kept for callers that rely on the in-memory view. It is a cache, never
        # the source of truth: the venue's answer is the truth.
        self.orders: Dict[str, Order] = {}
        self.max_price_guard = 0.99
        self.min_price_guard = 0.01
        self.max_spend_guard_usd = 1000.0

    # ------------------------------------------------------------------
    # guards and creation
    # ------------------------------------------------------------------

    def create_order(self, market_id: str, token_id: str, side: str,
                     max_price: float, max_spend_usd: float,
                     venue_id: str = "polymarket") -> tuple[bool, str, Optional[Order]]:
        """
        Create order with guard checks.
        Returns (allowed, reason, order)
        """
        checks = []
        if not (self.min_price_guard <= max_price <= self.max_price_guard):
            return False, (f"Price {max_price} out of guard bounds "
                           f"[{self.min_price_guard}, {self.max_price_guard}]"), None
        checks.append(f"Price {max_price} within bounds")
        if max_spend_usd <= 0:
            return False, f"Spend ${max_spend_usd} <= 0", None
        if max_spend_usd > self.max_spend_guard_usd:
            return False, (f"Spend ${max_spend_usd} > absolute max "
                           f"${self.max_spend_guard_usd}"), None
        checks.append(f"Spend ${max_spend_usd} within absolute max")
        if side not in ["YES", "NO", "BUY", "SELL"]:
            return False, f"Invalid side {side}", None
        checks.append(f"Side {side} valid")

        order_id = str(uuid.uuid4())[:8]
        order = Order(id=order_id, market_id=market_id, token_id=token_id,
                      side=side, max_price=max_price, max_spend_usd=max_spend_usd,
                      venue_id=venue_id, guard_checks=checks)
        self.orders[order_id] = order
        if self.storage:
            try:
                self.storage.conn.execute(
                    "INSERT INTO orders (id, market_id, side, max_price, max_spend, "
                    "status, created_at, venue_id, token_id, requested_usd, "
                    "limit_price) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (order_id, market_id, side, max_price, max_spend_usd,
                     order.status.value, order.created_at.isoformat(), venue_id,
                     token_id, max_spend_usd, max_price))
                self.storage.conn.commit()
            except Exception as e:
                logger.error(f"Order DB save failed: {type(e).__name__}: {e}")
        logger.info(f"Order created {order_id}: {market_id} {side} "
                    f"max_price {max_price} max_spend ${max_spend_usd} checks {checks}")
        return True, "Order created with guard checks", order

    def update_order(self, order_id: str, status: OrderStatus, amount: float = 0,
                     avg_price: float = 0, raw_response: Dict = None):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            order.amount = amount
            order.avg_price = avg_price
            if raw_response:
                order.raw_response = raw_response
            if status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                order.filled_at = datetime.now(timezone.utc)
            logger.info(f"Order {order_id} updated to {status} amount {amount} "
                        f"avg_price {avg_price}")

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self.orders.values()
                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED,
                                OrderStatus.PARTIAL, OrderStatus.UNCONFIRMED]]

    def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """
        Cancel an order in every place it exists.

        The in-memory cache AND the orders table: a cancel that only touched
        the cache was forgotten on restart, so the kill switch - which walks
        the storage rows - could not release the reservations it was written
        to release, and reserved capital sat locked behind an order the agent
        had already called off.
        """
        cancelled = False
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED,
                                OrderStatus.PARTIAL, OrderStatus.UNCONFIRMED]:
                order.status = OrderStatus.CANCELLED
                order.raw_response["cancel_reason"] = reason
                cancelled = True
        if self.storage is not None:
            try:
                row = self.storage.get_order_row(order_id)
            except Exception:
                row = None
            if row is not None:
                if str(row.get("status") or "") not in FINAL_STATUSES:
                    try:
                        self.storage.upsert_order({
                            "order_id": order_id,
                            "status": "cancelled",
                            "terminal_reason": f"cancelled: {reason}",
                            "last_synced_at":
                                datetime.now(timezone.utc).isoformat(),
                        })
                        cancelled = True
                    except Exception as e:
                        logger.error(
                            f"Could not record the cancel of {order_id}: "
                            f"{type(e).__name__}: {e} - the reservation "
                            f"stays until reconciliation proves it is gone")
        if cancelled:
            logger.info(f"Order {order_id} cancelled: {reason}")
        return cancelled

    def cancel_all(self, reason: str = "kill_switch") -> int:
        """
        Cancel every working order, in memory AND in storage.

        The in-memory cache is what a live process holds; the orders table is
        what a restarted process holds. Walking only the cache is how a kill
        switch after a restart would release nothing while the reservations
        stayed locked behind orders the agent no longer sees.
        """
        cancelled = set()
        for order in list(self.orders.values()):
            if self.cancel_order(order.id, reason=reason):
                cancelled.add(order.id)
        if self.storage is not None:
            try:
                rows = self.storage.get_open_orders()
            except Exception:
                rows = []
            for row in rows:
                row_id = str(row.get("order_id") or "")
                if row_id and self.cancel_order(row_id, reason=reason):
                    cancelled.add(row_id)
        logger.warning(
            f"Cancelled {len(cancelled)} open order(s): {reason}")
        return len(cancelled)

    def get_status_report(self) -> Dict:
        open_orders = self.get_open_orders()
        return {
            "total_orders": len(self.orders),
            "open_orders": len(open_orders),
            "open_orders_list": [
                {"id": o.id, "market_id": o.market_id, "side": o.side,
                 "max_spend": o.max_spend_usd, "status": o.status.value}
                for o in open_orders
            ],
            "guards": {
                "max_price": self.max_price_guard,
                "min_price": self.min_price_guard,
                "max_spend_usd": self.max_spend_guard_usd,
            },
        }

    # ------------------------------------------------------------------
    # recording what was submitted
    # ------------------------------------------------------------------

    def record_submission(self, exec_result, market_id: str,
                          token_id: Optional[str] = None,
                          venue_id: Optional[str] = None,
                          trade_id: Optional[int] = None,
                          side: Optional[str] = None,
                          forecast: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Persist an order that has been sent, so it can be reconciled later.

        Returns the order key. An order with no venue id still gets a local key,
        because the size_matched and original_size the venue reported are the
        only record of a partial fill and they must not be lost.

        An unconfirmed send is recorded deliberately: its whole purpose is to be
        asked about next cycle. `side` is the outcome side, kept so that a fill
        arriving later can open a position that settlement can actually resolve.

        `forecast` is the thesis behind the order - fair value, edge, confidence,
        strategy, category. It is written down HERE, with the order, because an
        order that rests and fills hours later is attributed to whatever the fill
        knows about it, and a fill knows nothing. Without this the position was
        recorded as strategy "resting_order_fill" with edge 0 and confidence 0,
        so the outcome taught the agent about a strategy that never chose the
        trade.
        """
        if self.storage is None:
            logger.error("OrderManager has no storage: a submitted order cannot be "
                         "reconciled and its fill will be lost")
            return None

        # What actually happened to the money, and what the trade was predicted
        # to earn, recorded WITH the order. A fill hours later has none of this
        # and cannot reconstruct it.
        forecast = dict(forecast or {})
        if getattr(exec_result, "is_simulated", None) is not None:
            forecast.setdefault(
                "execution_mode",
                "paper" if getattr(exec_result, "is_simulated") else "live")
        if side is not None:
            forecast.setdefault("side", side)

        venue_order_id = str(getattr(exec_result, "order_id", "") or "")
        key = venue_order_id or f"local-{uuid.uuid4().hex[:12]}"
        status = str(getattr(exec_result, "status", "") or "unknown")
        if getattr(exec_result, "unconfirmed_send", False):
            # A failed send is not final: it may be resting.
            status = OrderStatus.UNCONFIRMED.value

        matched_usd = float(getattr(exec_result, "filled_usd", 0.0) or 0.0)
        original_size = getattr(exec_result, "original_size", None)
        size_matched = getattr(exec_result, "size_matched", None)

        recorded = {
            "order_id": key,
            "market_id": market_id,
            "token_id": token_id,
            # The OUTCOME side (YES/NO), not the buy/sell direction. If this
            # order fills later, the position it opens must be settleable, and
            # settlement resolves a market by its outcome side.
            "side": side,
            # The resting LIMIT, not the average fill: a multi-level walk
            # fills below the limit, but the remainder rests AT the limit,
            # and that is the price reconciliation re-simulates a cross
            # against.
            "limit_price": float(getattr(exec_result, "limit_price", 0.0)
                                 or getattr(exec_result, "filled_price", 0.0)
                                 or getattr(exec_result, "price", 0.0) or 0.0),
            "requested_usd": float(getattr(exec_result, "amount_usd", 0.0) or 0.0),
            "status": status,
            "venue_id": venue_id or getattr(exec_result, "venue_id", ""),
            "original_size": original_size,
            "size_matched": size_matched,
            "matched_usd": matched_usd,
            "trade_id": trade_id,
            "raw": getattr(exec_result, "raw_response", {}) or {},
        }
        if forecast:
            recorded.update({
                "fair_price": forecast.get("fair_price"),
                "edge": forecast.get("edge"),
                "confidence": forecast.get("confidence"),
                "strategy": forecast.get("strategy"),
                "category": forecast.get("category"),
                "data_mode": forecast.get("data_mode"),
                # The execution facts. Named here or `upsert_order` drops them,
                # and this INSERT has silently lost columns before.
                "execution_mode": forecast.get("execution_mode"),
                "yes_price": forecast.get("yes_price"),
                "token_price": forecast.get("token_price"),
                "expected_net_ev": forecast.get("expected_net_ev"),
                "expected_net_ev_pct": forecast.get("expected_net_ev_pct"),
            })
        if not self.storage.upsert_order(recorded):
            return None
        return key

    def attach_position(self, order_key: str, trade_id: int) -> bool:
        """
        Tie an order to the position row its fills belong in.

        One position row per market, grown by later fills, because settlement
        finds an open trade by market id and would never close a second row.
        """
        if self.storage is None or not order_key:
            return False
        try:
            self.storage.conn.execute(
                "UPDATE orders SET trade_id = ? WHERE id = ?", (trade_id, order_key))
            self.storage.conn.commit()
            return True
        except Exception as e:
            logger.error(f"attach_position({order_key} -> {trade_id}) failed: {e}")
            return False

    def open_orders(self, venue_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Orders the agent believes are still working, from storage.

        Read from storage rather than the in-memory dict so a restart does not
        forget an order that is still resting at the venue.
        """
        if self.storage is None:
            return [self._order_to_dict(o) for o in self.get_open_orders()]
        return self.storage.get_open_orders(venue_id=venue_id)

    def resting_capital_usd(self, venue_id: Optional[str] = None) -> float:
        """
        Cash locked behind unfilled orders.

        Separate from booked position cost: a booked position's cost lives in
        the trades table, an unfilled order's reservation lives here. Adding
        both is correct; adding either twice is not.
        """
        if self.storage is None:
            return sum(max(0.0, o.max_spend_usd - o.matched_usd)
                       for o in self.get_open_orders())
        return self.storage.resting_capital_usd(venue_id=venue_id)

    @staticmethod
    def _order_to_dict(order: Order) -> Dict[str, Any]:
        return {
            "order_id": order.id,
            "market_id": order.market_id,
            "token_id": order.token_id,
            "side": order.side,
            "limit_price": order.max_price,
            "requested_usd": order.max_spend_usd,
            "status": order.status.value,
            "venue_id": order.venue_id,
            "original_size": order.original_size,
            "size_matched": order.size_matched,
            "matched_usd": order.matched_usd,
            "trade_id": order.trade_id,
        }

    # ------------------------------------------------------------------
    # reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self, adapter, venue_id: Optional[str] = None,
                        max_orders: int = 50,
                        position_opener=None) -> ReconciliationReport:
        """
        Ask the venue about every order that is not final, and believe it.

        For each working order:
          * read the venue's own view of the order;
          * if more of it has matched than last time, grow the position by the
            difference at the venue's reported price;
          * if it is now final, stop tracking it;
          * if it has vanished from the venue's working set without matching its
            full size, treat it as finished and release the capital;
          * if the venue cannot be asked at all, leave it open and say so.

        `position_opener` is how a fill on an order that was NOT a position gets
        booked. A resting order buys nothing, so no position row is created for
        it; when it fills later there is nothing for the fill to attach to. The
        opener comes from the caller because only the caller knows how a
        position is recorded (data mode, strategy, calibration fields). Without
        one, a later fill is refused and reported rather than silently dropped.
        """
        report = ReconciliationReport()
        orders = self.open_orders(venue_id=venue_id)
        if not orders:
            return report
        if adapter is None:
            report.unreconciled = len(orders)
            logger.warning(
                f"Order reconciliation skipped: no adapter, so {len(orders)} "
                f"working order(s) stay unreconciled and their capital stays "
                f"reserved")
            return report

        venue_open = None
        try:
            listing = await _call_adapter(adapter, "get_open_orders")
            if isinstance(listing, dict) and listing.get("available"):
                # The venue's working set, by id. Used only to decide that an
                # order is GONE, never to decide that it filled.
                venue_open = {str(o.get("id") or o.get("orderID") or ""): o
                              for o in listing.get("orders") or []}
        except Exception as e:
            logger.warning(f"Could not list venue open orders: {type(e).__name__}: {e}")

        for order in orders[:max_orders]:
            order_id = str(order.get("order_id") or "")
            if not order_id:
                continue
            report.checked += 1
            item = {"order_id": order_id, "market_id": order.get("market_id")}

            # A paper order never reached the venue, so asking the venue
            # about it would be a question with no answer. Its venue is the
            # current book, and its fate is re-simulated against it.
            if str(order.get("execution_mode") or "").lower() == "paper":
                await self._reconcile_paper_order(adapter, order, item, report,
                                                  position_opener)
                continue

            if order_id.startswith("local-"):
                # Never reached the venue with an id, so it cannot be looked up.
                # Left open and reported: it may exist under an id we do not have.
                report.unreconciled += 1
                item.update({"outcome": "no_venue_id",
                             "reason": "the order has no venue id, so it cannot be "
                                       "looked up; it may still be resting"})
                report.items.append(item)
                continue

            try:
                state = await _call_adapter(adapter, "get_order", order_id)
            except Exception as e:
                report.unreconciled += 1
                item.update({"outcome": "unreadable",
                             "reason": f"{type(e).__name__}: {e}"})
                report.items.append(item)
                continue

            readable = isinstance(state, dict) and bool(state.get("available"))
            gone_from_book = venue_open is not None and order_id not in venue_open

            if not readable:
                if gone_from_book:
                    # The order-level read failed, but the venue's working set
                    # is authoritative and the order is not in it. The listing
                    # answered the question the single read could not.
                    report.released += 1
                    report.released_usd += float(order.get("requested_usd") or 0.0)
                    self.storage.upsert_order({
                        "order_id": order_id, "status": "cancelled",
                        "size_matched": order.get("size_matched"),
                        "matched_usd": order.get("matched_usd"),
                        "terminal_reason": ("absent from the venue's open orders "
                                            "and no longer individually readable"),
                        "last_synced_at": datetime.now(timezone.utc).isoformat(),
                    })
                    item.update({
                        "outcome": "final", "status": "cancelled",
                        "reason": "no longer in the venue's open orders"})
                    report.items.append(item)
                    continue
                # The venue would not answer and its working set is unknown. An
                # unanswered order is not a finished order; it stays open and
                # keeps its capital reserved.
                report.unreconciled += 1
                item.update({"outcome": "unavailable",
                             "reason": (state or {}).get("reason", "venue did not answer")})
                report.items.append(item)
                continue

            venue_status = str(state.get("status") or "").lower()
            # The CLOB reports these as strings ("5.45"); an isinstance check on
            # a number would discard the venue's own fill size.
            matched_size = _num(state.get("size_matched"))
            reported_price = _num(state.get("price"))

            previous_usd = float(order.get("matched_usd") or 0.0)
            previous_size = float(order.get("size_matched") or 0.0)

            # Only a size the venue actually reported is bookable.
            grew_usd = 0.0
            if matched_size is not None and matched_size > previous_size + 1e-9:
                price = reported_price or float(order.get("limit_price") or 0.0)
                delta_size = matched_size - previous_size
                grew_usd = delta_size * price if price else 0.0
                if grew_usd > 0 and order.get("trade_id"):
                    ok = self.storage.add_to_position(
                        trade_id=int(order["trade_id"]), add_usd=grew_usd,
                        add_price=float(price))
                    if ok:
                        report.filled_more += 1
                        report.grew_usd += grew_usd
                    else:
                        report.unreconciled += 1
                        item["position_update_failed"] = True
                elif grew_usd > 0:
                    # A fill on an order that was never a position (it rested,
                    # then filled). There is no row to add it to, so one has to
                    # be opened or the exposure exists and the ledger does not
                    # know about it.
                    new_trade_id = None
                    if position_opener is not None:
                        try:
                            new_trade_id = position_opener(order, grew_usd, float(price))
                        except Exception as e:
                            logger.error(
                                f"Opening a position for order {order_id} raised "
                                f"{type(e).__name__}: {e}")
                    if new_trade_id:
                        self.attach_position(order_id, int(new_trade_id))
                        report.filled_more += 1
                        report.grew_usd += grew_usd
                        report.opened += 1
                        item["opened_trade_id"] = int(new_trade_id)
                    else:
                        logger.error(
                            f"Order {order_id} reports {delta_size} newly matched "
                            f"shares worth ${grew_usd:.4f} and no position was "
                            f"opened for them; the ledger will understate exposure")
                        report.unreconciled += 1
                        item["orphan_fill"] = round(grew_usd, 6)

            new_matched_usd = previous_usd + grew_usd
            if matched_size is not None:
                new_size = float(matched_size)
            else:
                new_size = previous_size

            original_size = order.get("original_size")
            fully_matched = bool(original_size) and new_size + 1e-9 >= float(original_size)

            final = False
            reason = ""
            if venue_status and venue_status in ("canceled", "cancelled", "expired",
                                                 "unmatched", "rejected", "failed"):
                final, reason = True, f"venue reports {venue_status}"
            elif venue_status == "matched" and fully_matched:
                final, reason = True, "fully matched"
            elif venue_open is not None and order_id not in venue_open:
                # Gone from the working set. Either it completed or it was
                # pulled; either way it is no longer holding capital at the
                # venue, so it stops being tracked here.
                final = True
                reason = ("no longer in the venue's open orders"
                          + ("; fully matched" if fully_matched else
                             "; unfilled remainder released"))

            if final:
                released = max(0.0, float(order.get("requested_usd") or 0.0)
                               - new_matched_usd)
                report.released += 1
                report.released_usd += released
                if fully_matched:
                    report.now_complete += 1
                self.storage.upsert_order({
                    "order_id": order_id,
                    "status": "filled" if fully_matched else "cancelled",
                    "size_matched": new_size,
                    "matched_usd": new_matched_usd,
                    "last_synced_at": datetime.now(timezone.utc).isoformat(),
                    "terminal_reason": reason,
                })
                item.update({"outcome": "final", "reason": reason,
                             "grew_usd": round(grew_usd, 6),
                             "released_usd": round(released, 6),
                             "status": "filled" if fully_matched else "cancelled"})
            else:
                self.storage.upsert_order({
                    "order_id": order_id,
                    "status": venue_status or order.get("status"),
                    "size_matched": new_size,
                    "matched_usd": new_matched_usd,
                    "last_synced_at": datetime.now(timezone.utc).isoformat(),
                })
                item.update({"outcome": "still_working", "status": venue_status,
                             "grew_usd": round(grew_usd, 6),
                             "size_matched": new_size,
                             "original_size": original_size})
            report.items.append(item)

        logger.info(
            f"Order reconciliation: checked {report.checked}, grew {report.filled_more} "
            f"(${report.grew_usd:.4f}), completed {report.now_complete}, released "
            f"{report.released} (${report.released_usd:.4f}), unreconciled "
            f"{report.unreconciled}")
        return report

    async def _reconcile_paper_order(self, adapter, order: Dict[str, Any],
                                     item: Dict[str, Any],
                                     report: "ReconciliationReport",
                                     position_opener=None) -> None:
        """
        Re-simulate a paper order's fate against the current book.

        A paper order never reached the venue, so `get_order` would be a
        question with no answer. The venue's CURRENT book is the simulator:
        a cross is a fill, booked exactly the way a live resting fill is
        (grow the position it belongs to, or open one); no cross means it
        is still resting and keeps reserving capital; a book that cannot be
        read leaves the order open - the same conservative rule as a live
        venue that does not answer.
        """
        from .paper_broker import PaperBroker

        order_id = str(order.get("order_id") or "")
        now = datetime.now(timezone.utc).isoformat()
        remaining_usd = max(0.0, float(order.get("requested_usd") or 0.0)
                            - float(order.get("matched_usd") or 0.0))
        if remaining_usd <= 0:
            self.storage.upsert_order({
                "order_id": order_id, "status": "filled",
                "size_matched": order.get("size_matched"),
                "matched_usd": order.get("matched_usd"),
                "terminal_reason": "paper order fully matched",
                "last_synced_at": now,
            })
            report.now_complete += 1
            item.update({"outcome": "final", "status": "filled",
                         "reason": "fully matched"})
            report.items.append(item)
            return

        limit = float(order.get("limit_price") or 0.0)
        side = str(order.get("side") or "YES").upper()
        token_id = str(order.get("token_id") or "")
        reader = getattr(adapter, "get_token_orderbook", None)
        if reader is None or not token_id or limit <= 0:
            report.unreconciled += 1
            item.update({"outcome": "unavailable",
                         "reason": ("no book can be read for this paper "
                                    "order, so it stays open and keeps its "
                                    "capital reserved")})
            report.items.append(item)
            return

        try:
            book_env = await reader(token_id)
        except Exception as e:
            book_env = {"available": False, "reason": f"{type(e).__name__}: {e}"}
        book = (book_env or {}).get("book") if isinstance(book_env, dict) else None
        if not isinstance(book, dict) or not (book.get("bids") or book.get("asks")):
            # A book that cannot be read is not a book that did not cross.
            # The order stays open and keeps reserving capital, exactly as
            # an unanswered live venue would.
            reason = (book_env or {}).get("reason", "no book") \
                if isinstance(book_env, dict) else "no book"
            report.unreconciled += 1
            item.update({"outcome": "unavailable",
                         "reason": f"paper book unreadable: {reason}"})
            report.items.append(item)
            return

        fill = PaperBroker().simulate_resting(limit, side, remaining_usd, book,
                                              book_source="orderbook")
        previous_usd = float(order.get("matched_usd") or 0.0)
        previous_size = float(order.get("size_matched") or 0.0)
        grew_usd = fill.filled_usd

        if grew_usd > 1e-9:
            price = float(fill.avg_price or limit)
            if order.get("trade_id"):
                ok = self.storage.add_to_position(
                    trade_id=int(order["trade_id"]), add_usd=grew_usd,
                    add_price=float(price))
                if ok:
                    report.filled_more += 1
                    report.grew_usd += grew_usd
                else:
                    report.unreconciled += 1
                    item["position_update_failed"] = True
            else:
                new_trade_id = None
                if position_opener is not None:
                    try:
                        new_trade_id = position_opener(order, grew_usd, price)
                    except Exception as e:
                        logger.error(
                            f"Opening a position for paper order {order_id} "
                            f"raised {type(e).__name__}: {e}")
                if new_trade_id:
                    self.attach_position(order_id, int(new_trade_id))
                    report.filled_more += 1
                    report.grew_usd += grew_usd
                    report.opened += 1
                    item["opened_trade_id"] = int(new_trade_id)
                else:
                    logger.error(
                        f"Paper order {order_id} filled ${grew_usd:.4f} and no "
                        f"position was opened for it; the ledger will "
                        f"understate exposure")
                    report.unreconciled += 1
                    item["orphan_fill"] = round(grew_usd, 6)

        new_usd = previous_usd + grew_usd
        new_size = previous_size + fill.filled_shares
        original_size = order.get("original_size")
        fully_matched = (bool(original_size)
                         and new_size + 1e-9 >= float(original_size))

        if fully_matched:
            released = max(0.0, float(order.get("requested_usd") or 0.0)
                           - new_usd)
            report.released += 1
            report.released_usd += released
            report.now_complete += 1
            self.storage.upsert_order({
                "order_id": order_id, "status": "filled",
                "size_matched": new_size, "matched_usd": new_usd,
                "terminal_reason": "paper order fully simulated",
                "last_synced_at": now,
            })
            item.update({"outcome": "final", "status": "filled",
                         "grew_usd": round(grew_usd, 6),
                         "released_usd": round(released, 6)})
        else:
            self.storage.upsert_order({
                "order_id": order_id,
                "status": order.get("status") or "dry_run",
                "size_matched": new_size, "matched_usd": new_usd,
                "last_synced_at": now,
            })
            item.update({"outcome": "still_resting",
                         "grew_usd": round(grew_usd, 6),
                         "size_matched": new_size,
                         "original_size": original_size})
        report.items.append(item)

    async def cancel_working(self, adapter, reason: str = "operator") -> int:
        """
        Cancel every working order at the venue. Returns how many were asked.

        Used by the kill switch: an order that is still restable is exposure the
        shutdown must not leave behind.
        """
        if adapter is None:
            return 0
        cancel = getattr(adapter, "cancel_order", None)
        if cancel is None:
            logger.error("Adapter cannot cancel orders; working orders remain exposed")
            return 0
        count = 0
        for order in self.open_orders():
            order_id = str(order.get("order_id") or "")
            if not order_id or order_id.startswith("local-"):
                continue
            try:
                result = await cancel(order_id)
            except Exception as e:
                logger.error(f"Cancel of {order_id} raised {type(e).__name__}: {e}")
                continue
            if isinstance(result, dict) and result.get("status") == "cancelled":
                count += 1
                self.storage.upsert_order({
                    "order_id": order_id, "status": "cancelled",
                    "terminal_reason": f"cancelled: {reason}",
                    "last_synced_at": datetime.now(timezone.utc).isoformat(),
                })
            else:
                logger.error(f"Cancel of {order_id} did not confirm: {result}")
        return count


def _num(value: Any) -> Optional[float]:
    """A number from a venue payload field, or None. Accepts numeric strings."""
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


async def _call_adapter(adapter, method_name: str, *args):
    """Call an adapter read, sync or async, without caring which."""
    import inspect

    method = getattr(adapter, method_name, None)
    if method is None:
        return {"available": False, "reason": f"adapter has no {method_name}()"}
    result = method(*args)
    if inspect.isawaitable(result):
        return await result
    return result
