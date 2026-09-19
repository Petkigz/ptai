"""
Order Manager - deterministic order management
Execution layer receives {market_id, token_id, side, max_price, max_spend} and nothing else
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import uuid
from loguru import logger


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    id: str
    market_id: str
    token_id: Optional[str]
    side: str  # BUY/SELL, YES/NO
    max_price: float
    max_spend_usd: float
    amount: float = 0.0  # filled amount
    avg_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    venue_id: str = "polymarket"
    raw_response: Dict = field(default_factory=dict)
    guard_checks: List[str] = field(default_factory=list)


class OrderManager:
    """
    Deterministic order manager - execution guard.
    LLM cannot say BUY $50000 because execution layer refuses.
    """
    def __init__(self, storage=None):
        self.storage = storage
        self.orders: Dict[str, Order] = {}
        self.max_price_guard = 0.99
        self.min_price_guard = 0.01
        self.max_spend_guard_usd = 1000.0  # absolute max per order

    def create_order(self, market_id: str, token_id: str, side: str, max_price: float, max_spend_usd: float, venue_id: str = "polymarket") -> tuple[bool, str, Optional[Order]]:
        """
        Create order with guard checks.
        Returns (allowed, reason, order)
        """
        checks = []

        # Guard: price bounds
        if not (self.min_price_guard <= max_price <= self.max_price_guard):
            return False, f"Price {max_price} out of guard bounds [{self.min_price_guard}, {self.max_price_guard}]", None
        checks.append(f"Price {max_price} within bounds")

        # Guard: spend bounds
        if max_spend_usd <= 0:
            return False, f"Spend ${max_spend_usd} <= 0", None
        if max_spend_usd > self.max_spend_guard_usd:
            return False, f"Spend ${max_spend_usd} > absolute max ${self.max_spend_guard_usd}", None
        checks.append(f"Spend ${max_spend_usd} within absolute max")

        # Guard: side valid
        if side not in ["YES", "NO", "BUY", "SELL"]:
            return False, f"Invalid side {side}", None
        checks.append(f"Side {side} valid")

        # Create order
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            id=order_id,
            market_id=market_id,
            token_id=token_id,
            side=side,
            max_price=max_price,
            max_spend_usd=max_spend_usd,
            venue_id=venue_id,
            guard_checks=checks
        )
        self.orders[order_id] = order

        # Persist
        if self.storage:
            try:
                self.storage.conn.execute(
                    "INSERT INTO orders (id, market_id, side, max_price, max_spend, status, created_at) VALUES (?,?,?,?,?,?,?)",
                    (order_id, market_id, side, max_price, max_spend_usd, order.status.value, order.created_at.isoformat())
                )
                self.storage.conn.commit()
            except Exception as e:
                logger.warning(f"Order DB save failed: {e}")

        logger.info(f"Order created {order_id}: {market_id} {side} max_price {max_price} max_spend ${max_spend_usd} checks {checks}")
        return True, "Order created with guard checks", order

    def update_order(self, order_id: str, status: OrderStatus, amount: float = 0, avg_price: float = 0, raw_response: Dict = None):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            order.amount = amount
            order.avg_price = avg_price
            if raw_response:
                order.raw_response = raw_response
            if status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                order.filled_at = datetime.now(timezone.utc)
            logger.info(f"Order {order_id} updated to {status} amount {amount} avg_price {avg_price}")

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]]

    def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                order.status = OrderStatus.CANCELLED
                order.raw_response["cancel_reason"] = reason
                logger.info(f"Order {order_id} cancelled: {reason}")
                return True
        return False

    def cancel_all(self, reason: str = "kill_switch"):
        for order in self.get_open_orders():
            self.cancel_order(order.id, reason=reason)
        logger.warning(f"Cancelled all open orders: {reason}")

    def get_status_report(self) -> Dict:
        open_orders = self.get_open_orders()
        return {
            "total_orders": len(self.orders),
            "open_orders": len(open_orders),
            "open_orders_list": [
                {"id": o.id, "market_id": o.market_id, "side": o.side, "max_spend": o.max_spend_usd, "status": o.status.value}
                for o in open_orders
            ],
            "guards": {
                "max_price": self.max_price_guard,
                "min_price": self.min_price_guard,
                "max_spend_usd": self.max_spend_guard_usd
            }
        }
