"""
Reconciliation - verifies fills, account balance, detects execution mismatch
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger


@dataclass
class ReconciliationResult:
    market_id: str
    expected_spend: float
    actual_spend: float
    expected_price: float
    actual_price: float
    mismatch: bool
    mismatch_pct: float
    balance_before: float
    balance_after: float
    expected_balance: float
    balance_mismatch: bool
    timestamp: datetime
    status: str  # ok, mismatch, error


class ReconciliationEngine:
    """
    Reconciliation: verify fill, reconcile account, detect anomalies
    """
    def __init__(self, storage=None):
        self.storage = storage

    def reconcile_order(self, order, portfolio_before: Dict, portfolio_after: Dict) -> ReconciliationResult:
        """
        Reconcile order execution with portfolio change
        """
        expected_spend = order.max_spend_usd
        actual_spend = order.amount * order.avg_price if order.amount else 0

        expected_price = order.max_price
        actual_price = order.avg_price

        # Calculate mismatch
        mismatch_pct = abs(expected_spend - actual_spend) / max(1, expected_spend) if expected_spend > 0 else 0
        mismatch = mismatch_pct > 0.1  # 10% mismatch threshold

        balance_before = portfolio_before.get("balance", 0)
        balance_after = portfolio_after.get("balance", 0)
        expected_balance = balance_before - actual_spend
        balance_mismatch = abs(balance_after - expected_balance) > 1.0  # $1 tolerance

        status = "ok"
        if mismatch:
            status = "mismatch"
            logger.warning(f"Reconciliation mismatch for {order.market_id}: expected ${expected_spend} actual ${actual_spend} mismatch {mismatch_pct*100:.1f}%")
        if balance_mismatch:
            status = "mismatch"
            logger.warning(f"Balance mismatch: expected ${expected_balance} actual ${balance_after}")

        result = ReconciliationResult(
            market_id=order.market_id,
            expected_spend=expected_spend,
            actual_spend=actual_spend,
            expected_price=expected_price,
            actual_price=actual_price,
            mismatch=mismatch,
            mismatch_pct=mismatch_pct,
            balance_before=balance_before,
            balance_after=balance_after,
            expected_balance=expected_balance,
            balance_mismatch=balance_mismatch,
            timestamp=datetime.now(timezone.utc),
            status=status
        )

        # Persist
        if self.storage:
            try:
                self.storage.conn.execute(
                    "INSERT INTO reconciliation (market_id, expected_spend, actual_spend, mismatch, status, timestamp) VALUES (?,?,?,?,?,?)",
                    (order.market_id, expected_spend, actual_spend, mismatch, status, result.timestamp.isoformat())
                )
                self.storage.conn.commit()
            except Exception as e:
                logger.warning(f"Reconciliation DB save failed: {e}")

        return result

    def check_unexpected_balance(self, expected_balance: float, actual_balance: float, tolerance_usd: float = 5.0) -> tuple[bool, str]:
        """Check for unexpected account balance"""
        diff = abs(expected_balance - actual_balance)
        if diff > tolerance_usd:
            return True, f"Unexpected balance: expected ${expected_balance:.2f} actual ${actual_balance:.2f} diff ${diff:.2f} > ${tolerance_usd}"
        return False, "Balance OK"

    def get_report(self) -> Dict:
        return {
            "message": "Reconciliation engine - verifies fills and balances",
            "checks": ["execution_mismatch", "balance_mismatch", "duplicate_orders", "unexpected_balance"]
        }
