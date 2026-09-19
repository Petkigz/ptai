"""
Execution Guard - final safety layer before real money
LLM proposes, Risk allows $2.71, Execution cannot exceed $2.71 or price 0.615 even if LLM insane
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GuardResult:
    allowed: bool
    reason: str
    max_spend_usd: float
    max_price: float
    checks_passed: list
    checks_failed: list


class ExecutionGuard:
    """
    Execution guard - deterministic, no LLM override possible.
    """
    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll
        self.absolute_max_per_trade_usd = 1000.0
        self.absolute_max_price = 0.99
        self.absolute_min_price = 0.01
        self.max_daily_trades = 50
        self.trades_today = 0

    def update_bankroll(self, bankroll: float):
        self.bankroll = bankroll

    def validate(self, proposal: Dict, risk_approved: Dict) -> GuardResult:
        """
        Validate execution request.
        proposal: from LLM
        risk_approved: from risk engine with max_spend and max_price
        """
        checks_passed = []
        checks_failed = []

        # Extract risk-approved values - these are the ONLY values that matter
        # LLM proposal is ignored for amounts
        max_spend = risk_approved.get("max_spend_usd", 0)
        max_price = risk_approved.get("max_price", 0)
        market_id = risk_approved.get("market_id", proposal.get("market_id", "unknown"))

        # Check 1: max_spend must be from risk engine, not LLM
        if max_spend <= 0:
            checks_failed.append(f"max_spend {max_spend} <= 0")
            return GuardResult(False, "max_spend <=0", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_spend ${max_spend} >0")

        # Check 2: absolute max per trade
        if max_spend > self.absolute_max_per_trade_usd:
            checks_failed.append(f"max_spend ${max_spend} > absolute max ${self.absolute_max_per_trade_usd}")
            return GuardResult(False, f"Exceeds absolute max ${self.absolute_max_per_trade_usd}", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_spend ${max_spend} <= absolute max ${self.absolute_max_per_trade_usd}")

        # Check 3: price bounds
        if not (self.absolute_min_price <= max_price <= self.absolute_max_price):
            checks_failed.append(f"max_price {max_price} out of bounds [{self.absolute_min_price}, {self.absolute_max_price}]")
            return GuardResult(False, f"Price out of bounds", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_price {max_price} within bounds")

        # Check 4: bankroll percentage - should already be enforced by risk, but double-check
        pct = max_spend / max(1, self.bankroll)
        if pct > 0.10:  # absolute 10% even if risk says 6%, extra safety
            checks_failed.append(f"Position {pct*100:.1f}% > 10% absolute max")
            return GuardResult(False, f"Position {pct*100:.1f}% > 10% absolute", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"Position {pct*100:.1f}% <= 10% absolute")

        # Check 5: daily trades limit
        if self.trades_today >= self.max_daily_trades:
            checks_failed.append(f"Daily trades {self.trades_today} >= {self.max_daily_trades}")
            return GuardResult(False, f"Daily trades limit {self.max_daily_trades}", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"Daily trades {self.trades_today} < {self.max_daily_trades}")

        # Check 6: market_id must exist
        if not market_id or market_id == "unknown":
            checks_failed.append("market_id missing")
            return GuardResult(False, "market_id missing", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"market_id {market_id} present")

        # All checks passed
        reason = f"Guard passed: spend ${max_spend} ({pct*100:.1f}% of ${self.bankroll}) price {max_price} market {market_id} | Checks: {checks_passed}"
        logger.info(reason)

        return GuardResult(
            allowed=True,
            reason=reason,
            max_spend_usd=max_spend,
            max_price=max_price,
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )

    def record_trade(self):
        self.trades_today += 1

    def reset_daily(self):
        self.trades_today = 0

    def get_status(self) -> Dict:
        return {
            "bankroll": self.bankroll,
            "absolute_max_per_trade_usd": self.absolute_max_per_trade_usd,
            "absolute_max_price": self.absolute_max_price,
            "absolute_min_price": self.absolute_min_price,
            "max_daily_trades": self.max_daily_trades,
            "trades_today": self.trades_today,
            "can_trade": self.trades_today < self.max_daily_trades
        }
