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
    V10 FIX #3: Absolute maximum derived from bankroll, not disconnected
    """
    def __init__(self, bankroll: float = 50.0, absolute_pct_ceiling: float = 0.10, configured_dollar_ceiling: float = 1000.0):
        self.bankroll = bankroll
        self.absolute_pct_ceiling = absolute_pct_ceiling  # 10% absolute max regardless
        self.configured_dollar_ceiling = configured_dollar_ceiling  # $1000 hard ceiling
        self.absolute_max_per_trade_usd = min(configured_dollar_ceiling, bankroll * absolute_pct_ceiling)
        self.absolute_max_price = 0.99
        self.absolute_min_price = 0.01
        self.max_daily_trades = 50
        self.trades_today = 0

    def update_bankroll(self, bankroll: float):
        self.bankroll = bankroll
        # V10 FIX #3: Recalculate absolute max when bankroll changes
        self.absolute_max_per_trade_usd = min(self.configured_dollar_ceiling, bankroll * self.absolute_pct_ceiling)

    @property
    def effective_absolute_max(self) -> float:
        # V10 FIX #3: Explicit derivation from bankroll
        return min(self.configured_dollar_ceiling, self.bankroll * self.absolute_pct_ceiling)

    def validate(self, proposal: Dict = None, risk_approved: Dict = None, **kwargs) -> GuardResult:
        """
        Validate execution request.
        V9 FIX #1 & #2: Hard LIVE/PAPER/MOCK separation + exact routing ABORT
        proposal: from LLM
        risk_approved: from risk engine with max_spend and max_price
        Also supports legacy kwargs: market_id, side, max_price, max_spend
        """
        checks_passed = []
        checks_failed = []

        # Backward compat: if called with kwargs market_id/side/max_price/max_spend
        if proposal is None and kwargs:
            proposal = {
                "market_id": kwargs.get("market_id", "unknown"),
                "side": kwargs.get("side", "YES"),
                "venue_id": kwargs.get("venue_id", "unknown"),
                "data_mode": kwargs.get("data_mode", "live"),
                "is_mock": kwargs.get("is_mock", False)
            }
            risk_approved = {
                "market_id": kwargs.get("market_id", "unknown"),
                "max_price": kwargs.get("max_price", 0),
                "max_spend_usd": kwargs.get("max_spend", kwargs.get("max_spend_usd", kwargs.get("risk_approved_amount", 0))),
                "venue_id": kwargs.get("venue_id", "unknown"),
                "data_mode": kwargs.get("data_mode", "live")
            }
        
        proposal = proposal or {}
        risk_approved = risk_approved or {}

        # V9 FIX #1 + V10 FIX #1: MOCK_DATA must be impossible to reach live execution - check 4 layers
        data_mode = proposal.get("data_mode") or risk_approved.get("data_mode") or "live"
        if hasattr(data_mode, 'value'):
            data_mode = data_mode.value
        data_mode = str(data_mode).lower()
        is_mock = proposal.get("is_mock", False) or risk_approved.get("is_mock", False) or data_mode == "mock"
        
        market_id_check = proposal.get("market_id") or risk_approved.get("market_id") or ""
        data_source_check = proposal.get("data_source") or risk_approved.get("data_source") or ""
        
        if "MOCK" in str(market_id_check).upper():
            is_mock = True
            data_mode = "mock"
        if "mock" in str(data_source_check).lower():
            # data_source contains mock_fallback -> treat as mock
            is_mock = True
            data_mode = "mock"
        # Also check data_mode tiers
        if data_mode in ("mock", "historical_sim"):
            is_mock = True
        
        if is_mock or data_mode in ("mock", "historical_sim"):
            checks_failed.append(f"MOCK_DATA detected market {market_id_check} data_mode={data_mode} data_source={data_source_check} is_mock={is_mock} - MUST NEVER reach live execution")
            return GuardResult(False, f"MOCK_DATA {market_id_check} blocked - cannot reach execution", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"data_mode {data_mode} not mock - LIVE/PAPER OK (source {data_source_check})")

        # Extract risk-approved values - these are the ONLY values that matter
        max_spend = risk_approved.get("max_spend_usd", 0)
        max_price = risk_approved.get("max_price", 0)
        market_id = risk_approved.get("market_id", proposal.get("market_id", "unknown"))

        # Check 1: max_spend must be from risk engine, not LLM
        if max_spend <= 0:
            checks_failed.append(f"max_spend {max_spend} <= 0")
            return GuardResult(False, "max_spend <=0", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_spend ${max_spend} >0")

        # Check 2: absolute max per trade - V10 FIX #3 derived from bankroll
        effective_max = self.effective_absolute_max
        if max_spend > effective_max:
            checks_failed.append(f"max_spend ${max_spend} > effective absolute max ${effective_max:.2f} = min(${self.configured_dollar_ceiling}, ${self.bankroll}*{self.absolute_pct_ceiling*100:.0f}%)")
            return GuardResult(False, f"Exceeds effective absolute max ${effective_max:.2f} (bankroll-derived)", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_spend ${max_spend} <= effective absolute max ${effective_max:.2f} = min(${self.configured_dollar_ceiling}, bankroll*{self.absolute_pct_ceiling*100:.0f}%)")

        # Check 3: price bounds
        if not (self.absolute_min_price <= max_price <= self.absolute_max_price):
            checks_failed.append(f"max_price {max_price} out of bounds [{self.absolute_min_price}, {self.absolute_max_price}]")
            return GuardResult(False, f"Price out of bounds", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"max_price {max_price} within bounds")

        # Check 4: bankroll percentage - V10 FIX #3 now explicit
        pct = max_spend / max(1, self.bankroll)
        if pct > self.absolute_pct_ceiling:
            checks_failed.append(f"Position {pct*100:.1f}% > {self.absolute_pct_ceiling*100:.0f}% absolute max (bankroll-derived)")
            return GuardResult(False, f"Position {pct*100:.1f}% > {self.absolute_pct_ceiling*100:.0f}% absolute", 0, 0, checks_passed, checks_failed)
        checks_passed.append(f"Position {pct*100:.1f}% <= {self.absolute_pct_ceiling*100:.0f}% absolute (bankroll ${self.bankroll} * {self.absolute_pct_ceiling})")

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
