"""
Risk Manager - Enforces all risk rules
- Max 6% per position
- Max open positions
- Daily loss limit
- Total drawdown limit
- Stop loss
- Self-preservation
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from loguru import logger

from .kelly import KellyCalculator, KellyResult
from ..config import get_settings
from ..storage.db import Storage

@dataclass
class RiskCheck:
    allowed: bool
    reason: str
    kelly_result: Optional[KellyResult] = None
    adjusted_size_usd: Optional[float] = None

class RiskManager:
    def __init__(self, storage: Storage, kelly_calculator: Optional[KellyCalculator] = None):
        self.storage = storage
        self.settings = get_settings()
        self.kelly = kelly_calculator or KellyCalculator(
            kelly_fraction=self.settings.kelly_fraction,
            max_pct=self.settings.max_position_pct,
            min_edge=self.settings.min_edge_pct
        )
        logger.info("RiskManager initialized")

    def check_all(self, market_price: float, fair_value: float, market_id: str = "", confidence: float = 0.7) -> RiskCheck:
        """Full risk check pipeline"""
        bankroll = self.storage.get_bankroll()
        perf = self.storage.get_performance_summary()

        # 1. Kelly calculation
        kelly_res = self.kelly.calculate(market_price, fair_value, bankroll)

        if not kelly_res.should_bet:
            return RiskCheck(allowed=False, reason=kelly_res.reason, kelly_result=kelly_res)

        # 2. Max open positions
        open_positions = self.storage.count_open_positions()
        if open_positions >= self.settings.max_open_positions:
            return RiskCheck(
                allowed=False,
                reason=f"Max open positions reached {open_positions}/{self.settings.max_open_positions}",
                kelly_result=kelly_res
            )

        # 3. Daily loss limit
        daily_pnl = sum([h["daily_pnl"] for h in perf["history"][:1]]) if perf["history"] else 0
        # Actually get today's loss
        today_loss_pct = 0
        if perf["history"]:
            # Find today's entries
            today_str = datetime.now(timezone.utc).date().isoformat()
            today_entries = [h for h in perf["history"] if today_str in h["timestamp"]]
            if today_entries:
                today_pnl = sum([e["daily_pnl"] or 0 for e in today_entries])
                today_loss_pct = abs(today_pnl) / bankroll if today_pnl < 0 else 0

        if today_loss_pct >= self.settings.max_daily_loss_pct:
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit hit {today_loss_pct:.1%} >= {self.settings.max_daily_loss_pct:.1%}",
                kelly_result=kelly_res
            )

        # 4. Total drawdown
        total_drawdown_pct = abs(perf["total_pnl_pct"]) / 100 if perf["total_pnl_pct"] < 0 else 0
        if total_drawdown_pct >= self.settings.max_total_drawdown_pct:
            return RiskCheck(
                allowed=False,
                reason=f"Total drawdown {total_drawdown_pct:.1%} >= {self.settings.max_total_drawdown_pct:.1%} - SHUTDOWN",
                kelly_result=kelly_res
            )

        # 5. Bankroll check
        if bankroll < 5.0:
            return RiskCheck(
                allowed=False,
                reason=f"Bankroll too low ${bankroll:.2f} < $5 - stop trading",
                kelly_result=kelly_res
            )

        # 6. Confidence check
        if confidence < 0.55:
            return RiskCheck(
                allowed=False,
                reason=f"Low confidence {confidence:.2f} < 0.55",
                kelly_result=kelly_res
            )

        # 7. Self-preservation check
        sp = self.storage.check_self_preservation(
            daily_cost=self.settings.daily_cost_to_cover,
            max_unprofitable_days=self.settings.shutdown_if_unprofitable_days
        )
        if sp["should_shutdown"]:
            return RiskCheck(
                allowed=False,
                reason=f"Self-preservation SHUTDOWN: {sp['shutdown_reason']}",
                kelly_result=kelly_res
            )

        # 8. Position size sanity
        adjusted_size = kelly_res.position_size_usd
        if adjusted_size > bankroll * self.settings.max_position_pct:
            adjusted_size = bankroll * self.settings.max_position_pct
        if adjusted_size < 1.0:
            return RiskCheck(
                allowed=False,
                reason=f"Position size ${adjusted_size:.2f} < $1 minimum",
                kelly_result=kelly_res
            )

        # All checks passed
        return RiskCheck(
            allowed=True,
            reason=f"OK - Edge {kelly_res.edge:.1%}, Size ${adjusted_size:.2f} ({adjusted_size/bankroll:.1%})",
            kelly_result=kelly_res,
            adjusted_size_usd=adjusted_size
        )

    def get_risk_summary(self) -> Dict:
        bankroll = self.storage.get_bankroll()
        perf = self.storage.get_performance_summary()
        sp = self.storage.check_self_preservation()

        return {
            "bankroll": bankroll,
            "open_positions": self.storage.count_open_positions(),
            "max_open_positions": self.settings.max_open_positions,
            "max_position_pct": self.settings.max_position_pct,
            "max_position_usd": bankroll * self.settings.max_position_pct,
            "daily_loss_limit_pct": self.settings.max_daily_loss_pct,
            "total_drawdown_pct": perf["total_pnl_pct"],
            "self_preservation": sp,
            "performance": perf
        }
