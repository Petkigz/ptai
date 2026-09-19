"""
Limits Engine - deterministic limits, LLM never directly controls money
"""
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger


@dataclass
class TradeLimits:
    max_position_pct: float = 0.06  # 6% hard cap
    max_position_usd: float = 1000.0  # absolute max
    min_edge_pct: float = 0.08  # 8%
    min_confidence: float = 0.60
    max_open_positions: int = 6
    max_category_pct: float = 0.15
    max_correlated_pct: float = 0.20
    max_total_exposure_pct: float = 0.50
    kelly_fraction: float = 0.5  # Half-Kelly
    min_liquidity: float = 500
    max_spread: float = 0.08
    daily_loss_limit_pct: float = 0.15
    total_drawdown_limit_pct: float = 0.30


class LimitsEngine:
    """
    Deterministic limits - receives proposal from LLM, decides if allowed.
    LLM proposes: {market, side, fair_prob, market_prob, edge, confidence, reasoning, trade: true}
    Python decides if actually allowed.
    """
    def __init__(self, limits: TradeLimits = None, bankroll: float = 50.0):
        self.limits = limits or TradeLimits()
        self.bankroll = bankroll

    def update_bankroll(self, bankroll: float):
        self.bankroll = bankroll

    def validate_proposal(self, proposal: Dict) -> tuple[bool, str, Dict]:
        """
        Validate LLM proposal deterministically.
        Returns (allowed, reason, adjusted_order)
        """
        # Extract proposal
        fair_prob = proposal.get("fair_probability", 0.5)
        market_prob = proposal.get("market_probability", 0.5)
        edge = proposal.get("edge", fair_prob - market_prob)
        confidence = proposal.get("confidence", 0.5)
        trade_flag = proposal.get("trade", False)

        # If LLM says don't trade, respect it
        if not trade_flag:
            return False, "LLM says no trade", {}

        # Check edge
        if abs(edge) < self.limits.min_edge_pct:
            return False, f"Edge {edge:.3f} < min {self.limits.min_edge_pct}", {}

        # Check confidence
        if confidence < self.limits.min_confidence:
            return False, f"Confidence {confidence:.3f} < min {self.limits.min_confidence}", {}

        # Check fair prob bounds
        if not (0.01 <= fair_prob <= 0.99):
            return False, f"Fair prob {fair_prob} out of bounds", {}

        # Calculate Kelly sizing
        from .kelly import KellyCalculator
        kelly = KellyCalculator(
            kelly_fraction=self.limits.kelly_fraction,
            max_pct=self.limits.max_position_pct,
            min_edge=self.limits.min_edge_pct
        )
        kelly_result = kelly.calculate(
            market_price=market_prob,
            fair_prob=fair_prob,
            bankroll=self.bankroll
        )

        if not kelly_result.should_bet:
            return False, f"Kelly says no bet: {kelly_result.reason}", {}

        # Enforce hard caps
        position_usd = kelly_result.position_size_usd
        position_pct = kelly_result.position_size_pct

        # Hard cap 6%
        max_usd_by_pct = self.bankroll * self.limits.max_position_pct
        if position_usd > max_usd_by_pct:
            logger.info(f"Kelly ${position_usd:.2f} capped to 6% ${max_usd_by_pct:.2f}")
            position_usd = max_usd_by_pct
            position_pct = self.limits.max_position_pct

        # Absolute max
        if position_usd > self.limits.max_position_usd:
            position_usd = self.limits.max_position_usd

        # Ensure minimum order size
        if position_usd < 1.0:
            return False, f"Position ${position_usd:.2f} < $1 min", {}

        adjusted_order = {
            "market_id": proposal.get("market_id", ""),
            "side": proposal.get("side", "YES"),
            "max_price": proposal.get("max_price", market_prob + 0.02),
            "max_spend_usd": round(position_usd, 2),
            "max_spend_pct": round(position_pct, 4),
            "fair_prob": fair_prob,
            "market_prob": market_prob,
            "edge": edge,
            "kelly_raw": kelly_result.kelly_fraction_raw,
            "kelly_adj": kelly_result.kelly_fraction_adj,
            "reason": kelly_result.reason
        }

        return True, f"Allowed: edge {edge:.3f} conf {confidence:.2f} size ${position_usd:.2f} ({position_pct*100:.1f}%)", adjusted_order

    def get_limits_report(self) -> Dict:
        return {
            "bankroll": self.bankroll,
            "max_position_pct": self.limits.max_position_pct,
            "max_position_usd": self.limits.max_position_usd,
            "min_edge_pct": self.limits.min_edge_pct,
            "min_confidence": self.limits.min_confidence,
            "max_open_positions": self.limits.max_open_positions,
            "max_category_pct": self.limits.max_category_pct,
            "max_correlated_pct": self.limits.max_correlated_pct,
            "max_total_exposure_pct": self.limits.max_total_exposure_pct,
            "kelly_fraction": self.limits.kelly_fraction,
            "daily_loss_limit_pct": self.limits.daily_loss_limit_pct,
            "total_drawdown_limit_pct": self.limits.total_drawdown_limit_pct
        }
