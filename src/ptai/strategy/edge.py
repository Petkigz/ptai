"""
Edge Calculator - sophisticated effective edge, not just fair - market >=0.08

raw_edge
    ↓
fees
    ↓
spread
    ↓
slippage
    ↓
liquidity
    ↓
model uncertainty
    ↓
correlation
    ↓
remaining time
    ↓
effective edge
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
from loguru import logger

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity
from ..markets.orderbook import read_spread


@dataclass
class EffectiveEdge:
    raw_edge: float
    fees: float
    spread: float
    slippage: float
    liquidity_penalty: float
    uncertainty_penalty: float
    correlation_penalty: float
    time_penalty: float
    effective_edge: float
    conservative_fair: float
    market_price: float
    reasoning: str
    should_trade: bool = False


class EdgeCalculator:
    def __init__(self, uncertainty_engine=None):
        self.uncertainty_engine = uncertainty_engine

    def calculate(self, market: Market, fair_prob: float, uncertainty: float = 0.1,
                  orderbook: Dict = None, amount_usd: float = 5.0,
                  correlation_penalty: float = 0.0, category_exposure: float = 0.0) -> EffectiveEdge:
        """
        Calculate effective edge with all deductions.
        Example: YES 0.61, fair 0.73, raw 12%, fees 0.8%, slippage 1.2%, uncertainty 2% -> effective much less
        """
        orderbook = orderbook or {}
        market_price = market.best_price
        raw_edge = fair_prob - market_price

        # Fees - Use accurate fee model from markets/fees.py
        # Polymarket formula: Fee = 0.06 × C × p × (1-p), at $0.50 $1.50 per 100 contracts, $3 position fee $0.09 (3%)
        # For $50 bankroll, fee is biggest constraint
        try:
            from ..markets.fees import FeeEngine
            fee_engine = FeeEngine()
            venue_id = getattr(market, 'source', 'polymarket')
            venue_str = venue_id.value if hasattr(venue_id, 'value') else str(venue_id)
            if market.raw.get("venue"):
                venue_str = market.raw["venue"]
            category = "politics" if "politics" in market.question.lower() or "election" in market.question.lower() else "general"
            fee_result = fee_engine.calculate(venue_id=venue_str, amount_usd=amount_usd, price=market_price, category=category)
            fees = fee_result.fee_pct
            # Also include gas for Polymarket
            from ..execution.gas import GasModel
            gas_model = GasModel()
            gas_result = gas_model.calculate_gas(operation="place_order", amount_usd=amount_usd, venue_id=venue_str)
            gas_pct = gas_result.gas_pct_of_position
        except Exception as e:
            # Fallback to old conservative model
            fees = 0.02 if "politics" in market.question.lower() or "election" in market.question.lower() else 0.008
            gas_pct = 0.016  # $0.05 / $3 = 1.6%

        # Spread from orderbook
        spread, _spread_is_real = read_spread(orderbook, 0.02)
        # If market has low liquidity, spread wider
        if market.liquidity < 1000:
            spread = max(spread, 0.04)
        elif market.liquidity < 5000:
            spread = max(spread, 0.025)

        # Slippage based on amount vs liquidity
        slippage = 0.0
        if market.liquidity > 0:
            ratio = amount_usd / market.liquidity
            slippage = min(0.05, ratio * 0.3)
        else:
            slippage = 0.03

        # Liquidity penalty - illiquid markets penalize
        liquidity_penalty = 0.0
        if market.liquidity < 500:
            liquidity_penalty = 0.03
        elif market.liquidity < 2000:
            liquidity_penalty = 0.01

        # Uncertainty penalty - from uncertainty engine
        uncertainty_penalty = uncertainty * 0.5  # 50% of uncertainty as penalty

        # Correlation penalty - if already exposed to correlated markets
        # category_exposure is current exposure to this category (0-1)
        # If already 15% in category, penalize new trades in same category
        if category_exposure > 0.15:
            correlation_penalty = (category_exposure - 0.15) * 0.5

        # Time penalty - very short time left increases risk, very long time increases uncertainty
        time_penalty = 0.0
        if market.end_date:
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                end = market.end_date
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                hours_left = (end - now).total_seconds() / 3600
                if hours_left < 2:
                    time_penalty = 0.02  # very short, risky
                elif hours_left > 24*60:
                    time_penalty = 0.01  # far out, more uncertainty
            except:
                pass

        # Effective edge - include gas for $50 bankroll math
        # From reality check: $3 position $0.05 gas = 1.6% significant
        try:
            gas_deduction = gas_pct
        except:
            gas_deduction = 0.016
        
        total_deductions = fees + gas_deduction + spread + slippage + liquidity_penalty + uncertainty_penalty + correlation_penalty + time_penalty
        effective_edge = raw_edge - total_deductions

        # Conservative fair after uncertainty
        if fair_prob > market_price:
            conservative_fair = fair_prob - uncertainty_penalty
        else:
            conservative_fair = fair_prob + uncertainty_penalty
        conservative_fair = max(0.01, min(0.99, conservative_fair))

        # Should trade? Effective edge >=8% and positive
        should_trade = effective_edge >= 0.08

        reasoning = (
            f"Raw {raw_edge:.3f} (fair {fair_prob:.3f} - mkt {market_price:.3f}) | "
            f"Deductions: fees {fees:.3f} gas {gas_deduction:.3f} spread {spread:.3f} slip {slippage:.3f} liq {liquidity_penalty:.3f} "
            f"unc {uncertainty_penalty:.3f} corr {correlation_penalty:.3f} time {time_penalty:.3f} total {total_deductions:.3f} | "
            f"Effective {effective_edge:.3f} conservative_fair {conservative_fair:.3f} | "
            f"Should trade: {should_trade} | $50 math: fee {fees*100:.1f}% + gas {gas_deduction*100:.1f}% + spread {spread*100:.1f}% = { (fees+gas_deduction+spread)*100:.1f}% cost must exceed to break even"
        )

        logger.info(f"Edge calc {market.id}: {reasoning}")

        return EffectiveEdge(
            raw_edge=raw_edge,
            fees=fees,
            spread=spread,
            slippage=slippage,
            liquidity_penalty=liquidity_penalty,
            uncertainty_penalty=uncertainty_penalty,
            correlation_penalty=correlation_penalty,
            time_penalty=time_penalty,
            effective_edge=effective_edge,
            conservative_fair=conservative_fair,
            market_price=market_price,
            reasoning=reasoning,
            should_trade=should_trade
        )

    def calculate_for_opportunity(self, opportunity: VenueOpportunity, orderbook: Dict = None) -> EffectiveEdge:
        """Calculate for VenueOpportunity"""
        # Create mock market from opportunity
        from ..markets.base import Market, MarketSource
        market = opportunity.market
        return self.calculate(
            market=market,
            fair_prob=opportunity.estimated_fair,
            uncertainty=opportunity.uncertainty,
            orderbook=orderbook,
            amount_usd=5.0,
            correlation_penalty=0.0
        )
