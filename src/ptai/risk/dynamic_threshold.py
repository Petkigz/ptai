"""
Dynamic Mispricing Threshold - 8% is start but make it fees+slippage+uncertainty+liquidity premium
Illiquid markets need 15%+

"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market


@dataclass
class DynamicThreshold:
    market_id: str
    base_threshold: float  # 8%
    fee_component: float
    slippage_component: float
    uncertainty_component: float
    liquidity_premium: float
    volatility_premium: float
    time_premium: float
    total_threshold: float
    market_price: float
    liquidity: float
    reasoning: str
    should_trade: bool


class DynamicThresholdEngine:
    """
    Dynamic threshold based on market conditions
    """
    def __init__(self, base_threshold: float = 0.08):
        self.base_threshold = base_threshold

    def calculate(self, market: Market, fees: float = 0.02, slippage: float = 0.01,
                 uncertainty: float = 0.1, orderbook: Dict = None) -> DynamicThreshold:
        orderbook = orderbook or {}
        
        # Fee component: already includes gas
        fee_component = fees
        
        # Slippage component: amount/liquidity
        slippage_component = slippage
        
        # Uncertainty component: 50% of uncertainty
        uncertainty_component = uncertainty * 0.5
        
        # Liquidity premium: illiquid markets need higher threshold
        liquidity = market.liquidity
        if liquidity < 1000:
            liquidity_premium = 0.07  # +7% for very illiquid
        elif liquidity < 5000:
            liquidity_premium = 0.04  # +4%
        elif liquidity < 10000:
            liquidity_premium = 0.02  # +2%
        else:
            liquidity_premium = 0.0
        
        # Volatility premium: high volatility needs higher threshold
        volatility = orderbook.get("volatility", 0.02)
        if volatility > 0.05:
            volatility_premium = 0.03
        elif volatility > 0.03:
            volatility_premium = 0.01
        else:
            volatility_premium = 0.0
        
        # Time premium: very short or very long time to resolution needs higher threshold
        time_premium = 0.0
        if market.end_date:
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                end = market.end_date
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                hours_left = (end - now).total_seconds() / 3600
                if hours_left < 24:
                    time_premium = 0.03  # <24h risky
                elif hours_left < 72:
                    time_premium = 0.01
                elif hours_left > 24*90:  # >90 days
                    time_premium = 0.02  # far out uncertain
            except:
                pass
        
        total_threshold = (
            self.base_threshold +
            fee_component +
            slippage_component +
            uncertainty_component +
            liquidity_premium +
            volatility_premium +
            time_premium
        )
        
        # Cap total threshold at 25% to avoid impossible
        total_threshold = min(0.25, total_threshold)
        
        reasoning = (
            f"Dynamic threshold for {market.id}: base {self.base_threshold*100:.1f}% + "
            f"fees {fee_component*100:.1f}% + slippage {slippage_component*100:.1f}% + "
            f"uncertainty {uncertainty_component*100:.1f}% + "
            f"liquidity premium {liquidity_premium*100:.1f}% (liq ${liquidity}) + "
            f"volatility {volatility_premium*100:.1f}% (vol {volatility*100:.1f}%) + "
            f"time {time_premium*100:.1f}% = total {total_threshold*100:.1f}% | "
            f"Illiquid markets need 15%+ vs liquid 8%"
        )
        
        return DynamicThreshold(
            market_id=market.id,
            base_threshold=self.base_threshold,
            fee_component=fee_component,
            slippage_component=slippage_component,
            uncertainty_component=uncertainty_component,
            liquidity_premium=liquidity_premium,
            volatility_premium=volatility_premium,
            time_premium=time_premium,
            total_threshold=total_threshold,
            market_price=market.best_price,
            liquidity=liquidity,
            reasoning=reasoning,
            should_trade=False  # will be set by caller comparing edge vs threshold
        )

    def should_trade(self, edge: float, threshold: DynamicThreshold) -> tuple[bool, str]:
        if edge >= threshold.total_threshold:
            return True, f"Edge {edge*100:.1f}% >= threshold {threshold.total_threshold*100:.1f}% => TRADE"
        else:
            return False, f"Edge {edge*100:.1f}% < threshold {threshold.total_threshold*100:.1f}% => NO TRADE"

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Dynamic Mispricing Threshold",
            "base": f"{self.base_threshold*100:.0f}% is start",
            "formula": "threshold = base 8% + fees + slippage + uncertainty*0.5 + liquidity_premium + volatility_premium + time_premium",
            "liquidity_premium": "<$1k +7%, <$5k +4%, <$10k +2%, >$10k +0% => illiquid needs 15%+",
            "importance": "Prevents trading illiquid markets with small edge that gets eaten by costs"
        }
