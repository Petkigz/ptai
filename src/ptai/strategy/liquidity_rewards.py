"""
Market Making / Liquidity Rewards - If Polymarket has zero maker fees or liquidity rewards, place two-sided quotes
Earn spread and rewards, use strict inventory limits

For $50 bankroll, arbitrage and liquidity rewards more realistic than directional bets
Directional bets need real edge, arbitrage just needs speed and execution
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market


@dataclass
class LiquidityRewardEstimate:
    market_id: str
    reward_per_day_usd: float
    reward_apr: float
    maker_fee: float
    spread_capture_per_trade: float
    inventory_limit: float
    should_provide_liquidity: bool
    reasoning: str


@dataclass
class MarketMakingQuote:
    market_id: str
    bid_price: float
    ask_price: float
    bid_size_usd: float
    ask_size_usd: float
    spread: float
    mid_price: float
    inventory: float
    inventory_limit: float
    reward_estimate: float
    should_quote: bool
    reasoning: str


class LiquidityRewardsEngine:
    """
    Market making with liquidity rewards
    """
    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll
        # Polymarket liquidity rewards - mock values, real would fetch from API
        # https://docs.polymarket.com/polyevents/liquidity-rewards
        self.mock_rewards = {
            "active": True,
            "reward_rate_per_day": 0.001,  # 0.1% per day on liquidity provided
            "maker_fee": 0.0,  # zero maker fees
            "min_spread": 0.02,  # need to quote within 2% to earn rewards
            "min_size": 10.0,  # min $10 size to earn rewards
        }

    def estimate_rewards(self, market: Market, amount_usd: float) -> LiquidityRewardEstimate:
        """
        Estimate liquidity rewards for providing liquidity
        """
        if not self.mock_rewards["active"]:
            return LiquidityRewardEstimate(
                market_id=market.id,
                reward_per_day_usd=0,
                reward_apr=0,
                maker_fee=0.02,
                spread_capture_per_trade=0,
                inventory_limit=0,
                should_provide_liquidity=False,
                reasoning="Liquidity rewards inactive"
            )
        
        # Reward per day = amount * reward_rate
        reward_per_day = amount_usd * self.mock_rewards["reward_rate_per_day"]
        reward_apr = self.mock_rewards["reward_rate_per_day"] * 365
        
        maker_fee = self.mock_rewards["maker_fee"]
        
        # Spread capture: if spread 2%, capture half =1% per round trip minus fees
        spread = 0.02  # assume 2% spread
        spread_capture = spread / 2 - maker_fee
        
        # Inventory limit: strict, max 10% bankroll per market for market making
        inventory_limit = self.bankroll * 0.10  # $5 on $50
        
        should_provide = (
            market.liquidity >= 5000 and
            market.volume_24h >= 10000 and
            amount_usd <= inventory_limit and
            reward_apr > 0.1  # >10% APR worthwhile
        )
        
        reasoning = (
            f"Liquidity rewards for {market.id}: amount ${amount_usd} reward {reward_per_day:.4f}/day APR {reward_apr*100:.1f}% | "
            f"Maker fee {maker_fee*100:.2f}% spread capture {spread_capture*100:.2f}% per trade | "
            f"Inventory limit ${inventory_limit} (10% bankroll) | "
            f"Should provide {should_provide} (liquid? {market.liquidity>=5000} vol? {market.volume_24h>=10000} APR>10%? {reward_apr>0.1})"
        )
        
        return LiquidityRewardEstimate(
            market_id=market.id,
            reward_per_day_usd=reward_per_day,
            reward_apr=reward_apr,
            maker_fee=maker_fee,
            spread_capture_per_trade=spread_capture,
            inventory_limit=inventory_limit,
            should_provide_liquidity=should_provide,
            reasoning=reasoning
        )

    def create_quotes(self, market: Market, inventory: float = 0,
                     amount_usd: float = 5.0) -> MarketMakingQuote:
        """
        Create two-sided quotes for market making
        """
        mid_price = market.best_price
        spread = 0.02  # target 2% spread
        
        # Adjust quotes based on inventory
        # If long inventory (positive), skew quotes down to sell, reduce bid size
        # If short inventory (negative), skew up to buy
        inventory_skew = inventory / 10.0 * 0.01  # $10 inventory => 1% skew
        
        bid_price = mid_price - spread/2 - inventory_skew
        ask_price = mid_price + spread/2 - inventory_skew
        
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))
        
        # Size based on inventory: if long, smaller bid larger ask
        if inventory > 0:
            bid_size = amount_usd * max(0.3, 1 - inventory/10)
            ask_size = amount_usd * min(1.7, 1 + inventory/10)
        elif inventory < 0:
            bid_size = amount_usd * min(1.7, 1 + abs(inventory)/10)
            ask_size = amount_usd * max(0.3, 1 - abs(inventory)/10)
        else:
            bid_size = amount_usd
            ask_size = amount_usd
        
        inventory_limit = self.bankroll * 0.10
        
        # Should quote?
        reward_est = self.estimate_rewards(market, amount_usd)
        should_quote = (
            abs(inventory) < inventory_limit and
            market.liquidity >= 5000
        )
        
        reasoning = (
            f"Market making quotes for {market.id}: mid {mid_price:.3f} spread {spread*100:.1f}% | "
            f"Bid {bid_price:.3f} size ${bid_size:.2f} Ask {ask_price:.3f} size ${ask_size:.2f} | "
            f"Inventory {inventory:.2f} limit {inventory_limit} skew {inventory_skew*100:.2f}% | "
            f"Reward APR {reward_est.reward_apr*100:.1f}% spread capture {reward_est.spread_capture_per_trade*100:.2f}% | "
            f"Should quote {should_quote}"
        )
        
        return MarketMakingQuote(
            market_id=market.id,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size_usd=bid_size,
            ask_size_usd=ask_size,
            spread=spread,
            mid_price=mid_price,
            inventory=inventory,
            inventory_limit=inventory_limit,
            reward_estimate=reward_est.reward_per_day_usd,
            should_quote=should_quote,
            reasoning=reasoning
        )

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Market Making / Liquidity Rewards",
            "importance": "For $50 bankroll, arbitrage and liquidity rewards more realistic than directional bets - Top 5",
            "rewards": "Polymarket zero maker fees or liquidity rewards, place two-sided quotes, earn spread and rewards",
            "inventory": "Strict inventory limits 10% bankroll per market $5 on $50, skew quotes based on inventory",
            "strategy": "If spread 2%, capture half 1% per round trip minus fees plus rewards APR, more realistic than directional",
            "mock": f"Mock rewards active {self.mock_rewards['active']} rate {self.mock_rewards['reward_rate_per_day']*100:.2f}%/day APR {self.mock_rewards['reward_rate_per_day']*365*100:.1f}%"
        }
