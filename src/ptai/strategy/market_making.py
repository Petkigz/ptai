"""
Market Making Strategy - provide liquidity, capture spread
Part of venue × market × strategy engine
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType
from ..markets.orderbook import read_spread


@dataclass
class MarketMakingSignal:
    market_id: str
    spread_pct: float
    depth: float
    volatility: float
    estimated_profit_per_trade: float
    inventory_risk: float
    reasoning: str
    should_trade: bool


class MarketMakingEngine:
    """
    Market making strategy:
    - High liquidity, tight spread markets
    - Capture bid-ask spread
    - Manage inventory risk
    - Requires good execution quality
    """
    def __init__(self, min_spread: float = 0.01, min_depth: float = 5000):
        self.min_spread = min_spread
        self.min_depth = min_depth

    def evaluate(self, market: Market, orderbook: Dict = None) -> MarketMakingSignal:
        orderbook = orderbook or {}
        spread, _spread_is_real = read_spread(orderbook, 0.02)
        spread_pct = orderbook.get("spread_pct", spread)
        depth = orderbook.get("depth", market.liquidity)
        volatility = orderbook.get("volatility", 0.02)  # default low vol
        
        # Estimate profit per trade: half spread minus fees
        fee = 0.001  # 0.1% typical
        profit_per_trade = spread_pct / 2 - fee
        
        # Inventory risk: if market trending strongly, risk of adverse selection
        # Simple: high volume + low spread = low inventory risk
        inventory_risk = 0.0
        if market.volume_24h > 0 and market.liquidity > 0:
            turnover = market.volume_24h / max(1, market.liquidity)
            if turnover > 5:  # high turnover = high risk of informed trading
                inventory_risk = 0.03
            elif turnover > 2:
                inventory_risk = 0.01
        
        # Adjust profit for inventory risk
        adjusted_profit = profit_per_trade - inventory_risk
        
        should_trade = (
            spread_pct >= self.min_spread and
            depth >= self.min_depth and
            adjusted_profit > 0.005 and  # 0.5% min profit
            market.liquidity > 2000
        )
        
        reasoning = (
            f"Spread {spread_pct:.4f} depth {depth:.0f} vol {volatility:.4f} | "
            f"Profit/trade {profit_per_trade:.4f} inv risk {inventory_risk:.4f} adj {adjusted_profit:.4f} | "
            f"Trade {should_trade}"
        )
        
        return MarketMakingSignal(
            market_id=market.id,
            spread_pct=spread_pct,
            depth=depth,
            volatility=volatility,
            estimated_profit_per_trade=adjusted_profit,
            inventory_risk=inventory_risk,
            reasoning=reasoning,
            should_trade=should_trade
        )

    def to_venue_opportunity(self, market: Market, signal: MarketMakingSignal) -> Optional[VenueOpportunity]:
        if not signal.should_trade:
            return None
        
        # For market making, edge is spread capture, not directional
        # Represent as opportunity with effective edge = profit per trade
        opp = VenueOpportunity(
            market=market,
            venue_id=getattr(market, 'source', 'unknown').value if hasattr(getattr(market, 'source', ''), 'value') else str(getattr(market, 'source', 'unknown')),
            venue_type=VenueType.PREDICTION,
            side="BOTH",  # Market making both sides
            market_price=market.best_price,
            estimated_fair=market.best_price,  # No directional edge
            raw_edge=signal.estimated_profit_per_trade,
            effective_edge=signal.estimated_profit_per_trade,
            confidence=0.7,  # Market making more predictable
            uncertainty=0.1,
            liquidity_score=min(1.0, market.liquidity / 10000),
            execution_quality=0.9,  # Need good execution
            category="market_making",
            sources=["market_making_spread"],
            reasoning=signal.reasoning,
            bull_case=f"Capture spread {signal.spread_pct:.3f} with depth {signal.depth:.0f}",
            bear_case=f"Inventory risk {signal.inventory_risk:.3f}",
            should_trade=signal.should_trade
        )
        opp.calculate_common_score()
        return opp
