"""
Momentum and Mean Reversion Strategies
Part of venue × market × strategy engine
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import math

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType


@dataclass
class MomentumSignal:
    market_id: str
    strategy: str  # momentum or mean_reversion
    price_velocity: float  # change per hour
    volume_trend: float
    estimated_edge: float
    confidence: float
    reasoning: str
    should_trade: bool


class MomentumEngine:
    """
    Momentum strategy: price trending continues
    Mean reversion: price overextended reverts
    Applies to crypto, stocks, and prediction markets with price history
    """
    def __init__(self):
        self.momentum_threshold = 0.02  # 2% velocity
        self.mean_rev_threshold = 0.05  # 5% overextended

    def evaluate_momentum(self, market: Market, context: Dict = None) -> MomentumSignal:
        context = context or {}
        price_history = context.get("price_history", [])
        orderbook = context.get("orderbook", {})
        
        # Simple momentum from price change
        # If market has raw with change data
        change_pct = 0.0
        if market.raw.get("change_pct"):
            change_pct = market.raw["change_pct"] / 100.0
        elif market.raw.get("real"):
            # Try to get from raw
            change_pct = market.raw.get("change_pct", 0) / 100.0
        
        # Price velocity: change per hour (simplified)
        velocity = change_pct / 24.0  # per hour if 24h change
        
        # Volume trend: is volume increasing?
        volume_trend = 0.0
        if market.volume_24h > 0 and market.volume > 0:
            volume_trend = (market.volume_24h / max(1, market.volume * 0.3) - 1)  # vs avg
        
        # Momentum edge: if velocity > threshold and volume confirms, predict continuation
        edge = 0.0
        confidence = 0.5
        
        if abs(velocity) > 0.001:  # 0.1% per hour
            # Momentum: recent up predicts further up, but with decay
            # Simple model: next 24h will have 30% of recent move
            edge = velocity * 24 * 0.3
            confidence = 0.6 if abs(velocity) > 0.002 else 0.55
            if volume_trend > 0.2:  # volume confirms
                confidence += 0.1
                edge *= 1.2
        
        should_trade = abs(edge) > 0.06 and confidence > 0.6
        
        reasoning = (
            f"Momentum: velocity {velocity:.4f}/h change {change_pct*100:.1f}% vol_trend {volume_trend:.2f} | "
            f"Edge {edge:.3f} conf {confidence:.2f} | Trade {should_trade}"
        )
        
        return MomentumSignal(
            market_id=market.id,
            strategy="momentum",
            price_velocity=velocity,
            volume_trend=volume_trend,
            estimated_edge=edge,
            confidence=confidence,
            reasoning=reasoning,
            should_trade=should_trade
        )

    def evaluate_mean_reversion(self, market: Market, context: Dict = None) -> MomentumSignal:
        context = context or {}
        
        # Mean reversion: if price overextended from historical mean, expect revert
        # For prediction markets, extreme prices 0.95, 0.05 often overextended
        # For crypto/stocks, RSI-like
        
        price = market.best_price
        # For binary markets, 0.5 is mean, extremes are 0.05, 0.95
        # Distance from 0.5
        distance_from_mean = abs(price - 0.5)
        
        edge = 0.0
        confidence = 0.5
        
        if distance_from_mean > 0.35:  # price >0.85 or <0.15
            # Overextended, expect reversion toward 0.5
            # Edge = reversion amount
            if price > 0.5:
                edge = -0.05  # expect down
            else:
                edge = 0.05  # expect up
            confidence = 0.55 + distance_from_mean * 0.2  # more extreme = higher conf revert
        
        should_trade = abs(edge) > 0.04 and confidence > 0.6 and market.liquidity > 1000
        
        reasoning = (
            f"MeanRev: price {price:.3f} dist_mean {distance_from_mean:.3f} | "
            f"Edge {edge:.3f} conf {confidence:.2f} | Trade {should_trade}"
        )
        
        return MomentumSignal(
            market_id=market.id,
            strategy="mean_reversion",
            price_velocity=0.0,
            volume_trend=0.0,
            estimated_edge=edge,
            confidence=confidence,
            reasoning=reasoning,
            should_trade=should_trade
        )

    def evaluate(self, market: Market, context: Dict = None) -> List[MomentumSignal]:
        """Evaluate both momentum and mean reversion, return best"""
        mom = self.evaluate_momentum(market, context)
        rev = self.evaluate_mean_reversion(market, context)
        return [mom, rev]

    def to_venue_opportunities(self, market: Market, signals: List[MomentumSignal]) -> List[VenueOpportunity]:
        opps = []
        for signal in signals:
            if not signal.should_trade:
                continue
            
            fair = market.best_price + signal.estimated_edge
            fair = max(0.01, min(0.99, fair))
            
            opp = VenueOpportunity(
                market=market,
                venue_id=getattr(market, 'source', 'unknown').value if hasattr(getattr(market, 'source', ''), 'value') else str(getattr(market, 'source', 'unknown')),
                venue_type=VenueType.FINANCIAL,
                side="YES" if signal.estimated_edge > 0 else "NO",
                market_price=market.best_price,
                estimated_fair=fair,
                raw_edge=signal.estimated_edge,
                effective_edge=signal.estimated_edge * signal.confidence,
                confidence=signal.confidence,
                uncertainty=1-signal.confidence,
                liquidity_score=min(1.0, market.liquidity / 10000),
                execution_quality=0.7,
                category=signal.strategy,
                sources=[f"{signal.strategy}_signal"],
                reasoning=signal.reasoning,
                bull_case=f"{signal.strategy} bullish: velocity {signal.price_velocity:.4f}" if signal.estimated_edge > 0 else "",
                bear_case=f"{signal.strategy} bearish: velocity {signal.price_velocity:.4f}" if signal.estimated_edge < 0 else "",
                should_trade=signal.should_trade
            )
            opp.calculate_common_score()
            opps.append(opp)
        
        return opps
