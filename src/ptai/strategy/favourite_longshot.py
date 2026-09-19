"""
Favourite-Longshot Bias - Empirically longshots <5% overpriced and favourites >95% underpriced
Systematically fade the tails, but only in liquid markets

Research: In prediction markets, extreme probabilities are miscalibrated
- Longshots (<5%) are overpriced: market says 3% but true 1% => fade (sell YES / buy NO)
- Favourites (>95%) underpriced: market says 97% but true 99% => buy YES

Only in liquid markets to avoid manipulation
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market


@dataclass
class LongshotBiasOpportunity:
    market: Market
    market_price: float
    is_longshot: bool  # <5%
    is_favourite: bool  # >95%
    fair_estimate: float
    edge: float
    liquidity_ok: bool
    should_trade: bool
    reasoning: str


class FavouriteLongshotEngine:
    """
    Exploits favourite-longshot bias
    """
    def __init__(self, longshot_threshold: float = 0.05, favourite_threshold: float = 0.95,
                 min_liquidity: float = 10000, min_volume: float = 10000):
        self.longshot_threshold = longshot_threshold
        self.favourite_threshold = favourite_threshold
        self.min_liquidity = min_liquidity
        self.min_volume = min_volume

    def evaluate(self, market: Market) -> Optional[LongshotBiasOpportunity]:
        price = market.best_price
        
        is_longshot = price < self.longshot_threshold
        is_favourite = price > self.favourite_threshold
        
        if not (is_longshot or is_favourite):
            return None
        
        # Check liquidity - only trade liquid markets
        liquidity_ok = market.liquidity >= self.min_liquidity and market.volume_24h >= self.min_volume
        if not liquidity_ok:
            return LongshotBiasOpportunity(
                market=market,
                market_price=price,
                is_longshot=is_longshot,
                is_favourite=is_favourite,
                fair_estimate=price,
                edge=0,
                liquidity_ok=False,
                should_trade=False,
                reasoning=f"Extreme price {price:.3f} but liquidity ${market.liquidity} volume ${market.volume_24h} < ${self.min_liquidity} min, avoid thin markets"
            )
        
        # Estimate fair based on bias correction
        # Longshots overpriced: true prob lower than market
        # Favourites underpriced: true prob higher than market
        if is_longshot:
            # Market 3% => fair maybe 1.5% (50% lower)
            # Use formula: fair = price * 0.6 for <5%
            fair_estimate = price * 0.6
            edge = price - fair_estimate  # edge for selling YES (buying NO)
            # Selling YES at 3% when fair 1.5% => profit 1.5% if you sell YES? Actually buying NO at 97% when fair NO 98.5%?
            # Let's calculate edge as price - fair for longshot (overpriced YES, so selling YES has edge)
            should_trade = edge > 0.02  # need 2% edge
            reasoning = (
                f"Longshot bias: market {price*100:.1f}% <5% overpriced, fair est {fair_estimate*100:.1f}% "
                f"edge {edge*100:.1f}% for selling YES / buying NO | "
                f"Empirically longshots overpriced, fade the tail | "
                f"Liquidity OK ${market.liquidity} | Should trade {should_trade}"
            )
        else:
            # Favourite: market 97% => fair 98.5% (underpriced)
            fair_estimate = price + (1 - price) * 0.4  # 97% => 97% + 3%*0.4 = 98.2%
            fair_estimate = min(0.99, fair_estimate)
            edge = fair_estimate - price  # edge for buying YES
            should_trade = edge > 0.02
            reasoning = (
                f"Favourite bias: market {price*100:.1f}% >95% underpriced, fair est {fair_estimate*100:.1f}% "
                f"edge {edge*100:.1f}% for buying YES | "
                f"Empirically favourites underpriced | "
                f"Liquidity OK ${market.liquidity} | Should trade {should_trade}"
            )
        
        return LongshotBiasOpportunity(
            market=market,
            market_price=price,
            is_longshot=is_longshot,
            is_favourite=is_favourite,
            fair_estimate=fair_estimate,
            edge=edge,
            liquidity_ok=liquidity_ok,
            should_trade=should_trade,
            reasoning=reasoning
        )

    def scan_markets(self, markets: List[Market]) -> List[LongshotBiasOpportunity]:
        opps = []
        for m in markets:
            opp = self.evaluate(m)
            if opp:
                opps.append(opp)
                if opp.should_trade:
                    logger.info(f"LONGSHOT BIAS: {opp.reasoning}")
        
        opps.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Longshot bias scan: {len(markets)} markets -> {len(opps)} extreme, {len([o for o in opps if o.should_trade])} tradeable")
        return opps

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Favourite-Longshot Bias",
            "bias": "Empirically longshots <5% overpriced and favourites >95% underpriced",
            "strategy": "Systematically fade the tails, but only in liquid markets",
            "longshot": f"Market <{self.longshot_threshold*100:.0f}% => fair = price*0.6, edge for selling YES",
            "favourite": f"Market >{self.favourite_threshold*100:.0f}% => fair = price + (1-price)*0.4, edge for buying YES",
            "liquidity_filter": f"Only liquid >${self.min_liquidity} liquidity >${self.min_volume} volume to avoid manipulation",
            "research": "In prediction markets extreme probabilities miscalibrated, longshots overpriced 3% true 1%, favourites underpriced 97% true 99%"
        }
