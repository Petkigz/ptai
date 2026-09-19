"""
Orderbook - market microstructure analysis
Looks at bid/ask, spread, orderbook depth, recent trades, price velocity, volume, liquidity, large orders
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class OrderbookSnapshot:
    market_id: str
    bid: float
    ask: float
    spread: float
    spread_pct: float
    bid_size: float
    ask_size: float
    depth: float
    mid_price: float
    imbalance: float  # bid_size vs ask_size, -1 to 1, positive = more bid pressure
    large_orders: List[Dict]
    recent_trades: List[Dict]
    price_velocity: float  # change per minute
    volume_24h: float
    liquidity: float


class OrderbookAnalyzer:
    def __init__(self):
        pass

    def analyze(self, market, raw_orderbook: Dict = None, recent_trades: List[Dict] = None) -> OrderbookSnapshot:
        raw_orderbook = raw_orderbook or {}
        recent_trades = recent_trades or []

        bid = raw_orderbook.get("bid", market.best_price - 0.01)
        ask = raw_orderbook.get("ask", market.best_price + 0.01)
        spread = ask - bid if bid and ask else raw_orderbook.get("spread", 0.02)
        spread_pct = spread / max(0.01, (bid + ask) / 2) if bid and ask else 0.02

        bid_size = raw_orderbook.get("bid_size", market.liquidity * 0.1)
        ask_size = raw_orderbook.get("ask_size", market.liquidity * 0.1)
        depth = raw_orderbook.get("depth", market.liquidity)

        mid_price = (bid + ask) / 2 if bid and ask else market.best_price

        # Imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        total_size = bid_size + ask_size
        imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0

        # Large orders detection
        large_orders = []
        if recent_trades:
            avg_size = sum(t.get("size", 0) for t in recent_trades) / len(recent_trades) if recent_trades else 0
            for t in recent_trades:
                if t.get("size", 0) > avg_size * 3 and avg_size > 0:
                    large_orders.append(t)

        # Price velocity
        price_velocity = 0.0
        if len(recent_trades) >= 2:
            try:
                prices = [t.get("price", market.best_price) for t in recent_trades[-10:]]
                if len(prices) >= 2:
                    price_velocity = prices[-1] - prices[0]
            except:
                pass

        snapshot = OrderbookSnapshot(
            market_id=market.id,
            bid=bid,
            ask=ask,
            spread=spread,
            spread_pct=spread_pct,
            bid_size=bid_size,
            ask_size=ask_size,
            depth=depth,
            mid_price=mid_price,
            imbalance=imbalance,
            large_orders=large_orders,
            recent_trades=recent_trades[-20:],
            price_velocity=price_velocity,
            volume_24h=market.volume_24h,
            liquidity=market.liquidity
        )

        logger.debug(f"Orderbook {market.id}: spread {spread:.3f} imbalance {imbalance:.2f} velocity {price_velocity:.3f} large_orders {len(large_orders)}")

        return snapshot

    def is_liquid(self, snapshot: OrderbookSnapshot, min_liquidity: float = 500, max_spread: float = 0.08) -> tuple[bool, str]:
        if snapshot.liquidity < min_liquidity:
            return False, f"Liquidity {snapshot.liquidity} < {min_liquidity}"
        if snapshot.spread > max_spread:
            return False, f"Spread {snapshot.spread:.3f} > {max_spread}"
        if snapshot.depth < min_liquidity * 0.5:
            return False, f"Depth {snapshot.depth} < {min_liquidity*0.5}"
        return True, "Liquid"

    def detect_manipulation(self, snapshot: OrderbookSnapshot) -> List[str]:
        """Detect spoofing, wash trading, etc - must prohibit manipulation"""
        risks = []
        if len(snapshot.large_orders) >= 3:
            risks.append(f"Multiple large orders {len(snapshot.large_orders)} - possible manipulation")
        if abs(snapshot.imbalance) > 0.8:
            risks.append(f"Extreme imbalance {snapshot.imbalance:.2f} - possible spoofing")
        if abs(snapshot.price_velocity) > 0.2:
            risks.append(f"High velocity {snapshot.price_velocity:.3f} - rapid move")
        return risks
