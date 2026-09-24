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
    # Where the spread came from. "orderbook" means it was read from a real
    # bid/ask; "assumed_default" means no book was available and a placeholder
    # was used. Without this the two are indistinguishable - every consumer
    # defaulted to a bare 0.02 and priced it as if the book had been read.
    spread_source: str = "assumed_default"
    is_assumed: bool = True
    assumed_fields: List[str] = None

    def __post_init__(self):
        if self.assumed_fields is None:
            self.assumed_fields = []


class OrderbookAnalyzer:
    def __init__(self):
        pass

    def analyze(self, market, raw_orderbook: Dict = None, recent_trades: List[Dict] = None) -> OrderbookSnapshot:
        raw_orderbook = raw_orderbook or {}
        recent_trades = recent_trades or []

        # Distinguish a real two-sided book from a placeholder.
        #
        # This used to read `bid = raw.get("bid", best_price - 0.01)` and the
        # same for ask, then compute `spread = ask - bid`. Because the defaults
        # always supplied both sides, that branch ALWAYS produced exactly 0.02 -
        # so the spread was a constant whenever no book was passed, and no
        # caller could tell the difference.
        assumed: List[str] = []
        raw_bid = raw_orderbook.get("bid")
        raw_ask = raw_orderbook.get("ask")

        if raw_bid and raw_ask and raw_ask > raw_bid:
            bid, ask = float(raw_bid), float(raw_ask)
            spread = ask - bid
            spread_source = "orderbook"
        elif raw_orderbook.get("spread") is not None:
            spread = float(raw_orderbook["spread"])
            bid = float(raw_bid) if raw_bid else market.best_price
            ask = float(raw_ask) if raw_ask else market.best_price
            spread_source = "reported_spread"
            assumed.append("bid/ask")
        else:
            # No usable book. Keep a numeric spread so callers doing arithmetic
            # do not crash, but label it, and derive the sides from the mid so
            # the numbers at least stay self-consistent.
            spread = 0.02
            bid = market.best_price - spread / 2
            ask = market.best_price + spread / 2
            spread_source = "assumed_default"
            assumed.extend(["bid", "ask", "spread"])

        mid_price = (bid + ask) / 2 if (bid and ask) else market.best_price
        spread_pct = spread / max(0.01, mid_price) if mid_price else 0.02

        bid_size = raw_orderbook.get("bid_size")
        if bid_size is None:
            bid_size = market.liquidity * 0.1
            assumed.append("bid_size")
        ask_size = raw_orderbook.get("ask_size")
        if ask_size is None:
            ask_size = market.liquidity * 0.1
            assumed.append("ask_size")
        depth = raw_orderbook.get("depth")
        if depth is None:
            depth = market.liquidity
            assumed.append("depth")

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
            liquidity=market.liquidity,
            spread_source=spread_source,
            is_assumed=spread_source == "assumed_default",
            assumed_fields=sorted(set(assumed)),
        )

        if snapshot.is_assumed:
            logger.debug(
                f"Orderbook {market.id}: NO REAL BOOK - spread {spread:.3f} is a "
                f"placeholder, assumed fields {snapshot.assumed_fields}")
        else:
            logger.debug(f"Orderbook {market.id}: spread {spread:.3f} from {spread_source} "
                         f"imbalance {imbalance:.2f} velocity {price_velocity:.3f} "
                         f"large_orders {len(large_orders)}")

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

def read_spread(orderbook: Optional[Dict] = None, default: float = 0.02) -> tuple:
    """
    Read a spread from an orderbook dict, distinguishing absent from measured.

    `orderbook.get("spread", default)` returns None when the key EXISTS with a
    None value, so a caller that honestly reports "no spread measured" breaks
    every consumer that used the default-argument idiom. Conversely the idiom
    silently substitutes a default when the key is merely missing, which is how
    a fabricated 2% spread travelled through the risk stack unnoticed.

    Returns (spread, is_real) where is_real is True only when the dict carried a
    usable spread AND did not declare itself assumed.
    """
    book = orderbook or {}
    raw = book.get("spread")
    if raw is None:
        return float(default), False
    try:
        spread = float(raw)
    except (TypeError, ValueError):
        return float(default), False
    if spread < 0:
        return float(default), False
    # An explicit provenance marker wins: a book that says is_real False is not
    # measured, whatever number it carries.
    if book.get("is_real") is False or book.get("spread_source") == "assumed_default":
        return spread, False
    return spread, True
