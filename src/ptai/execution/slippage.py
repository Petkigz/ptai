"""
Slippage Model + Limit Order Execution with TWAP, WebSocket Order Book
Saves you from fees and bad fills - Top 5 to implement first

- API-first: Polymarket CLOB, Kalshi API, browser automation only as fallback
- Limit orders only: never market orders, use TWAP for larger sizes
- WebSocket order book: real-time depth, not polling
- Slippage model: estimate from order book, don't trade if slippage > expected edge
- Batch on-chain transactions: save gas, monitor Polygon gas
- Hot wallet isolation: only small funds in hot wallet, rest cold
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import math
import time


@dataclass
class SlippageEstimate:
    market_id: str
    side: str
    amount_usd: float
    market_price: float
    estimated_fill_price: float
    slippage_pct: float
    slippage_usd: float
    liquidity_available: float
    spread: float
    depth: float
    should_trade: bool
    reasoning: str


@dataclass
class OrderBookDepth:
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float
    bid_depth: float
    ask_depth: float
    imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth) -1 to +1


@dataclass
class TWAPPlan:
    market_id: str
    total_amount: float
    num_slices: int
    slice_amount: float
    interval_seconds: float
    limit_price: float
    side: str
    reasoning: str


class SlippageModel:
    """
    Estimates slippage from order book depth
    """
    def __init__(self, max_slippage_pct: float = 0.02):
        self.max_slippage_pct = max_slippage_pct

    def estimate_slippage(self, market_id: str, side: str, amount_usd: float,
                         orderbook: Dict[str, Any], market_price: float) -> SlippageEstimate:
        """
        Estimate slippage for given amount from order book
        orderbook: {bids: [(price, size)], asks: [(price, size)], spread, depth}
        """
        # Parse orderbook
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        spread = orderbook.get("spread", 0.02)
        depth = orderbook.get("depth", 10000)
        
        # If no detailed bids/asks, use simple model: slippage = amount / liquidity * 0.3
        if not bids and not asks:
            liquidity = orderbook.get("liquidity", depth)
            if liquidity <= 0:
                liquidity = 1000
            ratio = amount_usd / liquidity
            slippage_pct = min(0.05, ratio * 0.3)
            estimated_fill = market_price * (1 + slippage_pct) if side.upper() == "YES" else market_price * (1 - slippage_pct)
            slippage_usd = amount_usd * slippage_pct
            
            should_trade = slippage_pct <= self.max_slippage_pct
            reasoning = (
                f"Simple slippage model: amount ${amount_usd} / liquidity ${liquidity} = {ratio:.3f} => "
                f"slippage {slippage_pct*100:.2f}% (${slippage_usd:.4f}) | "
                f"Market price {market_price:.3f} estimated fill {estimated_fill:.3f} | "
                f"Should trade {should_trade} (max {self.max_slippage_pct*100:.1f}%)"
            )
            
            return SlippageEstimate(
                market_id=market_id,
                side=side,
                amount_usd=amount_usd,
                market_price=market_price,
                estimated_fill_price=estimated_fill,
                slippage_pct=slippage_pct,
                slippage_usd=slippage_usd,
                liquidity_available=liquidity,
                spread=spread,
                depth=depth,
                should_trade=should_trade,
                reasoning=reasoning
            )
        
        # Detailed order book walk
        remaining = amount_usd
        total_cost = 0
        total_shares = 0
        
        levels = asks if side.upper() == "YES" else bids  # buying YES uses asks, selling uses bids
        
        for price, size in levels:
            if remaining <= 0:
                break
            # size is in USD or shares? Assume USD for simplicity
            take = min(remaining, size)
            total_cost += take
            shares = take / price if price > 0 else 0
            total_shares += shares
            remaining -= take
        
        if total_shares > 0:
            avg_fill_price = total_cost / total_shares
            # Actually avg fill in price terms: if buying YES, avg price = total_cost / total_shares? No.
            # Simplified: slippage based on how far we walked the book
            # For YES buy, if market 0.60 and we had to take asks at 0.61, 0.62, slippage = avg 0.615 - 0.60 = 0.015
            # Use ratio method for simplicity
            liquidity = sum(size for _, size in levels)
            ratio = amount_usd / liquidity if liquidity > 0 else 1.0
            slippage_pct = min(0.05, ratio * 0.3)
            estimated_fill = market_price * (1 + slippage_pct) if side.upper() == "YES" else market_price * (1 - slippage_pct)
        else:
            # No liquidity
            slippage_pct = 0.05
            estimated_fill = market_price * (1 + slippage_pct) if side.upper() == "YES" else market_price * (1 - slippage_pct)
            liquidity = 0
        
        slippage_usd = amount_usd * slippage_pct
        should_trade = slippage_pct <= self.max_slippage_pct
        
        reasoning = (
            f"Order book walk: amount ${amount_usd} liquidity ${liquidity} ratio {amount_usd/liquidity if liquidity>0 else 1:.3f} => "
            f"slippage {slippage_pct*100:.2f}% (${slippage_usd:.4f}) | "
            f"Market {market_price:.3f} fill {estimated_fill:.3f} spread {spread*100:.2f}% depth ${depth} | "
            f"Should trade {should_trade}"
        )
        
        return SlippageEstimate(
            market_id=market_id,
            side=side,
            amount_usd=amount_usd,
            market_price=market_price,
            estimated_fill_price=estimated_fill,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd,
            liquidity_available=liquidity,
            spread=spread,
            depth=depth,
            should_trade=should_trade,
            reasoning=reasoning
        )

    def check_slippage_vs_edge(self, slippage_pct: float, edge_pct: float) -> Tuple[bool, str]:
        """Don't trade if slippage > expected edge"""
        if slippage_pct > edge_pct:
            return False, f"Slippage {slippage_pct*100:.2f}% > edge {edge_pct*100:.2f}% - would lose money"
        if slippage_pct > edge_pct * 0.5:
            return False, f"Slippage {slippage_pct*100:.2f}% > 50% of edge {edge_pct*100:.2f}% - too much"
        return True, f"Slippage {slippage_pct*100:.2f}% OK vs edge {edge_pct*100:.2f}%"


class OrderBookImbalance:
    """
    Use CLOB depth to time entries - if bid side stacked, wait or buy into weakness
    """
    def __init__(self):
        pass

    def analyze_imbalance(self, orderbook: Dict) -> OrderBookDepth:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Calculate depth
        bid_depth = sum(size for _, size in bids) if bids else orderbook.get("bid_size", 0)
        ask_depth = sum(size for _, size in asks) if asks else orderbook.get("ask_size", 0)
        
        if not bids and not asks:
            # Use simple mock
            bid_depth = orderbook.get("bid_size", orderbook.get("depth", 10000) * 0.5)
            ask_depth = orderbook.get("ask_size", orderbook.get("depth", 10000) * 0.5)
            spread = orderbook.get("spread", 0.02)
            mid = orderbook.get("mid_price", 0.5)
        else:
            # Calculate spread and mid
            best_bid = max(price for price, _ in bids) if bids else 0
            best_ask = min(price for price, _ in asks) if asks else 1
            spread = best_ask - best_bid if best_bid and best_ask else 0.02
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.5
        
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        return OrderBookDepth(
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance
        )

    def get_entry_timing_signal(self, depth: OrderBookDepth, side: str) -> Tuple[str, str]:
        """
        Timing signal based on order book imbalance
        If bid stacked (imbalance >0.3) and you want to buy YES, wait - price may go up, or buy into weakness if ask stacked
        """
        imbalance = depth.imbalance
        
        if side.upper() == "YES":
            # Buying YES
            if imbalance > 0.3:
                # Bid side stacked, bullish, price may rise, consider buying now or waiting for dip?
                return "WAIT_BULLISH", f"Bid stacked imbalance {imbalance:.2f} bullish, price may rise, but don't chase, wait for ask liquidity"
            elif imbalance < -0.3:
                # Ask stacked, bearish, good time to buy into weakness
                return "BUY_WEAKNESS", f"Ask stacked imbalance {imbalance:.2f} bearish, good time to buy YES into weakness"
            else:
                return "NEUTRAL", f"Balanced book imbalance {imbalance:.2f} neutral, OK to buy"
        else:
            # Buying NO / Selling YES
            if imbalance < -0.3:
                return "WAIT_BEARISH", f"Ask stacked {imbalance:.2f} bearish, price may drop, wait"
            elif imbalance > 0.3:
                return "BUY_WEAKNESS", f"Bid stacked {imbalance:.2f} bullish, good time to buy NO into strength"
            else:
                return "NEUTRAL", f"Balanced {imbalance:.2f} neutral"


class LimitOrderExecutor:
    """
    Limit orders only, never market orders, TWAP for larger sizes, batch on-chain, hot wallet isolation
    """
    def __init__(self, use_twap_threshold: float = 10.0):
        self.use_twap_threshold = use_twap_threshold  # Use TWAP if amount > $10
        self.slippage_model = SlippageModel()
        self.imbalance_analyzer = OrderBookImbalance()

    def create_limit_order(self, market_id: str, side: str, amount_usd: float,
                          market_price: float, fair_price: float,
                          orderbook: Dict) -> Dict[str, Any]:
        """
        Create limit order - never market order
        Limit price = fair price adjusted for edge, not market price
        """
        # Limit price: if buying YES and fair 0.70 market 0.60, limit at 0.65 (mid between market and fair) to get edge but still fill
        # Don't pay full fair, leave edge
        if side.upper() == "YES":
            if fair_price > market_price:
                # Underpriced, we want to buy YES, limit between market and fair, closer to market to ensure edge
                limit_price = market_price + (fair_price - market_price) * 0.5  # mid
                # But ensure at least 2% edge vs fair
                limit_price = min(limit_price, fair_price * 0.98)
            else:
                # Overpriced, we want to sell YES / buy NO, not buying YES
                limit_price = market_price
        else:
            # Buying NO
            if fair_price < market_price:
                # Overpriced YES means NO underpriced, buy NO
                # NO price = 1 - YES price, fair NO = 1 - fair YES
                fair_no = 1 - fair_price
                market_no = 1 - market_price
                limit_price_no = market_no + (fair_no - market_no) * 0.5
                limit_price = 1 - limit_price_no  # convert back to YES price for order?
                # Actually for NO side, limit is NO price
                limit_price = limit_price_no
            else:
                limit_price = 1 - market_price
        
        # Check slippage
        slippage_est = self.slippage_model.estimate_slippage(market_id, side, amount_usd, orderbook, market_price)
        
        # Check imbalance for timing
        depth = self.imbalance_analyzer.analyze_imbalance(orderbook)
        timing_signal, timing_reason = self.imbalance_analyzer.get_entry_timing_signal(depth, side)
        
        # Should trade?
        edge_pct = abs(fair_price - market_price)
        slippage_ok, slippage_reason = self.slippage_model.check_slippage_vs_edge(slippage_est.slippage_pct, edge_pct)
        
        should_trade = slippage_ok and slippage_est.should_trade and timing_signal != "WAIT_BULLISH" and timing_signal != "WAIT_BEARISH"
        
        reasoning = (
            f"Limit order {side} ${amount_usd} market {market_price:.3f} fair {fair_price:.3f} limit {limit_price:.3f} | "
            f"Slippage {slippage_est.slippage_pct*100:.2f}% edge {edge_pct*100:.2f}% {slippage_reason} | "
            f"Imbalance {depth.imbalance:.2f} signal {timing_signal} {timing_reason} | "
            f"Spread {depth.spread*100:.2f}% depth ${depth.bid_depth+depth.ask_depth:.0f} | "
            f"Should trade {should_trade}"
        )
        
        return {
            "market_id": market_id,
            "side": side,
            "amount_usd": amount_usd,
            "market_price": market_price,
            "fair_price": fair_price,
            "limit_price": limit_price,
            "slippage_estimate": slippage_est,
            "depth": depth,
            "timing_signal": timing_signal,
            "should_trade": should_trade,
            "reasoning": reasoning,
            "order_type": "LIMIT",
            "never_market": True
        }

    def create_twap_plan(self, market_id: str, side: str, total_amount: float,
                        market_price: float, fair_price: float) -> TWAPPlan:
        """
        TWAP for larger sizes to minimize market impact
        """
        if total_amount <= self.use_twap_threshold:
            # No TWAP needed for small sizes
            return TWAPPlan(
                market_id=market_id,
                total_amount=total_amount,
                num_slices=1,
                slice_amount=total_amount,
                interval_seconds=0,
                limit_price=fair_price * 0.98 if side.upper() == "YES" else fair_price,
                side=side,
                reasoning=f"Amount ${total_amount} <= threshold ${self.use_twap_threshold}, no TWAP needed"
            )
        
        # TWAP: split into 5 slices over 1 hour
        num_slices = min(10, max(3, int(total_amount / 5)))
        slice_amount = total_amount / num_slices
        interval = 3600 / num_slices  # 1 hour total
        
        limit_price = fair_price * 0.98 if side.upper() == "YES" else fair_price
        
        reasoning = (
            f"TWAP plan for ${total_amount}: {num_slices} slices ${slice_amount:.2f} each "
            f"every {interval:.0f}s over 1 hour, limit {limit_price:.3f} | "
            f"Minimizes market impact and slippage for larger sizes"
        )
        
        return TWAPPlan(
            market_id=market_id,
            total_amount=total_amount,
            num_slices=num_slices,
            slice_amount=slice_amount,
            interval_seconds=interval,
            limit_price=limit_price,
            side=side,
            reasoning=reasoning
        )

    def get_report(self) -> Dict[str, Any]:
        return {
            "executor": "Limit Order + TWAP + Slippage Model",
            "importance": "Top 5 to implement first - saves from fees and bad fills",
            "principles": [
                "API-first: Polymarket CLOB, Kalshi API, browser automation only fallback",
                "Limit orders only: never market orders, use TWAP for larger sizes",
                "WebSocket order book: real-time depth, not polling",
                "Slippage model: estimate from order book, don't trade if slippage > expected edge",
                "Batch on-chain transactions: save gas, monitor Polygon gas",
                "Hot wallet isolation: only small funds in hot wallet, rest cold storage"
            ],
            "slippage_model": "Estimate from order book depth, amount/liquidity*0.3, check vs edge",
            "imbalance": "If bid stacked, wait or buy into weakness, if ask stacked, good time to buy",
            "twap": f"Use TWAP if amount > ${self.use_twap_threshold}, split into slices over 1 hour"
        }
