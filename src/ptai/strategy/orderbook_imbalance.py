"""
Order Book Imbalance - Use CLOB depth to time entries
If bid side stacked, wait or buy into weakness
"""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ImbalanceSignal:
    market_id: str
    imbalance: float
    bid_depth: float
    ask_depth: float
    spread: float
    signal: str
    timing: str
    should_trade: bool
    reasoning: str


class OrderBookImbalanceEngine:
    def __init__(self):
        pass

    def analyze(self, market_id: str = "test", orderbook: Dict = None, *args, **kwargs) -> ImbalanceSignal:
        # Support both signatures: analyze(market_id, orderbook) and analyze(orderbook)
        if isinstance(market_id, dict) and orderbook is None:
            orderbook = market_id
            market_id = orderbook.get("market_id", "test")
        if orderbook is None:
            orderbook = kwargs.get("orderbook", {})
            if not orderbook and args:
                if isinstance(args[0], dict):
                    orderbook = args[0]
        if orderbook is None:
            orderbook = {}

        bid_depth = orderbook.get("bid_size", orderbook.get("bid_depth", 5000))
        ask_depth = orderbook.get("ask_size", orderbook.get("ask_depth", 5000))
        spread = orderbook.get("spread", 0.02)
        
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0
        
        if imbalance > 0.4:
            signal = "BULLISH_STACKED_BID"
            timing = "Wait - bid stacked, price may rise, don't chase"
            should_trade = False
        elif imbalance < -0.4:
            signal = "BEARISH_STACKED_ASK"
            timing = "Buy weakness - ask stacked, good entry"
            should_trade = True
        elif imbalance > 0.2:
            signal = "MILD_BULLISH"
            timing = "Slight bullish, OK to buy but small size"
            should_trade = True
        elif imbalance < -0.2:
            signal = "MILD_BEARISH"
            timing = "Slight bearish, good time to buy"
            should_trade = True
        else:
            signal = "BALANCED"
            timing = "Balanced, neutral timing"
            should_trade = True
        
        reasoning = (
            f"Order book imbalance for {market_id}: bid ${bid_depth} ask ${ask_depth} "
            f"imbalance {imbalance:.2f} spread {spread*100:.2f}% | "
            f"Signal {signal} timing {timing} should_trade {should_trade}"
        )
        
        return ImbalanceSignal(
            market_id=market_id,
            imbalance=imbalance,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread=spread,
            signal=signal,
            timing=timing,
            should_trade=should_trade,
            reasoning=reasoning
        )

    def analyze_imbalance(self, orderbook: Dict, market_id: str = "test") -> ImbalanceSignal:
        return self.analyze(market_id=market_id, orderbook=orderbook)

    def get_entry_timing_signal(self, depth_signal, side: str = "YES"):
        """For compatibility with dashboard code"""
        # depth_signal can be ImbalanceSignal or dict
        if isinstance(depth_signal, dict):
            imbalance = depth_signal.get("imbalance", 0)
            bid_depth = depth_signal.get("bid_depth", 0)
            ask_depth = depth_signal.get("ask_depth", 0)
            spread = depth_signal.get("spread", 0.02)
            signal_obj = self.analyze(orderbook=depth_signal)
        else:
            signal_obj = depth_signal
            imbalance = signal_obj.imbalance
        return signal_obj.signal, signal_obj.timing

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Order Book Imbalance",
            "method": "Use CLOB depth to time entries, if bid stacked wait or buy into weakness",
            "signals": {
                "BULLISH_STACKED_BID": "imbalance >0.4 bid stacked, wait don't chase",
                "BEARISH_STACKED_ASK": "imbalance <-0.4 ask stacked, buy weakness good entry",
                "BALANCED": "imbalance -0.2 to 0.2 balanced neutral"
            }
        }
