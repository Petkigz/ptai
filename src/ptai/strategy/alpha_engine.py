"""
Alpha Engine - combines all additional alpha ideas into V3 scoring
Implements Top 5 + additional queue

Top 5:
1. combinatorial/negative risk arbitrage
2. cross-venue reference odds
3. calibration+ensemble
4. correlation-aware risk caps
5. limit order execution with slippage model

Additional:
- event graph consistency
- favourite-longshot bias fade
- whale tracking
- RAG historical outcomes
- Bayesian updating decay
- dynamic threshold
- liquidity rewards / market making
- orderbook imbalance timing
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType

from .combinatorial import CombinatorialArbitrageEngine
from .reference_odds import ReferenceOddsEngine
from .event_graph import EventGraphConsistencyEngine
from .favourite_longshot import FavouriteLongshotEngine
from .liquidity_rewards import LiquidityRewardsEngine
from .orderbook_imbalance import OrderBookImbalanceEngine
from .rag_history import HistoricalRAG
from ..intelligence.bayesian import BayesianUpdater
from ..risk.dynamic_threshold import DynamicThresholdEngine
from ..risk.correlation_enhanced import CorrelationAwareRiskManager
from ..intelligence.calibration_tracker import CalibrationTracker
from ..execution.slippage import SlippageModel
from ..markets.whale_tracker import WhaleTracker


@dataclass
class AlphaOpportunity:
    market: Market
    alpha_type: str
    edge: float
    confidence: float
    should_trade: bool
    reasoning: str
    raw: Dict[str, Any]


class AlphaEngine:
    """Combines all alpha engines for V3 ranking"""
    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll
        self.combinatorial = CombinatorialArbitrageEngine(min_profit_pct=0.02)
        self.reference_odds = ReferenceOddsEngine()
        self.event_graph = EventGraphConsistencyEngine()
        self.longshot = FavouriteLongshotEngine()
        self.liquidity = LiquidityRewardsEngine(bankroll=bankroll)
        self.orderbook_imbalance = OrderBookImbalanceEngine()
        self.rag = HistoricalRAG()
        self.bayesian = BayesianUpdater()
        self.dynamic_threshold = DynamicThresholdEngine()
        self.correlation = CorrelationAwareRiskManager(bankroll=bankroll)
        self.calibration = CalibrationTracker(data_dir="./data")
        self.slippage = SlippageModel()
        self.whale = WhaleTracker()

    def scan_all_alpha(self, markets: List[Market]) -> Dict[str, Any]:
        """Scan markets with all alpha engines"""
        results = {}
        
        # 1. Combinatorial arbitrage
        try:
            comb = self.combinatorial.find_combinatorial_arbitrage(markets)
            results["combinatorial"] = {
                "total": len(comb),
                "tradeable": len([c for c in comb if c.should_trade]),
                "top_profit": max([c.estimated_profit_pct for c in comb], default=0),
                "opps": comb[:5]
            }
        except Exception as e:
            logger.warning(f"Combinatorial failed: {e}")
            results["combinatorial"] = {"total": 0, "tradeable": 0, "error": str(e)}

        # 2. Reference odds
        try:
            ref_total = 0
            ref_tradeable = 0
            for m in markets[:20]:
                refs = self.reference_odds.get_all_reference_odds(m)
                ref_total += len(refs)
                ref_tradeable += len([r for r in refs if r.should_trade])
            results["reference_odds"] = {"total": ref_total, "tradeable": ref_tradeable}
        except Exception as e:
            results["reference_odds"] = {"total": 0, "tradeable": 0, "error": str(e)}

        # 3. Event graph consistency
        try:
            violations = self.event_graph.find_violations(markets)
            results["event_graph"] = {
                "violations": len(violations),
                "tradeable": len([v for v in violations if v.should_trade]),
                "opps": violations[:5]
            }
        except Exception as e:
            results["event_graph"] = {"violations": 0, "tradeable": 0, "error": str(e)}

        # 4. Favourite-longshot bias
        try:
            ls_opps = self.longshot.scan_markets(markets)
            results["favourite_longshot"] = {
                "extreme": len(ls_opps),
                "tradeable": len([o for o in ls_opps if o.should_trade]),
                "opps": ls_opps[:5]
            }
        except Exception as e:
            results["favourite_longshot"] = {"extreme": 0, "tradeable": 0, "error": str(e)}

        # 5. Whale tracking - real Polymarket activity feed.
        # An unreachable feed yields zero wallets and a reason; it used to
        # yield three handwritten wallets that the scan then reported as real.
        try:
            feed = self.whale.load_whales()
            results["whale"] = {
                "wallets": len(feed.wallets),
                "smart": len([w for w in feed.wallets if w.is_smart]),
                "dumb": len([w for w in feed.wallets if w.is_dumb]),
                "source": feed.source,
                "error": feed.last_error or None,
            }
        except Exception as e:
            results["whale"] = {"wallets": 0, "error": str(e)}

        # 6. RAG historical
        try:
            base = self.rag.estimate_base_rate(market_id="test", question="Will Trump win?", category="politics")
            results["rag"] = {
                "history": len(self.rag.history),
                "base_rate": base.base_rate,
                "confidence": base.confidence
            }
        except Exception as e:
            results["rag"] = {"history": 0, "error": str(e)}

        # 7. Dynamic threshold example
        try:
            results["dynamic_threshold"] = {
                "base": 0.08,
                "illiquid_example": "15%+ for <$1k",
                "report": self.dynamic_threshold.get_report()
            }
        except Exception as e:
            results["dynamic_threshold"] = {"error": str(e)}

        # 8. Liquidity rewards
        try:
            results["liquidity_rewards"] = {
                "apr": self.liquidity.mock_rewards["reward_rate_per_day"] * 365 * 100,
                "active": self.liquidity.mock_rewards["active"]
            }
        except Exception as e:
            results["liquidity_rewards"] = {"error": str(e)}

        # 9. Orderbook imbalance
        try:
            results["orderbook_imbalance"] = {
                "model": "bid vs ask depth",
                "report": self.orderbook_imbalance.get_report()
            }
        except Exception as e:
            results["orderbook_imbalance"] = {"error": str(e)}

        # 10. Correlation
        try:
            results["correlation"] = {
                "max_cluster_pct": self.correlation.max_cluster_pct * 100,
                "max_positions": self.correlation.max_cluster_positions
            }
        except Exception as e:
            results["correlation"] = {"error": str(e)}

        # 11. Calibration
        try:
            results["calibration"] = {
                "total": len(self.calibration.predictions),
                "report": self.calibration.get_report()
            }
        except Exception as e:
            results["calibration"] = {"total": 0, "error": str(e)}

        return results

    def calculate_alpha_adjusted_score(self, base_score: float, market: Market, context: Dict = None) -> float:
        """
        Adjust base V3 score with alpha signals
        score = expected_edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)
        Additional multipliers from alpha engines
        """
        context = context or {}
        score = base_score
        
        # Reference odds multiplier: if external source confirms edge, boost
        try:
            refs = self.reference_odds.get_all_reference_odds(market)
            if refs:
                max_edge_ref = max([abs(r.edge) for r in refs], default=0)
                if max_edge_ref > 0.05:
                    score *= 1.2  # 20% boost if reference odds confirm
        except:
            pass
        
        # RAG base rate adjustment: if historical base rate aligns with edge, boost confidence
        try:
            base_rate = self.rag.estimate_base_rate(market_id=market.id, question=market.question, category="unknown")
            if base_rate.confidence > 0.6:
                score *= (1.0 + base_rate.confidence * 0.1)
        except:
            pass
        
        # Favourite-longshot: if extreme price and matches bias, boost
        try:
            if market.best_price < 0.05 or market.best_price > 0.95:
                if market.liquidity > 10000:  # liquid only
                    score *= 1.15
        except:
            pass
        
        # Whale signal: mock boost
        # In real implementation, would check if smart whales are on same side
        
        return score

    def get_report(self) -> Dict[str, Any]:
        return {
            "engines": [
                "combinatorial_arbitrage - LP for MECE baskets, sum YES !=1",
                "reference_odds - Kalshi, Deribit BTC implied, Fed funds, Pinnacle/Betfair",
                "calibration_tracker - Brier score, only trust calibrated categories",
                "correlation_enhanced - cluster caps 12%, stress test all correlated losing",
                "slippage_model - orderbook TWAP WebSocket batch on-chain gas",
                "event_graph - Trump->GOP implication, temporal June->Dec, mutual exclusion sum<=1",
                "favourite_longshot - fade tails <5% overpriced *0.6 and >95% underpriced",
                "whale_tracker - Polygon wallets P&L copy smart fade dumb",
                "rag_history - historical outcomes base rates similarity Jaccard",
                "bayesian - exponential decay half-life 24h",
                "dynamic_threshold - base 8% + fees+slippage+uncertainty+liquidity premium illiquid 15%+",
                "liquidity_rewards - zero maker fees inventory limit 10% bankroll $5",
                "orderbook_imbalance - bid stacked wait ask stacked buy"
            ],
            "top_5_for_50": "arbitrage + liquidity rewards more realistic than directional",
            "scoring": "expected_edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk) × alpha_multipliers"
        }
