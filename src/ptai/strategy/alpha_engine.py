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
from dataclasses import dataclass, field
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
class AlphaAdjustment:
    """
    The multiplier applied to a base score, and exactly what produced it.

    A score adjustment with no record of what was applied cannot be audited,
    which is how a direction-blind multiplier survived unnoticed: nothing
    anywhere said which signal had moved the number.
    """
    multiplier: float
    applied: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def boosted(self) -> bool:
        return self.multiplier > 1.0 + 1e-9

    @property
    def penalised(self) -> bool:
        return self.multiplier < 1.0 - 1e-9


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
                "apr": round(self.liquidity.rewards.reward_apr * 100, 2),
                "reward_configured": self.liquidity.rewards.reward_available
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

    # ------------------------------------------------------------------
    # Alpha score adjustment
    # ------------------------------------------------------------------

    def _trade_direction(self, side: str) -> int:
        """+1 when the trade wants the probability higher than the market says."""
        s = (side or "YES").upper()
        if s in ("YES", "BUY", "LONG", "BACK", "OVER", "HOME"):
            return 1
        if s in ("NO", "SELL", "SHORT", "LAY", "UNDER", "AWAY"):
            return -1
        return 0

    def calculate_alpha_adjustment(self, market: Market, side: str = "YES",
                                   kalshi_markets: Optional[List[Market]] = None,
                                   whale_signals: Optional[List[Any]] = None,
                                   context: Optional[Dict] = None) -> AlphaAdjustment:
        """
        Directional multiplier from the alpha signals, with an audit trail.

        Every signal here is compared against the DIRECTION of the trade. The
        previous version tested magnitudes only:

            if max_edge_ref > 0.05: score *= 1.2

        so a reference price that contradicted the trade by 5% boosted it by
        20%, a RAG base rate pointing the other way boosted it, and the
        favourite-longshot rule boosted buying an overpriced longshot - which
        is precisely the trade the bias says to fade. Three signals that were
        supposed to be confirmation were confirming their own opposite.

        Signals that fail are recorded in `errors`, not swallowed by a bare
        `except: pass`.
        """
        context = context or {}
        direction = self._trade_direction(side)
        price = float(getattr(market, "best_price", 0.0) or 0.0)
        applied: List[str] = []
        errors: List[str] = []
        detail: Dict[str, Any] = {"side": side, "direction": direction, "price": price}
        multiplier = 1.0

        if direction == 0:
            errors.append(f"unknown trade side '{side}' - no alpha signals applied")
            return AlphaAdjustment(1.0, applied, errors, detail)

        # ---- 1. cross-venue reference odds -----------------------------
        try:
            refs = self.reference_odds.get_all_reference_odds(market, kalshi_markets)
            confirming = [r for r in refs if r.edge * direction > 0.05 and r.should_trade]
            contradicting = [r for r in refs if r.edge * direction < -0.05 and r.should_trade]
            detail["references"] = len(refs)
            detail["references_confirming"] = len(confirming)
            detail["references_contradicting"] = len(contradicting)
            if confirming and len(confirming) >= len(contradicting):
                best = max(confirming, key=lambda r: abs(r.edge) * r.confidence)
                multiplier *= 1.20
                applied.append(f"reference odds confirm ({best.source} edge "
                               f"{best.edge*direction*100:+.1f}% in favour) x1.20")
            elif contradicting:
                worst = max(contradicting, key=lambda r: abs(r.edge))
                multiplier *= 0.75
                applied.append(f"reference odds contradict ({worst.source} edge "
                               f"{worst.edge*direction*100:+.1f}% against) x0.75")
        except Exception as e:
            errors.append(f"reference_odds: {type(e).__name__}: {e}")

        # ---- 2. RAG historical base rate -------------------------------
        try:
            category = (getattr(market, "raw", {}) or {}).get("category") \
                or context.get("category") or "unknown"
            base_rate = self.rag.estimate_base_rate(
                market_id=market.id, question=market.question, category=category)
            detail["rag_base_rate"] = base_rate.base_rate
            detail["rag_confidence"] = base_rate.confidence
            detail["rag_similar"] = base_rate.num_similar
            # A base rate only confirms if it sits on the same side of the
            # market price as the trade does.
            if base_rate.confidence > 0.6 and base_rate.num_similar > 0:
                agreement = (base_rate.base_rate - price) * direction
                if agreement > 0.03:
                    boost = 1.0 + min(0.10, base_rate.confidence * 0.10)
                    multiplier *= boost
                    applied.append(f"RAG base rate {base_rate.base_rate:.2f} vs price "
                                   f"{price:.2f} agrees (n={base_rate.num_similar}) "
                                   f"x{boost:.3f}")
                elif agreement < -0.03:
                    multiplier *= 0.90
                    applied.append(f"RAG base rate {base_rate.base_rate:.2f} vs price "
                                   f"{price:.2f} disagrees x0.90")
        except Exception as e:
            errors.append(f"rag: {type(e).__name__}: {e}")

        # ---- 3. favourite-longshot bias --------------------------------
        try:
            liquidity = float(getattr(market, "liquidity", 0.0) or 0.0)
            if liquidity > 10000:
                # Longshots are OVERpriced, so fading them (side NO) is with
                # the bias. Favourites are UNDERpriced, so buying them (side
                # YES) is with the bias. Boosting either tail regardless of
                # side - as before - boosted the losing trade.
                with_bias = (price < 0.05 and direction < 0) or (price > 0.95 and direction > 0)
                against_bias = (price < 0.05 and direction > 0) or (price > 0.95 and direction < 0)
                if with_bias:
                    multiplier *= 1.15
                    applied.append(f"price {price:.3f} is a mispriced tail and this trade "
                                   "fades it with the bias x1.15")
                elif against_bias:
                    multiplier *= 0.85
                    applied.append(f"price {price:.3f} is a mispriced tail and this trade "
                                   "buys into the bias x0.85")
        except Exception as e:
            errors.append(f"favourite_longshot: {type(e).__name__}: {e}")

        # ---- 4. whale confirmation -------------------------------------
        try:
            signals = whale_signals if whale_signals is not None else context.get("whale_signals")
            if signals:
                aligned = [s for s in signals if getattr(s, "should_trade", False)
                           and self._whale_aligns(s, direction)]
                detail["whale_signals"] = len(signals)
                detail["whale_aligned"] = len(aligned)
                if aligned:
                    best = max(aligned, key=lambda s: abs(getattr(s, "edge_estimate", 0.0)))
                    multiplier *= 1.10
                    applied.append(f"{len(aligned)} whale signal(s) aligned, best edge "
                                   f"{getattr(best, 'edge_estimate', 0.0)*100:+.1f}% x1.10")
        except Exception as e:
            errors.append(f"whale: {type(e).__name__}: {e}")

        return AlphaAdjustment(multiplier=round(multiplier, 4), applied=applied,
                               errors=errors, detail=detail)

    @staticmethod
    def _whale_aligns(signal: Any, direction: int) -> bool:
        """
        Whether a whale signal points the same way as the trade.

        A copy signal follows the whale's side; a fade signal takes the
        opposite side. Comparing without accounting for that would treat
        "fade this dumb whale's YES" as agreement with a YES trade.
        """
        sig_type = str(getattr(signal, "signal_type", ""))
        side = str(getattr(signal, "side", "YES")).upper()
        whale_wants_yes = side in ("YES", "BUY")
        if sig_type == "fade_dumb":
            whale_wants_yes = not whale_wants_yes
        return (1 if whale_wants_yes else -1) == direction

    def calculate_alpha_adjusted_score(self, base_score: float, market: Market,
                                       context: Optional[Dict] = None) -> float:
        """
        Apply the alpha multiplier to a base score.

        Kept for backward compatibility. `context` may carry `side`,
        `kalshi_markets` and `whale_signals`; the default side is YES because
        that is what every existing caller assumed.
        """
        context = context or {}
        adjustment = self.calculate_alpha_adjustment(
            market,
            side=context.get("side", "YES"),
            kalshi_markets=context.get("kalshi_markets"),
            whale_signals=context.get("whale_signals"),
            context=context)
        return base_score * adjustment.multiplier

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
