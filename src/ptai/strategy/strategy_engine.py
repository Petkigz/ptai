"""
V3 Strategy Engine - evaluates venue × market × strategy
Core of PTAI V3: genuinely multi-venue, multi-strategy opportunity engine

Instead of "Find a Polymarket trade", it asks:
"Find the best legitimate opportunity available to my capital right now across all venues and strategies"

Reports:
I scanned 1,200 opportunities.
Polymarket: 3 candidates
Kalshi: 5 candidates
Manifold: 1 candidate
Crypto: 8 candidates
Stocks: 2 candidates
After fees/liquidity/uncertainty: 2 actually tradeable
Best opportunity: Venue B with strategy X

Then learning system determines which combinations actually demonstrate edge.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType
from ..venues.registry import VenueRegistry

from .fair_value import FairValueEngine
from .edge import HUNT_MISPRICING_MIN, EdgeCalculator, hunted_mispricing
from .strategy_selector import StrategyType, StrategySelector
from .arbitrage import ArbitrageEngine
from .event_trading import EventTradingEngine
from .market_making import MarketMakingEngine
from .momentum import MomentumEngine
from .expected_ev import ExpectedNetEVEngine


@dataclass
class VenueScanReport:
    venue_id: str
    venue_type: str
    total_discovered: int
    after_cheap: int
    after_liquidity: int
    candidates: int  # after fast model / strategy evaluation
    tradeable: int   # after effective edge + risk
    top_opportunity: Optional[VenueOpportunity] = None
    avg_edge: float = 0.0
    avg_score: float = 0.0


@dataclass
class MultiVenueScanResult:
    total_scanned: int
    total_after_cheap: int
    total_after_liquidity: int
    total_candidates: int
    total_tradeable: int
    venue_reports: List[VenueScanReport] = field(default_factory=list)
    all_opportunities: List[VenueOpportunity] = field(default_factory=list)
    final_selected: List[VenueOpportunity] = field(default_factory=list)
    arbitrage_opportunities: List = field(default_factory=list)
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)
    execution_time: float = 0.0
    reasoning: str = ""
    best_opportunity: Optional[VenueOpportunity] = None


def _execution_quality_from_book(orderbook, market) -> float:
    """Kept as the name this module's tests import; see the shared implementation."""
    from ..markets.orderbook import execution_quality_from_book
    return execution_quality_from_book(orderbook, market)


class StrategyEngineV3:
    """
    V3 Strategy Engine: venue × market × strategy
    Evaluates all combinations and ranks on common basis
    """
    def __init__(self, 
                 venue_registry: VenueRegistry = None,
                 fair_value_engine: FairValueEngine = None,
                 edge_calculator: EdgeCalculator = None,
                 strategy_selector: StrategySelector = None):
        self.venue_registry = venue_registry
        self.fair_value_engine = fair_value_engine or FairValueEngine()
        self.edge_calculator = edge_calculator or EdgeCalculator()
        self.strategy_selector = strategy_selector or StrategySelector()
        
        # Strategy engines
        self.arbitrage_engine = ArbitrageEngine(min_spread=0.03)
        self.event_engine = EventTradingEngine()
        self.mm_engine = MarketMakingEngine()
        self.momentum_engine = MomentumEngine()
        self.expected_ev_engine = ExpectedNetEVEngine()
        
        # Filters
        self.min_volume_24h = 500
        self.min_liquidity = 100
        self.max_spread = 0.10

    def cheap_filters(self, markets: List[Market]) -> List[Market]:
        filtered = []
        for m in markets:
            if m.volume_24h < self.min_volume_24h:
                continue
            if m.liquidity < self.min_liquidity:
                continue
            if not m.active or m.closed:
                continue
            # Allow extreme prices for mean reversion strategy, but filter for others
            # Keep all for V3, let strategy decide
            filtered.append(m)
        return filtered

    def liquidity_filter(self, markets: List[Market]) -> List[Market]:
        sorted_markets = sorted(markets, key=lambda x: (x.volume_24h, x.liquidity), reverse=True)
        limit = min(200, max(50, len(sorted_markets)))
        return sorted_markets[:limit]

    def evaluate_market_with_all_strategies(self, market: Market, context: Dict = None) -> List[VenueOpportunity]:
        """
        Core V3: evaluate single market with ALL strategies
        Returns list of VenueOpportunities, one per strategy that finds edge
        """
        context = context or {}
        opportunities: List[VenueOpportunity] = []
        venue_id = getattr(market, 'source', 'unknown')
        venue_id_str = venue_id.value if hasattr(venue_id, 'value') else str(venue_id)
        # Handle mock venues with raw venue field
        if market.raw.get("venue"):
            venue_id_str = market.raw["venue"]
        
        # Strategy 1: Mispricing (fair value vs market) - existing engine
        try:
            fv_result = self.fair_value_engine.estimate(market, context=context)
            # The opportunity is built on the hunt criterion plus a positive
            # post-cost edge. A 5% POST-COST floor also stood here, so the same
            # profitable NO trade (effective 0.028) was refused before it existed
            # - the mispricing was never evaluated by anything downstream.
            if fv_result.should_trade and fv_result.effective_edge > 0:
                opp = VenueOpportunity(
                    market=market,
                    venue_id=venue_id_str,
                    venue_type=VenueType.PREDICTION,
                    # The side the edge was measured on. Re-deriving it from the
                    # fair value here is how the two could disagree: the edge was
                    # computed for neither side in particular, and this line then
                    # picked one.
                    side=getattr(fv_result, "side", None) or (
                        "YES" if fv_result.fair_value > market.best_price else "NO"),
                    market_price=market.best_price,
                    estimated_fair=fv_result.fair_value,
                    raw_edge=fv_result.edge,
                    effective_edge=fv_result.effective_edge,
                    confidence=fv_result.confidence,
                    uncertainty=fv_result.uncertainty,
                    liquidity_score=min(1.0, market.liquidity / 10000),
                    # Measured from the book when there is one. This was the
                    # literal 0.8 for every mispricing opportunity, so a market
                    # with a wide spread and no depth scored the same as a deep
                    # one - and execution_quality feeds the opportunity score, the
                    # EV penalty and the ranking, so the ranking was partly made
                    # of a constant.
                    execution_quality=_execution_quality_from_book(
                        context.get("orderbook"), market),
                    category=context.get("category", "mispricing"),
                    sources=fv_result.forecast_result.sources if fv_result.forecast_result else ["fair_value"],
                    reasoning=fv_result.reasoning,
                    bull_case=fv_result.contradiction_report.bull_case if fv_result.contradiction_report else "",
                    bear_case=fv_result.contradiction_report.bear_case if fv_result.contradiction_report else "",
                    resolution_risks=fv_result.resolution_analysis.risks if fv_result.resolution_analysis else [],
                    should_trade=fv_result.should_trade
                )
                opp.calculate_common_score()
                # Tag strategy
                opp.raw = {"strategy": "mispricing", "venue": venue_id_str}
                opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Mispricing eval failed for {market.id}: {e}")

        # Strategy 2: Event trading
        try:
            event_signal = self.event_engine.evaluate(market, context=context)
            if event_signal.should_trade:
                opp = self.event_engine.to_venue_opportunity(market, event_signal, context=context)
                if opp:
                    opp.raw = {"strategy": "event_trading", "venue": venue_id_str}
                    opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Event trading eval failed for {market.id}: {e}")

        # Strategy 3: Market making (if orderbook available)
        try:
            orderbook = context.get("orderbook", {})
            if orderbook:
                mm_signal = self.mm_engine.evaluate(market, orderbook=orderbook)
                if mm_signal.should_trade:
                    opp = self.mm_engine.to_venue_opportunity(market, mm_signal)
                    if opp:
                        opp.raw = {"strategy": "market_making", "venue": venue_id_str}
                        opportunities.append(opp)
        except Exception as e:
            logger.debug(f"MM eval failed for {market.id}: {e}")

        # Strategy 4 & 5: Momentum and Mean Reversion
        try:
            mom_signals = self.momentum_engine.evaluate(market, context=context)
            mom_opps = self.momentum_engine.to_venue_opportunities(market, mom_signals)
            for opp in mom_opps:
                opp.raw = {"strategy": opp.category, "venue": venue_id_str}
                opportunities.append(opp)
        except Exception as e:
            logger.debug(f"Momentum eval failed for {market.id}: {e}")

        return opportunities

    async def scan_venue(self, venue_id: str, markets: List[Market], context_provider=None) -> VenueScanReport:
        """Scan single venue with all strategies"""
        start_total = len(markets)
        
        after_cheap = self.cheap_filters(markets)
        after_liquidity = self.liquidity_filter(after_cheap)
        
        # Evaluate each market with all strategies
        all_opps: List[VenueOpportunity] = []
        for market in after_liquidity[:100]:  # Limit per venue for performance
            context = {}
            if context_provider:
                try:
                    context = await context_provider.get_context(market)
                except Exception as e:
                    # NEVER silent. This except used to swallow an
                    # AttributeError - the provider had no get_context - and
                    # substitute a fabricated orderbook for every market on every
                    # cycle. The forecast was then built on a placeholder spread
                    # and a depth copied from the market's own liquidity field,
                    # with nothing logged, so the pipeline looked connected while
                    # no news, sentiment, research or real book reached it.
                    logger.error(
                        f"Context provider FAILED for {market.id}: "
                        f"{type(e).__name__}: {e}. Falling back to a degraded "
                        f"context - the forecast for this market is built WITHOUT "
                        f"news, sentiment, research or a real orderbook, and is "
                        f"marked degraded so nothing downstream mistakes it for a "
                        f"researched forecast.")
                    context = {
                        "orderbook": {"spread": 0.02, "is_real": False},
                        "category": "unknown",
                        "degraded": True,
                        "degraded_reason": f"{type(e).__name__}: {e}",
                        "sources": [],
                    }
            else:
                # No provider at all is also a degraded forecast, not a normal
                # one. Same reason: silence here is indistinguishable from a
                # researched market.
                context = {
                    "orderbook": {"spread": 0.02, "is_real": False},
                    "category": "unknown",
                    "degraded": True,
                    "degraded_reason": "no context provider supplied",
                    "sources": [],
                }
            
            opps = self.evaluate_market_with_all_strategies(market, context=context)
            all_opps.extend(opps)
        
        # Candidates after strategy evaluation
        candidates = [o for o in all_opps if o.effective_edge >= 0.03]
        # 8% of MISPRICING, not 8% of post-cost edge - see
        # `hunted_mispricing`. Costs are charged in the net EV terms.
        tradeable = [o for o in all_opps
                     if hunted_mispricing(o) >= HUNT_MISPRICING_MIN
                     and o.confidence >= 0.6 and o.should_trade]
        
        # Sort tradeable by score
        tradeable_sorted = sorted(tradeable, key=lambda x: x.score, reverse=True)
        top_opp = tradeable_sorted[0] if tradeable_sorted else (sorted(candidates, key=lambda x: x.score, reverse=True)[0] if candidates else None)
        
        avg_edge = sum(o.effective_edge for o in candidates) / len(candidates) if candidates else 0
        avg_score = sum(o.score for o in candidates) / len(candidates) if candidates else 0
        
        # Determine venue type
        venue_type = "unknown"
        if markets and len(markets) > 0:
            src = getattr(markets[0], 'source', 'unknown')
            if hasattr(src, 'value'):
                venue_type = src.value
            elif markets[0].raw.get("venue"):
                venue_type = markets[0].raw["venue"]
        
        return VenueScanReport(
            venue_id=venue_id,
            venue_type=venue_type,
            total_discovered=start_total,
            after_cheap=len(after_cheap),
            after_liquidity=len(after_liquidity),
            candidates=len(candidates),
            tradeable=len(tradeable),
            top_opportunity=top_opp,
            avg_edge=avg_edge,
            avg_score=avg_score
        ), all_opps

    async def scan_all_venues(self, markets_by_venue: Dict[str, List[Market]], context_provider=None, max_final_trades: int = 3) -> MultiVenueScanResult:
        """
        Main V3 entry: scan all venues, evaluate all strategies, rank on common basis
        Returns report like:
        I scanned 1,200 opportunities.
        Polymarket: 3 candidates
        Kalshi: 5 candidates
        After fees/liquidity/uncertainty: 2 actually tradeable
        Best opportunity: Venue B
        """
        start = time.time()
        
        total_scanned = sum(len(m) for m in markets_by_venue.values())
        venue_reports: List[VenueScanReport] = []
        all_opportunities: List[VenueOpportunity] = []
        strategy_breakdown: Dict[str, int] = {}
        
        # Scan each venue
        for venue_id, markets in markets_by_venue.items():
            try:
                report, opps = await self.scan_venue(venue_id, markets, context_provider=context_provider)
                venue_reports.append(report)
                all_opportunities.extend(opps)
                
                # Strategy breakdown
                for opp in opps:
                    strat = opp.raw.get("strategy", "unknown") if hasattr(opp, 'raw') and isinstance(opp.raw, dict) else "unknown"
                    strategy_breakdown[strat] = strategy_breakdown.get(strat, 0) + 1
                    
            except Exception as e:
                logger.error(f"Venue {venue_id} scan failed: {e}")
                venue_reports.append(VenueScanReport(
                    venue_id=venue_id,
                    venue_type="unknown",
                    total_discovered=len(markets),
                    after_cheap=0,
                    after_liquidity=0,
                    candidates=0,
                    tradeable=0
                ))
        
        # Arbitrage across all venues
        all_markets_flat = [m for markets in markets_by_venue.values() for m in markets]
        arbitrage_opps = []
        try:
            arbitrage_raw = self.arbitrage_engine.find_arbitrage(all_markets_flat[:500])  # Limit for performance
            arbitrage_opps = arbitrage_raw
            arb_venue_opps = self.arbitrage_engine.to_venue_opportunities(arbitrage_raw)
            all_opportunities.extend(arb_venue_opps)
            strategy_breakdown["arbitrage"] = len(arb_venue_opps)
        except Exception as e:
            logger.warning(f"Arbitrage scan failed: {e}")
        
        # V10 FIX #9: Common ranking using Expected Net EV - economically meaningful
        # Old: edge×prob×liquidity×execution×calibration×time/(fees+slippage+uncertainty+risk) - not $ profit
        # New: Σ prob×payoff − fees − spread − slippage − funding − gas − execution_loss − uncertainty_penalty
        # Then per dollar, per risk, per capital-time
        for opp in all_opportunities:
            try:
                ev = self.expected_ev_engine.calculate(opp, amount_usd=3.0, orderbook=opp.market.raw.get("orderbook", {}) if hasattr(opp.market, 'raw') and isinstance(opp.market.raw, dict) else {})
                opp._expected_ev = ev
                # Update score to be economically meaningful
                opp.score = ev.net_ev_usd * ev.ev_per_dollar * ev.ev_per_risk if ev.net_ev_usd > 0 else 0
            except Exception as e:
                logger.debug(f"EV calc failed for {opp.market.id}: {e}")
                opp._expected_ev = None
        
        # Rank by net EV first, then by old score
        if self.venue_registry:
            ranked = self.venue_registry.rank_opportunities(all_opportunities)
            # Re-rank by expected net EV where available
            ranked = sorted(ranked, key=lambda x: (getattr(x, '_expected_ev', None).net_ev_usd if hasattr(x, '_expected_ev') and x._expected_ev else 0, x.score), reverse=True)
        else:
            ranked = sorted(all_opportunities, key=lambda x: (getattr(x, '_expected_ev', None).net_ev_usd if hasattr(x, '_expected_ev') and x._expected_ev else 0, x.score), reverse=True)
        
        # Filter to tradeable - now also requires net EV >0.
        #
        # This is the filter that actually decides `final_selected`, so the same
        # double-count here would have made the EV gate's fix decoration: a NO
        # trade with raw +0.150, net +$1.07 and 2.8% of edge surviving the costs
        # cleared the EV gate and was then dropped on this line. The 8% is the
        # hunt criterion and is measured on the raw mispricing; the net EV check
        # below is where the costs are charged.
        tradeable = []
        for o in ranked:
            if (hunted_mispricing(o) < HUNT_MISPRICING_MIN
                    or o.confidence < 0.6 or not o.should_trade):
                continue
            ev = getattr(o, '_expected_ev', None)
            if ev and ev.net_ev_usd <= 0:
                continue
            tradeable.append(o)
        
        final_selected = tradeable[:max_final_trades]
        
        best_opp = final_selected[0] if final_selected else (ranked[0] if ranked else None)
        
        elapsed = time.time() - start
        
        # Build reasoning report
        venue_summary = "\n".join([
            f"{r.venue_id}: discovered {r.total_discovered}, candidates {r.candidates}, tradeable {r.tradeable}, "
            f"avg_edge {r.avg_edge:.3f}, top_score {r.top_opportunity.score:.3f} {r.top_opportunity.side} edge {r.top_opportunity.effective_edge:.3f} | {r.top_opportunity.market.question[:60]}" 
            if r.top_opportunity else f"{r.venue_id}: discovered {r.total_discovered}, candidates {r.candidates}, tradeable {r.tradeable}"
            for r in venue_reports
        ])
        
        strategy_summary = ", ".join([f"{k}: {v}" for k, v in strategy_breakdown.items()])
        
        # The None guard must cover every field, not just the first. This
        # previously read `best_opp.side if best_opp else ''` and then
        # dereferenced best_opp.effective_edge unconditionally, so the log line
        # raised AttributeError exactly when there was nothing to trade -
        # turning "no opportunities found" into a 500 on the endpoint.
        if best_opp is not None:
            _strategy = best_opp.raw.get("strategy") if hasattr(best_opp, "raw") else "unknown"
            best_line = (f"Best opportunity: {best_opp.venue_id} {best_opp.side} "
                         f"edge {best_opp.effective_edge:.3f} score {best_opp.score:.3f} "
                         f"strategy {_strategy} | {best_opp.market.question[:80]}")
        else:
            best_line = "Best opportunity: none - nothing cleared the gates | DO NOTHING"

        reasoning = (
            f"I scanned {total_scanned} opportunities across {len(markets_by_venue)} venues.\n"
            f"{venue_summary}\n"
            f"Strategies evaluated: {strategy_summary}\n"
            f"Arbitrage candidates: {len([a for a in arbitrage_opps if a.should_trade])} tradeable out of {len(arbitrage_opps)}\n"
            f"After fees/liquidity/uncertainty/risk: {len(tradeable)} actually tradeable opportunities\n"
            f"{best_line}\n"
            f"Final selected {len(final_selected)} trades (max {max_final_trades}) | Time {elapsed:.1f}s"
        )
        
        logger.info(reasoning)
        
        total_after_cheap = sum(r.after_cheap for r in venue_reports)
        total_after_liq = sum(r.after_liquidity for r in venue_reports)
        total_candidates = sum(r.candidates for r in venue_reports)
        total_tradeable = len(tradeable)
        
        return MultiVenueScanResult(
            total_scanned=total_scanned,
            total_after_cheap=total_after_cheap,
            total_after_liquidity=total_after_liq,
            total_candidates=total_candidates,
            total_tradeable=total_tradeable,
            venue_reports=venue_reports,
            all_opportunities=ranked,
            final_selected=final_selected,
            arbitrage_opportunities=arbitrage_opps,
            strategy_breakdown=strategy_breakdown,
            execution_time=elapsed,
            reasoning=reasoning,
            best_opportunity=best_opp
        )
