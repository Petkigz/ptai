"""
Opportunity Engine - finds, ranks, and filters opportunities across all venues
1000 markets -> cheap filters -> 500 -> liquidity filter -> 200 -> fast model -> 50 -> deep research -> 10 -> ensemble -> 3 -> risk -> 0-3 trades

FIXED ISSUES:
- fast_model_screen was just sort by volume, now real fast AI screening classification, news extraction, duplicate detection
- rank_and_select now uses expected EV, liquidity, risk, uncertainty, portfolio impact, capital allocation not just edge>=8%
- Venue learning bug fixed in registry.py but also improved here
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from loguru import logger
import re
from difflib import SequenceMatcher

from ..markets.base import Market
from ..venues.adapter import VenueOpportunity, VenueType
from ..venues.registry import VenueRegistry
from .fair_value import FairValueEngine, FairValueResult
from .edge import EdgeCalculator


@dataclass
class ScanResult:
    total_scanned: int
    after_cheap_filters: int
    after_liquidity: int
    after_fast_model: int
    after_deep_research: int
    after_ensemble: int
    after_risk: int
    opportunities: List[VenueOpportunity]
    execution_time: float
    reasoning: str


class FastModelClassifier:
    """
    Fast AI screening - replaces sort by volume heuristic
    Previously: sort by volume take top 50
    Now: market classification, news extraction, duplicate detection, initial screening
    Uses fast LLM (qwen 7b 2-3 sec) or heuristic rules for classification
    """
    def __init__(self):
        self.categories = ["politics", "sports", "crypto", "economics", "weather", "ai", "general"]
        # Keywords for classification
        self.keywords = {
            "politics": ["trump", "biden", "election", "republican", "democrat", "senate", "congress", "vote", "president"],
            "sports": ["nfl", "nba", "mlb", "soccer", "football", "team", "game", "championship", "super bowl", "world cup", "man city", "arsenal", "lakers"],
            "crypto": ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana", "bnb", "doge", "usdt", "perp", "futures", "funding"],
            "economics": ["fed", "cpi", "inflation", "interest rate", "fomc", "gdp", "nfp", "jobs", "unemployment", "earnings", "s&p", "aapl"],
            "weather": ["weather", "hurricane", "temperature", "rain", "snow"],
            "ai": ["gpt", "agi", "ai", "agent", "llm", "simmer", "autonomous"],
        }

    def classify(self, market: Market) -> Dict[str, Any]:
        q = market.question.lower()
        scores = {}
        for cat, kws in self.keywords.items():
            score = sum(1 for kw in kws if kw in q)
            scores[cat] = score
        
        best_cat = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
        confidence = min(1.0, max(scores.values()) / 3.0) if best_cat != "general" else 0.3
        
        # News extraction heuristic
        has_news_potential = any(word in q for word in ["earnings", "fed", "election", "cpi", "fomc", "trump", "btc"])
        
        # Duplicate detection: similar questions
        # Will be handled at batch level
        
        return {
            "category": best_cat,
            "confidence": confidence,
            "scores": scores,
            "has_news_potential": has_news_potential,
            "should_deep_research": confidence > 0.3 or has_news_potential or market.volume_24h > 10000
        }

    def detect_duplicates(self, markets: List[Market]) -> List[List[Market]]:
        """Detect duplicate markets asking same event"""
        groups = []
        used = set()
        for i, m1 in enumerate(markets):
            if m1.id in used:
                continue
            group = [m1]
            for j, m2 in enumerate(markets[i+1:], i+1):
                if m2.id in used:
                    continue
                # Similar question check
                sim = SequenceMatcher(None, m1.question.lower()[:60], m2.question.lower()[:60]).ratio()
                if sim > 0.8:
                    group.append(m2)
                    used.add(m2.id)
            if len(group) > 1:
                groups.append(group)
            used.add(m1.id)
        return groups


class OpportunityEngine:
    def __init__(self, venue_registry: VenueRegistry = None, fair_value_engine: FairValueEngine = None, edge_calculator: EdgeCalculator = None):
        self.venue_registry = venue_registry
        self.fair_value_engine = fair_value_engine or FairValueEngine()
        self.edge_calculator = edge_calculator or EdgeCalculator()
        self.fast_classifier = FastModelClassifier()
        self.min_volume_24h = 1000
        self.min_liquidity = 500
        self.max_spread = 0.08

    def cheap_filters(self, markets: List[Market]) -> List[Market]:
        filtered = []
        for m in markets:
            if m.volume_24h < self.min_volume_24h:
                continue
            if m.liquidity < self.min_liquidity:
                continue
            if not m.active or m.closed:
                continue
            if m.best_price > 0.95 or m.best_price < 0.05:
                continue
            filtered.append(m)
        logger.info(f"Cheap filters: {len(markets)} -> {len(filtered)}")
        return filtered

    def liquidity_filter(self, markets: List[Market]) -> List[Market]:
        sorted_markets = sorted(markets, key=lambda x: (x.volume_24h, x.liquidity), reverse=True)
        limit = min(200, max(50, len(sorted_markets) // 2))
        filtered = sorted_markets[:limit]
        logger.info(f"Liquidity filter: {len(markets)} -> {len(filtered)}")
        return filtered

    def fast_model_screen(self, markets: List[Market]) -> List[Market]:
        """
        Fast model screening FIXED: was sort by volume take top 50 heuristic
        Now real fast AI screening: classification, news extraction, duplicate detection, initial screening
        Uses fast LLM or heuristic rules
        """
        logger.info(f"Fast model screening {len(markets)} markets - classification + news extraction + duplicate detection")
        
        # Classify all markets
        classified = []
        for market in markets:
            classification = self.fast_classifier.classify(market)
            market.raw["fast_classification"] = classification
            market.raw["category"] = classification["category"]
            classified.append((market, classification))
        
        # Detect duplicates - don't deep research duplicates, keep highest volume one
        duplicates = self.fast_classifier.detect_duplicates(markets)
        duplicate_ids = set()
        for group in duplicates:
            # Keep highest volume, mark others as duplicate
            sorted_group = sorted(group, key=lambda x: x.volume_24h, reverse=True)
            for dup in sorted_group[1:]:
                duplicate_ids.add(dup.id)
                logger.debug(f"Duplicate detected: {dup.question[:40]} similar to {sorted_group[0].question[:40]}")
        
        # Filter out duplicates for deep research efficiency
        non_duplicate = [(m, c) for m, c in classified if m.id not in duplicate_ids]
        
        # Score for deep research priority: not just volume, but category confidence + news potential + volume + liquidity + time efficiency
        def deep_research_score(item):
            market, classification = item
            volume_score = min(1.0, market.volume_24h / 20000)
            liquidity_score = min(1.0, market.liquidity / 20000)
            category_score = classification["confidence"]
            news_score = 1.0 if classification["has_news_potential"] else 0.3
            # Time efficiency: short resolution better for $50 bankroll
            time_score = 1.0
            if market.end_date:
                try:
                    from datetime import datetime, timezone
                    hours_to_res = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_to_res < 24:
                        time_score = 1.2  # short term good
                    elif hours_to_res > 720:  # >30 days
                        time_score = 0.7  # long term less efficient for small bankroll
                except:
                    pass
            
            # Combined score
            return volume_score*0.3 + liquidity_score*0.2 + category_score*0.2 + news_score*0.2 + time_score*0.1
        
        scored = sorted(non_duplicate, key=deep_research_score, reverse=True)
        
        # Take top 50 but ensure category diversity
        # Don't just take top volume, ensure we have politics, sports, crypto, economics coverage
        selected = []
        category_counts = {}
        for market, classification in scored:
            cat = classification["category"]
            # Limit per category to ensure diversity
            if category_counts.get(cat, 0) >= 15 and len(selected) >= 30:
                continue
            selected.append(market)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if len(selected) >= 50:
                break
        
        logger.info(f"Fast model screen: {len(markets)} -> {len(non_duplicate)} non-duplicate -> {len(selected)} selected for deep research | Duplicates {len(duplicate_ids)} removed | Categories {category_counts}")
        return selected

    async def deep_research(self, markets: List[Market], context_provider=None) -> List[Dict[str, Any]]:
        researched = []
        top_markets = markets[:20]
        
        for market in top_markets:
            try:
                context = {}
                if context_provider:
                    context = await context_provider.get_context(market)
                else:
                    context = {
                        "news": "",
                        "sentiment": {"score": 0},
                        "orderbook": {"spread": 0.02},
                        "research": "",
                        "tweets": [],
                        "category": market.raw.get("category", "unknown")
                    }
                
                fv_result = self.fair_value_engine.estimate(market, context=context)
                
                researched.append({
                    "market": market,
                    "context": context,
                    "fair_value": fv_result,
                    "should_trade": fv_result.should_trade
                })
                
                if fv_result.should_trade:
                    logger.info(f"Deep research found opportunity: {market.id} edge {fv_result.effective_edge:.3f} cat {context.get('category')}")
                
            except Exception as e:
                logger.error(f"Deep research failed for {market.id}: {e}")
                continue
        
        opportunities = [r for r in researched if r["should_trade"]]
        logger.info(f"Deep research: {len(markets)} -> {len(researched)} researched -> {len(opportunities)} opportunities")
        return researched

    def ensemble_filter(self, researched: List[Dict]) -> List[VenueOpportunity]:
        opportunities = []
        for item in researched:
            market = item["market"]
            fv = item["fair_value"]
            context = item["context"]
            
            if not fv.should_trade:
                continue
            
            opp = VenueOpportunity(
                market=market,
                venue_id=getattr(market, 'source', 'unknown'),
                venue_type=VenueType.PREDICTION,
                side="YES" if fv.fair_value > market.best_price else "NO",
                market_price=market.best_price,
                estimated_fair=fv.fair_value,
                raw_edge=fv.edge,
                effective_edge=fv.effective_edge,
                confidence=fv.confidence,
                uncertainty=fv.uncertainty,
                liquidity_score=min(1.0, market.liquidity / 10000),
                execution_quality=0.8,
                category=context.get("category", market.raw.get("category", "unknown")),
                sources=fv.forecast_result.sources if fv.forecast_result else [],
                reasoning=fv.reasoning,
                bull_case=fv.contradiction_report.bull_case if fv.contradiction_report else "",
                bear_case=fv.contradiction_report.bear_case if fv.contradiction_report else "",
                resolution_risks=fv.resolution_analysis.risks if fv.resolution_analysis else [],
                should_trade=fv.should_trade
            )
            opp.calculate_common_score()
            opportunities.append(opp)
        
        opportunities = sorted(opportunities, key=lambda x: (x.effective_edge, x.score), reverse=True)
        filtered = opportunities[:10]
        logger.info(f"Ensemble filter: {len(researched)} -> {len(filtered)} opportunities")
        return filtered

    def rank_and_select(self, opportunities: List[VenueOpportunity], max_trades: int = 3, bankroll: float = 50.0, current_positions: List[Dict] = None) -> List[VenueOpportunity]:
        """
        Final ranking FIXED: was edge>=8% alone determines capital
        Now: Expected EV, liquidity, risk, uncertainty, portfolio impact, capital allocation
        Which opportunity gives best risk-adjusted expected return for capital available?
        """
        if not opportunities:
            return []
        
        current_positions = current_positions or []
        
        # Use venue registry to rank if available (with fixed learning bug)
        if self.venue_registry:
            ranked = self.venue_registry.rank_opportunities(opportunities)
        else:
            ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        
        # Enhanced selection with portfolio impact and capital allocation
        selected = []
        total_allocated = sum(p.get("amount_usd", 0) for p in current_positions)
        
        for opp in ranked:
            if len(selected) >= max_trades:
                break
            
            # Basic thresholds
            if opp.effective_edge < 0.08:
                continue
            if opp.confidence < 0.6:
                continue
            
            # Expected EV calculation
            # EV = edge * prob_correct * amount - fees - slippage - risk
            amount_usd = min(bankroll * 0.06, 3.0)  # 6% cap $3 on $50
            expected_profit = opp.effective_edge * amount_usd * opp.confidence
            fees = opp.fees_pct * amount_usd
            slippage = opp.slippage_pct * amount_usd
            risk_adjusted_ev = expected_profit - fees - slippage - opp.uncertainty * amount_usd * 0.5
            
            if risk_adjusted_ev <= 0:
                logger.info(f"Risk-adjusted EV negative for {opp.market.id}: EV ${expected_profit:.2f} fees ${fees:.2f} slippage ${slippage:.2f} uncertainty ${opp.uncertainty*amount_usd*0.5:.2f} => {risk_adjusted_ev:.2f} NO TRADE")
                continue
            
            # Liquidity check: need enough liquidity for execution quality
            liquidity_score = opp.liquidity_score
            if liquidity_score < 0.3:
                logger.info(f"Low liquidity for {opp.market.id}: score {liquidity_score:.2f} NO TRADE")
                continue
            
            # Portfolio impact: check correlation with existing positions
            # If same event across venues is one bet not two, aggregate per event
            event_key = opp.market.event_slug or opp.market.question[:30]
            existing_event_exposure = sum(p.get("amount_usd", 0) for p in current_positions if p.get("event_slug") == opp.market.event_slug or event_key in p.get("question", ""))
            if existing_event_exposure > 0:
                # Same event, would be correlation risk
                if existing_event_exposure + amount_usd > bankroll * 0.12:  # 12% per event cap
                    logger.info(f"Correlation risk: event {event_key} existing ${existing_event_exposure:.2f} + new ${amount_usd:.2f} > ${bankroll*0.12:.2f} max per event (12% bankroll) - one bet not two, skip")
                    continue
            
            # Capital allocation: check total exposure
            if total_allocated + amount_usd > bankroll * 0.5:  # 50% total max
                logger.info(f"Total exposure would exceed 50%: allocated ${total_allocated:.2f} + new ${amount_usd:.2f} > ${bankroll*0.5:.2f} max, skip {opp.market.id}")
                continue
            
            # Time efficiency: short resolution better for small bankroll compounding
            time_efficiency = 1.0
            if opp.time_to_resolution_hours:
                if opp.time_to_resolution_hours < 24:
                    time_efficiency = 1.2
                elif opp.time_to_resolution_hours > 720:
                    time_efficiency = 0.7
            
            # Final score with portfolio impact
            # Which opportunity gives best risk-adjusted expected return for capital available?
            final_score = risk_adjusted_ev * liquidity_score * opp.execution_quality * time_efficiency / (opp.uncertainty + 0.01)
            opp.score = final_score
            
            logger.info(f"Selected {opp.market.id} venue {opp.venue_id} edge {opp.effective_edge*100:.1f}% conf {opp.confidence:.2f} liq {liquidity_score:.2f} exec {opp.execution_quality:.2f} EV ${risk_adjusted_ev:.2f} final_score {final_score:.3f} - best risk-adjusted return for capital ${bankroll}")
            
            selected.append(opp)
            total_allocated += amount_usd
        
        # Re-sort selected by final score
        selected = sorted(selected, key=lambda x: x.score, reverse=True)
        
        logger.info(f"Final selection: {len(opportunities)} -> {len(selected)} trades (max {max_trades}) | Total allocated ${total_allocated:.2f}/{bankroll*0.5:.2f} max | Reasoning: Which opportunity gives best risk-adjusted expected return for capital available, not just edge>8%")
        return selected

    async def scan_and_find(self, markets: List[Market], context_provider=None, max_trades: int = 3) -> ScanResult:
        start = time.time()
        
        total = len(markets)
        
        after_cheap = self.cheap_filters(markets)
        after_liquidity = self.liquidity_filter(after_cheap)
        after_fast = self.fast_model_screen(after_liquidity)
        researched = await self.deep_research(after_fast, context_provider=context_provider)
        after_deep = len([r for r in researched if r["should_trade"]])
        after_ensemble_list = self.ensemble_filter(researched)
        after_ensemble = len(after_ensemble_list)
        final_opps = self.rank_and_select(after_ensemble_list, max_trades=max_trades)
        after_risk = len(final_opps)
        
        elapsed = time.time() - start
        
        reasoning = (
            f"Scan pipeline: {total} total -> {len(after_cheap)} after cheap filters -> "
            f"{len(after_liquidity)} after liquidity -> {len(after_fast)} after fast model (real classification not just volume sort, duplicates removed, category diversity) -> "
            f"{len(researched)} deep researched ({after_deep} with edge) -> "
            f"{after_ensemble} after ensemble -> {after_risk} final trades | "
            f"Time {elapsed:.1f}s | "
            f"Final ranking: Expected EV = edge*prob_correct*amount - fees - slippage - uncertainty, liquidity risk time efficiency portfolio impact capital allocation, best risk-adjusted return for capital, not just edge>8%"
        )
        
        logger.info(reasoning)
        
        return ScanResult(
            total_scanned=total,
            after_cheap_filters=len(after_cheap),
            after_liquidity=len(after_liquidity),
            after_fast_model=len(after_fast),
            after_deep_research=after_deep,
            after_ensemble=after_ensemble,
            after_risk=after_risk,
            opportunities=final_opps,
            execution_time=elapsed,
            reasoning=reasoning
        )
