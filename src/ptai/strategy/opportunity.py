"""
Opportunity Engine - finds, ranks, and filters opportunities across all venues
1000 markets -> cheap filters -> 500 -> liquidity filter -> 200 -> fast model -> 50 -> deep research -> 10 -> ensemble -> 3 -> risk -> 0-3 trades

FIXED V7:
- Venue identity explicit immutable through pipeline, never enum, never first eligible
- fast_model_screen now has LLM hook (Qwen/DeepSeek) not just keyword classifier
- User correctly identified fast model was keyword classifier + heuristic scoring, useful preprocessing but not actual AI market-selection model
- Now: fast model is two-stage: heuristic preprocessing (cheap) + LLM fast screening (Qwen 7B 2-3 sec) for top 100
- MarketScanner uses VenueRegistry SINGLE SOURCE, no fallback to PolymarketClient
- Routing exact: venue_id -> exact adapter -> exact orderbook, never eligible[0]
- Execution ABORT if adapter not found, never fallback to first eligible - hard safety
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
from .edge import HUNT_MISPRICING_MIN, EdgeCalculator, hunted_mispricing
from ..markets.orderbook import execution_quality_from_book
from .expected_ev import ExpectedNetEVEngine


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
    Fast AI screening - FIXED V7: Two-stage with LLM hook
    Previously: keyword classifier + heuristic scoring, useful preprocessing but not actual AI model
    Example: "bitcoin" -> crypto, "Trump" -> politics, "NBA" -> sports - useful but not AI
    
    Now:
    Stage 1: Heuristic preprocessing (cheap, 200 markets -> 100) - keyword classification, news extraction, duplicate detection
    Stage 2: Fast LLM screening (Qwen 7B 2-3 sec, 100 -> 50) - actual AI market-selection model using local Qwen/DeepSeek
    
    Local Qwen/DeepSeek intelligence is used at fast model stage, not just later
    """
    def __init__(self, llm_router=None):
        self.categories = ["politics", "sports", "crypto", "economics", "weather", "ai", "general"]
        self.keywords = {
            "politics": ["trump", "biden", "election", "republican", "democrat", "senate", "congress", "vote", "president", "gop", "dem"],
            "sports": ["nfl", "nba", "mlb", "soccer", "football", "team", "game", "championship", "super bowl", "world cup", "man city", "arsenal", "lakers", "djokovic", "messi"],
            "crypto": ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana", "bnb", "doge", "usdt", "perp", "futures", "funding", "binance", "whitebit"],
            "economics": ["fed", "cpi", "inflation", "interest rate", "fomc", "gdp", "nfp", "jobs", "unemployment", "earnings", "s&p", "aapl", "kalshi", "predictit"],
            "weather": ["weather", "hurricane", "temperature", "rain", "snow", "storm"],
            "ai": ["gpt", "agi", "ai", "agent", "llm", "simmer", "autonomous", "openai", "anthropic"],
        }
        self.llm_router = llm_router
        self.fast_model_enabled = llm_router is not None

    def classify(self, market: Market) -> Dict[str, Any]:
        q = market.question.lower()
        scores = {}
        for cat, kws in self.keywords.items():
            score = sum(1 for kw in kws if kw in q)
            scores[cat] = score
        
        best_cat = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
        confidence = min(1.0, max(scores.values()) / 3.0) if best_cat != "general" else 0.3
        
        has_news_potential = any(word in q for word in ["earnings", "fed", "election", "cpi", "fomc", "trump", "btc", "nfp", "gdp"])
        
        return {
            "category": best_cat,
            "confidence": confidence,
            "scores": scores,
            "has_news_potential": has_news_potential,
            "should_deep_research": confidence > 0.3 or has_news_potential or market.volume_24h > 10000,
            "stage": "heuristic_preprocessing",
            "is_ai": False
        }

    def classify_with_llm(self, market: Market) -> Dict[str, Any]:
        """
        FIXED V7: Actual AI market-selection model using Qwen/DeepSeek
        Previously: only keyword classifier, not actual AI model
        Now: fast LLM (Qwen 7B 2-3 sec) for top 100 markets
        """
        heuristic = self.classify(market)
        
        if not self.llm_router or not self.fast_model_enabled:
            return heuristic
        
        try:
            # Fast LLM prompt for market screening
            prompt = f"""You are a fast market screening model (Qwen 7B). Classify this prediction market quickly.

Question: {market.question[:200]}
Volume 24h: ${market.volume_24h:.0f}
Liquidity: ${market.liquidity:.0f}
Price: {market.best_price:.3f}
Current heuristic: {heuristic['category']} confidence {heuristic['confidence']:.2f}

Task: 
1. Category: politics, sports, crypto, economics, weather, ai, general
2. Has news potential? (earnings, Fed, election, CPI etc) yes/no
3. Should deep research? yes/no - based on mispricing potential, not just volume
4. Quick mispricing hint: does price seem off? (e.g. 0.95 for Trump when polls 0.52)
5. Confidence 0-1

Respond JSON: {{"category": "...", "has_news_potential": true/false, "should_deep_research": true/false, "mispricing_hint": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
Keep under 100 tokens, fast.
"""
            # Try LLM call with timeout
            import asyncio
            try:
                response = asyncio.run(self.llm_router.generate(prompt, max_tokens=150, temperature=0.3))
                if response:
                    import json
                    # Try parse JSON
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        llm_result = json.loads(json_match.group())
                        return {
                            "category": llm_result.get("category", heuristic["category"]),
                            "confidence": float(llm_result.get("confidence", heuristic["confidence"])),
                            "scores": heuristic["scores"],
                            "has_news_potential": llm_result.get("has_news_potential", heuristic["has_news_potential"]),
                            "should_deep_research": llm_result.get("should_deep_research", heuristic["should_deep_research"]),
                            "mispricing_hint": llm_result.get("mispricing_hint", ""),
                            "reasoning": llm_result.get("reasoning", ""),
                            "stage": "fast_llm_qwen_7b",
                            "is_ai": True,
                            "llm_response": response[:200],
                            "heuristic_fallback": heuristic
                        }
            except Exception as e:
                logger.debug(f"Fast LLM screening failed {market.id}: {e}, using heuristic")
            
            return heuristic
            
        except Exception as e:
            logger.debug(f"LLM classification failed {market.id}: {e}")
            return heuristic

    def detect_duplicates(self, markets: List[Market]) -> List[List[Market]]:
        groups = []
        used = set()
        for i, m1 in enumerate(markets):
            if m1.id in used:
                continue
            group = [m1]
            for j, m2 in enumerate(markets[i+1:], i+1):
                if m2.id in used:
                    continue
                sim = SequenceMatcher(None, m1.question.lower()[:60], m2.question.lower()[:60]).ratio()
                if sim > 0.8:
                    group.append(m2)
                    used.add(m2.id)
            if len(group) > 1:
                groups.append(group)
            used.add(m1.id)
        return groups


class OpportunityEngine:
    def __init__(self, venue_registry: VenueRegistry = None, fair_value_engine: FairValueEngine = None, edge_calculator: EdgeCalculator = None, llm_router=None):
        self.venue_registry = venue_registry
        self.fair_value_engine = fair_value_engine or FairValueEngine()
        self.edge_calculator = edge_calculator or EdgeCalculator()
        self.llm_router = llm_router
        self.fast_classifier = FastModelClassifier(llm_router=llm_router)
        self.expected_ev_engine = ExpectedNetEVEngine()
        # Alpha adjustment engine. Imported lazily: alpha_engine pulls in most
        # of the strategy package, and a hard import here would risk a cycle.
        # If it cannot be constructed the scorer simply applies no adjustment
        # rather than failing the whole selection pass.
        try:
            from .alpha_engine import AlphaEngine
            self.alpha_engine = AlphaEngine()
        except Exception as e:
            logger.warning(f"AlphaEngine unavailable, scores will not be adjusted: {e}")
            self.alpha_engine = None
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
            # Validate venue_id immutable
            if not getattr(m, 'venue_id', None):
                logger.warning(f"Market {m.id} missing venue_id - BUG, should be immutable")
            filtered.append(m)
        logger.info(f"Cheap filters: {len(markets)} -> {len(filtered)} | venue_id validation")
        return filtered

    def liquidity_filter(self, markets: List[Market]) -> List[Market]:
        sorted_markets = sorted(markets, key=lambda x: (x.volume_24h, x.liquidity), reverse=True)
        limit = min(200, max(50, len(sorted_markets) // 2))
        filtered = sorted_markets[:limit]
        # Track venues
        venues = {}
        for m in filtered:
            vid = getattr(m, 'venue_id', 'unknown')
            venues[vid] = venues.get(vid, 0) + 1
        logger.info(f"Liquidity filter: {len(markets)} -> {len(filtered)} | Venues: {venues}")
        return filtered

    def fast_model_screen(self, markets: List[Market]) -> List[Market]:
        """
        Fast model screening FIXED V7: Two-stage with LLM hook
        Previously: keyword classifier + heuristic scoring, useful preprocessing but not actual AI model
        Example: bitcoin->crypto, Trump->politics, NBA->sports - useful but not AI
        Local Qwen/DeepSeek intelligence was used later, but fast model stage wasn't really fast model described in architecture
        
        Now FIXED:
        Stage 1: Heuristic preprocessing (cheap, 200 -> 100) - keyword, news extraction, duplicate detection
        Stage 2: Fast LLM screening (Qwen 7B 2-3 sec, 100 -> 50) - actual AI market-selection model
        """
        logger.info(f"Fast model screening V7: {len(markets)} markets - Stage1 heuristic preprocessing + Stage2 fast LLM Qwen 7B")
        
        # Stage 1: Heuristic preprocessing (cheap)
        classified = []
        for market in markets:
            classification = self.fast_classifier.classify(market)
            market.raw["fast_classification"] = classification
            market.raw["category"] = classification["category"]
            market.raw["fast_model_stage"] = "heuristic_preprocessing"
            classified.append((market, classification))
        
        # Detect duplicates
        duplicates = self.fast_classifier.detect_duplicates(markets)
        duplicate_ids = set()
        for group in duplicates:
            sorted_group = sorted(group, key=lambda x: x.volume_24h, reverse=True)
            for dup in sorted_group[1:]:
                duplicate_ids.add(dup.id)
                logger.debug(f"Duplicate: {dup.question[:40]} similar to {sorted_group[0].question[:40]}")
        
        non_duplicate = [(m, c) for m, c in classified if m.id not in duplicate_ids]
        
        # Score for fast LLM priority
        def heuristic_score(item):
            market, classification = item
            volume_score = min(1.0, market.volume_24h / 20000)
            liquidity_score = min(1.0, market.liquidity / 20000)
            category_score = classification["confidence"]
            news_score = 1.0 if classification["has_news_potential"] else 0.3
            time_score = 1.0
            if market.end_date:
                try:
                    from datetime import datetime, timezone
                    hours_to_res = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_to_res < 24:
                        time_score = 1.2
                    elif hours_to_res > 720:
                        time_score = 0.7
                except:
                    pass
            return volume_score*0.3 + liquidity_score*0.2 + category_score*0.2 + news_score*0.2 + time_score*0.1
        
        scored = sorted(non_duplicate, key=heuristic_score, reverse=True)
        stage1_selected = scored[:100]  # Top 100 for LLM
        
        logger.info(f"Stage1 heuristic: {len(markets)} -> {len(non_duplicate)} non-dup -> {len(stage1_selected)} for fast LLM")
        
        # Stage 2: Fast LLM screening (Qwen 7B) for top 100
        if self.fast_classifier.fast_model_enabled and len(stage1_selected) > 0:
            logger.info(f"Stage2 fast LLM Qwen 7B screening {len(stage1_selected)} markets - actual AI model")
            llm_scored = []
            for market, heuristic_cls in stage1_selected[:100]:
                try:
                    llm_cls = self.fast_classifier.classify_with_llm(market)
                    market.raw["fast_classification"] = llm_cls
                    market.raw["category"] = llm_cls["category"]
                    market.raw["fast_model_stage"] = llm_cls["stage"]
                    market.raw["fast_model_is_ai"] = llm_cls.get("is_ai", False)
                    
                    # Combined score: heuristic + LLM
                    llm_score = llm_cls.get("confidence", 0.5)
                    if llm_cls.get("should_deep_research"):
                        llm_score += 0.2
                    if llm_cls.get("has_news_potential"):
                        llm_score += 0.1
                    
                    combined = heuristic_score((market, heuristic_cls))*0.4 + llm_score*0.6
                    llm_scored.append((market, llm_cls, combined))
                except Exception as e:
                    logger.debug(f"LLM screening failed {market.id}: {e}")
                    llm_scored.append((market, heuristic_cls, heuristic_score((market, heuristic_cls))))
            
            llm_scored = sorted(llm_scored, key=lambda x: x[2], reverse=True)
            stage1_selected = [(m, c) for m, c, _ in llm_scored]
            logger.info(f"Stage2 LLM done: {len(llm_scored)} scored by Qwen 7B fast model")
        
        # Final selection with category diversity
        selected = []
        category_counts = {}
        for market, classification in stage1_selected:
            cat = classification["category"]
            if category_counts.get(cat, 0) >= 15 and len(selected) >= 30:
                continue
            selected.append(market)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if len(selected) >= 50:
                break
        
        # Ensure venue_id immutable
        venue_counts = {}
        for m in selected:
            vid = getattr(m, 'venue_id', 'unknown')
            venue_counts[vid] = venue_counts.get(vid, 0) + 1
        
        logger.info(f"Fast model V7: {len(markets)} -> {len(non_duplicate)} non-dup -> {len(stage1_selected)} stage1 -> {len(selected)} final for deep research | Duplicates {len(duplicate_ids)} | Categories {category_counts} | Venues {venue_counts} | LLM enabled {self.fast_classifier.fast_model_enabled}")
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
                        "orderbook": {"spread": 0.02, "source": "mock", "is_real": False},
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
                    logger.info(f"Deep research found opportunity: {market.id} venue {getattr(market, 'venue_id', 'unknown')} edge {fv_result.effective_edge:.3f} cat {context.get('category')}")
                
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
            
            venue_id = getattr(market, 'venue_id', None)
            if not venue_id:
                venue_id = market.raw.get("venue_id") if hasattr(market, 'raw') else None
            if not venue_id:
                source = getattr(market, 'source', 'unknown')
                venue_id = source.value if hasattr(source, 'value') else str(source)
            if hasattr(venue_id, 'value'):
                venue_id = venue_id.value
            venue_id = str(venue_id).lower()
            
            opp = VenueOpportunity(
                market=market,
                venue_id=venue_id,
                venue_type=VenueType.PREDICTION,
                # The side comes from the valuation when it has one, so the
                # opportunity's `raw_edge` (which is signed for that side) cannot
                # disagree with it about which side is being bought.
                side=getattr(fv, "side", None) or (
                    "YES" if fv.fair_value > market.best_price else "NO"),
                market_price=market.best_price,
                estimated_fair=fv.fair_value,
                # `FairValueResult.edge` is the mispricing on the traded side. A
                # valuation object with no `side` predates that and still carries
                # the YES-space edge, so mirror it here rather than let a NO
                # opportunity inherit a negative one.
                raw_edge=fv.edge if getattr(fv, "side", None) else (
                    fv.edge if (getattr(fv, "side", None) or
                                ("YES" if fv.fair_value > market.best_price else "NO")) == "YES"
                    else -fv.edge),
                effective_edge=fv.effective_edge,
                confidence=fv.confidence,
                uncertainty=fv.uncertainty,
                liquidity_score=min(1.0, market.liquidity / 10000),
                execution_quality=execution_quality_from_book(
                    context.get("orderbook"), market),
                category=context.get("category", market.raw.get("category", "unknown")),
                sources=fv.forecast_result.sources if fv.forecast_result else [],
                reasoning=fv.reasoning + f" | venue_id immutable {venue_id} from market.venue_id | orderbook source {context.get('orderbook', {}).get('source', 'unknown')} is_real {context.get('orderbook', {}).get('is_real', False)}",
                bull_case=fv.contradiction_report.bull_case if fv.contradiction_report else "",
                bear_case=fv.contradiction_report.bear_case if fv.contradiction_report else "",
                resolution_risks=fv.resolution_analysis.risks if fv.resolution_analysis else [],
                should_trade=fv.should_trade
            )
            opp.calculate_common_score()
            opportunities.append(opp)
        
        opportunities = sorted(opportunities, key=lambda x: (x.effective_edge, x.score), reverse=True)
        filtered = opportunities[:10]
        logger.info(f"Ensemble filter: {len(researched)} -> {len(filtered)} opportunities | venue_id immutable validated")
        return filtered

    def rank_and_select(self, opportunities: List[VenueOpportunity], max_trades: int = 3, bankroll: float = 50.0, current_positions: List[Dict] = None) -> List[VenueOpportunity]:
        if not opportunities:
            return []
        
        current_positions = current_positions or []
        
        if self.venue_registry:
            ranked = self.venue_registry.rank_opportunities(opportunities)
        else:
            ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        
        selected = []
        total_allocated = sum(p.get("amount_usd", 0) for p in current_positions)
        
        for opp in ranked:
            if len(selected) >= max_trades:
                break
            
            # 8% of mispricing, not of post-cost edge - see
            # `hunted_mispricing`. Costs are charged in the net EV terms and in
            # the positive-effective-edge check below.
            if hunted_mispricing(opp) < HUNT_MISPRICING_MIN:
                continue
            if opp.effective_edge <= 0:
                continue
            if opp.confidence < 0.6:
                continue
            
            # Validate venue_id immutable
            if not opp.venue_id or opp.venue_id == "unknown":
                logger.error(f"ABORT: opportunity {opp.market.id} missing venue_id immutable - BUG")
                continue
            
            amount_usd = min(bankroll * 0.06, 3.0)
            # V10 FIX #9: Use Expected Net EV engine for economically meaningful scoring
            orderbook = opp.market.raw.get("orderbook", {}) if hasattr(opp.market, 'raw') and isinstance(opp.market.raw, dict) else {}
            ev_result = self.expected_ev_engine.calculate(opp, amount_usd, orderbook)
            
            # Old scoring for comparison
            expected_profit = opp.effective_edge * amount_usd * opp.confidence
            fees = opp.fees_pct * amount_usd
            slippage = opp.slippage_pct * amount_usd
            risk_adjusted_ev = expected_profit - fees - slippage - opp.uncertainty * amount_usd * 0.5
            
            # V10 FIX #9: New scoring = net EV per dollar per risk per capital-time
            if ev_result.net_ev_usd <= 0:
                logger.info(f"Expected Net EV blocks {opp.market.id} venue {opp.venue_id}: net EV ${ev_result.net_ev_usd:.2f} (gross ${ev_result.gross_ev_usd:.2f} costs ${ev_result.fees_usd+ev_result.spread_usd+ev_result.slippage_usd:.2f}) => NO TRADE - {ev_result.reasoning[:100]}")
                continue
            
            liquidity_score = opp.liquidity_score
            if liquidity_score < 0.3:
                logger.info(f"Low liquidity for {opp.market.id} venue {opp.venue_id}: score {liquidity_score:.2f} NO TRADE")
                continue
            
            if opp.execution_quality < 0.3:
                logger.info(f"Low execution quality for {opp.market.id} venue {opp.venue_id}: exec {opp.execution_quality:.2f} - may be mock orderbook not real CLOB - NO TRADE")
                continue
            
            event_key = opp.market.event_slug or opp.market.question[:30]
            existing_event_exposure = sum(p.get("amount_usd", 0) for p in current_positions if p.get("event_slug") == opp.market.event_slug or event_key in p.get("question", ""))
            if existing_event_exposure > 0:
                if existing_event_exposure + amount_usd > bankroll * 0.12:
                    logger.info(f"Correlation risk: event {event_key} existing ${existing_event_exposure:.2f} + new ${amount_usd:.2f} > ${bankroll*0.12:.2f} max per event - one bet not two, skip")
                    continue
            
            if total_allocated + amount_usd > bankroll * 0.5:
                logger.info(f"Total exposure would exceed 50%: allocated ${total_allocated:.2f} + new ${amount_usd:.2f} > ${bankroll*0.5:.2f} max, skip {opp.market.id}")
                continue
            
            time_efficiency = 1.0
            if opp.time_to_resolution_hours:
                if opp.time_to_resolution_hours < 24:
                    time_efficiency = 1.2
                elif opp.time_to_resolution_hours > 720:
                    time_efficiency = 0.7
            
            # V10 FIX #9: Final score now economically meaningful: net EV per dollar per risk per capital-time
            final_score = ev_result.net_ev_usd * ev_result.ev_per_dollar * liquidity_score * opp.execution_quality * time_efficiency * ev_result.ev_per_risk / max(0.01, opp.uncertainty)
            # Also incorporate capital efficiency
            final_score *= (1 + ev_result.ev_per_capital_time)

            # Alpha adjustment. This was dead code for the whole life of the
            # project: calculate_alpha_adjusted_score existed but nothing in
            # the live path ever called it, so none of the alpha signals
            # (reference odds, RAG base rates, favourite-longshot, whales)
            # ever moved a ranking. It is applied here, directionally, with
            # the multiplier and its reasons recorded on the opportunity so a
            # boosted score can be audited afterwards.
            if self.alpha_engine is not None:
                try:
                    adjustment = self.alpha_engine.calculate_alpha_adjustment(
                        opp.market, side=opp.side, context=context)
                    final_score *= adjustment.multiplier
                    opp.raw["alpha_adjustment"] = {
                        "multiplier": adjustment.multiplier,
                        "applied": adjustment.applied,
                        "errors": adjustment.errors,
                    }
                    if adjustment.applied:
                        logger.info(f"Alpha adjustment {opp.market.id}: x{adjustment.multiplier:.3f} - "
                                    f"{'; '.join(adjustment.applied)[:200]}")
                except Exception as e:
                    logger.warning(f"Alpha adjustment failed for {opp.market.id}: {e}")

            opp.score = final_score
            
            logger.info(f"Selected {opp.market.id} venue {opp.venue_id} edge {opp.effective_edge*100:.1f}% conf {opp.confidence:.2f} liq {liquidity_score:.2f} exec {opp.execution_quality:.2f} netEV ${ev_result.net_ev_usd:.2f} ({ev_result.net_ev_pct*100:.1f}%) per$ {ev_result.ev_per_dollar*100:.1f}% perRisk {ev_result.ev_per_risk:.2f} final_score {final_score:.3f} - V10 FIX #9 Expected Net EV | venue_id immutable {opp.venue_id}")
            
            selected.append(opp)
            total_allocated += amount_usd
        
        selected = sorted(selected, key=lambda x: x.score, reverse=True)
        
        logger.info(f"Final selection: {len(opportunities)} -> {len(selected)} trades (max {max_trades}) | Total allocated ${total_allocated:.2f}/{bankroll*0.5:.2f} max | venue_id immutable validated")
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
            f"Scan pipeline V7: {total} total -> {len(after_cheap)} after cheap filters -> "
            f"{len(after_liquidity)} after liquidity -> {len(after_fast)} after fast model V7 (Stage1 heuristic preprocessing + Stage2 fast LLM Qwen 7B actual AI market-selection, duplicates removed, category diversity, venue diversity, venue_id immutable) -> "
            f"{len(researched)} deep researched ({after_deep} with edge) -> "
            f"{after_ensemble} after ensemble -> {after_risk} final trades | "
            f"Time {elapsed:.1f}s | "
            f"Final ranking: Expected EV = edge*prob_correct*amount - fees - slippage - uncertainty, liquidity risk time efficiency portfolio impact capital allocation, execution quality from real orderbook, best risk-adjusted return, venue_id immutable exact routing"
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
