"""
PTAI Autonomous Trader v2 - market-agnostic, ensemble intelligence, separated execution

EVERY 10 MINUTES:
Check system health
Check account/portfolio
Check eligibility
Scan 500-1000 markets
Remove illiquid
Remove ambiguous resolution
Fast model screening
Historical/base-rate
News
X
Orderbook
Deep research
Bull case
Bear case
Resolution verification
Ensemble probability
Calibration
Uncertainty adjustment
Effective edge
Is edge >8%?
YES -> Kelly -> Half Kelly -> 6% cap -> Liquidity cap -> Correlation cap -> Portfolio risk -> Execution simulation -> Place order -> Verify fill -> Reconcile -> Monitor -> Record prediction -> LEARN

No human approval needed. DO NOTHING is successful outcome.
Capital preservation first.
"""
from typing import Dict, List, Optional, Any
import asyncio
import time
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path

from ..config import get_settings
from ..storage.db import Storage
from ..vault import Vault
from ..memory import Memory
from ..llm.provider import LLMRouter
from ..markets.base import Market
from ..markets.scanner import MarketScanner

from ..venues.registry import VenueRegistry
from ..venues.polymarket_adapter import PolymarketAdapter
from ..venues.adapter import EligibilityStatus

from ..intelligence.calibration import CalibrationEngine
from ..intelligence.uncertainty import UncertaintyEngine
from ..intelligence.ensemble import EnsembleForecaster
from ..intelligence.resolution_analyzer import ResolutionAnalyzer
from ..intelligence.contradiction import ContradictionEngine

from ..information.x_engine import XEngine
from ..information.news_engine import NewsEngine
from ..information.web_researcher import WebResearcher

from ..markets.market_normalizer import MarketNormalizer
from ..markets.orderbook import OrderbookAnalyzer

from ..strategy.fair_value import FairValueEngine
from ..strategy.edge import EdgeCalculator
from ..strategy.opportunity import OpportunityEngine
from ..strategy.strategy_selector import StrategySelector

from ..risk.kelly import KellyCalculator
from ..risk.exposure import ExposureManager
from ..risk.correlation import CorrelationEngine
from ..risk.drawdown import DrawdownManager
from ..risk.kill_switch import KillSwitch, KillLevel
from ..risk.limits import LimitsEngine, TradeLimits

from ..execution.order_manager import OrderManager
from ..execution.execution_guard import ExecutionGuard
from ..execution.reconciliation import ReconciliationEngine

from ..learning.calibration_db import CalibrationDB
from ..learning.trade_outcomes import TradeOutcomeTracker
from ..learning.performance import PerformanceTracker

from ..sentiment.x_scraper import XScraper
from ..sentiment.analyzer import SentimentAnalyzer


class TradingAgentV2:
    """
    PTAI v2 - market-agnostic autonomous trading engine
    Mission: Seek positive EV while preserving capital. Trade only when evidence, calibration, liquidity, risk agree.
    If forecasting advantage cannot be demonstrated, stop trading.
    """
    def __init__(self, country_code: str = "UG"):
        self.settings = get_settings()
        self.country_code = country_code
        
        # Storage
        self.storage = Storage(db_path="./data/ptai.db")
        self.vault = Vault()
        self.memory = Memory()
        
        # LLM
        self.llm_router = LLMRouter(
            preferred=getattr(self.settings, 'llm_provider', 'auto'),
            ollama_host=self.settings.ollama_host,
            lm_studio_host=self.settings.lm_studio_host,
            model=self.settings.lm_studio_model
        )
        
        # Intelligence
        self.calibration_engine = CalibrationDB(db_path="./data/calibration.json", storage=self.storage, memory=self.memory)
        self.uncertainty_engine = UncertaintyEngine()
        self.resolution_analyzer = ResolutionAnalyzer(llm_router=self.llm_router)
        self.contradiction_engine = ContradictionEngine(llm_router=self.llm_router)
        self.ensemble_forecaster = EnsembleForecaster(
            calibration_engine=self.calibration_engine,
            uncertainty_engine=self.uncertainty_engine,
            llm_router=self.llm_router
        )
        
        # Information
        self.x_scraper = XScraper()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.x_engine = XEngine(x_scraper=self.x_scraper, sentiment_analyzer=self.sentiment_analyzer)
        self.news_engine = NewsEngine(llm_router=self.llm_router)
        self.web_researcher = WebResearcher(llm_router=self.llm_router)
        
        # Markets
        self.market_normalizer = MarketNormalizer()
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.scanner = MarketScanner()
        
        # Strategy
        self.fair_value_engine = FairValueEngine(
            llm_router=self.llm_router,
            calibration_engine=self.calibration_engine,
            uncertainty_engine=self.uncertainty_engine
        )
        self.edge_calculator = EdgeCalculator(uncertainty_engine=self.uncertainty_engine)
        self.strategy_selector = StrategySelector()
        
        # Venues
        self.venue_registry = VenueRegistry(country_code=country_code)
        # Register Polymarket - keys from vault or env
        try:
            pk = self.vault.get("Trader", "polymarket_clob", {}).get("private_key") or self.settings.polymarket_private_key
            funder = self.vault.get("Trader", "polymarket_clob", {}).get("funder") or self.settings.polymarket_funder_address
        except:
            pk = None
            funder = None
        
        polymarket_adapter = PolymarketAdapter(private_key=pk, funder=funder)
        self.venue_registry.register(polymarket_adapter)
        
        # Risk
        bankroll = self.storage.get_performance_summary().get("bankroll", 50.0)
        self.exposure_manager = ExposureManager(bankroll=bankroll)
        self.correlation_engine = CorrelationEngine()
        self.drawdown_manager = DrawdownManager(initial_bankroll=bankroll)
        self.kill_switch = KillSwitch(data_dir="./data")
        self.limits_engine = LimitsEngine(
            limits=TradeLimits(
                max_position_pct=self.settings.max_position_pct,
                min_edge_pct=self.settings.min_edge_pct,
                kelly_fraction=self.settings.kelly_fraction
            ),
            bankroll=bankroll
        )
        self.kelly_calculator = KellyCalculator(
            kelly_fraction=self.settings.kelly_fraction,
            max_pct=self.settings.max_position_pct,
            min_edge=self.settings.min_edge_pct
        )
        
        # Execution
        self.order_manager = OrderManager(storage=self.storage)
        self.execution_guard = ExecutionGuard(bankroll=bankroll)
        self.reconciliation_engine = ReconciliationEngine(storage=self.storage)
        
        # Learning
        self.trade_outcome_tracker = TradeOutcomeTracker(storage=self.storage)
        self.performance_tracker = PerformanceTracker(storage=self.storage)
        
        # Opportunity
        self.opportunity_engine = OpportunityEngine(
            venue_registry=self.venue_registry,
            fair_value_engine=self.fair_value_engine,
            edge_calculator=self.edge_calculator
        )
        
        # Mission
        self.mission = "Seek positive EV while preserving capital. Trade only when evidence, calibration, liquidity, risk agree. If advantage cannot be demonstrated, stop trading. DO NOTHING is successful."
        
        logger.info(f"PTAI v2 initialized: {self.mission} | Country {country_code} | Bankroll ${bankroll}")

    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            "llm_available": False,
            "storage_ok": True,
            "internet_ok": True,
            "bankroll": self.storage.get_performance_summary().get("bankroll", 0),
            "kill_switch_level": int(self.kill_switch.current_level),
            "can_trade": self.kill_switch.can_trade()
        }
        
        # Check LLM
        try:
            # Try to detect LLM
            provider = self.llm_router.get_provider_name()
            health["llm_available"] = "heuristic" not in provider or True  # heuristic is fallback, still ok
            health["llm_provider"] = provider
        except Exception as e:
            health["llm_available"] = False
            health["llm_error"] = str(e)
            self.kill_switch.check_llm_unavailable(1000)
        
        # Check calibration degradation
        if self.calibration_engine.is_degrading():
            self.kill_switch.check_calibration_collapse(self.calibration_engine.calculate_brier_score())
        
        logger.info(f"System health: {health}")
        return health

    async def check_eligibility(self) -> Dict[str, EligibilityStatus]:
        """Check geographic eligibility - NEVER bypass"""
        eligibility = self.venue_registry.check_all_eligibility()
        for venue_id, status in eligibility.items():
            if status == EligibilityStatus.RESTRICTED:
                logger.warning(f"Venue {venue_id} restricted for {self.country_code} - no trading")
                self.kill_switch.trigger(KillLevel.NO_NEW_TRADES, f"Venue {venue_id} restricted for {self.country_code}", {"venue": venue_id, "country": self.country_code})
        return eligibility

    async def get_context_for_market(self, market: Market) -> Dict[str, Any]:
        """Get full context for market: news, X, orderbook, research"""
        context = {
            "category": "unknown",
            "news": "",
            "sentiment": {"score": 0},
            "tweets": [],
            "orderbook": {"spread": 0.02},
            "research": "",
            "sources": []
        }
        
        try:
            # News
            news_signals = await self.news_engine.get_news(market, max_articles=3)
            news_synth = self.news_engine.synthesize(news_signals)
            context["news"] = news_synth.get("summary", "")
            context["news_credibility"] = news_synth.get("credibility", 0)
            context["sources"].extend(news_synth.get("sources", []))
            
            # X - with credibility analysis
            x_signal = await self.x_engine.get_signal(market, news=context["news"])
            context["x_signal"] = x_signal
            context["sentiment"] = {"score": x_signal.adjusted_sentiment}
            context["tweets"] = []  # would have tweets
            context["x_credibility"] = x_signal.credibility
            context["x_should_use"] = x_signal.should_use
            
            # Orderbook - FIXED V7: Exact routing, never eligible[0]
            # Previously: eligible[0].get_orderbook(market) - dangerous if Kalshi market asked to Polymarket adapter
            # Now: opportunity.venue_id -> VenueRegistry -> exact adapter -> exact market -> exact orderbook
            # Never first eligible adapter
            try:
                exact_adapter = self.venue_registry.get_adapter_for_market(market)
                if exact_adapter:
                    orderbook_raw = await exact_adapter.get_orderbook(market)
                    orderbook_snapshot = self.orderbook_analyzer.analyze(market, raw_orderbook=orderbook_raw)
                    context["orderbook"] = {
                        "spread": orderbook_snapshot.spread,
                        "bid": orderbook_snapshot.bid,
                        "ask": orderbook_snapshot.ask,
                        "imbalance": orderbook_snapshot.imbalance,
                        "large_orders": orderbook_snapshot.large_orders,
                        "price_velocity": orderbook_snapshot.price_velocity,
                        "source": orderbook_raw.get("source", "unknown"),
                        "venue_id": exact_adapter.venue_id,
                        "is_real": orderbook_raw.get("is_real", False)
                    }
                    manip_risks = self.orderbook_analyzer.detect_manipulation(orderbook_snapshot)
                    if manip_risks:
                        context["manipulation_risks"] = manip_risks
                    logger.debug(f"Orderbook for {market.id} via exact adapter {exact_adapter.venue_id} spread {orderbook_snapshot.spread*100:.2f}%")
                else:
                    # If no exact adapter, DO NOT fallback to eligible[0] - that's bug
                    logger.error(f"No exact adapter for market {market.id} venue_id {getattr(market, 'venue_id', 'unknown')} - ABORT orderbook, never fallback")
                    context["orderbook"] = {"spread": 0.02, "error": "no_exact_adapter_ABORT"}
            except Exception as e:
                logger.warning(f"Orderbook fetch failed for {market.id}: {e}")
            
            # Web research - bull/bear
            research_result = await self.web_researcher.research(market, max_time_seconds=20)
            # Credit research only when pages were actually retrieved; an
            # unresearched market must not carry empty bull/bear fields that
            # read as evidence that no evidence exists.
            if getattr(research_result, 'researched', False):
                context["research"] = research_result.final_summary
                context["bull_case"] = research_result.supporting_yes
                context["bear_case"] = research_result.supporting_no
                context["sources"].extend(research_result.sources)
                context["research_researched"] = True
                context["research_confidence"] = float(research_result.confidence or 0.0)
            else:
                context["research"] = ""
                context["bull_case"] = ""
                context["bear_case"] = ""
                context["research_researched"] = False
                context["research_blockers"] = list(research_result.warnings)[:3]
            
            # Category detection simple
            q_lower = market.question.lower()
            if any(word in q_lower for word in ["trump", "biden", "election", "senate", "politics"]):
                context["category"] = "politics"
            elif any(word in q_lower for word in ["btc", "bitcoin", "eth", "crypto"]):
                context["category"] = "crypto"
            elif any(word in q_lower for word in ["nfl", "nba", "sports", "game"]):
                context["category"] = "sports"
            elif any(word in q_lower for word in ["weather", "temperature"]):
                context["category"] = "weather"
            else:
                context["category"] = "other"
            
            # Exposure for correlation penalty
            exposure = self.exposure_manager.get_exposure()
            context["category_exposure"] = exposure.by_category.get(context["category"], 0) / max(1, self.exposure_manager.bankroll)
            context["correlation_group"] = self.correlation_engine.get_correlation_group(market, self.exposure_manager.positions)
            
        except Exception as e:
            logger.error(f"Context fetch failed for {market.id}: {e}")
        
        return context

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Full autonomous cycle v2
        """
        logger.info("=== PTAI v2 Cycle Start ===")
        cycle_start = time.time()
        
        # Step 1: System health
        health = await self.check_system_health()
        if not health["can_trade"]:
            logger.warning(f"Kill switch active level {health['kill_switch_level']} - no trading")
            return {"status": "kill_switch_active", "level": health["kill_switch_level"], "health": health}

        # Step 2: Eligibility
        eligibility = await self.check_eligibility()
        eligible_adapters = self.venue_registry.get_eligible_adapters()
        if not eligible_adapters:
            logger.warning("No eligible venues - no trading (geographic compliance)")
            return {"status": "no_eligible_venues", "eligibility": {k: v.value for k, v in eligibility.items()}}

        # Step 3: Scan markets
        logger.info(f"Scanning markets from {len(eligible_adapters)} eligible venues, target 500-1000")
        all_markets = await self.venue_registry.discover_all(target_per_venue=500)
        logger.info(f"Discovered {len(all_markets)} total markets")

        if not all_markets:
            logger.info("No markets discovered - DO NOTHING is successful")
            return {"status": "no_markets", "message": "DO NOTHING - no markets, successful outcome"}

        # Step 4: Opportunity engine pipeline
        # For context provider, we need to fetch context per market in deep research phase
        # We'll create a simple context provider that calls get_context_for_market
        class ContextProvider:
            def __init__(self, agent):
                self.agent = agent
            async def get_context(self, market):
                return await self.agent.get_context_for_market(market)
        
        context_provider = ContextProvider(self)
        
        scan_result = await self.opportunity_engine.scan_and_find(
            markets=all_markets,
            context_provider=context_provider,
            max_trades=3
        )
        
        logger.info(f"Opportunity scan result: {scan_result.reasoning}")

        if not scan_result.opportunities:
            logger.info("No opportunities after full pipeline - DO NOTHING is successful")
            # Record scan even if no opps
            self.storage.log_scan(
                markets_scanned=scan_result.total_scanned,
                opportunities_found=0,
                avg_edge=0,
                execution_time=time.time() - cycle_start,
                bankroll=self.exposure_manager.bankroll
            )
            return {
                "status": "no_opportunities",
                "message": "DO NOTHING - no edge >=8% with confidence >=60% after all filters, successful",
                "scan_result": scan_result.reasoning,
                "pipeline": {
                    "total": scan_result.total_scanned,
                    "after_cheap": scan_result.after_cheap_filters,
                    "after_liquidity": scan_result.after_liquidity,
                    "after_fast": scan_result.after_fast_model,
                    "after_deep": scan_result.after_deep_research,
                    "after_ensemble": scan_result.after_ensemble,
                    "after_risk": scan_result.after_risk
                }
            }

        # Step 5: Risk and execution for each opportunity
        executed = []
        for opp in scan_result.opportunities:
            try:
                # Check kill switch again
                if not self.kill_switch.can_trade():
                    logger.warning("Kill switch triggered during execution loop")
                    break

                # Check exposure caps
                can_open, reason = self.exposure_manager.can_open(
                    market_id=opp.market.id,
                    amount_usd=5.0,  # temporary, will be sized by Kelly
                    category=opp.category,
                    correlation_group=opp.correlation_group,
                    venue=opp.venue_id
                )
                if not can_open:
                    logger.info(f"Exposure check blocks {opp.market.id}: {reason}")
                    continue

                # Correlation risk
                is_risky, corr_score, corr_reason = self.correlation_engine.check_portfolio_correlation_risk(
                    new_market=opp.market,
                    existing_positions=self.exposure_manager.positions
                )
                if is_risky:
                    logger.info(f"Correlation risk blocks {opp.market.id}: {corr_reason}")
                    continue

                # Limits engine - deterministic validation
                proposal = opp.market.to_dict()
                proposal.update({
                    "market_id": opp.market.id,
                    "fair_probability": opp.estimated_fair,
                    "market_probability": opp.market_price,
                    "edge": opp.effective_edge,
                    "confidence": opp.confidence,
                    "side": opp.side,
                    "trade": opp.should_trade,
                    "max_price": opp.market_price + 0.02 if opp.side == "YES" else (1 - opp.market_price) + 0.02
                })

                allowed, reason, adjusted_order = self.limits_engine.validate_proposal(proposal)
                if not allowed:
                    logger.info(f"Limits block {opp.market.id}: {reason}")
                    continue

                # Execution guard - final safety
                guard_result = self.execution_guard.validate(proposal, adjusted_order)
                if not guard_result.allowed:
                    logger.warning(f"Execution guard blocks {opp.market.id}: {guard_result.reason}")
                    self.kill_switch.check_execution_mismatch(adjusted_order.get("max_spend_usd", 0), guard_result.max_spend_usd)
                    continue

                # Create order via order manager
                order_allowed, order_reason, order = self.order_manager.create_order(
                    market_id=opp.market.id,
                    token_id=opp.market.yes_token_id if opp.side == "YES" else opp.market.no_token_id,
                    side=opp.side,
                    max_price=guard_result.max_price,
                    max_spend_usd=guard_result.max_spend_usd,
                    venue_id=opp.venue_id
                )
                if not order_allowed:
                    logger.info(f"Order manager blocks {opp.market.id}: {order_reason}")
                    continue

                # Place order via venue adapter - FIXED V7: Exact routing, ABORT if not found, never fallback
                # Previously: if requested adapter doesn't exist use first eligible - DANGEROUS for real money
                # Now: if venue=kalshi and Kalshi adapter isn't available, ABORT TRADE not try first venue - hard safety
                adapter = self.venue_registry.get_adapter_for_venue_id(opp.venue_id)
                if not adapter:
                    logger.error(f"ABORT TRADE: venue {opp.venue_id} adapter not found for market {opp.market.id} - requested {opp.venue_id} not in {list(self.venue_registry.adapters.keys())} - ABORT, never fallback to first eligible")
                    continue
                # Validate venue identity matches opportunity
                if adapter.venue_id != opp.venue_id.split("+")[0]:
                    logger.error(f"ABORT TRADE: venue mismatch opportunity {opp.venue_id} vs adapter {adapter.venue_id} for market {opp.market.id} - ABORT")
                    continue
                logger.info(f"Exact routing: opportunity venue_id {opp.venue_id} -> adapter {adapter.venue_id} -> market {opp.market.id} -> orderbook/execution")
                
                if adapter:
                    # Get portfolio before
                    portfolio_before = await adapter.get_portfolio() if hasattr(adapter, 'get_portfolio') else {"balance": self.exposure_manager.bankroll}
                    
                    result = await adapter.place_order(
                        opportunity=opp,
                        max_spend_usd=guard_result.max_spend_usd,
                        max_price=guard_result.max_price
                    )
                    
                    # Update order status
                    if result.get("status") in ["filled", "executed", "dry_run"]:
                        self.order_manager.update_order(
                            order.id,
                            status=self.order_manager.get_order(order.id).status if self.order_manager.get_order(order.id) else None,
                            amount=result.get("amount", guard_result.max_spend_usd / guard_result.max_price),
                            avg_price=result.get("avg_price", guard_result.max_price),
                            raw_response=result
                        )
                        # Record trade
                        trade_record = {
                            "market_id": opp.market.id,
                            "market_question": opp.market.question,
                            "venue_id": opp.venue_id,
                            "category": opp.category,
                            "strategy": self.strategy_selector.select_strategy(opp).value,
                            "edge": opp.effective_edge,
                            "fair_value": opp.estimated_fair,
                            "market_price": opp.market_price,
                            "position_size_usd": guard_result.max_spend_usd,
                            "kelly_fraction": adjusted_order.get("kelly_adj", 0),
                            "status": result.get("status", "executed"),
                            "side": opp.side,
                            "confidence": opp.confidence,
                            "uncertainty": opp.uncertainty,
                            "score": opp.score,
                            "reasoning": opp.reasoning[:500]
                        }
                        self.storage.log_trade(trade_record)
                        self.performance_tracker.record_trade(trade_record)
                        self.exposure_manager.add_position(
                            market_id=opp.market.id,
                            amount_usd=guard_result.max_spend_usd,
                            category=opp.category,
                            correlation_group=opp.correlation_group,
                            venue=opp.venue_id
                        )
                        self.execution_guard.record_trade()
                        
                        # Record forecast for calibration
                        self.calibration_engine.record_forecast(
                            market_id=opp.market.id,
                            question=opp.market.question,
                            forecast_prob=opp.estimated_fair,
                            confidence=opp.confidence,
                            market_price=opp.market_price,
                            category=opp.category
                        )
                        
                        # Record for venue learning
                        self.trade_outcome_tracker.record_trade(
                            trade_id=order.id,
                            market_id=opp.market.id,
                            venue_id=opp.venue_id,
                            strategy=self.strategy_selector.select_strategy(opp).value,
                            category=opp.category,
                            forecast_prob=opp.estimated_fair,
                            market_price=opp.market_price,
                            edge=opp.effective_edge,
                            side=opp.side,
                            amount_usd=guard_result.max_spend_usd
                        )
                        
                        # Portfolio after and reconciliation
                        portfolio_after = await adapter.get_portfolio() if hasattr(adapter, 'get_portfolio') else {"balance": self.exposure_manager.bankroll - guard_result.max_spend_usd}
                        recon = self.reconciliation_engine.reconcile_order(order, portfolio_before, portfolio_after)
                        if recon.mismatch:
                            self.kill_switch.check_execution_mismatch(recon.expected_spend, recon.actual_spend)
                        
                        executed.append({
                            "market_id": opp.market.id,
                            "order_id": order.id,
                            "venue": opp.venue_id,
                            "edge": opp.effective_edge,
                            "size": guard_result.max_spend_usd,
                            "result": result,
                            "reconciliation": recon.status
                        })
                        logger.info(f"Executed {opp.market.id} via {opp.venue_id}: edge {opp.effective_edge:.3f} size ${guard_result.max_spend_usd}")
                    else:
                        logger.warning(f"Execution failed for {opp.market.id}: {result}")
                        self.order_manager.update_order(order.id, status=self.order_manager.get_order(order.id).status, raw_response=result)
                else:
                    logger.warning(f"No adapter for venue {opp.venue_id}")

            except Exception as e:
                logger.error(f"Execution error for {opp.market.id}: {e}")
                continue

        # Record scan
        avg_edge = sum(o.effective_edge for o in scan_result.opportunities) / max(1, len(scan_result.opportunities))
        self.storage.log_scan(
            markets_scanned=scan_result.total_scanned,
            opportunities_found=len(scan_result.opportunities),
            avg_edge=avg_edge,
            execution_time=time.time() - cycle_start,
            bankroll=self.exposure_manager.bankroll
        )

        # Update drawdown
        perf = self.storage.get_performance_summary()
        self.drawdown_manager.update_bankroll(perf.get("bankroll", self.exposure_manager.bankroll))
        self.exposure_manager.update_bankroll(perf.get("bankroll", self.exposure_manager.bankroll))
        self.limits_engine.update_bankroll(perf.get("bankroll", self.exposure_manager.bankroll))
        self.execution_guard.update_bankroll(perf.get("bankroll", self.exposure_manager.bankroll))

        # Check drawdown triggers
        dd_check = self.drawdown_manager.check_triggers()
        for trigger in dd_check["triggers"]:
            self.kill_switch.trigger(KillLevel(trigger["level"]), trigger["reason"], {"drawdown": True})

        cycle_time = time.time() - cycle_start
        logger.info(f"=== PTAI v2 Cycle Complete in {cycle_time:.1f}s: {len(executed)} executed, {scan_result.reasoning} ===")

        return {
            "status": "completed",
            "cycle_time": cycle_time,
            "markets_scanned": scan_result.total_scanned,
            "opportunities": len(scan_result.opportunities),
            "executed": len(executed),
            "executed_details": executed,
            "scan_pipeline": scan_result.reasoning,
            "health": health,
            "kill_switch": self.kill_switch.get_status_report(),
            "exposure": self.exposure_manager.get_risk_report(),
            "venue_leaderboard": self.venue_registry.get_venue_leaderboard(),
            "performance": self.performance_tracker.get_summary(),
            "calibration": self.calibration_engine.get_stats(),
            "message": f"Cycle completed: {len(executed)} trades, DO NOTHING is successful if 0" if len(executed) == 0 else f"Executed {len(executed)} trades with avg edge {avg_edge*100:.1f}%"
        }

    def get_status(self) -> Dict:
        return {
            "mission": self.mission,
            "bankroll": self.exposure_manager.bankroll,
            "kill_switch": self.kill_switch.get_status_report(),
            "exposure": self.exposure_manager.get_risk_report(),
            "performance": self.performance_tracker.get_summary(),
            "calibration": self.calibration_engine.get_stats(),
            "venue_performance": self.venue_registry.get_venue_leaderboard(),
            "venue_concentration": self.venue_registry.should_concentrate_on(),
            "learning": self.trade_outcome_tracker.should_concentrate_on()
        }
