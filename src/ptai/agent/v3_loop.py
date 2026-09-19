"""
PTAI V3 - Genuinely multi-venue, multi-strategy opportunity engine
Polymarket and Kalshi are simply first two adapters, AI determines where to deploy capital based on measured edge

Core principle: venue × market × strategy evaluation, not just market

Mission: Find best legitimate opportunity available to capital right now across all venues and strategies
If nothing, DO NOTHING is successful

Architecture:
    PTAI Mission
        ↓
    Market Discovery (all venues)
        ↓
    ┌──────────┼──────────┬──────────┐
    Polymarket Kalshi  Manifold  Crypto  Stocks
        ↓         ↓        ↓        ↓       ↓
        └──────────┼────────────────┘
                   ▼
            NORMALIZED MARKET
                   │
                   ▼
         STRATEGY ENGINE V3
    venue × market × strategy
    ┌────────┼────────┬────────┐
    Mispricing Arbitrage Event  Momentum  MeanRev  MM
         ↓       ↓       ↓        ↓        ↓      ↓
         └───────┼────────────────┘
                 ▼
         COMMON SCORING
    edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)
                 │
                 ▼
         OPPORTUNITY RANK
    "Where is my edge?"
                 │
                 ▼
          RISK ENGINE
                 │
                 ▼
          EXECUTION
                 │
                 ▼
         MONITOR + LEARN
    Which venue×strategy combos demonstrate edge?
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

from ..venues.registry import VenueRegistry
from ..venues.polymarket_adapter import PolymarketAdapter
from ..venues.kalshi_adapter import KalshiAdapter
from ..venues.manifold_adapter import ManifoldAdapter
from ..venues.crypto_adapter import CryptoAdapter
from ..venues.stock_adapter import StockAdapter
from ..venues.predictit_adapter import PredictItAdapter
from ..venues.simmer_adapter import SimmerAdapter
from ..venues.cymetica_adapter import CymeticaAdapter
from ..venues.whitebit_adapter import WhiteBITAdapter
from ..venues.afx_adapter import AFXAdapter
from ..venues.grvt_adapter import GRVTAdapter
from ..venues.pionex_adapter import PionexAdapter
from ..venues.betfair_adapter import BetfairAdapter, BetdaqAdapter, BetConnectAdapter
from ..venues.ccxt_adapter import CCXTUnifiedAdapter
from ..venues.veynor_adapter import VeynorAdapter
from ..venues.openpx_adapter import OpenPXAdapter
from ..venues.apify_adapter import ApifyAdapter
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
from ..strategy.strategy_engine import StrategyEngineV3, MultiVenueScanResult
from ..strategy.alpha_engine import AlphaEngine

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


class TradingAgentV3:
    """
    PTAI V3 - genuinely multi-venue, multi-strategy
    Mission: Find best legitimate opportunity across all venues and strategies
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
        
        # Strategy
        self.fair_value_engine = FairValueEngine(
            llm_router=self.llm_router,
            calibration_engine=self.calibration_engine,
            uncertainty_engine=self.uncertainty_engine
        )
        self.edge_calculator = EdgeCalculator(uncertainty_engine=self.uncertainty_engine)
        self.strategy_selector = StrategySelector()
        self.strategy_engine_v3 = StrategyEngineV3(
            fair_value_engine=self.fair_value_engine,
            edge_calculator=self.edge_calculator,
            strategy_selector=self.strategy_selector
        )
        
        # Alpha engine - all additional alpha ideas
        self.alpha_engine = AlphaEngine(bankroll=self.storage.get_performance_summary().get("bankroll", 50.0))
        
        # Venues - multi-venue registry
        self.venue_registry = VenueRegistry(country_code=country_code)
        
        # Register all venues
        try:
            pk = self.vault.get("Trader", "polymarket_clob", {}).get("private_key") or self.settings.polymarket_private_key
            funder = self.vault.get("Trader", "polymarket_clob", {}).get("funder") or self.settings.polymarket_funder_address
        except:
            pk = None
            funder = None
        
        # Polymarket - primary prediction venue
        polymarket_adapter = PolymarketAdapter(private_key=pk, funder=funder)
        self.venue_registry.register(polymarket_adapter)
        
        # Kalshi - CFTC-regulated US event exchange, cross-venue arb highest prob
        kalshi_adapter = KalshiAdapter()
        self.venue_registry.register(kalshi_adapter)
        
        # Manifold - play money zero-cost testing ground for fair-value engine
        manifold_adapter = ManifoldAdapter()
        self.venue_registry.register(manifold_adapter)

        # PredictIt - public read-only feed heavy limits, sentiment source US politics only
        predictit_adapter = PredictItAdapter()
        self.venue_registry.register(predictit_adapter)

        # Simmer - AI agents trade against each other, Python SDK virtual + live
        simmer_adapter = SimmerAdapter(use_virtual=True)
        self.venue_registry.register(simmer_adapter)

        # Cymetica - perpetual prediction markets official Python SDK
        cymetica_adapter = CymeticaAdapter()
        self.venue_registry.register(cymetica_adapter)
        
        # Crypto - financial venue Binance
        crypto_adapter = CryptoAdapter(exchange="binance")
        self.venue_registry.register(crypto_adapter)

        # WhiteBIT - margin 10x futures 100x low minimums $50 bankroll can execute, HMAC-SHA512 no testnet
        whitebit_adapter = WhiteBITAdapter()
        self.venue_registry.register(whitebit_adapter)

        # AFX DEX - perpetual DEX wallet-signed EIP-712 min deposit 10 USDC withdrawal 2 USDC no API keys ideal local wallet
        afx_adapter = AFXAdapter(wallet_address=funder)
        self.venue_registry.register(afx_adapter)

        # GRVT - hybrid derivatives CLOB min $50 testing $200+ live Hummingbot integration
        grvt_adapter = GRVTAdapter()
        self.venue_registry.register(grvt_adapter)

        # Pionex - free REST WebSocket 10 req/sec spot bot futures built-in grid DCA via API
        pionex_adapter = PionexAdapter()
        self.venue_registry.register(pionex_adapter)
        
        # Stocks - financial venue
        stock_adapter = StockAdapter(broker="mock")
        self.venue_registry.register(stock_adapter)

        # Betfair - world's largest betting exchange flumine framework Betfair Betdaq Betconnect lay betting
        betfair_adapter = BetfairAdapter(use_flumine=True)
        self.venue_registry.register(betfair_adapter)
        betdaq_adapter = BetdaqAdapter()
        self.venue_registry.register(betdaq_adapter)
        betconnect_adapter = BetConnectAdapter()
        self.venue_registry.register(betconnect_adapter)

        # CCXT Unified - single Python client for cross-venue data prediction + crypto, one strategy same code
        ccxt_adapter = CCXTUnifiedAdapter(venues=["polymarket", "kalshi", "binance", "whitebit", "pionex"])
        self.venue_registry.register(ccxt_adapter)

        # Veynor - prediction market intelligence API Kalshi+Polymarket whale trades smart money arb signals 100 credits/month free
        veynor_adapter = VeynorAdapter()
        self.venue_registry.register(veynor_adapter)

        # OpenPX - Rust client sub-millisecond WebSocket Polymarket+Kalshi typed interfaces highest-performance arb
        openpx_adapter = OpenPXAdapter()
        self.venue_registry.register(openpx_adapter)

        # Apify - paid arb scanners Polymarket Kalshi PredictIt ranking fee-adjusted edge $2 per 1000 matched pairs expensive for $50 but useful if scale
        apify_adapter = ApifyAdapter()
        self.venue_registry.register(apify_adapter)
        
        # Link registry to strategy engine
        self.strategy_engine_v3.venue_registry = self.venue_registry
        
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
        
        # Mission
        self.mission = "Find best legitimate opportunity across all venues and strategies. Trade only when evidence, calibration, liquidity, risk agree. DO NOTHING is successful."
        
        logger.info(f"PTAI V3 initialized: {self.mission} | Country {country_code} | Bankroll ${bankroll} | Venues {list(self.venue_registry.adapters.keys())}")

    async def check_system_health(self) -> Dict[str, Any]:
        health = {
            "llm_available": False,
            "storage_ok": True,
            "internet_ok": True,
            "bankroll": self.storage.get_performance_summary().get("bankroll", 0),
            "kill_switch_level": int(self.kill_switch.current_level),
            "can_trade": self.kill_switch.can_trade(),
            "venues": list(self.venue_registry.adapters.keys())
        }
        
        try:
            provider = self.llm_router.get_provider_name()
            health["llm_available"] = True
            health["llm_provider"] = provider
        except Exception as e:
            health["llm_available"] = False
            health["llm_error"] = str(e)
            self.kill_switch.check_llm_unavailable(1000)
        
        if self.calibration_engine.is_degrading():
            self.kill_switch.check_calibration_collapse(self.calibration_engine.calculate_brier_score())
        
        return health

    async def check_eligibility(self) -> Dict[str, EligibilityStatus]:
        eligibility = self.venue_registry.check_all_eligibility()
        for venue_id, status in eligibility.items():
            if status == EligibilityStatus.RESTRICTED:
                logger.warning(f"Venue {venue_id} restricted for {self.country_code}")
        return eligibility

    async def discover_all_venues(self, target_per_venue: int = 300) -> Dict[str, List[Market]]:
        """Discover markets from all eligible venues - V3 multi-venue"""
        markets_by_venue: Dict[str, List[Market]] = {}
        
        eligible = self.venue_registry.get_eligible_adapters()
        # Also include REQUIRES_VERIFICATION for paper trading
        all_adapters = list(self.venue_registry.adapters.values())
        
        for adapter in all_adapters:
            venue_id = adapter.venue_id
            status = self.venue_registry.eligibility_cache.get(venue_id)
            if status is None:
                status = adapter.check_eligibility(self.country_code)
                self.venue_registry.eligibility_cache[venue_id] = status
            
            # For V3, include all except RESTRICTED in paper trading mode
            # In live mode, only ELIGIBLE
            if status == EligibilityStatus.RESTRICTED:
                # Still allow paper trading for learning, but mark
                logger.info(f"Venue {venue_id} restricted for {self.country_code}, including for paper trading learning")
                # Include but with paper flag
                pass
            
            try:
                markets = await adapter.discover_markets(target_count=target_per_venue)
                markets_by_venue[venue_id] = markets
                logger.info(f"{venue_id}: discovered {len(markets)} markets (eligibility {status.value})")
            except Exception as e:
                logger.error(f"{venue_id} discovery failed: {e}")
                markets_by_venue[venue_id] = []
        
        total = sum(len(m) for m in markets_by_venue.values())
        logger.info(f"V3 Multi-venue discovery: {total} total across {len(markets_by_venue)} venues")
        return markets_by_venue

    async def get_context_for_market(self, market: Market) -> Dict[str, Any]:
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
            # Orderbook from venue
            venue_id = market.raw.get("venue") or getattr(market, 'source', 'unknown')
            if isinstance(venue_id, str) and venue_id in self.venue_registry.adapters:
                adapter = self.venue_registry.adapters[venue_id]
                orderbook = await adapter.get_orderbook(market)
                context["orderbook"] = orderbook
            
            # News (mock for now, would use real news engine)
            # X sentiment
            # Web research for top markets only
        except Exception as e:
            logger.debug(f"Context fetch failed for {market.id}: {e}")
        
        return context

    async def run_cycle(self, target_per_venue: int = 200, max_trades: int = 3) -> Dict[str, Any]:
        """
        V3 Cycle: multi-venue × multi-strategy
        """
        start = time.time()
        logger.info("=== PTAI V3 Cycle Start: Multi-Venue × Multi-Strategy ===")
        
        # Health check
        health = await self.check_system_health()
        if not health["can_trade"]:
            return {
                "status": "blocked",
                "reason": f"Kill switch L{health['kill_switch_level']}",
                "health": health,
                "mission": self.mission,
                "execution_time": time.time() - start
            }
        
        # Eligibility
        eligibility = await self.check_eligibility()
        
        # Discover all venues
        markets_by_venue = await self.discover_all_venues(target_per_venue=target_per_venue)
        total_markets = sum(len(m) for m in markets_by_venue.values())
        
        if total_markets == 0:
            return {
                "status": "no_markets",
                "reason": "No markets discovered from any venue",
                "health": health,
                "eligibility": {k: v.value for k, v in eligibility.items()},
                "execution_time": time.time() - start,
                "mission": self.mission
            }
        
        # Alpha scan - all additional alpha ideas (Top 5 + queue)
        all_markets_flat = [m for markets in markets_by_venue.values() for m in markets]
        try:
            alpha_results = self.alpha_engine.scan_all_alpha(all_markets_flat[:200])
            logger.info(f"Alpha scan: {alpha_results}")
        except Exception as e:
            logger.warning(f"Alpha scan failed: {e}")
            alpha_results = {"error": str(e)}
        
        # V3 Strategy Engine: venue × market × strategy
        scan_result: MultiVenueScanResult = await self.strategy_engine_v3.scan_all_venues(
            markets_by_venue=markets_by_venue,
            context_provider=self,
            max_final_trades=max_trades
        )
        
        # Risk checks on final selected
        final_trades = []
        for opp in scan_result.final_selected:
            # Exposure check
            can_trade, reason = self.exposure_manager.can_open_position(
                market_id=opp.market.id,
                amount_usd=5.0,  # would be calculated via Kelly
                category=opp.category,
                correlation_group=opp.correlation_group
            )
            if not can_trade:
                logger.info(f"Risk blocks {opp.market.id}: {reason}")
                continue
            
            # Limits check
            limits_ok, limits_reason = self.limits_engine.validate(
                market_id=opp.market.id,
                edge=opp.effective_edge,
                confidence=opp.confidence,
                amount_usd=5.0,
                price=opp.market_price
            )
            if not limits_ok:
                logger.info(f"Limits block {opp.market.id}: {limits_reason}")
                continue
            
            # Kill switch check
            if not self.kill_switch.can_trade():
                logger.warning(f"Kill switch blocks trading L{self.kill_switch.current_level}")
                break
            
            final_trades.append(opp)
        
        # Execution (dry run for V3)
        execution_results = []
        for opp in final_trades[:max_trades]:
            try:
                venue_id = opp.venue_id.split("+")[0] if "+" in opp.venue_id else opp.venue_id
                # Handle composite venue ids
                if venue_id not in self.venue_registry.adapters:
                    # Try to find matching adapter
                    for vid in self.venue_registry.adapters.keys():
                        if vid in opp.venue_id or opp.venue_id in vid:
                            venue_id = vid
                            break
                
                adapter = self.venue_registry.adapters.get(venue_id)
                if not adapter:
                    # Use first adapter for dry run
                    adapter = list(self.venue_registry.adapters.values())[0]
                
                # Kelly sizing
                kelly_fraction = self.kelly_calculator.calculate(
                    edge=opp.effective_edge,
                    prob=opp.estimated_fair,
                    confidence=opp.confidence
                )
                amount_usd = bankroll = self.storage.get_performance_summary().get("bankroll", 50.0)
                amount_usd = amount_usd * kelly_fraction
                amount_usd = min(amount_usd, 3.0)  # Cap for $50 bankroll
                
                # Execution guard
                guard_result = self.execution_guard.validate(
                    market_id=opp.market.id,
                    side=opp.side,
                    max_price=opp.market_price + 0.02,
                    max_spend=amount_usd,
                    risk_approved_amount=amount_usd
                )
                
                if not guard_result.allowed:
                    logger.warning(f"Execution guard blocks {opp.market.id}: {guard_result.reason}")
                    continue
                
                result = await adapter.place_order(
                    opportunity=opp,
                    max_spend_usd=amount_usd,
                    max_price=opp.market_price + 0.02
                )
                execution_results.append({
                    "market_id": opp.market.id,
                    "venue": opp.venue_id,
                    "strategy": opp.raw.get("strategy", "unknown") if hasattr(opp, 'raw') and isinstance(opp.raw, dict) else "unknown",
                    "side": opp.side,
                    "edge": opp.effective_edge,
                    "score": opp.score,
                    "amount": amount_usd,
                    "result": result
                })
                
                # Record for calibration
                self.calibration_engine.record_prediction(
                    market_id=opp.market.id,
                    forecast_prob=opp.estimated_fair,
                    confidence=opp.confidence,
                    category=opp.category,
                    venue=opp.venue_id
                )
                
            except Exception as e:
                logger.error(f"Execution failed for {opp.market.id}: {e}")
                execution_results.append({
                    "market_id": opp.market.id,
                    "error": str(e)
                })
        
        elapsed = time.time() - start
        
        # Build V3 report
        result = {
            "status": "completed",
            "mission": self.mission,
            "health": health,
            "eligibility": {k: v.value for k, v in eligibility.items()},
            "discovery": {
                "total_scanned": scan_result.total_scanned,
                "per_venue": {
                    r.venue_id: {
                        "discovered": r.total_discovered,
                        "candidates": r.candidates,
                        "tradeable": r.tradeable,
                        "avg_edge": r.avg_edge,
                        "top_question": r.top_opportunity.market.question[:100] if r.top_opportunity else None,
                        "top_edge": r.top_opportunity.effective_edge if r.top_opportunity else 0,
                        "top_score": r.top_opportunity.score if r.top_opportunity else 0,
                        "top_strategy": r.top_opportunity.raw.get("strategy") if r.top_opportunity and hasattr(r.top_opportunity, 'raw') and isinstance(r.top_opportunity.raw, dict) else None
                    } for r in scan_result.venue_reports
                }
            },
            "strategy_breakdown": scan_result.strategy_breakdown,
            "arbitrage": {
                "total_found": len(scan_result.arbitrage_opportunities),
                "tradeable": len([a for a in scan_result.arbitrage_opportunities if a.should_trade]),
                "top": [
                    {
                        "venue_a": a.venue_a,
                        "venue_b": a.venue_b,
                        "spread": a.spread,
                        "profit_pct": a.estimated_profit_pct,
                        "confidence": a.confidence_same_event,
                        "question_a": a.market_a.question[:80],
                        "question_b": a.market_b.question[:80]
                    } for a in scan_result.arbitrage_opportunities[:3]
                ]
            },
            "opportunities": {
                "total_candidates": scan_result.total_candidates,
                "total_tradeable": scan_result.total_tradeable,
                "final_selected": len(final_trades),
                "best": {
                    "venue": scan_result.best_opportunity.venue_id if scan_result.best_opportunity else None,
                    "strategy": scan_result.best_opportunity.raw.get("strategy") if scan_result.best_opportunity and hasattr(scan_result.best_opportunity, 'raw') and isinstance(scan_result.best_opportunity.raw, dict) else None,
                    "question": scan_result.best_opportunity.market.question[:120] if scan_result.best_opportunity else "DO NOTHING",
                    "edge": scan_result.best_opportunity.effective_edge if scan_result.best_opportunity else 0,
                    "score": scan_result.best_opportunity.score if scan_result.best_opportunity else 0,
                    "side": scan_result.best_opportunity.side if scan_result.best_opportunity else None,
                    "reasoning": scan_result.best_opportunity.reasoning[:200] if scan_result.best_opportunity else "No edge found - DO NOTHING is successful"
                }
            },
            "alpha": alpha_results,
            "execution": execution_results,
            "reasoning": scan_result.reasoning,
            "execution_time": elapsed,
            "do_nothing_success": len(final_trades) == 0
        }
        
        logger.info(f"=== PTAI V3 Cycle Complete in {elapsed:.1f}s: {result['opportunities']['final_selected']} trades, DO NOTHING success: {result['do_nothing_success']} ===")
        logger.info(f"V3 Report: {scan_result.reasoning}")
        
        return result

    async def run_continuous(self, interval_minutes: int = 10):
        """Run V3 loop every 10 minutes"""
        logger.info(f"Starting PTAI V3 continuous loop every {interval_minutes} minutes")
        while True:
            try:
                if not self.kill_switch.can_trade():
                    logger.warning(f"Kill switch L{self.kill_switch.current_level} blocks trading, sleeping")
                    await asyncio.sleep(60)
                    continue
                
                result = await self.run_cycle()
                logger.info(f"V3 cycle result: {result['status']} {result['opportunities']['final_selected']} trades")
                
                # Sleep
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"V3 loop error: {e}")
                await asyncio.sleep(60)
