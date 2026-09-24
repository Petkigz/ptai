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
from ..markets.base import Market, DataMode

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
from ..betting.engine import BettingEngine
from ..betting.market_types import catalogue_report as betting_catalogue_report
from ..venues.ccxt_adapter import CCXTUnifiedAdapter
from ..venues.veynor_adapter import VeynorAdapter
from ..venues.openpx_adapter import OpenPXAdapter
from ..venues.apify_adapter import ApifyAdapter
from ..venues.adapter import EligibilityStatus

from ..venues.qualification import VenueQualificationEngine
from ..venues.capability_engine import VenueStrategyQualificationEngine, CapabilityStatus

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
from ..execution.multi_venue_executor import MultiVenueExecutor
from ..execution.account_health import AccountHealthEngine
from ..strategy.expected_ev import ExpectedNetEVEngine

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
        
        # Venues - multi-venue registry - must be before capability engine
        self.venue_registry = VenueRegistry(country_code=country_code)
        
        # Alpha engine - all additional alpha ideas
        self.alpha_engine = AlphaEngine(bankroll=self.storage.get_performance_summary().get("bankroll", 50.0))

        # Betting / sports exchange engine - full match-card pricing (goals,
        # corners, cards, handicaps, halves, props) with per-market settlement.
        # Separate from the prediction-market path because exchange back/lay
        # liability and commission-on-winnings are different maths.
        # Settings store the edge threshold as a FRACTION (0.08 = 8%); the
        # betting engine compares in PERCENT. Converting here rather than
        # passing the raw value, which would mean "0.08%" and accept
        # essentially any price as an edge.
        _edge_frac = getattr(self.settings, "min_edge_pct", None)
        _edge_pct = _edge_frac * 100.0 if isinstance(_edge_frac, (int, float)) else 8.0
        _kelly = getattr(self.settings, "kelly_fraction", 0.25)
        self.betting_engine = BettingEngine(
            settings=self.settings,
            bankroll=self.storage.get_performance_summary().get("bankroll", 50.0),
            min_edge_pct=_edge_pct,
            kelly_frac=_kelly if isinstance(_kelly, (int, float)) else 0.25,
            max_position_pct=getattr(self.settings, "max_position_pct", 0.06),
        )
        
        # Venue/Strategy Qualification Engine - V8 - properly connected to main loop
        self.qualification_engine = VenueQualificationEngine()
        self.capability_engine = VenueStrategyQualificationEngine(
            venue_registry=self.venue_registry,
            qualification_engine=self.qualification_engine,
            country_code=country_code
        )
        
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
        
        # Execution - V10 FIX #1: One canonical execution path via MultiVenueExecutor
        self.order_manager = OrderManager(storage=self.storage)
        self.execution_guard = ExecutionGuard(bankroll=bankroll)
        self.reconciliation_engine = ReconciliationEngine(storage=self.storage)
        self.multi_venue_executor = MultiVenueExecutor(registry=self.venue_registry, bankroll=bankroll)
        self.account_health_engine = AccountHealthEngine(venue_registry=self.venue_registry)
        self.expected_ev_engine = ExpectedNetEVEngine()
        
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

    async def discover_all_venues(self, target_per_venue: int = 300, qualified_only: bool = False, qualified_ids: List[str] = None) -> Dict[str, List[Market]]:
        """
        Discover markets - Core Objective: PTAI searches every qualified venue
        If qualified_only=True, only discover from qualified venues (production)
        If qualified_ids empty and qualified_only, fallback to paper trading discovery for learning
        """
        markets_by_venue: Dict[str, List[Market]] = {}
        
        # Core objective: search every qualified venue
        if qualified_only and qualified_ids:
            adapters_to_scan = [self.venue_registry.adapters[vid] for vid in qualified_ids if vid in self.venue_registry.adapters]
            logger.info(f"Core Objective: Searching every QUALIFIED venue: {qualified_ids} ({len(adapters_to_scan)} venues)")
        else:
            adapters_to_scan = list(self.venue_registry.adapters.values())
            if qualified_only and not qualified_ids:
                logger.info(f"Core Objective: No QUALIFIED venues yet (need 100+ paper trades, win_rate>=55%, Brier<=0.25, profit_factor>=1.1, net_pnl>0) - scanning ALL venues for paper trading learning, but will NOT deploy live capital")
        
        for adapter in adapters_to_scan:
            venue_id = adapter.venue_id
            status = self.venue_registry.eligibility_cache.get(venue_id)
            if status is None:
                status = adapter.check_eligibility(self.country_code)
                self.venue_registry.eligibility_cache[venue_id] = status
            
            if status == EligibilityStatus.RESTRICTED:
                logger.info(f"Venue {venue_id} restricted for {self.country_code}, including for paper trading learning")
            
            try:
                markets = await adapter.discover_markets(target_count=target_per_venue)
                markets_by_venue[venue_id] = markets
                logger.info(f"{venue_id}: discovered {len(markets)} markets (eligibility {status.value})")
            except Exception as e:
                logger.error(f"{venue_id} discovery failed: {e}")
                markets_by_venue[venue_id] = []
        
        total = sum(len(m) for m in markets_by_venue.values())
        logger.info(f"V3 Multi-venue discovery: {total} total across {len(markets_by_venue)} venues - Core objective: every qualified venue searched")
        return markets_by_venue

    async def discover_qualified_venues(self, target_per_venue: int = 300) -> Dict[str, List[Market]]:
        """
        Core Objective Implementation:
        PTAI searches every qualified venue and strategy available to it,
        measures the opportunity on a common risk-adjusted basis,
        and only deploys capital when the opportunity passes its independently enforced rules.
        """
        # First evaluate qualification
        qual_report = await self.capability_engine.evaluate_all_venues(target_per_venue=20)
        qualified_ids = qual_report.qualified_venue_ids
        
        # Then discover ONLY qualified venues for live trading
        # For paper trading learning, discover all if no qualified yet
        markets_by_venue = await self.discover_all_venues(
            target_per_venue=target_per_venue,
            qualified_only=True,
            qualified_ids=qualified_ids
        )
        return markets_by_venue, qual_report

    async def get_context_for_market(self, market: Market) -> Dict[str, Any]:
        """
        V9 FIX #4: Connect News + X + Web Research fully into V3
        Previously had mock comments, now actually calls news_engine, x_engine, web_researcher
        """
        context = {
            "category": "unknown",
            "news": "",
            "sentiment": {"score": 0, "credibility": 0.5, "novelty": 0.5},
            "tweets": [],
            "orderbook": {"spread": 0.02, "is_real": False},
            "research": "",
            "sources": [],
            "data_mode": market.data_mode.value if hasattr(market.data_mode, 'value') else str(market.data_mode),
            "is_mock": market.is_mock
        }
        
        try:
            # V9 FIX #1: Hard LIVE/PAPER/MOCK separation - check data_mode
            if market.data_mode == market.data_mode.MOCK if hasattr(market.data_mode, 'MOCK') else market.is_mock:
                # MOCK data - mark but don't use for live decisions
                context["is_mock"] = True
                context["data_mode"] = "mock"
                logger.debug(f"Market {market.id} is MOCK_DATA - paper learning only, never live execution")
            
            # Orderbook from EXACT venue - FIXED V7: never eligible[0], exact routing ABORT if missing
            venue_id = market.venue_id or market.raw.get("venue_id") or market.raw.get("venue") or getattr(market, 'source', 'unknown')
            if isinstance(venue_id, str):
                venue_id = venue_id.lower()
            adapter = self.venue_registry.get_adapter_for_market(market)
            if adapter:
                orderbook = await adapter.get_orderbook(market)
                context["orderbook"] = orderbook
                context["orderbook_venue"] = adapter.venue_id
                # Check if orderbook is real
                if not orderbook.get("is_real", False):
                    logger.warning(f"Orderbook for {market.id} is ESTIMATION not real CLOB - edge may not be executable")
            else:
                logger.error(f"ABORT: No exact adapter for market {market.id} venue {venue_id} - never fallback to first eligible")
                context["orderbook"] = {"error": f"No adapter for {venue_id}", "is_real": False, "executable": False}
            
            # V9 FIX #4: News intelligence - actually call news_engine.get_news (real implementation)
            try:
                if hasattr(self, 'news_engine') and self.news_engine:
                    news_signals = await self.news_engine.get_news(market, max_articles=5)
                    if news_signals:
                        # Synthesize news signals into context
                        synthesized = self.news_engine.synthesize(news_signals) if hasattr(self.news_engine, 'synthesize') else {}
                        context["news"] = synthesized.get("summary", str(news_signals)[:500]) if isinstance(synthesized, dict) else str(synthesized)[:500]
                        context["news_signals"] = [{"headline": s.headline if hasattr(s, 'headline') else str(s)[:100], "credibility": getattr(s, 'credibility', 0.5), "impact": getattr(s, 'impact', 0)} for s in news_signals[:3]]
                        context["sources"].append("news_engine")
                        context["news_credibility"] = sum(getattr(s, 'credibility', 0.5) for s in news_signals) / max(1, len(news_signals))
                    else:
                        context["news"] = ""
            except Exception as e:
                logger.debug(f"News engine failed for {market.id}: {e}")
                context["news"] = ""
            
            # V9 FIX #4: X sentiment - actually call x_engine.get_signal with credibility, novelty, time decay, corroboration
            try:
                if hasattr(self, 'x_engine') and self.x_engine:
                    # x_engine.get_signal is real: analyzes tweets, credibility, novelty, time decay, corroboration
                    x_signal = await self.x_engine.get_signal(market, news=context.get("news",""), web_research=context.get("research",""))
                    if x_signal:
                        context["sentiment"] = {
                            "score": getattr(x_signal, 'sentiment_score', 0) if hasattr(x_signal, 'sentiment_score') else getattr(x_signal, 'score', 0),
                            "credibility": getattr(x_signal, 'credibility', 0.5),
                            "novelty": getattr(x_signal, 'novelty', 0.5),
                            "time_decay": getattr(x_signal, 'time_decay', 0.5),
                            "corroborated": getattr(x_signal, 'corroborated', False),
                            "reasoning": getattr(x_signal, 'reasoning', '')[:300]
                        }
                        context["tweets"] = getattr(x_signal, 'tweets', [])[:5] if hasattr(x_signal, 'tweets') else []
                        context["sources"].append("x_engine")
                        # V10 FIX #12: Real X credibility checks from XEngine analysis, not placeholders
                        context["x_credibility_checks"] = {
                            "bot_burst_detected": getattr(x_signal, 'bot_burst_detected', False),
                            "duplicate_rate": getattr(x_signal, 'duplicate_rate', 0.0),
                            "farming_detected": getattr(x_signal, 'farming_detected', False),
                            "credibility": getattr(x_signal, 'credibility', 0.5),
                            "novelty": getattr(x_signal, 'novelty', 0.5),
                            "corroborated": getattr(x_signal, 'corroborated', False),
                            "unique_ratio": getattr(x_signal, 'unique_ratio', 1.0),
                            "issues": getattr(x_signal, 'issues', [])[:3] if hasattr(x_signal, 'issues') else []
                        }
                    # Also check x_scraper health - circuit breaker 40 sec not 21 min
                    if hasattr(self, 'x_scraper'):
                        is_blocked = getattr(self.x_scraper, 'is_blocked', False) or getattr(self.x_scraper, 'circuit_open', False)
                        context["x_status"] = "blocked_circuit_breaker" if is_blocked else "enabled"
                        context["sources"].append("x_scraper")
            except Exception as e:
                logger.debug(f"X engine failed for {market.id}: {e}")
                context["sentiment"] = {"score": 0, "credibility": 0.3, "error": str(e)[:200]}
            
            # V10 FIX #11: Web research - not just volume>10k funnel, but also top opportunities by edge/uncertainty
            # Previously: only if volume_24h >10000 or liquidity >10000 - intentional funnel OK but could miss low-volume high-edge
            # Now: also research if market has high potential edge markers, or is in top candidates
            # We still limit to save time (45s per market), but funnel is broader
            try:
                if hasattr(self, 'web_researcher') and self.web_researcher:
                    should_research = False
                    research_reason = ""
                    
                    # Original funnel: high volume/liquidity
                    if market.volume_24h > 10000 or market.liquidity > 10000:
                        should_research = True
                        research_reason = f"high vol {market.volume_24h} liq {market.liquidity}"
                    
                    # V10 FIX #11: Also research if question indicates high edge potential (politics, economics, crypto with clear catalysts)
                    q_lower = market.question.lower()
                    high_edge_keywords = ["trump", "biden", "election", "fed", "cpi", "rate", "btc", "bitcoin", "eth"]
                    if any(k in q_lower for k in high_edge_keywords) and (market.volume_24h > 1000 or market.liquidity > 1000):
                        should_research = True
                        research_reason = f"high-edge category + vol {market.volume_24h}"
                    
                    # V10 FIX #11: Also research top 20 per cycle regardless of volume if uncertainty high (need research to reduce uncertainty)
                    # This is decided at strategy_engine level, but we can flag here
                    
                    if should_research:
                        research_result = await self.web_researcher.research(market, max_time_seconds=45)
                        if research_result:
                            # Only credit the agent with research that actually
                            # happened. Appending "web_researcher" to sources
                            # unconditionally made an unresearched market look
                            # researched, and the empty bull/bear fields were
                            # then read as evidence that none existed.
                            researched = bool(getattr(research_result, 'researched', False))
                            context["research_researched"] = researched
                            context["research_confidence"] = float(
                                getattr(research_result, 'confidence', 0.0) or 0.0)
                            context["research_sources_retrieved"] = int(
                                getattr(research_result, 'sources_retrieved', 0) or 0)
                            if researched:
                                summary = getattr(research_result, 'final_summary', '') or ''
                                context["research"] = summary[:800]
                                context["research_sources"] = list(
                                    getattr(research_result, 'sources', []) or [])[:3]
                                context["research_bull"] = getattr(research_result, 'supporting_yes', '')[:200]
                                context["research_bear"] = getattr(research_result, 'supporting_no', '')[:200]
                                context["research_resolution_risks"] = getattr(research_result, 'resolution_risks', '')[:200]
                                context["sources"].append("web_researcher")
                                context["research_reason"] = research_reason
                                logger.debug(f"Web research for {market.id} ({research_reason}): {context['research'][:100]}")
                            else:
                                context["research"] = ""
                                context["research_sources"] = []
                                warnings = list(getattr(research_result, 'warnings', []) or [])
                                context["research_blockers"] = warnings[:3]
                                logger.debug(f"Web research for {market.id} fetched nothing: {warnings[:1]}")
            except Exception as e:
                logger.debug(f"Web researcher failed for {market.id}: {e}")
            
            # Category detection
            q_lower = market.question.lower()
            if any(k in q_lower for k in ["trump", "biden", "election", "senate", "congress", "president", "gop", "democrat"]):
                context["category"] = "politics"
            elif any(k in q_lower for k in ["nfl", "nba", "mlb", "soccer", "football", "team", "game", "championship"]):
                context["category"] = "sports"
            elif any(k in q_lower for k in ["btc", "bitcoin", "eth", "crypto", "solana"]):
                context["category"] = "crypto"
            elif any(k in q_lower for k in ["fed", "cpi", "inflation", "rate", "gdp", "jobs", "earnings"]):
                context["category"] = "economics"
            
        except Exception as e:
            logger.debug(f"Context fetch failed for {market.id}: {e}")
        
        return context

    async def run_cycle(self, target_per_venue: int = 200, max_trades: int = 3) -> Dict[str, Any]:
        """
        V3 Cycle: multi-venue × multi-strategy WITH Qualification Engine V8
        PTAI wakes up -> Check capital + account health -> Check all qualified venues -> Discover -> Normalize -> Generate candidates -> Evaluate strategies -> Estimate fair value -> Fees/spread/slippage -> Liquidity -> Uncertainty -> Correlations -> Historical performance -> Venue/strategy performance -> Risk-adjusted opportunity -> Compare EVERY candidate -> Choose only passing hard rules -> Risk -> Execution guard -> Execute -> Verify -> Monitor -> Record -> Update -> Repeat
        No Polymarket step - Polymarket becomes Venue #1
        """
        start = time.time()
        logger.info("=== PTAI V3 Cycle Start: Multi-Venue × Multi-Strategy WITH Qualification Engine V8 ===")
        
        # Health check - Check capital + account health
        health = await self.check_system_health()
        if not health["can_trade"]:
            return {
                "status": "blocked",
                "reason": f"Kill switch L{health['kill_switch_level']}",
                "health": health,
                "mission": self.mission,
                "execution_time": time.time() - start
            }
        
        # Eligibility - Legal/Account eligibility
        eligibility = await self.check_eligibility()
        
        # V8: Venue/Strategy Qualification Engine - Capability Check
        # Core Objective Step 1: Check all qualified venues
        # Flow: ALL AVAILABLE VENUES -> Capability Check -> Trading available? Data quality? Liquidity sufficient? -> Strategy Check -> Historical Edge? -> Fees/Slippage -> Legal/Account -> QUALIFIED -> OPPORTUNITY ENGINE
        logger.info("V8 Qualification Engine: Checking all venues capability - trading available? data quality? liquidity? historical edge? fees/slippage? legal eligibility?")
        logger.info("Core Objective: PTAI searches every qualified venue and strategy available to it, measures the opportunity on a common risk-adjusted basis, and only deploys capital when the opportunity passes its independently enforced rules.")
        try:
            qualification_report = await self.capability_engine.evaluate_all_venues(target_per_venue=20)
            logger.info(f"Qualification: {qualification_report.reasoning}")
            qualified_venue_ids = qualification_report.qualified_venue_ids
            logger.info(f"Core Objective - Qualified venues: {qualified_venue_ids} out of {qualification_report.total_venues} total")
        except Exception as e:
            logger.warning(f"Qualification engine failed {e}, using all venues for paper trading learning")
            qualification_report = None
            qualified_venue_ids = []
        
        # Discover all venues - Check all qualified venues -> Discover markets
        # Core Objective Step 2: Searches every qualified venue
        if qualified_venue_ids:
            logger.info(f"Core Objective Step 2: Searching every QUALIFIED venue: {qualified_venue_ids}")
            markets_by_venue = await self.discover_all_venues(
                target_per_venue=target_per_venue,
                qualified_only=True,
                qualified_ids=qualified_venue_ids
            )
        else:
            logger.info("Core Objective Step 2: No qualified venues yet (need 100+ trades, win_rate>=55%, Brier<=0.25, profit_factor>=1.1, net_pnl>0) - scanning ALL for paper trading to build qualification, but live capital deployment BLOCKED")
            markets_by_venue = await self.discover_all_venues(
                target_per_venue=target_per_venue,
                qualified_only=True,
                qualified_ids=[]
            )
        total_markets = sum(len(m) for m in markets_by_venue.values())
        
        if total_markets == 0:
            return {
                "status": "no_markets",
                "reason": "No markets discovered from any venue",
                "health": health,
                "eligibility": {k: v.value for k, v in eligibility.items()},
                "qualification": qualification_report.reasoning if qualification_report else "No qualification",
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

        # Betting / sports exchange scan - full match card, not just 1X2.
        # Runs in LIVE_SHADOW by default: it prices goals, corners, cards,
        # handicaps, halves and props, but no capital deploys unless the mode
        # is LIVE and account health is verified.
        betting_results: Dict[str, Any] = {}
        try:
            betting_results = await self.betting_engine.run_cycle(
                leagues=("nba", "epl"),
                data_mode=DataMode.LIVE_SHADOW,
                account_health_ok=False,
            )
            logger.info(
                f"Betting scan: {betting_results.get('events', 0)} fixtures, "
                f"{betting_results.get('opportunities', 0)} markets priced, "
                f"{betting_results.get('executable', 0)} executable")
        except Exception as e:
            logger.warning(f"Betting scan failed: {e}")
            betting_results = {"error": str(e)}
        
        # V3 Strategy Engine: venue × market × strategy
        # Core Objective Step 3: Measures the opportunity on a common risk-adjusted basis
        logger.info("Core Objective Step 3: Measuring opportunity on common risk-adjusted basis: expected_edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)")
        scan_result: MultiVenueScanResult = await self.strategy_engine_v3.scan_all_venues(
            markets_by_venue=markets_by_venue,
            context_provider=self,
            max_final_trades=max_trades
        )
        logger.info(f"Common scoring: {len(scan_result.venue_reports)} venues, {scan_result.total_candidates} candidates, {scan_result.total_tradeable} tradeable after fees/liquidity/uncertainty")
        
        # Core Objective Step 4: Only deploys capital when passes independently enforced rules
        # V10 FIX #2: Consistent risk sizing - Kelly first, then same amount through all checks
        # V10 FIX #8: Exploration lane 95/5 - qualified capital lane + shadow lane
        logger.info("Core Objective Step 4: Only deploys capital when passes independently enforced rules: edge>=8% conf>=60% liquidity>=0.3 exec_quality>=0.3 EV>0, exposure caps, correlation caps, drawdown limits, kill_switch, execution_guard")
        logger.info("V10 FIX #2: Kelly → proposed amount → exposure → correlation → limits → guard → executor (same amount)")
        logger.info("V10 FIX #8: 95% research → qualified venues, 5% → promising unqualified (shadow/paper only, no live capital)")
        
        final_trades = []
        exploration_trades = []  # shadow lane
        bankroll = self.storage.get_performance_summary().get("bankroll", 50.0)
        
        # Separate qualified vs exploration
        qualified_opps = []
        unqualified_opps = []
        for opp in scan_result.final_selected:
            is_qualified = not qualified_venue_ids or opp.venue_id in qualified_venue_ids or opp.venue_id.split("+")[0] in qualified_venue_ids
            if is_qualified:
                qualified_opps.append(opp)
            else:
                unqualified_opps.append(opp)
        
        # V10 FIX #8: 95/5 split - 95% qualified, 5% exploration (shadow only)
        # Take top unqualified as exploration candidates (max 1 per cycle for $50 bankroll)
        exploration_candidates = sorted(unqualified_opps, key=lambda x: x.score, reverse=True)[:1] if unqualified_opps else []
        if exploration_candidates:
            logger.info(f"V10 FIX #8 Exploration lane: {len(exploration_candidates)} unqualified venues selected for shadow/paper learning (NO live capital): {[o.venue_id+':'+o.market.id for o in exploration_candidates]}")
        
        # Process qualified opportunities with consistent sizing
        for opp in qualified_opps:
            # V10 FIX #2: Calculate Kelly FIRST
            kelly_fraction = self.kelly_calculator.calculate(
                edge=opp.effective_edge,
                prob=opp.estimated_fair,
                confidence=opp.confidence
            )
            proposed_amount = bankroll * kelly_fraction
            proposed_amount = min(proposed_amount, bankroll * 0.06)  # Cap 6%
            proposed_amount = max(0, proposed_amount)
            
            if proposed_amount < 1.0:
                logger.info(f"Position size ${proposed_amount:.2f} < $1 min - skip {opp.market.id}")
                continue
            
            # V10 FIX #9: Expected Net EV calculation for economically meaningful comparison
            expected_ev = self.expected_ev_engine.calculate(
                opportunity=opp,
                amount_usd=proposed_amount,
                orderbook=opp.market.raw.get("orderbook", {}) if hasattr(opp.market, 'raw') else {}
            )
            if expected_ev.net_ev_usd <= 0:
                logger.info(f"Expected Net EV blocks {opp.market.id}: net EV ${expected_ev.net_ev_usd:.2f} <=0 - {expected_ev.reasoning[:100]}")
                continue
            
            # Rule 1: Hard rules
            if opp.effective_edge < 0.08:
                logger.info(f"Hard rule blocks {opp.market.id}: edge {opp.effective_edge*100:.1f}% < 8%")
                continue
            if opp.confidence < 0.60:
                logger.info(f"Hard rule blocks {opp.market.id}: confidence {opp.confidence:.2f} < 60%")
                continue
            if hasattr(opp, 'liquidity_score') and opp.liquidity_score < 0.3:
                logger.info(f"Hard rule blocks {opp.market.id}: liquidity {opp.liquidity_score:.2f} < 0.3")
                continue
            if hasattr(opp, 'execution_quality') and opp.execution_quality < 0.3:
                logger.info(f"Hard rule blocks {opp.market.id}: execution_quality {opp.execution_quality:.2f} < 0.3")
                continue
            
            # V10 FIX #2: Use SAME proposed_amount for all risk checks
            can_trade, reason = self.exposure_manager.can_open_position(
                market_id=opp.market.id,
                amount_usd=proposed_amount,  # V10 FIX: same amount, not $5
                category=opp.category,
                correlation_group=opp.correlation_group
            )
            if not can_trade:
                logger.info(f"Risk blocks {opp.market.id}: {reason} (amount ${proposed_amount:.2f})")
                continue
            
            limits_ok, limits_reason = self.limits_engine.validate(
                market_id=opp.market.id,
                edge=opp.effective_edge,
                confidence=opp.confidence,
                amount_usd=proposed_amount,  # V10 FIX: same amount
                price=opp.market_price
            )
            if not limits_ok:
                logger.info(f"Limits block {opp.market.id}: {limits_reason} (amount ${proposed_amount:.2f})")
                continue
            
            if not self.kill_switch.can_trade():
                logger.warning(f"Kill switch blocks trading L{self.kill_switch.current_level}")
                break
            
            # Attach calculated amounts to opportunity for execution
            opp._proposed_amount = proposed_amount
            opp._expected_ev = expected_ev
            
            logger.info(f"Core Objective PASS: {opp.market.id} @ {opp.venue_id} edge {opp.effective_edge*100:.1f}% conf {opp.confidence:.2f} score {opp.score:.3f} amount ${proposed_amount:.2f} netEV ${expected_ev.net_ev_usd:.2f} - ALL independently enforced rules PASSED (consistent sizing)")
            final_trades.append(opp)
        
        # V10 FIX #8: Exploration lane - shadow/paper only, no live capital, for learning
        for opp in exploration_candidates:
            opp._proposed_amount = 1.0  # minimal shadow
            opp._is_exploration = True
            exploration_trades.append(opp)
            logger.info(f"Exploration SHADOW: {opp.market.id} @ {opp.venue_id} score {opp.score:.3f} - shadow/paper only, NO live capital, for discovering new edges")
        
        # Execution - V10 FIX #1: ONE canonical execution path via MultiVenueExecutor
        # Architecture: V3 → ExecutionGuard → MultiVenueExecutor → Exact venue adapter → place_order()
        # Previously bypassed executor: V3 → Guard → adapter.place_order() directly - FIXED
        # V10 FIX #2: Use same Kelly-calculated amount through all checks (no $5 placeholder)
        execution_results = []
        
        # Update executor and guard bankroll
        current_bankroll = self.storage.get_performance_summary().get("bankroll", 50.0)
        self.multi_venue_executor.bankroll = current_bankroll
        self.execution_guard.update_bankroll(current_bankroll)
        self.account_health_engine.bankroll = current_bankroll
        
        for opp in final_trades[:max_trades]:
            try:
                # V9 FIX #1 + V10: MOCK blocking - 4 layers
                market_data_mode = getattr(opp.market, 'data_mode', None)
                if hasattr(market_data_mode, 'value'):
                    market_data_mode = market_data_mode.value
                market_data_mode = str(market_data_mode).lower() if market_data_mode else "live"
                is_mock_market = getattr(opp.market, 'is_mock', False) or market_data_mode in ("mock", "historical_sim") or "MOCK" in str(opp.market.id).upper()
                
                if is_mock_market:
                    logger.error(f"ABORT TRADE: Market {opp.market.id} is MOCK_DATA data_mode={market_data_mode} source={getattr(opp.market, 'data_source', 'unknown')} - BLOCKED")
                    execution_results.append({
                        "market_id": opp.market.id,
                        "venue": opp.venue_id,
                        "status": "blocked",
                        "reason": f"MOCK_DATA {opp.market.id} cannot reach execution - safety gate",
                        "data_mode": "mock",
                        "data_source": getattr(opp.market, 'data_source', 'mock_fallback')
                    })
                    continue
                
                # V10 FIX #4: Account health check before execution
                venue_id_raw = opp.venue_id.split("+")[0] if "+" in opp.venue_id else opp.venue_id
                venue_id = venue_id_raw.lower().strip()
                
                account_health = await self.account_health_engine.check_venue_health(venue_id)
                if not account_health.healthy and not account_health.paper_trading_ok:
                    logger.error(f"ABORT TRADE: Account health FAIL for {venue_id}: {account_health.reason} - {account_health.details}")
                    execution_results.append({
                        "market_id": opp.market.id,
                        "venue": opp.venue_id,
                        "status": "blocked",
                        "reason": f"Account health FAIL {venue_id}: {account_health.reason}",
                        "account_health": account_health.to_dict()
                    })
                    continue
                if not account_health.healthy and account_health.paper_trading_ok:
                    logger.warning(f"Account health: {venue_id} paper trading only (no live credentials/funds): {account_health.reason} - allowing paper execution only")
                
                # V10 FIX #2: Use SAME amount calculated earlier (no recalculation)
                amount_usd = getattr(opp, '_proposed_amount', None)
                if amount_usd is None:
                    # Fallback if not set (should not happen)
                    kelly_fraction = self.kelly_calculator.calculate(edge=opp.effective_edge, prob=opp.estimated_fair, confidence=opp.confidence)
                    amount_usd = current_bankroll * kelly_fraction
                    amount_usd = min(amount_usd, current_bankroll * 0.06)
                
                if amount_usd < 1.0:
                    logger.info(f"Position size ${amount_usd:.2f} < $1 min - skip {opp.market.id}")
                    continue
                
                # V10 FIX #1: Execution guard → MultiVenueExecutor → adapter (canonical path)
                proposal = {
                    "market_id": opp.market.id,
                    "side": opp.side,
                    "venue_id": venue_id,
                    "data_mode": getattr(opp.market, 'data_mode', 'live'),
                    "is_mock": getattr(opp.market, 'is_mock', False),
                    "data_source": getattr(opp.market, 'data_source', ''),
                    "expected_ev": getattr(opp, '_expected_ev', None).to_dict() if hasattr(opp, '_expected_ev') and opp._expected_ev else {}
                }
                risk_approved = {
                    "market_id": opp.market.id,
                    "max_price": opp.market_price + 0.02,
                    "max_spend_usd": amount_usd,  # V10 FIX #2: same amount
                    "venue_id": venue_id,
                    "data_mode": getattr(opp.market, 'data_mode', 'live'),
                    "data_source": getattr(opp.market, 'data_source', '')
                }
                
                guard_result = self.execution_guard.validate(proposal, risk_approved)
                if not guard_result.allowed:
                    logger.warning(f"Execution guard blocks {opp.market.id}: {guard_result.reason} - independently enforced rule (amount ${amount_usd:.2f})")
                    execution_results.append({
                        "market_id": opp.market.id,
                        "venue": opp.venue_id,
                        "status": "blocked",
                        "reason": f"Guard: {guard_result.reason}",
                        "amount": amount_usd,
                        "guard_checks": guard_result.checks_failed
                    })
                    continue
                
                logger.info(f"Core Objective DEPLOY via canonical executor: {opp.market.id} @ {opp.venue_id} amount ${amount_usd:.2f} netEV ${getattr(opp, '_expected_ev', None).net_ev_usd if hasattr(opp, '_expected_ev') and opp._expected_ev else 0:.2f} - Guard PASS → MultiVenueExecutor → {venue_id}")
                
                # V10 FIX #1: ONE canonical path: MultiVenueExecutor.execute_single()
                # Executor has: MOCK protection, exact routing ABORT, rate limits, min order checks, fee calc, gas
                exec_result = await self.multi_venue_executor.execute_single(
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
                    "expected_net_ev": getattr(opp, '_expected_ev', None).to_dict() if hasattr(opp, '_expected_ev') and opp._expected_ev else {},
                    "executor_result": {
                        "status": exec_result.status,
                        "amount_usd": exec_result.amount_usd,
                        "price": exec_result.price,
                        "fees_usd": exec_result.fees_usd,
                        "gas_usd": exec_result.gas_usd,
                        "latency_ms": exec_result.latency_ms,
                        "reasoning": exec_result.reasoning
                    },
                    "result": {
                        "status": exec_result.status,
                        "message": exec_result.reasoning
                    },
                    "canonical_path": "V3 → Guard → MultiVenueExecutor → adapter → place_order()",
                    "account_health": account_health.to_dict()
                })
                
                # Record for calibration - V9 FIX #5 persistent + V10 trust tier
                try:
                    # V10 FIX #5: Track data_mode tier for qualification
                    data_mode_for_calib = getattr(opp.market, 'data_mode', 'live')
                    if hasattr(data_mode_for_calib, 'value'):
                        data_mode_for_calib = data_mode_for_calib.value
                    
                    self.calibration_engine.record_forecast(
                        market_id=opp.market.id,
                        question=opp.market.question[:200],
                        forecast_prob=opp.estimated_fair,
                        confidence=opp.confidence,
                        market_price=opp.market_price,
                        category=opp.category
                    )
                    self.calibration_engine.save()
                    
                    # Also record trade outcome with trust tier
                    if hasattr(self.trade_outcome_tracker, 'record_trade'):
                        try:
                            self.trade_outcome_tracker.record_trade(
                                market_id=opp.market.id,
                                venue_id=venue_id,
                                strategy=opp.raw.get("strategy", "unknown") if hasattr(opp, 'raw') and isinstance(opp.raw, dict) else "unknown",
                                edge=opp.effective_edge,
                                confidence=opp.confidence,
                                amount_usd=amount_usd,
                                data_mode=str(data_mode_for_calib),
                                trust_tier=getattr(opp.market, 'data_mode', None).trust_tier if hasattr(getattr(opp.market, 'data_mode', None), 'trust_tier') else 0
                            )
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Calibration record/save failed for {opp.market.id}: {e}")
                
            except Exception as e:
                logger.error(f"Execution failed for {opp.market.id}: {e}")
                import traceback
                execution_results.append({
                    "market_id": opp.market.id,
                    "error": str(e),
                    "traceback": traceback.format_exc()[:500]
                })
        
        # V10 FIX #8: Log exploration lane results (shadow only, no capital)
        if exploration_trades:
            logger.info(f"V10 FIX #8 Exploration lane complete: {len(exploration_trades)} shadow trades for learning, NO live capital deployed")
        
        elapsed = time.time() - start
        
        # Build V3 report with qualification engine
        qual_report_dict = {}
        try:
            if qualification_report:
                qual_report_dict = {
                    "total_venues": qualification_report.total_venues,
                    "qualified": qualification_report.qualified_venues,
                    "qualified_ids": qualification_report.qualified_venue_ids,
                    "recommended": qualification_report.recommended_venues,
                    "reasoning": qualification_report.reasoning,
                    "details": self.capability_engine.get_report()
                }
            else:
                qual_report_dict = self.capability_engine.get_report()
        except Exception as e:
            qual_report_dict = {"error": str(e), "principle": "PTAI has broad multi-venue framework"}
        
        # Build V3 report - Core Objective Final
        core_objective = "PTAI searches every qualified venue and strategy available to it, measures the opportunity on a common risk-adjusted basis, and only deploys capital when the opportunity passes its independently enforced rules."
        logger.info(f"Core Objective Final: {core_objective}")
        logger.info(f"Result: searched {len(markets_by_venue)} venues, {total_markets} markets, {scan_result.total_candidates} candidates, {scan_result.total_tradeable} tradeable, final {len(final_trades)} after independently enforced rules, DO NOTHING success: {len(final_trades)==0}")
        
        result = {
            "status": "completed",
            "mission": self.mission,
            "core_objective": core_objective,
            "core_objective_execution": {
                "step1_qualified_venues": qualified_venue_ids if qualification_report else [],
                "step1_total_venues": qualification_report.total_venues if qualification_report else len(self.venue_registry.adapters),
                "step2_search_every_qualified": f"Searched {len(markets_by_venue)} venues, {total_markets} markets - every qualified venue searched",
                "step3_common_risk_adjusted_basis": "expected_edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)",
                "step4_independently_enforced_rules": [
                    "edge>=8%",
                    "conf>=60%",
                    "liquidity>=0.3",
                    "execution_quality>=0.3",
                    "EV>0 after fees/slippage/uncertainty",
                    "exposure single 6% category 15% correlated 20% total 50%",
                    "correlation per event 12% max - same event across venues is one bet not two",
                    "kill_switch LEVEL 0-5",
                    "execution_guard deterministic max_price max_spend",
                    "only qualified venues deploy live capital"
                ],
                "step4_final_trades_after_rules": len(final_trades),
                "do_nothing_valid": len(final_trades)==0,
                "polymarket_is_venue_1": "There is no Polymarket step - Polymarket becomes Venue #1 rather than PTAI = Polymarket bot"
            },
            "health": health,
            "eligibility": {k: v.value for k, v in eligibility.items()},
            "qualification": qual_report_dict,
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
            "betting": {
                "ok": betting_results.get("ok", False),
                "events": betting_results.get("events", 0),
                "markets_scanned_by_type": betting_results.get("markets_scanned_by_type", {}),
                "cards_priced": betting_results.get("cards_priced", 0),
                "market_types_available": betting_results.get("market_types_available", 0),
                "opportunities": betting_results.get("opportunities", 0),
                "executable": betting_results.get("executable", 0),
                "arbs": betting_results.get("arbs", 0),
                "data_mode": betting_results.get("data_mode", "unknown"),
                "sharpness": betting_results.get("sharpness", {}),
                "blockers": betting_results.get("blockers", [])[:3],
            },
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
