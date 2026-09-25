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
import json
import time
from datetime import datetime, timezone
import os

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
from ..venues.betfair_adapter import BetdaqAdapter, BetConnectAdapter
from ..venues.betfair_exchange import BetfairExchangeAdapter
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
from ..strategy.edge import HUNT_MISPRICING_MIN, EdgeCalculator, hunted_mispricing
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
from ..execution.redemption import Redeemer
from ..execution.execution_guard import ExecutionGuard
from ..execution.reconciliation import ReconciliationEngine
from ..execution.multi_venue_executor import MultiVenueExecutor
from ..execution.account_health import AccountHealthEngine
from ..execution.settlement import SettlementEngine
from ..execution.position_ledger import PositionLedgerBuilder
from ..strategy.expected_ev import ExpectedNetEVEngine, executable_net_ev

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
    def __init__(self, country_code: str = "UG", dry_run: bool = True):
        """
        dry_run controls whether real capital can move, and it defaults True.

        V3 had no dry_run concept at all: the CLI's --dry-run flag (which
        defaults to settings.dry_run) had nowhere to go when the CLI was pointed
        at V3, and the betting scan hardcoded DataMode.LIVE_SHADOW. This
        restores the safety default so wiring the CLI to V3 cannot accidentally
        arm live trading - a caller has to ask for live explicitly.
        """
        self.settings = get_settings()
        self.country_code = country_code
        self.dry_run = bool(dry_run)
        
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
        #
        # The vault read was `self.vault.get(...)`, and Vault has no `get` - so
        # this raised AttributeError on EVERY construction, and the handler then
        # set both credentials to None. The settings fallback was never reached,
        # because the exception happened before it was evaluated. The result was
        # an agent that could not go live with correct credentials in the vault
        # AND in settings, reporting "unconfigured" as if nothing had been
        # supplied. Credentials are now read by their real names, and a vault
        # failure still falls through to settings rather than discarding them.
        try:
            stored = self.vault.get_tool_credentials("Trader", "polymarket_clob") or {}
        except Exception as e:
            # Not `except: pass`. Silent credential loss looks exactly like
            # "no credentials configured", and every downstream gate then
            # reports paper-only for a reason nobody can see.
            logger.error(f"Could not read Polymarket credentials from the vault: "
                         f"{type(e).__name__}: {e}. Falling back to settings.")
            stored = {}
        pk = stored.get("private_key") or self.settings.polymarket_private_key
        funder = stored.get("funder") or self.settings.polymarket_funder_address
        if not pk or not funder:
            logger.info(
                "No Polymarket credentials from the vault or settings, so live "
                "trading and redemption report as unconfigured. Paper trading "
                "needs neither.")

        # Kept on the instance because the venue's order probe, the redemption
        # client and the account-health ladder all need them. They used to be
        # local variables, so anything built later that asked the agent for its
        # credentials got None - which reads as "not configured" rather than
        # "not passed".
        self.private_key = pk
        self.funder = funder
        
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

        # Betfair - world's largest betting exchange. This is the REAL adapter
        # from betfair_exchange.py, not the legacy stub that returned invented
        # football questions. It is the feed that carries goals, corners, cards
        # and player props, so the derivative market models have something to
        # price against. Without credentials it returns no markets and says so.
        betfair_adapter = BetfairExchangeAdapter(
            username=os.getenv("BETFAIR_USERNAME", ""),
            password=os.getenv("BETFAIR_PASSWORD", ""),
            app_key=os.getenv("BETFAIR_APP_KEY", ""),
            sports=("football",),
            include_player_markets=False,
        )
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
        # Redemption closes the loop after settlement: a settled win that is
        # never redeemed is a bookkeeping profit and unusable capital. Refuses
        # to claim anything without a signer, and reports what it left locked.
        # Findings the cycle records for the operator: which venues have earned
        # live capital, and what the health engine actually saw of each account.
        self._last_qualified_venue_ids: List[str] = []
        self._last_qualification_total = 0
        self._last_account_health: Dict[str, Any] = {}
        self.redeemer = Redeemer(
            funder=getattr(self, "funder", None),
            private_key=getattr(self, "private_key", None),
            # The relayer accepts builder-key auth as an alternative to a
            # relayer key; read it from settings when present.
            builder_key=getattr(self.settings, "polymarket_builder_key", None),
            builder_secret=getattr(self.settings, "polymarket_builder_secret", None),
            builder_passphrase=getattr(self.settings, "polymarket_builder_passphrase", None),
            dry_run=self.dry_run,
        )
        self.execution_guard = ExecutionGuard(bankroll=bankroll)
        self.reconciliation_engine = ReconciliationEngine(storage=self.storage)
        self.multi_venue_executor = MultiVenueExecutor(registry=self.venue_registry, bankroll=bankroll)
        self.account_health_engine = AccountHealthEngine(
            venue_registry=self.venue_registry, storage=self.storage)
        self.expected_ev_engine = ExpectedNetEVEngine()

        # The one place dry_run becomes a data mode. LIVE_PAPER (not SHADOW) when
        # dry: paper trading still qualifies venues and exercises the full
        # prediction and risk path, it just cannot settle real money.
        self.data_mode = DataMode.LIVE_PAPER if self.dry_run else DataMode.LIVE

        # Propagate to every adapter. The adapter is the last gate before a real
        # order, and it holds its own flag, so this is the one line that decides
        # whether live capital is reachable at all. Without it the flag stopped
        # at the agent and a credentialed adapter would submit real orders
        # during what the operator believed was a dry run.
        _armed = []
        for _vid, _adapter in self.venue_registry.adapters.items():
            if hasattr(_adapter, "dry_run"):
                _adapter.dry_run = self.dry_run
                # Only report venues that could actually place an order, not
                # every adapter that happens to hold the flag. Most are honest
                # stubs with supports_trading=False and cannot trade either way.
                if not self.dry_run and getattr(
                        _adapter, "can_place_real_orders", False):
                    _armed.append(_vid)
        if not self.dry_run:
            if _armed:
                logger.warning(
                    f"LIVE MODE: {len(_armed)} venue(s) could submit real orders "
                    f"once account health passes: {_armed}. Execution still "
                    f"requires a TRADE_PERMITTED account health result "
                    f"(authenticated + funded + order probe verified).")
            else:
                logger.warning(
                    "LIVE MODE, but no venue can submit a real order: none of "
                    "the adapters has both live credentials and trading "
                    "support. Every execution this run is simulated.")
        
        # Learning
        self.trade_outcome_tracker = TradeOutcomeTracker(storage=self.storage)
        # Closes execution -> settlement -> outcome -> calibration. Without a
        # settlement pass, `record_resolution` has no caller, the resolved count
        # stays at 0, `is_degrading()` can never return True (it needs 50
        # resolved forecasts) and the kill switch on calibration collapse is
        # dead code. The agent would trade forever on its priors.
        # Sizing reads FREE cash from here, never the stored bankroll. The
        # stored figure does not distinguish money already committed to an open
        # position, so sizing against it lets concurrent positions each claim
        # 6% of the same dollars.
        self.ledger_builder = PositionLedgerBuilder(storage=self.storage)
        self.settlement_engine = SettlementEngine(
            venue_registry=self.venue_registry,
            storage=self.storage,
            calibration_engine=self.calibration_engine,
            trade_outcome_tracker=self.trade_outcome_tracker,
        )
        self.performance_tracker = PerformanceTracker(storage=self.storage)
        
        # Mission
        # The mission is to grow capital under a risk budget. It is NOT to pay
        # for hardware, cover a server bill, or earn a daily figure - those are
        # the operator's decisions about what to do with the money, and
        # hardwiring them into the agent made a $50 account chase a daily target
        # instead of taking the opportunities that were actually there.
        self.mission = (
            "Given available capital, continuously search accessible markets and "
            "strategies, identify opportunities with positive expected net "
            "return, manage risk, execute, learn from outcomes, and grow the "
            "capital. Trade only when evidence, calibration, liquidity and risk "
            "agree. DO NOTHING is a successful outcome."
        )
        
        logger.info(f"PTAI V3 initialized: {self.mission} | Country {country_code} | Bankroll ${bankroll} | Venues {list(self.venue_registry.adapters.keys())}")
        logger.info(
            f"V3 data mode: {self.data_mode.value} (dry_run={self.dry_run}) - "
            + ("no real orders can be placed" if self.dry_run
               else (f"LIVE, {len(_armed)} venue(s) could submit after account "
                     f"health passes {_armed}" if _armed
                     else "LIVE, but no venue can submit a real order")))

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

    async def get_context(self, market: Market) -> Dict[str, Any]:
        """
        The name the strategy engine calls.

        StrategyEngine does `context_provider.get_context(market)`. This class
        only ever had `get_context_for_market()`, so every call raised
        AttributeError into a bare `except:` which substituted
        {"orderbook": {"spread": 0.02, "depth": market.liquidity}} - a fabricated
        spread and a number copied from the market's own liquidity field.

        The effect was that the entire intelligence pipeline - news, X sentiment,
        web research, the real orderbook - never reached the forecast on the V3
        path, while the code implementing all of it sat right here. The forecast
        was built on a placeholder and nothing said so.

        Kept as an alias rather than a rename so any existing caller that does use
        the longer name keeps working.
        """
        return await self.get_context_for_market(market)

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
                # The venue's OWN declared taker fee, carried to the stage that
                # costs the trade. Without this the EV stage fell back to its
                # 2% default for every venue - overcharging Manifold and Kalshi
                # (which charge nothing) and WhiteBIT (0.1%), and hiding the
                # difference between venues entirely.
                try:
                    caps_fee = getattr(adapter.capabilities, "fee_taker_pct", None)
                    if caps_fee is not None:
                        context["fee_taker_pct"] = float(caps_fee)
                    caps_gas = getattr(adapter.capabilities, "order_gas_usd", None)
                    if caps_gas is not None:
                        context["order_gas_usd"] = float(caps_gas)
                except Exception as e:
                    logger.debug(f"Could not read the fee schedule for {market.id}: {e}")
                # ALSO attached to the market, because the expected-EV stage
                # reads `opp.market.raw["orderbook"]` and nothing ever put it
                # there. The result was two different cost estimates for the same
                # trade: EdgeCalculator priced the spread from the real book,
                # then ExpectedNetEVEngine - handed {} - fell back to a default
                # 0.02 spread, so the two stages disagreed about the trade's own
                # costs and the later one decided.
                try:
                    if isinstance(getattr(market, "raw", None), dict):
                        market.raw["orderbook"] = orderbook
                except Exception as e:
                    logger.debug(f"Could not attach the orderbook to {market.id}: {e}")
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

    async def _reconcile_working_orders(self):
        """
        Ask the venue about every order that has not reached a final state.

        The adapter is the source of truth. An adapter that cannot answer leaves
        its orders open, which is the conservative result: an order whose fate
        is unknown keeps holding its capital.
        """
        if self.order_manager is None:
            logger.debug("No order manager: resting orders cannot be reconciled")
            return None
        pending = self.order_manager.open_orders()
        if not pending:
            return None

        logger.info(
            f"Reconciling {len(pending)} working order(s) against the venue - "
            f"${self.order_manager.resting_capital_usd():.4f} of capital is "
            f"reserved behind them")

        total = {"checked": 0, "filled_more": 0, "now_complete": 0,
                 "released": 0, "unreconciled": 0, "grew_usd": 0.0,
                 "released_usd": 0.0}
        by_venue = {}
        for order in pending:
            by_venue.setdefault(order.get("venue_id") or "", []).append(order)

        for venue_id in by_venue:
            adapter = self.venue_registry.get_adapter(venue_id) if venue_id else None
            if adapter is None:
                logger.warning(
                    f"No adapter for {venue_id or 'unknown venue'}: "
                    f"{len(by_venue[venue_id])} order(s) stay unreconciled")
                total["unreconciled"] += len(by_venue[venue_id])
                continue
            report = await self.order_manager.reconcile(
                adapter, venue_id=venue_id,
                position_opener=self._open_position_from_fill)
            for key in total:
                total[key] += getattr(report, key, 0) or 0

        if total["grew_usd"] or total["released_usd"]:
            # Fills move capital, so the derived figures must not stay stale into
            # sizing - the same reason settlement refreshes the bankroll.
            self.bankroll = self.storage.get_bankroll()
            self.execution_guard.update_bankroll(self.bankroll)
            self.account_health_engine.bankroll = self.bankroll
        if total["unreconciled"]:
            logger.warning(
                f"{total['unreconciled']} order(s) could not be reconciled; their "
                f"capital stays reserved until the venue answers")
        return total

    def _open_position_from_fill(self, order: Dict[str, Any], add_usd: float,
                                 price: float) -> int:
        """
        Book a position for a fill that arrived on an order we did not hold one
        for.

        A resting order buys nothing, so no position is created when it is
        submitted; when it fills hours later the fill has nowhere to go. This is
        that destination. The row is labelled with the order it came from, so a
        later audit can trace it back to the venue's order rather than to a
        strategy that never chose it.
        """
        if not add_usd or add_usd <= 0:
            return 0
        market_id = str(order.get("market_id") or "")
        if not market_id:
            logger.error("Cannot open a position from a fill with no market id")
            return 0
        side = str(order.get("side") or "").upper()
        if side not in ("YES", "NO"):
            # The order's side is the outcome side. Without it the position
            # cannot be settled, so it is refused rather than guessed.
            logger.error(
                f"Cannot open a position for {market_id}: the order carries side "
                f"{side!r}, which is not a settleable outcome side (YES/NO)")
            return 0
        # The forecast the order was placed WITH. Not the fill price standing in
        # for a fair value, and not edge 0: the fill is where the trade executed,
        # not what the agent believed when it decided. Falling back to the fill
        # price is honest only when the order genuinely has no forecast (an order
        # placed by an older version, or one written by reconciliation itself),
        # and it is labelled so the outcome can be recognised as unattributed.
        fair_price = order.get("fair_price")
        edge = order.get("edge")
        confidence = order.get("confidence")
        strategy = order.get("strategy")
        category = order.get("category")
        attributed = fair_price is not None and strategy is not None
        if not attributed:
            fair_price = float(price)
            edge = 0.0
            confidence = 0.0
            strategy = "resting_order_fill"

        try:
            # The fill price is the price of the TOKEN this order was for, so
            # the YES price is derived from it and both are stored. Writing the
            # fill price into `market_price` - which means the YES price - is how
            # a NO position came to settle as though its shares had cost the YES
            # price.
            fill_token_price = float(price)
            bought_yes = str(side or "").upper() in ("YES", "1", "LONG", "BUY", "TRUE")
            fill_yes_price = (fill_token_price if bought_yes
                              else round(1.0 - fill_token_price, 6))

            trade_id = self.storage.log_trade({
                "market_id": market_id,
                "venue_id": order.get("venue_id") or "polymarket",
                "side": side,
                "position_size_usd": float(add_usd),
                "market_price": fill_yes_price,
                "yes_price_at_entry": fill_yes_price,
                "token_price_at_entry": fill_token_price,
                # The execution mode the ORDER was submitted with. A delayed
                # fill must state its own mode rather than inherit whatever
                # `status` happens to imply, or a simulated order that filled
                # hours later is filed as a live position.
                "execution_mode": (order.get("execution_mode")
                                   or ("paper" if str(order.get("data_mode") or "")
                                       in ("paper", "live_paper") else "live")),
                # Costs and the prediction, from the order they were recorded
                # on. None where the order does not carry them, which the
                # learning layer reads as "not measured" rather than as zero.
                "fees_usd": order.get("fees_usd"),
                # `fair_value` is the column that exists. The old call passed
                # `fair_price`, which log_trade ignores, so this row carried no
                # fair value at all.
                "fair_value": float(fair_price),
                "edge": float(edge or 0.0),
                "confidence": float(confidence or 0.0),
                "strategy": strategy,
                "category": category,
                "data_mode": (order.get("data_mode")
                              or getattr(self.data_mode, "value", str(self.data_mode))),
                "order_id": order.get("order_id"),
            })
        except Exception as e:
            logger.error(f"Could not record the resting-order fill: "
                         f"{type(e).__name__}: {e}")
            return 0

        # And into the learning record, which is the whole point of keeping the
        # forecast: a delayed fill must be learnable, or the agent's most
        # expensive decisions - the ones it committed to and waited on - are the
        # ones it never studies.
        try:
            self.trade_outcome_tracker.record_trade(
                trade_id=str(trade_id),
                market_id=market_id,
                venue_id=order.get("venue_id") or "polymarket",
                strategy=str(strategy),
                category=str(category or ""),
                forecast_prob=float(fair_price),
                market_price=float(price),
                edge=float(edge or 0.0),
                side=side,
                amount_usd=float(add_usd),
                # The execution facts the order carried. Without them a delayed
                # fill's outcome was unclassified in the paper/live split, and
                # carried no costs and no prediction - so the trades the agent
                # waited longest for taught it the least.
                execution_mode=(order.get("execution_mode") or None),
                expected_net_ev=order.get("expected_net_ev"),
                expected_net_ev_pct=order.get("expected_net_ev_pct"),
                fees_usd=order.get("fees_usd"),
                slippage_bps=order.get("slippage_bps"),
                execution_quality=order.get("execution_quality"),
                data_mode=order.get("data_mode"),
                delayed_fill=True,
                attributed=bool(attributed),
            )
        except Exception as e:
            # Loud, and attached to the result: a position the agent cannot learn
            # from is a fact the operator needs, not a debug line.
            logger.error(
                f"Position {trade_id} opened from a delayed fill but could not be "
                f"recorded for learning: {type(e).__name__}: {e}")

        logger.info(
            f"Resting order {order.get('order_id')} filled ${add_usd:.4f} at "
            f"{price:.4f} - booked as position {trade_id} on {market_id} {side} "
            f"(strategy {strategy}, fair {float(fair_price):.4f}, "
            f"{'attributed' if attributed else 'UNATTRIBUTED - no forecast on the order'})")
        return int(trade_id or 0)

    async def _redeem_settled_wins(self):
        """
        Claim winnings that have settled.

        Driven by the venue's own redeemable list, not by the agent's belief
        that it won: only the contract knows whether collateral is still
        claimable, and redeeming an already-redeemed condition is a no-op rather
        than a double payment.
        """
        redeemer = getattr(self, "redeemer", None)
        if redeemer is None:
            return None
        if not redeemer.funder:
            logger.info(
                "Redemption skipped: no funder address, so there is no position "
                "list to read and nothing can be claimed")
            return None

        positions, available, reason = await asyncio.to_thread(
            redeemer.read_redeemable)
        if not available:
            logger.warning(
                f"Redemption could not read claimable positions ({reason}); "
                f"settled winnings, if any, stay locked this cycle")
            return None
        if not positions:
            return None

        # Skip anything already claimed in an earlier run: the venue is
        # idempotent, but re-submitting burns relayer quota for nothing.
        fresh = [p for p in positions
                 if not redeemer.already_redeemed(self.storage, p.condition_id)]
        if not fresh:
            return None

        report = await asyncio.to_thread(redeemer.redeem, fresh)
        redeemer.record(self.storage, report, fresh)
        if report.claimed:
            self.bankroll = self.storage.get_bankroll()
            self.execution_guard.update_bankroll(self.bankroll)
        return report.to_dict()

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
        
        # Settlement - close positions whose markets have resolved.
        #
        # Runs before discovery and sizing on purpose: the bankroll, win rate and
        # calibration all feed the next decisions, so settling late would size
        # the new cycle against a stale bankroll. This is the step that was
        # missing entirely, and its absence is why calibration never accumulated
        # a single resolved forecast.
        settlement_report = None
        try:
            settlement_report = await self.settlement_engine.settle_pending()
            if settlement_report.settled:
                # Bankroll moved; refresh the derived values before sizing.
                self.bankroll = self.storage.get_bankroll()
                self.execution_guard.update_bankroll(self.bankroll)
                self.account_health_engine.bankroll = self.bankroll
        except Exception as e:
            logger.error(
                f"SETTLEMENT PASS FAILED: {type(e).__name__}: {e}. Resolved "
                f"markets will not be recorded this cycle, so calibration and "
                f"win rate will be stale.")

        # Reconciliation - find out what the venue did with the orders we sent.
        #
        # Runs straight after settlement because it is the same question asked
        # of a different source: settlement says which MARKETS resolved, this
        # says which ORDERS are still working. An order resting in the book
        # holds capital that sizing must not spend twice, and a partial fill is
        # a position that grows after we booked it.
        reconciliation = None
        try:
            reconciliation = await self._reconcile_working_orders()
        except Exception as e:
            logger.error(
                f"ORDER RECONCILIATION FAILED: {type(e).__name__}: {e}. Any order "
                f"resting at the venue stays invisible this cycle, so its capital "
                f"may be sized against again.")

        # Redemption - claim settled winnings.
        #
        # Without this step a won position is marked settled in the ledger and
        # the collateral stays locked in the contract forever, which looks like
        # profit on paper and is not spendable.
        redemption = None
        try:
            redemption = await self._redeem_settled_wins()
        except Exception as e:
            logger.error(
                f"REDEMPTION FAILED: {type(e).__name__}: {e}. Settled winnings "
                f"stay locked in the contract and cannot be traded with.")

        # Eligibility - Legal/Account eligibility
        eligibility = await self.check_eligibility()
        
        # V8: Venue/Strategy Qualification Engine - Capability Check
        # Core Objective Step 1: Check all qualified venues
        # Flow: ALL AVAILABLE VENUES -> Capability Check -> Trading available? Data quality? Liquidity sufficient? -> Strategy Check -> Historical Edge? -> Fees/Slippage -> Legal/Account -> QUALIFIED -> OPPORTUNITY ENGINE
        logger.info("V8 Qualification Engine: Checking all venues capability - trading available? data quality? liquidity? historical edge? fees/slippage? legal eligibility?")
        logger.info("Core Objective: PTAI searches every qualified venue and strategy available to it, measures the opportunity on a common risk-adjusted basis, and only deploys capital when the opportunity passes its independently enforced rules.")
        # Requalification from what actually resolved. This is the connection
        # that was missing: the gate reads adapter.performance_stats (never
        # written) and a JSON file only tests ever wrote, so its sample size was
        # permanently zero and no venue could ever earn its way to live - while a
        # stale file was read back as evidence. Every cycle, every registered
        # venue's numbers are rebuilt from the outcome log.
        try:
            refreshed = self._refresh_qualifications()
            logger.info(
                f"Qualification inputs refreshed from recorded outcomes: "
                f"{refreshed} venue(s)")
        except Exception as e:
            logger.error(
                f"Could not refresh qualification inputs: {type(e).__name__}: {e}. "
                f"The gate will read whatever it last held, so treat the "
                f"qualification result as stale.")

        try:
            qualification_report = await self.capability_engine.evaluate_all_venues(target_per_venue=20)
            logger.info(f"Qualification: {qualification_report.reasoning}")
            qualified_venue_ids = qualification_report.qualified_venue_ids
            # Kept on the instance: "which venue has earned live capital" is the
            # first question an operator asks, and it was a local variable that
            # died with the cycle.
            self._last_qualified_venue_ids = list(qualified_venue_ids)
            self._last_qualification_total = qualification_report.total_venues
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
            # Same shape as the full result. The no-markets path used to return
            # a different set of keys, so a UI had to special-case it to render
            # "nothing to trade" - and the honest empty case is the common one
            # whenever a venue is down or unconfigured.
            empty_venues = {vid: 0 for vid in markets_by_venue} or {}
            return {
                "status": "no_markets",
                "reason": "No markets discovered from any venue",
                "health": health,
                "eligibility": {k: v.value for k, v in eligibility.items()},
                "qualification": qualification_report.reasoning if qualification_report else "No qualification",
                "execution_time": time.time() - start,
                "mission": self.mission,
                "discovery": {
                    "total_scanned": 0,
                    "per_venue": empty_venues,
                    "message": ("I scanned 0 markets across "
                                f"{len(markets_by_venue)} venues. No venue returned markets - "
                                "check credentials and eligibility before assuming a quiet market."),
                },
                # Key-for-key identical to the success path. run_continuous reads
                # result['opportunities']['final_selected'] every cycle, and
                # do_nothing_success is read by the loop and the dashboard - both
                # KeyErrored on this path, so a fresh deployment with no venue
                # credentials spun on errors forever instead of idling.
                "opportunities": {
                    "total_candidates": 0,
                    "total_tradeable": 0,
                    "final_selected": 0,
                    "best": {
                        "venue": None, "strategy": None,
                        "question": "DO NOTHING",
                        "edge": 0, "score": 0, "side": None,
                        "reasoning": "No markets discovered - nothing to evaluate",
                    },
                },
                "settlement": (settlement_report.to_dict()
                               if settlement_report else None),
                "alpha": {},
                "betting": {"ok": False, "events": 0, "data_mode": "none",
                            "blockers": ["no markets discovered"]},
                "execution": [],
                "do_nothing_success": True,
                "reasoning": ("No markets discovered from any venue, so no opportunities were "
                              "evaluated. This is not a signal to hold: a venue that fails to "
                              "return markets is unavailable, not quiet."),
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
                data_mode=self.data_mode,
                # Real capital needs a verified account. Until the account-health
                # probe can actually place and cancel an order this stays False,
                # so a live data mode cannot by itself deploy money.
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

        # The VENUE's view of the account, read before sizing.
        #
        # Free cash was computed from the local trade log alone, so a position
        # the venue holds and PTAI does not know about - placed by hand, or
        # filled while the process was down - was invisible, and the next cycle
        # could spend that capital again. The adapter's read is cached, so the
        # readiness ladder later in the cycle uses this same snapshot rather than
        # asking the venue a second time and possibly getting a different answer.
        venue_positions = None
        venue_state_complete = None
        portfolio = None
        try:
            # The venue that actually holds capital, READ from storage - the
            # same place the console reads it, so the agent and the operator
            # cannot disagree about which account is live. Until one is chosen,
            # the only venue with a real order path is the one worth asking.
            from ..strategy.venue_selection import VenueSelector
            live_venue = (VenueSelector(storage=self.storage)
                          .remembered_live_venue() or "polymarket")
            adapter = self.venue_registry.adapters.get(live_venue)
            if adapter is not None and hasattr(adapter, "get_portfolio"):
                portfolio = await adapter.get_portfolio()
                venue_positions = portfolio.get("venue_only_positions")
                # An unreadable account only blocks sizing when there is an
                # account to read. In paper mode no credentials exist yet, so the
                # venue legitimately has nothing to tell us and "could not read"
                # would be a false alarm that stops the simulation - which is
                # step 6 of the sequence, the thing that has to run before any
                # real capital moves. Armed for real orders, the verdict is
                # enforced.
                if bool(getattr(adapter, "can_place_real_orders", False)):
                    venue_state_complete = not portfolio.get(
                        "account_state_incomplete", True)
                else:
                    venue_state_complete = None
        except Exception as e:
            # Fail closed: an account we could not read is not a flat account.
            logger.warning(
                f"Venue account state unavailable ({type(e).__name__}: {e}) - "
                f"treating committed capital as unknown")
            venue_state_complete = False

        # Free cash, not bankroll. `bankroll` is still used for the per-trade
        # percentage caps because those are expressed against account equity,
        # but the amount actually deployable is what is uncommitted.
        ledger = self.ledger_builder.build(
            venue_positions=venue_positions,
            venue_state_complete=venue_state_complete)

        # ------------------------------------------------------------------
        # THE CONTROL BOUNDARY.
        #
        # Venue selection and the operator's authorised budget existed, and
        # neither was a gate at the point where money actually moved. The
        # selector said "one venue holds the live capital" and stored it; the
        # capital ledger computed an authorised budget per venue; and then the
        # execution loop dispatched to whatever venue the opportunity happened to
        # be on, sized against the GLOBAL bankroll. So the architecture's rule
        # and the system's behaviour were two different things:
        #
        #     selection:  only Polymarket is the live venue
        #     execution:  Polymarket live, and any other funded venue live too
        #
        # Selection and authorisation are now computed once per cycle and
        # enforced at three points below: the sizing cap, the pre-dispatch hard
        # gate, and the executor call itself.
        # ------------------------------------------------------------------
        self._cycle_live_venue, self._cycle_live_capital = \
            self._resolve_live_capital(
                portfolio=portfolio,
                free_cash=ledger.free_cash)
        free_capital = ledger.free_cash
        self.last_ledger = ledger
        logger.info(
            f"Capital: equity ${ledger.equity:.2f} = free ${ledger.free_cash:.2f} "
            f"+ reserved ${ledger.reserved_capital:.2f} "
            f"({ledger.live_position_count} live, {ledger.paper_position_count} paper); "
            f"realised ${ledger.realised_pnl:+.2f}, unrealised ${ledger.unrealised_pnl:+.2f}")
        for _warning in ledger.warnings:
            logger.warning(f"Ledger: {_warning}")
        
        # Separate qualified vs exploration.
        #
        # `not qualified_venue_ids` used to be OR-ed in here, which inverted the
        # gate at the worst possible moment: when NO venue had qualified,
        # everything was treated as qualified. Combined with the fact that no
        # venue qualifies on a fresh install (they are all unverified), the
        # default state of a new deployment was "live capital is permitted".
        #
        # The rule is now one-way: a venue that has not qualified can only ever
        # be explored in paper/shadow. An empty qualified set means the
        # qualified lane is empty, not that the gate is open.
        qualified_opps = []
        unqualified_opps = []
        for opp in scan_result.final_selected:
            base_venue = opp.venue_id.split("+")[0]
            is_qualified = (
                bool(qualified_venue_ids)
                and (opp.venue_id in qualified_venue_ids
                     or base_venue in qualified_venue_ids)
            )
            if is_qualified:
                qualified_opps.append(opp)
            else:
                unqualified_opps.append(opp)

        if not qualified_venue_ids:
            logger.warning(
                f"No venue qualified this cycle ({len(unqualified_opps)} "
                f"candidate(s) found). Everything goes to the paper/shadow "
                f"exploration lane - no live capital is deployable without a "
                f"qualified venue.")
        
        # V10 FIX #8: 95/5 split - 95% qualified, 5% exploration (shadow only)
        # Take top unqualified as exploration candidates (max 1 per cycle for $50 bankroll)
        exploration_candidates = sorted(unqualified_opps, key=lambda x: x.score, reverse=True)[:1] if unqualified_opps else []
        if exploration_candidates:
            logger.info(f"V10 FIX #8 Exploration lane: {len(exploration_candidates)} unqualified venues selected for shadow/paper learning (NO live capital): {[o.venue_id+':'+o.market.id for o in exploration_candidates]}")
        
        # Live-capital refusals that happen before dispatch (at SIZING, where a
        # cap can zero an order). Kept here because `execution_results` is
        # created after this loop, and an operator-visible refusal must not be
        # dropped on the floor just because the place that records it comes later.
        sizing_refusals: List[Dict[str, Any]] = []

        # Process qualified opportunities with consistent sizing
        for opp in qualified_opps:
            # V10 FIX #2: Calculate Kelly FIRST.
            #
            # This call used to pass edge=/prob=/confidence=, which are not
            # parameters of KellyCalculator.calculate() - its signature is
            # (market_price, fair_prob, bankroll) and it returns a KellyResult.
            # Every qualified opportunity therefore raised TypeError here and
            # no position was ever sized. No test caught it because the path
            # only runs once a venue has qualified, which nothing in the suite
            # did: the sizing code was reachable in principle and dead in
            # practice.
            #
            # Sizing against free_capital, not equity, so positions in one batch
            # cannot each claim 6% of the same dollars.
            kelly_result = self.kelly_calculator.calculate(
                market_price=opp.market_price,
                fair_prob=opp.estimated_fair,
                bankroll=free_capital,
            )
            if not kelly_result.should_bet:
                logger.info(
                    f"Kelly declines {opp.market.id}: {kelly_result.reason} "
                    f"(edge {kelly_result.edge:+.3f}, EV "
                    f"${kelly_result.expected_value:+.2f})")
                continue
            # kelly_fraction_adj is already half-Kelly and already capped at 6%
            # of the bankroll it was given; the min() below is a second cap
            # against equity, not against the free cash we just used.
            proposed_amount = free_capital * kelly_result.kelly_fraction_adj
            proposed_amount = min(proposed_amount, bankroll * 0.06)  # Cap 6% of equity

            # ...and by the capital THIS VENUE may actually spend.
            #
            # Sizing was global: free cash times Kelly, capped at 6% of the
            # account's equity. The operator's per-venue budget and the venue's
            # own balance were computed by the capital ledger and never consulted
            # here, so the agent could size a trade at a venue it had been
            # authorised $0 for - or more than the venue actually held.
            #
            # Only binds for capital that can move for real; paper sizing is
            # governed by free cash alone.
            venue_cap = None
            cap_detail = None
            cap_entry = (getattr(self, "_cycle_live_capital", None) or {}).get(
                opp.venue_id)
            if cap_entry is not None and not self._adapter_is_paper(opp):
                venue_cap = float(cap_entry.get("cap_usd") or 0.0)
                cap_detail = cap_entry.get("detail")
                if proposed_amount > venue_cap:
                    logger.info(
                        f"Live capital cap binds for {opp.market.id} @ "
                        f"{opp.venue_id}: ${proposed_amount:.2f} -> "
                        f"${venue_cap:.2f} ({cap_detail})")
                    proposed_amount = venue_cap
            proposed_amount = max(0, proposed_amount)
            
            if proposed_amount < 1.0:
                if venue_cap is not None and venue_cap < 1.0:
                    # A live order the capital boundary sized to nothing is not
                    # "too small to bother with" - it is a refusal by the
                    # boundary, and the operator's record has to name which one.
                    logger.warning(
                        f"LIVE CAPITAL BLOCKED {opp.market.id} @ {opp.venue_id} "
                        f"at sizing: ${venue_cap:.2f} deployable - "
                        f"{cap_detail}")
                    sizing_refusals.append(self._blocked_live_entry(
                        opp, proposed_amount,
                        cap_detail or "no live capital available"))
                    continue
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
            rules_ok, rules_reason = self._hard_rules_pass(opp)
            if not rules_ok:
                logger.info(f"Hard rule blocks {opp.market.id}: {rules_reason}")
                continue
            
            # V10 FIX #2: Use SAME proposed_amount for all risk checks
            #
            # Three consecutive calls in this risk chain were dead:
            # ExposureManager has no `can_open_position` (it is `can_open`),
            # LimitsEngine has no `validate` (it is `validate_proposal`), and
            # the Kelly call above passed parameters that do not exist. Each
            # would have raised on the first qualified opportunity. They were
            # written as a chain of independent guards, executed as a chain of
            # independent exceptions - and none of it had ever run, because no
            # venue had qualified.
            can_trade, reason = self.exposure_manager.can_open(
                market_id=opp.market.id,
                amount_usd=proposed_amount,  # V10 FIX: same amount, not $5
                category=opp.category,
                correlation_group=opp.correlation_group,
                venue=opp.venue_id,
            )
            if not can_trade:
                logger.info(f"Risk blocks {opp.market.id}: {reason} (amount ${proposed_amount:.2f})")
                continue
            
            # LimitsEngine reads a specific proposal shape and returns an
            # ADJUSTED order. It is given the fields it actually reads, or it
            # defaults `trade` to False and rejects every opportunity as "LLM
            # says no trade" - a rejection that looks like a risk decision and
            # is really a missing key.
            limits_ok, limits_reason, adjusted = self.limits_engine.validate_proposal({
                "market_id": opp.market.id,
                "side": opp.side,
                "venue_id": opp.venue_id,
                "trade": True,
                "fair_probability": opp.estimated_fair,
                "market_probability": opp.market_price,
                "edge": opp.effective_edge,
                "confidence": opp.confidence,
            })
            if not limits_ok:
                logger.info(f"Limits block {opp.market.id}: {limits_reason} (amount ${proposed_amount:.2f})")
                continue
            # Use the size limits approved, not the size we proposed.
            if isinstance(adjusted, dict):
                _approved = adjusted.get("max_spend_usd")
                if _approved is not None and float(_approved) < proposed_amount:
                    logger.info(
                        f"Limits reduced {opp.market.id} from "
                        f"${proposed_amount:.2f} to ${float(_approved):.2f}: "
                        f"{limits_reason}")
                    proposed_amount = float(_approved)
                opp._max_price = adjusted.get("max_price", opp.market_price)
            
            if not self.kill_switch.can_trade():
                logger.warning(f"Kill switch blocks trading L{self.kill_switch.current_level}")
                break
            
            # Attach calculated amounts to opportunity for execution
            opp._proposed_amount = proposed_amount
            opp._expected_ev = expected_ev
            
            logger.info(f"Core Objective PASS: {opp.market.id} @ {opp.venue_id} edge {opp.effective_edge*100:.1f}% conf {opp.confidence:.2f} score {opp.score:.3f} amount ${proposed_amount:.2f} netEV ${expected_ev.net_ev_usd:.2f} - ALL independently enforced rules PASSED (consistent sizing)")
            final_trades.append(opp)
        
        # V10 FIX #8: Exploration lane - shadow/paper only, no live capital, for
        # learning.
        #
        # These candidates used to be collected here, marked, appended to a list,
        # and then COUNTED IN A LOG LINE. Nothing executed them. The consequence
        # was a deadlock on every fresh installation:
        #
        #     no qualified venues -> pick an exploration candidate
        #     -> do not execute it -> zero trade outcomes
        #     -> qualification stays at zero -> no qualified venues -> repeat
        #
        # The agent could never bootstrap its own qualification evidence from
        # zero, so the "100 paper trades -> qualified -> live" lifecycle was
        # unreachable by construction. They now go through the SAME execution
        # path as everything else, forced to paper, so they produce real
        # simulated fills, real positions and real learning records.
        for opp in exploration_candidates:
            opp._proposed_amount = 1.0  # minimal shadow
            opp._is_exploration = True
            # Measure the shadow trade the same way a qualified one is measured.
            #
            # Exploration is the ONLY source of evidence on a fresh install, so
            # the venue's qualification rests on these rows. Scored records carry
            # the expected net EV that was predicted BEFORE the trade; unscored
            # ones carry NULL, and a gate that requires measured EV coverage
            # would then be unsatisfiable for exactly the venue that has only
            # exploration evidence - the bootstrap deadlock, one level down. The
            # shadow size is what the EV is computed on, because fees, slippage
            # and uncertainty all scale with the amount.
            try:
                opp._expected_ev = self.expected_ev_engine.calculate(
                    opportunity=opp,
                    amount_usd=opp._proposed_amount,
                    orderbook=(opp.market.raw.get("orderbook", {})
                               if hasattr(opp.market, "raw") else {}),
                )
            except Exception as e:
                # Refusing to explore is worse than exploring unscored, but the
                # refusal is never silent: the row says the EV was not measured.
                logger.warning(
                    f"Exploration EV unavailable for {opp.market.id} @ "
                    f"{opp.venue_id}: {type(e).__name__}: {e} - the trade is "
                    f"still simulated, and its outcome will carry no expected EV")
                opp._expected_ev = None
            exploration_trades.append(opp)
            _ev = getattr(opp._expected_ev, "net_ev_usd", None)
            logger.info(f"Exploration SHADOW: {opp.market.id} @ {opp.venue_id} score {opp.score:.3f} amount ${opp._proposed_amount:.2f} netEV ${_ev if _ev is not None else 'unmeasured'} - shadow/paper only, NO live capital, for discovering new edges")

        # The exploration lane is downstream of qualification, so on a fresh
        # install it is the ONLY source of evidence. Route it into the execution
        # loop, after the qualified trades so it can never displace a real one.
        for opp in exploration_trades:
            if opp not in final_trades:
                final_trades.append(opp)
        
        # Execution - V10 FIX #1: ONE canonical execution path via MultiVenueExecutor
        # Architecture: V3 → ExecutionGuard → MultiVenueExecutor → Exact venue adapter → place_order()
        # Previously bypassed executor: V3 → Guard → adapter.place_order() directly - FIXED
        # V10 FIX #2: Use same Kelly-calculated amount through all checks (no $5 placeholder)
        execution_results = list(sizing_refusals)
        
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
                
                # The opportunity is passed through so the order probe can test
                # against the very market we are about to trade. Permission is
                # demonstrated on a real market or not at all.
                account_health = await self.account_health_engine.check_venue_health(
                    venue_id, opportunity=opp)
                # Cached for the venue selection at the end of the cycle. These
                # reads already happened, so reporting which venues are funded
                # and verified costs no additional calls.
                try:
                    self._last_account_health[venue_id] = account_health.to_dict()
                except Exception as e:
                    logger.debug(f"Could not cache account health for {venue_id}: {e}")
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
                    # Fallback if not set (should not happen). Same corrected
                    # signature as the primary sizing path above.
                    _kr = self.kelly_calculator.calculate(
                        market_price=opp.market_price,
                        fair_prob=opp.estimated_fair,
                        bankroll=current_bankroll,
                    )
                    if not _kr.should_bet:
                        logger.info(
                            f"Kelly declines {opp.market.id} at the execution "
                            f"gate: {_kr.reason}")
                        continue
                    amount_usd = current_bankroll * _kr.kelly_fraction_adj
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
                
                # THE HARD GATE. Everything upstream may have approved this
                # trade, and none of it decides where real money goes: the
                # selected live venue and the operator's authorised budget do,
                # here, immediately before the order is placed.
                live_ok, live_reason = self._live_execution_allowed(opp, amount_usd)
                if not live_ok:
                    logger.warning(
                        f"LIVE CAPITAL BLOCKED {opp.market.id} @ {opp.venue_id} "
                        f"${amount_usd:.2f}: {live_reason}")
                    execution_results.append(
                        self._blocked_live_entry(opp, amount_usd, live_reason))
                    continue

                logger.info(f"Core Objective DEPLOY via canonical executor: {opp.market.id} @ {opp.venue_id} amount ${amount_usd:.2f} netEV ${getattr(opp, '_expected_ev', None).net_ev_usd if hasattr(opp, '_expected_ev') and opp._expected_ev else 0:.2f} - Guard PASS → MultiVenueExecutor → {venue_id} | {live_reason}")
                
                # V10 FIX #1: ONE canonical path: MultiVenueExecutor.execute_single()
                # Executor has: MOCK protection, exact routing ABORT, rate limits, min order checks, fee calc, gas
                #
                # The price cap is on the side being bought. It used to be
                # market_price + 0.02 - the YES price - which for a NO order caps
                # the wrong token and either rejects a valid order or lets it
                # through at a price the NO token never trades at.
                exec_result = await self._execute_with_side_aware_cap(
                    opp=opp, amount_usd=amount_usd,
                    exploration=bool(getattr(opp, "_is_exploration", False)))
                
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
                
                # --- ONLY a proven fill may become a position ------------------
                #
                # Recording used to happen unconditionally after execution, so a
                # result of rejected, error, rate_limited or blocked still
                # created an "open" position - sized at the REQUESTED amount and
                # priced at the REQUESTED price, because the executor copied
                # those straight through. That is a position in the ledger that
                # never existed at the venue: settlement would later close it
                # against a real outcome and book a P&L for a trade that never
                # happened, and the bankroll would drift away from reality.
                if not exec_result.should_record_position:
                    logger.warning(
                        f"NO POSITION: {opp.market.id} @ {venue_id} status="
                        f"{exec_result.status} - recorded nothing. {exec_result.reasoning[:120]}")
                    execution_results[-1]["position_recorded"] = False
                    execution_results[-1]["position_reason"] = (
                        f"execution status {exec_result.status} committed no "
                        f"capital, so no position exists to settle"
                    )
                    # No position - but possibly still an ORDER. A resting order
                    # reserves capital at the venue, and a send whose response
                    # was lost may be resting too. Both have to be recorded or
                    # the next cycle cannot ask the venue about them, the
                    # reserved cash is invisible to sizing, and a later fill
                    # never becomes a position.
                    if exec_result.reserves_capital or exec_result.needs_reconciliation:
                        order_key = self.order_manager.record_submission(
                            exec_result, market_id=opp.market.id,
                            token_id=getattr(opp.market, "tokens", [{}])[0].token_id
                            if getattr(opp.market, "tokens", None) else None,
                            venue_id=venue_id, side=str(opp.side).upper(),
                        forecast=self._forecast_for_order(
                            opp, getattr(opp.market, "data_mode", "live")))
                        execution_results[-1]["order_recorded"] = bool(order_key)
                        execution_results[-1]["order_key"] = order_key
                        if order_key:
                            logger.info(
                                f"Order {order_key} recorded for reconciliation: "
                                f"status {exec_result.status}, "
                                f"${exec_result.resting_usd:.4f} reserved, "
                                f"unfilled {exec_result.unfilled_shares} shares")
                    continue

                execution_results[-1]["position_recorded"] = True
                execution_results[-1]["fill"] = exec_result.to_position_dict()

                # Draw the committed amount down as we go. Within one cycle the
                # ledger is a snapshot taken before any of these trades, so
                # without this every position in the batch would size against
                # the same free cash - the original bug, one level down.
                _committed = (exec_result.filled_usd
                              if exec_result.committed_capital else amount_usd)
                free_capital = max(0.0, free_capital - _committed)
                if free_capital <= 0:
                    logger.warning(
                        "Free capital exhausted mid-cycle - no further positions "
                        "will be opened in this batch.")

                # --- Record the position so it can later be SETTLED -----------
                #
                # This block did not persist anything usable:
                #   * `except: pass` around trade_outcome_tracker.record_trade
                #     hid any failure inside it. A silent failure to record is
                #     worse than a loud one: the agent keeps trading on unlearned
                #     priors while the logs look fine.
                #   * the trade was never written to the trades table at all, so
                #     there was nothing for settlement to close, nothing for win
                #     rate to be computed over, and nothing for the learning
                #     chain to attribute an outcome to.
                #   * the forecast carried no venue or trade id, so even a
                #     settlement could not have routed it back to a position.
                #
                # Failures here are now loud and recorded on the execution result,
                # so a position that was taken but cannot be learned from is
                # visible rather than assumed.
                # TWO labels, and they answer different questions.
                #
                # `data_mode` describes the market data - real books and prices
                # from a live venue. `execution_mode` describes what happened to
                # the money. An exploration trade is executed in paper against
                # live data, so it is data_mode=live AND execution_mode=paper,
                # and the learning chain must read the second one or it counts
                # simulated trades as live outcomes.
                data_mode_for_calib = getattr(opp.market, 'data_mode', 'live')
                if hasattr(data_mode_for_calib, 'value'):
                    data_mode_for_calib = data_mode_for_calib.value
                execution_mode = "paper" if exec_result.is_simulated else "live"

                # The price actually paid per share, on the token being bought.
                # `filled_price` is that price; `opp.market_price` is the YES
                # price, so it is only usable as the token price for a YES buy.
                yes_price_at_entry = float(opp.market_price or 0.0)
                if exec_result.filled_price:
                    token_price_at_entry = float(exec_result.filled_price)
                elif str(opp.side or "").upper() == "YES":
                    token_price_at_entry = yes_price_at_entry
                else:
                    token_price_at_entry = round(1.0 - yes_price_at_entry, 6)
                strategy_name = (opp.raw.get("strategy", "unknown")
                                 if hasattr(opp, 'raw') and isinstance(opp.raw, dict)
                                 else "unknown")
                learning_problems = []

                trade_id = None
                try:
                    trade_id = self.storage.log_trade({
                        "market_id": opp.market.id,
                        "market_question": opp.market.question[:200],
                        "side": opp.side,
                        # The FILL price, not the requested max price. A limit
                        # order at 0.52 that filled at 0.49 changes the P&L and
                        # the edge, and settlement uses this price. For a paper
                        # fill it is the average of the levels the simulation
                        # walked, which is how slippage enters the P&L at all.
                        # Kept as the YES price it means to the strategy, with
                        # the token price beside it. This column used to hold
                        # the fill price when there was one and the YES price
                        # when there was not, so it meant two different numbers
                        # in the same column - and settlement, which needs the
                        # token price, was given whichever it happened to be.
                        "yes_price_at_entry": yes_price_at_entry,
                        "token_price_at_entry": token_price_at_entry,
                        "market_price": yes_price_at_entry,
                        "execution_mode": execution_mode,
                        "fees_usd": getattr(exec_result, "fees_usd", None),
                        "fair_value": opp.estimated_fair,
                        "edge": opp.effective_edge,
                        "kelly_fraction": getattr(opp, "_kelly_fraction", None),
                        # The FILLED size, not the intended size.
                        # What actually went into the position, for real AND
                        # paper. A simulated fill is usually smaller than the
                        # request, because the book has finite depth; charging
                        # the ledger the requested amount overstates exposure and
                        # is how a paper equity curve drifts into fiction.
                        # Guaranteed positive by should_record_position, which
                        # refuses a simulated result that filled nothing. The old
                        # `or amount_usd` fallback was a live trap: a paper fill
                        # of $0 booked the whole request.
                        "position_size_usd": exec_result.position_size_usd,
                        "position_size_pct": (amount_usd / current_bankroll
                                              if current_bankroll else None),
                        "confidence": opp.confidence,
                        # A simulated execution is recorded as paper, never as a
                        # live position holding real capital.
                        "status": "paper" if exec_result.is_simulated else "open",
                        "notes": f"venue={venue_id} strategy={strategy_name} "
                                 f"mode={data_mode_for_calib} "
                                 f"exec_status={exec_result.status} "
                                 f"order_id={exec_result.order_id} "
                                 f"filled=${exec_result.filled_usd:.2f}"
                                 f"@{exec_result.filled_price or 0:.4f}",
                    })
                except Exception as e:
                    learning_problems.append(f"trade not persisted: {type(e).__name__}: {e}")
                    logger.error(
                        f"TRADE NOT PERSISTED for {opp.market.id}: {e}. Position "
                        f"cannot be settled, so it cannot be learned from.")

                # Link the order to the position it produced. A partial fill
                # that later completes must grow THIS row: settlement finds an
                # open trade by market id, so a second row for the same market
                # would never be closed and its P&L would never be realised.
                if trade_id and (exec_result.needs_reconciliation
                                 or exec_result.unfilled_shares > 0):
                    order_key = self.order_manager.record_submission(
                        exec_result, market_id=opp.market.id,
                        token_id=getattr(opp.market, "tokens", [{}])[0].token_id
                        if getattr(opp.market, "tokens", None) else None,
                        venue_id=venue_id, trade_id=int(trade_id),
                        side=str(opp.side).upper(),
                        forecast=self._forecast_for_order(
                            opp, getattr(opp.market, "data_mode", "live")))
                    execution_results[-1]["order_recorded"] = bool(order_key)
                    execution_results[-1]["order_key"] = order_key
                    if order_key:
                        logger.info(
                            f"Order {order_key} linked to position {trade_id} "
                            f"({exec_result.unfilled_shares} shares still working); "
                            f"further fills will grow this position")

                # Forecast, tied to the venue and the trade so settlement can
                # find it and close the position when the market resolves.
                try:
                    self.calibration_engine.record_forecast(
                        market_id=opp.market.id,
                        question=opp.market.question[:200],
                        forecast_prob=opp.estimated_fair,
                        confidence=opp.confidence,
                        market_price=opp.market_price,
                        category=opp.category,
                        venue_id=venue_id,
                        trade_id=trade_id,
                    )
                except Exception as e:
                    learning_problems.append(f"forecast not recorded: {type(e).__name__}: {e}")
                    logger.error(
                        f"FORECAST NOT RECORDED for {opp.market.id}: {e}. The "
                        f"probability behind this trade will not be calibrated.")

                # Venue/strategy/category performance, for allocation later.
                try:
                    # The fields this tracker actually needs. It used to
                    # receive confidence/data_mode/trust_tier and none of
                    # trade_id/category/forecast_prob/market_price/side, so
                    # every call raised TypeError and no venue or strategy
                    # outcome was ever recorded.
                    self.trade_outcome_tracker.record_trade(
                        trade_id=trade_id,
                        market_id=opp.market.id,
                        venue_id=venue_id,
                        strategy=strategy_name,
                        category=opp.category,
                        forecast_prob=opp.estimated_fair,
                        market_price=(exec_result.filled_price
                                      or opp.market_price),
                        edge=opp.effective_edge,
                        side=opp.side,
                        confidence=opp.confidence,
                        # PAPER or LIVE - the money, not the data.
                        execution_mode=execution_mode,
                        # The expected net EV computed BEFORE the trade was
                        # taken, so qualification can weigh what was actually
                        # predicted instead of the average edge.
                        #
                        # Read from THIS opportunity, not from a local: `expected_ev`
                        # belongs to the sizing loop, and on the exploration path
                        # it was never assigned at all - passing it raised
                        # UnboundLocalError, which refused the whole learning
                        # record. An outcome that is not written teaches nothing.
                        # None when this trade was never scored, which the gate
                        # treats as unmeasured rather than as zero.
                        expected_net_ev=getattr(
                            getattr(opp, "_expected_ev", None), "net_ev_usd", None),
                        expected_net_ev_pct=getattr(
                            getattr(opp, "_expected_ev", None), "net_ev_pct", None),
                        # The market data mode, kept for context. The
                        # paper/live split reads execution_mode, above.
                        data_mode=str(data_mode_for_calib),
                        # What the trade actually cost. Recorded so the
                        # qualification gate can MEASURE fees, slippage and
                        # execution quality instead of reading the placeholders
                        # it used to hold - a constant that happened to equal the
                        # execution-quality threshold, so every venue passed it.
                        fees_usd=getattr(exec_result, "fees_usd", None),
                        slippage_bps=(
                            (getattr(exec_result, "paper_fill", None) or {})
                            .get("slippage_bps")),
                        execution_quality=self._execution_quality(exec_result),
                        # WHICH BOOK this fill was priced against, from the
                        # broker that walked it. A paper fill can be perfectly
                        # simulated and still be evidence about nothing; the
                        # qualification gate refuses a sample that is mostly
                        # priced against an assumed book, and it can only do
                        # that if the label reaches the outcome row.
                        # THE SAME EV, REPRICED AT THE FILL THAT HAPPENED.
                        #
                        # `expected_net_ev` above is the prediction, made at the
                        # price the book showed when the opportunity was found.
                        # These are that quantity recomputed at the price the
                        # order actually paid and the fees it actually incurred,
                        # so the qualification gate can judge what the fills were
                        # worth instead of trusting a model of execution. The
                        # slippage is already inside `filled_price` - the ladder
                        # walk is how the order got there - so it is not
                        # deducted a second time here.
                        **self._executable_ev_fields(
                            opp=opp,
                            exec_result=exec_result,
                            # Read from THIS opportunity, never from a local:
                            # `expected_ev` belongs to the sizing loop and on the
                            # exploration path it is never assigned, so passing
                            # it raises UnboundLocalError inside the record call
                            # and the whole outcome - score, costs, forecast and
                            # all - is lost. Reading it here rather than the
                            # local is what stopped that happening before.
                            modelled=getattr(opp, "_expected_ev", None),
                            amount_usd=amount_usd,
                        ),
                        book_source=(
                            (getattr(exec_result, "paper_fill", None) or {})
                            .get("book_source")
                            # A live fill is its own evidence: real money,
                            # real book, no simulation involved.
                            or ("venue_fill" if execution_mode == "live" else "")),
                        gas_usd=getattr(exec_result, "gas_usd", None),
                        amount_usd=(exec_result.filled_usd
                                    if exec_result.committed_capital
                                    else amount_usd),
                    )
                except Exception as e:
                    learning_problems.append(f"venue/strategy outcome not recorded: {type(e).__name__}: {e}")
                    logger.error(
                        f"VENUE OUTCOME NOT RECORDED for {opp.market.id}: {e}. "
                        f"Allocation will keep treating this venue as untested.")

                if learning_problems:
                    logger.error(
                        f"LEARNING GAPS on {opp.market.id}: " + "; ".join(learning_problems))
                    execution_results[-1]["learning_problems"] = learning_problems
                    execution_results[-1]["learning_complete"] = False
                else:
                    execution_results[-1]["learning_complete"] = True
                
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
            "capital": self.last_ledger.to_dict() if getattr(self, "last_ledger", None) else None,
            "settlement": settlement_report.to_dict() if settlement_report else None,
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
                    "reasoning": scan_result.best_opportunity.reasoning[:200] if scan_result.best_opportunity else "No edge found - DO NOTHING is successful",
                    # The number the choice was actually made on: NET EV per
                    # dollar-day of locked capital per unit of execution risk,
                    # with the denominator kept so the figure can be argued
                    # with rather than just believed.
                    "capital_efficiency": (
                        (scan_result.best_opportunity.raw or {}).get("capital_efficiency")
                        if scan_result.best_opportunity else None),
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
            # Which venue the money is on, and why. Computed from the evidence
            # this cycle gathered: qualification, the balances the health engine
            # read, and the per-venue results in the trades table.
            "venue_selection": self._venue_selection(qualified_venue_ids),
            "reasoning": scan_result.reasoning,
            "execution_time": elapsed,
            "do_nothing_success": len(final_trades) == 0
        }
        
        # .get() throughout: a cycle that returns an unexpected shape must not
        # be able to kill the loop. A long run has to survive its own reporting.
        _n_trades = (result.get("opportunities") or {}).get("final_selected", 0)
        logger.info(f"=== PTAI V3 Cycle Complete in {elapsed:.1f}s: {_n_trades} trades, DO NOTHING success: {result.get('do_nothing_success', True)} ===")
        logger.info(f"V3 Report: {scan_result.reasoning}")

        # Write down WHAT WAS DECIDED, so the answer survives the process.
        #
        # "What did the agent do last cycle, and why?" is a question the operator
        # asks between cycles, from a console that is not holding the cycle in
        # memory - and after a restart there is nothing to ask. The full cycle
        # result is per-venue detail measured in hundreds of kilobytes, which
        # would make this a log, not state; this is the decision and the reasons
        # behind it.
        try:
            selection = result.get("venue_selection") or {}
            execution = result.get("execution") or []
            blocked = [e for e in execution
                       if e.get("status") == "blocked_live_capital"]
            recorded = [e for e in execution if e.get("position_recorded")]
            ledger_dict = (self.last_ledger.to_dict()
                           if getattr(self, "last_ledger", None) else {})
            best = ((result.get("opportunities") or {}).get("best") or {})
            self.storage.set_state("operator.last_cycle", json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "verdict": ("DEPLOYED" if recorded else "DO NOTHING"),
                "why": scan_result.reasoning[:400] if scan_result.reasoning else "",
                "cycle_seconds": round(elapsed, 1),
                "venues_searched": len(scan_result.venue_reports),
                "markets_scanned": scan_result.total_scanned,
                "candidates": scan_result.total_candidates,
                "qualified_venues": list(qualified_venue_ids),
                "decided": {
                    "capital_efficiency": best.get("capital_efficiency"),
                    "venue": best.get("venue"),
                    "strategy": best.get("strategy"),
                    "question": best.get("question"),
                    "side": best.get("side"),
                    "edge": best.get("edge"),
                    "score": best.get("score"),
                    "reasoning": best.get("reasoning"),
                },
                "live_venue": selection.get("live_venue"),
                "live_venue_verdict": selection.get("verdict"),
                "venue_to_fund": selection.get("candidate"),
                "venue_reasons": list(selection.get("reasons") or [])[:3],
                "orders": {
                    "attempted": len(execution),
                    "positions_recorded": len(recorded),
                    "blocked_live_capital": len(blocked),
                    "blocked_reason": blocked[0].get("reason") if blocked else None,
                },
                "settlement": result.get("settlement") or {},
                "capital": {
                    "equity": ledger_dict.get("equity"),
                    "free_cash": ledger_dict.get("free_cash"),
                    "reserved_capital": ledger_dict.get("reserved_capital"),
                    "realised_pnl": ledger_dict.get("realised_pnl"),
                },
                "risk": {
                    "kill_switch_level": int(self.kill_switch.current_level),
                    "can_trade": bool(self.kill_switch.can_trade()),
                },
            }))
        except Exception as e:
            # Never silently: a cycle that cannot be reported is a decision the
            # operator cannot audit, which is worse than a noisy log line.
            logger.warning(
                f"Could not record the cycle decision for the operator view: "
                f"{type(e).__name__}: {e}")

        return result

    async def _execute_with_side_aware_cap(self, opp, amount_usd: float,
                                           exploration: bool = False):
        """
        Send the order with a price cap on the side actually being bought, and
        with real money forbidden for exploration.

        Two corrections in one place, because both are properties of the call:

        1. THE CAP. `opp.market_price` is the YES price. A NO order buys the NO
           token, which trades near 1 - yes, so a cap of yes + 0.02 is a cap on a
           price that token never reaches - the order is rejected as
           non-marketable, or worse, accepted at a nonsense limit. The cap is
           derived from the side, using the opportunity's own price for that side
           when it has one.

        2. EXPLORATION IS PAPER. The adapter's dry_run flag is the last gate
           before real money, so it is set for the duration of this call and
           restored in a finally. Exploration exists to earn qualification, and
           qualification is earned in paper - an exploration trade that could
           touch real capital would be live trading at an unqualified venue,
           which is the one thing the qualification gate exists to prevent.
        """
        if str(opp.side).upper() == "NO":
            yes_price = float(opp.market_price)
            cap = max(0.01, min(0.99, 1.0 - yes_price + 0.02))
        else:
            cap = float(opp.market_price) + 0.02

        adapter = None
        saved_dry_run = None
        if exploration:
            adapter = self.venue_registry.get_adapter_for_market(opp.market)
            if adapter is not None:
                saved_dry_run = adapter.dry_run
                adapter.dry_run = True
                logger.info(
                    f"Exploration is PAPER-ONLY: {opp.market.id} @ {opp.venue_id} "
                    f"executed with the adapter forced to dry_run for this call "
                    f"so it can never touch live capital")
        try:
            return await self.multi_venue_executor.execute_single(
                opportunity=opp,
                max_spend_usd=amount_usd,
                max_price=cap,
            )
        finally:
            if adapter is not None and saved_dry_run is not None:
                adapter.dry_run = saved_dry_run

    def _resolve_live_capital(self, portfolio, free_cash: float) -> tuple:
        """
        Which venue may spend real money this cycle, and how much.

        Returns (live_venue_id, {venue_id: {"cap_usd": float, "detail": str}}).

        The cap is the smallest of the three things that all have to agree before
        real money moves:

          * the VENUE's own available balance, as the venue reported it,
          * the budget the OPERATOR authorised for that venue,
          * local free cash, i.e. what is not already committed.

        A missing term is not treated as unlimited. No venue balance read means no
        venue-confirmed money to spend, which means no live deployment - the agent
        may still paper-trade there, and paper is where it belongs until the
        account can be read.
        """
        from ..execution.capital import authorised_budget
        from ..strategy.venue_selection import VenueSelector

        caps: Dict[str, Any] = {}
        try:
            live_venue = VenueSelector(storage=self.storage).remembered_live_venue()
        except Exception as e:
            logger.warning(f"Could not read the selected live venue: {e}")
            live_venue = None

        if not live_venue:
            return None, caps

        try:
            authorised = float(authorised_budget(self.storage, live_venue) or 0.0)
        except Exception as e:
            logger.warning(f"Could not read the authorised budget: {e}")
            authorised = 0.0

        venue_balance = None
        if isinstance(portfolio, dict):
            raw = portfolio.get("venue_confirmed_balance")
            try:
                venue_balance = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                venue_balance = None

        parts = {
            "venue_balance_usd": venue_balance,
            "authorised_usd": authorised,
            "free_cash_usd": float(free_cash or 0.0),
        }
        # Live capital needs all three. Any one missing is a refusal, not a
        # default: venue balance unread -> fail closed; nothing authorised ->
        # the operator has not given this agent the money; free cash 0 -> already
        # committed.
        if venue_balance is None:
            caps[live_venue] = {
                "cap_usd": 0.0, "parts": parts,
                "detail": ("the venue's balance could not be read, so no live "
                           "capital is deployable at this venue")}
        elif authorised <= 0:
            caps[live_venue] = {
                "cap_usd": 0.0, "parts": parts,
                "detail": (f"no budget authorised for {live_venue}: paper only "
                           f"until the operator sets one")}
        else:
            cap = max(0.0, min(venue_balance, authorised, float(free_cash or 0.0)))
            caps[live_venue] = {
                "cap_usd": cap, "parts": parts,
                "detail": (f"min(venue ${venue_balance:.2f}, authorised "
                           f"${authorised:.2f}, free ${float(free_cash or 0.0):.2f})")}
        return live_venue, caps

    def _live_execution_allowed(self, opp, amount_usd: float) -> tuple:
        """
        May this specific order move real money? Returns (allowed, reason).

        The last gate before dispatch, and the one that makes the architecture
        true: if the opportunity's venue is not the selected live venue, this
        trade does not touch real capital regardless of how good it looks or how
        qualified the venue is. Paper is not affected - an unarmed adapter has no
        real money to move - and neither is exploration, which is forced to paper
        before it reaches here.
        """
        venue_id = str(getattr(opp, "venue_id", "") or "")
        if not venue_id:
            return False, "opportunity has no venue_id, so its capital is unaccountable"

        # Paper executions are governed by the paper engine, not this boundary.
        try:
            adapter = self.venue_registry.adapters.get(venue_id)
        except Exception:
            adapter = None
        armed = bool(getattr(adapter, "can_place_real_orders", False))
        if not armed:
            return True, "paper: the adapter cannot place real orders"
        if getattr(opp, "_is_exploration", False):
            return True, "exploration: forced to paper before dispatch"

        live_venue = getattr(self, "_cycle_live_venue", None)
        if not live_venue:
            return False, ("no live venue is selected, so real capital may not "
                           "be deployed anywhere")
        if venue_id != live_venue:
            return False, (f"{venue_id} is not the selected live venue "
                           f"({live_venue}) - one venue holds the live capital "
                           f"at a time")

        entry = (getattr(self, "_cycle_live_capital", None) or {}).get(venue_id)
        if not entry:
            return False, (f"no authorised capital computed for {venue_id} this "
                           f"cycle")
        cap = float(entry.get("cap_usd") or 0.0)
        if cap <= 0:
            return False, entry.get("detail") or "no live capital available"
        if float(amount_usd) > cap + 1e-9:
            return False, (f"amount ${float(amount_usd):.2f} exceeds the live "
                           f"capital cap ${cap:.2f} for {venue_id}")
        return True, f"within {entry.get('detail')}"

    def _hard_rules_pass(self, opp):
        """
        Rule 1 of the core objective, in one place.

        The 8% here was applied to the POST-COST edge, so it charged the fees,
        spread, slippage and uncertainty penalties a second time - on top of the
        terms in the net EV gate three lines above, which is where costs belong.
        A NO trade that cleared net EV (+$1.07 on a $3 stake) was refused by
        this line because only 2.8% of edge had survived the costs.

        The 8% is the mispricing rule the strategy hunts on; the costs are
        enforced once, as a positive effective edge here and as net EV above.

        Returns (ok, reason) so the live gate can be driven directly by a test
        instead of only being reachable through a whole cycle.
        """
        mispricing = hunted_mispricing(opp)
        if mispricing < HUNT_MISPRICING_MIN:
            return False, (f"mispricing {mispricing*100:.1f}% < 8% "
                           f"(effective {opp.effective_edge*100:.1f}% after costs)")
        if opp.effective_edge <= 0:
            return False, (f"costs consumed the edge: effective "
                           f"{opp.effective_edge*100:.1f}% from mispricing "
                           f"{mispricing*100:.1f}%")
        if opp.confidence < 0.60:
            return False, f"confidence {opp.confidence:.2f} < 60%"
        if hasattr(opp, 'liquidity_score') and opp.liquidity_score < 0.3:
            return False, f"liquidity {opp.liquidity_score:.2f} < 0.3"
        if hasattr(opp, 'execution_quality') and opp.execution_quality < 0.3:
            return False, f"execution_quality {opp.execution_quality:.2f} < 0.3"
        return True, (f"mispricing {mispricing*100:.1f}% effective "
                      f"{opp.effective_edge*100:.1f}% conf {opp.confidence:.2f}")

    @staticmethod
    def _executable_ev_fields(opp, exec_result, modelled,
                              amount_usd: float) -> dict:
        """
        The executable-EV columns for one outcome row, or an empty dict.

        Empty is a deliberate outcome: a trade that never filled has no fill to
        reprice at, and the columns stay NULL so the gate reads "not measured"
        rather than a zero that looks like a break-even execution.
        """
        filled_usd = getattr(exec_result, "filled_usd", None)
        filled_price = getattr(exec_result, "filled_price", None)
        if not filled_usd or not filled_price:
            return {}

        modelled_pct = getattr(modelled, "net_ev_pct", None)
        # The price the DECISION was made at, in the same token space as the
        # fill, so `fill_price_vs_modelled` compares like with like. A NO
        # position models the NO token; the venue reports the NO price.
        modelled_price = None
        try:
            if str(getattr(opp, "side", "YES")).upper() == "YES":
                modelled_price = float(opp.market_price)
            else:
                modelled_price = 1.0 - float(opp.market_price)
        except (TypeError, ValueError):
            modelled_price = None

        try:
            result = executable_net_ev(
                opportunity=opp,
                filled_usd=float(filled_usd),
                filled_price=float(filled_price),
                fees_usd=float(getattr(exec_result, "fees_usd", 0.0) or 0.0),
                gas_usd=float(getattr(exec_result, "gas_usd", 0.0) or 0.0),
                modelled_net_ev_pct=modelled_pct,
                modelled_price=modelled_price,
            )
        except Exception as e:
            # An unpriceable fill must not cost the whole learning record - the
            # outcome row is worth more than these four columns, and a lost row
            # teaches nothing. The columns stay NULL, which the gate reads as
            # "not measured" and fails closed on, and the failure is logged
            # rather than swallowed.
            logger.error(
                f"Could not reprice {getattr(opp, 'market_id', 'unknown')} at "
                f"its fill ({type(e).__name__}: {e}); the outcome is recorded "
                f"WITHOUT an executable EV, and the qualification gate will "
                f"treat it as unmeasured")
            return {}
        if result is None:
            return {}
        return {
            "executable_net_ev": result.net_ev_usd,
            "executable_net_ev_pct": result.net_ev_pct,
            "fill_price_vs_modelled": result.price_paid_vs_modelled,
        }

    @staticmethod
    def _execution_quality(exec_result) -> Optional[float]:
        """
        How well the order executed, on 0-1, from what was actually observed.

        Measured from slippage against the touch: a fill at the best available
        price is 1.0, and 5% worse than the touch is 0.0. Returns None when
        nothing was measured, so the record says "not measured" rather than
        asserting a quality nobody observed - the qualification gate fails closed
        on an unmeasured value, which is the point.

        Realised slippage is the only execution fact the venue actually reports.
        A score derived from anything else would be a number about nothing.
        """
        paper = getattr(exec_result, "paper_fill", None) or {}
        bps = paper.get("slippage_bps")
        if bps is None:
            return None
        try:
            return max(0.0, 1.0 - abs(float(bps)) / 500.0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _forecast_for_order(opp, data_mode) -> Dict[str, Any]:
        """
        The thesis behind an order, in the shape the orders table stores.

        Written down at submission so a fill that arrives hours later is still
        attributed to the strategy that chose the trade - rather than to
        "resting_order_fill" with edge 0, which teaches the agent nothing about
        the decision it actually made.
        """
        # The entry prices are recorded on the SIDE being bought: `token_price`
        # is what a share of the token this order is for costs, `yes_price` is
        # the same market on the YES scale. Both, so a fill that arrives later
        # cannot confuse them - which is exactly how a NO position ended up
        # settling on the wrong price.
        yes_price = float(getattr(opp, "market_price", 0.0) or 0.0)
        side = str(getattr(opp, "side", "YES") or "YES").upper()
        bought_yes = side in ("YES", "1", "LONG", "BUY", "TRUE")
        token_price = yes_price if bought_yes else round(1.0 - yes_price, 6)

        expected_ev = getattr(opp, "_expected_ev", None)
        return {
            "fair_price": getattr(opp, "estimated_fair", None),
            "edge": getattr(opp, "effective_edge", None),
            "confidence": getattr(opp, "confidence", None),
            "strategy": getattr(opp, "strategy", None) or getattr(opp, "strategy_name", None),
            "category": getattr(opp, "category", None),
            "data_mode": str(getattr(data_mode, "value", data_mode)),
            # What the trade was predicted to earn, at entry.
            "expected_net_ev": getattr(expected_ev, "net_ev_usd", None),
            "expected_net_ev_pct": getattr(expected_ev, "net_ev_pct", None),
            # The prices, on both scales.
            "yes_price": yes_price,
            "token_price": token_price,
            # PAPER or LIVE. Read from the adapter rather than assumed: a
            # dry-run venue simulates, and that is what the row must say.
            "execution_mode": ("paper" if self._adapter_is_paper(opp)
                               else "live"),
        }

    def _blocked_live_entry(self, opp, amount_usd: float, reason: str) -> Dict[str, Any]:
        """
        The record for a live order the capital boundary refused.

        One definition, because every entry in `execution_results` is read the
        same way downstream: a refusal that omitted a field made callers that
        index it raise instead of read the refusal. Nothing was sent, so the
        amount committed is honestly zero rather than the amount proposed.
        """
        return {
            "market_id": opp.market.id,
            "venue": opp.venue_id,
            "strategy": (opp.raw.get("strategy", "unknown")
                         if hasattr(opp, "raw") and isinstance(opp.raw, dict)
                         else "unknown"),
            "side": opp.side,
            "edge": opp.effective_edge,
            "score": opp.score,
            "amount": amount_usd,
            "status": "blocked_live_capital",
            "reason": reason,
            "live_venue": getattr(self, "_cycle_live_venue", None),
            "executor_result": {
                "status": "blocked",
                "amount_usd": 0.0,
                "price": 0.0,
                "fees_usd": 0.0,
                "gas_usd": 0.0,
                "latency_ms": 0,
                "reasoning": reason,
            },
            "result": {"status": "blocked", "message": reason},
            "canonical_path": ("V3 -> live-capital gate -> BLOCKED "
                               "(no order sent)"),
            "position_recorded": False,
            "position_reason": "no live capital was authorised here",
        }

    def _adapter_is_paper(self, opp) -> bool:
        """
        Whether the venue for this opportunity can only simulate.

        THE predicate for "will this order spend real money": the sizing clamp
        and the last gate before dispatch both read it. When they disagreed, the
        clamp sized a PAPER order against the live venue's budget - so a venue
        with no authorisation (or an unreadable balance) had every paper trade
        clamped to $0 and skipped, which is how exploration stops executing and
        a fresh install can never earn the qualification it needs.
        """
        try:
            adapter = self.venue_registry.adapters.get(
                str(getattr(opp, "venue_id", "") or ""))
        except Exception:
            adapter = None
        if adapter is None:
            return True
        if getattr(opp, "_is_exploration", False):
            return True
        return not bool(getattr(adapter, "can_place_real_orders", False))

    def _refresh_qualifications(self) -> int:
        """
        Rebuild the qualification gate's inputs from the recorded outcomes.

        Returns the number of venues refreshed. A venue with no recorded
        outcomes is refreshed too, with zeros - which fail every statistical
        check. That is deliberate: an unmeasured venue must not inherit the
        result of a measurement that was never made, and a venue that loses its
        evidence must lose its qualification with it.
        """
        from ..learning.trade_outcomes import qualification_stats_from_outcomes

        registered = list(getattr(self.venue_registry, "adapters", {}) or {})
        for venue_id in registered:
            stats = qualification_stats_from_outcomes(self.storage, venue_id)
            self.qualification_engine.evaluate_qualification(venue_id, stats)
        return len(registered)

    def _venue_selection(self, qualified_venue_ids=None) -> Optional[Dict[str, Any]]:
        """
        Decide and record which venue holds the live capital.

        Recorded every cycle rather than computed on demand in the UI, so the
        answer is part of the run's own output and can be read back later.
        """
        try:
            from ..execution.capital import (
                FUNDING_ROUTES,
                CapitalLedger,
                authorised_budgets,
                operator_mode,
            )
            from ..strategy.venue_selection import VenueSelector

            selector = VenueSelector(storage=self.storage,
                                     funding_routes=FUNDING_ROUTES)
            # Falls back to what the last cycle recorded, so the selection can be
            # recomputed on demand (by the console, or by the operator) without
            # running a cycle first.
            qualified_venue_ids = list(
                qualified_venue_ids or self._last_qualified_venue_ids or [])
            registered = list(getattr(self.venue_registry, "adapters", {}) or {})
            # The health engine's own evidence, not a parallel reading. A venue
            # counts as funded only if it reached the FUNDED rung or above, which
            # is the rung the engine raises after reading a real balance from an
            # authenticated call.
            funded_rungs = {"funded", "trade_permitted"}
            balances = {}
            for venue in registered:
                health = self._last_account_health.get(venue) or {}
                evidence = health.get("evidence") or {}
                reached = str(health.get("readiness") or "")
                balances[venue] = {
                    "available": reached in funded_rungs,
                    "balance": float(evidence.get("balance_usd") or 0.0),
                    "source": str(evidence.get("balance_provenance") or ""),
                }
            # The operator's own authorisation, read from the same place the
            # console writes it. Passing an empty budget map here would make the
            # agent's choice blind to the money the operator authorised, and the
            # screen and the loop would then disagree about what is funded.
            plan = CapitalLedger(storage=self.storage).build(
                mode=operator_mode(self.storage),
                budgets=authorised_budgets(self.storage),
                venue_labels={v: r["label"] for v, r in FUNDING_ROUTES.items()},
                balances=balances)
            selection = selector.select(
                selector.assess(
                    registered or sorted(qualified_venue_ids),
                    accounts=[a.to_dict() for a in plan.accounts],
                    qualified_ids=qualified_venue_ids,
                    tracker=getattr(self, "trade_outcome_tracker", None),
                    labels={v: r["label"] for v, r in FUNDING_ROUTES.items()},
                ),
                # Falls back to the bankroll the agent is actually sizing
                # against, which is the stored one - there is no self.bankroll.
                total_budget_usd=(plan.total_budget_usd
                                  or self.storage.get_bankroll()))
            logger.info(f"Venue selection: {selection.verdict}")
            for reason in selection.reasons:
                logger.info(f"  venue: {reason}")
            return selection.to_dict()
        except Exception as e:
            # Never silent: if the selection cannot be computed, the operator
            # must not be left reading a stale or absent answer.
            logger.error(f"Venue selection failed: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

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
                logger.info(
                    f"V3 cycle result: {result.get('status', 'unknown')} "
                    f"{(result.get('opportunities') or {}).get('final_selected', 0)} trades")
                
                # Sleep
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"V3 loop error: {e}")
                await asyncio.sleep(60)
