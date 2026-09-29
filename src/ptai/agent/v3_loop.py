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
from datetime import datetime, timedelta, timezone
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
from ..venues.inventory import inventory_line, record_inventory
from ..validation.rule_bench import benched_rules, bench_live_authorisation
from ..venues import credentials as credential_store
from ..venues import preferences as venue_preferences
from ..validation.walk_forward import (record_verdict, rows_from_storage,
                                        run_walk_forward)
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
from ..betting.positions import SportsBook
from ..betting.ratings import RatingsBook
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
from ..risk.money_guard import LANE_LIVE, LANE_PAPER, default_guard
from ..risk.kill_switch import KillSwitch, KillLevel
from ..risk.limits import LimitsEngine, TradeLimits

from ..execution.order_manager import OrderManager
from ..execution.round import (MARKS_KEY, RoundReport, load_rounds,
                               record_round, round_history_summary)
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


def _side_token_id(market, side):
    """
    The token id for the SIDE being ordered.

    The order row stores it, and paper reconciliation re-simulates a cross
    against THAT token's book. The first token on the market is not
    necessarily the one for the side: on a binary market it is the YES
    token, and a NO order resting against a YES book would be re-simulated
    against prices it never faces.
    """
    want = str(side or "").upper()
    tokens = getattr(market, "tokens", None) or []
    for token in tokens:
        outcome = str(getattr(token, "outcome", "") or "").upper()
        if outcome == want:
            token_id = getattr(token, "token_id", None) or getattr(token, "id", None)
            if token_id:
                return str(token_id)
    for token in tokens:
        token_id = getattr(token, "token_id", None) or getattr(token, "id", None)
        if token_id:
            return str(token_id)
    raw = getattr(market, "raw", None) or {}
    for key in ("clobTokenIds", "clob_token_ids", "token_ids"):
        value = raw.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                if parsed:
                    return str(parsed[0])
            except Exception:
                pass
    return None


def _credential_source(data_dir: str, tool: str) -> Dict[str, Any]:
    """What the agent will actually use for a login, for the record."""
    try:
        from ..venues import credentials as credential_store  # noqa: WPS433

        return credential_store.describe(tool, data_dir)
    except Exception as e:  # noqa: BLE001
        return {"configured": False, "error": f"{type(e).__name__}: {e}"}


# How long a single sleep in the main loop is allowed to be. The wait between
# cycles is sliced into pieces this long so a changed interval, or a manual "run a
# round now", is noticed within one slice instead of after the whole old interval.
WAIT_SLICE_SECONDS = 60.0


def _thinking_note(on: bool) -> str:
    """
    One sentence: is the model reasoning before it answers, and why that way.

    The operator asked why thinking was disabled, and the honest answer has two
    halves: PTAI's own prompt forbade chain-of-thought (a speed decision from a
    log where one market took ~9 minutes), and the request never asked the
    server to enable it. Both are now controlled by ONE setting, and this
    sentence is what the log and the console print, from the same place.
    """
    if on:
        return ("thinking: ON - the model is asked to reason through the market "
                "before answering (PTAI_LLM_THINKING=1); each forecast takes "
                "longer, and a call that overruns the call limit gives no "
                "forecast for that market. Turn it off in Setup > Brain to go "
                "back to direct answers")
    return ("thinking: OFF - the model answers directly; PTAI's prompt asks it "
            "not to think and the request does not ask the server to enable it "
            "(PTAI_LLM_THINKING=0). Turn it on in Setup > Brain to make the "
            "model reason first")


def _why_the_model_was_not_used(status: Dict[str, Any], asked: int,
                                problems: List[str]) -> str:
    """
    The one sentence explaining, from evidence, why no market was answered.

    Ordered by where the cycle actually stopped: no router, no server, an ask
    that never became a call, the model never answering a market it was given,
    the shortlist never reaching the pricing stage, the pricing stage refusing a
    market before a forecast was built, or nothing being shortlisted at all.
    Every branch names a counter the operator can check against the same log,
    and none of them is a plan.

    A module function rather than a method: the console, the CLI and the tests
    all read the cycle's model line through objects that are not a full agent,
    and a reason builder that lives on the agent is reachable by only one of
    them.
    """
    if not status["available"]:
        return ("no local model server answered, so every market was priced "
                "without a model")
    if asked and not status["answered_by_model"]:
        why = (problems[0] if problems else
               (f"the router recorded {status['calls']} call(s); none came back "
                f"with an answer" if status["calls"] else
                "every one of them reached the model step and the router counted no "
                "call for them, so the ask never reached the provider"))
        return (f"{asked} market(s) reached the model step and none was answered "
                f"({why})")
    if asked == 0 and status["calls"]:
        # Calls the router saw that never came from the pricing path
        # (contradiction, resolution analysis). If every one of them failed, that
        # is the reason the model is not used - and it is still true that no
        # MARKET was sent, so both facts are said.
        why = status["last_error"] or "no reason recorded"
        if status["failed"] and status["failed"] == status["calls"]:
            return (f"every model call failed ({why}); no market reached the model "
                    f"step either")
        return (f"the router recorded {status['calls']} call(s) but no market "
                f"reached the model step ({status['failed']} failed): {why}")
    if status["deep_shortlist"] and not status["deep_priced"]:
        # The shortlist is a plan. If none of the shortlisted markets ever
        # reached the pricing stage, the model was never asked - and the line
        # must not claim otherwise.
        return (f"{status['deep_shortlist']} market(s) were chosen for deep "
                f"analysis but none of them reached the pricing stage this cycle, "
                f"so the model was never asked: they were filtered out (volume, "
                f"liquidity or an unvalidated book) before any model time was "
                f"spent")
    if status["deep_priced"] and not status.get("forecasts_run"):
        # REACHED THE PRICING STAGE, NEVER REACHED A FORECAST. The gates in front
        # of the ensemble - resolution wording, contradiction - can refuse a
        # market on their own. The 16:53 log reported exactly this market as
        # "this process has no model router wired to the forecast engine", which
        # was false: the router was wired and was simply never reached.
        return (f"{status['deep_priced']} market(s) reached the pricing stage and "
                f"were refused before a forecast was built, so the model was never "
                f"asked - the cycle's own refusal line for each one names the gate "
                f"(resolution risk, contradiction)")
    if status["deep_shortlist"] and not asked:
        if status.get("ensemble_has_router"):
            return (f"{status['deep_shortlist']} market(s) were chosen for deep "
                    f"analysis and {status.get('forecasts_run', 0)} forecast(s) "
                    f"were built, but not one of them was handed to the model: "
                    f"every market the forecast engine priced was marked as "
                    f"outside the deep shortlist, so the model step was skipped "
                    f"for all of them")
        return (f"{status['deep_shortlist']} market(s) were chosen for deep "
                f"analysis and {status.get('forecasts_run', 0)} forecast(s) were "
                f"built, but not one of them was handed to the model: this process "
                f"has no model router wired to the forecast engine")
    if status["deep_shortlist"] == 0:
        return (f"no market reached the deep shortlist this cycle "
                f"({status['considered']} market(s) considered), so the model was "
                f"never asked")
    return (f"the model was available ({status['describe']}) and this cycle asked "
            f"it for {asked} market(s) without a usable answer")


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
        # Storage FIRST, because everything else is placed relative to it. It was
        # built further down with a hardcoded "./data/ptai.db", so an agent
        # started inside the console wrote to a DIFFERENT database than the page
        # was reading (PTAI_DB) - "Run one cycle" could record trades, ratings and
        # settlements that the operator then could not see. One reader of the
        # environment, in the one place every entry point goes through (db.py).
        self.storage = Storage()
        # Every adapter below is built from `self.settings`. Putting the operator's
        # SAVED logins into those settings first is what makes "I pasted my Kalshi
        # key into the console" mean anything to a process that started
        # afterwards - without it the key sat in the vault while the loops read
        # empty strings, exactly as before.
        self.data_dir = str(self.storage_data_dir())
        try:
            self.credential_state = credential_store.apply_to_settings(
                self.settings, self.data_dir)
            _ready = sorted(k for k, ok in self.credential_state.items() if ok)
            if _ready:
                logger.info("Saved logins applied from the vault: "
                            + ", ".join(_ready))
        except Exception as e:  # noqa: BLE001 - a bad vault must not stop trading
            self.credential_state = {}
            logger.error(f"Could not apply saved logins: "
                         f"{type(e).__name__}: {e}. The agent will run with "
                         f"whatever .env provides.")
        
        # The bankroll is loaded here, not lazily: every cycle path (the
        # blocked one included) reports it to the console, and a first
        # cycle must not crash on an attribute that appears only after a
        # settlement or redemption happens to run.
        self.bankroll = self.storage.get_bankroll()
        # ...and the vault sits BESIDE the database this agent is using, so a
        # login saved in the console is the same file the agent reads. A
        # fixed "./data/vault.json" is the same bug as the fixed database path.
        self.vault = Vault(str(Path(self.data_dir) / "vault.json"))
        self.memory = Memory()
        
        # LLM
        self.llm_router = LLMRouter(
            preferred=getattr(self.settings, 'llm_provider', 'auto'),
            ollama_host=self.settings.ollama_host,
            lm_studio_host=self.settings.lm_studio_host,
            model=self.settings.lm_studio_model,
            timeout_seconds=getattr(self.settings, "llm_timeout_seconds",
                                    180.0),
            # The model reasons before answering only if the operator asked it
            # to (PTAI_LLM_THINKING / the console's Brain panel). Sent in the
            # request, and the cycle says which way it is set and why.
            thinking=bool(getattr(self.settings, "llm_thinking", False)),
        )
        
        # Intelligence
        self.calibration_engine = CalibrationDB(db_path=str(Path(self.data_dir) / "calibration.json"),
                              storage=self.storage, memory=self.memory)
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
        # ONE forecast ensemble, not two. FairValueEngine builds its own by
        # default, so the agent had a second, separate one: the ensemble that
        # PRICES every market was not the ensemble the agent could ask anything
        # - including "what were you asked this cycle". Sharing it means the one
        # object that runs the forecasts is also the one that reports on them,
        # so a count cannot be missing from the sentence that needs it.
        self.fair_value_engine.ensemble_forecaster = self.ensemble_forecaster
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
        
        # THE SPORTS LEDGER. The betting lane could price a match card and had
        # nowhere to put the bets: no position, no settlement, no P&L, no
        # learning. These two give it the same lifecycle the prediction path has -
        # and the ratings book is what gives the sports models an independent view
        # of a fixture, which is the input nothing was producing.
        self.sports_ratings = RatingsBook(storage=self.storage)
        self.sports_book = SportsBook(
            storage=self.storage, tracker=None,   # tracker attached below
            data_engine=self.betting_engine.data,
            min_stake=1.0, max_bets_per_cycle=1)

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
        # Read through the credential store, which is also what the console
        # writes to and what the per-cycle refresh reads. One reader, one answer:
        # "saved in the console" and "what the adapter got" cannot disagree.
        polymarket_login = _credential_source(self.data_dir, "polymarket")
        stored = credential_store.resolve("polymarket", self.data_dir)
        # ...and through THIS agent's own vault handle as well. The store reads
        # the vault file directly, so a vault that was handed in or replaced
        # (which is how the runtime-wiring test injects credentials) would
        # otherwise be bypassed. The store wins per field; the handle fills the
        # gaps, so both paths reach the same adapter.
        try:
            legacy = {}
            for name in ("polymarket_clob", "polymarket"):
                legacy.update(self.vault.get_tool_credentials("Trader", name) or {})
            if legacy:
                merged = {k: v for k, v in legacy.items() if v}
                merged.update({k: v for k, v in stored.items() if v})
                stored = merged
        except Exception as e:  # noqa: BLE001 - not `except: pass`
            logger.error(f"Could not read Polymarket credentials from this "
                         f"agent's vault handle: {type(e).__name__}: {e}. "
                         f"Falling back to the credential store.")
        if not stored:
            # In paper mode this is a statement of fact, not a failure: paper
            # trading needs no credentials at all, and an ERROR line in a paper
            # run reads like the agent is broken. Same sentence, honest level -
            # and it says which mode it is about.
            message = (
                "Polymarket credentials are not configured (vault or environment)"
                + (f": {polymarket_login.get('error')}"
                   if polymarket_login.get("error") else "")
                + (". This process is in PAPER mode, which needs no credentials; "
                   "live trading and redemption would report as unconfigured."
                   if getattr(self, "dry_run", True) else
                   ". Live trading and redemption will report as unconfigured "
                   "until they are saved."))
            if getattr(self, "dry_run", True):
                logger.info(message)
            else:
                logger.error(message)
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
        # Kalshi was constructed as `KalshiAdapter()` - no arguments - so an
        # operator's own API key could never reach it. It has no order path yet
        # (the adapter says so), and the key is what raises the read quota that
        # builds the record in paper.
        _kalshi = credential_store.resolve("kalshi", self.data_dir)
        kalshi_adapter = KalshiAdapter(
            api_key=_kalshi.get("api_key"),
            api_secret=_kalshi.get("api_secret"),
            member_id=_kalshi.get("member_id"),
        )
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
        # ...and its login comes from the vault/env store, not from three
        # os.getenv calls buried in the middle of the registration list, which
        # the console could not write to and the operator could not set from the
        # product. This is the feed that carries goals, corners and cards.
        _betfair = credential_store.resolve("betfair", self.data_dir)
        betfair_adapter = BetfairExchangeAdapter(
            username=_betfair.get("username", ""),
            password=_betfair.get("password", ""),
            app_key=_betfair.get("app_key", ""),
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
        # The limits that actually stop an order. `max_daily_loss_pct` lived in
        # Settings and was read by nothing; CircuitBreaker, DrawdownManager and
        # two of the kill switch's triggers were constructed or defined and never
        # called. This one is consulted immediately before every order.
        self.money_guard = default_guard(self.storage, self.settings)
        # Rules the recorded verdicts refuse; filled at the top of every cycle.
        self._cycle_benched: Dict[str, Any] = {}
        # Venue switches the operator set; filled at the top of every cycle by
        # the discovery path, which is the only place they apply.
        self._cycle_venue_skips: List[str] = []
        self._cycle_credential_changes: Dict[str, Any] = {}
        # The fixtures the previous cycle fetched, used to settle sports bets
        # before the scan fetches this cycle's (settlement runs first on purpose:
        # the bankroll it returns feeds the sizing that follows).
        self._sports_fixtures: Optional[List[Any]] = None
        self._sports_rating_update: Dict[str, Any] = {}
        self._sports_placement: Dict[str, Any] = {}
        self._sports_settlement_this_cycle: Optional[Dict[str, Any]] = None
        # Two-stage scan state: books read in the cheap pass, and the markets the
        # deep pass (X, web research, LLM) is allowed to spend time on.
        self._cycle_books: Dict[str, Any] = {}
        self._deep_market_ids: set = set()
        self._deep_shortlist_active = False
        # What actually reached the pricing stage this cycle, and how many of
        # those were markets the screen had chosen for deep analysis. The model
        # line is built from these counts plus the model's own record of what it
        # was asked - never from the shortlist, which is only a plan.
        self._cycle_context_calls = 0
        self._cycle_deep_context_calls = 0
        self._cycle_closest_call = {}
        self._last_cycle_model = ""
        self._screen: Dict[str, Any] = {}
        # Round state: the opening book value, the report being built, and what
        # settled inside it.
        self._round_open_ledger = None
        self._round_report = None
        self._round_settled = 0
        # Which account the exposure ceilings were seeded with. None means no
        # seed has run, so nothing may be added to the view (see
        # `_seed_exposure_manager`).
        self._exposure_account: Optional[str] = None
        self._round_started_at = ""
        # Base rates: real counted frequencies, refreshed at most once a day.
        self.base_rates = None
        self._base_rate_refresh = {"attempted_at": "", "ok": False, "reason": ""}
        self.deep_analysis_limit = max(1, int(
            getattr(self.settings, "deep_analysis_limit", 8) or 8))
        self.kill_switch = KillSwitch(data_dir=self.data_dir)
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
        # The sports book learns through the same tracker as everything else, so
        # a settled bet reaches qualification, the venue x strategy matrix and the
        # money guard's loss limits - not a private scoreboard of its own.
        if getattr(self, "sports_book", None) is not None:
            self.sports_book.tracker = self.trade_outcome_tracker
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

        # Two of the kill switch's twelve advertised triggers were defined and
        # never called by anything: `check_daily_loss` and `check_drawdown`. A
        # trigger that cannot fire is decoration, so the guard's own numbers now
        # feed them - but ONLY from the LIVE lane, because the kill switch latches
        # until a human resets it and a simulated loss must never leave the agent
        # stopped and waiting for somebody to notice.
        try:
            live_capital = sum(
                float((entry or {}).get("cap_usd") or 0.0)
                for entry in (getattr(self, "_cycle_live_capital", None) or {}).values())
            live = self.money_guard.usage(LANE_LIVE, live_capital)
            health["money_guard"] = {
                "live": live.to_dict(),
                "paper": self.money_guard.usage(
                    LANE_PAPER, float(self.storage.get_paper_bankroll() or 0.0)).to_dict(),
            }
            # The drawdown trigger is fed a fraction of CAPITAL, in its own
            # units. The daily-loss trigger is left unfed on purpose: the guard
            # already stops the live lane for the day and lifts on its own, and
            # latching the whole system - paper included - at 15% would leave the
            # agent stopped until a human noticed, which is the opposite of
            # running unattended. 30% of real capital is the point worth a look.
            if live_capital > 0:
                self.kill_switch.check_drawdown(live.loss_pct_of_bankroll)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Money guard health read failed: {type(e).__name__}: {e}")

        return health

    async def check_eligibility(self) -> Dict[str, EligibilityStatus]:
        eligibility = self.venue_registry.check_all_eligibility()
        for venue_id, status in eligibility.items():
            if status == EligibilityStatus.RESTRICTED:
                # The registry's eligibility check already warned about this
                # venue and country in the same second; this is the same fact.
                logger.debug(f"Venue {venue_id} restricted for {self.country_code}")
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

        # THE OPERATOR'S SWITCHES. Applied last, so they also bind the
        # qualified-only path: a venue the operator switched off is not scanned
        # and not traded, whatever qualification thinks of it. The skipped list
        # is logged as a CHOICE rather than a failure - a venue that is missing
        # from the scan because the operator said so must not read as broken.
        try:
            self._cycle_venue_skips = []
            _prefs = venue_preferences.load(self.storage)
            adapters_to_scan, self._cycle_venue_skips = venue_preferences.filter_adapters(
                self.storage, adapters_to_scan, _prefs)
            if self._cycle_venue_skips:
                logger.info(
                    "Venues switched OFF by you, so not scanned this cycle: "
                    + ", ".join(sorted(self._cycle_venue_skips))
                    + " (turn them back on in the console's Setup tab)")
        except Exception as e:  # noqa: BLE001 - a preference read must not stop a scan
            self._cycle_venue_skips = []
            logger.warning(f"Could not read the venue switches: "
                           f"{type(e).__name__}: {e} - scanning every venue")
        
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

    # How many markets a cycle will spend deep analysis on, and how many it will
    # even look at before choosing. The operator's 2026-09-27 log showed 60-75
    # seconds per market of LLM time, one market at a time, which no 10-minute
    # interval can absorb. Deep analysis is now spent where it can change an
    # answer, and the rest of the scan is cheap.
    PRESCAN_LIMIT = 200
    PRESCAN_CONCURRENCY = 8

    def _screen_score(self, market: Market, book: Dict[str, Any]) -> tuple:
        """
        A cheap, explainable reason to look closely at this market.

        No model runs here, so this cannot claim an edge - it scores whether the
        market is PRICEABLE (a validated two-sided book) and whether there is
        enough liquidity and volume for a small order to mean anything. Markets
        that cannot be priced are not worth 70 seconds of LLM time, and markets
        nobody trades are not either.
        """
        if not getattr(market, "is_probability_market", True):
            return -1.0, ("this venue quotes a price, not a probability, so a "
                          "forecast cannot be turned into an edge here")
        if not isinstance(book, dict) or not book.get("validated", False):
            why = ((book or {}).get("validation") or {}).get("identity") \
                or ((book or {}).get("warning") or "no validated orderbook")
            return -1.0, f"book not validated: {str(why)[:120]}"
        spread = book.get("spread")
        if spread is None:
            return -1.0, "no measured spread"
        if spread > self.strategy_engine_v3.max_spread:
            return -1.0, (f"spread {spread:.1%} is wider than the "
                          f"{self.strategy_engine_v3.max_spread:.0%} this system "
                          f"will trade")
        # THE SCREEN MUST ONLY SHORTLIST MARKETS THE PRICING STAGE WILL PRICE.
        # A market below the scan's own volume or liquidity floor is discarded
        # before the context provider is ever called, so shortlisting it spends
        # the cycle's model budget on nothing: the operator's log showed 8
        # markets "sent to the model" in a cycle where the model was never
        # asked at all. These are the same two floors `cheap_filters` applies.
        engine = self.strategy_engine_v3
        if float(market.volume_24h or 0.0) < float(engine.min_volume_24h):
            return -1.0, (f"volume_24h {float(market.volume_24h or 0.0):,.0f} is below "
                          f"the {engine.min_volume_24h:,} the scan will trade")
        if float(market.liquidity or 0.0) < float(engine.min_liquidity):
            return -1.0, (f"liquidity {float(market.liquidity or 0.0):,.0f} is below "
                          f"the {engine.min_liquidity:,} the scan will trade")
        if not bool(getattr(market, "active", True)) or bool(getattr(market, "closed", False)):
            return -1.0, "the market is not active"
        depth = float(book.get("depth") or 0.0)
        liquidity_score = min(1.0, (float(market.liquidity) + depth) / 20000.0)
        book_quality = max(0.0, 1.0 - float(spread) / 0.10)
        volume_score = min(1.0, float(market.volume_24h) / 20000.0)
        score = 0.45 * liquidity_score + 0.40 * book_quality + 0.15 * volume_score
        return round(score, 6), (
            f"liquidity {liquidity_score:.2f}, book quality {book_quality:.2f} "
            f"(spread {float(spread):.1%}), volume {volume_score:.2f}")

    async def _prescan_and_rank(self, markets: List[Market]) -> Dict[str, Any]:
        """
        Stage 1 of the scan: read every book cheaply, then rank.

        Fetching one orderbook per market is milliseconds; asking a local model
        to think about each market is 60-75 seconds. Doing the expensive one on
        every market is what made a 10-minute cycle impossible, so the cycle now
        reads books for many markets, ranks them on measurable execution quality
        and liquidity, and spends the deep analysis - X, web research and the LLM -
        only on the shortlist.
        """
        self._cycle_books = {}
        self._deep_market_ids = set()
        self._deep_shortlist_active = False
        self._cycle_scan_counts = {}
        # ONE COUNT PER CYCLE for the refusals the venues report per market.
        for _adapter in (getattr(self.venue_registry, "adapters", {}) or {}).values():
            _reset = getattr(_adapter, "reset_clob_rejections", None)
            if callable(_reset):
                try:
                    _reset()
                except Exception as e:  # noqa: BLE001 - a counter must never stop a cycle
                    logger.debug(f"Could not reset {type(_adapter).__name__} "
                                 f"rejection counter: {e}")

        candidates = []
        try:
            candidates = self.strategy_engine_v3.cheap_filters(list(markets))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Cheap screen filters failed, using every market: "
                           f"{type(e).__name__}: {e}")
            candidates = list(markets)
        candidates = candidates[:self.PRESCAN_LIMIT]
        if not candidates:
            self._screen = {"considered": 0, "limit": self.deep_analysis_limit,
                            "shortlist": [], "screened_out": 0,
                            "criteria": "no market passed the cheap filters"}
            logger.info("Cheap screen: no market passed the cheap filters "
                        "(volume/liquidity/active) - nothing to deep-analyse")
            return self._screen

        semaphore = asyncio.Semaphore(self.PRESCAN_CONCURRENCY)

        async def read_book(market: Market):
            async with semaphore:
                try:
                    adapter = self.venue_registry.get_adapter_for_market(market)
                    if not adapter:
                        return market, None, "no adapter for this market"
                    return market, await adapter.get_orderbook(market), ""
                except Exception as e:  # noqa: BLE001
                    return market, None, f"{type(e).__name__}: {str(e)[:120]}"

        started = time.time()
        try:
            results = await asyncio.gather(*(read_book(m) for m in candidates))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Prescan failed: {type(e).__name__}: {e}")
            results = []

        # ONLY THE MARKETS THE SCAN WILL ACTUALLY EVALUATE CAN BE SHORTLISTED.
        # Stage 2 reads (cheap filters -> liquidity order -> per-venue cap) and
        # that cap is why the operator's cycle chose 8 markets for deep analysis
        # and priced none of them: the screen had ranked them on book depth, and
        # the scan dropped every one at its own top-100-by-volume cut. Model time
        # was then spent on nothing and the model line blamed the model. The
        # same helper now decides both.
        by_venue: Dict[str, List[Market]] = {}
        for market, _book, _error in results:
            key = str(getattr(market, "venue_id", "") or
                      (market.raw or {}).get("venue_id") or
                      (market.raw or {}).get("venue") or
                      getattr(market, "source", "unknown") or "unknown").lower()
            by_venue.setdefault(key, []).append(market)

        ranked = []
        for market, book, error in results:
            if book is not None:
                self._cycle_books[market.id] = book
            score, why = self._screen_score(market, book or {})
            if error and not book:
                why = error
            ranked.append((score, market, why))
        ranked.sort(key=lambda row: row[0], reverse=True)

        def _venue_key(market: Market) -> str:
            return str(getattr(market, "venue_id", "") or
                       (market.raw or {}).get("venue_id") or
                       (market.raw or {}).get("venue") or
                       getattr(market, "source", "unknown") or "unknown").lower()

        engine = self.strategy_engine_v3
        _scope = getattr(engine, "markets_that_will_be_evaluated", None)
        # THE CAP IS APPLIED TO MARKETS THAT CAN ACTUALLY BE PRICED.
        #
        # It used to be applied to every market the venue returned, ranked by
        # volume, BEFORE any book was known. The slots then went to markets
        # whose books the venue refused (69 CLOB rejections in the operator's
        # 2026-09-29 19:33 cycle) and the shortlist was starved: two markets
        # chosen from two hundred, the model asked about ONE, and no trade. The
        # book is already read here - a market with no validated book cannot
        # consume a pricing slot, because the scan would refuse it anyway.
        priceable_by_venue: Dict[str, List[Market]] = {}
        refused_at_screen = 0
        for score, market, _why in ranked:
            if score < 0:
                refused_at_screen += 1
                continue
            priceable_by_venue.setdefault(_venue_key(market), []).append(market)
        will_evaluate = set()
        if callable(_scope):
            for venue_markets in priceable_by_venue.values():
                for market in _scope(venue_markets):
                    will_evaluate.add(market.id)
        else:  # a caller with no such helper: every priceable market
            will_evaluate = {market.id for _s, market, _w in ranked
                             if _s >= 0}
        dropped_by_scan = [market for score, market, _why in ranked
                           if score >= 0 and market.id not in will_evaluate]

        shortlist = [row for row in ranked
                     if row[0] >= 0 and row[1].id in will_evaluate][
                         :self.deep_analysis_limit]
        self._deep_market_ids = {row[1].id for row in shortlist}
        self._deep_shortlist_active = True
        self._screen = {
            "considered": len(candidates),
            "books_read": len(self._cycle_books),
            "limit": self.deep_analysis_limit,
            "seconds": round(time.time() - started, 1),
            "shortlist": [{"market_id": m.id, "venue": str(getattr(m, "source", "")),
                           "question": (m.question or "")[:90],
                           "score": score, "why": why}
                          for score, m, why in shortlist],
            "screened_out": len(ranked) - len(shortlist),
            # EVERY MARKET READ IS ACCOUNTED FOR, because the operator reads
            # these numbers to answer "how long do I have to run this".
            #   read = chosen + dropped by the scan's cap + refused at the
            #          screen (no validated book, wide spread, below the
            #          volume/liquidity floors).
            "refused_at_screen": refused_at_screen,
            "priceable": sum(len(v) for v in priceable_by_venue.values()),
            # How many markets the venue scan's own per-venue cap will drop
            # before pricing. Named, because a screen that ignores it shortlists
            # markets that can never be priced.
            "dropped_by_scan": len(dropped_by_scan),
            "screened_out_reasons": (
                "deep analysis - X sentiment, web research and the LLM - is limited "
                "to the shortlist above. Every other market was read and ranked, and "
                "was not given model time this cycle. Whether it is priced at all is "
                "decided by the venue scan, which refuses a market with no usable "
                "book and counts it."),
            "criteria": ("validated two-sided book, spread inside the tradable "
                         "limit, then liquidity, book quality and volume"),
        }
        unreadable = sum(1 for _, book, _ in results
                         if book and book.get("source") in ("no_clob_book", "no_token"))
        self._screen["no_book"] = unreadable
        self._screen["no_book_note"] = (
            "markets the venue has no book for. They are not priced, not ranked "
            "and not traded; their cost and edge are absent rather than estimated.")
        # The screen's own line must ADD UP, because the operator reads these
        # numbers to answer "how long do I have to run this". Four kinds of
        # market come out of one read:
        #
        #   chosen for deep analysis ... 8
        #   below the scan's per-venue cap (priceable, no model time) ... M
        #   refused here (no validated book, spread, floors) ... R
        #   priceable but not shortlisted (the scan still prices them) ... P
        #
        # and 8 + M + R + P = what was read. The 19:31 cycle named only two of
        # the four and 41 markets vanished from the arithmetic.
        _priceable = sum(len(v) for v in priceable_by_venue.values())
        self._screen["priceable_not_shortlisted"] = max(
            0, _priceable - len(shortlist) - len(dropped_by_scan))
        self._screen["unaccounted"] = max(
            0, len(ranked) - len(shortlist) - len(dropped_by_scan)
            - refused_at_screen - self._screen["priceable_not_shortlisted"])
        # PLANS AND PRICES ARE DIFFERENT FACTS. This line said "199 priced on the
        # cheap context only" while the same cycle's scan line said 99 of those
        # markets were "not evaluated - their book is an estimate or absent".
        # Nothing here has been priced yet: this is the screen's plan, and it
        # says so. What was actually priced appears after the scan, in the model
        # line, from the venues' own counts.
        logger.info(
            f"Cheap screen: {len(candidates)} market(s) read in "
            f"{self._screen['seconds']}s, {len(shortlist)} chosen for deep analysis "
            f"(limit {self.deep_analysis_limit}); the rest are NOT given model time "
            f"and none of them has been priced yet"
            + (f"; {unreadable} had no venue book at all (they cannot be priced)"
               if unreadable else "")
            + (f"; {len(dropped_by_scan)} sit below their venue's own per-venue cap "
               f"and will not be priced this cycle" if dropped_by_scan else "")
            + (f"; {refused_at_screen} were refused by the screen itself (no "
               f"validated book, a spread wider than it will trade, or below the "
               f"volume/liquidity floors) - the reasons are counted in the "
               f"console's screen panel" if refused_at_screen else ""))
        for row in self._screen["shortlist"][:5]:
            logger.info(f"  deep {row['market_id']}: {row['why']} "
                        f"(score {row['score']})")
        return self._screen

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
        # Per-market progress, written as it happens.
        #
        # This is the seam every market passes through, and the operator's log
        # showed why it matters: one model call took 524 seconds (21:11:05 ->
        # 21:20:20), and between those two lines the console had nothing to show
        # but a stale heartbeat. Counting the markets - and naming how long the
        # previous one took - turns "is it stuck?" into a number that moves.
        self._cycle_market_index = int(getattr(self, "_cycle_market_index", 0)) + 1
        _elapsed = time.time() - float(getattr(self, "_cycle_market_started", time.time()))
        self._cycle_last_market_seconds = round(_elapsed, 1)
        self._cycle_market_started = time.time()
        if _elapsed > float(getattr(self, "_slowest_model_call_seconds", 0.0) or 0.0) \
                and self._cycle_market_index > 1:
            self._slowest_model_call_seconds = _elapsed
        _total = getattr(self, "_cycle_markets_total", None)
        _question = (getattr(market, "question", "") or "")[:70]
        self._set_phase(
            "evaluating",
            f"market {self._cycle_market_index}"
            + (f" of {_total}" if _total else "")
            + f": {_question}"
            + (f" (previous market took {_elapsed:.0f}s)" if self._cycle_market_index > 1
               else ""))
        return await self.get_context_for_market(market)

    # ------------------------------------------------------------------
    # A ROUND.
    #
    # One press of the button, from research to a bankroll figure. The round is
    # scored on the account that is actually trading it - the paper account in
    # paper mode, the real one otherwise - and its result is equity_end minus
    # equity_start with the realised and marked parts kept apart.
    # ------------------------------------------------------------------
    def _round_marks(self) -> Dict[str, float]:
        """The marks stored at the end of the last round, for the opening book."""
        try:
            raw = self.storage.get_state(MARKS_KEY)
            return json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001
            return {}

    def _store_round_marks(self, marks: Dict[str, float]) -> None:
        try:
            self.storage.set_state(MARKS_KEY, json.dumps(marks))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not store position marks: "
                           f"{type(e).__name__}: {e}")

    def _current_marks(self, markets: List[Market]) -> Dict[str, float]:
        """
        What each market is worth now, from the books this cycle read.

        A validated two-sided book gives the midpoint - the honest mark for a
        position that would have to cross the spread to close. Without one, the
        venue's own reference price is used and the position is still marked,
        because a price the venue published is evidence; a book that FAILED
        validation is not used at all, and those positions are counted as
        unmarked rather than priced off a book we refused.
        """
        marks: Dict[str, float] = {}
        for market in markets or []:
            book = (getattr(self, "_cycle_books", {}) or {}).get(market.id) or {}
            mid = None
            if isinstance(book, dict) and book.get("validated") and book.get("midpoint"):
                mid = float(book["midpoint"])
            elif not isinstance(book, dict) or book.get("validated", True):
                price = getattr(market, "yes_price", None)
                if price:
                    mid = float(price)
            if mid is not None and 0.0 < mid < 1.0:
                marks[str(market.id)] = round(mid, 6)
        return marks

    def _held_market_ids(self, account: str = "paper") -> set:
        """
        The markets this account already has an open position in.

        Read from the trade log, which is the same record the ledger and the
        console use, so the guard cannot disagree with what the operator sees.
        An unreadable log returns an empty set - the guard is an extra refusal,
        not the only one, and refusing every market because the log hiccuped
        would stop the round for no reason.
        """
        try:
            rows = self.storage.get_open_positions() or []
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not read open positions for the held-market "
                         f"guard: {type(e).__name__}: {e}")
            return set()
        held = set()
        for row in rows:
            mode = str(row.get("execution_mode") or "").lower()
            if mode and mode != account:
                continue
            # Anything unresolved that is not an explicit refusal is a position.
            # The status a paper trade carries is "paper" (the real fill is
            # "executed"), so an allow-list silently held nothing and the guard
            # let the same market be bought again every round.
            status = str(row.get("status") or "").lower()
            if status in ("rejected", "blocked", "error", "cancelled",
                          "canceled", "failed", "unknown"):
                continue
            market_id = row.get("market_id")
            if market_id:
                held.add(str(market_id))
        return held

    def _round_ledger(self, price_lookup) -> Any:
        """The ledger as it stands, with whatever marks we have."""
        try:
            return self.ledger_builder.build(price_lookup=price_lookup)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not build the ledger for the round: "
                           f"{type(e).__name__}: {e}")
            return None

    def _score_round(self, report: RoundReport, markets: List[Market],
                     scan_result: Any, execution_results: List[Dict[str, Any]],
                     discovered: int) -> RoundReport:
        """
        Close the round: mark the book, compare it with the opening book, store it.
        """
        # WHAT THE BOOK IS WORTH NOW: this cycle's own prices where it managed to
        # read a book, and otherwise the last price this agent saw for that
        # market. The stored marks are the fallback, not the fresh read, because
        # a round that could not reach a venue must not turn a marked position
        # into an unmarked one and "lose" a move it already reported - that
        # reports a loss the account did not suffer, which is the same class of
        # error as inventing a gain.
        marks = dict(self._round_marks())
        marks.update(self._current_marks(markets))
        end = self._round_ledger(lambda mid: marks.get(str(mid)))
        if end is None:
            report.warnings.append(
                "the ledger could not be read at the end of the round, so no "
                "result can be reported")
            report.markets_discovered = discovered
            return report

        account = "paper" if report.mode == "paper" else "live"
        if account == "paper":
            report.equity_end = float(end.paper_equity)
            report.cash_end = float(end.paper_bankroll)
            report.unrealised_pnl = float(end.paper_unrealised_pnl)
            report.realised_pnl = float(end.paper_realised_pnl)
            report.positions_held = int(end.paper_position_count)
            report.open_positions_marked = int(end.paper_marked)
            report.open_positions_unmarked = int(end.paper_unmarked)
        else:
            report.equity_end = float(end.equity)
            report.cash_end = float(end.free_cash + end.reserved_capital
                                    + end.resting_order_cost)
            report.unrealised_pnl = float(end.unrealised_pnl)
            report.realised_pnl = float(end.realised_pnl)
            report.positions_held = int(end.live_position_count)
            report.open_positions_marked = 1 if end.open_position_value else 0
        report.warnings.extend(end.warnings[:4])

        opening = getattr(self, "_round_open_ledger", None)
        if opening is None:
            report.notes.append(
                "no opening book value was recorded for this round, so its "
                "result cannot be measured - the next round will have one")
        else:
            if account == "paper":
                report.equity_start = float(opening.paper_equity)
                report.cash_start = float(opening.paper_bankroll)
            else:
                report.equity_start = float(opening.equity)
                report.cash_start = float(opening.free_cash
                                          + opening.reserved_capital
                                          + opening.resting_order_cost)

        report.markets_discovered = discovered
        report.markets_screened = int((self._screen or {}).get("considered") or 0)
        report.markets_researched = len(self._deep_market_ids or [])
        if scan_result is not None:
            report.markets_priced = int(getattr(scan_result, "total_scanned", 0) or 0)
            report.candidates = int(getattr(scan_result, "total_candidates", 0) or 0)
        opened = [e for e in (execution_results or []) if e.get("position_recorded")]
        report.positions_opened = len(opened)
        # What actually left the account, from the FILL rather than from the
        # request: a $3 order that filled $1.38 against a thin book staked
        # $1.38, and the slot's `amount` is the request. Reading the wrong one
        # made every round report $0.00 staked.
        _staked = 0.0
        for _entry in opened:
            _fill = _entry.get("fill")
            _value = (_fill.get("filled_usd") if isinstance(_fill, dict) else None)
            if _value is None:
                _value = _entry.get("amount_usd", _entry.get("amount"))
            try:
                _staked += float(_value or 0.0)
            except (TypeError, ValueError):
                logger.warning(
                    f"Round staked total skipped an unreadable amount on "
                    f"{_entry.get('market_id')}: {_value!r}")
        report.staked_usd = round(_staked, 4)
        report.positions_settled = int(
            (getattr(self, "_round_settled", 0) or 0))
        # WHAT THE ROUND ACTUALLY TRADED, one row per order it opened. The
        # round's headline is a number; this is the answer to "what did it do to
        # earn or lose it". A refusal is not a trade - those stay in the
        # execution list, counted by reason.
        trades: List[Dict[str, Any]] = []
        for entry in (execution_results or []):
            if not entry.get("position_recorded"):
                continue
            fill = entry.get("fill") if isinstance(entry.get("fill"), dict) else {}
            ev = (entry.get("expected_net_ev")
                  if isinstance(entry.get("expected_net_ev"), dict) else {})
            trades.append({
                "market_id": entry.get("market_id"),
                "venue": entry.get("venue"),
                "side": entry.get("side"),
                "strategy": entry.get("strategy"),
                "question": entry.get("question"),
                "amount_usd": round(float(
                    fill.get("filled_usd") or entry.get("amount") or 0.0), 4),
                "price": fill.get("filled_price"),
                "shares": fill.get("filled_shares"),
                "edge": entry.get("edge"),
                "expected_net_ev": ev.get("net_ev_usd"),
                "execution_mode": entry.get("execution_mode"),
                "paper_because": entry.get("paper_because", ""),
                "exploration": bool(entry.get("exploration")),
                "status": fill.get("status") or entry.get("status"),
                "pnl": entry.get("pnl"),
            })
        report.trades = trades
        report.closest_call = dict(getattr(self, "_cycle_closest_call", None) or {})
        if report.open_positions_unmarked:
            report.warnings.append(
                f"{report.open_positions_unmarked} position(s) could not be priced "
                f"and are carried at cost; the result reflects only the rest")
        if report.verdict == "flat" and report.positions_opened:
            report.notes.append(
                "positions were opened and the book has not moved yet - a round "
                "that opens positions usually ends marked, not settled")
        report.notes.append(
            "marked = the book at current prices (it can move back). realised = "
            "markets that resolved (that money is banked)")
        self._store_round_marks(marks)
        return report

    BASE_RATES_REFRESH_HOURS = 24.0

    def _refresh_base_rates(self, force: bool = False) -> Dict[str, Any]:
        """
        Count YES frequencies from closed markets, at most once a day.

        Cheap (one paged GET), real (they are settled), and refreshed rather than
        recomputed per cycle. A failure keeps yesterday's dataset instead of
        emptying it - and if there is none, the model keeps saying "no data".
        """
        from ..intelligence.base_rates import BaseRateBook

        book = self.base_rates or BaseRateBook(storage=self.storage)
        self.base_rates = book

        if not force and book.dataset.get("built_at"):
            try:
                from datetime import datetime, timezone
                built = datetime.fromisoformat(book.dataset["built_at"])
                age_hours = (datetime.now(timezone.utc) - built).total_seconds() / 3600.0
                if age_hours < self.BASE_RATES_REFRESH_HOURS:
                    self._base_rate_refresh = {
                        "attempted_at": book.dataset["built_at"], "ok": True,
                        "reason": f"dataset is {age_hours:.1f}h old (refreshes every "
                                  f"{self.BASE_RATES_REFRESH_HOURS:.0f}h)"}
                    return self._base_rate_refresh
            except (TypeError, ValueError):
                pass

        from datetime import datetime, timezone
        attempted = datetime.now(timezone.utc).isoformat()
        rows = []
        try:
            client = getattr(self, "polymarket_client", None)
            if client is None:
                from ..markets.polymarket import PolymarketClient
                client = PolymarketClient()
                self.polymarket_client = client
            rows = client.fetch_closed_markets(limit=500)
        except Exception as e:  # noqa: BLE001
            rows = []
            logger.warning(f"Could not fetch closed markets for base rates: "
                           f"{type(e).__name__}: {e}")

        if not rows:
            self._base_rate_refresh = {
                "attempted_at": attempted, "ok": False,
                "reason": ("no closed markets returned - the base-rate model "
                           "keeps contributing no weight rather than a constant"),
            }
            if not book.dataset:
                logger.info("Base rates: no dataset and nothing fetched this "
                            "cycle - the base-rate component stays absent")
            return self._base_rate_refresh

        book.build_from_markets(rows, source="Polymarket closed markets (Gamma)")
        stored = book.save()
        self._base_rate_refresh = {
            "attempted_at": attempted, "ok": True,
            "markets_read": len(rows), "stored": stored,
            "categories_usable": [c for c, v in
                                  (book.dataset.get("categories") or {}).items()
                                  if v.get("usable")],
            "reason": "",
        }
        return self._base_rate_refresh

    def _base_rate_prior_for(self, category: str):
        """The prior for a category, or None when there is nothing usable."""
        if self.base_rates is None:
            return None
        try:
            return self.base_rates.prior_for(category)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Base-rate prior lookup failed for {category}: {e}")
            return None

    def _deep_analysis_for(self, market: Market) -> tuple:
        """
        Should this market get the expensive evidence, or only its own book?

        Returns (deep, reason). When no screen has run in this cycle - a direct
        call, or a caller outside the cycle - the answer is yes, so the context
        provider keeps behaving the way every existing caller expects. Inside a
        cycle the screen always runs first, and then the shortlist is the only
        thing that gets model time.
        """
        if not self._deep_shortlist_active:
            return True, "no screen this cycle - analysed in full"
        if market.id in self._deep_market_ids:
            for row in (self._screen.get("shortlist") or []):
                if row.get("market_id") == market.id:
                    return True, f"deep shortlist: {row.get('why', '')}"
            return True, "deep shortlist"
        ranked_out = self._screen.get("screened_out", 0)
        # `ranked_out` is a CYCLE total - how many markets were not given model
        # time this cycle - and it was printed for every market as "N ranked
        # below it", which claimed a per-market ranking fact the screen never
        # computed. The sentence now says what the number is.
        return False, (
            f"not in this cycle's deep shortlist ({self.deep_analysis_limit} of "
            f"{self._screen.get('considered', '?')} markets chosen for model time; "
            f"{ranked_out} market(s) in total were priced on their measured book "
            f"alone this cycle) - not given model time")

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
        
        # Is this market in the deep shortlist? If a screen ran this cycle, the
        # answer decides whether X, web research and the LLM run at all. Anything
        # else is priced on its measured book alone, and says so, rather than
        # being silently priced on nothing.
        deep, deep_reason = self._deep_analysis_for(market)
        context["deep_analysis"] = deep
        context["screen_reason"] = deep_reason
        # COUNTED HERE, where a market actually reaches the pricing stage. The
        # cycle's model line needs the difference between "chosen for deep
        # analysis" (a plan, made by the screen) and "priced" (a fact), because
        # the operator's log said 8 markets "were sent to the model" in a cycle
        # where the router recorded no calls at all.
        self._cycle_context_calls = int(getattr(self, "_cycle_context_calls", 0)) + 1
        if deep:
            self._cycle_deep_context_calls = int(
                getattr(self, "_cycle_deep_context_calls", 0)) + 1

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
                # The cheap pass already read this book. Re-reading it per market
                # doubled the I/O of every cycle for an answer the same second
                # would give; the cache is per cycle, so nothing goes stale
                # between cycles.
                orderbook = self._cycle_books.get(market.id)
                if orderbook is None:
                    orderbook = await adapter.get_orderbook(market)
                    self._cycle_books[market.id] = orderbook
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
                # Check if orderbook is real. Three different facts, three
                # different sentences: the venue has no book for this market (a
                # refusal), we could not reach the venue (an estimate), or the
                # venue answered with something unusable. The operator's log had
                # all three printed as "ESTIMATION", which hid the first - the one
                # that says the market is not tradable there at all.
                if not orderbook.get("is_real", False):
                    source = str(orderbook.get("source") or "")
                    if source in ("no_clob_book", "no_token"):
                        logger.warning(
                            f"No venue book for {market.id} ({source}): "
                            f"{orderbook.get('warning') or 'no book returned'} "
                            f"- no cost or edge is computed from it")
                    elif source == "clob_unvalidated":
                        logger.warning(
                            f"Unusable venue book for {market.id} - it failed "
                            f"validation, so edge may not be executable")
                    else:
                        logger.warning(
                            f"Orderbook for {market.id} is ESTIMATION not real "
                            f"CLOB (the venue did not answer) - edge may not be "
                            f"executable")
            else:
                logger.error(f"ABORT: No exact adapter for market {market.id} venue {venue_id} - never fallback to first eligible")
                context["orderbook"] = {"error": f"No adapter for {venue_id}", "is_real": False, "executable": False}
            
            # V9 FIX #4: News intelligence - actually call news_engine.get_news (real implementation)
            try:
                if not deep:
                    context["news"] = ""
                    context["news_skipped"] = deep_reason
                elif hasattr(self, 'news_engine') and self.news_engine:
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
                if not deep:
                    context["x_status"] = "skipped"
                    context["x_unavailable_reason"] = deep_reason
                elif hasattr(self, 'x_engine') and self.x_engine:
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
                        if is_blocked:
                            # Say it out loud. A blocked scraper produced
                            # `raw 0.00 / adjusted 0.00 / use False` for every
                            # market, which reads like a neutral signal that was
                            # measured rather than a source that was never
                            # available - so nobody looking at the log could see
                            # that the sentiment component was simply absent.
                            context["x_unavailable_reason"] = (
                                "X scraper is blocked (circuit breaker open after "
                                "repeated failures) - no posts were read, so there "
                                "is no sentiment signal, not a neutral one")
                        context["sources"].append("x_scraper")
                else:
                    context["x_status"] = "no_engine"
                    context["x_unavailable_reason"] = (
                        "no X engine configured - sentiment was not collected")
            except Exception as e:
                logger.debug(f"X engine failed for {market.id}: {e}")
                context["x_status"] = "failed"
                context["x_unavailable_reason"] = (
                    f"X engine failed: {type(e).__name__}: {str(e)[:150]} - no "
                    f"sentiment signal was collected for this market")
                context["sentiment"] = {"score": 0, "credibility": 0.3, "error": str(e)[:200]}
            
            # V10 FIX #11: Web research - not just volume>10k funnel, but also top opportunities by edge/uncertainty
            # Previously: only if volume_24h >10000 or liquidity >10000 - intentional funnel OK but could miss low-volume high-edge
            # Now: also research if market has high potential edge markers, or is in top candidates
            # We still limit to save time (45s per market), but funnel is broader
            try:
                if not deep:
                    context["research_researched"] = False
                    context["research_blockers"] = [deep_reason]
                    context["research"] = ""
                elif hasattr(self, 'web_researcher') and self.web_researcher:
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
                                # The OBJECT, not only its text. The
                                # contradiction engine decides `researched` from
                                # the ResearchResult it is given, and V3 was
                                # passing the text alone - so the engine's own
                                # state said "no sources" and its early return
                                # threw away the five sources that had just been
                                # retrieved. The most expensive stage in the
                                # stack was reporting "NOT RESEARCHED (no
                                # sources)" over a market it had researched.
                                context["research_result"] = research_result
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
                                # Nothing was retrieved: say so explicitly and
                                # do NOT pass a result on, so the contradiction
                                # engine cannot mistake an empty fetch for
                                # evidence.
                                context["research_result"] = None
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

            # The base-rate prior for THIS category, from resolved markets. None
            # means the model has nothing to say, and says that.
            prior = self._base_rate_prior_for(context["category"])
            if prior:
                context["base_rate_prior"] = prior
                logger.debug(f"Base-rate prior for {market.id} "
                             f"({context['category']}): {prior['rate']:.3f} "
                             f"from n={prior['n']}")
            
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
                # The market's YES price at submission, kept with the order so a
                # fill that arrives later can still be scored against the price
                # the decision was made at.
                yes_price=order.get("yes_price"),
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

    async def run_round(self, target_per_venue: int = 200,
                        max_trades: Optional[int] = None) -> Dict[str, Any]:
        """
        Run one complete round: research, decide, bet, and report the bankroll.

        The difference from `run_cycle` is the number of positions. A round is
        meant to leave the account measurably different, and three positions on a
        $50 bankroll is a thin sample of a market the operator asked to be
        involved in "more at once". The ceiling is the RISK rules, not this
        method: it fills every position slot the limits leave open and no more,
        reads them from the same ledger sizing reads, and never widens a cap.

        Paper rounds are the same call with the same rules - the paper account
        carries its own cash, so six positions of $3 is $18 of a $50 simulation,
        under both the 6%-a-position and 50%-exposure ceilings.
        """
        if max_trades is None:
            try:
                limits_max = int(self.limits_engine.limits.max_open_positions)
            except Exception:  # noqa: BLE001
                limits_max = 6
            ledger = self._round_ledger(
                lambda mid: self._round_marks().get(str(mid)))
            if ledger is not None:
                account = "paper" if self.dry_run else "live"
                held = (ledger.paper_position_count if account == "paper"
                        else ledger.live_position_count)
            else:
                held = 0
            free_slots = max(0, limits_max - int(held))
            # Never at least one: a book that already holds every position the
            # rules allow is a full book, and widening the ceiling to keep the
            # round busy is exactly the thing the ceiling exists to prevent.
            # Nothing is left to open - the round still researches and reports.
            max_trades = free_slots
            logger.info(
                f"Round sizing: {held} position(s) already open, the risk rules "
                f"allow {limits_max} - this round may open up to {max_trades}"
                if max_trades else
                f"Round sizing: the {held} open position(s) fill every slot the "
                f"risk rules allow ({limits_max}) - this round will research and "
                f"score the book, and will not open anything")
        return await self.run_cycle(target_per_venue=target_per_venue,
                                    max_trades=max_trades)

    def round_history(self, limit: int = 20) -> Dict[str, Any]:
        """Completed rounds and the running score, for the console."""
        rounds = load_rounds(self.storage, limit=limit)
        return {"rounds": rounds, "summary": round_history_summary(rounds)}

    async def run_cycle(self, target_per_venue: int = 200, max_trades: int = 3) -> Dict[str, Any]:
        """
        V3 Cycle: multi-venue × multi-strategy WITH Qualification Engine V8
        PTAI wakes up -> Check capital + account health -> Check all qualified venues -> Discover -> Normalize -> Generate candidates -> Evaluate strategies -> Estimate fair value -> Fees/spread/slippage -> Liquidity -> Uncertainty -> Correlations -> Historical performance -> Venue/strategy performance -> Risk-adjusted opportunity -> Compare EVERY candidate -> Choose only passing hard rules -> Risk -> Execution guard -> Execute -> Verify -> Monitor -> Record -> Update -> Repeat
        No Polymarket step - Polymarket becomes Venue #1
        """
        start = time.time()
        logger.info("=== PTAI V3 Cycle Start: Multi-Venue × Multi-Strategy WITH Qualification Engine V8 ===")
        # Prove the process is alive BEFORE the work starts: the first cycle
        # can take far longer than the console's liveness window, and a cycle
        # in progress must not look like a dead agent.
        self._write_agent_heartbeat("cycle")
        # ...and what it is DOING, so the console is not blank for the minutes
        # a cycle spends working. Written at each step, not at the end.
        self._set_phase("scanning")
        # A cycle is a fresh read of the venues. The per-market fee and
        # neg-risk facts the arbitrage engines consult are re-read here, not
        # carried over from the previous cycle: a fee schedule that changed
        # must not be traded on this cycle's stale copy.
        self._arb_facts_cache = {}
        # ...and a fresh count of what the local model did, so "this cycle
        # called qwen 8 times and 8 answers came back" is a statement about
        # this cycle rather than about the process's lifetime.
        try:
            self.llm_router.reset_usage()
            # ...and the forecast engine's own record of which markets were
            # actually handed to the model. Two records of two different facts:
            # the router counts CALLS, the ensemble counts MARKETS ASKED.
            self.ensemble_forecaster.reset_llm_accounting()
        except Exception as e:  # noqa: BLE001 - counting must not stop a cycle
            logger.debug(f"Could not reset the model call count: {e}")
        self._cycle_context_calls = 0
        self._cycle_deep_context_calls = 0
        # PAPER CAPITAL. A paper account with no purse cannot size a trade, and
        # the operator's report was exactly that: "paper mode has zero balance
        # available but it supposed to operate on capitaal". The purse is
        # re-seeded here - at a cycle start, never on a read - only when it is
        # empty AND nothing is open, and the log line says which of those it
        # was. Money committed to open positions is in the equity figure and is
        # never topped up.
        try:
            from ..execution.capital import ensure_paper_purse
            _purse = ensure_paper_purse(self.storage, at="this cycle's start")
            self._paper_purse = _purse
        except Exception as e:  # noqa: BLE001 - a purse problem must not stop the cycle
            self._paper_purse = {"available": False,
                                 "reason": f"{type(e).__name__}: {e}"}
            logger.warning(f"Could not check the paper purse: {type(e).__name__}: {e}")
        # Hand the console the venue list it cannot build for itself. The .bat
        # starts the agent and the console as two processes, so the console has
        # no registry to read and used to fall back to "the fundable venues" -
        # which showed an operator two of nineteen and no way to tell whether
        # the others were broken, absent, or simply not listed.
        _inventory = record_inventory(self.storage, self.venue_registry)
        if _inventory:
            logger.info("Venue inventory recorded: " + inventory_line(_inventory))
        # Out-of-sample validation, re-run on the record as it stands. Cheap next
        # to a scan, and it means the console shows the verdict for the record the
        # agent is actually trading rather than the one it had last time somebody
        # remembered to run the command. Refusals are logged; nothing about this
        # can open the gate - `may_qualify()` answers False always.
        try:
            _wf_rows = rows_from_storage(self.storage)
            _wf = run_walk_forward(_wf_rows)
            record_verdict(self.storage, _wf, scope="all")
            if _wf.refused_names():
                logger.warning(
                    f"Out-of-sample validation does not support: "
                    f"{', '.join(_wf.refused_names())} - {_wf.summary}")
            else:
                logger.info(f"Out-of-sample validation: {_wf.verdict}")
        except Exception as e:  # noqa: BLE001 - a report must never stop a cycle
            logger.warning(f"Out-of-sample validation failed: "
                           f"{type(e).__name__}: {e}")
        # THE BENCH. The verdict above is a finding; this is what the agent does
        # about it. Read once per cycle from the persisted verdicts - the same
        # answer the console shows - and consulted immediately before an order
        # would spend real money. A refused rule keeps choosing WHICH trades are
        # candidates (that is the entry gates' job, unchanged) and loses the
        # right to vouch for them with real capital: entries carried only by
        # refused rules run in paper and say so.
        # LOGINS SAVED WHILE THIS PROCESS IS RUNNING. The adapters were built at
        # startup; an operator who pastes a key into the console must not have to
        # find a window and restart it. Applied here, at the top of every cycle,
        # so the next scan uses it and the log says which venue changed.
        try:
            self.credential_state = credential_store.apply_to_settings(
                self.settings, self.data_dir)
            self._cycle_credential_changes = credential_store.refresh_adapters(
                getattr(self, "venue_registry", None), self.settings, self.data_dir)
        except Exception as e:  # noqa: BLE001 - never stop a cycle over a login
            self._cycle_credential_changes = {}
            logger.warning(f"Could not refresh saved logins this cycle: "
                           f"{type(e).__name__}: {e}")
        try:
            self._cycle_benched = benched_rules(self.storage)
            if self._cycle_benched:
                logger.warning(
                    "Out-of-sample bench: real money is withdrawn from entries "
                    "carried only by " + ", ".join(sorted(self._cycle_benched))
                    + " (they still run in paper)")
        except Exception as e:  # noqa: BLE001
            self._cycle_benched = {}
            logger.warning(f"Could not read the out-of-sample bench: "
                           f"{type(e).__name__}: {e}")
        
        # Health check - Check capital + account health
        health = await self.check_system_health()
        if not health["can_trade"]:
            blocked = {
                "status": "blocked",
                "reason": f"Kill switch L{health['kill_switch_level']}",
                "health": health,
                "mission": self.mission,
                "execution_time": time.time() - start
            }
            # A blocked agent is still a LIVE agent. The console's running
            # indicator must not read "refused to trade" as "process died",
            # so even a blocked cycle leaves its scan row.
            self._set_phase("blocked", blocked["reason"])
            self._record_scan_log(blocked, markets_scanned=0,
                                  opportunities_found=0, avg_edge=0.0)
            return blocked
        
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
            self._round_settled = int(
                getattr(settlement_report, "settled", 0) or 0)
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

        # Sports settlement - close the bets whose FIXTURE has finished.
        #
        # Uses the fixtures this cycle fetched on the same feed that priced them,
        # so a bet settles against a result, never against an assumption. A bet
        # whose result cannot be read from what came back stays OPEN and is
        # reported - refunding it would invent money the venue never returned.
        sports_settlement = None
        try:
            sports_settlement = await self.sports_book.settle(
                events=self._sports_events_for_settlement())
            if sports_settlement and sports_settlement.get("settled"):
                self._credit_sports_settlements(sports_settlement)
            # The settlement runs BEFORE the scan, so it is written onto the
            # block this cycle will report rather than lost.
            self._sports_settlement_this_cycle = sports_settlement
        except Exception as e:  # noqa: BLE001
            logger.error(f"[sports] settlement pass failed: "
                         f"{type(e).__name__}: {e}. Open sports bets stay open "
                         f"and nothing is guessed.")

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
        # Counters the per-market progress line reads. Reset per cycle, so the
        # screen never shows one cycle's progress during another.
        self._cycle_market_index = 0
        self._cycle_markets_total = total_markets
        self._cycle_market_started = time.time()
        self._cycle_last_market_seconds = None
        self._set_phase(
            "evaluating",
            f"{total_markets} market(s) from {len(markets_by_venue)} venue(s); "
            f"pricing them against fees, depth and uncertainty")
        
        # ------------------------------------------------------------------
        # OPEN THE ROUND.
        #
        # Recorded BEFORE any work: the opening book value uses the marks stored
        # at the end of the last round, so a position carried in is measured
        # against what it was worth then rather than against what it cost - an
        # unrealised gain already reported must not be reported a second time.
        # Chances and costs are all measured against this snapshot.
        # ------------------------------------------------------------------
        self._round_settled = 0
        self._round_started_at = datetime.now(timezone.utc).isoformat()
        _opening_marks = self._round_marks()
        self._round_open_ledger = self._round_ledger(
            lambda mid: _opening_marks.get(str(mid)))
        _mode = "paper" if self.dry_run else "live"
        self._round_report = RoundReport(
            number=len(load_rounds(self.storage, limit=9999)) + 1,
            started_at=self._round_started_at,
            mode=_mode,
            account=_mode,
        )
        if self._round_open_ledger is not None:
            if _mode == "paper":
                self._round_report.equity_start = float(
                    self._round_open_ledger.paper_equity)
                self._round_report.cash_start = float(
                    self._round_open_ledger.paper_bankroll)
            else:
                self._round_report.equity_start = float(self._round_open_ledger.equity)
        logger.info(
            f"=== ROUND {self._round_report.number} opened ({_mode} account, "
            f"opening value "
            + (f"${self._round_report.equity_start:,.2f})"
               if self._round_report.equity_start is not None
               else "unknown - no prior marks)") + " ===")

        if total_markets == 0:
            # Same shape as the full result. The no-markets path used to return
            # a different set of keys, so a UI had to special-case it to render
            # "nothing to trade" - and the honest empty case is the common one
            # whenever a venue is down or unconfigured.
            empty_venues = {vid: 0 for vid in markets_by_venue} or {}
            # Base rates and the forecast evidence are refreshed even here: they
            # describe what the engine HAS, and a quiet cycle is exactly when the
            # operator needs to see that the base-rate component is loaded or
            # that X is blocked.
            try:
                self._refresh_base_rates()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Base-rate refresh failed: {type(e).__name__}: {e}")
            self._screen = {
                "considered": 0, "shortlist": [], "screened_out": 0,
                "limit": self.deep_analysis_limit,
                "criteria": ("no market was discovered, so no screen ran - "
                             "nothing was priced on a model opinion either"),
            }
            self._store_forecast_evidence([])
            # A cycle that discovered nothing still says which model it had and
            # why the model did no work - "nowhere to be found in any of these
            # runs" is exactly what an early-returning cycle used to look like.
            logger.info(self._local_model_line())
            # A round that found nothing still closes: the book from earlier
            # rounds is marked, so the operator sees the account value either
            # way instead of a button that reports only that it was quiet.
            _quiet_round = self._round_report or RoundReport(
                mode="paper" if self.dry_run else "live")
            try:
                self._score_round(_quiet_round, markets=[], scan_result=None,
                                  execution_results=[], discovered=0)
                _quiet_round.notes.append(
                    "no venue returned a market this round - the account value "
                    "below is the book carried in, marked at the prices from "
                    "the last round that had any")
            except Exception as e:  # noqa: BLE001
                _quiet_round.warnings.append(
                    f"the round could not be closed: {type(e).__name__}: {e}")
            _quiet_round.finished_at = datetime.now(timezone.utc).isoformat()
            _quiet_round.duration_seconds = time.time() - start
            record_round(self.storage, _quiet_round)
            logger.info("ROUND RESULT: " + _quiet_round.headline())
            self._round_report = _quiet_round
            # ...and the sports lane still runs. Its feed is not the prediction
            # markets' feed, and "no prediction markets" is not a reason to skip
            # the half of the product that has fixtures today.
            sports_block = await self._run_sports_lane(
                "paper" if self.dry_run else "live")
            no_markets_result = {
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
                # The sports lane ran on its OWN feed and its result belongs here:
                # a cycle with no prediction markets is not a cycle with nothing
                # to do, and the operator's button must show what it did.
                "sports": sports_block,
                "deep_analysis": {"considered": 0, "shortlist": [],
                                  "limit": self.deep_analysis_limit,
                                  "screened_out": 0,
                                  "criteria": "no market was discovered, so no "
                                              "screen ran"},
                "local_model": self._local_model_status(),
                "betting": {
                    "ok": bool(sports_block.get("opportunities")),
                    "events": sports_block.get("opportunities") and None or 0,
                    "opportunities": sports_block.get("opportunities", 0),
                    "executable": sports_block.get("executable", 0),
                    "bets_placed": sports_block.get("bets_placed", 0),
                    "data_mode": self.data_mode.value,
                    "blockers": (["no prediction markets discovered"]
                                 + (sports_block.get("blockers") or []))[:3],
                },
                "execution": [],
                "do_nothing_success": True,
                "reasoning": ("No markets discovered from any venue, so no opportunities were "
                              "evaluated. This is not a signal to hold: a venue that fails to "
                              "return markets is unavailable, not quiet."),
            }
            # The round, after the dict it belongs to exists: a quiet round still
            # states the account's value rather than reporting only that it ran.
            no_markets_result["round"] = _quiet_round.to_dict()
            # A cycle that discovered nothing is still a cycle that RAN. On a
            # fresh deployment - no venue credentials yet, or a venue down -
            # every cycle takes this path, and without the row the console
            # would report "Not running" and "Last Scan: Never" for an agent
            # that is honestly trying to scan on every interval.
            self._set_phase(
                "no_markets",
                f"asked {len(markets_by_venue)} venue(s) and none returned a "
                f"market to price - an unavailable venue is not a quiet one")
            self._record_scan_log(no_markets_result, markets_scanned=0,
                                  opportunities_found=0, avg_edge=0.0)
            return no_markets_result

        # Alpha scan - all additional alpha ideas (Top 5 + queue)
        all_markets_flat = [m for markets in markets_by_venue.values() for m in markets]


        # BASE RATES: a prior counted from resolved markets, refreshed at most
        # once a day. Until it exists the base-rate model contributes nothing and
        # says so - which is what the operator's log showed for every market.
        try:
            self._refresh_base_rates()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Base-rate refresh failed: {type(e).__name__}: {e}. "
                           f"The base-rate model keeps contributing no weight.")

        # STAGE 1: the cheap screen. Books for every market, then a ranking.
        # STAGE 2 (inside the strategy scan): X, web research and the LLM for the
        # shortlist only. Doing stage 2 on every market is what made a 10-minute
        # cycle impossible - 60-75 seconds per market of model time on the
        # operator's log.
        self._set_phase("screening",
                        f"reading order books for {min(len(all_markets_flat), self.PRESCAN_LIMIT)} "
                        f"market(s) to choose {self.deep_analysis_limit} for deep analysis")
        screen = await self._prescan_and_rank(all_markets_flat)
        try:
            # The books just read, and the venue's own fee rate, go with the
            # markets: an arbitrage basket is only an arbitrage at executable
            # prices (see strategy/combinatorial.py), and this cycle already has
            # the validated book for every market it screened.
            alpha_results = self.alpha_engine.scan_all_alpha(
                all_markets_flat[:200],
                book_lookup=lambda mid: (getattr(self, "_cycle_books", {}) or {}).get(mid),
                fee_rate_lookup=self._arb_fee_rate,
                neg_risk_lookup=self._arb_neg_risk)
            comb = alpha_results.get("combinatorial") or {}
            # A summary, not the whole result. The dict holds candidate baskets
            # with every Market in them, and `logger.info(f"...{alpha_results}")`
            # wrote one enormous line per cycle that buried the cycle it belonged
            # to - the operator's own log shows it.
            logger.info(
                f"Alpha scan: combinatorial {comb.get('total', 0)} candidate(s) - "
                f"{comb.get('verified', 0)} verified basket(s), "
                f"{comb.get('tradeable', 0)} tradeable at executable prices "
                f"(research only, no orders placed)"
                + (f"; refused {comb.get('refused', 0)}"
                   + (f" ({', '.join(f'{k}: {v}' for k, v in list(comb.get('refused_reasons', {}).items())[:2])})"
                      if comb.get("refused_reasons") else "")
                   if comb.get("refused") else ""))
            ref = alpha_results.get("reference_odds") or {}
            if ref and not ref.get("error"):
                absent = ref.get("unavailable") or {}
                logger.info(
                    f"Reference odds: {ref.get('total', 0)} anchor(s) from "
                    f"{ref.get('markets_read', 0)} market(s), "
                    f"{ref.get('tradeable', 0)} actionable"
                    + ("; unavailable: " + "; ".join(
                        f"{k} - {v}" for k, v in list(absent.items())[:3])
                       if absent else ""))
        except Exception as e:
            logger.warning(f"Alpha scan failed: {e}")
            alpha_results = {"error": str(e)}

        # Which purse a SPORTS bet would come out of. `dry_run` is the flag that
        # propagates to every adapter and decides whether an order could really
        # leave - and no sports book has an order path at all, so in practice this
        # is always "paper" today. It is read rather than hardcoded so the label
        # is the same one the prediction trades carry, from the same source.
        _mode = self.effective_execution_mode()
        if _mode["mode"] != _mode["asked"]:
            logger.warning(f"This cycle runs in {_mode['mode']}: {_mode['why']}")
        execution_mode = _mode["mode"]
        # Betting / sports exchange scan - full match card, not just 1X2.
        # Runs in LIVE_SHADOW by default: it prices goals, corners, cards,
        # handicaps, halves and props, but no capital deploys unless the mode
        # is LIVE and account health is verified.
        # THE SPORTS LANE. Separate feed, separate model, same lifecycle as the
        # prediction path - and it runs even when prediction discovery finds
        # nothing, which is why it is a method rather than an inline block.
        sports_block = await self._run_sports_lane(execution_mode)

        # V3 Strategy Engine: venue × market × strategy
        # Core Objective Step 3: Measures the opportunity on a common risk-adjusted basis
        logger.info("Core Objective Step 3: Measuring opportunity on common risk-adjusted basis: expected_edge × prob_correct × liquidity × execution × calibration × time / (fees+slippage+uncertainty+risk)")
        scan_result: MultiVenueScanResult = await self.strategy_engine_v3.scan_all_venues(
            markets_by_venue=markets_by_venue,
            context_provider=self,
            max_final_trades=max_trades,
            # The books this cycle already read, and the venue's own fee rate:
            # the arbitrage engine costs a pair off executable prices or refuses
            # it, and the per-venue scan does not re-price markets with no book.
            book_lookup=lambda mid: (getattr(self, "_cycle_books", {}) or {}).get(mid),
            fee_rate_lookup=self._arb_fee_rate,
            # THE SCREEN'S SHORTLIST IS PRICED FIRST. The two stages used to
            # disagree in practice (the screen chose on book quality, the scan
            # capped on volume), and the model line then counted markets that
            # were never priced - the operator's 19:33 cycle asked the model
            # about one market of nine hundred.
            must_price=set(getattr(self, "_deep_market_ids", None) or set()),
        )
        # WHAT THE VENUES ACTUALLY PRICED, from their own reports - the number
        # the model line needs. `evaluated` is markets that reached the pricing
        # stage; `skipped_no_book` is the ones refused before it; `beyond_cap`
        # is the ones the per-venue cap left unread. Together they account for
        # every market the screen read.
        _reports = list(scan_result.venue_reports or [])
        self._cycle_scan_counts = {
            "venues": len(_reports),
            "evaluated": sum(int(getattr(r, "evaluated", 0) or 0) for r in _reports),
            "skipped_no_book": sum(int(getattr(r, "skipped_no_book", 0) or 0)
                                   for r in _reports),
            "beyond_cap": sum(int(getattr(r, "beyond_cap", 0) or 0) for r in _reports),
            "discovered": sum(int(getattr(r, "total_discovered", 0) or 0)
                              for r in _reports),
        }
        if isinstance(getattr(self, "_screen", None), dict):
            self._screen.update(self._cycle_scan_counts)
            # ...and what the PRICING stage saw, so the console's screen panel
            # and the log's model line are built from one set of numbers.
            self._screen.update({
                "priced": int(getattr(self, "_cycle_context_calls", 0) or 0),
                "deep_priced": int(getattr(self, "_cycle_deep_context_calls", 0) or 0),
            })
        # HOW CLOSE WAS IT. Kept on the agent for the round, the cycle report
        # and the console's round panel: the operator asking "how long do I have
        # to run this" is asking about this number, and a round that reports
        # only zeros cannot answer him.
        self._cycle_closest_call = dict(
            getattr(scan_result, "best_near_miss", None) or {})
        _sc = self._cycle_scan_counts
        _unpriced = max(0, int(_sc.get("discovered", 0) or 0)
                        - int(_sc.get("evaluated", 0) or 0)
                        - int(_sc.get("skipped_no_book", 0) or 0)
                        - int(_sc.get("beyond_cap", 0) or 0))
        _sc["unpriced_venues"] = _unpriced
        logger.info(
            f"Venue scan priced {_sc['evaluated']} market(s) of "
            f"{_sc['discovered']} read across {_sc['venues']} venue(s); "
            f"{_sc['skipped_no_book']} had no usable book, "
            f"{_sc['beyond_cap']} were beyond their venue's per-venue cap, and "
            f"{_unpriced} were read in venues or below the floors this cycle "
            f"does not price (their own lines say which)")
        # HOW MANY BOOKS THE VENUES REFUSED, in one line. Each venue said the
        # first refusal itself (with its reason); this is the count, so forty
        # identical warnings become one sentence.
        for _adapter in (getattr(self.venue_registry, "adapters", {}) or {}).values():
            _summary = getattr(_adapter, "clob_rejection_summary", None)
            if not callable(_summary):
                continue
            try:
                _counts = _summary() or {}
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not read {type(_adapter).__name__} "
                             f"rejection counts: {e}")
                continue
            if not _counts:
                continue
            _total = sum(int(v) for v in _counts.values())
            _reasons = ", ".join(f"{k} x{int(v)}" for k, v in _counts.items())
            logger.info(
                f"{getattr(_adapter, 'venue_id', 'venue')}: {_total} orderbook(s) "
                f"refused this cycle and never priced against ({_reasons}). The "
                f"first of each reason was reported where it happened.")
        logger.info(f"Common scoring: {len(scan_result.venue_reports)} venues, {scan_result.total_candidates} candidates, {scan_result.total_tradeable} tradeable after fees/liquidity/uncertainty")
        # WHICH model worked this cycle, and what it did. The operator went
        # looking for their LM Studio model id in the log and found it nowhere:
        # the only startup line named the CONFIGURED value (the "local-model"
        # placeholder), the per-forecast line named the provider CLASS, and a
        # market the model never saw was indistinguishable from one it priced.
        logger.info(self._local_model_line())
        logger.info(self._closest_call_line())
        # WHY nothing traded. A cycle that refused everything used to look
        # identical to a cycle that found nothing, which is how a resolution
        # rule blocking a whole category stayed invisible for a week.
        try:
            summary = self.strategy_engine_v3.fair_value_engine.outcome_summary()
            logger.info(f"Market outcomes this cycle: {summary}")
            self.strategy_engine_v3.fair_value_engine.reset_outcomes()
        except Exception as e:
            logger.debug(f"Could not summarise market outcomes: {e}")
        
        # Core Objective Step 4: Only deploys capital when passes independently enforced rules
        # V10 FIX #2: Consistent risk sizing - Kelly first, then same amount through all checks
        # V10 FIX #8: Exploration lane 95/5 - qualified capital lane + shadow lane
        logger.info("Core Objective Step 4: Only deploys capital when passes independently enforced rules: edge>=8% conf>=60% liquidity>=0.3 exec_quality>=0.3 EV>0, exposure caps, correlation caps, drawdown limits, kill_switch, execution_guard")
        logger.info("V10 FIX #2: Kelly → proposed amount → exposure → correlation → limits → guard → executor (same amount)")
        logger.info("Capital lanes: qualified venues may be traded live once one "
                    "qualifies; everything else is paper/shadow only. Nothing "
                    "live is deployed without a qualified venue.")
        
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
        # ...and EVERY venue that holds live capital, not just the first: the
        # dispatch gate reads this set, so a second funded venue can deploy
        # without the first venue's id being treated as the only answer.
        self._cycle_live_venues = set(
            getattr(self, "_cycle_live_capital", None) or {})
        # Two pools, and a trade draws only from the one it will spend.
        # free_capital is the operator's live account; free_capital_paper is
        # its shadow, carrying the same rules. Sizing a paper trade against
        # the live pool would let a paper run trade capital it does not have,
        # and the evidence would measure a bankroll the strategy never had.
        free_capital = ledger.free_cash
        free_capital_paper = ledger.paper_free_cash
        self.last_ledger = ledger
        logger.info(
            f"Capital: equity ${ledger.equity:.2f} = free ${ledger.free_cash:.2f} "
            f"+ reserved ${ledger.reserved_capital:.2f} "
            f"({ledger.live_position_count} live, {ledger.paper_position_count} paper); "
            f"realised ${ledger.realised_pnl:+.2f}, unrealised ${ledger.unrealised_pnl:+.2f}")
        logger.info(
            f"Capital (paper): equity ${ledger.paper_equity:.2f} = free "
            f"${ledger.paper_free_cash:.2f} + positions "
            f"${ledger.paper_position_cost:.2f} + resting "
            f"${ledger.paper_resting_order_cost:.2f} "
            f"({ledger.paper_position_count} paper positions)")
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
        
        # V10 FIX #8: 95/5 split - 95% qualified, 5% exploration (shadow only).
        #
        # The pool is the scan's own paper candidates plus any opportunity that
        # reached `final_selected` but sits on an unqualified venue. It did not
        # exist in practice before 2026-09-29: `final_selected` requires the
        # LIVE gates, so on a fresh install (nothing qualified) the pool was
        # always empty and the lane that is supposed to bootstrap the
        # qualification evidence never placed a trade - the operator's words for
        # it were "I have not seen 1 trade".
        _paper_pool = list(getattr(scan_result, "exploration_candidates", None) or [])
        _paper_pool.extend(unqualified_opps)
        # A market this account already holds is not a new learning opportunity,
        # it is the same bet placed again: buying it every round stacks one
        # market's outcome into every slot and trains the qualification record on
        # a single question. The held markets are named, so a round that explored
        # nothing new says why instead of looking idle.
        _held = self._held_market_ids("paper")
        if _held:
            _before = len(_paper_pool)
            _paper_pool = [o for o in _paper_pool
                           if str(o.market.id) not in _held]
            if _before != len(_paper_pool):
                logger.info(
                    f"Exploration lane: {_before - len(_paper_pool)} candidate(s) "
                    f"skipped - this account already holds "
                    f"{len(_held)} market(s) ({', '.join(sorted(_held)[:5])}); "
                    f"re-buying one is the same bet twice, not a new one")
        exploration_candidates = sorted(
            _paper_pool, key=lambda x: float(getattr(x, "score", 0.0) or 0.0),
            reverse=True)[:1] if _paper_pool else []
        if exploration_candidates:
            logger.info(
                f"Exploration lane (paper): {len(exploration_candidates)} "
                f"candidate(s) for shadow/paper learning, no live capital: "
                f"{[o.venue_id + ':' + o.market.id for o in exploration_candidates]}")
        
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
            # The pool this trade will draw from. `_will_simulate` is THE
            # predicate for "will this order spend imaginary money" - the
            # venue-cap clamp below and the dispatch gate both read the same
            # helper, so sizing, the clamp and the purse the fill draws down
            # cannot disagree about which account the trade comes out of.
            is_paper_trade = self._will_simulate(opp)
            pool_free = free_capital_paper if is_paper_trade else free_capital
            pool_bankroll = (ledger.paper_bankroll if is_paper_trade
                             else bankroll)
            kelly_result = self.kelly_calculator.calculate(
                market_price=opp.market_price,
                fair_prob=opp.estimated_fair,
                bankroll=pool_free,
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
            proposed_amount = pool_free * kelly_result.kelly_fraction_adj
            proposed_amount = min(proposed_amount, pool_bankroll * 0.06)
            opp._is_paper_trade = is_paper_trade

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
            if cap_entry is not None and not is_paper_trade:
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
            _why = (opp.raw or {}).get("exploration_because", "")
            logger.info(
                f"PAPER/EXPLORATION trade: {opp.market.id} @ {opp.venue_id} "
                f"{str(opp.side).upper()} ${opp._proposed_amount:.2f} - the "
                f"model's {float(opp.estimated_fair):.3f} against the "
                f"{float(opp.raw.get('price_paid') or opp.market_price):.3f} a "
                f"share costs, which leaves "
                f"{float(opp.effective_edge or 0.0):+.3f} after the cash costs "
                f"(net EV ${_ev if _ev is not None else 'unmeasured'}); "
                f"{_why}. Paper only, no live capital, for learning")
            if opp.raw is not None:
                # The console and the round read this rather than re-deriving it.
                opp.raw["paper_note"] = (
                    f"paper/exploration: model {float(opp.estimated_fair):.3f} vs "
                    f"executable {float(opp.raw.get('price_paid') or opp.market_price):.3f}"
                    f" - {float(opp.effective_edge or 0.0):+.3f} after cash costs")

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

        # THE CAPS HAVE TO SEE THE BOOK. `ExposureManager.can_open` is the check
        # that enforces the position COUNT and the category / correlated / total
        # exposure ceilings, and it was being consulted with an empty list on
        # every cycle: nothing here ever told it what was already open, so those
        # ceilings were enforced against zero. A round is now asked to open every
        # slot the limits leave free ("do more at once"), which is only safe if
        # the rules that bound "more at once" can see what is already held.
        _exposure_seed = self._seed_exposure_manager(ledger)
        if (not _exposure_seed.get("readable", True)
                and self._round_report is not None):
            self._round_report.warnings.append(str(_exposure_seed.get("warning")))
        
        self._set_phase(
            "executing",
            f"{len(final_trades)} opportunit"
            f"{'y' if len(final_trades) == 1 else 'ies'} passed the hard rules; "
            f"sizing and placing them now")
        for opp in final_trades[:max_trades]:
            try:
                # V9 FIX #1 + V10: MOCK blocking - 4 layers
                market_data_mode, is_mock_market = self._market_is_mock(opp.market)

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
                    self._record_venue_health()
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
                # Same pool rule as sizing: a trade that will only simulate
                # draws the paper pool, one that can spend real money draws
                # the live one.
                is_paper_trade = self._will_simulate(opp)
                pool_free = free_capital_paper if is_paper_trade else free_capital
                amount_usd = getattr(opp, '_proposed_amount', None)
                if amount_usd is None:
                    # Fallback if not set (should not happen). Same corrected
                    # signature as the primary sizing path above.
                    _kr = self.kelly_calculator.calculate(
                        market_price=opp.market_price,
                        fair_prob=opp.estimated_fair,
                        bankroll=pool_free,
                    )
                    if not _kr.should_bet:
                        logger.info(
                            f"Kelly declines {opp.market.id} at the execution "
                            f"gate: {_kr.reason}")
                        continue
                    amount_usd = pool_free * _kr.kelly_fraction_adj
                    amount_usd = min(
                        amount_usd,
                        (ledger.paper_bankroll if is_paper_trade
                         else current_bankroll) * 0.06)
                
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
                #
                # A refusal here forbids REAL money; it does not forbid the
                # trade. The order is placed in paper instead, on the same code
                # path, with the adapter's dry_run forced for the call so no real
                # order can leave - because the alternative (dropping it) meant
                # an armed-but-unfunded install simulated NOTHING, and the paper
                # record is what earns the qualification that unlocks live
                # capital in the first place.
                lane, lane_reason = self._money_lane(opp, amount_usd)
                paper_because = ""
                # ...and by what the out-of-sample record will vouch for. This is
                # a refusal to RISK real money, so it sends the order to paper; it
                # never widens or rewrites the entry filters, which have already
                # had their say above.
                if lane == "live":
                    _bench_allowed, _bench_reason = bench_live_authorisation(
                        opp, getattr(self, "_cycle_benched", {}) or {})
                    if not _bench_allowed:
                        logger.warning(
                            f"OUT-OF-SAMPLE BENCH refuses real money for "
                            f"{opp.market.id} @ {venue_id}: {_bench_reason} - "
                            f"running it in paper so the trade and its outcome "
                            f"are still recorded")
                        lane = "paper"
                        paper_because = f"out-of-sample bench: {_bench_reason}"
                # THE MONEY GUARD. Session and daily loss limits, measured
                # against settled outcomes in the trade log rather than a counter
                # in memory, plus the stake cap for the first live money at a
                # venue. A refusal here is a refusal to RISK real money, so the
                # order still runs in paper - on the same code path, with the
                # adapter forced to dry_run - and the reason is recorded either
                # way. What it must never do is put the order through anyway.
                lane_bankroll = 0.0
                if lane == "live":
                    lane_bankroll = float(
                        ((getattr(self, "_cycle_live_capital", None) or {}).get(
                            venue_id) or {}).get("cap_usd") or 0.0)
                else:
                    try:
                        lane_bankroll = float(self.storage.get_paper_bankroll())
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Could not read the paper bankroll for the money "
                            f"guard: {type(e).__name__}: {e}")
                guard_decision = self.money_guard.check(
                    amount_usd, lane, lane_bankroll, venue_id=venue_id)
                self._last_guard_decision = guard_decision
                if guard_decision.refused:
                    if lane == "live":
                        logger.warning(
                            f"MONEY GUARD refuses real money for {opp.market_id} "
                            f"@ {venue_id}: {guard_decision.reason} - running it "
                            f"in paper so the trade and its outcome are still "
                            f"recorded")
                        lane = "paper"
                        paper_because = f"money guard: {guard_decision.reason}"
                    else:
                        logger.warning(
                            f"MONEY GUARD stops the {lane} lane for "
                            f"{opp.market_id}: {guard_decision.reason}")
                        execution_results.append({
                            "market_id": opp.market.id,
                            "venue": opp.venue_id,
                            "status": "blocked_loss_limit",
                            "reason": guard_decision.reason,
                            "amount": amount_usd,
                            "guard": guard_decision.to_dict(),
                        })
                        continue
                elif guard_decision.approved_usd < amount_usd - 1e-9:
                    logger.info(
                        f"Money guard caps {opp.market_id} at "
                        f"${guard_decision.approved_usd:.2f} (proposed "
                        f"${amount_usd:.2f}): {guard_decision.binding}"
                        + (f" - {guard_decision.notes[0]}"
                           if guard_decision.notes else ""))
                    amount_usd = guard_decision.approved_usd
                if lane == "live":
                    entry = (getattr(self, "_cycle_live_capital", None) or {}).get(
                        venue_id) or {}
                    cap = float(entry.get("cap_usd") or 0.0)
                    if cap > 0 and amount_usd > cap + 1e-9:
                        logger.info(
                            f"Live capital cap binds at dispatch for "
                            f"{opp.market.id}: ${amount_usd:.2f} -> ${cap:.2f}")
                        amount_usd = cap
                    logger.info(f"Core Objective DEPLOY via canonical executor: {opp.market.id} @ {opp.venue_id} amount ${amount_usd:.2f} netEV ${getattr(opp, '_expected_ev', None).net_ev_usd if hasattr(opp, '_expected_ev') and opp._expected_ev else 0:.2f} - Guard PASS → MultiVenueExecutor → {venue_id} | {lane_reason}")
                else:
                    # The lane and the money guard can BOTH have refused real
                    # money, and overwriting here recorded only the lane's
                    # reason - the softer of the two, and never the one about
                    # losses. Both are kept.
                    if lane_reason and lane_reason not in (paper_because or ""):
                        paper_because = (f"{paper_because}; {lane_reason}"
                                         if paper_because else lane_reason)
                    logger.info(
                        f"PAPER EXECUTION {opp.market.id} @ {opp.venue_id} "
                        f"${amount_usd:.2f}: real money refused ({lane_reason}) - "
                        f"placing the same order in paper so the trade and its "
                        f"outcome are still recorded")
                
                # V10 FIX #1: ONE canonical path: MultiVenueExecutor.execute_single()
                # Executor has: MOCK protection, exact routing ABORT, rate limits, min order checks, fee calc, gas
                #
                # The price cap is on the side being bought. It used to be
                # market_price + 0.02 - the YES price - which for a NO order caps
                # the wrong token and either rejects a valid order or lets it
                # through at a price the NO token never trades at.
                exec_result = await self._execute_with_side_aware_cap(
                    opp=opp, amount_usd=amount_usd,
                    # Forcing the adapter to dry_run for a paper trade is what
                    # makes "no real money" true at the venue boundary rather
                    # than merely intended here.
                    exploration=bool(getattr(opp, "_is_exploration", False)
                                     or paper_because))
                
                execution_results.append({
                    "market_id": opp.market.id,
                    "venue": opp.venue_id,
                    # What the money guard allowed and why, on the order itself.
                    # A size the operator cannot trace is a size they cannot
                    # argue with - and the guard is the one that trimmed it.
                    "guard": guard_decision.to_dict(),
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
                    "account_health": account_health.to_dict(),
                    # Was this a shadow trade? The round lists its trades and
                    # the label has to travel with the row.
                    "exploration": bool(getattr(opp, "_is_exploration", False)),
                    "question": (opp.market.question or "")[:120],
                    # Which purse paid, read from the FILL rather than from the
                    # intent: a simulated fill is paper money even when the venue
                    # holds credentials and could have moved real money.
                    "execution_mode": ("paper" if getattr(
                        exec_result, "is_simulated", False) else "live"),
                    # Present exactly when the live-capital lane refused and the
                    # order was simulated instead, so the operator can tell a
                    # paper trade the boundary sent there from one it never
                    # touched.
                    "execution_lane": "paper" if paper_because else "live",
                    **({"paper_because": paper_because} if paper_because else {}),
                })
                
                # Draw down the pool the trade actually drew from. A simulated
                # fill spends paper capital; subtracting it from the live pool
                # would let a paper run spend a number it must never touch.
                if getattr(exec_result, "is_simulated", False):
                    free_capital_paper = self._record_execution(
                        opp, exec_result, execution_results[-1],
                        venue_id=venue_id, amount_usd=amount_usd,
                        current_bankroll=current_bankroll,
                        free_capital=free_capital_paper)
                else:
                    free_capital = self._record_execution(
                        opp, exec_result, execution_results[-1],
                        venue_id=venue_id, amount_usd=amount_usd,
                        current_bankroll=current_bankroll,
                        free_capital=free_capital)
            except Exception as e:
                logger.error(f"Execution failed for {opp.market.id}: {e}")
                import traceback
                execution_results.append({
                    "market_id": opp.market.id,
                    "error": str(e),
                    "traceback": traceback.format_exc()[:500]
                })
        
        # --- the arbitrage lane -------------------------------------------
        # Discovered, reported, and until now never traded. Same executor, same
        # recorder, same gates as the path above.
        free_capital, free_capital_paper = await self._execute_arbitrage_lane(
            scan_result, execution_results,
            current_bankroll=current_bankroll, free_capital=free_capital,
            free_capital_paper=free_capital_paper)

        # V10 FIX #8: Log exploration lane results (shadow only, no capital)
        if exploration_trades:
            logger.info(
                f"Exploration lane complete: {len(exploration_trades)} paper "
                f"trade(s) for learning, no live capital deployed")
        
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
            # Where the cycle's model time went, and what it did not look at.
            "deep_analysis": screen or getattr(self, "_screen", {}),
            # WHICH local model, and what it did, this cycle.
            "local_model": self._local_model_status(),
            # The paper account's own purse: its balance, and whether this cycle
            # had to re-seed it (and why not, when it did not).
            "paper_purse": dict(getattr(self, "_paper_purse", None) or {}),
            # Whether the base-rate component had real counted frequencies this
            # cycle, and from how many resolved markets.
            "base_rates": (self.base_rates.status() if self.base_rates is not None
                           else {"available": False,
                                 "reason": "base rates were never loaded"}),
            "base_rate_refresh": self._base_rate_refresh,
            "arbitrage": {
                "total_found": len(scan_result.arbitrage_opportunities),
                "tradeable": len([a for a in scan_result.arbitrage_opportunities if a.should_trade]),
                # What the lane DID with them. Without this the report looked
                # identical whether the pair executor ran or was unreachable.
                "execution": dict(getattr(self, "_last_arb_execution", None) or {}),
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
                "ok": bool(sports_block.get("opportunities")),
                "events": (sports_block.get("placement") or {}).get("events")
                          or self._sports_events_seen(),
                "market_types_available": sports_block.get("market_types_available", 0),
                "opportunities": sports_block.get("opportunities", 0),
                "executable": sports_block.get("executable", 0),
                "bets_placed": sports_block.get("bets_placed", 0),
                # Kept for the readers that ask for them. What the lane settled
                # is on `sports` - this summary is the headline, not the ledger.
                "arbs": 0,
                "data_mode": self.data_mode.value,
                "blockers": sports_block.get("blockers", []),
            },
            "sports": sports_block,
            # What the exposure caps could see when they ran. An unreadable book
            # must be on the report, not only in a log line: a caps check that
            # cannot see the book cannot refuse anything.
            "exposure": _exposure_seed,
            # The per-market forecast chains, for the trades that were proposed.
            "pricing": self._pricing_rows(scan_result.all_opportunities)[:20],
            # The closest call of the round and the paper candidates the scan
            # found, so "how close was it" and "why did nothing trade" are on
            # the report rather than only in a log line.
            "closest_call": dict(getattr(self, "_cycle_closest_call", None) or {}),
            "paper_candidates": len(
                getattr(scan_result, "exploration_candidates", None) or []),
            # The round is attached AFTER it is closed, below - closing it is
            # what marks the book and states the bankroll, so there is nothing
            # honest to report at this point in the cycle yet.
            "execution": execution_results,
            # Which venue the money is on, and why. Computed from the evidence
            # this cycle gathered: qualification, the balances the health engine
            # read, and the per-venue results in the trades table.
            "venue_selection": self._venue_selection(qualified_venue_ids),
            "reasoning": scan_result.reasoning,
            "execution_time": elapsed,
            "do_nothing_success": len(final_trades) == 0
        }
        
        # The settlement pass ran before the scan; attach its outcome to the block
        # the operator will read, so "settled 2" is not a log line nobody sees.
        if isinstance(sports_block, dict):
            sports_block["settlement"] = getattr(
                self, "_sports_settlement_this_cycle", None)
            sports_block["book"] = self.sports_book.summary()

        # Record the scan row the console's System Health reads ("Last Scan",
        # "Agent: Running") and the /api/scans history is built from. Every
        # legacy loop wrote this row; the V3 loop - the one `ptai run`
        # actually executes - never did, so a live agent read as "Not
        # running" and "Last Scan: Never" on the dashboard.
        _edges = [o.effective_edge for o in (scan_result.all_opportunities or [])
                  if getattr(o, "effective_edge", None) is not None]
        self._record_scan_log(
            result,
            markets_scanned=int(scan_result.total_scanned or 0),
            opportunities_found=int(scan_result.total_candidates or 0),
            avg_edge=(sum(_edges) / len(_edges)) if _edges else 0.0,
        )
        # The forecast evidence, in the database rather than only in the log.
        # The console runs as its own process (the .bat starts both), so anything
        # the operator should be able to read about WHY a fair value came out the
        # way it did has to survive this process.
        self._store_forecast_evidence(
            self._pricing_rows(scan_result.all_opportunities))

        # ------------------------------------------------------------------
        # CLOSE THE ROUND: mark the book and state the result.
        # ------------------------------------------------------------------
        _round = self._round_report or RoundReport(
            mode="paper" if self.dry_run else "live")
        try:
            self._score_round(
                _round,
                markets=[m for group in markets_by_venue.values() for m in group],
                scan_result=scan_result,
                execution_results=execution_results,
                discovered=int(total_markets or 0),
            )
        except Exception as e:  # noqa: BLE001
            _round.warnings.append(
                f"the round could not be closed: {type(e).__name__}: {e}")
            logger.error(f"Round close failed: {type(e).__name__}: {e}")
        _round.finished_at = datetime.now(timezone.utc).isoformat()
        try:
            _round.duration_seconds = (datetime.now(timezone.utc)
                                       - datetime.fromisoformat(_round.started_at)
                                       ).total_seconds()
        except (TypeError, ValueError):
            _round.duration_seconds = float(result.get("execution_time") or 0.0)
        record_round(self.storage, _round)
        logger.info("ROUND RESULT: " + _round.headline())
        self._round_report = _round
        # The round: what it earned or lost, and on which account. Now it exists.
        result["round"] = _round.to_dict()

        _slowest = getattr(self, "_slowest_model_call_seconds", None)
        self._set_phase(
            "cycle_complete",
            f"{int(scan_result.total_scanned or 0)} market(s) scanned, "
            f"{int(scan_result.total_candidates or 0)} candidate(s), "
            f"{len([e for e in (result.get('execution') or []) if e.get('position_recorded')])} "
            f"position(s) recorded"
            + (f"; slowest single market took {_slowest:.0f}s"
               if isinstance(_slowest, (int, float)) else ""))

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

    # One order per venue per second, which is what the executor enforces. The
    # number here is the executor's own window plus a small margin.
# How long a single sleep in the main loop is allowed to be. The wait between
# cycles is sliced into pieces this long so a changed interval, or a manual "run a
# round now", is noticed within one slice instead of after the whole old interval.
# It is a module constant so a test can shrink it; the loop itself never changes
# it.
    VENUE_ORDER_SPACING_SECONDS = 1.05

    async def _wait_for_venue_spacing(self, venue_id: str) -> float:
        """
        Wait out the venue's order spacing instead of losing the order.

        `MultiVenueExecutor.check_rate_limit` allows one request per venue per
        second and answers anything faster with `rate_limited`. That rule is
        right; dropping the order is not. In a round that opens several
        positions on the one venue that holds the capital, the first position
        filled and the rest came back rate_limited and were never retried - the
        round reported one position and quietly lost the others, which is the
        exact opposite of "do more at once". Waiting the second out keeps both
        the venue's rule and the round.
        """
        if not venue_id:
            return 0.0
        try:
            last = float((getattr(self.multi_venue_executor, "rate_limits", {}) or {})
                         .get(venue_id) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if last <= 0:
            return 0.0
        wait = self.VENUE_ORDER_SPACING_SECONDS - (time.time() - last)
        if wait <= 0:
            return 0.0
        logger.info(f"Order spacing: waiting {wait:.2f}s before the next order at "
                    f"{venue_id} - the venue allows one per second")
        await asyncio.sleep(wait)
        return wait

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
        # Space the orders the venue will accept. Checked HERE, on the single
        # canonical dispatch path, so paper and live orders are spaced the same
        # way and a second caller cannot bypass it.
        await self._wait_for_venue_spacing(
            str(getattr(opp, "venue_id", "") or ""))

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
        selector = VenueSelector(storage=self.storage)
        try:
            live_venues = selector.remembered_live_venues()
        except Exception as e:
            logger.warning(f"Could not read the selected live venues: {e}")
            live_venues = []

        if not live_venues:
            return None, caps

        # FREE CASH IS SPENT DOWN THE LIST, NOT HANDED TO EVERY VENUE.
        #
        # With one live venue this could not matter. With `PTAI_MAX_LIVE_VENUES`
        # above 1 it is the protection that stops the same dollar being promised
        # to two accounts at once: each venue's cap is what is left of free cash
        # after the venues before it, so the caps can never sum to more money
        # than the account actually has.
        remaining_free = float(free_cash or 0.0)
        for live_venue in live_venues:
            try:
                authorised = float(authorised_budget(self.storage, live_venue) or 0.0)
            except Exception as e:
                logger.warning(f"Could not read the authorised budget for "
                               f"{live_venue}: {e}")
                authorised = 0.0

            venue_balance = None
            if isinstance(portfolio, dict):
                raw = (portfolio.get("venue_confirmed_balances") or {}).get(
                    live_venue, portfolio.get("venue_confirmed_balance"))
                try:
                    venue_balance = float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    venue_balance = None

            parts = {
                "venue_balance_usd": venue_balance,
                "authorised_usd": authorised,
                "free_cash_usd": round(remaining_free, 2),
            }
            # Live capital needs all three. Any one missing is a refusal, not a
            # default: venue balance unread -> fail closed; nothing authorised ->
            # the operator has not given this agent the money; free cash 0 ->
            # already committed.
            if venue_balance is None:
                caps[live_venue] = {
                    "cap_usd": 0.0, "parts": parts,
                    "detail": ("the venue's balance could not be read, so no live "
                               "capital is deployable at this venue")}
            elif authorised <= 0:
                caps[live_venue] = {
                    "cap_usd": 0.0, "parts": parts,
                    "detail": (f"no budget authorised for {live_venue}: the agent "
                               f"runs in paper there until the operator authorises "
                               f"an amount")}
            else:
                cap = max(0.0, min(venue_balance, authorised, remaining_free))
                remaining_free = max(0.0, remaining_free - cap)
                caps[live_venue] = {
                    "cap_usd": cap, "parts": parts,
                    "detail": (f"min(venue ${venue_balance:.2f}, authorised "
                               f"${authorised:.2f}, free "
                               f"${parts['free_cash_usd']:.2f})")}
        return live_venues[0], caps

    def _money_lane(self, opp, amount_usd: float) -> tuple:
        """
        Which purse this order comes out of: ("live", why) or ("paper", why).

        This is the operator's question answered directly. Before it, a refusal
        from the live-capital boundary dropped the trade on the floor:

            LIVE CAPITAL BLOCKED STUB-M1 @ polymarket $3.00: no live venue is
            selected, so real capital may not be deployed anywhere

        ...so on any install where the adapter is ARMED (credentials present)
        but the operator has not funded a venue, the agent bought nothing at all.
        Paper stopped working the moment the account was linked - which is the
        one moment the operator most needs the simulation, because the paper
        record is what earns the qualification that unlocks live capital.

        The rule is now: the boundary forbids REAL money, it does not forbid the
        trade. Everything that cannot spend real money runs in paper, on the same
        code path, at the size the paper bankroll justifies - and says so.

        The AMOUNT is deliberately not part of this question: where the operator
        has authorised LESS than the trade is worth, the order is trimmed to the
        cap at dispatch rather than turned into a paper trade. Dropping it is the
        bug this method exists to prevent, and paper is for money that cannot be
        spent, not for money that will not stretch.
        """
        venue_id = str(getattr(opp, "venue_id", "") or "")
        if self._adapter_is_paper(opp):
            return "paper", "the adapter cannot place real orders"
        if getattr(opp, "_is_exploration", False):
            return "paper", "exploration is paper-only"
        live_venues = set(getattr(self, "_cycle_live_venues", None)
                          or ([getattr(self, "_cycle_live_venue", None)]
                              if getattr(self, "_cycle_live_venue", None) else []))
        if not live_venues:
            return "paper", ("no live venue holds capital, so no real money may "
                             "be deployed anywhere")
        if venue_id and venue_id not in live_venues:
            return "paper", (
                f"{venue_id} does not hold live capital this cycle "
                f"({', '.join(sorted(live_venues))} "
                f"{'does' if len(live_venues) == 1 else 'do'}) - money goes to "
                f"the venue(s) the operator funded, and the agent cannot move it "
                f"between venues")
        entry = (getattr(self, "_cycle_live_capital", None) or {}).get(venue_id)
        if not entry:
            return "paper", f"no authorised capital computed for {venue_id} this cycle"
        cap = float(entry.get("cap_usd") or 0.0)
        if cap <= 0:
            return "paper", (entry.get("detail")
                             or f"no live capital available at {venue_id}")
        return "live", str(entry.get("detail") or f"cap ${cap:.2f}")

    @staticmethod
    def _trade_row_mode(row: Dict[str, Any]) -> str:
        """Which account a trades row belongs to. `status` first, as the schema intends."""
        mode = str(row.get("execution_mode") or "").strip().lower()
        if mode in ("paper", "live"):
            return mode
        status = str(row.get("status") or "").strip().lower()
        if status == "paper" or "PAPER" in str(row.get("notes") or "").upper():
            return "paper"
        return "live"

    def _seed_exposure_manager(self, ledger: Any) -> Dict[str, Any]:
        """
        Load the open book into the exposure manager before the risk rules run.

        `ExposureManager.can_open` enforces the position COUNT and the category,
        correlated-group and total-exposure ceilings. Nothing in this loop ever
        told it what was already open, so it was consulted with an empty list
        every cycle and those four ceilings were enforced against zero. That was
        survivable while a cycle opened three positions; it is not survivable now
        that a round fills every slot the limits leave free.

        Seeded from the trades table rather than from memory: the process that
        placed the positions may not be this one. Only the account this round is
        trading is seeded - a paper round must not be refused because the live
        book is full, and a live round must not spend against a paper position.

        A row whose category was never recorded is counted in the position count
        and the total ceiling, but keyed on its own market for the category and
        correlated-group ceilings: its membership cannot be reconstructed, and
        inventing one would either refuse trades that should pass or pass trades
        the ceiling exists to stop. That gap is reported, not hidden.
        """
        mode = "paper" if self.dry_run else "live"
        report: Dict[str, Any] = {
            "account": mode,
            "positions": 0,
            "unknown_category": 0,
            "book_value_usd": 0.0,
            "bankroll": None,
            "readable": True,
            "warning": None,
        }
        try:
            rows = self.storage.get_open_positions() or []
        except Exception as e:  # noqa: BLE001
            report["readable"] = False
            report["warning"] = (
                f"the open book could not be read ({type(e).__name__}: {e}), so the "
                f"exposure ceilings cannot see it - positions opened this round "
                f"were checked against an empty book")
            logger.warning("Exposure seed FAILED: " + report["warning"])
            # Nothing may be added to a view that was never loaded. The round
            # carries the failure; it is not dressed up as an empty book.
            self._exposure_account = None
            return report

        seeded: List[Dict[str, Any]] = []
        for row in rows:
            if self._trade_row_mode(row) != mode:
                continue
            try:
                amount = float(row.get("position_size_usd") or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            if amount <= 0:
                continue
            market_id = str(row.get("market_id") or "")
            category = str(row.get("category") or "").strip()
            correlation = str(row.get("correlation_group") or "").strip()
            if not category:
                report["unknown_category"] += 1
                category = f"unknown:{market_id}"
            seeded.append({
                "market_id": market_id,
                "amount_usd": amount,
                "category": category,
                "correlation_group": correlation or category,
                "venue": str(row.get("venue_id") or "polymarket"),
            })
            report["book_value_usd"] = round(
                report["book_value_usd"] + amount, 2)

        # The denominator the caps are percentages of is the account's own
        # bankroll, read from the same place sizing reads it.
        try:
            bankroll = (float(self.storage.get_paper_bankroll() or 0.0)
                        if mode == "paper"
                        else float(self.storage.get_performance_summary()
                                   .get("bankroll", 50.0) or 0.0))
        except Exception:  # noqa: BLE001
            bankroll = float(getattr(ledger, "initial_bankroll", 50.0) or 50.0)
        if bankroll <= 0:
            bankroll = float(getattr(ledger, "initial_bankroll", 50.0) or 50.0)
        report["bankroll"] = round(bankroll, 2)
        report["positions"] = len(seeded)

        self.exposure_manager.positions = seeded
        self.exposure_manager.update_bankroll(bankroll)
        self._exposure_account = mode
        logger.info(
            f"Exposure caps seeded with the {mode} book: {len(seeded)} position(s) "
            f"worth ${report['book_value_usd']:.2f} against a ${bankroll:.2f} "
            f"bankroll, {self.exposure_manager.max_open_positions - len(seeded)} "
            f"slot(s) open"
            + (f"; {report['unknown_category']} position(s) have no recorded "
               f"category and count only against the count and total ceilings"
               if report["unknown_category"] else ""))
        return report

    def _will_simulate(self, opp) -> bool:
        """
        Will this order spend imaginary money?

        THE predicate, read by sizing and by dispatch, so the pool a trade is
        sized from cannot disagree with the purse it draws down. It used to be
        `_adapter_is_paper`, which says "can this venue take real orders" - a
        different question from "is this particular trade live", and the two
        answers differ for every venue that is armed but not funded.
        """
        return self._money_lane(opp, 0.0)[0] == "paper"

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
    def _market_is_mock(market) -> tuple:
        """
        (data_mode, is_mock) for a market - one definition.

        The arbitrage lane needs the same refusal as the single-opportunity
        path, and a second copy of this test would eventually disagree with this
        one about what "mock" means. The fallback is `live`, which is the
        direction that REFUSES: a market whose mode cannot be read is not
        silently admitted to execution.
        """
        mode = getattr(market, "data_mode", None)
        if hasattr(mode, "value"):
            mode = mode.value
        mode = str(mode).lower() if mode else "live"
        is_mock = (bool(getattr(market, "is_mock", False))
                   or mode in ("mock", "historical_sim")
                   or "MOCK" in str(getattr(market, "id", "")).upper())
        return mode, is_mock

    async def _execute_arbitrage_lane(self, scan_result, execution_results, *,
                                      current_bankroll, free_capital,
                                      free_capital_paper: float = 0.0,
                                      max_pairs: int = 1):
        """
        Execute the arbitrage the scan found, and record both legs.

        This lane did not exist. The pair executor was written, tested and
        unreachable: arbitrage was discovered, reported in the cycle report, and
        never traded - so the sequential state machine, the price recheck and
        the hedge were bounding an exposure window that never opened. The
        strategy with the cleanest economics in the whole system was the one
        wired to nothing.

        The same gates the single path passes are applied here, per leg:

          * a mock market is refused,
          * the venue has to answer an account-health check,
          * a leg that would move REAL money has to pass the live-capital
            boundary - which, with one venue funded at a time, means a
            cross-venue pair cannot go live: the honest answer is to say so
            rather than to fund it anyway.

        The pair is not put through the venue qualification gate the way a
        directional trade is: it has no forecast to be qualified on, and its
        risk is execution risk, which the recheck and the hedge bound directly.
        The operator's live venue and budget still apply, because that is where
        real money is authorised.

        Returns the free capital left after the legs are drawn down.
        """
        pairs = list(getattr(scan_result, "arbitrage_opportunities", None) or [])
        candidates = [a for a in pairs if getattr(a, "should_trade", False)]
        summary = {"found": len(pairs), "tradeable": len(candidates),
                   "attempted": 0, "legs_filled": 0, "hedged": 0, "results": []}
        self._last_arb_execution = summary
        if not candidates:
            return free_capital, free_capital_paper

        for arb in candidates[:max_pairs]:
            try:
                opp_a, opp_b = self.multi_venue_executor.arb_leg_opportunities(arb)
            except Exception as e:
                # One line, with the consequence: the pair is not attempted and
                # it is recorded as such. This used to be an ERROR, which reads
                # in the operator's log like a crash rather than a refused pair.
                logger.warning(
                    f"Arb pair {arb.venue_a} vs {arb.venue_b} not attempted: its "
                    f"legs could not be built ({type(e).__name__}: {e})")
                summary["results"].append({
                    "venue_a": arb.venue_a, "venue_b": arb.venue_b,
                    "attempted": False, "reason": f"leg build failed: {e}"})
                execution_results.append({
                    "lane": "arbitrage", "venue": f"{arb.venue_a}+{arb.venue_b}",
                    "market_id": f"{arb.market_a.id}+{arb.market_b.id}",
                    "status": "blocked", "reason": f"leg build failed: {e}",
                    "position_recorded": False})
                continue

            reason = await self._arbitrage_blocked(arb, opp_a, opp_b, free_capital)
            if reason:
                logger.warning(f"Arb blocked: {arb.venue_a} vs {arb.venue_b} - {reason}")
                summary["results"].append({
                    "venue_a": arb.venue_a, "venue_b": arb.venue_b,
                    "attempted": False, "reason": reason})
                execution_results.append({
                    "lane": "arbitrage", "venue": f"{arb.venue_a}+{arb.venue_b}",
                    "market_id": f"{arb.market_a.id}+{arb.market_b.id}",
                    "status": "blocked", "reason": reason,
                    "position_recorded": False})
                continue

            # A pair cannot straddle the pools: with one venue holding the
            # live capital at a time, either both legs simulate or the pair
            # is refused. Budget the pool the legs will actually draw.
            #
            # The pair is simulated when real money is not allowed for BOTH
            # legs - which, with one venue funded at a time, is every
            # cross-venue pair. That used to end the lane: `_arbitrage_blocked`
            # consulted the live-capital boundary and refused the whole thing,
            # so the strategy with the cleanest economics in the system ran only
            # while the adapter happened to hold no credentials. As soon as the
            # operator linked an account, arbitrage stopped executing. Real money
            # is still forbidden - the adapters are forced to dry_run for the
            # call, exactly as an exploration trade is.
            # The amount is not part of this decision - it is venue-level, which
            # is why the pair's own sizing below can follow it.
            lanes = [self._money_lane(o, 0.0) for o in (opp_a, opp_b)]
            legs_paper = any(mode == "paper" for mode, _why in lanes)
            if legs_paper:
                logger.info(
                    f"Arb pair {arb.venue_a}+{arb.venue_b} runs in PAPER: "
                    + "; ".join(f"{o.venue_id}: {why}"
                                for o, (_m, why) in zip((opp_a, opp_b), lanes)
                                if _m == "paper"))
            pool = free_capital_paper if legs_paper else free_capital
            pool_bankroll = (self.storage.get_paper_bankroll() if legs_paper
                             else current_bankroll)
            pair_budget = min(pool_bankroll * 0.06, pool)
            amount_per_leg = pair_budget / 2.0
            summary["attempted"] += 1
            _forced_dry_run = []
            if legs_paper:
                for _leg in (opp_a, opp_b):
                    _adapter = self.venue_registry.adapters.get(
                        str(getattr(_leg, "venue_id", "") or ""))
                    if _adapter is not None and not _adapter.dry_run:
                        _forced_dry_run.append((_adapter, _adapter.dry_run))
                        _adapter.dry_run = True
            try:
                results = await self.multi_venue_executor.execute_arbitrage_pair(
                    arb, amount_per_leg=amount_per_leg)
            except Exception as e:
                logger.error(
                    f"Arb lane failed to execute {arb.venue_a} vs {arb.venue_b}: "
                    f"{type(e).__name__}: {e}")
                summary["results"].append({
                    "venue_a": arb.venue_a, "venue_b": arb.venue_b,
                    "attempted": True, "error": f"{type(e).__name__}: {e}"})
                continue
            finally:
                # Restore the venue's own posture, whatever the pair did.
                for _adapter, _saved in _forced_dry_run:
                    _adapter.dry_run = _saved

            for opp, result in zip((opp_a, opp_b), list(results)):
                slot = {
                    "lane": "arbitrage",
                    "venue": opp.venue_id,
                    "market_id": opp.market.id,
                    "side": opp.side,
                    "amount": amount_per_leg,
                    "expected_net_ev": {},
                    "executor_result": {
                        "status": result.status,
                        "amount_usd": result.amount_usd,
                        "price": result.price,
                        "filled_usd": result.filled_usd,
                        "filled_price": result.filled_price,
                        "filled_shares": result.filled_shares,
                        "fees_usd": result.fees_usd,
                        "gas_usd": result.gas_usd,
                        "latency_ms": result.latency_ms,
                        "reasoning": result.reasoning,
                    },
                    "result": {"status": result.status,
                               "message": result.reasoning},
                    "canonical_path": ("V3 -> Guard -> MultiVenueExecutor -> "
                                       "execute_arbitrage_pair -> adapter -> "
                                       "place_order()"),
                }
                execution_results.append(slot)
                # The SAME recorder the single path uses. A leg is a position
                # like any other: it has to be recorded, settled, and learned
                # from, or the pair's P&L is folklore.
                if legs_paper:
                    free_capital_paper = self._record_execution(
                        opp, result, slot, venue_id=opp.venue_id,
                        amount_usd=amount_per_leg,
                        current_bankroll=current_bankroll,
                        free_capital=free_capital_paper)
                else:
                    free_capital = self._record_execution(
                        opp, result, slot, venue_id=opp.venue_id,
                        amount_usd=amount_per_leg,
                        current_bankroll=current_bankroll,
                        free_capital=free_capital)
                if result.bought_something:
                    summary["legs_filled"] += 1
                if "HEDGE" in str(result.reasoning):
                    summary["hedged"] += 1
                # The hedge is sized in SHARES on the filled part. Any
                # remainder still resting would fill later as a one-sided
                # position - exactly the exposure the pair exists to avoid.
                # Leg A's remainder is unhedged A; leg B's remainder is
                # over-hedged B. Both are cancelled, not left to drift.
                if result.unfilled_shares > 0 and slot.get("order_key"):
                    await self._cancel_arb_remainder(opp, slot)
                summary["results"].append({
                    "venue": opp.venue_id, "market_id": opp.market.id,
                    "side": opp.side, "status": result.status,
                    "filled_usd": result.filled_usd,
                    "filled_shares": result.filled_shares,
                    "remainder_cancelled": slot.get("remainder_cancelled"),
                    "reasoning": result.reasoning[:200]})
        return free_capital, free_capital_paper

    async def _cancel_arb_remainder(self, opp, slot) -> None:
        """
        Cancel the resting remainder of an arbitrage leg, at the venue first
        where there is a venue order, then in the local record.

        The hedge is sized in shares on what FILLED. A remainder that is left
        resting fills later as a one-sided position - precisely the exposure
        the pair exists to avoid - so it is cancelled rather than reconciled
        into the portfolio. A venue cancel that is not confirmed leaves the
        local row open: the reservation stays, and reconciliation verifies
        the order is gone before it is released.
        """
        order_key = slot.get("order_key")
        reason = ("arbitrage leg hedged on the filled shares; the resting "
                  "remainder would become one-sided exposure")
        row = None
        try:
            row = self.storage.get_order_row(order_key)
        except Exception:
            row = None
        order_id = str((row or {}).get("order_id") or order_key or "")
        venue_cancelled = None
        if order_id and not order_id.startswith("local-"):
            adapter = None
            try:
                adapter = self.venue_registry.get_adapter(
                    str((row or {}).get("venue_id") or opp.venue_id))
            except Exception:
                adapter = None
            cancel = getattr(adapter, "cancel_order", None) if adapter else None
            if cancel is not None:
                try:
                    out = await cancel(order_id)
                    venue_cancelled = bool(
                        isinstance(out, dict)
                        and out.get("status") == "cancelled")
                except Exception as e:
                    logger.warning(
                        f"Venue cancel of arb remainder {order_id} failed: "
                        f"{type(e).__name__}: {e}")
        if order_id.startswith("local-") or venue_cancelled:
            slot["remainder_cancelled"] = bool(
                self.order_manager.cancel_order(order_key, reason=reason))
        else:
            slot["remainder_cancelled"] = False
            logger.warning(
                f"Arb remainder {order_id}: the venue did not confirm the "
                f"cancel, so the reservation stays open until reconciliation "
                f"proves the order is gone")

    async def _arbitrage_blocked(self, arb, opp_a, opp_b, free_capital) -> str:
        """
        Why this pair must not be sent, or an empty string if it may.

        Returns the reason instead of raising: a refused pair is a normal
        outcome, and the cycle report has to be able to say which gate refused
        it - the same discipline the single path uses.
        """
        for label, opp in (("leg A", opp_a), ("leg B", opp_b)):
            _, is_mock = self._market_is_mock(opp.market)
            if is_mock:
                return (f"{label} market {opp.market.id} is MOCK_DATA - it "
                        f"cannot reach execution")
            try:
                health = await self.account_health_engine.check_venue_health(
                    opp.venue_id, opportunity=opp)
            except Exception as e:
                return f"{label} venue {opp.venue_id} health unreadable: {e}"
            try:
                self._last_account_health[opp.venue_id] = health.to_dict()
                self._record_venue_health()
            except Exception:
                pass
            if not health.healthy and not health.paper_trading_ok:
                return (f"{label} venue {opp.venue_id} health FAIL: "
                        f"{health.reason}")
        return ""

    def _record_execution(self, opp, exec_result, slot, *, venue_id,
                          amount_usd, current_bankroll, free_capital):
        """
        Turn a proven fill into a position, and the position into learning.

        Returns the free capital left after this fill's cost is drawn down.

        `slot` is the report entry this execution belongs to; it is annotated
        with what happened so the cycle report can be read without parsing logs.
        The two labels this writes are not the same question: `data_mode`
        describes the MARKET DATA and `execution_mode` describes the MONEY, and
        the learning chain reads the second one - an exploration trade runs in
        paper against live data and must never be counted as a live outcome.

        Extracted from the body of the single-opportunity loop so that the
        arbitrage lane records its legs through the SAME path. A second copy of
        this logic would eventually disagree with this one about what a position
        is, and the copy that is wrong is always the one nobody is looking at.
        """
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
            slot["position_recorded"] = False
            slot["position_reason"] = (
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
                    token_id=_side_token_id(opp.market, opp.side),
                    venue_id=venue_id, side=str(opp.side).upper(),
                forecast=self._forecast_for_order(
                    opp, getattr(opp.market, "data_mode", "live")))
                slot["order_recorded"] = bool(order_key)
                slot["order_key"] = order_key
                if order_key:
                    logger.info(
                        f"Order {order_key} recorded for reconciliation: "
                        f"status {exec_result.status}, "
                        f"${exec_result.resting_usd:.4f} reserved, "
                        f"unfilled {exec_result.unfilled_shares} shares")
            return free_capital

        slot["position_recorded"] = True
        slot["fill"] = exec_result.to_position_dict()

        # Draw the committed amount down as we go. Within one cycle the
        # ledger is a snapshot taken before any of these trades, so
        # without this every position in the batch would size against
        # the same free cash - the original bug, one level down.
        # A simulated fill put only the FILLED amount at risk; the unfilled
        # remainder is an ORDER that rests in the book and reserves its cash
        # in the order table - the same structure as a live partial fill.
        # Charging the ledger the whole request for a paper trade would
        # double-reserve the remainder.
        _committed = (exec_result.filled_usd
                      if exec_result.committed_capital
                      else (exec_result.position_size_usd
                            if exec_result.is_simulated else amount_usd))
        free_capital = max(0.0, free_capital - _committed)
        if free_capital <= 0:
            logger.warning(
                "Free capital exhausted mid-cycle - no further positions "
                "will be opened in this batch.")

        # ...and the exposure view grows with it, so the NEXT trade in this round
        # is checked against the positions this one just opened. Without this the
        # caps would only ever see the book as it stood at the start of the round
        # and the sixth position would be judged as though the other five did not
        # exist - the original bug, one level down. Only the account this round is
        # trading is updated: a paper exploration fill in a live round must not
        # consume a live ceiling.
        if ("paper" if exec_result.is_simulated else "live") == self._exposure_account:
            try:
                self.exposure_manager.add_position(
                    market_id=opp.market.id,
                    amount_usd=float(_committed),
                    category=str(getattr(opp, "category", "") or "unknown"),
                    correlation_group=str(
                        getattr(opp, "correlation_group", "") or ""),
                    venue=str(venue_id or ""),
                )
            except Exception as e:  # noqa: BLE001 - the caps must not fail silently
                logger.warning(
                    f"Could not add {opp.market.id} to the exposure view: "
                    f"{type(e).__name__}: {e} - later positions in this round "
                    f"are checked against a book that does not include it")

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
                # Which ceiling this position counts against, stored at the
                # moment the membership was known. Without these two columns
                # the category (15%) and correlated-group (20%) caps could
                # never be re-checked against the open book.
                "category": getattr(opp, "category", None),
                "correlation_group": getattr(opp, "correlation_group", None),
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
                token_id=_side_token_id(opp.market, opp.side),
                venue_id=venue_id, trade_id=int(trade_id),
                side=str(opp.side).upper(),
                forecast=self._forecast_for_order(
                    opp, getattr(opp.market, "data_mode", "live")))
            slot["order_recorded"] = bool(order_key)
            slot["order_key"] = order_key
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
                # THE NULL, stored on the YES scale so it can be compared with
                # `forecast_prob` and `actual_outcome` row by row. This is what
                # the market thought when the decision was made; without it
                # "skill" is a rescaled Brier score that never saw a price.
                yes_price=getattr(opp, "market_price", None),
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
            slot["learning_problems"] = learning_problems
            slot["learning_complete"] = False
        else:
            slot["learning_complete"] = True
        return free_capital
        

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

    def _forecast_for_order(self, opp, data_mode) -> Dict[str, Any]:
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
            # PAPER or LIVE, from the same decision that chose the purse: a
            # venue that cannot spend real money - or that is not where the real
            # money is - simulates, and that is what the row must say. Reading
            # `_adapter_is_paper` alone labelled an armed-but-unfunded venue's
            # simulated fill as LIVE.
            "execution_mode": ("paper" if self._will_simulate(opp)
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
        Whether the venue for this opportunity is incapable of real orders.

        This is about the ADAPTER, not about this trade: credentials absent or
        the venue in dry run. An armed venue returns False here even when the
        trade will be simulated anyway - because the operator funded nothing, or
        funded somewhere else - so it is NOT the predicate for "will this order
        spend real money". That question is `_will_simulate`, which reads this
        and then the live-capital boundary the way dispatch does; sizing and
        dispatch must agree, and when they disagreed the clamp sized a PAPER
        order against the live venue's budget, so a venue with no authorisation
        (or an unreadable balance) had every paper trade clamped to $0 and
        skipped - which is how exploration stops executing and a fresh install
        can never earn the qualification it needs.
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

    def _record_venue_health(self) -> None:
        """
        Persist the venue health this cycle read, for whoever is not this process.

        The console has to show the same balances the agent is sizing against,
        and when the agent runs in the command-line window the console has no
        adapter objects to ask. Cached health was already in memory for venue
        selection; writing it here costs nothing and is the difference between a
        page that says "no venue answered" and one that shows the balance, its
        provenance and the moment it was read.
        """
        try:
            payload = {
                "at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "role": getattr(self, "engine_role", None),
                "venues": self._last_account_health,
            }
            self.storage.set_state("agent.venue_health",
                                   json.dumps(payload, default=str))
        except Exception as e:  # noqa: BLE001 - reporting must not break a cycle
            logger.debug(f"Could not record venue health: {type(e).__name__}: {e}")

    def _write_agent_heartbeat(self, status: str) -> None:
        """
        Leave the "the agent process is alive" mark the dashboard reads.

        The console decides "running" from evidence, not faith. A completed
        scan row is the strongest evidence, but the first cycle can take far
        longer than the freshness window (a slow local LLM makes that worse),
        and while the kill switch blocks, the loop completes no cycle at all.
        So the agent ALSO leaves a heartbeat at cycle start and while
        blocked: "alive" and "a scan has completed" are two different facts,
        and the console must be able to tell them apart.
        """
        try:
            self.storage.set_state("agent.heartbeat", json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "kill_switch_level": int(self.kill_switch.current_level),
            }))
        except Exception as e:
            logger.warning(
                f"Could not write the agent heartbeat: {type(e).__name__}: {e}")

    def _store_forecast_evidence(self, pricing: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Write what the forecast engine had to work with, for the console.

        Stored for BOTH a busy cycle and a quiet one. On a quiet cycle the useful
        answer is not "nothing happened" - it is which components were even
        available (base rates loaded? X blocked? a screen run?) and the operator
        cannot see that from an empty log line.
        """
        try:
            self.storage.set_state("intelligence.last_forecast_evidence", json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "deep_analysis": getattr(self, "_screen", {}) or {},
                # Which model was asked, which answered, and what it was not
                # asked to look at - the same dict the cycle's log line is
                # printed from.
                "local_model": self._local_model_status(),
                "base_rates": (self.base_rates.status()
                               if self.base_rates is not None else {"available": False}),
                "base_rate_refresh": self._base_rate_refresh,
                "pricing": list(pricing or []),
            }))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not store the forecast evidence for the "
                           f"console: {type(e).__name__}: {e}")

    def _pricing_rows(self, opportunities) -> List[Dict[str, Any]]:
        """The per-market forecast chains, as the console reads them."""
        rows = []
        for opp in list(opportunities or [])[:30]:
            if not isinstance(getattr(opp, "raw", None), dict):
                continue
            rows.append({**((opp.raw or {}).get("fair_value_chain") or {}),
                         "market_id": opp.market.id,
                         "question": (opp.market.question or "")[:110],
                         "venue": opp.venue_id,
                         "side": opp.side,
                         "tradeable": bool(opp.should_trade),
                         "explain": (opp.raw or {}).get("explain", ""),
                         "components": (opp.raw or {}).get("components", [])})
        return rows

    def _local_model_status(self) -> Dict[str, Any]:
        """
        What the local model is, and what it did this cycle.

        One dict, read by the cycle's log line, by the console's forecast
        evidence and by the cycle report, so the three cannot describe the same
        cycle differently. Everything is read from the router that actually made
        the calls - never from settings, which name what was ASKED for.
        """
        router = getattr(self, "llm_router", None)
        status: Dict[str, Any] = {
            "available": False, "model": "", "provider": "", "where": "",
            "reason": "", "describe": "", "calls": 0, "answered": 0, "failed": 0,
            "seconds": 0.0, "models": {}, "last_error": "", "used": False,
            "not_used_reason": "", "thinking": False, "thinking_note": "",
            "ensemble_has_router": False, "forecasts_run": 0, "priced": 0,
            "deep_priced": 0, "asked": 0, "answered_by_model": 0,
        }
        screen = getattr(self, "_screen", None) or {}
        status["considered"] = int(screen.get("considered") or 0)
        status["deep_shortlist"] = len(screen.get("shortlist") or [])
        # NAMED FOR WHAT IT IS: markets the screen did not give model time to.
        # It used to be called `priced_without_model`, and that name is what
        # printed "199 priced on their measured book alone" in a cycle where one
        # market was priced.
        status["screened_out_no_model_time"] = int(screen.get("screened_out") or 0)
        if router is None:
            status["not_used_reason"] = "this agent was built without a model router"
            status["describe"] = "none"
            return status
        try:
            status["describe"] = router.describe()
        except Exception as e:  # noqa: BLE001
            status["describe"] = f"unknown ({type(e).__name__}: {e})"
        status["provider"] = router.get_provider_name() if hasattr(router, "get_provider_name") else ""
        status["model"] = str(getattr(router, "active_model", "") or "")
        status["reason"] = str(getattr(router, "active_model_reason", "") or "")
        provider = getattr(router, "provider", None)
        if provider is not None:
            status["where"] = str(getattr(provider, "base_url", None)
                                  or getattr(provider, "host", "") or "")
            if not status["model"]:
                status["model"] = str(getattr(provider, "model", "") or "")
        status["available"] = bool(provider is not None and status["model"])
        try:
            usage = router.usage_report()
        except Exception:  # noqa: BLE001 - a router that cannot report is still described
            usage = {}
        status["calls"] = int(usage.get("calls", 0) or 0)
        status["answered"] = int(usage.get("answered", 0) or 0)
        status["failed"] = int(usage.get("failed", 0) or 0)
        status["seconds"] = float(usage.get("seconds", 0.0) or 0.0)
        status["models"] = dict(usage.get("models", {}) or {})
        status["last_error"] = str(usage.get("last_error", "") or "")
        # WHAT THE MODEL WAS ASKED, counted where the asking happened.
        #
        # THREE COUNTERS, THREE DIFFERENT FACTS, and the line is built from all
        # three because confusing them is what the operator kept seeing:
        #   * `priced` / `deep_priced` - markets that reached the pricing stage
        #     (the ensemble got as far as being asked for a forecast, or the
        #     gate in front of it refused first);
        #   * `forecasts_run` - forecasts the ensemble actually built;
        #   * `asked` / `answered_by_model` - markets handed to the model step
        #     and markets the model answered.
        # A shortlist is a PLAN and appears nowhere in this list.
        asked = answered_by_model = 0
        problems = []
        try:
            accounting = self.ensemble_forecaster.llm_accounting()
            asked = int(accounting.get("asked", 0) or 0)
            answered_by_model = int(accounting.get("answered", 0) or 0)
            problems = [p for p in (accounting.get("problems") or []) if p]
            status["forecasts_run"] = int(accounting.get("forecasts_run", 0) or 0)
        except Exception:  # noqa: BLE001 - a counter that cannot be read is not a crash
            pass
        status["asked"] = asked
        status["answered_by_model"] = answered_by_model
        status["thinking"] = bool(getattr(router, "thinking", False))
        status["thinking_note"] = _thinking_note(status["thinking"])
        status["priced"] = int(getattr(self, "_cycle_context_calls", 0) or 0)
        status["deep_priced"] = int(getattr(self, "_cycle_deep_context_calls", 0) or 0)
        status["question_problems"] = problems[-3:]
        # WHICH ENGINE PRICED THIS CYCLE, so "not one market was handed to the
        # model" can say whether the forecast engine HAS a router (a wiring
        # question) or the markets it priced were simply not marked deep (a
        # shortlist question). Two different defects, two different sentences.
        status["ensemble_has_router"] = bool(getattr(
            getattr(self, "ensemble_forecaster", None), "llm_router", None))
        # A CALL THAT CAME BACK IS NOT A MODEL ANSWER.
        #
        # It used to be `answered_by_model > 0 or status["answered"] > 0`, and
        # the router's `answered` only means the HTTP call returned a response.
        # A model that reasons and is cut off mid-thought - or answers with
        # prose - returns 200 with nothing parseable, so every market fell back
        # to the heuristic while the line said the model was used and printed
        # "8 of 8 call(s) answered". If markets were asked and none of them got
        # a model answer, the model was not used FOR PRICING and the line says
        # so; the router's own answers still count when no market was asked
        # (contradiction and resolution analysis make calls of their own).
        status["used"] = (answered_by_model > 0
                          or (asked == 0 and status["answered"] > 0))
        # AN ASK THAT NEVER BECAME A CALL IS A WIRING DEFECT, and it is visible
        # only by comparing the two counters. The 16:23 log said "8 market(s)
        # were sent to the model but no answer was recorded" while the router
        # had recorded no calls at all: the ask never reached the provider.
        status["asks_without_calls"] = max(0, asked - int(status["calls"] or 0))
        if not status["used"]:
            status["not_used_reason"] = _why_the_model_was_not_used(status, asked,
                                                                    problems)
        # A MODEL THAT CHANGED BETWEEN CYCLES IS A FACT THE OPERATOR NEEDS. The
        # log named qwen3.8-27b in one cycle and qwen/qwen3-14b later with no
        # line connecting them; the answer is a line that says so.
        previous = str(getattr(self, "_last_cycle_model", "") or "")
        if status["model"] and previous and status["model"] != previous:
            status["model_changed_from"] = previous
        if status["model"]:
            self._last_cycle_model = status["model"]
        return status

    def _closest_call_line(self) -> str:
        """
        The best market this round priced that did not qualify to trade.

        "How long am I supposed to run it to produce results" has a measurable
        answer: watch this number against the bar. When the executable edge on
        the closest call reaches 8% on the mid AND the conservative estimate
        clears the price, the live gates open; below that, the number says how
        far away the round was instead of leaving the operator with a zero.
        """
        row = getattr(self, "_cycle_closest_call", None) or {}
        if not row:
            return ("Closest call this round: nothing was priced, so there is "
                    "nothing to measure against the 8% bar")
        edge = float(row.get("executable_edge") or 0.0)
        cons = float(row.get("conservative_executable_edge") or 0.0)
        model = ("the model answered" if row.get("model_answered")
                 else "no model answer was recorded for it")
        return (
            f"Closest call this round: {row.get('market_id')} "
            f"{str(row.get('side') or 'YES').upper()} - the model's "
            f"{float(row.get('fair') or 0.0):.3f} against the "
            f"{float(row.get('market') or 0.0):.3f} mid leaves "
            f"{edge:+.3f} a share after every cash cost (conservative estimate "
            f"{cons:+.3f}); refused: {str(row.get('refusal') or 'unknown')[:160]} "
            f"({model}; the live bar is 8% mispricing and the conservative "
            f"estimate clearing the price)")

    def _local_model_line(self) -> str:
        """The cycle's model line, from the status dict. Never raises."""
        try:
            status = self._local_model_status()
        except Exception as e:  # noqa: BLE001 - reporting must not stop a cycle
            return f"Local model: unknown - could not be read ({type(e).__name__}: {e})"
        ids = ", ".join(f"{k} x{v}" for k, v in (status["models"] or {}).items())
        line = f"Local model: {status['describe']}"
        if status.get("model_changed_from"):
            line += (f" | CHANGED since the last cycle: it was "
                     f"{status['model_changed_from']}, and this cycle it is "
                     f"{status['model']}")
        if status.get("asked") or status["calls"]:
            # "returned", not "answered": the router counts responses, and a
            # response with no JSON in it is not a model answer. The two facts
            # are counted separately (see `answered_by_model`), so the sentence
            # must not read as though both were the same thing.
            line += (f" | markets asked: {status.get('asked', 0)} | "
                     f"{status['answered']} of {status['calls']} call(s) returned "
                     f"in {status['seconds']:.1f}s"
                     + (f" ({ids})" if ids else ""))
            if status.get("asks_without_calls"):
                line += (f" | WIRING GAP: {status['asks_without_calls']} of those "
                         f"ask(s) produced no recorded call at all, so they never "
                         f"reached the provider")
        if not status["used"]:
            line += f" | NOT USED THIS CYCLE: {status['not_used_reason']}"
        else:
            # THE MODEL WAS USED, BUT NOT FOR EVERY DEEP MARKET.
            #
            # The 18:14 run asked the model about 3 markets while 5 reached the
            # pricing stage: two were refused by the resolution gate before a
            # forecast existed. The line said "markets asked: 3" and "5 reached
            # the pricing stage" with nothing connecting them, which reads like
            # two markets disappeared. It is one fact and it is named here.
            _deep = int(status.get("deep_priced") or 0)
            _asked = int(status.get("asked") or 0)
            if _deep > _asked:
                line += (f" | {_deep - _asked} of the {_deep} deep market(s) "
                         f"were refused before a forecast was built, so the "
                         f"model was not asked about them (the per-market "
                         f"refusal lines name the gate)")
        # THE TAIL IS ABOUT PRICES, NOT PLANS. "199 priced on their measured book
        # alone" was printed in a cycle where the venues priced one market, and
        # "model time went to 1 of 200" called a plan model time. The screen's
        # shortlist is a plan; what was priced comes from the venues' counts.
        _scan = dict(getattr(self, "_cycle_scan_counts", None) or {})
        line += (f" | the screen chose {status['deep_shortlist']} of "
                 f"{status['considered']} market(s) for deep analysis (a plan: "
                 f"{status.get('deep_priced', 0)} of them reached the pricing stage)")
        if _scan:
            # THE FOUR NUMBERS ADD UP TO THE TOTAL. The 18:14 line said "priced
            # 5 market(s) of the 900 read (95 had no usable book, 100 beyond
            # their venue's cap)" - which leaves 700 markets unaccounted for,
            # and a reader is right to ask where they went. The remainder is
            # the markets this cycle never priced at all: venues that quote
            # prices rather than probabilities, restricted venues, and markets
            # below the scan's floors. Named, and counted, so the sentence adds
            # up in front of the operator.
            _read = int(_scan.get("discovered", 0) or 0)
            _priced = int(_scan.get("evaluated", 0) or 0)
            _nobook = int(_scan.get("skipped_no_book", 0) or 0)
            _cap = int(_scan.get("beyond_cap", 0) or 0)
            _unpriced = max(0, _read - _priced - _nobook - _cap)
            line += (f" | venue scan: priced {_priced} market(s) of the {_read} "
                     f"read ({_nobook} had no usable book, {_cap} beyond their "
                     f"venue's cap, {_unpriced} in venues or below the floors "
                     f"this cycle does not price - each line says why)")
            _scan["unpriced_venues"] = _unpriced
        else:
            line += " | venue scan: had not run when this line was written"
        if status["failed"] and status["last_error"]:
            line += f" | last failure: {status['last_error']}"
        # THINKING, IN THE OPERATOR'S TERMS.
        #
        # He asked why thinking was off. It was off by PTAI's own choice - the
        # prompt forbade chain-of-thought (a speed decision from a 9-minute
        # market) and the request never asked the server to enable it. The
        # choice is now his (`PTAI_LLM_THINKING`, or the console's Brain
        # panel), and the model line says which way it is set and what it
        # costs, so the question never has to be asked of the log again.
        line += " | " + str(status.get("thinking_note") or _thinking_note(
            bool(status.get("thinking"))))
        return line
    def _record_scan_log(self, result: Dict[str, Any], *, markets_scanned: int,
                         opportunities_found: int, avg_edge: float) -> None:
        """
        Write the market_scans row the console's System Health reads.

        Written once per cycle - including a cycle the kill switch blocked -
        so the dashboard's "Agent: Running" and "Last Scan" reflect what the
        agent actually did instead of what a loop that nobody wired ever did.
        """
        try:
            bankroll = getattr(self, "bankroll", None)
            if bankroll is None:
                bankroll = self.storage.get_bankroll()
            self.storage.log_scan(
                markets_scanned=markets_scanned,
                opportunities_found=opportunities_found,
                avg_edge=avg_edge,
                execution_time=float(result.get("execution_time") or 0.0),
                bankroll=float(bankroll or 0.0),
            )
        except Exception as e:
            logger.warning(
                f"Could not record the scan for the console: "
                f"{type(e).__name__}: {e}")

    # What each phase is called on screen. The loop names a phase; the reading
    # side never invents wording for it, so "what is it doing right now" is the
    # same sentence in the console, in the CLI and in a bug report.
    _PHASE_LABELS = {
        "scanning": "scanning the venues for markets",
        "screening": "reading order books to choose which markets get model time",
        "evaluating": "pricing what it found",
        "executing": "sizing and placing orders",
        "cycle_complete": "cycle complete",
        "no_markets": "nothing to trade from any venue",
        "blocked": "blocked by the risk check",
        "sleeping": "waiting for the next cycle",
        "stopped": "stopped by the operator",
    }

    def storage_data_dir(self) -> str:
        """The folder the database lives in - where the vault lives beside it."""
        try:
            return str(Path(getattr(self.storage, "db_path", "./data/ptai.db")).parent)
        except Exception:  # noqa: BLE001
            return "./data"

    async def _run_sports_lane(self, execution_mode: str) -> Dict[str, Any]:
        """
        Price the sports card, place what passes, and report the ratings.

        A METHOD because this lane does not depend on prediction-market
        discovery at all, and it used to be unreachable whenever that discovery
        came back empty: the cycle returned early with "no markets" and the
        sports feed - a completely different provider - was never asked. On a
        fresh install with no venue credentials that is exactly when the
        simulation needs to run, and it was the one thing that never did.
        """
        events: List[Any] = []
        try:
            events = await self.betting_engine.data.fetch_events(("nba", "epl"))
            self._sports_fixtures = list(events)
            self._sports_rating_update = self.sports_ratings.update(events)
            strengths, ratings = self.sports_ratings.engine_inputs(events)
            if ratings:
                logger.info(
                    f"[sports] model inputs: {len(ratings)} fixture(s) have a "
                    f"rating on BOTH sides "
                    f"({self.sports_ratings.snapshot()['rated_teams']} rated "
                    f"team(s) in total)")
            else:
                logger.info(
                    "[sports] no fixture has a rating on both sides yet, so the "
                    "models have no independent view and will not claim an edge - "
                    "ratings are built from finished results, three each")
            betting_results = await self.betting_engine.run_cycle(
                leagues=("nba", "epl"),
                data_mode=self.data_mode,
                # Real capital needs a verified account. Until the account-health
                # probe can actually place and cancel an order this stays False,
                # so a live data mode cannot by itself deploy money.
                account_health_ok=False,
                strengths=strengths,
                ratings=ratings,
                events=events,
            )
            logger.info(
                f"Betting scan: {betting_results.get('events', 0)} fixtures, "
                f"{betting_results.get('opportunities', 0)} markets priced, "
                f"{betting_results.get('executable', 0)} executable")
        except Exception as e:  # noqa: BLE001
            self._sports_rating_update = {"error": str(e)}
            logger.warning(f"Betting scan failed: {e}")
            betting_results = {"error": str(e), "ok": False}

        try:
            paper_bankroll = float(self.storage.get_paper_bankroll() or 0.0)
            self._sports_placement = self.sports_book.place(
                betting_results,
                execution_mode=(execution_mode if betting_results.get("ok")
                                else "paper"),
                bankroll=paper_bankroll,
                guard=self.money_guard,
                limit=1,
            )
            for bet in self._sports_placement.get("bets") or []:
                # Draw the stake from the PAPER purse as it is placed, so paper
                # equity reflects money committed the way the prediction path's
                # fills do - a paper bankroll that ignores its sports bets would
                # be a second, disagreeing account.
                if bet.get("execution_mode") == "paper":
                    self.storage.set_paper_bankroll(
                        float(self.storage.get_paper_bankroll() or 0.0)
                        - float(bet["stake_usd"] or 0))
        except Exception as e:  # noqa: BLE001
            self._sports_placement = {"placed": 0,
                                      "error": f"{type(e).__name__}: {e}"}
            logger.warning(f"[sports] placement failed: {type(e).__name__}: {e}")

        return {
            "bets_placed": self._sports_placement.get("placed", 0),
            "placement": self._sports_placement,
            "settlement": None,
            "ratings": self.sports_ratings.snapshot(),
            "ratings_update": self._sports_rating_update,
            "book": self.sports_book.summary(),
            "market_types_available": betting_results.get("market_types_available", 0),
            "opportunities": betting_results.get("opportunities", 0),
            "executable": betting_results.get("executable", 0),
            "blockers": (betting_results.get("blockers") or [])[:3],
            "how_a_bet_works": ("quoted from a real feed, placed as a position, "
                                "settled on the final score, P&L written through "
                                "the same learning record as every trade"),
        }

    def _sports_events_for_settlement(self):
        """The fixtures this cycle already fetched, if it fetched any."""
        return getattr(self, "_sports_fixtures", None)

    def _sports_events_seen(self) -> int:
        return len(self._sports_fixtures or [])

    def _credit_sports_settlements(self, settlement: Dict[str, Any]) -> None:
        """
        Return the stake and the profit to the paper purse.

        Without this the stake would leave the paper bankroll at placement and
        never come back - a simulation that loses every bet by construction.
        """
        credited = 0.0
        for row in settlement.get("bets") or []:
            if row.get("status") not in ("won", "lost", "void"):
                continue
            # PAPER only. A live bet's stake and winnings come back into the
            # venue account, not into the local simulation's purse - crediting
            # them here would inflate the paper bankroll with real money.
            if row.get("execution_mode") != "paper":
                continue
            # The stake is money that belongs to the account again; the P&L is
            # what it earned or lost while it was committed.
            credited += float(row.get("stake_usd") or 0.0) + float(row.get("pnl") or 0.0)
        if credited:
            before = float(self.storage.get_paper_bankroll() or 0.0)
            self.storage.set_paper_bankroll(before + credited)
            logger.info(f"[sports] paper purse: ${before:.2f} -> "
                        f"${self.storage.get_paper_bankroll():.2f} "
                        f"(stakes returned and P&L applied)")

    def _arb_venue_facts(self, market: Market) -> Dict[str, Any]:
        """
        What the VENUE says about this market: its taker fee, and whether it is
        part of a mutually exclusive basket ("neg risk" in Polymarket's words).

        Both come from the venue's own market info, read once and cached per
        market, because the arbitrage engine needs both and neither may be
        assumed:

          * the fee, because `2% x legs x 0.5` was a constant this codebase
            invented and then called "after fees"; and
          * the neg-risk answer, because "the ten outcomes of this contest are
            mutually exclusive" is the premise of the whole basket - it is what
            makes buying every NO pay n-1 - and a shared event slug does not
            establish it. Extracting price strikes share a slug and are not
            exclusive at all.

        Falls back to the venue's declared capability rate for the fee when the
        per-market info cannot be read, and reports `neg_risk: None` in that case
        rather than guessing: an unknown answer leaves the basket unverified.
        """
        cache = getattr(self, "_arb_facts_cache", None)
        if cache is None:
            cache = self._arb_facts_cache = {}
        key = str(getattr(market, "id", "") or id(market))
        if key in cache:
            return cache[key]

        facts: Dict[str, Any] = {"fee": None, "neg_risk": None, "source": None}
        try:
            adapter = (self.venue_registry.get_adapter_for_market(market)
                       if self.venue_registry else None)
            if adapter is not None:
                token_id = (getattr(market, "yes_token_id", None) or "")
                mechanics = None
                # ASK ONLY AN ADAPTER THAT HAS ONE. Every adapter was asked for
                # `get_mechanics`, so a venue without that reader - PredictIt,
                # whose markets have no CLOB mechanics at all - answered with an
                # AttributeError on every market, every cycle. It was caught and
                # hidden at DEBUG, so it looked like nothing was wrong while the
                # per-market fee and neg-risk facts were silently never read.
                mechanics_reader = getattr(adapter, "get_mechanics", None)
                if token_id and callable(mechanics_reader):
                    try:
                        # A shim carrying `.market` so the adapter resolves the
                        # condition id too - the same path the order flow uses.
                        from types import SimpleNamespace
                        mechanics = mechanics_reader(
                            SimpleNamespace(market=market), token_id=token_id)
                    except Exception as e:  # noqa: BLE001 - unread falls back
                        facts["mechanics_error"] = f"{type(e).__name__}: {e}"
                        logger.debug(f"Mechanics for {key} unreadable: "
                                     f"{type(e).__name__}: {e}")
                elif mechanics_reader is None:
                    facts["mechanics_source"] = (
                        f"{getattr(adapter, 'venue_id', 'this venue')} publishes no "
                        f"per-market mechanics reader")
                    # SAID ONCE PER VENUE, at INFO - not once per market at
                    # DEBUG. The operator's log carried
                    # `Mechanics for predictit-8544-33624 unreadable:
                    # AttributeError: 'PredictItAdapter' object has no attribute
                    # 'get_mechanics'` for every PredictIt market of every cycle,
                    # which reads like a broken agent. It is a venue that has no
                    # CLOB mechanics to read; the declared fee is used instead,
                    # and neg_risk stays unknown rather than being guessed.
                    # The memory lives on the ADAPTER, not on this agent: the
                    # console builds a fresh agent per request, and a per-agent
                    # memory would print the line again for every one of them.
                    _noted = getattr(adapter, "_mechanics_absent_noted", None)
                    if _noted is None:
                        _noted = set()
                        try:
                            adapter._mechanics_absent_noted = _noted
                        except Exception:  # noqa: BLE001 - a frozen adapter is fine
                            _noted = getattr(self, "_mechanics_absent_noted", set())
                            self._mechanics_absent_noted = _noted
                    _venue = str(getattr(adapter, "venue_id", "") or "")
                    if _venue not in _noted:
                        _noted.add(_venue)
                        logger.info(
                            f"{_venue}: this venue publishes no per-market "
                            f"mechanics (no get_mechanics reader), so the fee comes "
                            f"from its declared capability and exclusivity stays "
                            f"unknown. Said once, not once per market.")
                if mechanics is not None and getattr(mechanics, "is_real", False):
                    facts["neg_risk"] = bool(getattr(mechanics, "neg_risk", False))
                    facts["fee"] = {
                        "rate": float(getattr(mechanics, "taker_fee_rate", 0.0) or 0.0),
                        "source": (f"{adapter.venue_id} clob market info "
                                   f"({getattr(mechanics, 'tick_size', '?')} tick)")}
                    facts["source"] = "clob_market_info"
                if facts["fee"] is None:
                    declared = getattr(adapter.capabilities, "fee_taker_pct", None)
                    if declared is not None:
                        facts["fee"] = {
                            "rate": float(declared),
                            "source": (f"{adapter.venue_id} declared capability "
                                       f"fee_taker_pct")}
                        facts["source"] = facts["source"] or "capability"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Venue facts for {key} not readable: "
                         f"{type(e).__name__}: {e}")
        cache[key] = facts
        return facts

    def _arb_fee_rate(self, market: Market) -> Optional[Dict[str, Any]]:
        """The venue's own taker fee, or None when it could not be read."""
        return self._arb_venue_facts(market).get("fee")

    def _arb_neg_risk(self, market: Market) -> Optional[bool]:
        """The venue's own answer on whether this market's event is exclusive."""
        return self._arb_venue_facts(market).get("neg_risk")

    def effective_execution_mode(self) -> Dict[str, Any]:
        """
        Paper or live, from BOTH halves of the decision, with the reason.

        The operator's switch (`console.mode`, which the page writes) is the
        INTENT. Whether this process can sign an order at all is the CAPABILITY,
        fixed when it started (`dry_run`). Acting on intent alone would let a
        page click try to send real orders from a paper process; acting on
        capability alone is what the operator complained about - the switch
        "did nothing" because the running process never re-read it.

        Both must agree for a live cycle, and when they do not, this says which
        half is missing. Re-read every cycle, so flipping the switch in the page
        takes effect at the next cycle instead of at the next restart.
        """
        asked = "paper"
        try:
            from ..execution.capital import operator_mode
            asked = str(operator_mode(self.storage) or "paper").lower()
        except Exception as e:  # noqa: BLE001 - unreadable intent means paper
            logger.debug(f"Operator mode unreadable: {type(e).__name__}: {e}")
        if asked != "live":
            return {"mode": "paper", "asked": asked,
                    "why": "the operator's switch is on paper"}
        if self.dry_run:
            return {"mode": "paper", "asked": asked,
                    "why": ("the switch is on live, but this process was started "
                            "in dry run, so no order can leave it - restart the "
                            "agent in live mode (" + self._live_command() + ")")}
        return {"mode": "live", "asked": asked,
                "why": "the switch is on live and this process can sign orders"}

    @staticmethod
    def _live_command() -> str:
        return "python main.py run --live"

    def request_immediate_cycle(self) -> bool:
        """
        Ask the running loop to start its next cycle now.

        The console's "Run a round" button used to build a SECOND engine inside
        the web process, which then wrote to the same liveness keys and round
        history as the loop that was already running. Waking the one loop is what
        the button always meant.

        THREAD SAFETY, and this is not theoretical: the web handler runs in the
        console's event loop while the agent runs in its own loop in another
        thread. `asyncio.Event.set()` called from a foreign thread sets the flag
        but does not wake the waiter on the other loop, so a round request sat
        there until the next scheduled cycle - the button looked like it did
        nothing. Setting the event THROUGH the agent's loop is what wakes it.
        """
        wake = getattr(self, "_wake_event", None)
        if wake is None:
            return False
        try:
            loop = getattr(self, "_wake_loop", None)
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if loop is not None and loop is not running:
                loop.call_soon_threadsafe(wake.set)
            else:
                wake.set()
            logger.info("A cycle was requested by the operator; waking the loop")
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not wake the loop: {type(e).__name__}: {e}")
            return False

    async def _wait_for_next_cycle(self, minutes: int) -> Optional[int]:
        """
        Sleep until the next cycle, wakeable early.

        Two things shorten the wait, and both are settings the operator changes
        in the page:

          * a manual "run a round now" sets the wake event;
          * a changed interval means this wait is for the wrong length, so the
            wait is abandoned and the caller re-reads it (within a minute, not
            after the whole old interval).

        The wait is sliced at a minute so neither change needs a restart.
        """
        from .engine_host import operator_interval_minutes

        wake = getattr(self, "_wake_event", None)
        remaining = max(0.0, float(minutes) * 60.0)
        started_with = int(minutes)
        while remaining > 0:
            if wake is not None and wake.is_set():
                wake.clear()
                return operator_interval_minutes(self.storage, started_with)
            now_interval = operator_interval_minutes(self.storage, started_with)
            if now_interval != started_with:
                logger.info(f"Cycle interval changed to {now_interval} minute(s) "
                            f"(was {started_with}) - applying it now instead of "
                            f"waiting out the old one")
                return now_interval
            slice_seconds = min(WAIT_SLICE_SECONDS, remaining)
            if wake is not None:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=slice_seconds)
                    continue
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(slice_seconds)
            remaining -= slice_seconds
        return None

    def _set_phase(self, phase: str, detail: Optional[str] = None,
                   next_cycle_at: Optional[str] = None) -> None:
        """
        Say what the cycle is doing WHILE it is doing it.

        The heartbeat answers "alive"; a scan row answers "finished a cycle".
        Neither answers the question the operator actually asks between cycles -
        "what is it doing right now?" - because a cycle can run for minutes and
        leaves no trace until it ends. This is that trace.

        Never raises: a cycle must not die because a progress note could not be
        written. The failure is logged instead, because a console that shows
        nothing while the agent works is a bug report, not a crash.
        """
        label = self._PHASE_LABELS.get(phase, phase.replace("_", " "))
        # The lease is renewed on every phase write. This is the pulse the agent
        # produces most often (once per market), so a cycle that runs for minutes
        # keeps its claim alive without anyone remembering to renew it.
        if getattr(self, "engine_role", None):
            try:
                from .engine_host import renew_engine_lease
                renew_engine_lease(self.storage, kind=self.engine_role)
            except Exception as e:  # noqa: BLE001 - never break a cycle over this
                logger.debug(f"Lease renew failed: {type(e).__name__}: {e}")
        try:
            self.storage.set_state("agent.phase", json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "label": label,
                "detail": detail or label,
                "next_cycle_at": next_cycle_at,
            }))
        except Exception as e:
            logger.warning(
                f"Could not write the agent phase '{phase}': "
                f"{type(e).__name__}: {e}")

    async def run_continuous(self, interval_minutes: Optional[int] = None,
                             kind: Optional[str] = None,
                             force_lease: bool = False) -> Dict[str, Any]:
        """
        The agent's loop: one cycle, a wait, repeat - owned by exactly one process.

        Three things changed for the operator's 2026-09-29 report, and all three
        were about the loop being deaf to the page:

          * it CLAIMS A LEASE first, so a second engine is refused by pid instead
            of quietly writing to the same keys (the console used to run a rival
            engine on every button press);
          * the interval is re-read from the operator's setting for every wait, so
            a change applies to the next cycle instead of the next restart; and
          * the wait is interruptible, so "run a round now" runs the round on THIS
            engine rather than a second one.

        Returns a small dict so a caller (the console host) can put the reason on
        the page instead of guessing from silence.
        """
        from .engine_host import (claim_engine_lease,
                                  operator_interval_minutes,
                                  release_engine_lease)

        self.engine_role = kind or getattr(self, "engine_role", None) or "cli"
        role = self.engine_role
        requested = operator_interval_minutes(self.storage, interval_minutes)
        claim = claim_engine_lease(self.storage, role, interval_min=requested,
                                   force=force_lease)
        if not claim.get("claimed"):
            holder = claim.get("holder") or {}
            logger.error(
                f"NOT starting a second engine: {claim.get('reason')} "
                f"(last seen {holder.get('age_seconds')}s ago). Stop that one "
                f"first - from the console's Stop button, or by closing its "
                f"window - or start this one with --force to take over.")
            return {"started": False, "reason": claim.get("reason"),
                    "holder": holder}

        # The wake event lives as long as the loop does, and remembers ITS loop:
        # a request to run now arrives from the console's thread, and only a
        # threadsafe call INTO this loop can wake a waiter on it.
        self._wake_event = asyncio.Event()
        self._wake_loop = asyncio.get_running_loop()
        interval = int(requested)
        logger.info(f"Starting PTAI V3 continuous loop every {interval} minutes "
                    f"(role: {role}, pid {os.getpid()})")
        try:
            while True:
                try:
                    if not self.kill_switch.can_trade():
                        logger.warning(f"Kill switch L{self.kill_switch.current_level} blocks trading, sleeping")
                        # The agent is REFUSING, not dead: keep the heartbeat
                        # fresh while the kill switch holds the loop.
                        self._write_agent_heartbeat(
                            f"kill_switch_L{self.kill_switch.current_level}")
                        await asyncio.sleep(60)
                        continue

                    mode = self.effective_execution_mode()
                    if mode["asked"] != mode["mode"]:
                        logger.warning(f"Execution mode: {mode['mode']} "
                                       f"({mode['why']})")

                    # A WORKING phase before the cycle starts, not after its first
                    # step. A cycle's first step can take minutes (a slow model, a
                    # venue that does not answer), and while it ran the page still
                    # showed the RESTING phase - "waiting until 08:04" - which is
                    # what a stalled agent looks like. This also gives the loop a
                    # lease renewal and a heartbeat at the moment it needs them.
                    self.cycles_started = getattr(self, "cycles_started", 0) + 1
                    cycle_started = datetime.now(timezone.utc)
                    self._set_phase(
                        "scanning",
                        f"cycle {self.cycles_started} started at "
                        f"{cycle_started.strftime('%H:%M')} UTC: refreshing "
                        f"qualification and scanning the venues")
                    self._write_agent_heartbeat("running")
                    logger.info(f"Starting cycle {self.cycles_started} at "
                                f"{cycle_started.isoformat()}")

                    result = await self.run_cycle()
                    elapsed = (datetime.now(timezone.utc)
                               - cycle_started).total_seconds()
                    logger.info(
                        f"V3 cycle result: {result.get('status', 'unknown')} "
                        f"{(result.get('opportunities') or {}).get('final_selected', 0)} trades "
                        f"(cycle {self.cycles_started} took {elapsed:.0f}s)")

                    # Announce the wait as well, with the time the next cycle is
                    # due. "Waiting until 14:35" is the answer to the operator's
                    # next question, and without it the console goes blank for the
                    # whole interval, which is what a stalled agent looks like.
                    next_at = datetime.now(timezone.utc) + timedelta(minutes=interval)
                    self._set_phase(
                        "sleeping",
                        f"last cycle: {result.get('status', 'unknown')}. The next "
                        f"cycle starts at {next_at.strftime('%H:%M')} UTC "
                        f"(every {interval} min, set in the console)",
                        next_cycle_at=next_at.isoformat())

                    changed = await self._wait_for_next_cycle(interval)
                    if changed:
                        interval = int(changed)
                except asyncio.CancelledError:
                    logger.info("The agent loop was stopped by the operator")
                    raise
                except Exception as e:
                    logger.error(f"V3 loop error: {type(e).__name__}: {e}")
                    await asyncio.sleep(60)
        finally:
            release_engine_lease(self.storage,
                                 reason=f"the {role} engine stopped")
            self._set_phase("stopped", "the agent was stopped; it is not working "
                                       "and nothing is being researched")
            logger.info(f"The {role} engine released the agent")
