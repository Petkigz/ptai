"""
Market Adapter Interface - market-agnostic design
Polymarket is one adapter among many. Each venue must pass qualification.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

from loguru import logger

from ..markets.base import Market
from ..markets.orderbook import read_spread


class VenueType(str, Enum):
    PREDICTION = "prediction"  # Polymarket, Kalshi
    FINANCIAL = "financial"    # Stocks, ETFs, Crypto
    OTHER = "other"            # Auctions, marketplaces


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    RESTRICTED = "restricted"  # Geographic or regulatory
    UNKNOWN = "unknown"
    REQUIRES_VERIFICATION = "requires_verification"


# Implementation status. Distinct from eligibility, which is a regulatory
# question: a venue can be perfectly legal to trade and still have no code
# behind it.
STATUS_LIVE = "live"                    # issues real requests to the venue
STATUS_UNIMPLEMENTED = "unimplemented"  # no client exists; must never return markets
STATUS_SCANNER = "scanner"              # read-only aggregator, cannot place orders


@dataclass
class AdapterCapability:
    supports_market_discovery: bool = True
    supports_orderbook: bool = False
    supports_trading: bool = False
    supports_portfolio: bool = False
    supports_history: bool = False
    supports_browser_fallback: bool = False
    fee_taker_pct: float = 0.0  # e.g. 0.02 = 2%
    fee_maker_pct: float = 0.0
    min_order_usd: float = 1.0
    # Which of the above are actually true. An adapter that has never issued an
    # HTTP request must not advertise market discovery, and must not appear
    # eligible for trading.
    implementation_status: str = STATUS_LIVE
    # One line on what is missing, shown in the UI so a user can tell why a
    # venue they linked produces nothing.
    implementation_note: str = ""

    @property
    def is_implemented(self) -> bool:
        return self.implementation_status in (STATUS_LIVE, STATUS_SCANNER)


@dataclass
class VenueOpportunity:
    """Normalized opportunity across any venue - common basis for ranking
    V9 FIX #1: Hard LIVE/PAPER/MOCK separation - data_mode on Opportunity as well as Market
    """
    market: Market
    venue_id: str
    venue_type: VenueType
    side: str  # YES/NO, BUY/SELL, etc
    market_price: float  # 0-1 for prediction, normalized for others
    estimated_fair: float
    raw_edge: float
    effective_edge: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.1
    liquidity_score: float = 0.5
    execution_quality: float = 0.5
    time_to_resolution_hours: Optional[float] = None
    fees_pct: float = 0.0
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    correlation_group: str = ""  # For portfolio correlation cap
    category: str = ""  # politics, sports, crypto, etc
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    bull_case: str = ""
    bear_case: str = ""
    resolution_risks: List[str] = field(default_factory=list)
    should_trade: bool = False
    score: float = 0.0  # Common opportunity score
    # V9 FIX #1: DataMode on Opportunity as well - plus market_id for dashboard audit
    data_mode: str = "live"  # live/paper/mock - derived from market.data_mode
    is_mock: bool = False
    data_source: str = ""
    market_id: str = ""  # explicit for dashboard /api/v3/opportunities
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # V9: Sync data_mode/is_mock/market_id from market
        try:
            from ..markets.base import DataMode
            if not self.market_id:
                self.market_id = getattr(self.market, 'id', '')
            m_mode = getattr(self.market, 'data_mode', DataMode.LIVE)
            if hasattr(m_mode, 'value'):
                m_mode_val = m_mode.value
            else:
                m_mode_val = str(m_mode).lower()
            self.data_mode = m_mode_val
            self.is_mock = getattr(self.market, 'is_mock', False) or m_mode_val == "mock" or "MOCK" in str(self.market.id).upper() or "MOCK" in str(self.market_id).upper()
            self.data_source = getattr(self.market, 'data_source', '')
            # MOCK must be impossible to reach live execution
            if self.is_mock or self.data_mode == "mock":
                self.should_trade = False
        except Exception:
            pass
        # Ensure data_mode string
        if hasattr(self.data_mode, 'value'):
            self.data_mode = self.data_mode.value
        self.data_mode = str(self.data_mode).lower()
        if self.data_mode == "mock":
            self.is_mock = True

    def calculate_common_score(self) -> float:
        """
        Common score:
        expected_edge * prob_correct * liquidity * execution * calibration * time_efficiency
        / (fees + slippage + uncertainty + risk)
        """
        numerator = (
            max(0, self.effective_edge) *
            self.confidence *
            self.liquidity_score *
            self.execution_quality *
            (1.0 / max(1.0, (self.time_to_resolution_hours or 24) / 24))  # time efficiency
        )
        denominator = (
            self.fees_pct +
            self.slippage_pct +
            self.spread_pct +
            self.uncertainty +
            0.01  # avoid div0
        )
        self.score = numerator / denominator if denominator > 0 else 0
        return self.score


class MarketAdapter(ABC):
    """
    Interface every venue must implement.
    Qualification: must prove positive EV through paper trading before real capital.
    """
    def __init__(self, venue_id: str, venue_type: VenueType):
        self.venue_id = venue_id
        self.venue_type = venue_type
        self.capabilities = AdapterCapability()
        self.is_qualified = False  # Must pass paper trading
        self.performance_stats = {
            "total_paper_trades": 0,
            "win_rate": 0.0,
            "avg_edge": 0.0,
            "brier_score": 1.0,
            "calibration": 0.0,
            "profit_paper": 0.0,
            "profit_live": 0.0,
            "forecast_skill": 0.5
        }

    @abstractmethod
    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        """Check geographic/regulatory eligibility - NEVER bypass restrictions"""
        pass

    @abstractmethod
    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        """Discover markets - fast path, 500-1000 in seconds"""
        pass

    @abstractmethod
    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        """Get orderbook depth, spread, recent trades"""
        pass

    @abstractmethod
    async def get_portfolio(self) -> Dict[str, Any]:
        """Get positions, balance"""
        pass

    def calculate_fees(self, market: Market, amount_usd: float) -> float:
        """Calculate taker fees for this venue"""
        return amount_usd * self.capabilities.fee_taker_pct

    def estimate_slippage(self, market: Market, amount_usd: float) -> float:
        """Estimate slippage based on liquidity and orderbook"""
        if market.liquidity <= 0:
            return 0.02  # 2% default for illiquid
        # Simple model: slippage ~ amount / liquidity
        ratio = amount_usd / max(1, market.liquidity)
        return min(0.05, ratio * 0.5)  # cap 5%

    def estimate_spread(self, orderbook: Dict) -> float:
        """Estimate spread from orderbook"""
        if not orderbook:
            return 0.01
        bid = orderbook.get("bid", 0)
        ask = orderbook.get("ask", 1)
        if bid and ask:
            return abs(ask - bid)
        return read_spread(orderbook, 0.01)[0]

    @abstractmethod
    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        """
        Deterministic execution - receives ONLY:
        market_id, token_id, side, max_price, max_spend
        LLM can never directly control money amount.
        """
        pass

    async def qualify_via_paper_trading(self, min_trades: int = 100, min_win_rate: float = 0.55) -> bool:
        """
        Venue must prove profitability via paper trading before live capital.
        """
        stats = self.performance_stats
        if stats["total_paper_trades"] < min_trades:
            return False
        if stats["win_rate"] < min_win_rate:
            return False
        if stats["brier_score"] > 0.25:  # poorly calibrated
            return False
        self.is_qualified = True
        return True

    def get_performance_summary(self) -> Dict:
        return {
            "venue_id": self.venue_id,
            "type": self.venue_type.value,
            "qualified": self.is_qualified,
            "implementation_status": self.capabilities.implementation_status,
            **self.performance_stats
        }


class UnimplementedVenueAdapter(MarketAdapter):
    """
    Base for a venue with no working client.

    These adapters previously fabricated markets: `discover_markets` returned
    hardcoded rows with `id="...-MOCK-..."` describing invented prices, which
    then flowed into the scanner, were scored and ranked, and appeared in the
    dashboard as opportunities. The execution guard refused them at the last
    step, so nothing was actually traded - but everything upstream treated them
    as real, which is why the scanner appeared to see hundreds of markets across
    many venues when only a handful were ever reachable.

    Declaring the venue unimplemented fixes that at the source:
      - discover_markets returns nothing, so no fabricated market enters scoring
      - eligibility is UNKNOWN, not ELIGIBLE, so it is never routed to
      - place_order refuses rather than returning a success-shaped message

    The venue still appears in the registry and in the UI, labelled with what is
    missing, because a user needs to know a venue is known-but-unavailable
    rather than silently absent.
    """

    def __init__(self, venue_id: str, venue_type: VenueType, note: str = ""):
        super().__init__(venue_id=venue_id, venue_type=venue_type)
        self.capabilities = AdapterCapability(
            supports_market_discovery=False,
            supports_orderbook=False,
            supports_trading=False,
            supports_portfolio=False,
            implementation_status=STATUS_UNIMPLEMENTED,
            implementation_note=note or "no client implementation",
        )
        self.implementation_note = self.capabilities.implementation_note

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        """
        UNKNOWN, not ELIGIBLE.

        Claiming eligibility is a claim the venue can be traded on, and an
        adapter with no client cannot be traded on by definition.
        """
        return EligibilityStatus.UNKNOWN

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        logger.info(
            f"{self.venue_id}: no client implementation, returning no markets. "
            f"{self.capabilities.implementation_note}")
        return []

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "available": False,
            "reason": self.capabilities.implementation_note,
            "note": "no client implementation - this is not an empty orderbook, it is no orderbook",
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "available": False,
            "balance": None,
            "positions": [],
            "reason": self.capabilities.implementation_note,
        }

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float,
                          max_price: float) -> Dict[str, Any]:
        logger.error(f"{self.venue_id}: order refused - {self.capabilities.implementation_note}")
        return {
            "status": "unimplemented",
            "success": False,
            "venue_id": self.venue_id,
            "error": self.capabilities.implementation_note,
            "message": (f"{self.venue_id} has no client implementation; no order was placed. "
                        f"This is a refusal, not a fill."),
        }
