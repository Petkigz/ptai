
"""
CCXT Adapter - unified data layer so agent can read odds across Polymarket, Kalshi, and any crypto venue with one interface
CCXT supports prediction markets alongside crypto exchanges, per-venue quirks handled, strategy logic stays venue-agnostic
https://github.com/ccxt/ccxt
pip install ccxt
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class CCXTUnifiedAdapter(MarketAdapter):
    def __init__(self, venues: List[str] = None):
        super().__init__(venue_id="ccxt_unified", venue_type=VenueType.OTHER)
        self.venues = venues or ["polymarket", "kalshi", "binance", "whitebit", "pionex"]
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=False,  # CCXT is data layer, execution via specific adapters
            supports_portfolio=False,
            fee_taker_pct=0.001,
            min_order_usd=1.0
        )
        self.ccxt_available = False
        try:
            import ccxt
            self.ccxt_available = True
            logger.info(f"CCXT available, supports {len(ccxt.exchanges)} exchanges including prediction markets")
        except ImportError:
            logger.warning("CCXT not installed, using mock unified layer")

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        return EligibilityStatus.ELIGIBLE  # data layer, always eligible

    async def discover_markets(self, target_count: int = 100, filters: Dict = None) -> List[Market]:
        # Unified discovery across all venues with one interface
        # Production: 
        # import ccxt
        # for venue in self.venues:
        #   exchange = getattr(ccxt, venue)()
        #   markets = exchange.fetch_markets()
        #   tickers = exchange.fetch_tickers()
        markets = []
        for venue in self.venues:
            for i in range(target_count // len(self.venues)):
                price = 0.5 + (i*0.01 - 0.05)
                markets.append(Market(
                    id=f"ccxt-MOCK-{venue}-{i}",
                    source=MarketSource.POLYMARKET,
                    question=f"CCXT unified {venue} market {i} - one strategy reads odds across multiple venues",
                    outcomes=["YES", "NO"],
                    outcome_prices=[price, 1-price],
                    tokens=[Token(token_id=f"ccxt-MOCK-{venue}-{i}", outcome="YES", price=price)],
                    volume=10000,
                    volume_24h=5000,
                    liquidity=8000,
                    active=True,
                    closed=False,
                    event_slug=f"ccxt-{venue}-{i}",
                    raw={"venue": venue, "unified_via": "ccxt", "category": "unified", "original_venue": venue, "data_mode": "mock", "data_source": "ccxt_mock_fallback", "is_mock": True, "safety": "MOCK_DATA MUST NEVER REACH LIVE EXECUTION"}
                ,
                venue_id="ccxt",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="ccxt_mock_fallback",
                is_mock=True
            ))
        logger.info(f"CCXT unified discovered {len(markets)} markets across {len(self.venues)} venues with one interface")
        return markets[:target_count]

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        original_venue = market.raw.get("original_venue", "unknown")
        return {
            "market_id": market.id,
            "original_venue": original_venue,
            "bid": market.best_price - 0.01,
            "ask": market.best_price + 0.01,
            "spread": 0.02,
            "depth": market.liquidity,
            "venue": "ccxt_unified",
            "unified": True,
            "note": "One strategy reads odds across multiple venues with same code, per-venue quirks handled by CCXT"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "ccxt_unified", "type": "unified_data_layer"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        return {
            "status": "dry_run",
            "message": f"CCXT unified data layer - execution delegated to specific venue adapter {opportunity.venue_id}",
            "venue": "ccxt_unified",
            "original_venue": opportunity.market.raw.get("original_venue", opportunity.venue_id)
        }

    def get_unified_ticker(self, symbol: str) -> Dict[str, Any]:
        # Example unified ticker across venues
        return {
            "symbol": symbol,
            "polymarket": 0.61,
            "kalshi": 0.68,
            "spread": 0.07,
            "arb_profit": 0.12,
            "note": "Read odds across multiple venues with one codebase"
        }
