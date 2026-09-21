
"""
Pionex Adapter - free REST and WebSocket API 10 req/sec, spot bot futures
Built-in bot strategies grid DCA can be controlled via API, reduces need to build execution logic
API: https://pionex.com/apidocs
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class PionexAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(venue_id="pionex", venue_type=VenueType.FINANCIAL)
        self.api_key = api_key
        self.api_secret = api_secret
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key and api_secret),
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.0005,
            fee_maker_pct=0.0005,
            min_order_usd=1.0
        )
        self.rate_limit = "10 req/sec"

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        cc = country_code.upper()
        if cc == "UG":
            return EligibilityStatus.ELIGIBLE
        if cc == "US":
            return EligibilityStatus.RESTRICTED
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 50, filters: Dict = None) -> List[Market]:
        # Mock - production: GET https://api.pionex.com/api/v1/market/tickers
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "MATIC_USDT", "AVAX_USDT", "DOT_USDT"]
        markets = []
        for i, sym in enumerate(symbols[:target_count]):
            prob = 0.5 + (i*0.02 - 0.06)
            markets.append(Market(
                id=f"pionex-MOCK-{sym}",
                source=MarketSource.POLYMARKET,
                question=f"Will {sym} grid bot profitable? Pionex built-in bots",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=[Token(token_id=sym, outcome="YES", price=prob)],
                volume=80000,
                volume_24h=30000,
                liquidity=15000,
                active=True,
                closed=False,
                event_slug=sym,
                raw={"venue": "pionex", "symbol": sym, "type": "spot", "category": "crypto", "bots": ["grid", "DCA", "infinity_grid"], "rate_limit": "10 req/sec", "data_mode": "mock", "data_source": "pionex_mock_fallback", "is_mock": True, "safety": "MOCK_DATA MUST NEVER REACH LIVE EXECUTION"}
            ,
                venue_id="pionex",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="pionex_mock_fallback",
                is_mock=True
            ))
        logger.info(f"Pionex discovered {len(markets)} markets (rate limit {self.rate_limit}, built-in bots grid/DCA)")
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "market_id": market.id,
            "symbol": market.raw.get("symbol"),
            "bid": market.best_price - 0.001,
            "ask": market.best_price + 0.001,
            "spread": 0.002,
            "depth": market.liquidity,
            "venue": "pionex",
            "rate_limit": self.rate_limit,
            "bots": "grid, DCA, infinity_grid controllable via API"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "pionex", "bots": ["grid", "DCA"]}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        # Pionex can use built-in bots via API, reducing execution logic needed
        return {
            "status": "dry_run",
            "message": f"Pionex order {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id} - can use built-in grid/DCA bots via API, rate limit 10 req/sec",
            "venue": "pionex",
            "bots_available": ["grid", "DCA", "infinity_grid"]
        }
