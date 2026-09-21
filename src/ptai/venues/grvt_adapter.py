
"""
GRVT Adapter - hybrid derivatives exchange with central limit order book CLOB
Hummingbot integration guide recommends min ~$50 for testing and $200+ for live
Slightly above $50 bankroll but viable after initial growth
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class GRVTAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, private_key: str = None):
        super().__init__(venue_id="grvt", venue_type=VenueType.FINANCIAL)
        self.api_key = api_key
        self.private_key = private_key
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key),
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.0003,
            fee_maker_pct=0.0,
            min_order_usd=1.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        cc = country_code.upper()
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        if cc == "US":
            return EligibilityStatus.RESTRICTED
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 20, filters: Dict = None) -> List[Market]:
        # Mock - production: GRVT API https://docs.grvt.io/
        # Hummingbot integration: https://hummingbot.org/exchanges/grvt/
        symbols = ["BTC_USDT-PERP", "ETH_USDT-PERP", "SOL_USDT-PERP"]
        markets = []
        for i, sym in enumerate(symbols[:target_count]):
            prob = 0.5 + (i*0.02 - 0.02)
            markets.append(Market(
                id=f"grvt-MOCK-{sym}",
                source=MarketSource.POLYMARKET,
                question=f"Will {sym} funding rate positive? GRVT hybrid CLOB",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=[Token(token_id=sym, outcome="YES", price=prob)],
                volume=100000,
                volume_24h=50000,
                liquidity=25000,
                active=True,
                closed=False,
                event_slug=sym,
                raw={"venue": "grvt", "symbol": sym, "type": "hybrid_perp", "category": "crypto", "min_testing": 50, "min_live": 200, "clob": True, "data_mode": "mock", "data_source": "grvt_mock_fallback", "is_mock": True, "safety": "MOCK_DATA MUST NEVER REACH LIVE EXECUTION"}
            ,
                venue_id="grvt",
                venue_type="other",
                data_mode=DataMode.MOCK,
                data_source="grvt_mock_fallback",
                is_mock=True
            ))
        logger.info(f"GRVT discovered {len(markets)} hybrid perpetual markets (min testing $50 live $200+)")
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "market_id": market.id,
            "symbol": market.raw.get("symbol"),
            "bid": market.best_price - 0.001,
            "ask": market.best_price + 0.001,
            "spread": 0.002,
            "depth": market.liquidity,
            "venue": "grvt",
            "type": "hybrid CLOB",
            "hummingbot": "supported"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "grvt", "min_testing": 50, "min_live": 200}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd < 50:
            return {"status": "dry_run", "message": f"GRVT min $50 testing, you have ${max_spend_usd} - viable after initial growth", "venue": "grvt"}
        return {
            "status": "dry_run",
            "message": f"GRVT hybrid CLOB order {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id}",
            "venue": "grvt"
        }
