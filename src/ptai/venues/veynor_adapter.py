
"""
Veynor Adapter - prediction market intelligence API for Kalshi and Polymarket
Whale trades, top markets, cross-venue arb opportunities, smart money signals in single Python import
Free tier 100 credits/month enough for light scanning
https://veynor.com/
pip install veynor
"""
from typing import List, Dict, Any
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token

class VeynorAdapter(MarketAdapter):
    def __init__(self, api_key: str = None):
        super().__init__(venue_id="veynor", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=False,
            supports_trading=False,  # intelligence layer, not trading
            supports_portfolio=False,
            supports_history=True,
            fee_taker_pct=0.0,
            min_order_usd=0.0
        )
        self.credits_free = 100
        self.base_url = "https://api.veynor.com"

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        return EligibilityStatus.ELIGIBLE  # intelligence API, worldwide

    async def discover_markets(self, target_count: int = 50, filters: Dict = None) -> List[Market]:
        # Production: from veynor import Veynor; client = Veynor(api_key); markets = client.get_top_markets()
        # Provides whale trades, smart money signals, arb opportunities
        try:
            # Mock Veynor intelligence
            mock_markets = [
                {"question": "Will Trump win? Veynor whale smart money buying YES", "price": 0.62, "whale": "smart", "arb": 0.05},
                {"question": "Will Fed cut rates? Veynor cross-venue arb Polymarket 0.61 Kalshi 0.68", "price": 0.61, "whale": "neutral", "arb": 0.07},
                {"question": "Will BTC >$100k? Veynor top market volume high", "price": 0.55, "whale": "smart", "arb": 0.02},
            ]
            markets = []
            for i, m in enumerate(mock_markets[:target_count]):
                markets.append(Market(
                    id=f"veynor-{i}",
                    source=MarketSource.POLYMARKET,
                    question=m["question"],
                    outcomes=["YES", "NO"],
                    outcome_prices=[m["price"], 1-m["price"]],
                    tokens=[Token(token_id=f"veynor-{i}", outcome="YES", price=m["price"])],
                    volume=50000,
                    volume_24h=20000,
                    liquidity=30000,
                    active=True,
                    closed=False,
                    event_slug=f"veynor-{i}",
                    raw={"venue": "veynor", "whale_signal": m["whale"], "arb_spread": m["arb"], "category": "intelligence", "credits": self.credits_free}
                ))
            logger.info(f"Veynor discovered {len(markets)} intelligence markets (whale trades, smart money, arb) free tier {self.credits_free} credits/month")
            return markets
        except Exception as e:
            logger.error(f"Veynor failed: {e}")
            return []

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {
            "market_id": market.id,
            "venue": "veynor",
            "whale_trades": "Available via Veynor API",
            "smart_money": market.raw.get("whale_signal"),
            "arb_opportunities": market.raw.get("arb_spread"),
            "type": "intelligence_layer",
            "note": "Single Python import for cross-venue data Kalshi and Polymarket"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "veynor", "type": "intelligence", "credits_free": 100}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        return {
            "status": "intelligence",
            "message": f"Veynor intelligence layer - whale trades, smart money, arb opportunities, not trading venue. Market {opportunity.market.id} whale {opportunity.market.raw.get('whale_signal')} arb {opportunity.market.raw.get('arb_spread')}",
            "venue": "veynor",
            "action": "Use intelligence to inform trading on Polymarket/Kalshi adapters"
        }

    def get_whale_trades(self, limit: int = 20) -> List[Dict]:
        # Veynor provides whale trades
        return [
            {"whale": "0x1234...smart", "market": "Trump win", "side": "YES", "amount": 10000, "price": 0.60, "pnl": 5000, "win_rate": 0.75},
            {"whale": "0x5678...dumb", "market": "Biden win", "side": "YES", "amount": 5000, "price": 0.55, "pnl": -2000, "win_rate": 0.35},
        ]

    def get_arb_opportunities(self) -> List[Dict]:
        # Cross-venue arb via Veynor
        return [
            {"event": "Fed cut June", "polymarket": 0.61, "kalshi": 0.68, "spread": 0.07, "profit": 0.12, "confidence_same_event": 0.85},
        ]
