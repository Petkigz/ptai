
"""
PredictIt Adapter - public read-only feed heavy limits, US politics only
Treat as data source for sentiment, not trading venue
API: https://www.predictit.org/api/marketdata/all/  (public, heavy limits)
"""
from typing import List, Dict, Any
from loguru import logger
import requests

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class PredictItAdapter(MarketAdapter):
    def __init__(self):
        super().__init__(venue_id="predictit", venue_type=VenueType.PREDICTION)
        self.last_error: str = ""
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=False,
            supports_trading=False,  # read-only for PTAI, not trading venue
            supports_portfolio=False,
            supports_history=False,
            fee_taker_pct=0.10,  # 10% fee on profits + 5% withdrawal
            fee_maker_pct=0.10,
            min_order_usd=1.0
        )
        self.api_url = "https://www.predictit.org/api/marketdata/all/"

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # US only for trading, but data worldwide
        cc = country_code.upper()
        if cc == "US":
            return EligibilityStatus.ELIGIBLE
        # For data sentiment, eligible everywhere but trading restricted
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 100, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        try:
            resp = requests.get(self.api_url, timeout=10, headers={"User-Agent": "PTAI/1.0"})
            if resp.status_code != 200:
                self.last_error = f"PredictIt API returned HTTP {resp.status_code}"
                logger.warning(f"PredictIt: {self.last_error}; no markets returned")
                return []
            data = resp.json()
            markets = []
            for m in data.get("markets", [])[:target_count]:
                try:
                    # PredictIt structure: markets have contracts
                    q = m.get("name", "Unknown")
                    contracts = m.get("contracts", [])
                    if not contracts:
                        continue
                    c = contracts[0]
                    yes_price = c.get("lastTradePrice", 0.5)
                    if yes_price > 1:
                        yes_price = yes_price / 100.0
                    market = Market(
                        id=f"predictit-{m.get('id')}-{c.get('id')}",
                        source=MarketSource.PREDICTIT if hasattr(MarketSource, 'PREDICTIT') else MarketSource.POLYMARKET,
                        question=q[:200],
                        description=m.get("shortName", ""),
                        outcomes=["YES", "NO"],
                        outcome_prices=[yes_price, 1-yes_price],
                        tokens=[Token(token_id=str(c.get("id")), outcome="YES", price=yes_price)],
                        volume=c.get("volume", 0) or 0,
                        volume_24h=c.get("volume", 0) or 0,
                        liquidity=1000,
                        active=True,
                        closed=False,
                        slug=str(m.get("id")),
                        event_slug=m.get("shortName", "")[:50],
                        raw={"venue": "predictit", "api": "public", "original": m, "data_mode": "live", "data_source": "predictit_api"},
                        venue_id="predictit",
                        venue_type="prediction",
                        data_mode=DataMode.LIVE,
                        data_source="predictit_api",
                        is_mock=False
                    )
                    markets.append(market)
                except Exception as e:
                    logger.debug(f"PredictIt parse fail: {e}")
                    continue
            if not markets:
                self.last_error = "PredictIt returned no parseable markets"
                return []
            logger.info(f"PredictIt discovered {len(markets)} markets")
            return markets[:target_count]
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(f"PredictIt discovery failed: {self.last_error}; no markets returned")
            return []

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        return {"spread": 0.05, "bid": market.best_price-0.02, "ask": market.best_price+0.02, "depth": market.liquidity, "source": "predictit_mock"}

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "note": "PredictIt read-only sentiment source, not trading venue"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        return {"status": "dry_run", "message": f"PredictIt sentiment only - would not trade {opportunity.market.id}, use for reference odds", "venue": "predictit"}
