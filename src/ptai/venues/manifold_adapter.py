"""
Manifold Markets Adapter
API: https://docs.manifold.markets/api
Public: GET https://api.manifold.markets/v0/markets
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode


MANIFOLD_API = "https://api.manifold.markets/v0"


class ManifoldAdapter(MarketAdapter):
    def __init__(self, api_key: str = None):
        super().__init__(venue_id="manifold", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        self.base_url = MANIFOLD_API
        self.last_error: str = ""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=False,  # Manifold uses AMM, not orderbook
            supports_trading=bool(api_key),
            supports_portfolio=True,
            supports_history=True,
            supports_browser_fallback=True,
            fee_taker_pct=0.0,  # No fees, but 5% liquidity pool
            fee_maker_pct=0.0,
            min_order_usd=1.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        # Manifold is play money (Mana), generally available worldwide
        # Real money via sweepstakes may be restricted
        country_code = country_code.upper()
        if country_code in {"US", "UG", "GB", "CA", "AU", "DE", "FR", "KE", "NG", "ZA"}:
            return EligibilityStatus.ELIGIBLE
        return EligibilityStatus.ELIGIBLE

    def _parse_manifold_market(self, raw: Dict) -> Optional[Market]:
        try:
            if raw.get("outcomeType") != "BINARY":
                return None  # Only binary for now
            
            question = raw.get("question", "")
            prob = raw.get("probability", 0.5)
            volume = float(raw.get("volume", 0))
            volume_24h = float(raw.get("volume24Hours", volume * 0.2))
            liquidity = float(raw.get("pool", {}).get("NO", 0) + raw.get("pool", {}).get("YES", 0)) if isinstance(raw.get("pool"), dict) else float(raw.get("totalLiquidity", 1000))
            
            close_time = raw.get("closeTime")
            end_date = None
            if close_time:
                try:
                    end_date = datetime.fromtimestamp(close_time / 1000)
                except:
                    pass

            tokens = [
                Token(token_id=f"{raw.get('id')}_YES", outcome="YES", price=prob),
                Token(token_id=f"{raw.get('id')}_NO", outcome="NO", price=1-prob)
            ]

            market = Market(
                id=str(raw.get("id")),
                source=MarketSource.PREDICTIT,
                question=question,
                description=raw.get("description", "")[:500] if raw.get("description") else "",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=tokens,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                active=not raw.get("isResolved", False) and not raw.get("closeTime", 0) < (datetime.now().timestamp()*1000),
                closed=raw.get("isResolved", False),
                slug=raw.get("slug", ""),
                event_slug="",
                condition_id=str(raw.get("id")),
                market_type="binary",
                raw={**raw, "venue": "manifold", "data_mode": "live", "data_source": "manifold_api"},
                venue_id="manifold",
                venue_type="prediction",
                data_mode=DataMode.LIVE,
                data_source="manifold_api",
                is_mock=False
            )
            # Override source string
            market.raw["venue"] = "manifold"
            return market
        except Exception as e:
            logger.debug(f"Failed to parse Manifold market: {e}")
            return None

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        markets: List[Market] = []
        
        try:
            params = {"limit": min(100, target_count), "sort": "24-hour-vol", "filter": "open"}
            resp = self.session.get(f"{self.base_url}/markets", params=params, timeout=10)
            if resp.status_code == 200:
                raw_markets = resp.json()
                if isinstance(raw_markets, list):
                    for rm in raw_markets[:target_count]:
                        m = self._parse_manifold_market(rm)
                        if m:
                            markets.append(m)
                if markets:
                    logger.info(f"Manifold discovered {len(markets)} real markets")
                    return markets[:target_count]
        except Exception as e:
            logger.warning(f"Manifold API failed (expected offline): {e}")

        # No mock fallback. This used to fabricate up to 150 Manifold questions
        # from templates with random prices on ANY failure - so a network
        # outage, which is the common case, silently became invented markets
        # flowing into the scanner. An empty result with a stated reason is the
        # only honest outcome.
        self.last_error = ("Manifold API unreachable; no markets returned. "
                           "Refusing to fabricate markets on a network failure.")
        logger.warning(f"Manifold discovery returned no markets: {self.last_error}")
        return []

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # Manifold uses AMM, not orderbook - simulate spread based on liquidity
        spread = 0.03 if market.liquidity > 2000 else 0.06
        return {
            "market_id": market.id,
            "token_id": market.yes_token_id,
            "bid": max(0.01, market.yes_price - spread/2),
            "ask": min(0.99, market.yes_price + spread/2),
            "spread": spread,
            "depth": market.liquidity,
            "amm": True
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "orders": [], "venue": "manifold", "paper": True}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd <= 0 or max_price <= 0 or max_price >= 1:
            return {"status": "rejected", "reason": "Invalid guard params", "venue": "manifold"}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000", "venue": "manifold"}
        
        logger.info(f"Manifold guard: market={opportunity.market.id} side={opportunity.side} max_price={max_price} max_spend=${max_spend_usd}")
        
        return {
            "status": "dry_run",
            "venue": "manifold",
            "message": f"Would place {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id}",
            "market_id": opportunity.market.id
        }
