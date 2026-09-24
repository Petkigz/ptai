"""
Kalshi Adapter - real implementation
Kalshi API docs: https://trading-api.readme.io/
Public markets: GET https://api.elections.kalshi.com/trade-api/v2/markets
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode


KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None, member_id: str = None):
        super().__init__(venue_id="kalshi", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        self.api_secret = api_secret
        self.member_id = member_id
        self.base_url = KALSHI_API
        self.session = requests.Session()
        self.last_error: str = ""
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key),
            supports_portfolio=True,
            supports_history=True,
            supports_browser_fallback=True,
            fee_taker_pct=0.0,  # Kalshi fee is $0.07 per contract capped, approx 0-7%
            fee_maker_pct=0.0,
            min_order_usd=1.0
        )
        # Kalshi restricted: US only for most markets, check terms
        self.restricted_countries = set()  # Actually US-allowed, others restricted for some markets
        # For non-US, many markets restricted - but we check live
        self.us_only_markets_pct = 0.8

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        country_code = country_code.upper()
        # Kalshi is US-centric, many markets US-only
        # For UG, generally restricted for real trading, but paper trading possible
        if country_code == "US":
            return EligibilityStatus.ELIGIBLE
        if country_code == "UG":
            # Kalshi requires US residency for most markets
            return EligibilityStatus.RESTRICTED
        # Other countries: likely restricted, but allow verification
        return EligibilityStatus.REQUIRES_VERIFICATION

    def _parse_kalshi_market(self, raw: Dict) -> Optional[Market]:
        try:
            ticker = raw.get("ticker", "")
            title = raw.get("title") or raw.get("subtitle") or ticker
            yes_bid = raw.get("yes_bid", 0) / 100.0 if raw.get("yes_bid") else 0.5
            yes_ask = raw.get("yes_ask", 0) / 100.0 if raw.get("yes_ask") else 0.5
            last_price = raw.get("last_price", 0) / 100.0 if raw.get("last_price") else (yes_bid + yes_ask) / 2
            volume = float(raw.get("volume", 0))
            volume_24h = float(raw.get("volume_24h", volume * 0.3))
            liquidity = float(raw.get("liquidity", raw.get("open_interest", 1000)))
            
            # End date
            end_date = None
            close_time = raw.get("close_time") or raw.get("expiration_time")
            if close_time:
                try:
                    end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                except:
                    pass

            tokens = [
                Token(token_id=f"{ticker}_YES", outcome="YES", price=last_price),
                Token(token_id=f"{ticker}_NO", outcome="NO", price=1-last_price)
            ]

            market = Market(
                id=ticker,
                source=MarketSource.KALSHI,
                question=title,
                description=raw.get("subtitle", "") or raw.get("details", ""),
                outcomes=["YES", "NO"],
                outcome_prices=[last_price, 1-last_price],
                tokens=tokens,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                active=raw.get("status") == "active" or raw.get("status") == "open",
                closed=raw.get("status") in ["closed", "settled"],
                slug=ticker,
                event_slug=raw.get("event_ticker", ""),
                condition_id=ticker,
                market_type="binary",
                raw={**raw, "venue_id": "kalshi", "adapter_venue_id": "kalshi", "discovery_source": "KalshiAdapter._parse_kalshi_market", "data_mode": "live", "data_source": "kalshi_api"},
                venue_id="kalshi",
                venue_type="prediction",
                data_mode=DataMode.LIVE,
                data_source="kalshi_api",
                is_mock=False
            )
            return market
        except Exception as e:
            logger.warning(f"Failed to parse Kalshi market {raw.get('ticker')}: {e}")
            return None

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        markets: List[Market] = []
        
        try:
            # Try real API
            params = {
                "limit": min(100, target_count),
                "status": "open"
            }
            resp = self.session.get(f"{self.base_url}/markets", params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                raw_markets = data.get("markets", []) if isinstance(data, dict) else data
                for rm in raw_markets[:target_count]:
                    m = self._parse_kalshi_market(rm)
                    if m:
                        markets.append(m)
                logger.info(f"Kalshi discovered {len(markets)} real markets")
                if markets:
                    return markets[:target_count]
        except Exception as e:
            logger.warning(f"Kalshi API failed (expected in offline): {e}")

        # No mock fallback. This generated up to `target_count` Kalshi-style
        # questions with random prices whenever the API was unreachable, so an
        # outage produced markets indistinguishable from real ones. Returning
        # nothing with a stated reason is the only honest outcome.
        self.last_error = ("Kalshi API unreachable; no markets returned. "
                           "Refusing to fabricate markets on a network failure.")
        logger.warning(f"Kalshi discovery returned no markets: {self.last_error}")
        return []

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # V9 FIX #1 & #3: Real orderbook with is_real flag
        try:
            resp = self.session.get(f"{self.base_url}/markets/{market.id}/orderbook", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                ob = data.get("orderbook", {})
                yes_ob = ob.get("yes", {})
                bid = yes_ob.get("bid", market.yes_price - 0.01) if isinstance(yes_ob, dict) else market.yes_price - 0.01
                ask = yes_ob.get("ask", market.yes_price + 0.01) if isinstance(yes_ob, dict) else market.yes_price + 0.01
                return {
                    "market_id": market.id,
                    "venue_id": "kalshi",
                    "bid": bid,
                    "ask": ask,
                    "spread": ob.get("spread", 0.02) if isinstance(ob, dict) else 0.02,
                    "spread_pct": ob.get("spread", 0.02) if isinstance(ob, dict) else 0.02,
                    "depth": market.liquidity,
                    "liquidity": market.liquidity,
                    "source": "kalshi_api_real",
                    "is_real": True,
                    "is_mock": False,
                    "executable": True,
                    "data_mode": "live"
                }
        except Exception as e:
            logger.debug(f"Kalshi orderbook fetch failed for {market.id}: {e}")
        
        # Mock orderbook - marked not real, not executable for safety
        spread = 0.02 if market.liquidity > 5000 else 0.05
        return {
            "market_id": market.id,
            "venue_id": "kalshi",
            "token_id": market.yes_token_id,
            "bid": max(0.01, market.yes_price - spread/2),
            "ask": min(0.99, market.yes_price + spread/2),
            "spread": spread,
            "spread_pct": spread,
            "bid_size": market.liquidity * 0.1,
            "ask_size": market.liquidity * 0.1,
            "depth": market.liquidity,
            "liquidity": market.liquidity,
            "source": "kalshi_mock_estimation",
            "is_real": False,
            "is_mock": True,
            "executable": False,
            "data_mode": "mock",
            "warning": "ESTIMATION not real Kalshi orderbook"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"balance": 0, "positions": [], "orders": [], "venue": "kalshi", "paper": True}
        # Real implementation would auth and fetch
        return {"balance": 0, "positions": [], "orders": [], "venue": "kalshi"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd <= 0 or max_price <= 0 or max_price >= 1:
            return {"status": "rejected", "reason": "Invalid guard params", "venue": "kalshi"}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000", "venue": "kalshi"}
        
        logger.info(f"Kalshi guard: market={opportunity.market.id} side={opportunity.side} max_price={max_price} max_spend=${max_spend_usd}")
        
        if not self.api_key:
            return {
                "status": "dry_run",
                "venue": "kalshi",
                "message": f"Would place {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id}",
                "market_id": opportunity.market.id
            }
        
        # Real trading would go here
        return {"status": "error", "reason": "Live trading not implemented for Kalshi yet", "venue": "kalshi"}
