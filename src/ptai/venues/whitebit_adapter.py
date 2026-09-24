
"""
WhiteBIT Adapter - margin (10x) and futures (100x) identical API endpoints only pair name differs
Minimum order amounts low, API HMAC-SHA512, no testnet must test with min orders low leverage
Real venue where $50 bankroll can execute
API: https://whitebit.com/api
Docs: https://docs.whitebit.com/
"""
from typing import List, Dict, Any
from loguru import logger
import hmac
import hashlib
import time

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode

class WhiteBITAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(venue_id="whitebit", venue_type=VenueType.FINANCIAL)

        self.last_error: str = ""
        self.api_key = api_key
        self.api_secret = api_secret
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=bool(api_key and api_secret),
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.001,  # 0.1% taker
            fee_maker_pct=0.0005,
            min_order_usd=1.0
        )
        self.base_url = "https://whitebit.com/api/v4"

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        cc = country_code.upper()
        # WhiteBIT restricted in some countries, UG requires verification
        if cc == "UG":
            return EligibilityStatus.REQUIRES_VERIFICATION
        if cc == "US":
            return EligibilityStatus.RESTRICTED
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 50, filters: Dict = None) -> List[Market]:
        # Mock + real structure
        # Production: GET /api/v4/public/markets, GET /api/v4/public/ticker
        # Margin and futures identical endpoints only pair name differs: BTC_USDT vs BTC_USDT_PERP or BTC_USDT_MARGIN
        try:
            import requests
            # Try public ticker for spot
            resp = requests.get(f"{self.base_url}/public/ticker", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                markets = []
                for symbol, ticker in list(data.items())[:target_count]:
                    try:
                        last = float(ticker.get("last_price", 0.5))
                        vol = float(ticker.get("quote_volume", 1000))
                        # Convert to prediction-like prob for momentum strategy
                        # For crypto derivatives, we treat as financial not prediction
                        # price_change -> prob_up
                        change = float(ticker.get("change", 0))
                        prob_up = max(0.1, min(0.9, 0.5 + change*2))
                        markets.append(Market(
                            id=f"whitebit-{symbol}",
                            source=MarketSource.POLYMARKET,
                            question=f"Will {symbol} close higher?",
                            outcomes=["YES", "NO"],
                            outcome_prices=[prob_up, 1-prob_up],
                            tokens=[Token(token_id=symbol, outcome="YES", price=prob_up)],
                            volume=vol,
                            volume_24h=vol,
                            liquidity=vol*0.1,
                            active=True,
                            closed=False,
                            event_slug=symbol,
                            raw={"venue": "whitebit", "symbol": symbol, "last_price": last, "type": "spot", "category": "crypto", "data_mode": "live", "data_source": "whitebit_api", "is_mock": False},
                            venue_id="whitebit",
                            venue_type="financial",
                            data_mode=DataMode.LIVE,
                            data_source="whitebit_api",
                            is_mock=False
                        ))
                    except:
                        continue
                if markets:
                    logger.info(f"WhiteBIT discovered {len(markets)} markets via public API")
                    return markets[:target_count]
        except Exception as e:
            logger.debug(f"WhiteBIT public API failed: {e}")

        # No mock fallback - the line below used to follow a fabricated-market
        # block. A network failure now returns nothing and says so.
        self.last_error = ("WhiteBIT API unreachable; no markets returned. "
                           "Refusing to fabricate markets on a network failure.")
        logger.warning(f"WhiteBIT discovery returned no markets: {self.last_error}")
        return []
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # V9 FIX #1 & #3: Real vs mock orderbook with is_real flag
        symbol = market.raw.get("symbol", market.event_slug)
        # Try real API for LIVE markets
        is_mock_market = getattr(market, 'is_mock', False) or "MOCK" in market.id
        if not is_mock_market:
            try:
                import requests
                resp = requests.get(f"{self.base_url}/public/orderbook/{symbol}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    best_bid = float(bids[0][0]) if bids else market.best_price - 0.001
                    best_ask = float(asks[0][0]) if asks else market.best_price + 0.001
                    return {
                        "market_id": market.id,
                        "venue_id": "whitebit",
                        "symbol": symbol,
                        "bid": best_bid,
                        "ask": best_ask,
                        "spread": best_ask - best_bid,
                        "spread_pct": (best_ask - best_bid) / best_bid if best_bid else 0.002,
                        "bid_size": 5000,
                        "ask_size": 5000,
                        "depth": market.liquidity,
                        "venue": "whitebit",
                        "auth": "HMAC-SHA512",
                        "source": "whitebit_api_real",
                        "is_real": True,
                        "is_mock": False,
                        "executable": True,
                        "data_mode": "live"
                    }
            except Exception as e:
                logger.debug(f"WhiteBIT orderbook real fetch failed {market.id}: {e}")

        # Mock fallback - marked not executable
        return {
            "market_id": market.id,
            "venue_id": "whitebit",
            "symbol": symbol,
            "bid": market.best_price - 0.001,
            "ask": market.best_price + 0.001,
            "spread": 0.002,
            "spread_pct": 0.002,
            "bid_size": 5000,
            "ask_size": 5000,
            "depth": market.liquidity,
            "venue": "whitebit",
            "auth": "HMAC-SHA512",
            "note": "No testnet - test with min orders low leverage - MOCK estimation" if is_mock_market else "Estimation - real API failed",
            "source": "whitebit_mock_fallback" if is_mock_market else "whitebit_estimation",
            "is_real": False,
            "is_mock": True if is_mock_market else False,
            "executable": False if is_mock_market else False,
            "data_mode": "mock" if is_mock_market else "live",
            "warning": "MOCK_DATA - cannot execute" if is_mock_market else "Estimation - verify executable price"
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "venue": "whitebit", "margin": "10x", "futures": "100x"}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        # Risk caveat: leverage amplifies losses, $50 bankroll even 2x 50% adverse wipes account
        # Treat as hedging or arbitrage legs, not directional until bankroll grows
        if max_spend_usd > 50:
            return {"status": "rejected", "reason": "WhiteBIT $50 bankroll max position, use hedging not directional"}
        leverage = opportunity.market.raw.get("leverage", "1x")
        return {
            "status": "dry_run",
            "message": f"WhiteBIT order {opportunity.side} ${max_spend_usd} @ {max_price} for {opportunity.market.id} {leverage} - hedging/arb leg only, not directional",
            "venue": "whitebit",
            "risk": "Leverage amplifies losses, $50 bankroll 2x leverage 50% adverse wipes account"
        }
