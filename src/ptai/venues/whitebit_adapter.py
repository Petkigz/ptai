
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
from ..markets.base import Market, MarketSource, Token

class WhiteBITAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(venue_id="whitebit", venue_type=VenueType.FINANCIAL)
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
                            raw={"venue": "whitebit", "symbol": symbol, "last_price": last, "type": "spot", "category": "crypto"}
                        ))
                    except:
                        continue
                if markets:
                    logger.info(f"WhiteBIT discovered {len(markets)} markets via public API")
                    return markets[:target_count]
        except Exception as e:
            logger.debug(f"WhiteBIT public API failed: {e}")

        # Mock fallback
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT", "DOGE_USDT"]
        markets = []
        for i, sym in enumerate(symbols[:target_count]):
            prob = 0.5 + (i*0.02 - 0.06)
            markets.append(Market(
                id=f"whitebit-{sym}",
                source=MarketSource.POLYMARKET,
                question=f"Will {sym} close higher? (WhiteBIT margin/futures)",
                outcomes=["YES", "NO"],
                outcome_prices=[prob, 1-prob],
                tokens=[Token(token_id=sym, outcome="YES", price=prob)],
                volume=100000,
                volume_24h=50000,
                liquidity=20000,
                active=True,
                closed=False,
                event_slug=sym,
                raw={"venue": "whitebit", "symbol": sym, "type": "futures" if "PERP" in sym else "spot", "category": "crypto", "leverage": "10x margin 100x futures"}
            ))
        logger.info(f"WhiteBIT mock discovered {len(markets)} markets")
        return markets

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        symbol = market.raw.get("symbol", market.event_slug)
        return {
            "market_id": market.id,
            "symbol": symbol,
            "bid": market.best_price - 0.001,
            "ask": market.best_price + 0.001,
            "spread": 0.002,
            "bid_size": 5000,
            "ask_size": 5000,
            "depth": market.liquidity,
            "venue": "whitebit",
            "auth": "HMAC-SHA512",
            "note": "No testnet - test with min orders low leverage"
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
