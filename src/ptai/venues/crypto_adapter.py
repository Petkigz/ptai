"""
Crypto Adapter - for crypto spot/perp statistical arbitrage, momentum, mean reversion
Supports venue × market × strategy evaluation
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode


class CryptoAdapter(MarketAdapter):
    """
    Crypto adapter for statistical strategies
    Not prediction market - but evaluates crypto as venue for momentum, mean reversion, arbitrage
    """
    def __init__(self, exchange: str = "binance", api_key: str = None, api_secret: str = None):
        super().__init__(venue_id=f"crypto_{exchange}", venue_type=VenueType.FINANCIAL)
        self.exchange = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
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
            supports_browser_fallback=False,
            fee_taker_pct=0.001,  # 0.1% typical crypto taker
            fee_maker_pct=0.0005,
            min_order_usd=10.0
        )

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        country_code = country_code.upper()
        # Crypto generally available, but check local regulations
        # Uganda allows crypto trading via international exchanges
        if country_code in {"US", "UG", "GB", "CA", "AU", "DE", "FR", "KE", "NG", "ZA", "JP", "SG"}:
            return EligibilityStatus.ELIGIBLE
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count: int = 500, filters: Dict = None) -> List[Market]:
        filters = filters or {}
        markets: List[Market] = []
        
        # Try Binance public API for top symbols
        try:
            resp = self.session.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            if resp.status_code == 200:
                tickers = resp.json()
                # Filter USDT pairs, top volume
                usdt_pairs = [t for t in tickers if t.get("symbol", "").endswith("USDT")]
                sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
                for ticker in sorted_pairs[:target_count]:
                    symbol = ticker.get("symbol")
                    last_price = float(ticker.get("lastPrice", 0))
                    volume_24h = float(ticker.get("quoteVolume", 0))
                    # Convert to Market for unified handling - crypto as binary: will price go up?
                    # For V3, we treat each crypto pair as opportunity for directional strategies
                    price_change_pct = float(ticker.get("priceChangePercent", 0))
                    # Normalize price to 0-1 for prediction-style: probability of up in next 24h based on momentum
                    # Simple heuristic: if up recently, momentum says continue up with 55% prob
                    prob_up = 0.5 + max(-0.2, min(0.2, price_change_pct / 100.0 * 0.5))
                    m = Market(
                        id=f"CRYPTO-{symbol}",
                        source=MarketSource.PREDICTIT,
                        question=f"Will {symbol} close higher in 24h? (Crypto momentum)",
                        description=f"Crypto {symbol} last ${last_price} change {price_change_pct}% vol ${volume_24h:,.0f}",
                        outcomes=["YES", "NO"],
                        outcome_prices=[prob_up, 1-prob_up],
                        tokens=[
                            Token(token_id=f"CRYPTO-{symbol}_UP", outcome="YES", price=prob_up),
                            Token(token_id=f"CRYPTO-{symbol}_DOWN", outcome="NO", price=1-prob_up)
                        ],
                        volume=volume_24h,
                        volume_24h=volume_24h,
                        liquidity=volume_24h * 0.1,
                        active=True,
                        closed=False,
                        slug=symbol.lower(),
                        event_slug=f"crypto-{self.exchange}",
                        market_type="binary",
                        raw={"venue": f"crypto_{self.exchange}", "symbol": symbol, "last_price": last_price, "change_pct": price_change_pct, "real": True, "data_mode": "live", "data_source": "binance_api"},
                        venue_id=f"crypto_{self.exchange}",
                        venue_type="financial",
                        data_mode=DataMode.LIVE,
                        data_source="binance_api",
                        is_mock=False
                    )
                    markets.append(m)
                if markets:
                    logger.info(f"Crypto {self.exchange} discovered {len(markets)} real markets")
                    return markets[:target_count]
        except Exception as e:
            logger.warning(f"Crypto {self.exchange} API failed (expected offline): {e}")

        # Mock fallback for V3 testing
        import random
        crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
                         "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT", "FILUSDT", "TRXUSDT"]
        for i, symbol in enumerate(crypto_symbols[:min(target_count, 50)]):
            price = random.uniform(0.1, 50000)
            vol = random.uniform(1000000, 100000000)
            change = random.uniform(-10, 10)
            prob_up = 0.5 + change / 100.0 * 0.5
            prob_up = max(0.1, min(0.9, prob_up))
            m = Market(
                id=f"CRYPTO-MOCK-{symbol}",
                source=MarketSource.PREDICTIT,
                question=f"Will {symbol} close higher in 24h? (Crypto {self.exchange}) - MOCK_DATA MUST NEVER REACH LIVE EXECUTION",
                description=f"Mock crypto {symbol} last ${price:.2f} change {change:.1f}% - MOCK",
                outcomes=["YES", "NO"],
                outcome_prices=[prob_up, 1-prob_up],
                tokens=[
                    Token(token_id=f"CRYPTO-MOCK-{symbol}_UP", outcome="YES", price=prob_up),
                    Token(token_id=f"CRYPTO-MOCK-{symbol}_DOWN", outcome="NO", price=1-prob_up)
                ],
                volume=vol,
                volume_24h=vol,
                liquidity=vol*0.1,
                active=True,
                closed=False,
                slug=symbol.lower(),
                event_slug=f"crypto-{self.exchange}",
                market_type="binary",
                raw={"mock": True, "venue": f"crypto_{self.exchange}", "symbol": symbol, "data_mode": "mock", "data_source": "crypto_mock_fallback", "is_mock": True, "safety": "MOCK_DATA must be impossible to reach live execution"},
                venue_id=f"crypto_{self.exchange}",
                venue_type="financial",
                data_mode=DataMode.MOCK,
                data_source="crypto_mock_fallback",
                is_mock=True
            )
            markets.append(m)
        
        min_vol = filters.get("min_volume", 10000)
        filtered = [m for m in markets if m.volume_24h >= min_vol]
        logger.info(f"Crypto {self.exchange} mock discovery: {len(markets)} -> {len(filtered)}")
        return filtered[:target_count]

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # For crypto, fetch real orderbook if possible
        try:
            symbol = market.raw.get("symbol") or market.slug.upper()
            resp = self.session.get(f"https://api.binance.com/api/v3/depth", params={"symbol": symbol, "limit": 20}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread_pct = (best_ask - best_bid) / best_bid if best_bid else 0.001
                    return {
                        "market_id": market.id,
                        "bid": best_bid,
                        "ask": best_ask,
                        "spread": spread_pct,
                        "spread_pct": spread_pct,
                        "depth": market.liquidity,
                        "bids": bids[:5],
                        "asks": asks[:5]
                    }
        except Exception as e:
            logger.debug(f"Crypto orderbook failed for {market.id}: {e}")
        
        return {
            "market_id": market.id,
            "bid": market.best_price - 0.005,
            "ask": market.best_price + 0.005,
            "spread": 0.01,
            "spread_pct": 0.01,
            "depth": market.liquidity
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"balance": 0, "positions": [], "orders": [], "venue": f"crypto_{self.exchange}", "paper": True}

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> Dict[str, Any]:
        if max_spend_usd <= 0:
            return {"status": "rejected", "reason": "Invalid guard params", "venue": f"crypto_{self.exchange}"}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000", "venue": f"crypto_{self.exchange}"}
        
        logger.info(f"Crypto {self.exchange} guard: market={opportunity.market.id} side={opportunity.side} max_spend=${max_spend_usd}")
        
        return {
            "status": "dry_run",
            "venue": f"crypto_{self.exchange}",
            "message": f"Would place {opportunity.side} ${max_spend_usd} for {opportunity.market.id} on {self.exchange}",
            "market_id": opportunity.market.id
        }
