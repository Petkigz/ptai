"""
Crypto Adapter - for crypto spot/perp statistical arbitrage, momentum, mean reversion
Supports venue × market × strategy evaluation

A crypto exchange is NOT a prediction market, and this adapter used to blur that
in the one place it mattered: `discover_markets` invented a probability of the
price rising (`prob_up = 0.5 + change% x 0.5`) and wrote it into `outcome_prices`
as though a venue had quoted it. Nothing quoted it. The market's real content is
the QUOTE - symbol, last price, 24h volume, the book - and that is now all this
record claims, with `probability_market: False` so the probability lane refuses
it for the right reason.

Where the venue is paper-traded is `execution/directional.py`: a spot position at
the venue's own prices, with a stop and target from the venue's own measured
volatility. This adapter contributes the measured inputs - `directional_quote()`
and `recent_candles()` - and never a probability.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode
from ..execution.directional import DirectionalQuote


class CryptoAdapter(MarketAdapter):
    """
    Crypto adapter for statistical strategies
    Not prediction market - but evaluates crypto as venue for momentum, mean reversion, arbitrage
    """
    def __init__(self, exchange: str = "binance", api_key: str = None, api_secret: str = None):
        super().__init__(venue_id=f"crypto_{exchange}", venue_type=VenueType.FINANCIAL)

        self.last_error: str = ""
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
            # Reads the exchange's public market data; never submits. The
            # capability must say what the code does, not what an API key in
            # settings implies.
            supports_trading=False,
            requires_credentials=False,
            supports_portfolio=True,
            supports_history=True,
            supports_browser_fallback=False,
            fee_taker_pct=0.001,  # 0.1% typical crypto taker
            fee_maker_pct=0.0005,
            min_order_usd=10.0,
            # Quotes a price, not a probability: paper-traded in the directional
            # lane, refused by the probability lane.
            quotes_prices_not_probabilities=True,
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
                    # NO INVENTED PROBABILITY. The UP token's price is the
                    # quote itself (this is a price market), the outcome list
                    # carries the symbol rather than YES/NO, and the momentum
                    # heuristic that used to be written into `outcome_prices` is
                    # gone - a strategy may still form a view, but nothing here
                    # pretends the venue priced one.
                    m = Market(
                        id=f"CRYPTO-{symbol}",
                        source=MarketSource.PREDICTIT,
                        question=f"{symbol} on {self.exchange}: price higher or lower in 24h?",
                        description=f"{symbol} last {last_price:,.8g}, 24h change {price_change_pct:+.2f}%, 24h quote volume {volume_24h:,.0f}",
                        outcomes=[symbol],
                        outcome_prices=[last_price],
                        tokens=[Token(token_id=symbol, outcome=symbol, price=last_price)],
                        volume=volume_24h,
                        volume_24h=volume_24h,
                        liquidity=volume_24h * 0.1,
                        active=True,
                        closed=False,
                        slug=symbol.lower(),
                        event_slug=f"crypto-{self.exchange}",
                        market_type="directional",
                        raw={"venue": f"crypto_{self.exchange}", "symbol": symbol, "last_price": last_price, "change_pct": price_change_pct, "real": True, "data_mode": "live", "data_source": "binance_api", "probability_market": False, "quote_scale": "currency", "market_kind": "directional"},
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

        # No mock fallback. This fabricated markets for every symbol in the
        # pair list whenever the exchange was unreachable, so a network failure
        # became invented prices.
        self.last_error = (f"{self.exchange} unreachable; no markets returned. "
                           f"Refusing to fabricate markets on a network failure.")
        logger.warning(f"Crypto {self.exchange} discovery returned no markets: {self.last_error}")
        return []

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

    # ------------------------------------------------------------------
    # the directional lane's measured inputs
    # ------------------------------------------------------------------
    def directional_quote(self, symbol, market: Market = None) -> DirectionalQuote:
        """
        The venue's own two-sided quote for this symbol, in its quoted currency.

        This is what a spot position is opened and closed at: the ask to buy, the
        bid to sell, both straight from the exchange. No quote, no trade - the
        returned object carries `is_real: False` and the reason instead.

        Accepts a `Market` as the first argument too, because the lane has one
        during discovery and only a symbol when it settles a position later.
        """
        if not isinstance(symbol, str):
            market, symbol = symbol, ""
        if not symbol and market is not None:
            symbol = str((market.raw or {}).get("symbol") or market.slug or "")
        symbol = str(symbol or "").upper()
        venue_id = f"crypto_{self.exchange}"
        quote = DirectionalQuote(venue_id=venue_id, symbol=symbol, bid=0.0, ask=0.0)
        if not symbol:
            quote.reason = "the market record carries no symbol"
            return quote
        try:
            resp = self.session.get(
                "https://api.binance.com/api/v3/ticker/bookTicker",
                params={"symbol": symbol}, timeout=5)
            if resp.status_code != 200:
                quote.reason = f"HTTP {resp.status_code} from the book ticker"
                return quote
            data = resp.json() or {}
            bid = float(data.get("bidPrice") or 0)
            ask = float(data.get("askPrice") or 0)
        except Exception as e:  # noqa: BLE001
            quote.reason = f"{type(e).__name__}: {e}"
            return quote
        if bid <= 0 or ask <= bid:
            quote.reason = f"the venue quoted bid {bid} / ask {ask}"
            return quote
        quote.bid, quote.ask = bid, ask
        raw = (market.raw or {}) if market is not None else {}
        quote.last = float(raw.get("last_price") or 0) or None
        quote.change_pct = raw.get("change_pct")
        quote.volume_24h = getattr(market, "volume_24h", None)
        quote.source = "binance_book_ticker"
        quote.is_real = True
        return quote

    def recent_candles(self, symbol: str, hours: int = 48,
                       interval: str = "1h") -> List[Dict[str, Any]]:
        """
        The venue's own candles, newest last. Empty means unmeasurable - and the
        directional lane refuses to size a trade on an unmeasured volatility.
        """
        out: List[Dict[str, Any]] = []
        try:
            resp = self.session.get("https://api.binance.com/api/v3/klines",
                                    params={"symbol": str(symbol).upper(),
                                            "interval": interval,
                                            "limit": max(2, min(1000, int(hours)))},
                                    timeout=5)
            if resp.status_code != 200:
                self.last_error = f"HTTP {resp.status_code} from klines"
                return out
            for row in (resp.json() or []):
                if not isinstance(row, (list, tuple)) or len(row) < 7:
                    continue
                out.append({"open_time": int(row[0]), "open": float(row[1]),
                            "high": float(row[2]), "low": float(row[3]),
                            "close": float(row[4]), "volume": float(row[5])})
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            logger.debug(f"Crypto {self.exchange} candles failed for {symbol}: {e}")
        return out

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
