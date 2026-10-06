
"""
WhiteBIT Adapter - margin (10x) and futures (100x) identical API endpoints only pair name differs
Minimum order amounts low, API HMAC-SHA512, no testnet must test with min orders low leverage
Real venue where $50 bankroll can execute
API: https://whitebit.com/api
Docs: https://docs.whitebit.com/

An exchange quotes PRICES. This adapter used to write an invented probability of
the price rising (`prob_up = 0.5 + change x 2`) into every market record, which
is a forecast nobody made dressed as a quote - and the operator's log carried
the consequence: `whitebit-BCH_TRY ... spread 6300.0%`. The record now carries
the quote and nothing else, `probability_market: False` keeps the probability
lane away from it, and the venue's real inputs (`directional_quote()`,
`recent_candles()`) feed the directional lane instead.
"""
from typing import List, Dict, Any
from loguru import logger
import hmac
import hashlib
import time

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, MarketSource, Token, DataMode
from ..execution.directional import DirectionalQuote

class WhiteBITAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(venue_id="whitebit", venue_type=VenueType.FINANCIAL)

        self.last_error: str = ""
        self.api_key = api_key
        self.api_secret = api_secret
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            # Public ticker only: place_order returns a dry-run stub that never
            # signs a request, so it cannot hold a real position.
            supports_trading=False,
            requires_credentials=False,
            supports_portfolio=True,
            supports_history=True,
            fee_taker_pct=0.001,  # 0.1% taker
            fee_maker_pct=0.0005,
            min_order_usd=1.0,
            # Quotes a price, not a probability: paper-traded in the directional
            # lane, refused by the probability lane.
            quotes_prices_not_probabilities=True,
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
                        change = float(ticker.get("change", 0))
                        # NO INVENTED PROBABILITY, for the same reason as the
                        # Binance adapter: the venue quoted a price, and the
                        # market record now says only that.
                        markets.append(Market(
                            id=f"whitebit-{symbol}",
                            source=MarketSource.POLYMARKET,
                            question=f"{symbol} on WhiteBIT: price higher or lower in 24h?",
                            description=f"{symbol} last {last:,.8g}, 24h change {change:+.2f}%, 24h quote volume {vol:,.0f}",
                            outcomes=[symbol],
                            outcome_prices=[last],
                            tokens=[Token(token_id=symbol, outcome=symbol, price=last)],
                            volume=vol,
                            volume_24h=vol,
                            liquidity=vol*0.1,
                            active=True,
                            closed=False,
                            event_slug=symbol,
                            market_type="directional",
                            raw={"venue": "whitebit", "symbol": symbol, "last_price": last, "change_pct": change, "type": "spot", "category": "crypto", "data_mode": "live", "data_source": "whitebit_api", "probability_market": False, "quote_scale": "currency", "market_kind": "directional", "is_mock": False},
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

    def _not_a_probability_book(self, market: Market, symbol: str,
                                bid: Any = None, ask: Any = None) -> Dict[str, Any]:
        """
        The refusal for a quote that is a price, not a probability.

        The operator's log carried `whitebit-BCH_TRY ... spread 6300.0%` and
        `whitebit-ADA_PERP ... pay 0.748` against a 'fair 0.100', because a
        crypto order book (TRY, or a perpetual's price in dollars) was handed to
        the probability pricing stack. This venue is an exchange: what it quotes
        is the price of a coin, and no Yes share exists to buy at that price.
        The book is REAL - it is simply not the kind of book the question needs.
        """
        return {
            "market_id": market.id,
            "venue_id": "whitebit",
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "source": "not_a_probability_book",
            "scale": "currency",
            "is_real": True,
            "validated": False,
            "executable": False,
            "is_mock": False,
            "warning": (
                f"WhiteBIT is a crypto exchange: it quotes the price of a coin "
                f"(bid {bid} / ask {ask}), not a probability in 0-1, so there is "
                f"no Yes share to buy and no edge is computed from this book."),
        }

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        # V9 FIX #1 & #3: Real vs mock orderbook with is_real flag
        symbol = market.raw.get("symbol", market.event_slug)
        # ASKED FIRST, BEFORE ANY FETCH: a market whose price is not a
        # probability gets the refusal whether or not the venue answers, so a
        # 6300% spread can never be computed from a currency quote again.
        if not getattr(market, "is_probability_market", True):
            return self._not_a_probability_book(market, symbol)
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
                    # A book whose quotes are not in 0-1 is not a probability
                    # book, whatever the market says.
                    if max(abs(best_bid), abs(best_ask)) > 1.0:
                        return self._not_a_probability_book(
                            market, symbol, best_bid, best_ask)
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

    # ------------------------------------------------------------------
    # the directional lane's measured inputs
    # ------------------------------------------------------------------
    def directional_quote(self, symbol, market: Market = None) -> DirectionalQuote:
        """
        WhiteBIT's own two-sided quote for this market, from its public book.

        Read through the same endpoint the orderbook uses, so the lane trades on
        exactly the prices the venue is showing. No quote, no trade. Accepts a
        `Market` as the first argument too: the lane holds one during discovery
        and only a symbol when it settles a position in a later cycle.
        """
        if not isinstance(symbol, str):
            market, symbol = symbol, ""
        if not symbol and market is not None:
            symbol = str((market.raw or {}).get("symbol") or market.event_slug or "")
        symbol = str(symbol or "").upper()
        quote = DirectionalQuote(venue_id="whitebit", symbol=symbol, bid=0.0, ask=0.0)
        if not symbol:
            quote.reason = "the market record carries no symbol"
            return quote
        try:
            import requests
            resp = requests.get(f"{self.base_url}/public/orderbook/{symbol}",
                                timeout=5)
            if resp.status_code != 200:
                quote.reason = f"HTTP {resp.status_code} from the public book"
                return quote
            data = resp.json() or {}
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            if not bids or not asks:
                quote.reason = "the venue returned a one-sided book"
                return quote
            bid = float(bids[0][0])
            ask = float(asks[0][0])
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
        quote.source = "whitebit_public_orderbook"
        quote.is_real = True
        return quote

    def recent_candles(self, symbol: str, hours: int = 48,
                       interval: str = "1h") -> List[Dict[str, Any]]:
        """
        The venue's own candles. WhiteBIT returns [time, open, close, high, low,
        volume, ...] per row; the parse is defensive because a shape this lane
        does not recognise must produce NO volatility (and therefore no trade),
        never a plausible-looking one.
        """
        out: List[Dict[str, Any]] = []
        try:
            import requests
            resp = requests.get(f"{self.base_url}/public/kline",
                                params={"market": symbol, "interval": interval,
                                        "limit": max(2, min(1000, int(hours)))},
                                timeout=5)
            if resp.status_code != 200:
                self.last_error = f"HTTP {resp.status_code} from kline"
                return out
            payload = resp.json() or {}
            rows = payload.get("result") if isinstance(payload, dict) else payload
            for row in (rows or []):
                if not isinstance(row, (list, tuple)) or len(row) < 5:
                    continue
                out.append({"open_time": int(row[0]), "open": float(row[1]),
                            "close": float(row[2]), "high": float(row[3]),
                            "low": float(row[4]),
                            "volume": float(row[5]) if len(row) > 5 else 0.0})
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            logger.debug(f"WhiteBIT candles failed for {symbol}: {e}")
        return out

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
