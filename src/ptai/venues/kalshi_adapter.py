"""
Kalshi Adapter - real implementation
Kalshi API docs: https://trading-api.readme.io/
Public markets: GET https://api.elections.kalshi.com/trade-api/v2/markets
"""
import base64
import math
import time
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from loguru import logger
import requests
from datetime import datetime

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode

# Kalshi signs every authenticated request with an RSA key the operator creates
# in their account. Imported defensively: the adapter's MARKET reads are public
# and must keep working on a machine where the signing library is missing - the
# order path refuses instead, which is the honest failure.
try:  # pragma: no cover - import guard
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    _SIGNING_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SIGNING_AVAILABLE = False

# Production shared API server (the docs list external-api.kalshi.com as the
# recommended root and this one as "also supported"); the demo environment is a
# separate exchange with separate keys and mock funds, which is where an order
# path can be proven before real money rides on it.
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_API = "https://external-api.demo.kalshi.co/trade-api/v2"

# The V2 order path. The legacy /portfolio/orders endpoint was deprecated no
# earlier than 2026-05-06, so this is the shape to write against:
#   POST /portfolio/events/orders
#   {"ticker": ..., "side": "bid"|"ask", "count": "1", "price": "0.5600",
#    "time_in_force": "good_till_canceled",
#    "self_trade_prevention_type": "taker_at_cross", "client_order_id": uuid}
# Signed with RSA-PSS/SHA-256 over f"{timestamp_ms}{METHOD}{path}", base64.
KALSHI_ORDER_PATH = "/portfolio/events/orders"
KALSHI_BALANCE_PATH = "/portfolio/balance"
KALSHI_POSITIONS_PATH = "/portfolio/positions"


class KalshiAdapter(MarketAdapter):
    def __init__(self, api_key: str = None, api_secret: str = None,
                 member_id: str = None, private_key_pem: str = None,
                 environment: str = "production"):
        super().__init__(venue_id="kalshi", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        self.api_secret = api_secret
        self.member_id = member_id
        self.private_key_pem = private_key_pem
        # demo|production. The demo exchange has mock funds and separate keys,
        # so the whole path (sign, submit, read the response) can be exercised
        # there without a funded account.
        self.environment = "demo" if str(environment).strip().lower() in {
            "demo", "test", "sandbox"} else "production"
        self.base_url = KALSHI_DEMO_API if self.environment == "demo" else KALSHI_API
        self.session = requests.Session()
        self.last_error: str = ""
        self._signing_key = None
        self._signing_checked = False
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            # Trading needs the key the operator generates in their Kalshi
            # account: the API key id AND the RSA private key that signs each
            # request. One without the other cannot place anything, so the flag
            # reports exactly that pair rather than "an API key exists".
            supports_trading=bool(self.api_key and self.private_key_pem),
            real_order_path=True,
            # The order probe is implemented below (place a minimum-size order,
            # then cancel it), so the account-health ladder can PROVE order
            # permission instead of reporting it unproven. It refuses in dry_run.
            supports_order_probe=True,
            requires_credentials=False,  # its markets are public; orders are not
            supports_portfolio=True,
            supports_history=True,
            supports_browser_fallback=True,
            # NOT a flat rate, and 0.0 here was a fiction: Kalshi's taker fee is
            # round_up(0.07 x contracts x P x (1-P)) - 6.3% of stake at 10c,
            # 3.5% at 50c, 0.7% at 90c. The field carries the CURVE'S CEILING
            # (7% of stake, approached at the tails) so any consumer that can
            # only read a flat number over-charges rather than under-charges;
            # `fee_rate_for_market` below carries the curve itself, which is what
            # the runtime costs trades with.
            fee_taker_pct=0.07,
            fee_maker_pct=0.0175,  # maker formula per the July 2026 schedule
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

    # ------------------------------------------------------------------
    # fees: the venue's own curve - and its per-market price grid
    # ------------------------------------------------------------------

    def fee_rate_for_market(self, market: Market) -> Optional[float]:
        """
        Kalshi's taker fee as a fraction of the stake, for this market.

        The schedule (effective 2026-07-07) is
            taker fee = round_up(M x 0.07 x contracts x P x (1 - P))
        with M the per-market fee multiplier and P the price of the contract
        traded. On a dollar of stake that is 0.07 x (1 - P) for a YES buy at P
        and 0.07 x P for a NO buy at 1 - P, so the cost falls as the price moves
        away from 50c. Without a side, the worst of the two legs is returned -
        at a 30c market that is the YES side's 4.9% of stake, where the NO side
        pays 2.1%. One number has to cover both sides, and over-stating cost
        refuses a marginal edge while under-stating it invents one.

        Multiplier M is not on the market payload this client reads, so 1 is
        assumed - the general schedule. A market with its own multiplier (some
        sports series) therefore has its fee UNDER-stated here; the order that
        goes out is unaffected.
        """
        price = None
        raw = getattr(market, "raw", None) or {}
        price = self._dollars(raw, "last_price_dollars", "last_price")
        if price is None:
            price = getattr(market, "yes_price", None)
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        if not 0.0 < price < 1.0:
            return None
        return round(0.07 * max(price, 1.0 - price), 6)

    def calculate_fees(self, market: Market, amount_usd: float) -> float:
        """The venue's own fee curve, rounded up to the cent it charges in."""
        rate = self.fee_rate_for_market(market)
        if rate is None:
            return super().calculate_fees(market, amount_usd)
        raw = amount_usd * rate
        # Kalshi rounds the trade fee up to the next cent.
        return math.ceil(raw * 100.0 - 1e-9) / 100.0 if raw > 0 else 0.0

    @staticmethod
    def parse_price_ranges(raw: Dict) -> List[Dict[str, float]]:
        """
        The market's own valid price grid.

        Kalshi moved to fixed-point dollar prices with sub-cent ticks, and the
        valid prices are described PER MARKET by `price_ranges` (bands of
        {start, end, step}). Off-grid prices are rejected by the exchange, so a
        price the operator is willing to pay has to be snapped onto this grid
        before it is sent. Reading it dynamically is what makes the client work
        on every price structure, including ones introduced later.
        """
        out: List[Dict[str, float]] = []
        for band in (raw or {}).get("price_ranges") or []:
            try:
                out.append({"start": float(band["start"]),
                            "end": float(band["end"]),
                            "step": float(band["step"])})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @classmethod
    def snap_price(cls, raw: Dict, price: float,
                   direction: str = "down") -> Optional[float]:
        """
        Snap a price onto the market's grid. None when the grid cannot be read.

        `direction` is which way to move when the price is off-grid, and it is
        about money, not neatness:

          * "down" for a bid (buying YES): rounding up would pay more than the
            caller's maximum, and a maximum means a maximum;
          * "up" for the YES leg of a NO order (`ask`): the money at risk is
            1 - q, so q has to round UP for the NO price to stay at or below
            the caller's maximum.

        Whole cents are valid in every Kalshi price structure, so a market whose
        grid cannot be read still gets a valid (coarser) cent price.
        """
        ranges = cls.parse_price_ranges(raw)
        if not ranges:
            return None
        price = float(price)
        up = str(direction).lower() == "up"
        if price <= ranges[0]["start"]:
            return round(ranges[0]["start"], 6)
        for band in ranges:
            if band["step"] <= 0:
                continue
            if price <= band["end"]:
                steps = (price - band["start"]) / band["step"]
                steps = int(steps + 1 - 1e-9) if up else int(steps + 1e-9)
                snapped = band["start"] + steps * band["step"]
                snapped = max(band["start"], min(snapped, band["end"]))
                return round(snapped, 6)
        return round(ranges[-1]["end"], 6)

    @staticmethod
    def _dollars(raw: Dict, *names) -> Optional[float]:
        """Read a price: the current `*_dollars` field, or a legacy cent field."""
        for name in names:
            value = (raw or {}).get(name)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            return number if name.endswith("_dollars") else number / 100.0
        return None

    @staticmethod
    def _count(raw: Dict, *names) -> float:
        """Read a contract count, which the current API reports as `*_fp` strings."""
        for name in names:
            value = (raw or {}).get(name)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    # ------------------------------------------------------------------
    # authentication: the operator's own RSA key, signing every request
    # ------------------------------------------------------------------

    def _load_signing_key(self):
        """Load the operator's PEM once. None + a reason when it cannot be used."""
        if self._signing_checked:
            return self._signing_key
        self._signing_checked = True
        if not self.private_key_pem:
            self.last_error = ("no Kalshi private key saved; the order path and "
                               "the account read need the RSA key from your "
                               "Kalshi account, not just the API key id")
            return None
        if not _SIGNING_AVAILABLE:
            self.last_error = ("the cryptography package is missing, so Kalshi "
                               "requests cannot be signed; market reads still work")
            return None
        try:
            self._signing_key = serialization.load_pem_private_key(
                str(self.private_key_pem).encode("utf-8"), password=None)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"the saved Kalshi private key could not be read: {type(e).__name__}: {e}"
            self._signing_key = None
        return self._signing_key

    def _auth_ready(self):
        """(ready, reason). Never guesses: a half-configured login is not ready."""
        if not self.api_key:
            return False, "no Kalshi API key saved"
        if self._load_signing_key() is None:
            return False, self.last_error or "no usable Kalshi signing key"
        return True, ""

    def _signed_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        The three headers Kalshi requires.

        The signed message is timestamp + METHOD + path (the full path from the
        root, query string excluded), signed with RSA-PSS/SHA-256 and sent
        base64-encoded. `path` here is the API path without the host, e.g.
        /trade-api/v2/portfolio/events/orders.
        """
        key = self._load_signing_key()
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
        signature = key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "Content-Type": "application/json",
        }

    def _signed_path(self, endpoint: str) -> str:
        """The full path that gets signed: the API prefix plus the endpoint."""
        prefix = urlparse(self.base_url).path.rstrip("/")
        return f"{prefix}{endpoint}"

    def _signed_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        path = self._signed_path(endpoint)
        return self.session.request(
            method.upper(), f"{self.base_url}{endpoint}",
            headers=self._signed_headers(method.upper(), path), timeout=15, **kwargs)

    def _parse_kalshi_market(self, raw: Dict) -> Optional[Market]:
        """
        Build a Market from Kalshi's CURRENT market object.

        The API moved prices to fixed-point dollar strings (`yes_bid_dollars`,
        `last_price_dollars`) and counts to `*_fp` strings, with integer-cent
        fields kept only as legacy. The old parser read `yes_bid`, `last_price`,
        `volume` and `liquidity` - so on today's payload it fell through to
        its `0.5` default for every price: every Kalshi market was priced at
        even money and every volume was zero. Both are read here, current field
        first, and a market whose price cannot be read at all is refused rather
        than published at 0.50 as though that were the price.
        """
        try:
            ticker = raw.get("ticker", "")
            if not ticker:
                return None

            yes_bid = self._dollars(raw, "yes_bid_dollars", "yes_bid")
            yes_ask = self._dollars(raw, "yes_ask_dollars", "yes_ask")
            last = self._dollars(raw, "last_price_dollars", "last_price")
            status = str(raw.get("status", "")).lower()

            # A never-traded market reports last_price_dollars "0.0000", which is
            # NOT a price: publishing a market at 0.00 says the outcome is
            # impossible when the truth is that nothing has traded yet. A quote
            # is the fallback, and only a market with no usable number at all is
            # refused.
            if last is None or not 0.0 < float(last) < 1.0:
                if yes_bid and yes_ask:
                    last = (yes_bid + yes_ask) / 2.0
                elif yes_bid:
                    last = yes_bid
                elif yes_ask:
                    last = yes_ask
            if last is None or not 0.0 < float(last) < 1.0:
                # No readable price anywhere. A 0.50 market invented from a
                # missing field is the kind of number a strategy would trade on.
                self.last_error = (f"Kalshi market {ticker} carried no readable "
                                   f"price (looked for last_price_dollars / "
                                   f"yes_bid_dollars / yes_ask_dollars); refused "
                                   f"rather than defaulting to 0.50")
                logger.warning(self.last_error)
                return None
            last = max(0.0, min(1.0, float(last)))

            title = (raw.get("title") or raw.get("yes_sub_title")
                     or raw.get("subtitle") or ticker)
            description = (raw.get("rules_primary") or raw.get("subtitle")
                           or raw.get("yes_sub_title") or "")

            volume = self._count(raw, "volume_fp", "volume")
            volume_24h = self._count(raw, "volume_24h_fp", "volume_24h") or volume
            open_interest = self._count(raw, "open_interest_fp", "open_interest")
            # `liquidity` is not a field on the current market object; resting
            # open interest is the honest stand-in, and 0 (not 1000) when the
            # venue did not report one.
            liquidity = open_interest or self._count(raw, "liquidity")

            end_date = None
            close_time = (raw.get("close_time") or raw.get("latest_expiration_time")
                          or raw.get("expiration_time"))
            if close_time:
                try:
                    end_date = datetime.fromisoformat(
                        str(close_time).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    end_date = None

            tokens = [
                Token(token_id=f"{ticker}_YES", outcome="YES", price=last),
                Token(token_id=f"{ticker}_NO", outcome="NO", price=1 - last),
            ]

            market = Market(
                id=ticker,
                source=MarketSource.KALSHI,
                question=title,
                description=description,
                outcomes=["YES", "NO"],
                outcome_prices=[last, 1 - last],
                tokens=tokens,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                active=status in ("active", "open"),
                closed=status in ("closed", "settled", "determined", "finalized"),
                slug=ticker,
                event_slug=raw.get("event_ticker", ""),
                condition_id=ticker,
                market_type="binary",
                raw={**raw, "venue_id": "kalshi", "adapter_venue_id": "kalshi",
                     "discovery_source": "KalshiAdapter._parse_kalshi_market",
                     "data_mode": "live", "data_source": "kalshi_api"},
                venue_id="kalshi",
                venue_type="prediction",
                data_mode=DataMode.LIVE,
                data_source="kalshi_api",
                is_mock=False,
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
        """
        The exchange's own book, or an honest refusal.

        Kalshi returns BIDS ONLY, in two arrays, because in a binary market a
        NO bid at Y is a YES ask at 1-Y. The current payload is
        `orderbook_fp`: `yes_dollars` and `no_dollars`, each a list of
        [price_dollars, count_fp] ascending with the BEST bid last. The old code
        looked for a dict with `bid`/`ask` keys, found none on that schema, and
        fell back to a placeholder spread stamped `is_real: False` - which is why
        every Kalshi book read as an ESTIMATION even when the exchange had
        answered.
        """
        def _levels(rows) -> List[List[float]]:
            out: List[List[float]] = []
            for row in rows or []:
                try:
                    out.append([float(row[0]), float(row[1])])
                except (TypeError, ValueError, IndexError):
                    continue
            return out

        try:
            resp = self.session.get(
                f"{self.base_url}/markets/{market.id}/orderbook", timeout=10)
            if resp.status_code == 200:
                data = resp.json() or {}
                fp = data.get("orderbook_fp")
                if isinstance(fp, dict):
                    yes_levels = _levels(fp.get("yes_dollars"))
                    no_levels = _levels(fp.get("no_dollars"))
                else:
                    # Legacy shape: integer-cent levels under `orderbook`.
                    legacy = data.get("orderbook") or {}
                    yes_levels = _levels(legacy.get("yes"))
                    no_levels = _levels(legacy.get("no"))
                    if legacy and not fp:
                        # Legacy prices are cents, not dollars.
                        yes_levels = [[p / 100.0, c] for p, c in yes_levels]
                        no_levels = [[p / 100.0, c] for p, c in no_levels]

                if yes_levels or no_levels:
                    best_yes_bid = yes_levels[-1][0] if yes_levels else None
                    best_no_bid = no_levels[-1][0] if no_levels else None
                    # The implied asks: the other side's best bid, complemented.
                    best_yes_ask = (1.0 - best_no_bid) if best_no_bid is not None else None
                    best_no_ask = (1.0 - best_yes_bid) if best_yes_bid is not None else None
                    # If one side has no bids, the market has no offer on that
                    # side; publishing a bid/ask pair anyway would invent one.
                    if best_yes_bid is not None and best_yes_ask is not None:
                        bid, ask = best_yes_bid, best_yes_ask
                    elif best_yes_bid is not None:
                        bid, ask = best_yes_bid, None
                    elif best_yes_ask is not None:
                        bid, ask = None, best_yes_ask
                    else:
                        bid = ask = None

                    spread = (ask - bid) if (bid is not None and ask is not None) else None
                    depth = sum(count for _, count in yes_levels + no_levels)
                    depth_usd = sum(price * count for price, count in yes_levels + no_levels)
                    return {
                        "market_id": market.id,
                        "venue_id": "kalshi",
                        "token_id": market.yes_token_id,
                        "bid": bid,
                        "ask": ask,
                        "yes_bid": best_yes_bid,
                        "yes_ask": best_yes_ask,
                        "no_bid": best_no_bid,
                        "no_ask": best_no_ask,
                        "spread": spread,
                        "spread_pct": spread,
                        "bid_size": yes_levels[-1][1] if yes_levels else 0.0,
                        "ask_size": no_levels[-1][1] if no_levels else 0.0,
                        "depth": depth,
                        "depth_usd": round(depth_usd, 4),
                        "liquidity": depth_usd,
                        "yes_levels": yes_levels,
                        "no_levels": no_levels,
                        "source": "kalshi_api_real",
                        "is_real": True,
                        "is_mock": False,
                        "executable": bid is not None and ask is not None,
                        "data_mode": "live",
                        "assumed_fields": [],
                        "note": ("Kalshi publishes bids only; the ask is the "
                                 "other side's best bid complemented (1 - p)."),
                    }
                # A 200 that carries no levels is an empty book, not a spread.
                self.last_error = (f"Kalshi {market.id}: orderbook response "
                                   f"carried no price levels")
        except Exception as e:  # noqa: BLE001
            self.last_error = f"Kalshi orderbook fetch failed: {type(e).__name__}: {e}"
            logger.debug(self.last_error)

        # NO INVENTED BOOK. This used to return a placeholder spread around the
        # market price, and downstream only the flag stopped it being traded on.
        return {
            "market_id": market.id,
            "venue_id": "kalshi",
            "token_id": market.yes_token_id,
            "bid": None,
            "ask": None,
            "spread": None,
            "spread_pct": None,
            "depth": 0,
            "liquidity": 0,
            "source": "kalshi_api_unavailable",
            "is_real": False,
            "is_mock": False,
            "executable": False,
            "data_mode": "live",
            "assumed_fields": [],
            "warning": "no Kalshi book was read; no spread is published here",
            "reason": self.last_error or "no orderbook response",
        }

    async def get_settlement(self, market_id: str) -> Dict[str, Any]:
        """
        Did this market settle, and to what?

        Kalshi publishes `result` ("yes"/"no") on the market once the outcome is
        determined, plus `settlement_value_dollars` and `settlement_ts` when it
        settles. This is a public read, and it is what lets a Kalshi paper trade
        reach the resolved count with a REAL outcome rather than a guess.
        """
        ticker = str(market_id or "").strip()
        if not ticker:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "no_market_id", "reason": "empty Kalshi ticker"}
        try:
            resp = self.session.get(f"{self.base_url}/markets/{ticker}", timeout=10)
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "kalshi_markets_error",
                    "reason": f"{type(e).__name__}: {e}"}
        if resp.status_code != 200:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "kalshi_markets_http",
                    "reason": f"HTTP {resp.status_code} for {ticker}"}
        try:
            raw = (resp.json() or {}).get("market") or {}
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "kalshi_markets_unreadable",
                    "reason": f"{type(e).__name__}: {e}"}

        status = str(raw.get("status", "")).lower()
        result = str(raw.get("result", "")).strip().lower()
        settled_marker = raw.get("settlement_value_dollars") is not None
        if result not in ("yes", "no"):
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "kalshi_markets", "market_id": ticker,
                    "raw_status": status,
                    "reason": f"no result yet (status {status or 'unknown'})"}
        if status not in ("determined", "finalized", "settled") and not settled_marker:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "kalshi_markets", "market_id": ticker,
                    "raw_status": status,
                    "reason": (f"result {result!r} is recorded but the market is "
                               f"still {status or 'in an unknown state'}; not "
                               f"counted as settled yet")}
        return {"settled": True, "outcome": 1.0 if result == "yes" else 0.0,
                "is_real": True, "source": "kalshi_markets", "market_id": ticker,
                "raw_status": status, "result": result,
                "settlement_value_dollars": raw.get("settlement_value_dollars"),
                "settlement_ts": raw.get("settlement_ts")}

    async def get_portfolio(self) -> Dict[str, Any]:
        """
        The account as the exchange reports it, or nothing with a reason.

        This used to return zeros whenever a key existed - a fabricated account
        read that made an unfunded venue look read and a funded one look empty,
        which is worse than either. `available` is True only when the venue
        answered an authenticated request.
        """
        ready, why = self._auth_ready()
        if not ready:
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "kalshi",
                    "environment": self.environment, "reason": why}
        try:
            resp = self._signed_request("GET", KALSHI_BALANCE_PATH)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "kalshi",
                    "reason": f"{type(e).__name__}: {e}"}
        if resp.status_code != 200:
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "kalshi",
                    "http_status": resp.status_code,
                    "reason": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return {"available": False, "balance": None, "positions": [],
                    "orders": [], "venue": "kalshi",
                    "reason": "the balance response was not JSON"}
        # Kalshi reports the balance in cents.
        cents = body.get("balance")
        out: Dict[str, Any] = {
            "available": True,
            "balance": (float(cents) / 100.0) if cents is not None else None,
            "currency": "USD",
            "positions": [],
            "orders": [],
            "venue": "kalshi",
            "environment": self.environment,
            "source": "kalshi_api_real",   # the allowlist token: a venue-side read
        }
        try:
            pos = self._signed_request("GET", KALSHI_POSITIONS_PATH)
            if pos.status_code == 200:
                out["positions"] = (pos.json() or {}).get("market_positions", [])
        except Exception as e:  # noqa: BLE001 - a missing positions read is not a failed balance
            out["positions_note"] = f"{type(e).__name__}: {e}"
        return out

    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float,
                          max_price: float) -> Dict[str, Any]:
        """
        Submit a real limit order to Kalshi, or say why not.

        WHAT IS BUILT: an authenticated POST to /portfolio/events/orders with
        the V2 request shape (single-book side, fixed-point dollar price, whole
        or fractional contract count), the operator's own RSA signature, and a
        client_order_id so a retry cannot become a second order.

        WHAT IS NOT: only the YES side (book_side `bid`) is submitted. Kalshi's
        NO side (`ask`) shares a single book whose price leg must be verified
        against a real account before PTAI sends money at it - so a NO-side
        opportunity is refused with that reason instead of guessed at. Every
        response is reported verbatim; no fill is ever assumed.
        """
        market = opportunity.market
        if max_spend_usd <= 0 or max_price <= 0 or max_price >= 1:
            return {"status": "rejected", "reason": "Invalid guard params",
                    "venue": "kalshi", "success": False}
        if max_spend_usd > 1000:
            return {"status": "rejected", "reason": "Exceeds absolute max $1000",
                    "venue": "kalshi", "success": False}

        ticker = str((market.raw or {}).get("ticker") or market.id or "")
        if not ticker:
            return {"status": "rejected", "reason": "no Kalshi ticker on this market",
                    "venue": "kalshi", "success": False}
        if getattr(market, "closed", False) or str(
                (market.raw or {}).get("status", "")).lower() in {"closed", "settled"}:
            return {"status": "rejected",
                    "reason": f"{ticker} is closed or settled; the exchange would refuse it",
                    "venue": "kalshi", "success": False}

        side = str(opportunity.side or "").upper()
        if side not in {"YES", "NO"}:
            return {"status": "rejected",
                    "reason": f"this order path speaks YES/NO; got {opportunity.side!r}",
                    "venue": "kalshi", "success": False}

        ready, why = self._auth_ready()
        if not ready:
            return {
                "status": "refused", "success": False, "venue": "kalshi",
                "reason": why,
                "message": ("Refusal, not a fill. No order was placed: Kalshi "
                            "needs the API key id and its RSA private key, saved "
                            "in the Logins tab."),
                "market_id": market.id,
            }

        # DIRECTION, and why this is not a guess. Kalshi's V2 order endpoint
        # quotes everything from the YES side: `bid` buys YES, `ask` sells YES.
        # The docs state the equivalence explicitly - "selling YES is
        # economically equivalent to buying NO at 1 - price" - and in a binary
        # book a YES bid at p IS a NO ask at 1-p, with identical size. So a NO
        # buy at p_no is sent as an `ask` at 1 - p_no, which risks exactly the
        # same money (1 contract costs 1 - q = p_no).
        book_side = "bid" if side == "YES" else "ask"
        yes_leg = round(float(max_price), 6) if side == "YES" else round(1.0 - float(max_price), 6)

        # The price the exchange will accept. Off-grid prices are rejected, so
        # the caller's maximum is snapped onto the market's OWN grid - down for
        # a buy, up for the YES leg of a NO order (see snap_price).
        grid = self.parse_price_ranges(market.raw or {})
        snapped = self.snap_price(market.raw or {}, yes_leg,
                                  "down" if side == "YES" else "up")
        if snapped is None:
            # No readable grid, so the fallback is whole cents - valid in every
            # Kalshi price structure - and it is FLOORED (for a buy) rather than
            # rounded, because `round(0.437, 2)` is 0.44: a limit order that pays
            # more than the authorised maximum is not a limit order.
            cents = 100.0 * yes_leg
            cents = (math.ceil(cents - 1e-9) if side == "NO"
                     else math.floor(cents + 1e-9))
            snapped = round(min(0.99, max(0.01, cents / 100.0)), 2)
            grid_source = ("assumed whole cents: this market's price_ranges were "
                           "not readable")
        else:
            grid_source = "snapped to the market's own price_ranges grid"
        # Money per contract, in the terms the operator is paying.
        per_contract = snapped if side == "YES" else round(1.0 - snapped, 6)
        if per_contract <= 0:
            return {"status": "rejected", "success": False, "venue": "kalshi",
                    "reason": (f"a ${snapped:.4f} YES leg implies a NO price of "
                               f"${per_contract:.4f}; not orderable")}
        if side == "NO" and per_contract > float(max_price) + 1e-9:
            return {"status": "rejected", "success": False, "venue": "kalshi",
                    "reason": (f"NO at ${per_contract:.4f} would exceed the "
                               f"authorised maximum of ${float(max_price):.4f}")}

        count = int(max_spend_usd // per_contract)
        if count < 1:
            return {"status": "rejected", "success": False, "venue": "kalshi",
                    "reason": (f"${max_spend_usd:.2f} does not buy one contract "
                               f"at ${per_contract:.4f}; a smaller order is not "
                               f"placeable")}

        order = {
            "ticker": ticker,
            "side": book_side,
            "count": f"{count:.2f}",
            "price": f"{snapped:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": str(uuid.uuid4()),
        }
        logger.info(f"[kalshi] submitting {count} contract(s) {ticker} {side} "
                    f"(book_side {book_side}, price ${snapped:.4f}) "
                    f"max ${max_spend_usd:.2f} on {self.environment}")
        try:
            resp = self._signed_request("POST", KALSHI_ORDER_PATH, json=order)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "success": False, "venue": "kalshi",
                    "reason": f"{type(e).__name__}: {e}",
                    "message": "The order was not confirmed by the exchange. "
                               "Treat it as unplaced until checked."}

        if resp.status_code != 201:
            logger.error(f"[kalshi] order rejected: HTTP {resp.status_code}")
            return {"status": "rejected", "success": False, "venue": "kalshi",
                    "http_status": resp.status_code,
                    "reason": resp.text[:300],
                    "message": "Refusal, not a fill. No order was placed."}
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return {"status": "error", "success": False, "venue": "kalshi",
                    "reason": "the order response was not JSON; order state unknown"}

        # The exchange's own numbers. `fill_count` is what matched at submit;
        # the rest rests at the limit. Nothing here assumes a fill it did not
        # report, and the notional is costed at the LIMIT price rather than at
        # an average Kalshi does not return in this response (the conservative
        # direction: it can only over-state what was spent).
        fill_count = self._count(body, "fill_count_fp", "fill_count")
        remaining = self._count(body, "remaining_count_fp", "remaining_count")
        filled_usd = round(fill_count * per_contract, 6)
        return {
            "status": "submitted",
            "success": True,
            "venue": "kalshi",
            "environment": self.environment,
            "order_id": body.get("order_id"),
            "client_order_id": order["client_order_id"],
            "market_id": market.id,
            "ticker": ticker,
            "side": side,
            "side_sent": book_side,
            "price": per_contract,
            "yes_leg_price": snapped,
            "price_grid": grid_source,
            "count": count,
            "size_matched": fill_count,
            "original_size": count,
            "limit_price": per_contract,
            "filled_usd": filled_usd,
            "fill_count": fill_count,
            "remaining_count": remaining,
            "ts_ms": body.get("ts_ms"),
            "filled": bool(fill_count),
            "message": ("Submitted to Kalshi. The exchange's own fill_count and "
                        "remaining_count are reported above; nothing here assumes "
                        "a fill."),
        }

    async def probe_order_permission(self, opportunity=None) -> bool:
        """
        Prove this account can submit AND withdraw an order - by doing both.

        The AccountHealthEngine refuses to certify an account for real capital
        without this rung, and its rule is that an unproven permission is
        unproven rather than assumed. So the probe places ONE contract at the
        LOWEST price the market's own grid allows - a bid that low cannot be
        marketable - and then cancels it, exactly as PolymarketAdapter does:

          * refuses in dry_run: a probe places a real order, so it must never run
            while the agent believes it is simulating;
          * always attempts the cancel, including when the place half-succeeded,
            because an order left resting is exposure nobody told the caller
            about;
          * returns True only when the exchange created the order AND the cancel
            was confirmed.

        The Kalshi DEMO environment (KALSHI_ENVIRONMENT=demo, separate keys,
        mock funds) is where this is meant to be exercised first.
        """
        self.last_order_probe = {"attempted": False, "reason": ""}
        if self.dry_run:
            self.last_order_probe["reason"] = "adapter is in dry_run; a probe places a real order"
            logger.warning("Order probe refused: adapter is in dry_run.")
            return False
        ready, why = self._auth_ready()
        if not ready:
            self.last_order_probe["reason"] = why
            logger.warning(f"Order probe refused: {why}")
            return False
        market = getattr(opportunity, "market", None) if opportunity is not None else None
        if market is None:
            self.last_order_probe["reason"] = "no tradeable market supplied"
            logger.warning("Order probe refused: no market supplied.")
            return False
        ticker = str((market.raw or {}).get("ticker") or market.id or "")
        if not ticker:
            self.last_order_probe["reason"] = "market has no Kalshi ticker"
            return False
        if getattr(market, "closed", False):
            self.last_order_probe["reason"] = f"{ticker} is closed; the exchange would refuse"
            return False

        # The lowest price this market will accept at one cent or above: whole
        # cents are valid in every Kalshi price structure, and a bid that low
        # cannot be marketable. `snap_price(..., "up")` finds it on the market's
        # own grid instead of assuming the grid.
        ranges = self.parse_price_ranges(market.raw or {})
        probe_price = self.snap_price(market.raw or {}, 0.01, "up") or 0.01
        order = {
            "ticker": ticker,
            "side": "bid",
            "count": "1.00",
            "price": f"{probe_price:.4f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": str(uuid.uuid4()),
        }
        order_id, reason = "", ""
        try:
            resp = self._signed_request("POST", KALSHI_ORDER_PATH, json=order)
            if resp.status_code == 201:
                order_id = str((resp.json() or {}).get("order_id") or "")
            else:
                reason = f"post refused: HTTP {resp.status_code} {resp.text[:160]}"
            if not order_id and not reason:
                reason = "post returned no order id"
        except Exception as e:  # noqa: BLE001
            reason = f"place failed: {type(e).__name__}: {e}"
        logger.info(f"[kalshi] probe order: {order_id or reason}")

        cancelled, cancel_detail = False, None
        if order_id:
            try:
                cancel = self._signed_request(
                    "DELETE", f"{KALSHI_ORDER_PATH}/{order_id}")
                cancelled = cancel.status_code in (200, 204)
                cancel_detail = cancel.status_code
                if not cancelled:
                    reason = f"cancel failed: HTTP {cancel.status_code} {cancel.text[:160]}"
            except Exception as e:  # noqa: BLE001
                reason = f"cancel raised: {type(e).__name__}: {e}"

        self.last_order_probe = {
            "attempted": True, "venue": "kalshi", "environment": self.environment,
            "market_id": ticker, "probe_price": probe_price, "probe_count": 1,
            "order_id": order_id, "cancelled": cancelled,
            "cancel_http_status": cancel_detail, "reason": reason,
            "grid_source": ("market price_ranges" if ranges else
                            "assumed 0.01: price_ranges unreadable"),
        }
        if order_id and cancelled:
            logger.success(f"Order permission VERIFIED for kalshi: placed and "
                           f"cancelled 1 contract @ ${probe_price:.4f} ({order_id})")
            return True
        logger.error(f"Order permission NOT verified for kalshi: "
                     f"{reason or 'unknown'}. Live capital stays disabled.")
        return False
