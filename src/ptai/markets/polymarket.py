"""
Polymarket integration - Gamma API for discovery, CLOB for execution
Fully local, uses public APIs, no cloud agent needed
"""
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import aiohttp
from loguru import logger

from .base import Market, Token, MarketSource, DataMode
from ..config import get_settings

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

class PolymarketClient:
    def __init__(self, gamma_api: str = GAMMA_API, clob_api: str = CLOB_API):
        self.gamma_api = gamma_api.rstrip("/")
        self.clob_api = clob_api.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PTAI/1.0 Local Trading Agent",
            "Accept": "application/json"
        })
        settings = get_settings()
        self.min_liquidity = settings.min_liquidity
        self.min_volume_24h = settings.min_volume_24h

    def _parse_outcomes(self, market_data: Dict) -> tuple:
        """Parse outcomes which come as JSON strings"""
        outcomes = market_data.get("outcomes", "[]")
        outcome_prices = market_data.get("outcomePrices", "[]")
        clob_token_ids = market_data.get("clobTokenIds", "[]")

        # Handle if already list or stringified
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except:
                outcomes = [outcomes]
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except:
                outcome_prices = []
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except:
                clob_token_ids = []

        # Convert prices to float
        prices = []
        for p in outcome_prices:
            try:
                prices.append(float(p))
            except:
                prices.append(0.5)

        return outcomes, prices, clob_token_ids

    def _event_to_markets(self, event: Dict) -> List[Market]:
        """Convert Gamma event -> list of Market objects"""
        markets = []
        event_slug = event.get("slug", "")
        event_title = event.get("title", "")

        for m in event.get("markets", []):
            try:
                outcomes, prices, token_ids = self._parse_outcomes(m)

                tokens = []
                for i, outcome in enumerate(outcomes):
                    token_id = token_ids[i] if i < len(token_ids) else f"{m.get('id')}_{i}"
                    price = prices[i] if i < len(prices) else 0.5
                    tokens.append(Token(token_id=token_id, outcome=outcome, price=price))

                # Volume & liquidity can be at event or market level
                volume = float(m.get("volume") or event.get("volume") or 0)
                volume_24h = float(m.get("volume24hr") or event.get("volume24hr") or m.get("volume24h") or event.get("volume24h") or 0)
                liquidity = float(m.get("liquidity") or event.get("liquidity") or 0)

                # End date
                end_date_str = m.get("endDate") or event.get("endDate") or event.get("end_date")
                end_date = None
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    except:
                        pass

                market = Market(
                    id=str(m.get("id") or m.get("conditionId") or ""),
                    source=MarketSource.POLYMARKET,
                    question=m.get("question") or event.get("title") or "",
                    description=m.get("description") or event.get("description") or "",
                    outcomes=outcomes,
                    outcome_prices=prices,
                    tokens=tokens,
                    volume=volume,
                    volume_24h=volume_24h,
                    liquidity=liquidity,
                    end_date=end_date,
                    active=event.get("active", True) and not event.get("closed", False),
                    closed=event.get("closed", False),
                    slug=m.get("slug", ""),
                    event_slug=event_slug,
                    condition_id=m.get("conditionId") or m.get("condition_id") or "",
                    market_type="binary" if len(outcomes) == 2 else "categorical",
                    raw={"event": event, "market": m, "venue": "polymarket", "data_mode": "live", "data_source": "gamma_api", "is_mock": False},
                    venue_id="polymarket",
                    venue_type="prediction",
                    data_mode=DataMode.LIVE,
                    data_source="gamma_api",
                    is_mock=False
                )
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market {m.get('id')}: {e}")
                continue
        return markets

    def fetch_events(self, limit: int = 100, offset: int = 0, order: str = "volume24hr", active: bool = True, closed: bool = False) -> List[Dict]:
        """Fetch raw events from Gamma API"""
        params = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": False,
            "active": str(active).lower(),
            "closed": str(closed).lower()
        }
        try:
            resp = self.session.get(f"{self.gamma_api}/events", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Gamma API fetch failed offset={offset}: {e}")
            return []

    def scan_markets(self, target_count: int = 750, order_by: str = "volume24hr") -> List[Market]:
        """
        Scan 500-1000 markets as required
        Paginates through events endpoint
        """
        logger.info(f"Scanning Polymarket for {target_count} markets, order_by={order_by}")
        all_markets: List[Market] = []
        offset = 0
        batch = 100
        seen_ids = set()

        # Map order_by aliases
        order_map = {
            "volume24hr": "volume24hr",
            "volume_24hr": "volume24hr",
            "volume": "volume",
            "liquidity": "liquidity",
            "volume24h": "volume24hr"
        }
        order = order_map.get(order_by, "volume24hr")

        while len(all_markets) < target_count:
            events = self.fetch_events(limit=batch, offset=offset, order=order, active=True, closed=False)
            if not events:
                logger.warning(f"No more events at offset {offset}, got {len(all_markets)} markets")
                break

            for ev in events:
                markets = self._event_to_markets(ev)
                for m in markets:
                    if m.id in seen_ids:
                        continue
                    # Apply filters
                    if m.liquidity < self.min_liquidity and self.min_liquidity > 0:
                        continue
                    if m.volume_24h < self.min_volume_24h and self.min_volume_24h > 0:
                        continue
                    if not m.active:
                        continue
                    # Must have valid token
                    if not m.tokens:
                        continue
                    seen_ids.add(m.id)
                    all_markets.append(m)
                    if len(all_markets) >= target_count:
                        break
                if len(all_markets) >= target_count:
                    break

            logger.info(f"Scanned offset {offset}, total markets {len(all_markets)}/{target_count}")
            offset += batch

            # Avoid hammering API
            time.sleep(0.3)

            # Safety: don't loop forever
            if offset > 2000:
                break

        logger.success(f"Scan complete: {len(all_markets)} markets found")
        return all_markets[:target_count]

    async def scan_markets_async(self, target_count: int = 750, order_by: str = "volume24hr") -> List[Market]:
        """Async version for faster scanning"""
        # For simplicity, use sync in async wrapper
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.scan_markets(target_count, order_by))

    def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """Get orderbook for a token from CLOB"""
        try:
            resp = self.session.get(f"{self.clob_api}/book", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price"""
        try:
            resp = self.session.get(f"{self.clob_api}/midpoint", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception as e:
            logger.debug(f"Midpoint fetch failed {token_id}: {e}")
            return None

    def get_market_resolution(self, market_id: str) -> Optional[Dict]:
        """
        Ask the Gamma API whether a market has closed and how it settled.

        Returns the raw market dict, or None when the lookup failed. It does NOT
        synthesise an answer: the caller must be able to tell "not settled yet"
        from "could not find out", because the second must never be recorded as
        an outcome.

        Gamma reports a settled market with `closed: true` and `outcomePrices`
        set to the settlement values - exactly ["1", "0"] or ["0", "1"]. Those
        are settlement marks, not tradeable prices.
        """
        try:
            resp = self.session.get(f"{self.gamma_api}/markets",
                                    params={"id": market_id}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Market resolution fetch failed for {market_id}: {e}")
            return None

        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            # Some Gamma responses wrap the market in a list under "markets".
            for key in ("markets",):
                inner = data.get(key)
                if isinstance(inner, list) and inner:
                    return inner[0]
            return data
        return None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        try:
            resp = self.session.get(f"{self.clob_api}/price", params={"token_id": token_id, "side": "buy"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except:
            return None

class PolymarketExecutor:
    """
    The Polymarket CLOB boundary, on the CLOB **V2** client.

    This was written against `py-clob-client` (the V1 client), whose newest
    release is 0.34.6 from 2026-02-19 - two months BEFORE Polymarket's V2
    exchange went live on 2026-04-28, documented as "no backward compatibility
    after go-live". The V1 Neg Risk Adapter was retired on 2026-07-17, the same
    day `py-clob-client-v2` shipped. The integration now uses the V2 client.

    V2 API differences that matter and are handled here:
      * `create_or_derive_api_creds()` no longer exists. Credentials come from
        `derive_api_key()` / `create_api_key()` or are supplied explicitly.
      * `create_order` takes `CreateOrderOptions(tick_size, neg_risk)`. If they
        are omitted the client fetches them per order, which means the agent
        cannot know the tick at the moment it models a price.
      * `cancel_order` takes an `OrderPayload`, not a bare id string.
      * Order, trade and market-info reads exist as first-class calls:
        `get_open_orders`, `get_order`, `get_trades`, `get_clob_market_info`.
        None of them were used, so resting orders were invisible.

    Everything venue-facing returns a dict with an explicit `is_real` flag. A
    caller must never have to guess whether a number came from the venue.
    """

    # The venue's own vocabulary for an order that is resting rather than done.
    ORDER_TYPES = ("GTC", "GTD", "FAK", "FOK")

    def __init__(self, private_key: Optional[str] = None,
                 funder: Optional[str] = None, chain_id: int = 137,
                 signature_type: int = 1, host: str = CLOB_API,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 api_passphrase: Optional[str] = None,
                 builder_code: Optional[str] = None):
        self.private_key = private_key
        self.funder = funder
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.host = host
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.builder_code = builder_code
        self.client = None
        self.client_error: Optional[str] = None
        # Mechanics are stable for a market but not free to fetch, so cache them
        # for the life of the executor. Tick size can change mid-life, so the
        # cache is deliberately short-lived and refreshable.
        self._mechanics_cache: Dict[str, Any] = {}
        self._init_client()

    # ------------------------------------------------------------------
    # client
    # ------------------------------------------------------------------

    def _init_client(self):
        if not self.private_key:
            logger.warning("No private key - Polymarket is read-only, no orders can be signed")
            self.client_error = "no private key"
            return
        try:
            from py_clob_client_v2 import ClobClient
            from py_clob_client_v2.clob_types import ApiCreds, BuilderConfig

            creds = None
            if self.api_key and self.api_secret and self.api_passphrase:
                creds = ApiCreds(api_key=self.api_key, api_secret=self.api_secret,
                                 api_passphrase=self.api_passphrase)

            builder_config = None
            if self.builder_code:
                builder_config = BuilderConfig(builder_code=self.builder_code)

            kwargs: Dict[str, Any] = {
                "host": self.host,
                "chain_id": self.chain_id,
                "key": self.private_key,
                "signature_type": self.signature_type,
                "funder": self.funder,
            }
            if creds is not None:
                kwargs["creds"] = creds
            if builder_config is not None:
                kwargs["builder_config"] = builder_config

            self.client = ClobClient(**kwargs)

            if creds is None:
                # V1 called this create_or_derive_api_creds(); V2 separates
                # derive (existing key) from create (new key). Deriving first is
                # the non-destructive choice: creating a key would invalidate
                # any credentials the operator already uses elsewhere.
                derived = None
                try:
                    derived = self.client.derive_api_key()
                except Exception as e:
                    logger.info(f"Polymarket: no existing API key to derive ({e}); creating one")
                if derived is None:
                    derived = self.client.create_api_key()
                self.client.set_api_creds(derived)
            logger.success("Polymarket CLOB V2 client initialized")
        except ImportError as e:
            self.client_error = f"py_clob_client_v2 not installed: {e}"
            logger.error(
                "Polymarket execution unavailable: the CLOB V2 client is not "
                f"installed ({e}). The V1 client cannot talk to the V2 exchange.")
            self.client = None
        except Exception as e:
            self.client_error = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to init Polymarket CLOB V2 client: {type(e).__name__}: {e}")
            self.client = None

    @property
    def can_sign(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------
    # market mechanics
    # ------------------------------------------------------------------

    def get_clob_market_info(self, condition_id: str) -> Dict[str, Any]:
        """
        Every CLOB parameter for a market in one call: tokens, tick size,
        neg-risk, minimum order size, fees, rewards, delay.
        """
        if not self.client or not condition_id:
            return {}
        try:
            response = self.client.get_clob_market_info(condition_id)
        except Exception as e:
            logger.debug(f"get_clob_market_info({condition_id}) failed: {e}")
            return {}
        return _unwrap(response)

    def get_tick_size(self, token_id: str) -> Optional[str]:
        if not self.client or not token_id:
            return None
        try:
            return str(self.client.get_tick_size(token_id))
        except Exception as e:
            logger.debug(f"get_tick_size failed: {e}")
            return None

    def get_neg_risk(self, token_id: str) -> Optional[bool]:
        if not self.client or not token_id:
            return None
        try:
            return bool(self.client.get_neg_risk(token_id))
        except Exception as e:
            logger.debug(f"get_neg_risk failed: {e}")
            return None

    def get_mechanics(self, token_id: Optional[str],
                      condition_id: Optional[str] = None,
                      refresh: bool = False) -> "MarketMechanics":
        """
        The venue's order rules for one market.

        Read at DECISION time, not at signing time. The SDK fetches the tick
        itself when an order is built, but by then the edge has already been
        computed against a price on a grid the venue may not share.
        """
        from .mechanics import MarketMechanics, mechanics_for_market

        key = str(condition_id or token_id or "")
        if key and not refresh and key in self._mechanics_cache:
            return self._mechanics_cache[key]

        mechanics = mechanics_for_market(token_id, self, condition_id=condition_id)
        if key:
            self._mechanics_cache[key] = mechanics
        return mechanics

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------

    def place_order(self, token_id: str, price: float, size: float,
                    side: str = "BUY", order_type: str = "GTC",
                    dry_run: bool = True, mechanics=None,
                    post_only: bool = False, condition_id: Optional[str] = None,
                    crossing: bool = False) -> Dict[str, Any]:
        """
        Place a limit order, on the venue's tick grid, after checking the
        venue's own rules.

        `size` is in SHARES. `price` is rounded to the tick here - conservatively
        on a buy, so the signed price is never worse than the modelled price -
        and the rounded values are returned alongside the venue's response so a
        caller can record what was actually signed rather than what it asked
        for.
        """
        from .mechanics import MarketMechanics

        if mechanics is None:
            mechanics = self.get_mechanics(token_id, condition_id=condition_id)

        side_const = str(side).upper()

        # Rounding snaps an OFF-GRID price onto the tick, but a price outside
        # the venue's legal band is not a rounding problem: the venue would
        # reject it (its own validity check is exactly this bound), and silently
        # clamping it would trade a different price than the caller asked for.
        low, high = mechanics.tick, 1 - mechanics.tick
        if not (low <= float(price) <= high):
            prepared = {
                "status": "rejected", "is_real": False,
                "reason": (f"price {price} is outside the venue's legal range "
                           f"[{low}, {high}] for a {mechanics.tick_size} tick"),
                "requested_price": price, "requested_size": size,
                "token_id": token_id, "side": side_const,
                "mechanics": mechanics.to_dict(),
            }
            logger.warning(f"Order refused before signing: {prepared['reason']}")
            return prepared

        signed_price = mechanics.round_price(price, side)
        signed_size = mechanics.round_size(size)

        prepared = {
            "status": "not_sent", "is_real": False,
            "signed_price": signed_price, "signed_size": signed_size,
            "requested_price": price, "requested_size": size,
            "token_id": token_id, "side": side_const,
            "order_type": order_type,
            "mechanics": mechanics.to_dict(),
        }

        if signed_price != price or signed_size != size:
            prepared["mechanics_adjustment"] = {
                "price": round(signed_price - price, 8),
                "size": round(signed_size - size, 8),
                "reason": (f"price and size snapped to the venue's tick "
                           f"{mechanics.tick_size} and size step"),
            }
        if not mechanics.is_real:
            prepared["mechanics_warning"] = (
                "mechanism values are assumed, not reported by the venue")

        ok, reason = mechanics.validate_order(signed_price, signed_size)
        if not ok:
            prepared.update({"status": "rejected", "reason": reason})
            logger.warning(f"Order refused before signing: {reason}")
            return prepared

        if dry_run or not self.client:
            prepared.update({
                "status": "dry_run",
                "simulated": True,
                "message": (f"DRY RUN - would place {side_const} {signed_size} "
                            f"@ {signed_price} for {token_id[:20]}"),
                "reason": ("no client" if not self.client else
                           "dry_run=True: real capital disabled at the venue boundary"),
            })
            return prepared

        try:
            from py_clob_client_v2.clob_types import (
                CreateOrderOptions, OrderArgs, OrderType,
            )
            from py_clob_client_v2.order_builder.constants import BUY, SELL

            if order_type not in self.ORDER_TYPES:
                prepared.update({"status": "rejected",
                                 "reason": f"unsupported order type {order_type}"})
                return prepared

            order_args = OrderArgs(
                token_id=token_id,
                price=signed_price,
                size=signed_size,
                side=BUY if side_const in ("BUY", "YES", "LONG", "TRUE", "1") else SELL,
            )
            options = CreateOrderOptions(
                tick_size=mechanics.tick_size,
                neg_risk=mechanics.neg_risk,
            )
            response = self.client.create_and_post_order(
                order_args=order_args,
                options=options,
                order_type=getattr(OrderType, order_type),
                post_only=bool(post_only),
            )
        except Exception as e:
            # A failure here may or may not mean the order reached the book. It
            # is reported as its own status so the caller reconciles rather than
            # assuming either way.
            logger.error(f"Order placement failed: {type(e).__name__}: {e}")
            prepared.update({
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "unconfirmed_send": True,
                "reason": ("the send failed without a venue confirmation; an "
                           "order may still be resting and must be reconciled "
                           "against the venue's open orders"),
            })
            return prepared

        payload = _unwrap(response) or {}
        prepared.update({
            "status": payload.get("status") or "submitted",
            "is_real": True,
            "orderID": payload.get("orderID") or payload.get("id") or "",
            "order_id": payload.get("orderID") or payload.get("id") or "",
            # The venue's own accounting of the order.
            "size_matched": _as_float(payload.get("size_matched")),
            "original_size": _as_float(payload.get("original_size")) or signed_size,
            "making_amount": payload.get("makingAmount"),
            "taking_amount": payload.get("takingAmount"),
            "transactions_hashes": payload.get("transactionsHashes") or [],
            "trade_ids": payload.get("tradeIDs") or payload.get("tradeIds") or [],
            "raw": payload,
        })
        logger.info(
            f"Polymarket order {prepared['status']} id={prepared['order_id'] or '-'} "
            f"{side_const} {signed_size} @ {signed_price} matched={prepared['size_matched']}")
        return prepared

    def place_market_order(self, token_id: str, amount_usd: float,
                           side: str = "BUY", order_type: str = "FOK",
                           dry_run: bool = True, mechanics=None,
                           max_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a marketable order, priced from the book by the venue itself.

        The price is still reported so the agent can account for what it paid
        rather than what it hoped to pay.
        """
        from .mechanics import MarketMechanics

        if mechanics is None:
            mechanics = self.get_mechanics(token_id)
        prepared = {
            "status": "not_sent", "is_real": False, "amount_usd": amount_usd,
            "side": str(side).upper(), "order_type": order_type,
            "mechanics": mechanics.to_dict(),
        }
        if not mechanics.accepting_orders:
            prepared.update({"status": "rejected",
                             "reason": "venue is not accepting orders in this market"})
            return prepared
        if amount_usd < mechanics.min_order_notional_usd:
            prepared.update({
                "status": "rejected",
                "reason": (f"notional ${amount_usd:.2f} below the "
                           f"${mechanics.min_order_notional_usd} minimum"),
            })
            return prepared
        if dry_run or not self.client:
            prepared.update({"status": "dry_run", "simulated": True,
                             "reason": "dry_run or no client"})
            return prepared
        try:
            from py_clob_client_v2.clob_types import MarketOrderArgs, MarketOrderArgsV2, OrderType
            from py_clob_client_v2.order_builder.constants import BUY, SELL

            price = 0.0
            try:
                price = float(self.client.calculate_market_price(
                    token_id, "BUY" if str(side).upper().startswith("B") else "SELL",
                    amount_usd, getattr(OrderType, order_type)))
            except Exception as e:
                logger.warning(f"calculate_market_price unavailable: {e}")
            if max_price is not None and price and price > max_price:
                prepared.update({
                    "status": "rejected",
                    "reason": (f"venue market price {price} exceeds the "
                               f"{max_price} limit"),
                })
                return prepared

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usd,
                side=BUY if str(side).upper().startswith("B") else SELL,
                price=price,
                order_type=getattr(OrderType, order_type),
            )
            response = self.client.create_and_post_market_order(
                order_args=order_args,
                order_type=getattr(OrderType, order_type),
            )
        except Exception as e:
            logger.error(f"Market order failed: {type(e).__name__}: {e}")
            prepared.update({"status": "failed", "error": f"{type(e).__name__}: {e}",
                             "unconfirmed_send": True})
            return prepared
        payload = _unwrap(response) or {}
        prepared.update({
            "status": payload.get("status") or "submitted",
            "is_real": True,
            "orderID": payload.get("orderID") or payload.get("id") or "",
            "order_id": payload.get("orderID") or payload.get("id") or "",
            "quoted_price": price,
            "size_matched": _as_float(payload.get("size_matched")),
            "raw": payload,
        })
        return prepared

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel a live order. A cancel that fails is reported, never swallowed:
        an order left resting in the book is real exposure the agent believes
        it does not have.
        """
        if not self.client:
            return {"status": "no_client", "is_real": False,
                    "reason": "no CLOB client (missing key or credentials failed)"}
        if not order_id:
            return {"status": "rejected", "reason": "no order_id to cancel"}
        # V2 takes an OrderPayload. Older/V1-shaped clients took the bare id.
        # Both shapes are attempted so a client cannot be silently mis-called,
        # and the last failure is the one reported.
        attempts = []
        try:
            from py_clob_client_v2.clob_types import OrderPayload
            attempts.append(lambda: self.client.cancel_order(OrderPayload(orderID=order_id)))
        except ImportError:
            pass
        attempts.append(lambda: self.client.cancel_order(order_id))
        attempts.append(lambda: self.client.cancel(order_id))

        response = None
        last_error = None
        for attempt in attempts:
            try:
                response = attempt()
                last_error = None
                break
            except Exception as e:
                last_error = e
        if last_error is not None:
            logger.error(f"Cancel failed for {order_id}: {type(last_error).__name__}: {last_error}")
            return {"status": "error", "order_id": order_id,
                    "error": f"{type(last_error).__name__}: {last_error}"}

        if isinstance(response, dict):
            not_cancelled = response.get("not_canceled") or response.get("notCanceled") or {}
            if not_cancelled:
                logger.error(
                    f"Order {order_id} was NOT cancelled: {not_cancelled}. It may "
                    f"still be resting in the book.")
                return {"status": "not_cancelled", "order_id": order_id,
                        "is_real": True, "raw": response}
        logger.info(f"Order cancelled: {order_id}")
        return {"status": "cancelled", "order_id": order_id, "is_real": True,
                "raw": response}

    def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        if not self.client:
            return {"status": "no_client", "is_real": False}
        if not order_ids:
            return {"status": "nothing_to_do", "is_real": True}
        try:
            from py_clob_client_v2.clob_types import OrderPayload
            response = self.client.cancel_orders(
                [OrderPayload(orderID=oid) for oid in order_ids])
        except Exception as e:
            logger.error(f"Batch cancel failed: {type(e).__name__}: {e}")
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}
        return {"status": "cancelled", "is_real": True, "count": len(order_ids),
                "raw": response}

    def cancel_all(self) -> Dict[str, Any]:
        if not self.client:
            return {"status": "no_client", "is_real": False}
        try:
            response = self.client.cancel_all()
        except Exception as e:
            logger.error(f"Cancel-all failed: {type(e).__name__}: {e}")
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}
        return {"status": "cancelled", "is_real": True, "raw": response}

    # ------------------------------------------------------------------
    # order and trade reads - the reconciliation inputs
    # ------------------------------------------------------------------

    def get_open_orders(self, market_id: Optional[str] = None,
                        asset_id: Optional[str] = None,
                        order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Orders resting in the book right now.

        Nothing called this before, so an order that rested was invisible: the
        agent would record no position and go on believing it had no exposure.
        """
        if not self.client:
            return {"available": False, "is_real": False, "orders": [],
                    "reason": "no client"}
        try:
            from py_clob_client_v2.clob_types import OpenOrderParams
            params = OpenOrderParams(market=market_id, asset_id=asset_id, id=order_id)
            response = self.client.get_open_orders(params)
        except Exception as e:
            logger.warning(f"get_open_orders failed: {type(e).__name__}: {e}")
            return {"available": False, "is_real": False, "orders": [],
                    "reason": f"{type(e).__name__}: {e}"}
        orders = response if isinstance(response, list) else _unwrap(response) or []
        if not isinstance(orders, list):
            orders = []
        return {"available": True, "is_real": True, "orders": orders,
                "count": len(orders), "source": "clob_open_orders"}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """One order's full state, including how much of it has matched."""
        if not self.client or not order_id:
            return {"available": False, "is_real": False}
        try:
            response = self.client.get_order(order_id)
        except Exception as e:
            logger.warning(f"get_order({order_id}) failed: {type(e).__name__}: {e}")
            return {"available": False, "is_real": False,
                    "reason": f"{type(e).__name__}: {e}"}
        payload = _unwrap(response) or {}
        if not isinstance(payload, dict):
            return {"available": False, "is_real": False,
                    "reason": f"unexpected payload {type(payload).__name__}"}
        return {
            "available": True,
            "is_real": True,
            "order_id": order_id,
            "status": payload.get("status"),
            "size_matched": _as_float(payload.get("size_matched")),
            "original_size": _as_float(payload.get("original_size")),
            "price": _as_float(payload.get("price")),
            "side": payload.get("side"),
            "asset_id": payload.get("asset_id") or payload.get("assetId"),
            "market": payload.get("market"),
            "outcome": payload.get("outcome"),
            "raw": payload,
        }

    def get_trades(self, market_id: Optional[str] = None,
                   asset_id: Optional[str] = None,
                   trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fills. On V2 a successful FAK/FOK match no longer returns transaction
        hashes - it returns `tradeIDs`, and the hashes must be resolved by
        polling the trade. This is that read.
        """
        if not self.client:
            return {"available": False, "is_real": False, "trades": []}
        try:
            from py_clob_client_v2.clob_types import TradeParams
            params = TradeParams(market=market_id, asset_id=asset_id, id=trade_id)
            response = self.client.get_trades(params)
        except Exception as e:
            logger.warning(f"get_trades failed: {type(e).__name__}: {e}")
            return {"available": False, "is_real": False, "trades": [],
                    "reason": f"{type(e).__name__}: {e}"}
        trades = response if isinstance(response, list) else _unwrap(response) or []
        if not isinstance(trades, list):
            trades = []
        return {"available": True, "is_real": True, "trades": trades,
                "count": len(trades), "source": "clob_trades"}

    # ------------------------------------------------------------------
    # account
    # ------------------------------------------------------------------

    def get_balance_allowance(self, token_id: Optional[str] = None,
                              asset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        The venue's own view of this account's collateral and allowance.

        This is the honest source for "is there money and is it permitted to
        trade": a local figure proves nothing about either.
        """
        if not self.client:
            return {"available": False, "is_real": False,
                    "reason": "no CLOB client"}
        try:
            from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
            params = BalanceAllowanceParams(
                asset_type=getattr(AssetType, asset_type) if asset_type else None,
                token_id=token_id,
                signature_type=self.signature_type)
            response = self.client.get_balance_allowance(params)
        except Exception as e:
            logger.warning(f"Balance/allowance read failed: {type(e).__name__}: {e}")
            return {"available": False, "is_real": False,
                    "reason": f"{type(e).__name__}: {e}"}

        payload = _unwrap(response) or {}
        if not isinstance(payload, dict):
            return {"available": False, "is_real": False,
                    "reason": f"unexpected response type {type(response).__name__}"}

        # Collateral is a 6-decimal fixed point integer in base units.
        raw_balance = payload.get("balance")
        balance = None
        try:
            if raw_balance is not None:
                balance = float(raw_balance) / 1e6
        except (TypeError, ValueError):
            balance = None

        return {
            "available": True,
            "is_real": True,
            "source": "clob_balance_allowance",
            "balance": balance,
            "balance_base_units": raw_balance,
            "allowance": payload.get("allowance"),
            "raw": payload,
        }

    def get_balance(self) -> Dict:
        return self.get_balance_allowance()


def _unwrap(response: Any) -> Any:
    """
    Unwrap V2's `{"data": ...}` envelope, which some endpoints use and some do
    not, so callers do not have to know which.
    """
    if isinstance(response, dict) and "data" in response and len(response) == 1:
        return response["data"]
    return response


def _as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
