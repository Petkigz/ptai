"""
Simmer - a prediction-market venue with its own SDK, and a venue that will
actually take a paper order.

Simmer (simmer.markets) is a harness for agents that trade prediction markets
through one API. It publishes a Python SDK (`simmer-sdk` on PyPI) whose client
speaks to `https://api.simmer.markets`, and - this is the part that matters here
- it runs a SYNTHETIC venue called `sim` where every active market can be traded
in $SIM, a virtual currency. Simmer's own documentation: "$SIM fills instantly
(AMM, no spread)".

WHY THAT IS WORTH AN ADAPTER. Every other venue PTAI can read either needs money
it does not have (an exchange account, KYC, a funded wallet) or cannot report a
resolution. This one needs one free API key from simmer.markets/dashboard and
nothing else: the markets are real questions, the orders are real orders at the
venue, the fills come back from the venue's own engine, and the venue resolves
the market - so a paper position here closes on a real outcome and counts toward
the record. No money can move: $SIM cannot be withdrawn or converted, and this
adapter never touches Simmer's real-money venues.

WHAT THIS ADAPTER WILL NOT DO:

  * It will not submit a REAL-money order. Simmer can route orders to Polymarket
    and Kalshi with a signed wallet; handing a third party the operator's signing
    key is not something this product does, and it is not needed - PTAI talks to
    Polymarket directly. `venue="sim"` is pinned at construction,
    `real_order_path` is False, live mode gets a refusal that says so, and the
    client is built with `_ignore_env_wallets=True` so not even a stray
    WALLET_PRIVATE_KEY in the environment can be picked up by it.
  * It will not invent a book. When the venue publishes a top of book
    (`best_bid`/`best_ask` and their sizes) those are the prices. When it does
    not - the synthetic venue publishes one number, its current probability -
    the book is that number on both sides with `spread_basis` and `size_basis`
    saying exactly why, because Simmer's own documentation says a $SIM order
    fills at the AMM price with no spread.
  * It will not guess a resolution. `Market.outcome` is the venue's own field:
    True/False when the market has resolved, None while it is open. A resolved
    market with no outcome is refused, not interpreted.
  * It will not pretend the SDK is installed when it is not. The SDK is an
    optional dependency: `pip install simmer-sdk`. Without it - or without a
    saved key - every call returns what is missing, and the venue row says so.

FEES. The synthetic venue charges no fee and no gas: the $SIM cost of an order is
what the venue's own fill reports. (`fee_taker_pct` is therefore 0.0, and the
figures PTAI records come from the venue's TradeResult rather than from arithmetic
of ours.)
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from loguru import logger

from .adapter import (AdapterCapability, EligibilityStatus, MarketAdapter,
                      VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


SIMMER_ID_PREFIX = "simmer"

# The venue's synthetic, virtual-currency book. Pinned: the adapter will not use
# a real-money venue even if a caller asks for one.
SIMMER_PAPER_VENUE = "sim"

# Simmer publishes no size for the synthetic venue; a $SIM order fills at the
# AMM price. The ladder therefore carries one share a side - the smallest unit
# the venue trades - and `size_basis` states that this is not a depth claim.
PUBLISHED_QUANTITY_SHARES = 1.0

# The widest two-sided quote this adapter will call executable.
MAX_TRUSTED_SPREAD = 0.10

# What to ask for when the caller does not say.
DEFAULT_TARGET_MARKETS = 100


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a field from an SDK object or from a plain dict.

    The SDK returns dataclasses (`Market`, `TradeResult`, `Position`), and the
    same shapes arrive as dicts when a caller asks for compact output. Both are
    read through here so no accessor silently answers "unknown" for one of them.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


class SimmerAdapter(MarketAdapter):
    def __init__(self, api_key: Optional[str] = None, use_virtual: bool = True):
        super().__init__(venue_id="simmer", venue_type=VenueType.PREDICTION)
        self.api_key = api_key
        # Kept for the existing call site (`SimmerAdapter(use_virtual=True)`);
        # only the synthetic venue is wired, so this is documentation rather
        # than a switch.
        self.use_virtual = bool(use_virtual)
        self.last_error: str = ""
        self._client_obj: Any = None
        self._sdk_module: Any = None
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            # NOT a real-money path: PTAI never signs for a third party, and
            # Simmer's live venues need a signed wallet. `supports_trading` is
            # false for exactly that reason - the $SIM submission path is used
            # through place_order's paper branch (see the module docstring).
            supports_trading=False,
            real_order_path=False,
            # Every call - markets included - carries the API key.
            requires_credentials=True,
            supports_portfolio=bool(api_key),
            supports_history=False,
            fee_taker_pct=0.0,
            fee_maker_pct=0.0,
            min_order_usd=1.0,
            implementation_status="live",
        )

    # ------------------------------------------------------------------
    # the client
    # ------------------------------------------------------------------
    @property
    def sdk_available(self) -> bool:
        """Is `simmer-sdk` importable in this interpreter?"""
        try:
            import simmer_sdk  # noqa: F401,WPS433
        except Exception:  # noqa: BLE001 - any import failure is the same fact
            return False
        return True

    def _client(self):
        """(client, error). One place where the SDK or the key can be missing."""
        if not self.api_key:
            return None, ("no Simmer API key saved. The key is free at "
                          "simmer.markets/dashboard and is the only thing this "
                          "venue needs - it moves no money")
        if self._client_obj is not None:
            return self._client_obj, ""
        try:
            import simmer_sdk  # noqa: WPS433 - optional dependency
        except Exception as e:  # noqa: BLE001
            return None, (f"the Simmer SDK is not installed ({type(e).__name__}: "
                          f"{e}); install it with 'pip install simmer-sdk'")
        try:
            # `live=True` for the SYNTHETIC venue means the venue records and
            # resolves the $SIM order server-side, which is the whole point of
            # this adapter. The venue is pinned to `sim`: nothing here can reach
            # a real-money market.
            #
            # `_ignore_env_wallets=True` is deliberate and is about the one thing
            # this adapter must never do: the SDK otherwise auto-detects a wallet
            # from WALLET_PRIVATE_KEY / OWS_WALLET in the environment, and PTAI
            # signs for nobody. With wallets ignored, a stray key in the
            # operator's environment cannot reach this client at all.
            self._client_obj = simmer_sdk.SimmerClient(
                api_key=self.api_key, venue=SIMMER_PAPER_VENUE, live=True,
                _ignore_env_wallets=True)
            self._sdk_module = simmer_sdk
        except Exception as e:  # noqa: BLE001
            return None, (f"the Simmer client could not be built "
                          f"({type(e).__name__}: {e})")
        return self._client_obj, ""

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        """
        The synthetic venue is open everywhere; it holds no money.

        Simmer's real-money venues inherit their own rules (Polymarket's
        residency rules, Kalshi's US gate), and this adapter does not use them -
        so the only venue being described here is the one that can be traded for
        free from anywhere.
        """
        return EligibilityStatus.ELIGIBLE

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    async def discover_markets(self, target_count: int = DEFAULT_TARGET_MARKETS,
                               filters: Dict = None) -> List[Market]:
        """
        Active markets on the synthetic venue, newest liquidity first.

        The SDK's `venue="sim"` filter returns the markets that are tradeable
        with $SIM, which is exactly the set this adapter can fill an order in.
        """
        filters = filters or {}
        client, error = self._client()
        if client is None:
            self.last_error = error
            logger.info(f"Simmer: {error}")
            return []
        limit = int(filters.get("limit") or target_count or DEFAULT_TARGET_MARKETS)
        limit = max(1, min(limit, 200))
        try:
            rows = client.get_markets(status="active", venue=SIMMER_PAPER_VENUE,
                                      sort="volume", limit=limit)
        except Exception as e:  # noqa: BLE001 - a venue read may fail any way
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Simmer: could not read markets: {self.last_error}")
            return []
        rows = list(rows or [])
        markets: List[Market] = []
        skipped = 0
        for row in rows:
            market = self._build_market(row)
            if market is None:
                skipped += 1
                continue
            markets.append(market)
        self.last_error = ""
        logger.info(
            f"Simmer discovered {len(markets)} tradeable market(s) on the $SIM "
            f"venue" + (f"; {skipped} row(s) had no usable price or were not "
                        f"tradeable" if skipped else ""))
        return markets[:target_count]

    def _build_market(self, row: Any) -> Optional[Market]:
        """One SDK market, as a Market. None when it cannot be traded or priced."""
        market_id = _text(_field(row, "id"))
        if not market_id:
            return None
        if _field(row, "sim_tradeable") is False:
            # The venue's own paper-trading guard: an order here would be
            # refused, so the market is not offered as a candidate.
            return None
        probability = self._probability(_field(row, "current_probability"))
        if probability is None:
            return None
        status = _text(_field(row, "status")) or "active"
        if status.lower() not in ("active", "open"):
            return None
        question = _text(_field(row, "question")) or market_id
        reference_bid = self._probability(_field(row, "best_bid"))
        reference_ask = self._probability(_field(row, "best_ask"))
        return Market(
            id=f"{SIMMER_ID_PREFIX}-{market_id}",
            # `MarketSource` has no SIMMER member (adding one would change a
            # shared enum for one venue); the same normalised carrier Betfair
            # uses is passed here and `venue_id` below is the authoritative
            # identity, exactly as it is for every adapter.
            source=MarketSource.POLYMARKET,
            question=question[:400],
            description=_text(_field(row, "resolution_criteria"))[:400],
            outcomes=["YES", "NO"],
            outcome_prices=[probability, round(1.0 - probability, 6)],
            tokens=[Token(token_id=market_id, outcome="YES", price=probability)],
            # SIMILAR TO PREDICTIT'S ROW: the venue publishes a price and no
            # traded volume, so both floors are recorded as unpublished rather
            # than as zero ("nobody trades here") - the scan treats those
            # differently on purpose.
            volume=0.0,
            volume_24h=0.0,
            liquidity=0.0,
            end_date=self._parse_date(_field(row, "resolves_at")),
            active=True,
            closed=False,
            slug=market_id,
            event_slug=_text(_field(row, "import_source"))[:50],
            raw={
                "venue": "simmer",
                "api": "simmer_sdk",
                "market_id": market_id,
                "sim_venue": SIMMER_PAPER_VENUE,
                "status": status,
                "import_source": _field(row, "import_source"),
                "is_sdk_only": _field(row, "is_sdk_only"),
                "sim_tradeable": _field(row, "sim_tradeable"),
                "polymarket_condition_id": _field(row, "polymarket_condition_id"),
                # The venue's own field names, kept as it published them, so the
                # book reader below reads the same keys whether the row arrived
                # as a dataclass or as a dict.
                "best_bid": reference_bid,
                "best_ask": reference_ask,
                "best_bid_size": _field(row, "best_bid_size"),
                "best_ask_size": _field(row, "best_ask_size"),
                "reference_note": ("the venue's top of book, when it publishes "
                                   "one; it is the IMPORTED venue's book, not "
                                   "the price a $SIM order fills at"),
                "volume_basis": "not_published",
                "liquidity_basis": "not_published",
            },
            venue_id="simmer",
            venue_type="prediction",
            data_mode=DataMode.LIVE,
            data_source="simmer_sdk",
            is_mock=False,
        )

    # ------------------------------------------------------------------
    # the book
    # ------------------------------------------------------------------
    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        """
        The price a $SIM order would fill at, from the venue's own numbers.

        Two cases, told apart in the payload rather than in a comment:

          * the venue published a top of book for this market (`best_bid` /
            `best_ask`, with sizes) - those are used, and the book is that book;
          * it published none. The synthetic venue quotes ONE number, its
            current probability, and Simmer's documentation states that a $SIM
            order fills at the AMM price with no spread - so the quote is that
            number on both sides, `spread` is 0.0, and every field that would
            otherwise imply depth says where the number comes from.
        """
        probability = self._probability(
            (getattr(market, "outcome_prices", None) or [None])[0]) \
            or self._probability((market.raw or {}).get("current_probability"))
        raw_bid = self._probability((market.raw or {}).get("best_bid"))
        raw_ask = self._probability((market.raw or {}).get("best_ask"))
        bid, ask = raw_bid, raw_ask
        basis = "venue_published_top_of_book"
        if bid is None or ask is None:
            bid = ask = probability
            basis = "synthetic_venue_amm_price"
        if bid is None or ask is None:
            return self._no_quote(market, "the venue published no probability "
                                          "for this market, so there is no price "
                                          "to fill at")
        spread = round(ask - bid, 6)
        if spread < 0:
            return self._no_quote(
                market, f"the venue's quote is crossed (buy {ask} is below sell "
                        f"{bid}); there is no marketable price in it")
        depth_usd = self._published_depth_usd(market, bid, ask)
        executable = spread <= MAX_TRUSTED_SPREAD
        size_basis = (
            "the venue published sizes at the touch for this market" if depth_usd
            else ("Simmer publishes no size for the synthetic venue: a $SIM order "
                  "fills at the AMM price, so the ladder carries one share a side "
                  "and no depth is claimed"))
        return {
            "venue_id": "simmer",
            "market_id": getattr(market, "id", ""),
            "market_question": getattr(market, "question", ""),
            "bids": [{"price": bid, "size": self._published_size(market, "bid")}],
            "asks": [{"price": ask, "size": self._published_size(market, "ask")}],
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "mid": round((ask + bid) / 2.0, 6),
            "depth": depth_usd,
            "depth_basis": ("the smaller published size at the touch, in dollars"
                            if depth_usd else None),
            "size_basis": size_basis,
            "spread_basis": basis,
            "executable": executable,
            "executable_price_yes": ask,
            "executable_price_no": round(1.0 - bid, 6),
            "executable_price": ask,
            "executable_price_note": ("a YES share costs the ask; a NO share costs "
                                      "one minus the bid on this venue's binary "
                                      "book"),
            "is_real": True,
            "is_mock": False,
            "validated": executable,
            "warning": ("" if executable else
                        f"the spread {spread:.2%} is wider than the "
                        f"{MAX_TRUSTED_SPREAD:.0%} this system will trade"),
            "source": "simmer_synthetic_book",
            "venue": SIMMER_PAPER_VENUE,
            "currency": "SIM (virtual)",
        }

    def _no_quote(self, market: Market, why: str) -> Dict[str, Any]:
        """A refusal, not a book - no price, no spread, nothing executable."""
        return {
            "venue_id": "simmer",
            "market_id": getattr(market, "id", ""),
            "available": False,
            "is_real": False,
            "is_mock": False,
            "validated": False,
            "source": "simmer_no_quote",
            "warning": why,
            "reason": why,
            "bids": [],
            "asks": [],
            "depth": None,
            "size_basis": "no quote was published, so there is nothing to size",
            "executable": False,
        }

    def _published_size(self, market: Market, side: str) -> float:
        raw = market.raw or {}
        key = "best_bid_size" if side == "bid" else "best_ask_size"
        value = raw.get(key)
        try:
            size = float(value)
        except (TypeError, ValueError):
            size = 0.0
        return round(size, 6) if size > 0 else PUBLISHED_QUANTITY_SHARES

    def _published_depth_usd(self, market: Market, bid: float,
                             ask: float) -> Optional[float]:
        raw = market.raw or {}
        sizes = []
        for key in ("best_bid_size", "best_ask_size"):
            try:
                size = float(raw.get(key))
            except (TypeError, ValueError):
                size = 0.0
            if size > 0:
                sizes.append(size)
        if not sizes:
            return None
        return round(min(sizes) * ((bid + ask) / 2.0), 2)

    # ------------------------------------------------------------------
    # orders: the venue's own $SIM venue, never real money
    # ------------------------------------------------------------------
    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float,
                          max_price: float) -> Dict[str, Any]:
        """
        Submit to the venue's synthetic venue, and report what IT says filled.

        This is the difference between this adapter and a local paper fill: the
        order goes to Simmer, the fill comes back from Simmer, the position is
        tracked by Simmer, and the outcome will be published by Simmer. The
        currency is $SIM, which cannot be withdrawn or converted, so no real
        money is involved and the trade is recorded as a paper trade.
        """
        market = getattr(opportunity, "market", None)
        side = _text(getattr(opportunity, "side", "")).upper()
        market_key = self._market_key(getattr(market, "id", ""))
        if not self.dry_run:
            return self._refusal(
                market, "this adapter will not place REAL-money orders: Simmer's "
                        "live venues need a signed wallet, and PTAI signs for no "
                        "one. The synthetic $SIM venue is the only path here")
        if side not in ("YES", "NO"):
            return self._refusal(
                market, f"this venue speaks YES/NO; got "
                        f"{getattr(opportunity, 'side', '')!r}")
        if not market_key:
            return self._refusal(market, "no Simmer market id on this record")
        if max_spend_usd <= 0 or not (0 < float(max_price) < 1):
            return self._refusal(
                market, f"invalid order guard (spend {max_spend_usd}, limit "
                        f"{max_price})")
        client, error = self._client()
        if client is None:
            return self._refusal(market, error)

        ask = self._probability((market.raw or {}).get("current_probability"))
        if ask is None:
            ask = self._probability((getattr(market, "outcome_prices", None) or
                                     [None])[0])
        if ask is not None and side == "YES" and ask > float(max_price) + 1e-9:
            return self._refusal(
                market, f"the venue's price {ask:.2f} is above the limit "
                        f"{float(max_price):.2f}; the order was not sent")
        if ask is not None and side == "NO" and ask is not None \
                and round(1.0 - ask, 6) > float(max_price) + 1e-9:
            return self._refusal(
                market, f"the venue's no price {round(1.0 - ask, 6):.2f} is above "
                        f"the limit {float(max_price):.2f}; the order was not sent")

        try:
            result = client.trade(
                market_id=market_key, side=side.lower(),
                amount=round(float(max_spend_usd), 2),
                venue=SIMMER_PAPER_VENUE,
                reasoning="PTAI paper trade on the synthetic venue",
                source="sdk:ptai")
        except Exception as e:  # noqa: BLE001 - the SDK raises on transport errors
            return self._refusal(market, f"the venue call failed "
                                         f"({type(e).__name__}: {e})")

        if not bool(_field(result, "success")):
            why = (_field(result, "error") or _field(result, "skip_reason")
                   or _field(result, "fill_status") or "the venue refused the order")
            return self._refusal(market, f"the venue did not fill: {why}")
        shares = _field(result, "shares_bought")
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            shares = 0.0
        price = self._probability(_field(result, "new_price"))
        cost = _field(result, "cost")
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = 0.0
        if shares <= 0 or price is None:
            return self._refusal(
                market, "the venue reported success without a filled size and "
                        "price, which is not a fill this agent can account for")
        filled_usd = round(cost if cost > 0 else shares * price, 6)
        return {
            "status": "paper",
            "success": True,
            "is_real": False,
            "simulated": True,
            "venue_id": "simmer",
            "market_id": getattr(market, "id", ""),
            "venue_market_id": market_key,
            "side": side,
            "filled_price": price,
            "price": price,
            "size": shares,
            "size_matched": shares,
            "filled_usd": filled_usd,
            "cost_usd": filled_usd,
            "fees_usd": 0.0,
            "currency": "SIM (virtual)",
            "venue": SIMMER_PAPER_VENUE,
            "venue_trade_id": _field(result, "trade_id"),
            "fill_status": _field(result, "fill_status"),
            "sim_balance": _field(result, "balance"),
            "source": "simmer_sdk_sim_venue",
            "reason": (f"filled on Simmer's $SIM venue: {shares:.4f} share(s) at "
                       f"{price:.4f}, ${filled_usd:.2f} of virtual currency. No "
                       f"real money was involved; the venue tracks and resolves "
                       f"this position"),
            "warnings": list(_field(result, "warnings") or []),
        }

    def _refusal(self, market: Any, why: str) -> Dict[str, Any]:
        """A refusal, not a fill. Nothing was opened."""
        return {
            "status": "refused",
            "success": False,
            "is_real": False,
            "venue_id": "simmer",
            "market_id": getattr(market, "id", ""),
            "reason": why,
            "message": (f"Refusal, not a fill. No Simmer position was opened: "
                        f"{why}"),
        }

    # ------------------------------------------------------------------
    # the venue's order rules
    # ------------------------------------------------------------------
    def get_mechanics(self, opportunity=None, token_id: Optional[str] = None):
        """
        What this venue's orders obey, as far as it publishes.

        The synthetic venue's fills come back from the venue itself, so its fee
        is not derived here: `taker_fee_rate` is 0.0 and the recorded cost is
        whatever the venue's TradeResult reported. `is_real` is False because
        this is the venue's documented shape rather than a per-market read -
        the same honesty rule the other adapters follow.
        """
        from ..markets.mechanics import MarketMechanics  # noqa: WPS433

        return MarketMechanics(
            tick_size="0.01",
            min_order_size=1.0,
            min_order_notional_usd=0.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            source="simmer_documented_rules",
            is_real=False,
            warnings=[
                "the $SIM venue charges no fee and its orders fill at the AMM "
                "price; the cost PTAI records is the venue's own fill, not an "
                "estimate made here",
            ],
        )

    # ------------------------------------------------------------------
    # the outcome
    # ------------------------------------------------------------------
    async def get_settlement(self, market_id: str) -> Dict[str, Any]:
        """
        What the venue says this market resolved to.

        `Market.outcome` is the venue's own field: True for YES, False for NO,
        None while the market has not resolved. A resolved market whose outcome
        is missing is REFUSED rather than interpreted - the same rule every
        other adapter here follows, because an inferred outcome is a fabricated
        one once it reaches the calibration record.
        """
        market_key = self._market_key(market_id)
        if not market_key:
            return self._unsettled("no_market_id", "empty Simmer market id")
        client, error = self._client()
        if client is None:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "simmer_unreachable", "reason": error,
                    "venue_id": "simmer"}
        try:
            row = client.get_market_by_id(market_key)
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "simmer_unreadable",
                    "reason": (f"could not read market {market_key}: "
                               f"{type(e).__name__}: {e}"),
                    "venue_id": "simmer"}
        if row is None:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "simmer_market_missing",
                    "reason": (f"the venue does not have market {market_key}"),
                    "venue_id": "simmer"}
        status = _text(_field(row, "status")).lower()
        outcome = _field(row, "outcome")
        if outcome is True or outcome is False:
            value = 1.0 if outcome else 0.0
            return {"settled": True, "outcome": value, "is_real": True,
                    "source": "simmer_market_outcome",
                    "reason": (f"the venue reports this market resolved "
                               f"{'YES' if value == 1.0 else 'NO'} (status "
                               f"{status or 'resolved'})"),
                    "venue_id": "simmer", "market_id": market_key}
        if status in ("resolved", "closed", "settled"):
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "simmer_outcome_missing",
                    "reason": (f"market {market_key} is {status} and the venue "
                               f"published no outcome; this adapter does not "
                               f"infer one"),
                    "venue_id": "simmer", "market_id": market_key}
        return {"settled": False, "outcome": None, "is_real": True,
                "source": "simmer_open",
                "reason": (f"market {market_key} is still {status or 'active'}; "
                           f"no outcome is published yet"),
                "venue_id": "simmer", "market_id": market_key}

    @staticmethod
    def _unsettled(source: str, reason: str) -> Dict[str, Any]:
        return {"settled": False, "outcome": None, "is_real": False,
                "source": f"simmer_{source}", "reason": reason,
                "venue_id": "simmer"}

    # ------------------------------------------------------------------
    # account
    # ------------------------------------------------------------------
    async def get_portfolio(self) -> Dict[str, Any]:
        """
        The $SIM account, when a key is saved.

        The venue's positions ARE the record of what this adapter opened here,
        which makes this read the reconciliation for the paper lane: the balance
        is reported in $SIM and the currency is named so nothing about it can be
        mistaken for dollars the operator holds.
        """
        client, error = self._client()
        if client is None:
            return {"venue_id": "simmer", "available": False, "balance": None,
                    "positions": [], "paper": True, "virtual": True,
                    "reason": error}
        try:
            rows = list(client.get_positions(venue=SIMMER_PAPER_VENUE) or [])
        except Exception as e:  # noqa: BLE001
            return {"venue_id": "simmer", "available": False, "balance": None,
                    "positions": [], "paper": True, "virtual": True,
                    "reason": f"could not read positions: {type(e).__name__}: {e}"}
        positions: List[Dict[str, Any]] = []
        balance = None
        for row in rows:
            if balance is None:
                value = _field(row, "sim_balance")
                if isinstance(value, (int, float)):
                    balance = float(value)
            positions.append({
                "market_id": _field(row, "market_id"),
                "question": _field(row, "question"),
                "shares_yes": _field(row, "shares_yes"),
                "shares_no": _field(row, "shares_no"),
                "current_value": _field(row, "current_value"),
                "pnl": _field(row, "pnl"),
                "status": _field(row, "status"),
            })
        return {"venue_id": "simmer", "available": True, "balance": balance,
                "currency": "SIM (virtual)", "paper": True, "virtual": True,
                "currency_note": ("$SIM is the venue's own virtual currency: it "
                                  "cannot be withdrawn, deposited or converted, "
                                  "so it is not capital"),
                "positions": positions, "position_count": len(positions),
                "source": "simmer_sdk"}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _market_key(market_id: Any) -> str:
        """The venue's own market id, from `simmer-<id>` or a bare id."""
        raw = _text(market_id).strip()
        if not raw:
            return ""
        if raw.lower().startswith(SIMMER_ID_PREFIX + "-"):
            return raw[len(SIMMER_ID_PREFIX) + 1:]
        return raw

    @staticmethod
    def _probability(value: Any) -> Optional[float]:
        """A probability in (0, 1), or None. Never a fallback."""
        if isinstance(value, bool) or value is None:
            return None
        try:
            probability = float(value)
        except (TypeError, ValueError):
            return None
        if not (0.0 < probability < 1.0):
            return None
        return round(probability, 6)

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """`resolves_at`, an ISO timestamp."""
        if value is None:
            return None
        text = _text(value).strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            logger.debug(f"Simmer: unparseable resolves_at {value!r}")
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
