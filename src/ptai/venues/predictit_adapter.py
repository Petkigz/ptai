"""
PredictIt Adapter - the venue's own two-sided quotes, and its own resolutions.

API (public, no account, ~1 request/second, non-commercial use):
    GET https://www.predictit.org/api/marketdata/all/
    GET https://www.predictit.org/api/marketdata/markets/{market_id}

WHAT THIS ADAPTER USED TO DO, AND WHY IT WAS WRONG. `get_orderbook` answered
`bid = last_price - 0.02, ask = last_price + 0.02` with a fabricated depth and
the tell in its own `source` field ("predictit_mock"). That is a spread nobody
quoted: the venue publishes a real bid and a real ask per contract
(`bestSellYesCost`, `bestBuyYesCost`) and the mock numbers were used wherever
they fit, including the cost model. Meanwhile the adapter also invented
liquidity (`liquidity=1000` on every market), invented an `end_date` by omitting
it entirely, discarded every contract but the first (so a multi-outcome market
became a fake two-way market on the first name), and defaulted a missing price
to 0.50. It could not report a resolution at all, so the executor - which since
V68 refuses to open a position on a venue that cannot say how its market ended -
would not have paper-traded it either way.

WHAT IT DOES NOW, in the venue's own numbers:

  * ONE MARKET PER CONTRACT. A PredictIt market is a field of contracts
    ("Republican", "Democratic", ...); each contract is its own tradeable
    instrument with its own id, quote and end date. The market name and the
    contract name are both carried, so the question says which instrument it is.
  * THE BOOK IS THE PUBLISHED QUOTE. `bestBuyYesCost` is the cost to buy one YES
    share - the ask; `bestSellYesCost` is the bid. Neither side's SIZE is
    published anywhere in the API, so the ladder is one share a side and the book
    says so (`size_basis`). Depth is `None`, not a number.
  * NOTHING IS INVENTED FOR VOLUME. PredictIt publishes no volume and no
    liquidity, so both are 0.0 and the market record carries
    `volume_basis: "not_published"` - which the scan honours by not applying its
    volume/liquidity floors to a venue that never publishes them, instead of the
    old `liquidity=1000` placeholder that made the floors meaningless.
  * THE SETTLEMENT READ IS THE VENUE'S OWN RECORD. A closed contract whose
    published final prices agree on a decided side ($1.00 or $0.00) settles to
    YES or NO. Anything ambiguous is refused with the numbers that were read:
    this adapter never converts an ambiguous price into an outcome.
  * NO ORDER PATH, AND IT SAYS SO. PredictIt has no trading API - orders go
    through the website - so `real_order_path` stays False and `place_order`
    refuses real money while simulating a paper fill at the venue's own quote.

FEES. The venue charges 10% of the profit on a position CLOSED before
settlement, and 5% on profits withdrawn. It charges nothing to open, and a
contract held to settlement redeems at $1.00. The paper lane holds to
settlement, so the entry-time fee this adapter declares is 0.0 - declaring 10%
on the notional (which is what `fee_taker_pct` used to be set to) would have
charged a cost the venue never charges on these trades.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from loguru import logger
import requests

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity, AdapterCapability
from ..markets.base import Market, Token, MarketSource, DataMode


PREDICTIT_LIST_URL = "https://www.predictit.org/api/marketdata/all/"
PREDICTIT_MARKET_URL = "https://www.predictit.org/api/marketdata/markets/{market_id}"

# The market id PTAI stores carries both ids: `predictit-<market>-<contract>`.
PREDICTIT_ID_PREFIX = "predictit"

# The venue publishes the price to buy ONE share, not the size behind it. One
# share is therefore the whole ladder - the venue's own unit, and the largest
# quantity its published quote actually justifies.
PUBLISHED_QUANTITY_SHARES = 1.0

# How long a quote read during discovery may be reused as this market's book.
# The venue updates its feed about once a minute; past this the adapter asks for
# the market again rather than pricing against a stale touch.
QUOTE_MAX_AGE_SECONDS = 600

# A settlement price is a decided one. Both of the venue's published final
# numbers must sit on the same side of this before an outcome is reported: a
# closed contract that stopped trading at 0.62 says nothing about who won.
DECIDED_PRICE_YES = 0.99
DECIDED_PRICE_NO = 0.01

# The widest two-sided quote this adapter will call executable. It is the same
# bar the rest of the system trades on, named here so the venue's own row can
# carry the reason when a quote is too wide to use.
MAX_TRUSTED_SPREAD = 0.10


class PredictItAdapter(MarketAdapter):
    def __init__(self):
        super().__init__(venue_id="predictit", venue_type=VenueType.PREDICTION)
        self.last_error: str = ""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PTAI/1.0 (personal research)",
        })
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            # The venue publishes a real two-sided quote per contract, so this
            # adapter can price a fill. There is still no submission path: the
            # order book itself is web-only.
            supports_orderbook=True,
            supports_trading=False,   # read-only for PTAI; orders are web-only
            supports_portfolio=False,  # the public API exposes no account
            supports_history=False,   # price history is a CSV download
            real_order_path=False,
            # The venue's schedule: 10% of profit on a position closed before
            # settlement, 5% on profits withdrawn, nothing to open, $1.00 per
            # share at settlement. A hold-to-settlement paper trade pays none of
            # that at entry, and the cost model charges a fee at entry.
            fee_taker_pct=0.0,
            fee_maker_pct=0.0,
            min_order_usd=1.0,
            implementation_status="live",
        )
        self.api_url = PREDICTIT_LIST_URL
        # Quotes read from the venue's own list payload, keyed by contract id.
        # The list call returns every contract's best buy and sell price, so the
        # scan does not have to make a second request per market to price one -
        # and the venue's rate limit (~1/second) is not put at risk by a cycle
        # that reads two hundred books.
        self._quotes: Dict[str, Dict[str, Any]] = {}
        self._quotes_at: Optional[float] = None

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        """
        PredictIt is US-only to TRADE; its data is public everywhere.

        Returning ELIGIBLE for every non-US country used to say this operator
        could open an account there. It cannot: PredictIt requires verified US
        residency, so trading needs that verification - for Uganda and for every
        other country outside the US - while reading is open to anyone. The
        status only ever affects what the venue panel promises, because this
        adapter has no way to submit an order at all.
        """
        cc = (country_code or "").upper()
        if cc == "US":
            return EligibilityStatus.ELIGIBLE
        return EligibilityStatus.REQUIRES_VERIFICATION

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    async def discover_markets(self, target_count: int = 100,
                               filters: Dict = None) -> List[Market]:
        """
        Every OPEN contract the venue lists, one Market each.

        A closed market in the feed is skipped rather than returned: the scan
        refuses a closed market, and its resolution is read by
        `get_settlement` from the venue's own record instead.
        """
        filters = filters or {}
        payload, error = self._get_json(self.api_url)
        if payload is None:
            self.last_error = error
            logger.warning(f"PredictIt: {error}; no markets returned")
            return []

        raw_markets = payload.get("markets") if isinstance(payload, dict) else None
        if not isinstance(raw_markets, list):
            self.last_error = ("the venue's payload carried no 'markets' list "
                               f"(keys: {sorted(list(payload))[:6]})")
            logger.warning(f"PredictIt: {self.last_error}")
            return []

        markets: List[Market] = []
        quotes: Dict[str, Dict[str, Any]] = {}
        skipped_closed = 0
        skipped_unpriced = 0
        for raw_market in raw_markets:
            if not isinstance(raw_market, dict):
                continue
            market_status = str(raw_market.get("status") or "Open")
            raw_contracts = raw_market.get("contracts")
            if not isinstance(raw_contracts, list) or not raw_contracts:
                continue
            multi = len(raw_contracts) > 1
            for contract in raw_contracts:
                if not isinstance(contract, dict):
                    continue
                contract_status = str(contract.get("status") or market_status)
                if (market_status.lower() != "open"
                        or contract_status.lower() != "open"):
                    skipped_closed += 1
                    continue
                market = self._build_market(raw_market, contract, multi=multi)
                if market is None:
                    skipped_unpriced += 1
                    continue
                markets.append(market)
                contract_id = str(contract.get("id"))
                quotes[contract_id] = self._quote_from_contract(
                    contract, market_id=str(raw_market.get("id")),
                    venue_timestamp=raw_market.get("timeStamp"))
            if len(markets) >= target_count:
                break

        if not markets:
            self.last_error = ("PredictIt returned no open, priced contracts "
                               f"({skipped_closed} closed, {skipped_unpriced} "
                               f"with no usable price)")
            logger.warning(f"PredictIt: {self.last_error}")
            return []

        # The cache is replaced, never merged: a contract the venue stopped
        # listing must not keep being priced from this cycle's earlier read.
        self._quotes = quotes
        self._quotes_at = datetime.now(timezone.utc).timestamp()
        self.last_error = ""
        logger.info(
            f"PredictIt discovered {len(markets)} open contract(s) from "
            f"{len(raw_markets)} market(s)"
            + (f"; {skipped_closed} closed, {skipped_unpriced} with no publishable "
               f"price were skipped" if (skipped_closed or skipped_unpriced) else ""))
        return markets[:target_count]

    def _build_market(self, raw_market: Dict[str, Any], contract: Dict[str, Any],
                      multi: bool) -> Optional[Market]:
        """
        One contract, as a Market. Returns None when the venue published no
        usable price for it - a contract with no quote is not a 0.50 market.
        """
        contract_id = contract.get("id")
        market_id = raw_market.get("id")
        if contract_id is None or market_id is None:
            return None
        market_name = str(raw_market.get("name") or raw_market.get("shortName") or "")
        contract_name = str(contract.get("name") or contract.get("shortName") or "")
        # A single-contract market IS the question; in a multi-contract market
        # the contract name is the instrument, and dropping it would price
        # "Republican" as though it were the whole market.
        question = market_name if not multi else f"{market_name} - {contract_name}"
        # THE RECORD'S PRICE IS THE VENUE'S OWN TWO-SIDED QUOTE, not the last
        # trade: a contract last traded at 0.67 yesterday whose book now reads
        # 0.51/0.49 is a 0.50 market, and pricing it off the last trade is how a
        # stale print becomes this cycle's cost. Both sides must be published -
        # a one-sided contract cannot be priced at all (its book is a refusal),
        # so it is skipped here rather than discovered and refused later.
        ask = self._price(contract.get("bestBuyYesCost"))
        bid = self._price(contract.get("bestSellYesCost"))
        if ask is None or bid is None:
            return None
        yes_price = round((ask + bid) / 2.0, 6)
        end_date = self._parse_date(contract.get("dateEnd"))
        return Market(
            id=f"{PREDICTIT_ID_PREFIX}-{market_id}-{contract_id}",
            source=MarketSource.PREDICTIT,
            question=question[:200],
            description=str(contract.get("shortName") or contract_name or "")[:200],
            outcomes=["YES", "NO"],
            outcome_prices=[yes_price, round(1.0 - yes_price, 6)],
            tokens=[Token(token_id=str(contract_id), outcome="YES",
                          price=yes_price)],
            # THE VENUE PUBLISHES NEITHER. Volume and liquidity are used for
            # ranking and for the scan's floors, so a placeholder here (the old
            # `liquidity=1000`) is not a neutral default: it is a fabricated
            # number deciding which markets get model time. `volume_basis` is
            # what tells the scan the difference between "no trading" and "not
            # published", and the floors are only applied when a venue actually
            # publishes the figures.
            volume=0.0,
            volume_24h=0.0,
            liquidity=0.0,
            end_date=end_date,
            active=True,
            closed=False,
            slug=str(market_id),
            event_slug=str(raw_market.get("shortName") or "")[:50],
            raw={
                "venue": "predictit",
                "api": "public",
                "market_id": str(market_id),
                "contract_id": str(contract_id),
                "contract_name": contract_name,
                "url": raw_market.get("url"),
                "venue_timestamp": raw_market.get("timeStamp"),
                "volume_basis": "not_published",
                "liquidity_basis": "not_published",
                "depth_basis": "not_published",
                "original": {
                    "lastTradePrice": contract.get("lastTradePrice"),
                    "bestBuyYesCost": contract.get("bestBuyYesCost"),
                    "bestSellYesCost": contract.get("bestSellYesCost"),
                    "bestBuyNoCost": contract.get("bestBuyNoCost"),
                    "bestSellNoCost": contract.get("bestSellNoCost"),
                    "dateEnd": contract.get("dateEnd"),
                },
            },
            venue_id="predictit",
            venue_type="prediction",
            data_mode=DataMode.LIVE,
            data_source="predictit_api",
            is_mock=False,
        )

    # ------------------------------------------------------------------
    # the book
    # ------------------------------------------------------------------
    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        """
        The venue's own published quote for this contract.

        Two real prices and no invented ones: the ask is `bestBuyYesCost` (the
        cost to buy one YES share) and the bid is `bestSellYesCost`. Sizes are
        not published anywhere in the API, so each side carries ONE share - the
        venue's own unit - and the book states that basis instead of implying a
        depth nobody quoted.
        """
        market_id, contract_id = self._split_market_id(market)
        if contract_id is None:
            return self._refusal(
                market, "no contract id on this market record, so there is "
                        "nothing to ask the venue for")

        quote = self._cached_quote(contract_id)
        if quote is None:
            quote = self._fetch_quote(market_id, contract_id)
        if quote is None:
            return self._refusal(
                market, self.last_error or
                f"the venue published no quote for contract {contract_id}")
        return self._book(market, contract_id, quote)

    def _book(self, market: Market, contract_id: str,
              quote: Dict[str, Any]) -> Dict[str, Any]:
        yes_ask = self._price(quote.get("yes_ask"))
        yes_bid = self._price(quote.get("yes_bid"))
        if yes_ask is None or yes_bid is None:
            return self._refusal(
                market, "the venue published only one side of this contract's "
                        "quote, and a one-sided quote is not a price")
        if yes_ask < yes_bid:
            return self._refusal(
                market, f"the venue's own quote is crossed (bid {yes_bid:.2f} > "
                        f"ask {yes_ask:.2f}); nothing is priced from it")

        # The NO side. Buying NO at p is the venue's own `bestBuyNoCost`; where
        # the venue publishes no NO price, the equivalence the system already
        # uses for a binary book applies - buying NO is selling YES at 1 - p -
        # and which of the two was used is recorded rather than blurred.
        no_ask = self._price(quote.get("no_ask"))
        no_ask_source = "venue_bestBuyNoCost"
        if no_ask is None:
            no_ask = round(1.0 - yes_bid, 6)
            no_ask_source = "1 - the venue's yes bid (binary equivalence)"
        no_bid = self._price(quote.get("no_bid"))
        no_bid_source = "venue_bestSellNoCost"
        if no_bid is None:
            no_bid = round(1.0 - yes_ask, 6)
            no_bid_source = "1 - the venue's yes ask (binary equivalence)"

        spread = round(yes_ask - yes_bid, 6)
        mid = (yes_ask + yes_bid) / 2.0
        age = self._quote_age_seconds(quote)
        # ONE SHAPE NOTE, because it decides which price a fill pays: `bids` and
        # `asks` here are the YES share's book (buying YES takes the ask). A NO
        # buy consumes a DIFFERENT ladder - the venue's own no price - and
        # `PaperBroker` reads `bids` for a non-YES side, so `place_order` builds
        # the side's own book rather than letting the broker read this one
        # backwards.
        return {
            "market_id": market.id,
            "venue_id": "predictit",
            "contract_id": contract_id,
            "bids": [{"price": yes_bid, "size": PUBLISHED_QUANTITY_SHARES}],
            "asks": [{"price": yes_ask, "size": PUBLISHED_QUANTITY_SHARES}],
            "bid": yes_bid,
            "ask": yes_ask,
            "spread": spread,
            "spread_pct": round(spread / mid, 6) if mid > 0 else None,
            "bid_size": PUBLISHED_QUANTITY_SHARES,
            "ask_size": PUBLISHED_QUANTITY_SHARES,
            # NOT PUBLISHED. `execution_quality_from_book` scores on the spread
            # alone when depth is None, which is the honest reading: a tight
            # quote with unknown size is a good price and an unmeasured fill.
            "depth": None,
            "depth_basis": "not_published",
            "size_basis": ("the venue publishes the price to buy ONE share, not "
                           "the size behind it; the ladder is that one share"),
            "liquidity": market.liquidity,
            "volume_24h": market.volume_24h,
            "liquidity_basis": "not_published",
            "source": quote.get("source") or "predictit_published_quote",
            "is_real": True,
            "is_mock": False,
            "validated": True,
            "validation": {
                "identity": (f"contract {contract_id} is the contract this "
                             f"market record names"),
                "two_sided": ("the venue published both a buy and a sell price "
                              "for this contract"),
                "ordered": "the ask is at or above the bid",
            },
            "executable": spread <= MAX_TRUSTED_SPREAD,
            "executable_price_yes": yes_ask,
            "executable_price_no": no_ask,
            "executable_price": yes_ask,
            "executable_price_note": ("a YES share costs the venue's yes ask; a "
                                      "NO share costs executable_price_no"),
            "executable_price_no_source": no_ask_source,
            "executable_price_no_bid": no_bid,
            "executable_price_no_bid_source": no_bid_source,
            "quote_age_seconds": age,
            "venue_timestamp": quote.get("venue_timestamp"),
            "warning": (None if spread <= MAX_TRUSTED_SPREAD else
                        f"spread {spread:.1%} is wider than the "
                        f"{MAX_TRUSTED_SPREAD:.0%} this system will trade"),
            "reasoning": ("the venue's own best buy and best sell prices for this "
                          "contract; no depth is published, so the ladder is the "
                          "one share the price is quoted for"),
        }

    def _refusal(self, market: Market, why: str) -> Dict[str, Any]:
        """
        No book, and no numbers dressed up as one. `is_real` False and
        `validated` False keep this out of the cost model and the shortlist.
        """
        return {
            "market_id": getattr(market, "id", ""),
            "venue_id": "predictit",
            "bids": [], "asks": [],
            "bid": None, "ask": None, "spread": None, "spread_pct": None,
            "depth": None, "depth_basis": "not_published",
            "liquidity": getattr(market, "liquidity", 0.0),
            "volume_24h": getattr(market, "volume_24h", 0.0),
            "source": "predictit_no_quote",
            "is_real": False,
            "is_mock": False,
            "validated": False,
            "executable": False,
            "executable_price": None,
            "warning": why,
            "reasoning": ("the venue published no usable quote for this contract, "
                          "so no cost or edge is computed from it"),
        }

    # ------------------------------------------------------------------
    # settlement: the venue's own record
    # ------------------------------------------------------------------
    async def get_settlement(self, market_id: str) -> Dict[str, Any]:
        """
        How this contract ended, from the venue's own published record.

        PredictIt has no results endpoint, but for a contract the venue has
        closed it publishes the final traded and closing prices - and a settled
        contract's price IS its outcome ($1.00 a share for the side that won,
        $0.00 for the side that lost). The rule here is deliberately strict:

          * the market or the contract must be closed (not "Open"), AND
          * every final price the venue published for that contract ($1.00 or
            $0.00) must agree on the same side, at the decided price.

        Anything else is refused with the numbers that were read. A contract
        that stopped trading at 0.62 is a closed contract with an undecided
        price, and calling it either way would be an invented outcome - the one
        thing a settlement read must never do.
        """
        market_id_part, contract_id = self._split_market_id(market_id)
        if market_id_part is None:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "no_market_id",
                    "reason": "empty PredictIt market id"}
        payload, error = self._get_json(
            PREDICTIT_MARKET_URL.format(market_id=market_id_part))
        if payload is None:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "predictit_market_error", "reason": error}
        raw_market = self._first_market(payload)
        if raw_market is None:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "predictit_market_unreadable",
                    "reason": (f"the venue answered for market "
                               f"{market_id_part} with no market record")}

        contracts = raw_market.get("contracts")
        contract = None
        if isinstance(contracts, list):
            if contract_id is not None:
                contract = next(
                    (c for c in contracts
                     if isinstance(c, dict) and str(c.get("id")) == str(contract_id)),
                    None)
                if contract is None:
                    return {"settled": False, "outcome": None, "is_real": True,
                            "source": "predictit_contract_missing",
                            "reason": (f"the venue's record for market "
                                       f"{market_id_part} does not list contract "
                                       f"{contract_id}")}
            elif len(contracts) == 1 and isinstance(contracts[0], dict):
                contract = contracts[0]
        if contract is None:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "predictit_contract_missing",
                    "reason": (f"market {market_id_part} carries several contracts "
                               f"and the id names none of them")}

        market_status = str(raw_market.get("status") or "")
        contract_status = str(contract.get("status") or market_status)
        if (market_status.lower() == "open"
                and contract_status.lower() == "open"):
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "predictit_open", "market_id": str(market_id_part),
                    "reason": "the venue still lists this market as open"}

        finals = [self._final_price(contract.get(field))
                  for field in ("lastClosePrice", "lastTradePrice")]
        finals = [value for value in finals if value is not None]
        if not finals:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "predictit_no_final_price",
                    "market_id": str(market_id_part),
                    "reason": ("the venue closed this contract but published no "
                               "final price for it, so its outcome is not known "
                               "from the venue's record")}
        if all(value >= DECIDED_PRICE_YES for value in finals):
            return {"settled": True, "outcome": 1.0, "is_real": True,
                    "source": "predictit_closed_contract",
                    "market_id": str(market_id_part),
                    "contract_id": str(contract.get("id")),
                    "final_prices": finals, "status": contract_status,
                    "reason": ("the venue's own final prices for this contract are "
                               f"{', '.join(f'{v:.2f}' for v in finals)} - the "
                               f"named outcome won")}
        if all(value <= DECIDED_PRICE_NO for value in finals):
            return {"settled": True, "outcome": 0.0, "is_real": True,
                    "source": "predictit_closed_contract",
                    "market_id": str(market_id_part),
                    "contract_id": str(contract.get("id")),
                    "final_prices": finals, "status": contract_status,
                    "reason": ("the venue's own final prices for this contract are "
                               f"{', '.join(f'{v:.2f}' for v in finals)} - the "
                               f"named outcome lost")}
        return {"settled": False, "outcome": None, "is_real": True,
                "source": "predictit_undecided",
                "market_id": str(market_id_part),
                "final_prices": finals, "status": contract_status,
                "reason": ("the venue closed this contract without a decided price "
                           f"({', '.join(f'{v:.2f}' for v in finals)}). No outcome "
                           f"is invented from an ambiguous final price.")}

    # ------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------
    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float,
                          max_price: float) -> Dict[str, Any]:
        """
        A paper fill at the venue's own quote, or a refusal. Never an order.

        There is no order API for this venue, so nothing here can submit. In
        dry run the fill walks the published ladder - one share at the quoted
        price - and reports exactly that, including the fact that only one
        share is backed by anything the venue published.
        """
        market = opportunity.market
        side = str(getattr(opportunity, "side", "") or "").upper()
        if not self.dry_run:
            # Live mode: the standard refusal, which names the REAL reason
            # (no submission path) rather than inviting a switch that would
            # change nothing.
            return self.real_order_refusal(max_spend_usd, max_price, side,
                                           getattr(market, "id", ""))
        if side not in ("YES", "NO"):
            return self._order_refusal(market, f"this venue speaks YES/NO; got "
                                               f"{getattr(opportunity, 'side', '')!r}")
        if max_spend_usd <= 0 or not (0 < float(max_price) < 1):
            return self._order_refusal(
                market, f"invalid order guard (spend {max_spend_usd}, "
                        f"limit {max_price})")

        book = await self.get_orderbook(market)
        if not book.get("is_real") or not book.get("validated"):
            return self._order_refusal(
                market, book.get("warning") or "no usable quote for this contract")
        price = (book.get("executable_price_yes") if side == "YES"
                 else book.get("executable_price_no"))
        price = self._price(price)
        if price is None:
            return self._order_refusal(
                market, f"the venue published no price to buy {side} on this "
                        f"contract")

        from ..markets.mechanics import MarketMechanics  # noqa: WPS433
        from ..execution.paper_broker import PaperBroker  # noqa: WPS433

        mechanics = self.get_mechanics(opportunity)
        limit = mechanics.round_price(float(max_price), side)
        if limit <= 0 or limit >= 1:
            return self._order_refusal(
                market, f"limit {max_price} is outside the venue's accepted range")
        if side == "YES" and price > limit + 1e-9:
            return self._order_refusal(
                market, f"the venue's ask {price:.2f} is above the limit "
                        f"{limit:.2f}; the order would rest and never fill")
        if side == "NO" and price > limit + 1e-9:
            return self._order_refusal(
                market, f"the venue's no price {price:.2f} is above the limit "
                        f"{limit:.2f}; the order would rest and never fill")

        requested_shares = mechanics.shares_for_usd(float(max_spend_usd), limit, side)
        ok, why = mechanics.validate_order(limit, requested_shares)
        if not ok:
            logger.warning(f"Paper order refused for {market.id}: {why}")
            return self._order_refusal(market, why, mechanics=mechanics)

        # THE LADDER FOR THE SIDE BEING BOUGHT. Buying YES consumes the venue's
        # yes ask; buying NO consumes its no price, and `PaperBroker` reads the
        # `bids` list for any side that is not YES - so a NO buy handed the
        # YES-space book would fill at the YES BID (0.66) and report a price
        # nobody would have paid for a 0.34 share. The side's own one-share
        # ladder is built here instead.
        if side == "YES":
            side_book = {
                "bids": [{"price": book.get("bid"),
                          "size": PUBLISHED_QUANTITY_SHARES}],
                "asks": [{"price": price, "size": PUBLISHED_QUANTITY_SHARES}],
            }
        else:
            side_book = {
                "bids": [{"price": price, "size": PUBLISHED_QUANTITY_SHARES}],
                "asks": [{"price": price, "size": PUBLISHED_QUANTITY_SHARES}],
            }
        # NO LIMIT IS HANDED TO THE BROKER. `PaperBroker` reads the `bids` list
        # for any side that is not YES and marketable as `limit <= best`, which
        # is a SELL rule: a NO order carrying a normal max-price cap (limit
        # above the price) would be reported as resting when it crosses
        # perfectly well. The cap has already been applied two checks above,
        # where the only question is whether the price fits inside it.
        broker = PaperBroker(mechanics=mechanics, taker_fee_rate=0.0)
        fill = broker.simulate(side_book, side, float(max_spend_usd),
                              limit_price=None, mechanics=mechanics,
                              book_source="orderbook")
        result = {
            "status": "paper",
            "is_real": False,
            "simulated": True,
            "venue_id": "predictit",
            "market_id": getattr(market, "id", ""),
            "contract_id": book.get("contract_id"),
            "side": side,
            "requested_price": float(max_price),
            "limit_price": limit,
            "price": fill.avg_price or price,
            "size": fill.filled_shares,
            "simulated_filled_usd": fill.filled_usd,
            "filled_usd": fill.filled_usd,
            "filled_price": fill.avg_price,
            "fees_usd": fill.fee_usd,
            "paper_fill": fill.to_dict(),
            "mechanics": mechanics.to_dict(),
            # WHAT THE FILL ASSUMES, in one line, so a small position is never
            # read as a small edge.
            "size_basis": book.get("size_basis"),
            "quote_age_seconds": book.get("quote_age_seconds"),
            "reason": (fill.reason or
                       f"filled at the venue's own published {side} price; the "
                       f"venue publishes no size, so the ladder is one share"),
        }
        if fill.unfilled_usd > 1e-9:
            result["size_matched"] = fill.filled_shares
            result["original_size"] = requested_shares
            result["resting_usd"] = fill.unfilled_usd
        return result

    def _order_refusal(self, market: Market, why: str,
                       mechanics: Any = None) -> Dict[str, Any]:
        """A refusal, not a fill. Nothing was opened."""
        out = {
            "status": "refused",
            "success": False,
            "is_real": False,
            "venue_id": "predictit",
            "market_id": getattr(market, "id", ""),
            "reason": why,
            "message": (f"Refusal, not a fill. No PredictIt position was opened: "
                        f"{why}"),
        }
        if mechanics is not None:
            out["mechanics"] = mechanics.to_dict()
        return out

    def get_mechanics(self, opportunity=None, token_id: Optional[str] = None):
        """
        The venue's order rules, as far as the venue publishes them.

        PredictIt quotes in whole cents, sells in whole shares, charges nothing
        to open, and settles at $1.00 - all published rules, but not read
        per-market, so this object says `is_real: False` and carries the reason
        rather than pretending a per-market read happened. The account's own $1
        floor is applied to the order REQUEST by the executor, not re-derived
        here from a rounded share count.
        """
        from ..markets.mechanics import MarketMechanics  # noqa: WPS433

        return MarketMechanics(
            tick_size="0.01",
            min_order_size=1.0,
            min_order_notional_usd=0.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            source="predictit_documented_rules",
            is_real=False,
            warnings=[
                "PredictIt's penny grid, one-share minimum and zero entry fee "
                "come from the venue's published rules, not from a per-market "
                "read; the venue publishes no size with its quotes",
            ],
        )

    # ------------------------------------------------------------------
    # account
    # ------------------------------------------------------------------
    async def get_portfolio(self) -> Dict[str, Any]:
        return {"venue_id": "predictit", "available": False, "balance": None,
                "positions": [],
                "reason": ("the public PredictIt API exposes no account: balances "
                           "and positions cannot be read, and there is no "
                           "submission path")}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_json(self, url: str):
        """(payload, error). One place where a venue read can fail."""
        try:
            resp = self.session.get(url, timeout=10)
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
        if resp.status_code != 200:
            return None, f"the venue returned HTTP {resp.status_code}"
        try:
            return resp.json(), ""
        except Exception as e:  # noqa: BLE001
            return None, f"the venue's answer was not JSON ({type(e).__name__}: {e})"

    @staticmethod
    def _first_market(payload: Any) -> Optional[Dict[str, Any]]:
        """
        The market record inside a per-market answer.

        The endpoint answers with the market object; some deployments wrap it in
        a one-element list. Both are accepted, and neither is guessed at.
        """
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if isinstance(payload, dict):
            if isinstance(payload.get("markets"), list) and payload["markets"]:
                first = payload["markets"][0]
                return first if isinstance(first, dict) else None
            return payload
        return None

    def _quote_from_contract(self, contract: Dict[str, Any], market_id: str,
                             venue_timestamp: Any = None) -> Dict[str, Any]:
        return {
            "yes_ask": self._price(contract.get("bestBuyYesCost")),
            "yes_bid": self._price(contract.get("bestSellYesCost")),
            "no_ask": self._price(contract.get("bestBuyNoCost")),
            "no_bid": self._price(contract.get("bestSellNoCost")),
            "last_trade": self._price(contract.get("lastTradePrice")),
            "market_id": str(market_id),
            "venue_timestamp": venue_timestamp,
            "captured_at": datetime.now(timezone.utc).timestamp(),
            "source": "predictit_published_quote",
        }

    def _cached_quote(self, contract_id: str) -> Optional[Dict[str, Any]]:
        quote = self._quotes.get(str(contract_id))
        if not quote:
            return None
        age = self._quote_age_seconds(quote)
        if age is None or age > QUOTE_MAX_AGE_SECONDS:
            return None
        return quote

    @staticmethod
    def _quote_age_seconds(quote: Dict[str, Any]) -> Optional[float]:
        captured = quote.get("captured_at")
        if not isinstance(captured, (int, float)):
            return None
        return round(datetime.now(timezone.utc).timestamp() - float(captured), 1)

    def _fetch_quote(self, market_id: Optional[str],
                     contract_id: str) -> Optional[Dict[str, Any]]:
        """
        One per-market request, for a contract discovery did not hand us.

        A book read that is not in the cycle's own discovery payload has to ask
        the venue - and the answer is cached for the same short window, so a
        scan that prices ten contracts from one market does not ask ten times.
        """
        if market_id is None:
            self.last_error = "no market id on this record"
            return None
        payload, error = self._get_json(PREDICTIT_MARKET_URL.format(market_id=market_id))
        if payload is None:
            self.last_error = error
            return None
        raw_market = self._first_market(payload)
        contracts = (raw_market or {}).get("contracts")
        if not isinstance(contracts, list):
            self.last_error = f"market {market_id} carried no contracts list"
            return None
        for contract in contracts:
            if (isinstance(contract, dict)
                    and str(contract.get("id")) == str(contract_id)):
                quote = self._quote_from_contract(
                    contract, market_id=str(market_id),
                    venue_timestamp=(raw_market or {}).get("timeStamp"))
                self._quotes[str(contract_id)] = quote
                return quote
        self.last_error = (f"market {market_id} does not list contract "
                           f"{contract_id}")
        return None

    @staticmethod
    def _split_market_id(market: Any):
        """
        (market_id, contract_id) from `predictit-<market>-<contract>`.

        Accepts a Market or the raw id string. Returns (None, None) when the id
        is not a PredictIt one, and (market_id, None) when only one part is
        present - the caller decides what that is worth rather than being handed
        a guess.
        """
        raw = market if isinstance(market, str) else getattr(market, "id", "")
        raw = str(raw or "").strip()
        if not raw.lower().startswith(PREDICTIT_ID_PREFIX + "-"):
            return None, None
        parts = raw.split("-")
        if len(parts) < 3:
            return (parts[1], None) if len(parts) == 2 else (None, None)
        return parts[1], parts[2]

    @staticmethod
    def _final_price(value: Any) -> Optional[float]:
        """
        A FINAL price, which may legitimately be 0.00 or 1.00.

        `_price` refuses 0 and 1 because neither is a quotable PRICE - but a
        settled contract's price IS its outcome, and both ends of the range are
        decided ones. Values above 1 are read as cents (the venue reports both
        shapes); everything else that is not a number is None.
        """
        if isinstance(value, bool) or value is None:
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if price > 1.0:
            price = price / 100.0
        if not (0.0 <= price <= 1.0):
            return None
        return round(price, 6)

    @staticmethod
    def _price(value: Any) -> Optional[float]:
        """
        A price in (0, 1), or None. Never a fallback.

        The venue's API reports dollars (0.52). Values above 1 that are still
        within the venue's cent scale are read as cents, because both shapes
        appear in the wild; anything else is not a price and is refused rather
        than clipped into one (a missing price used to become 0.50).
        """
        if isinstance(value, bool) or value is None:
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if 1.0 < price <= 100.0:
            price = price / 100.0
        if not (0.0 < price < 1.0):
            return None
        return round(price, 6)

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """`dateEnd`, which the venue spells 'N/A' when it has none."""
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.upper() in ("N/A", "NA", "NONE"):
            return None
        text = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%m/%d/%Y %I:%M:%S %p")
            except ValueError:
                logger.debug(f"PredictIt: unparseable dateEnd {value!r}")
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
