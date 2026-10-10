"""
Betdaq - a betting exchange with a published Python SDK, and a venue that
will take an order with no money behind it.

Betdaq (api.betdaq.com, API v2.0) is a sports betting exchange: markets with
runners, decimal-odds ladders on both sides, matched orders, and a settlement
that names a result for every runner. A Python wrapper for that API ships on
PyPI as `betdaq` (the package this adapter imports), and - this is the part
that matters here - the venue runs PLAY markets alongside its real-money ones:
`WantPlayMarkets` is a first-class flag in its own market-data calls, and every
market row carries `is_play_market`. Play markets are the venue's own
simulation, settled in play money, exactly like Simmer's synthetic `sim` venue
in spirit: real questions, real orders at the venue, fills from the venue's own
matching engine, and a result the venue publishes.

WHY THAT IS WORTH AN ADAPTER. It is the second venue PTAI can submit an order
to with nothing but a login: the markets are real questions, the orders are
real orders at the venue, the fills come back from the venue, and the venue
resolves the market - so a paper position here closes on a real outcome and
counts toward the record. Play money cannot be withdrawn, deposited or
converted, so no money can move.

WHAT THIS ADAPTER WILL NOT DO:

  * It will not submit a REAL-money order. The adapter is pinned to play
    markets at discovery (`WantPlayMarkets=True`, and only rows the venue marks
    `is_play_market` are published), so every stake it ever places is play
    money. `supports_trading` and `real_order_path` are False, live mode gets a
    refusal that says so, and the venue's real-money markets are never read.
  * It will not invent a book. The book is the venue's own ladders
    (`batb`/`batl`, best-first, [price, stake] levels) converted to the
    probability space the rest of PTAI prices in: decimal odds become 1/odds,
    and a backer's stake becomes stake x odds contracts. A market the venue
    does not return prices for gets an empty book that says so.
  * It will not guess a resolution. Betdaq publishes results per RUNNER through
    its selection-changes feed (`SettlementResultString`), not as a field on
    the market - the market row only says SETTLED. This adapter polls that feed
    and maps the venue's own result string onto the winning runner; a settled
    market whose feed names no winner is refused, not interpreted. The
    published outcome order is the venue's own runner order (the same
    first-runner rule the Betfair adapter enforces), so "did YES win" is exact.
  * It will not pretend the SDK is installed when it is not. The `betdaq`
    package is an optional dependency: `pip install betdaq`. Without it - or
    without a saved login - every call returns what is missing, and the venue
    row says so.

SETTLEMENT, PRECISELY. The feed is a diff: `get_selection_changes(since)`
returns every selection change after a sequence number, and each change row
carries `settlement_info` with the venue's own `result` string for that runner.
The adapter keeps a cursor (0 on the first read: the feed is the only place the
venue says who won, and the adapter has no way to know where its own markets'
settlements sit in it, so the first read is the full history, once - every
later read is incremental), caches the verdict per market, and matches the
result string against the winner wording. An unrecognised string is treated as
NOT the winner: the failure direction is a refusal, never a guess.

FEES. Betdaq charges commission on net winnings at settlement, not on the
order; the fill receipt carries no commission, and the paper record's P&L is
gross of the venue's commission. `get_mechanics` says so rather than quoting a
rate this adapter cannot verify.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone

from loguru import logger

from .adapter import (AdapterCapability, EligibilityStatus, MarketAdapter,
                      VenueOpportunity, VenueType)
from ..markets.base import DataMode, Market, MarketSource, Token


BETDAQ_ID_PREFIX = "betdaq"

# The venue's play-money markets are the only ones this adapter reads. Pinned
# at discovery: a real-money market is never published, so no order path to one
# can exist here.
PLAY_MARKETS_ONLY = True

# The widest two-sided quote this adapter will call executable.
MAX_TRUSTED_SPREAD = 0.10

# What to ask for when the caller does not say.
DEFAULT_TARGET_MARKETS = 100

# The smallest stake the venue's order filter is built with, in the play
# currency. The paper lane sizes in dollars; a play-money stake of the same
# number is the honest equivalent.
MIN_STAKE = 1.0

# Decimal-odds tick used when the venue's own ladder cannot be read.
DEFAULT_ODDS_TICK = 0.01

# The venue's own result strings, matched case-insensitively against the
# winner wording. Anything else - "loser", "void", a non-runner, an empty
# string, a wording this adapter has not seen - is NOT the winner, and a
# settled market with no recognised winner is refused rather than guessed.
WINNER_RESULT_WORDS = {"winner", "won", "win", "1st", "first"}

# Runner/market statuses that mean "no result can be stated".
VOIDED_MARKET_STATUS = "VOIDED"

# Order statuses that mean the venue accepted and (at least partially) matched.
# Compared against `_status_name`, which upper-cases.
MATCHED_ORDER_STATUSES = {"MATCHED", "UNMATCHED"}


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or an object; never silently answer "unknown"."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_name(value: Any) -> str:
    """
    A status as a name, from an enum object OR an already-parsed string.

    The wrapper is inconsistent on purpose here: some of its parsers keep the
    enum (`MarketStatus(8)`), others keep `.name` ("SETTLED"). Both are read
    through this so no status silently answers "unknown" for one of them.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().upper()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.strip().upper()
    return _text(value).strip().upper()


def _runner_id(value: Any) -> Optional[int]:
    """A runner id as an int, or None. Ids are ints on the venue's wire."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _market_id_of(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> Optional[datetime]:
    """The wrapper hands back 'YYYY-MM-DD HH:MM:SS.ffffff' strings (tz-naive UTC)."""
    if value is None:
        return None
    text = _text(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.debug(f"Betdaq: unparseable date {value!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _best_level(ladder: Any) -> Optional[Tuple[float, float]]:
    """The best [price, stake] level of a venue ladder (best-first), or None."""
    if not ladder:
        return None
    try:
        first = list(ladder)[0]
        price, stake = float(first[0]), float(first[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if price <= 0 or stake <= 0:
        return None
    return price, stake


def _ladder_to_probabilities(ladder: Any) -> List[Dict[str, float]]:
    """
    A venue ladder in the terms the rest of the system trades in.

    Two conversions, both needed (the same two the Betfair adapter makes):
    decimal odds become the probability 1/odds, and the BACKER'S STAKE quoted
    at a level becomes the number of 1.0-paying contracts at that level
    (stake x odds). Without the second, a 2.50 back with 100 of stake reads as
    100 contracts of a 0.40 asset, which understates the book by 2.5x.
    """
    out: List[Dict[str, float]] = []
    for level in ladder or []:
        try:
            odds, stake = float(level[0]), float(level[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if odds <= 1.0 or stake <= 0:
            continue
        out.append({
            "price": round(1.0 / odds, 6),
            "size": round(stake * odds, 4),
            "odds": odds,
            "stake": stake,
        })
    return out


class BetdaqAdapter(MarketAdapter):
    """
    The Betdaq exchange, read and paper-traded through its own SDK.

    Pinned to the venue's play-money markets: discovery asks for play markets
    and publishes only rows the venue marks `is_play_market`, so every stake
    this adapter places is play money and no real order can leave.
    """

    def __init__(self, username: Optional[str] = None,
                 password: Optional[str] = None):
        super().__init__(venue_id="betdaq", venue_type=VenueType.OTHER)
        self.username = username or ""
        self.password = password or ""
        self.last_error: str = ""
        self._client_obj: Any = None
        self._create_order: Any = None
        self._polarity_back: Any = None
        self.skipped_markets: Dict[str, int] = {}
        # The venue's play-market ids from the last discovery, so the account
        # read can tell this adapter's positions from the account's other ones.
        self._play_market_ids: set = set()
        # The selection-changes cursor and the per-market settlement verdicts.
        self._selection_sequence: Optional[int] = None
        self._settlement_verdicts: Dict[str, Dict[str, Any]] = {}
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            # NOT a real-money path: this adapter is pinned to play markets, so
            # there is no submission path to a real-money market here.
            # `supports_trading` is false for exactly that reason - the
            # play-money submission path is used through place_order's paper
            # branch (see the module docstring).
            supports_trading=False,
            real_order_path=False,
            # Every call - market data included - carries the login in its
            # SOAP header, so the venue's feed needs the login before it
            # answers anything.
            requires_credentials=True,
            supports_portfolio=bool(self.username and self.password),
            supports_history=False,
            fee_taker_pct=0.0,
            fee_maker_pct=0.0,
            min_order_usd=MIN_STAKE,
            implementation_status="live",
            implementation_note=(
                "client on the betdaq SDK (pip install betdaq); pinned to the "
                "venue's play-money markets, so every stake is play money and "
                "no real order can be placed"),
        )

    # ------------------------------------------------------------------
    # the client
    # ------------------------------------------------------------------
    @property
    def sdk_available(self) -> bool:
        """Is the `betdaq` package importable in this interpreter?"""
        try:
            import betdaq.apiclient  # noqa: F401,WPS433
        except Exception:  # noqa: BLE001 - any import failure is the same fact
            return False
        return True

    @property
    def configured(self) -> bool:
        return bool(self.username and self.password)

    def _client(self):
        """(client, error). One place where the SDK or the login can be missing."""
        if not self.configured:
            return None, ("no Betdaq login saved. The venue's feed carries the "
                          "login on every call, so there is nothing to read "
                          "without it; the login form is on the venue page")
        if self._client_obj is not None:
            return self._client_obj, ""
        try:
            from betdaq.apiclient import APIClient  # noqa: WPS433 - optional dep
            from betdaq.enums import Polarity  # noqa: WPS433
            from betdaq.filters import create_order  # noqa: WPS433
        except Exception as e:  # noqa: BLE001
            return None, (f"the Betdaq SDK is not installed "
                          f"({type(e).__name__}: {e}); install it with "
                          f"'pip install betdaq'")
        try:
            # The wrapper builds its SOAP clients from the venue's WSDL at
            # construction, so a construction failure IS a venue answer: it is
            # reported, never retried in a loop and never papered over.
            self._client_obj = APIClient(self.username, self.password)
            self._create_order = create_order
            self._polarity_back = Polarity.back
        except Exception as e:  # noqa: BLE001
            self._client_obj = None
            return None, (f"the Betdaq client could not be built "
                          f"({type(e).__name__}: {e}); the venue's WSDL is at "
                          f"api.betdaq.com and the login is sent with every call")
        return self._client_obj, ""

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        """
        Play money only, and a login even for that.

        This adapter offers no real-money path at all, so the status only ever
        affects what the venue panel promises - the same position PredictIt is
        in. Whether Betdaq serves this country for an account is unverified,
        and the play-money markets are the only thing here that could be used
        anyway; a real-money account would be a different venue row.
        """
        return EligibilityStatus.REQUIRES_VERIFICATION

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    async def discover_markets(self, target_count: int = DEFAULT_TARGET_MARKETS,
                               filters: Dict = None) -> List[Market]:
        """
        The venue's ACTIVE play markets, priced from the venue's own ladders.

        Two venue calls: the market tree (runners, statuses) and the prices
        (ladders, matched amounts). Only single-winner play markets with a
        two-sided price on every published runner are published - the same
        rules the Betfair adapter applies, for the same reason: the outcome
        order must match the venue's settlement order exactly.
        """
        filters = filters or {}
        client, error = self._client()
        if client is None:
            self.last_error = error
            logger.info(f"Betdaq: {error}")
            return []
        try:
            sports = list(client.marketdata.get_sports() or [])
        except Exception as e:  # noqa: BLE001 - a venue read may fail any way
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Betdaq: could not read sports: {self.last_error}")
            return []
        sport_ids = [s.get("sport_id") for s in sports if s.get("sport_id")]
        if not sport_ids:
            self.last_error = "the venue returned no sports"
            logger.warning("Betdaq: no sports returned")
            return []
        try:
            rows = list(client.marketdata.get_sport_markets(
                sport_ids, include_selections=True, WantPlayMarkets=True) or [])
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Betdaq: could not read markets: {self.last_error}")
            return []

        # The skip counters are reset BEFORE the shape filter runs, so they
        # describe THIS discovery (a counter wiped after the filter would
        # report nothing at all).
        self.skipped_markets = {}
        candidates = [r for r in rows if self._is_publishable_shape(r)]
        markets: List[Market] = []
        if candidates:
            price_rows = self._read_prices(client, [r.get("market_id")
                                                    for r in candidates])
            prices_by_id = {self._key(row.get("market_id")): row
                            for row in price_rows}
            for row in candidates:
                market = self._to_market(row, prices_by_id.get(
                    self._key(row.get("market_id"))))
                if market is None:
                    continue
                markets.append(market)
        self._play_market_ids = {self._key(m.id[len(BETDAQ_ID_PREFIX) + 1:])
                                 for m in markets}
        self.last_error = ""
        logger.info(
            f"Betdaq discovered {len(markets)} play market(s) "
            f"({len(rows)} row(s) read, {len(candidates)} in shape, "
            f"{dict(self.skipped_markets) or 'nothing'} skipped)")
        return markets[:max(1, int(target_count or DEFAULT_TARGET_MARKETS))]

    def _read_prices(self, client: Any, market_ids: Sequence[Any]) -> List[Dict]:
        """The venue's price rows for these markets; a failure is a reason."""
        ids = [i for i in market_ids if i is not None]
        if not ids:
            return []
        try:
            return list(client.marketdata.get_prices(ids) or [])
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Betdaq: could not read prices: {self.last_error}")
            return []

    def _is_publishable_shape(self, row: Dict[str, Any]) -> bool:
        """
        The venue-side shape filter, with every skip counted and named.

        Play markets only (pinned), ACTIVE only, single-winner only (a place
        market's "winner" is not one outcome), at least two runners, and a
        market type the venue names - an unnamed type is refused rather than
        settled against wrong facts.
        """
        def skip(why: str) -> bool:
            self.skipped_markets[why] = self.skipped_markets.get(why, 0) + 1
            return False

        if not PLAY_MARKETS_ONLY or not row.get("is_play_market"):
            return skip("not a play market (this adapter reads play markets only)")
        status = _status_name(row.get("market_status"))
        if status != "ACTIVE":
            return skip(f"market status {status or 'unknown'}")
        winners = _num(row.get("number_of_winners"))
        if winners is None or winners != 1.0:
            return skip("not a single-winner market")
        market_type = _status_name(row.get("market_type"))
        if not market_type or market_type in ("UNSPECIFIED", "UNKNOWN"):
            return skip("market type not named by the venue")
        runners = row.get("runners") or []
        if len(runners) < 2:
            return skip("fewer than two runners")
        return True

    def _to_market(self, row: Dict[str, Any],
                   prices_row: Optional[Dict[str, Any]]) -> Optional[Market]:
        """
        One venue market as a Market, or None with the reason counted.

        The published outcome order is the VENUE'S OWN RUNNER ORDER, and a
        market whose first runner has no two-sided price is not published at
        all - the same rule the Betfair adapter enforces, because settlement
        maps the winning runner onto outcome index 0 through exactly this
        order, and a wrong outcome written into calibration is permanent.
        """
        market_id = _market_id_of(row.get("market_id"))
        if market_id is None:
            self.skipped_markets["no market id"] = \
                self.skipped_markets.get("no market id", 0) + 1
            return None
        runners = row.get("runners") or []
        price_runners = {self._key(r.get("runner_id")): r
                         for r in ((prices_row or {}).get("runners") or [])}

        bookable: List[Dict[str, Any]] = []
        for runner in runners:
            runner_key = self._key(runner.get("runner_id"))
            price_runner = price_runners.get(runner_key) or {}
            book = price_runner.get("runner_book") or {}
            # batb = the back side: the prices a backer pays (the asks).
            # batl = the lay side: the prices a layer sells at (the bids).
            best_back = _best_level(book.get("batb"))
            best_lay = _best_level(book.get("batl"))
            if best_back is None or best_lay is None:
                continue
            back_odds, back_stake = best_back
            lay_odds, lay_stake = best_lay
            if back_odds <= 1.0 or lay_odds <= 1.0:
                continue
            bookable.append({
                "runner_id": runner.get("runner_id"),
                "name": _text(runner.get("runner_name")) or str(runner.get("runner_id")),
                "back_odds": back_odds, "back_stake": back_stake,
                "lay_odds": lay_odds, "lay_stake": lay_stake,
                "runner_status": _status_name(runner.get("runner_status")),
                "reset_count": runner.get("reset_count"),
            })
        if not bookable:
            self.skipped_markets["no two-sided price"] = \
                self.skipped_markets.get("no two-sided price", 0) + 1
            return None
        first_id = self._key(runners[0].get("runner_id"))
        if self._key(bookable[0]["runner_id"]) != first_id:
            self.skipped_markets["first runner unpriced"] = \
                self.skipped_markets.get("first runner unpriced", 0) + 1
            logger.info(f"[betdaq] {market_id} skipped: its first runner has no "
                        f"two-sided price, so the published outcome order would "
                        f"not match the venue's settlement order")
            return None

        labels = [r["name"] for r in bookable]
        back_odds = [r["back_odds"] for r in bookable]
        lay_odds = [r["lay_odds"] for r in bookable]
        event_name = _text(row.get("event_name"))
        market_name = _text(row.get("market_name")) or str(market_id)
        sport_name = _text(row.get("sport_name"))
        total_matched = _num((prices_row or {}).get("market_total_matched")) or 0.0
        ladder_stakes = sum(r["back_stake"] + r["lay_stake"] for r in bookable)
        return Market(
            id=f"{BETDAQ_ID_PREFIX}-{market_id}",
            # `MarketSource` has no BETDAQ member (adding one would change a
            # shared enum for one venue); the same normalised carrier Betfair
            # and Simmer use is passed here and `venue_id` is authoritative.
            source=MarketSource.POLYMARKET,
            question=(f"{event_name} - {market_name}" if event_name
                      else market_name)[:400],
            description=(f"Betdaq play market ({_status_name(row.get('market_type'))}"
                         f"); {len(bookable)} selections; two-sided decimal-odds "
                         f"ladders; stakes are play money")[:400],
            outcomes=labels,
            outcome_prices=[round(1.0 / o, 4) for o in back_odds],
            tokens=[Token(token_id=f"{market_id}:{r['runner_id']}",
                          outcome=r["name"], price=round(1.0 / r["back_odds"], 4))
                    for r in bookable],
            # The venue publishes matched amounts, but in its PLAY currency -
            # not the dollars the scan's floors and scores are written in - so
            # the figures are carried in raw with their basis, and the
            # volume floors are skipped for this venue rather than fed numbers
            # from a different unit. The one-turn-per-venue reservation is
            # what still gives it model time.
            volume=total_matched,
            volume_24h=0.0,
            liquidity=ladder_stakes,
            end_date=_parse_date(row.get("market_start_time")),
            active=True,
            closed=False,
            slug=str(market_id),
            event_slug=_text(row.get("event_id"))[:50],
            market_type="categorical",
            raw={
                "venue": "betdaq",
                "api": "betdaq",
                "market_id": market_id,
                "is_play_market": True,
                "market_status": _status_name(row.get("market_status")),
                "market_type": _status_name(row.get("market_type")),
                "number_of_winners": row.get("number_of_winners"),
                "event_name": event_name,
                "sport_name": sport_name,
                "competition_name": _text(row.get("competition_name")),
                "tournament_name": _text(row.get("tournament_name")),
                "withdrawal_sequence_number": row.get("withdrawal_sequence_number"),
                "back_odds": back_odds,
                "lay_odds": lay_odds,
                "back_stakes": [r["back_stake"] for r in bookable],
                "lay_stakes": [r["lay_stake"] for r in bookable],
                "spreads": [round(1.0 / r["lay_odds"] - 1.0 / r["back_odds"], 6)
                            for r in bookable],
                # The PUBLISHED order of the outcomes, by venue runner id.
                # Settlement maps the winning runner back onto an outcome index
                # through this list and through nothing else.
                "selection_ids": [r["runner_id"] for r in bookable],
                "runner_names": labels,
                "runner_reset_counts": [r["reset_count"] for r in bookable],
                "is_exchange": True,
                "lay_available": True,
                "currency": "play money (virtual)",
                # The scan's volume/liquidity floors are written in dollars;
                # this venue's matched amounts are published in its play
                # currency, which is not that unit - so the basis is stated
                # as play money and the floors are skipped for this venue
                # rather than fed numbers from a different unit.
                "volume_basis": "play_money",
                "volume_basis_note": ("the venue publishes matched amounts, but "
                                      "in its play currency, not the dollars "
                                      "the scan's floors are written in"),
                "liquidity_basis": "play_money_ladder_stakes",
            },
            venue_id="betdaq",
            venue_type="other",
            data_mode=DataMode.LIVE,
            data_source="betdaq_sdk",
            is_mock=False,
        )

    # ------------------------------------------------------------------
    # the book
    # ------------------------------------------------------------------
    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        """
        The venue's own ladders for one market, in the project's price space.

        The primary outcome's ladders become `bids` and `asks`: buying the
        primary outcome is backing it (the venue's back side, `batb`, is what a
        backer pays - the asks), and selling it is laying it (`batl` - the
        bids). Both are carried because on an exchange the two sides are
        separate tradable prices, not one spread.
        """
        market_key = self._market_key(getattr(market, "id", ""))
        if not market_key:
            return self._empty_book(market, "no Betdaq market id on this market")
        client, error = self._client()
        if client is None:
            return self._empty_book(market, error)
        rows = self._read_prices(client, [market_key])
        row = next((r for r in rows
                    if self._key(r.get("market_id")) == self._key(market_key)),
                   None)
        if row is None:
            return self._empty_book(
                market, f"the venue returned no prices for market {market_key}")
        status = _status_name(row.get("status"))
        if status and status != "ACTIVE":
            return self._empty_book(
                market, f"the venue reports market {market_key} as {status}; "
                        f"it is not accepting orders")

        published = [self._key(s) for s in
                     ((getattr(market, "raw", None) or {}).get("selection_ids") or [])]
        runners = row.get("runners") or []
        ordered = sorted(
            runners,
            key=lambda r: published.index(self._key(r.get("runner_id")))
            if self._key(r.get("runner_id")) in published else 999)

        runner_books: List[Dict[str, Any]] = []
        for runner in ordered:
            book = runner.get("runner_book") or {}
            runner_books.append({
                "runner_id": runner.get("runner_id"),
                "name": _text(runner.get("runner_name")),
                "status": _status_name(runner.get("runner_status")),
                # The back side is what a backer pays: the asks.
                "back": _ladder_to_probabilities(book.get("batb")),
                # The lay side is what a layer sells at: the bids.
                "lay": _ladder_to_probabilities(book.get("batl")),
                "back_odds": [level[0] for level in (book.get("batb") or [])],
                "lay_odds": [level[0] for level in (book.get("batl") or [])],
            })

        primary = runner_books[0] if runner_books else {"back": [], "lay": []}
        asks = [{"price": level["price"], "size": level["size"]}
                for level in primary["back"]]
        bids = [{"price": level["price"], "size": level["size"]}
                for level in primary["lay"]]
        best_ask = asks[0] if asks else None
        best_bid = bids[0] if bids else None
        # The spread convention the rest of the system reads: ask minus bid,
        # positive in a normal market (the Simmer and PredictIt books carry it
        # the same way). A negative spread is a crossed quote.
        spread = (round(best_ask["price"] - best_bid["price"], 6)
                  if best_ask and best_bid else None)
        if spread is not None and spread < 0:
            return self._empty_book(
                market, f"the venue's quote is crossed (bid {best_bid['price']} "
                        f"is above ask {best_ask['price']}); there is no "
                        f"marketable price in it")
        executable = bool(asks and bids
                          and (spread is not None and spread <= MAX_TRUSTED_SPREAD))
        return {
            "available": True,
            "market_id": getattr(market, "id", ""),
            "venue_id": "betdaq",
            "token_id": getattr(market, "yes_token_id", None),
            "bids": bids,
            "asks": asks,
            "bid": best_bid["price"] if best_bid else None,
            "ask": best_ask["price"] if best_ask else None,
            "bid_size": best_bid["size"] if best_bid else 0.0,
            "ask_size": best_ask["size"] if best_ask else 0.0,
            "spread": spread,
            "spread_pct": spread,
            "mid": round((best_bid["price"] + best_ask["price"]) / 2.0, 6)
                   if best_bid and best_ask else None,
            "depth": sum(level["size"] for r in runner_books
                         for level in r["back"] + r["lay"]),
            "executable": executable,
            "is_real": True,
            "is_mock": False,
            "validated": executable,
            "data_mode": "live",
            "assumed_fields": [],
            "warning": ("" if executable else
                        f"the spread {spread:.2%} is wider than the "
                        f"{MAX_TRUSTED_SPREAD:.0%} this system will trade"
                        if spread is not None else
                        "the venue published no two-sided quote for the "
                        "primary outcome"),
            # The exchange's own view, undeformed: odds and stakes.
            "back": [{"price": r["back_odds"][0] if r["back_odds"] else None,
                      "size": (r["back"][0]["stake"] if r["back"] else 0.0)}
                     for r in runner_books if r["back_odds"]],
            "lay": [{"price": r["lay_odds"][0] if r["lay_odds"] else None,
                     "size": (r["lay"][0]["stake"] if r["lay"] else 0.0)}
                    for r in runner_books if r["lay_odds"]],
            "runners": runner_books,
            "total_matched": _num(row.get("market_total_matched")) or 0.0,
            "status": status,
            "source": "betdaq_play_book",
            "currency": "play money (virtual)",
            "price_basis": ("decimal odds converted to probability (1/odds); a "
                            "back price is a genuine tradable price, not a "
                            "derived spread"),
            "note": ("bids are lay prices and asks are back prices: on an "
                     "exchange the two sides are separate tradable prices. "
                     "Stakes are the venue's play money."),
        }

    def _empty_book(self, market: Market, reason: str) -> Dict[str, Any]:
        """No book, and no invented one either."""
        return {
            "available": False,
            "market_id": getattr(market, "id", ""),
            "venue_id": "betdaq",
            "token_id": getattr(market, "yes_token_id", None),
            "bids": [], "asks": [],
            "bid": None, "ask": None,
            "spread": None, "spread_pct": None,
            "depth": 0,
            "executable": False,
            "is_real": False,
            "is_mock": False,
            "validated": False,
            "data_mode": "live",
            "assumed_fields": [],
            "source": "betdaq_unavailable",
            "warning": "no Betdaq book was read; no spread is published here",
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # orders: the venue's play markets, never real money
    # ------------------------------------------------------------------
    async def place_order(self, opportunity: VenueOpportunity, max_spend_usd: float,
                          max_price: float) -> Dict[str, Any]:
        """
        Submit to the venue's play market, and report what IT says matched.

        This is the difference between this adapter and a local paper fill:
        the order goes to Betdaq, the match comes back from Betdaq's engine,
        and the venue will settle it. The stake is play money, which cannot be
        withdrawn or converted, so no real money is involved and the trade is
        recorded as a paper trade.

        Only a BACK is built, and only on the primary outcome (YES) or - on a
        two-outcome market - the other runner (NO). A lay risks
        stake x (odds - 1), which is not the quantity this executor sizes, and
        a NO on a market with more than two outcomes is not one selection; both
        are refused with their reason, exactly as the Betfair order path does.
        """
        market = getattr(opportunity, "market", None)
        side = _text(getattr(opportunity, "side", "")).upper()

        def refusal(reason: str, status: str = "refused") -> Dict[str, Any]:
            return {"status": status, "success": False, "venue": "betdaq",
                    "venue_id": "betdaq",
                    "market_id": getattr(market, "id", ""),
                    "side": side, "reason": reason,
                    "message": "Refusal, not a fill. No Betdaq order was placed."}

        if not self.dry_run:
            return refusal(
                "this adapter will not place REAL-money orders: it is pinned to "
                "the venue's play-money markets, and PTAI holds no Betdaq "
                "real-money account")
        if market is None:
            return refusal("no market on this opportunity", "rejected")
        market_key = self._market_key(getattr(market, "id", ""))
        if not market_key:
            return refusal("no Betdaq market id on this market", "rejected")
        if max_spend_usd <= 0 or not (0 < float(max_price) < 1):
            return refusal("invalid order guard (spend and limit must be "
                           "positive, and the limit is a probability)",
                           "rejected")
        selection_id, why = self._runner_for_side(market, side)
        if selection_id is None:
            return refusal(why, "rejected")
        client, error = self._client()
        if client is None:
            return refusal(error)

        # Fresh ladders for the exact selection: a book read at discovery time
        # is not what the venue would match against now, and the order carries
        # the venue's own reset/withdrawal sequence numbers, which must match
        # the server's state or the bet is rejected.
        rows = self._read_prices(client, [market_key])
        row = next((r for r in rows
                    if self._key(r.get("market_id")) == self._key(market_key)),
                   None)
        if row is None:
            return refusal(f"the venue returned no prices for market "
                           f"{market_key}")
        runner = next((r for r in (row.get("runners") or [])
                       if self._key(r.get("runner_id")) == self._key(selection_id)),
                      None)
        if runner is None:
            return refusal(f"runner {selection_id} is not in the prices for "
                           f"market {market_key}")
        book = runner.get("runner_book") or {}
        best_back = _best_level(book.get("batb"))
        if best_back is None:
            return refusal(f"the venue is showing no back price for runner "
                           f"{selection_id}", "rejected")
        best_odds, best_stake = best_back
        runner_status = _status_name(runner.get("runner_status"))
        if runner_status and runner_status not in ("ACTIVE", ""):
            return refusal(f"runner {selection_id} is {runner_status} at the "
                           f"venue; it is not accepting orders", "rejected")

        # The price the venue will accept, and the price we are allowed to
        # pay. `max_price` is a probability, so the odds floor is 1/cap,
        # rounded UP to the tick - the safe direction for a back price.
        floor = self._snap_odds(1.0 / float(max_price), up=True)
        if floor is None:
            return refusal(f"a limit of {float(max_price):.4f} is not a price "
                           f"the venue can quote", "rejected")
        crosses = best_odds >= floor
        price = best_odds if crosses else floor
        available_contracts = best_stake * best_odds if crosses else 0.0
        stake = round(min(float(max_spend_usd), available_contracts), 2) \
            if crosses else round(float(max_spend_usd), 2)
        if stake < MIN_STAKE:
            return refusal(f"a stake of {stake:.2f} is below the venue minimum "
                           f"of {MIN_STAKE:.2f} (or {available_contracts:.2f} "
                           f"is all that is offered at {best_odds})", "rejected")

        order = self._create_order(
            SelectionId=selection_id,
            Stake=stake,
            Price=price,
            Polarity=self._polarity_back,
            ExpectedSelectionResetCount=runner.get("runner_reset_count") or 0,
            ExpectedWithdrawalSequenceNumber=(
                row.get("withdrawal_sequence_number") or 0),
        )
        try:
            receipts = list(client.betting.place_orders([order], receipt=True)
                            or [])
        except Exception as e:  # noqa: BLE001 - the SDK raises on transport
            return refusal(f"the venue call failed ({type(e).__name__}: {e})",
                           "error")
        if not receipts:
            return refusal("the venue returned no receipt for the order", "error")
        receipt = receipts[0]
        return_code = _num(receipt.get("return_code"))
        if return_code not in (None, 0.0):
            return refusal(f"the venue refused the order (return code "
                           f"{int(return_code)})", "rejected")
        matched = _num(receipt.get("matched_size")) or 0.0
        matched_price = _num(receipt.get("matched_price"))
        order_id = receipt.get("order_id")
        if matched <= 0 or matched_price is None or matched_price <= 1.0:
            return refusal(
                "the venue matched nothing: the order is resting, not filled, "
                "and a resting order is not a position this agent can account "
                "for", "submitted")

        probability = round(1.0 / matched_price, 6)
        contracts = round(matched * matched_price, 4)
        return {
            "status": "paper",
            "success": True,
            "is_real": False,
            "simulated": True,
            "venue": "betdaq",
            "venue_id": "betdaq",
            "market_id": getattr(market, "id", ""),
            "betdaq_market_id": market_key,
            "side": side,
            "side_sent": "BACK",
            "selection_id": selection_id,
            "order_id": order_id,
            "venue_trade_id": order_id,
            # What the venue matched, in the terms the ledger records: the
            # stake at risk is the cost, and the contracts are stake x odds.
            "filled_price": probability,
            "price": probability,
            "odds": matched_price,
            "size": contracts,
            "size_matched": contracts,
            "filled_usd": round(matched, 6),
            "cost_usd": round(matched, 6),
            "fees_usd": 0.0,
            "currency": "play money (virtual)",
            "fill_status": _status_name(receipt.get("status")) or "Matched",
            "resting": False,
            "source": "betdaq_sdk_play_market",
            "reason": (f"matched on Betdaq's play market: {matched:.2f} of play "
                       f"money at odds {matched_price} on runner "
                       f"{selection_id} ({contracts:.4f} contracts at "
                       f"{probability:.4f}). No real money was involved; the "
                       f"venue settles this position"),
        }

    def _runner_for_side(self, market: Market, side: str) -> Tuple[Optional[int], str]:
        """
        Which venue runner a YES or NO view is a bet on, or why not.

        YES is the market's first published outcome (the venue's own first
        runner - see `_to_market` for why that alignment is enforced at
        publish time). NO is the OTHER runner, and that is only a single bet
        on a two-outcome market.
        """
        published = [self._key(s) for s in
                     ((getattr(market, "raw", None) or {}).get("selection_ids")
                      or [])]
        if not published:
            return None, "this market carries no venue runner ids"
        if side == "YES":
            return published[0], ""
        if side == "NO":
            if len(published) != 2:
                return None, (f"a NO view on a {len(published)}-outcome Betdaq "
                              f"market is not one runner, and this order path "
                              f"will not synthesise it from a lay")
            return published[1], ""
        return None, f"this order path speaks YES/NO; got {side!r}"

    def _snap_odds(self, odds: float, up: bool = True) -> Optional[float]:
        """Snap decimal odds onto the venue's ladder, or the default tick."""
        if odds is None or odds <= 1.0:
            return None
        tick = self._odds_tick()
        snapped = round(round(odds / tick) * tick, 10)
        if not up and snapped > odds:
            snapped = round(snapped - tick, 10)
        if up and snapped < odds:
            snapped = round(snapped + tick, 10)
        return round(snapped, 10) if snapped > 1.0 else None

    def _odds_tick(self) -> float:
        """The venue's own price increment, when its ladder can be read."""
        client, error = self._client()
        if client is None:
            return DEFAULT_ODDS_TICK
        try:
            ladder = list(client.marketdata.get_odds_ladder() or [])
        except Exception:  # noqa: BLE001 - the tick is a convenience, not a gate
            return DEFAULT_ODDS_TICK
        prices = sorted({_num(level.get("price")) for level in ladder
                         if _num(level.get("price"))})
        if len(prices) >= 2:
            steps = [round(b - a, 10) for a, b in zip(prices, prices[1:])
                     if b > a]
            if steps:
                return min(steps)
        return DEFAULT_ODDS_TICK

    # ------------------------------------------------------------------
    # the outcome
    # ------------------------------------------------------------------
    async def get_settlement(self, market_id: str) -> Dict[str, Any]:
        """
        What the venue says this market resolved to, if it has resolved.

        The market row only says SETTLED; the winner is published per RUNNER
        in the venue's selection-changes feed, as the venue's own result string.
        This adapter polls that feed (a cursor, full history on the first read
        only), matches the result string against the winner wording, and maps
        the winning runner onto the published outcome order. A settled market
        whose feed names no winner is REFUSED, not interpreted - an inferred
        outcome is a fabricated one once it reaches the calibration record.
        """
        market_key = self._market_key(market_id)
        if not market_key:
            return self._unsettled("no_market_id", "empty Betdaq market id")
        cached = self._settlement_verdicts.get(str(market_key))
        if cached is not None:
            return cached
        client, error = self._client()
        if client is None:
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "betdaq_unreachable", "reason": error,
                    "venue_id": "betdaq"}
        try:
            rows = list(client.marketdata.get_markets([market_key]) or [])
        except Exception as e:  # noqa: BLE001
            return {"settled": False, "outcome": None, "is_real": False,
                    "source": "betdaq_unreadable",
                    "reason": (f"could not read market {market_key}: "
                               f"{type(e).__name__}: {e}"),
                    "venue_id": "betdaq"}
        row = next((r for r in rows
                    if self._key(r.get("market_id")) == self._key(market_key)),
                   None)
        if row is None:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "betdaq_market_missing",
                    "reason": f"the venue does not have market {market_key}",
                    "venue_id": "betdaq", "market_id": market_key}
        status = _status_name(row.get("market_status"))
        runners = row.get("runners") or []
        first_runner = self._key(runners[0].get("runner_id")) if runners else None
        if status == VOIDED_MARKET_STATUS:
            verdict = {"settled": False, "outcome": None, "is_real": True,
                       "source": "betdaq_voided", "market_id": market_key,
                       "raw_status": status,
                       "reason": (f"market {market_key} is voided: all matched "
                                  f"orders in it are voided too, so it resolves "
                                  f"nothing and no outcome is recorded")}
            self._settlement_verdicts[str(market_key)] = verdict
            return verdict
        if status != "SETTLED":
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "betdaq_open", "market_id": market_key,
                    "raw_status": status,
                    "reason": (f"market {market_key} is "
                               f"{status or 'in an unknown state'}; no result "
                               f"is published yet")}
        if not runners:
            return {"settled": False, "outcome": None, "is_real": True,
                    "source": "betdaq_settled_no_runners",
                    "market_id": market_key, "raw_status": status,
                    "reason": f"market {market_key} is settled with no runners "
                              f"read; the venue's result cannot be mapped"}

        winner_id, winner_name, results = self._poll_results(client, market_key)
        if winner_id is None:
            verdict = {"settled": False, "outcome": None, "is_real": True,
                       "source": "betdaq_settled_no_winner",
                       "market_id": market_key, "raw_status": status,
                       "runner_results": results,
                       "reason": (f"market {market_key} is settled and the "
                                  f"venue's feed names no winner among its "
                                  f"runners (results: "
                                  f"{results or 'none published'}); this "
                                  f"adapter does not infer one")}
            self._settlement_verdicts[str(market_key)] = verdict
            return verdict
        verdict = {
            "settled": True,
            "outcome": 1.0 if self._key(winner_id) == first_runner else 0.0,
            "is_real": True,
            "source": "betdaq_selection_result",
            "market_id": market_key,
            "raw_status": status,
            "mapping": ("venue runner order; index 0 is the first published "
                        "outcome"),
            "winner_runner_id": winner_id,
            "winner_name": winner_name,
            "runner_results": results,
            "reason": (f"{winner_name or winner_id} won, and it is "
                       f"{'the first' if self._key(winner_id) == first_runner else 'not the first'} "
                       f"published outcome"),
        }
        self._settlement_verdicts[str(market_key)] = verdict
        return verdict

    def _poll_results(self, client: Any, market_key: Any) -> Tuple[
            Optional[int], str, Dict[str, str]]:
        """
        (winner runner id, winner name, {runner id: result string}).

        The feed is a diff since a sequence number. The cursor starts at 0 on
        the first read - the feed is the only place the venue says who won,
        and the adapter has no way to know where its own markets' settlements
        sit in it - so the first read is the full history, once; every later
        read is incremental. Results are matched against the venue's own
        wording; anything unrecognised is not the winner.
        """
        since = self._selection_sequence if self._selection_sequence is not None else 0
        try:
            rows = list(client.marketdata.get_selection_changes(since) or [])
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Betdaq: could not read selection changes: "
                           f"{self.last_error}")
            return None, "", {}
        sequences = [_num(r.get("sequence_number")) for r in rows]
        sequences = [int(s) for s in sequences if s is not None]
        if sequences:
            self._selection_sequence = max(sequences)
        results: Dict[str, str] = {}
        winner_id: Optional[int] = None
        winner_name = ""
        for row in rows:
            if self._key(row.get("market_id")) != self._key(market_key):
                continue
            runner_id = _runner_id(row.get("runner_id"))
            if runner_id is None:
                continue
            for info in (row.get("settlement_info") or []):
                result = _text(info.get("result")).strip()
                if not result:
                    continue
                results[str(runner_id)] = result
                if result.lower() in WINNER_RESULT_WORDS and winner_id is None:
                    winner_id = runner_id
                    winner_name = _text(row.get("runner_name"))
        return winner_id, winner_name, results

    @staticmethod
    def _unsettled(source: str, reason: str) -> Dict[str, Any]:
        return {"settled": False, "outcome": None, "is_real": False,
                "source": f"betdaq_{source}", "reason": reason,
                "venue_id": "betdaq"}

    # ------------------------------------------------------------------
    # account
    # ------------------------------------------------------------------
    async def get_portfolio(self) -> Dict[str, Any]:
        """
        The play-money positions this adapter opened, when a login is saved.

        The venue publishes ONE account balance, and it is real money - this
        adapter trades only play markets and never reports that figure as this
        venue's, because a balance shown next to play-money positions invites
        the reading "this venue holds my money", which is false here: play
        money cannot be withdrawn, deposited or converted, and the venue
        cannot hold money from this operator's country in any case.
        """
        client, error = self._client()
        if client is None:
            return {"venue_id": "betdaq", "available": False, "balance": None,
                    "positions": [], "paper": True, "virtual": True,
                    "reason": error}
        try:
            rows = list(client.betting.get_orders() or [])
        except Exception as e:  # noqa: BLE001
            return {"venue_id": "betdaq", "available": False, "balance": None,
                    "positions": [], "paper": True, "virtual": True,
                    "reason": f"could not read orders: {type(e).__name__}: {e}"}
        positions: List[Dict[str, Any]] = []
        skipped = 0
        for row in rows:
            market_key = self._key(row.get("market_id"))
            if market_key not in {self._key(i) for i in self._play_market_ids}:
                skipped += 1
                continue
            status = _status_name(row.get("status"))
            if status not in MATCHED_ORDER_STATUSES:
                continue
            positions.append({
                "market_id": row.get("market_id"),
                "runner_id": row.get("runner_id"),
                "side": _status_name(row.get("side")),
                "stake_matched": _num(row.get("matched_size")) or 0.0,
                "average_price": _num(row.get("average_price")),
                "status": status,
                "currency": "play money (virtual)",
                "virtual": True,
            })
        return {
            "venue_id": "betdaq",
            "available": True,
            "balance": None,
            "currency": "play money (virtual)",
            "paper": True,
            "virtual": True,
            "balance_note": ("the venue publishes one account balance and it is "
                             "real money; this adapter trades only play markets, "
                             "so no balance figure is reported as this venue's"),
            "currency_note": ("play money is the venue's own virtual currency: "
                              "it cannot be withdrawn, deposited or converted, "
                              "so it is not capital"),
            "positions": positions,
            "position_count": len(positions),
            "orders_skipped_not_play": skipped,
            "source": "betdaq_sdk",
        }

    # ------------------------------------------------------------------
    # the venue's order rules
    # ------------------------------------------------------------------
    def get_mechanics(self, opportunity=None, token_id: Optional[str] = None):
        """
        What this venue's orders obey, as far as it publishes.

        The tick is the venue's own ladder increment when it can be read.
        Betdaq charges commission on net winnings AT SETTLEMENT, not on the
        order: the fill receipt carries no commission, so the paper record's
        P&L is gross of the venue's commission - stated here rather than hidden
        behind a rate this adapter cannot verify.
        """
        from ..markets.mechanics import MarketMechanics  # noqa: WPS433

        return MarketMechanics(
            tick_size=f"{self._odds_tick():.2f}",
            min_order_size=MIN_STAKE,
            min_order_notional_usd=0.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            source="betdaq_documented_rules",
            is_real=False,
            warnings=[
                "the venue charges commission on net winnings at settlement, "
                "not on the order; the paper record's P&L is gross of that "
                "commission",
                "stakes are the venue's play money; the tick is the venue's "
                "own ladder increment when its ladder can be read, else "
                f"{DEFAULT_ODDS_TICK:.2f}",
            ],
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "venue_id": "betdaq",
            "sdk_available": self.sdk_available,
            "configured": self.configured,
            "play_markets": len(self._play_market_ids),
            "selection_sequence": self._selection_sequence,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _key(value: Any) -> Optional[int]:
        """A market/runner id as an int, or None. Ids are ints on the wire."""
        return _market_id_of(value)

    @staticmethod
    def _market_key(market_id: Any) -> str:
        """The venue's own market id, from `betdaq-<id>` or a bare id."""
        raw = _text(market_id).strip()
        if not raw:
            return ""
        if raw.lower().startswith(BETDAQ_ID_PREFIX + "-"):
            return raw[len(BETDAQ_ID_PREFIX) + 1:]
        return raw
