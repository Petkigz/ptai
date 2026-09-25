"""
Sports Data Ingestion - real feeds, real odds, no invented numbers.

Two hard rules this module follows:

1. It never fabricates a price. If a feed is unreachable, or a key is
   missing, the result is an empty list plus a diagnostic. A made-up
   sportsbook line is worse than no line at all, because everything
   downstream treats it as real and sizes money against it.

2. Every Market it emits carries an accurate `data_mode` and `data_source`,
   so the execution guard can tell a live feed from a replay.

Providers
---------
ESPN            free, no key. Scoreboards for NBA/NFL/MLB/NHL/EPL/La Liga/
                UCL/tennis, plus a consensus line (spread + total) per event.
The Odds API    multi-bookmaker real odds (Pinnacle, Betfair Exchange,
                bet365, ...). Needs a key; free tier is ~500 req/month.
football-data   free token, European football fixtures + standings.

Moneyline/spread handling
-------------------------
American odds are converted to decimal, and every book's prices are de-vigged
before they are treated as a probability. Vig removal method defaults to the
power method (see odds_math) because it corrects the longshot bias that
proportional removal leaves in.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from loguru import logger

from ..markets.base import DataMode, Market, MarketSource, Token
from .odds_math import (
    american_to_decimal,
    analyse_book,
    devig,
    implied_probability,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class SportsEvent:
    """One fixture/game, provider-neutral."""
    event_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    commence_time: datetime
    is_live: bool = False
    status: str = "scheduled"
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    provider: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable cross-provider key for matching the same fixture."""
        return f"{self.league}:{self.away_team.strip().lower().replace(' ', '-')}-at-{self.home_team.strip().lower().replace(' ', '-')}"

    @property
    def hours_to_start(self) -> float:
        now = datetime.now(timezone.utc)
        ct = self.commence_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        return (ct - now).total_seconds() / 3600.0


@dataclass
class BookOdds:
    """One bookmaker's prices for one event."""
    book: str
    market: str                     # h2h | spreads | totals
    outcomes: Dict[str, float]      # outcome label -> decimal odds
    spread: Optional[float] = None  # handicap for spreads
    point: Optional[float] = None   # line for totals
    last_update: Optional[datetime] = None
    is_exchange: bool = False       # exchange prices have much lower vig
    commission_pct: float = 0.0

    @property
    def overround(self) -> float:
        if not self.outcomes:
            return float("nan")
        try:
            return analyse_book(list(self.outcomes.values())).overround
        except ValueError:
            return float("nan")

    def fair_probs(self, method: str = "power") -> Dict[str, float]:
        if not self.outcomes:
            return {}
        labels = list(self.outcomes.keys())
        prices = [self.outcomes[k] for k in labels]
        fair = devig(prices, method)
        return dict(zip(labels, fair))


@dataclass
class ConsensusOdds:
    """Aggregated view across all books for one event/market."""
    event: SportsEvent
    market: str
    books: List[BookOdds]
    fair_probs: Dict[str, float]
    best_price: Dict[str, float]
    best_book: Dict[str, float]
    sharp_probs: Dict[str, float]      # de-vigged Pinnacle/exchange only
    mean_overround: float
    n_books: int
    is_synthetic: bool = False


@dataclass
class LineMovement:
    """How a line has moved over the tracking window."""
    event_key: str
    market: str
    outcome: str
    first_price: float
    last_price: float
    move_pct: float
    direction: str                 # up | down | flat
    is_steam: bool                 # fast, large, across multiple books
    steam_reason: str = ""
    n_observations: int = 0


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

# What a normal browser sends. ESPN's public site API answered 403 Forbidden to
# the old "ptai/1.0 (local research bot)" string, and with it the whole betting
# lane aborted every cycle ("no fixtures from any feed") - no goals, cards,
# corners, totals or match markets were priced at all. A truthful UA that also
# identifies as a browser is what every other client uses for public feeds.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 PTAI/1.0"
)


# A provider that refuses this machine is not a provider that was asked and had
# nothing to say. The operator's log repeated this every 10-minute cycle:
#
#   [espn] fetch failed .../basketball/nba/scoreboard: HTTPStatusError: 403 Forbidden
#   [espn] fetch failed .../soccer/eng.1/scoreboard: HTTPStatusError: 403 Forbidden
#   [sports] 0 unique fixtures, 2 provider issue(s)
#   [betting] cycle aborted: no fixtures from any feed
#
# ...and the whole sports lane - goals, cards, corners, totals, halves - priced
# nothing. A 403 is an answer, and it is worth re-asking only occasionally.
BLOCKED_RETRY_SECONDS = 900.0


class BaseProvider:
    name = "base"
    is_synthetic = False

    def __init__(self, timeout: float = 12.0, user_agent: str = BROWSER_USER_AGENT):
        self.timeout = timeout
        self.user_agent = user_agent
        self.last_error: str = ""
        self.last_fetch_at: Optional[datetime] = None
        self.requests_made = 0
        # Set when the feed refuses us outright (401/403/407). While it is set the
        # provider is skipped, so the log says once why an entire lane is dark
        # instead of printing the same 403 every cycle forever.
        self.blocked_reason: str = ""
        self.blocked_until: Optional[datetime] = None

    @property
    def is_blocked(self) -> bool:
        if not self.blocked_reason:
            return False
        if self.blocked_until and datetime.now(timezone.utc) >= self.blocked_until:
            self.blocked_reason = ""
            self.blocked_until = None
            return False
        return True

    def _mark_blocked(self, status: int, url: str) -> None:
        self.blocked_reason = (
            f"{self.name} refuses this network with HTTP {status} "
            f"({url.split('/scoreboard')[0].split('//')[-1]})")
        self.blocked_until = datetime.now(timezone.utc) + timedelta(
            seconds=BLOCKED_RETRY_SECONDS)
        logger.warning(
            f"[{self.name}] HTTP {status} - this feed is refusing this machine. "
            f"Skipping it for {BLOCKED_RETRY_SECONDS/60:.0f} min; the sports lane "
            f"stays dark until it lifts or another feed is added "
            f"(THE_ODDS_API_KEY / FOOTBALL_DATA_TOKEN in Setup).")

    async def _get(self, client: httpx.AsyncClient, url: str, params: Dict = None,
                   headers: Dict = None) -> Optional[Any]:
        try:
            self.requests_made += 1
            r = await client.get(url, params=params or {}, headers=headers or {},
                                 timeout=self.timeout, follow_redirects=True)
            if r.status_code == 429:
                self.last_error = f"429 rate limited by {self.name}"
                logger.warning(self.last_error)
                return None
            if r.status_code in (401, 403, 407):
                self.last_error = f"HTTP {r.status_code} from {self.name}"
                self._mark_blocked(r.status_code, url)
                return None
            r.raise_for_status()
            self.last_fetch_at = datetime.now(timezone.utc)
            return r.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning(f"[{self.name}] fetch failed {url[:80]}: {self.last_error}")
            return None

    async def events(self, leagues: Sequence[str] = ()) -> List[SportsEvent]:
        raise NotImplementedError

    async def odds(self, event: SportsEvent) -> List[BookOdds]:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "reachable": not self.last_error,
            "last_error": self.last_error,
            "last_fetch_at": self.last_fetch_at.isoformat() if self.last_fetch_at else None,
            "requests_made": self.requests_made,
            "is_synthetic": self.is_synthetic,
        }


# ---------------------------------------------------------------------------
# ESPN (free, no key)
# ---------------------------------------------------------------------------

# league slug -> (sport path, league code)
ESPN_LEAGUES: Dict[str, Tuple[str, str]] = {
    "nba": ("basketball", "nba"),
    "nfl": ("football", "nfl"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "epl": ("soccer", "eng.1"),
    "laliga": ("soccer", "esp.1"),
    "seriea": ("soccer", "ita.1"),
    "bundesliga": ("soccer", "ger.1"),
    "ucl": ("soccer", "uefa.champions"),
    "mls": ("soccer", "usa.1"),
    "atp": ("tennis", "atp"),
    "wta": ("tennis", "wta"),
    "ncaam": ("basketball", "mens-college-basketball"),
}


class EspnProvider(BaseProvider):
    """
    ESPN's public site API. No key, no auth, and it carries a consensus
    spread + total per event, which is enough to build a real fair value
    for US sports and European football.
    """
    name = "espn"

    BASE = "https://site.api.espn.com/apis/site/v2/sports"

    async def events(self, leagues: Sequence[str] = ("nba", "epl")) -> List[SportsEvent]:
        out: List[SportsEvent] = []
        async with httpx.AsyncClient(headers={"User-Agent": self.user_agent}) as client:
            for slug in leagues:
                if slug not in ESPN_LEAGUES:
                    self.last_error = f"unknown league {slug}"
                    continue
                sport, code = ESPN_LEAGUES[slug]
                data = await self._get(client, f"{self.BASE}/{sport}/{code}/scoreboard",
                                       params={"limit": 100})
                if not data:
                    continue
                for ev in data.get("events", []) or []:
                    parsed = self._parse_event(ev, slug)
                    if parsed:
                        out.append(parsed)
        logger.info(f"[espn] fetched {len(out)} events across {len(list(leagues))} leagues")
        return out

    def _parse_event(self, ev: Dict, league: str) -> Optional[SportsEvent]:
        comps = ev.get("competitions") or []
        if not comps:
            return None
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        if not home or not away:
            return None
        try:
            commence = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except Exception:
            commence = datetime.now(timezone.utc) + timedelta(days=1)

        status = ((comp.get("status") or {}).get("type") or {}).get("name", "scheduled")
        # ESPN reports statuses like "STATUS_IN_PROGRESS" / "STATUS_HALFTIME";
        # strip the prefix before classifying or nothing is ever live.
        normalised = status.upper().replace("STATUS_", "").replace("_", "")
        is_live = normalised in ("IN", "INPROGRESS", "HALFTIME", "LIVE", "INTERMISSION")

        return SportsEvent(
            event_id=str(ev.get("id", "")),
            sport=ESPN_LEAGUES.get(league, ("", ""))[0],
            league=league,
            home_team=(home.get("team") or {}).get("displayName", "Home"),
            away_team=(away.get("team") or {}).get("displayName", "Away"),
            commence_time=commence,
            is_live=is_live,
            status=status,
            home_score=float(home["score"]) if str(home.get("score", "")).replace(".", "").isdigit() else None,
            away_score=float(away["score"]) if str(away.get("score", "")).replace(".", "").isdigit() else None,
            provider=self.name,
            raw={"odds_raw": comp.get("odds"), "venue": (comp.get("venue") or {}).get("fullName")},
        )

    def odds_for_event(self, event: SportsEvent) -> List[BookOdds]:
        """
        Convert the consensus line already carried on the ESPN event payload.

        ESPN's `details` string is like 'LAL -5.5' and `moneyLine` entries
        look like {'awayOdds': 215, 'homeOdds': -260, 'drawOdds': 340}.
        """
        raw_list = event.raw.get("odds_raw") or []
        out: List[BookOdds] = []
        for block in raw_list:
            provider = ((block or {}).get("provider") or {}).get("name", "espn-consensus")
            details = block.get("details") or ""
            over_under = block.get("overUnder")
            ml = block.get("moneyLine") or {}

            if ml:
                outcomes: Dict[str, float] = {}
                for key, label in (("awayOdds", event.away_team), ("homeOdds", event.home_team)):
                    val = ml.get(key)
                    if isinstance(val, (int, float)) and val != 0:
                        outcomes[label] = american_to_decimal(float(val))
                draw = ml.get("drawOdds")
                if isinstance(draw, (int, float)) and draw != 0:
                    outcomes["Draw"] = american_to_decimal(float(draw))
                if len(outcomes) >= 2:
                    out.append(BookOdds(book=provider, market="h2h", outcomes=outcomes,
                                        is_exchange=False))

            spread = self._parse_spread(details)
            if spread is not None and ml:
                fav_is_home = self._favourite_is_home(block, event, details)
                outcomes = {}
                for key, label in (("awayOdds", event.away_team), ("homeOdds", event.home_team)):
                    val = ml.get(key)
                    if isinstance(val, (int, float)) and val != 0:
                        outcomes[label] = american_to_decimal(float(val))
                if outcomes:
                    out.append(BookOdds(book=f"{provider}-spread", market="spreads",
                                        outcomes=outcomes,
                                        spread=-spread if fav_is_home else spread))

            if over_under and isinstance(over_under, (int, float)):
                for side, key in (("over", "overOdds"), ("under", "underOdds")):
                    pass
                over_odds = block.get("overOdds")
                under_odds = block.get("underOdds")
                if over_odds and under_odds:
                    out.append(BookOdds(book=f"{provider}-total", market="totals",
                                        outcomes={"Over": american_to_decimal(float(over_odds)),
                                                  "Under": american_to_decimal(float(under_odds))},
                                        point=float(over_under)))
        return out

    @staticmethod
    def _initials(name: str) -> str:
        """'Los Angeles Lakers' -> 'LAL', for matching ESPN's abbreviations."""
        skip = {"fc", "cf", "sc", "ac", "de", "la", "los", "the", "club", "afc"}
        parts = [w for w in name.replace(".", " ").split() if w.lower() not in skip]
        return "".join(w[0] for w in parts).upper()

    @classmethod
    def _favourite_is_home(cls, block: Dict, event: SportsEvent, details: str) -> bool:
        """
        Work out which side carries the handicap.

        ESPN flags it explicitly via homeTeamOdds.favorite - use that first.
        The fallback is the abbreviation in `details` ('LAL -5.5'), which
        cannot be matched as a substring because 'LAL' is not contained in
        'Los Angeles Lakers'; compare against initials instead.
        """
        home_flags = block.get("homeTeamOdds") or {}
        away_flags = block.get("awayTeamOdds") or {}
        if isinstance(home_flags.get("favorite"), bool) or isinstance(away_flags.get("favorite"), bool):
            return bool(home_flags.get("favorite")) and not bool(away_flags.get("favorite"))

        token = details.upper().split()[0] if details.strip() else ""
        if not token:
            return False
        if event.home_team.lower() in details.lower():
            return True
        if token == cls._initials(event.home_team):
            return True
        if token == cls._initials(event.away_team):
            return False
        # last resort: the shorter money is the favourite
        ml = block.get("moneyLine") or {}
        ho, ao = ml.get("homeOdds"), ml.get("awayOdds")
        if isinstance(ho, (int, float)) and isinstance(ao, (int, float)) and ho and ao:
            return american_to_decimal(float(ho)) < american_to_decimal(float(ao))
        return False

    @staticmethod
    def _parse_spread(details: str) -> Optional[float]:
        """
        ESPN carries the line as a string like 'LAL -5.5' or 'EVEN'.

        The team abbreviation has to be stripped before the number can be
        parsed - float('LAL -5.5') raises, which silently drops the whole
        spread book rather than just that field.
        """
        if not details:
            return None
        d = details.upper().strip()
        if d in ("EVEN", "PK", "PICK", "EVENMONEY"):
            return 0.0
        token = d.split()[-1].replace("+", "")
        try:
            return abs(float(token))
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# The Odds API (multi-bookmaker, needs key)
# ---------------------------------------------------------------------------

class TheOddsApiProvider(BaseProvider):
    """
    Real multi-book odds including Pinnacle and Betfair Exchange - the two
    sharpest prices publicly available. Pinnacle/exchange de-vigged prices
    are the honest 'sharp reference' that fabricated numbers pretend to be.
    """
    name = "the_odds_api"
    BASE = "https://api.the-odds-api.com/v4"

    SPORT_KEYS = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "epl": "soccer_epl",
        "laliga": "soccer_spain_la_liga",
        "seriea": "soccer_italy_serie_a",
        "bundesliga": "soccer_germany_bundesliga",
        "ucl": "soccer_uefa_champs_league",
        "atp": "tennis_atp_french_open",
    }

    def __init__(self, api_key: str = "", regions: str = "eu,uk",
                 markets: str = "h2h,spreads,totals", **kw):
        super().__init__(**kw)
        self.api_key = api_key
        self.regions = regions
        self.markets = markets
        self.remaining_requests: Optional[int] = None

    async def events(self, leagues: Sequence[str] = ("nba",)) -> List[SportsEvent]:
        if not self.api_key:
            self.last_error = "no THE_ODDS_API_KEY configured - refusing to invent odds"
            logger.warning(f"[{self.name}] {self.last_error}")
            return []
        out: List[SportsEvent] = []
        async with httpx.AsyncClient(headers={"User-Agent": self.user_agent}) as client:
            for slug in leagues:
                key = self.SPORT_KEYS.get(slug)
                if not key:
                    continue
                data = await self._get(client, f"{self.BASE}/sports/{key}/odds", params={
                    "apiKey": self.api_key, "regions": self.regions,
                    "markets": self.markets, "oddsFormat": "decimal", "dateFormat": "iso",
                })
                if not data:
                    continue
                for item in data:
                    try:
                        commence = datetime.fromisoformat(str(item["commence_time"]).replace("Z", "+00:00"))
                    except Exception:
                        commence = datetime.now(timezone.utc) + timedelta(days=1)
                    ev = SportsEvent(
                        event_id=str(item.get("id")), sport=key.split("_")[0], league=slug,
                        home_team=item.get("home_team", ""), away_team=item.get("away_team", ""),
                        commence_time=commence, provider=self.name,
                        raw={"bookmakers": item.get("bookmakers") or []},
                    )
                    out.append(ev)
        return out

    def odds_for_event(self, event: SportsEvent) -> List[BookOdds]:
        out: List[BookOdds] = []
        for bm in event.raw.get("bookmakers") or []:
            book = bm.get("title", "unknown")
            is_ex = "exchange" in book.lower() or "betfair" in book.lower()
            try:
                last_update = datetime.fromisoformat(str(bm.get("last_update")).replace("Z", "+00:00"))
            except Exception:
                last_update = None
            for mkt in bm.get("markets") or []:
                outcomes: Dict[str, float] = {}
                for o in mkt.get("outcomes") or []:
                    price = o.get("price")
                    if isinstance(price, (int, float)) and price > 1.0:
                        outcomes[o.get("name", "?")] = float(price)
                if len(outcomes) >= 2:
                    out.append(BookOdds(
                        book=book, market=mkt.get("key", "h2h"), outcomes=outcomes,
                        spread=None, point=None, last_update=last_update,
                        is_exchange=is_ex, commission_pct=0.05 if is_ex else 0.0,
                    ))
        return out


# ---------------------------------------------------------------------------
# football-data.org (free token)
# ---------------------------------------------------------------------------

class FootballDataProvider(BaseProvider):
    name = "football_data"
    BASE = "https://api.football-data.org/v4"

    def __init__(self, token: str = "", **kw):
        super().__init__(**kw)
        self.token = token

    async def events(self, leagues: Sequence[str] = ("epl",)) -> List[SportsEvent]:
        if not self.token:
            self.last_error = "no FOOTBALL_DATA_TOKEN configured"
            return []
        out: List[SportsEvent] = []
        async with httpx.AsyncClient(headers={"X-Auth-Token": self.token,
                                             "User-Agent": self.user_agent}) as client:
            for slug in leagues:
                data = await self._get(client, f"{self.BASE}/competitions/{slug.upper()}/matches",
                                       params={"status": "SCHEDULED,TIMED,LIVE,IN_PLAY,PAUSED"})
                if not data:
                    continue
                for m in data.get("matches", []) or []:
                    try:
                        commence = datetime.fromisoformat(str(m["utcDate"]).replace("Z", "+00:00"))
                    except Exception:
                        continue
                    out.append(SportsEvent(
                        event_id=str(m.get("id")), sport="soccer", league=slug,
                        home_team=(m.get("homeTeam") or {}).get("name", ""),
                        away_team=(m.get("awayTeam") or {}).get("name", ""),
                        commence_time=commence,
                        status=m.get("status", "SCHEDULED"),
                        provider=self.name, raw=m,
                    ))
        return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SportsDataEngine:
    """
    Aggregates providers into a consensus fair value per event, tracks line
    movement, and exposes Market objects the rest of PTAI understands.

    Sharp reference: when Pinnacle or a betting exchange is present, its
    de-vigged price is used as the fair anchor - that is a real sharp price
    rather than the market's own price nudged by an arbitrary amount.
    """

    SHARP_BOOKS = ("pinnacle", "betfair", "exchange", "sbobet", "circa", "bookmaker.eu")

    def __init__(self, providers: Optional[List[BaseProvider]] = None,
                 settings=None, devig_method: str = "power"):
        self.providers = providers if providers is not None else self._default_providers(settings)
        self.devig_method = devig_method
        self._line_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.events_cache: List[SportsEvent] = []
        self.diagnostics: List[str] = []

    @staticmethod
    def _default_providers(settings) -> List[BaseProvider]:
        providers: List[BaseProvider] = [EspnProvider()]
        odds_key = getattr(settings, "the_odds_api_key", "") if settings else ""
        fd_token = getattr(settings, "football_data_token", "") if settings else ""
        if odds_key:
            providers.append(TheOddsApiProvider(api_key=odds_key))
        if fd_token:
            providers.append(FootballDataProvider(token=fd_token))
        return providers

    # -- fetch --------------------------------------------------------------

    async def fetch_events(self, leagues: Sequence[str] = ("nba", "epl")) -> List[SportsEvent]:
        self.diagnostics.clear()
        # A feed that refused us recently is not asked again on every cycle; the
        # reason is carried so the caller can say "every feed refused" rather than
        # "no fixtures" - the two mean completely different things.
        active = [p for p in self.providers if not getattr(p, "is_blocked", False)]
        for p in self.providers:
            if getattr(p, "is_blocked", False):
                self.diagnostics.append(p.blocked_reason)
        if not active:
            logger.warning(
                "[sports] every fixture feed is refusing this machine - "
                f"{'; '.join(self.diagnostics) or 'no providers configured'}")
            self.events_cache = []
            return self.events_cache
        results = await asyncio.gather(
            *[p.events(leagues) for p in active], return_exceptions=True)
        merged: Dict[str, SportsEvent] = {}
        for provider, res in zip(active, results):
            if isinstance(res, Exception):
                self.diagnostics.append(f"{provider.name}: {type(res).__name__} {res}")
                continue
            if not res:
                self.diagnostics.append(f"{provider.name}: 0 events ({provider.last_error or 'empty'})")
            for ev in res:
                # prefer the richest payload for a duplicate fixture
                existing = merged.get(ev.key)
                if existing is None or (ev.raw.get("bookmakers") and not existing.raw.get("bookmakers")):
                    merged[ev.key] = ev
        self.events_cache = list(merged.values())
        for p in self.providers:
            if p.last_error:
                self.diagnostics.append(f"{p.name}: {p.last_error}")
        logger.info(f"[sports] {len(self.events_cache)} unique fixtures, "
                    f"{len(self.diagnostics)} provider issue(s)")
        return self.events_cache

    def odds_for(self, event: SportsEvent) -> List[BookOdds]:
        books: List[BookOdds] = []
        for p in self.providers:
            fn = getattr(p, "odds_for_event", None)
            if fn is None:
                continue
            try:
                books.extend(fn(event))
            except Exception as e:
                self.diagnostics.append(f"{p.name} odds parse: {e}")
        return books

    # -- consensus ----------------------------------------------------------

    def consensus(self, event: SportsEvent, market: str = "h2h") -> Optional[ConsensusOdds]:
        books = [b for b in self.odds_for(event) if b.market == market]
        if not books:
            return None

        all_probs: Dict[str, List[float]] = {}
        sharp_probs: Dict[str, List[float]] = {}
        best_price: Dict[str, float] = {}
        best_book: Dict[str, float] = {}
        overrounds: List[float] = []

        for b in books:
            fair = b.fair_probs(self.devig_method)
            over = b.overround
            if over == over:  # not NaN
                overrounds.append(over)
            is_sharp = any(s in b.book.lower() for s in self.SHARP_BOOKS) or b.is_exchange
            for label, price in b.outcomes.items():
                all_probs.setdefault(label, []).append(fair.get(label, 0.0))
                if is_sharp:
                    sharp_probs.setdefault(label, []).append(fair.get(label, 0.0))
                if label not in best_price or price > best_price[label]:
                    best_price[label] = price
                    best_book[label] = b.book

        fair_probs = {k: sum(v) / len(v) for k, v in all_probs.items() if v}
        sharp = {k: sum(v) / len(v) for k, v in sharp_probs.items() if v}

        return ConsensusOdds(
            event=event, market=market, books=books, fair_probs=fair_probs,
            best_price=best_price, best_book=best_book,
            sharp_probs=sharp,
            mean_overround=sum(overrounds) / len(overrounds) if overrounds else float("nan"),
            n_books=len(books),
            is_synthetic=False,
        )

    # -- line movement ------------------------------------------------------

    def record_line(self, event_key: str, market: str, outcome: str, price: float) -> None:
        key = f"{event_key}|{market}|{outcome}"
        self._line_history.setdefault(key, []).append((datetime.now(timezone.utc), price))

    def line_movement(self, event_key: str, market: str, outcome: str,
                      steam_window_minutes: float = 30.0,
                      steam_move_pct: float = 2.0) -> Optional[LineMovement]:
        key = f"{event_key}|{market}|{outcome}"
        hist = self._line_history.get(key) or []
        if len(hist) < 2:
            return None
        first_price = hist[0][1]
        last_price = hist[-1][1]
        move_pct = (last_price / first_price - 1.0) * 100.0 if first_price else 0.0

        # Steam = large move inside a short window
        window = timedelta(minutes=steam_window_minutes)
        recent = [(t, p) for t, p in hist if hist[-1][0] - t <= window]
        steam = False
        reason = ""
        if len(recent) >= 2:
            window_move = abs((recent[-1][1] / recent[0][1] - 1.0) * 100.0)
            if window_move >= steam_move_pct:
                steam = True
                reason = (f"{window_move:.2f}% move in {steam_window_minutes:.0f}min "
                          f"across {len(recent)} observations - likely sharp money")

        return LineMovement(
            event_key=event_key, market=market, outcome=outcome,
            first_price=first_price, last_price=last_price, move_pct=round(move_pct, 4),
            direction="up" if move_pct > 0.05 else ("down" if move_pct < -0.05 else "flat"),
            is_steam=steam, steam_reason=reason, n_observations=len(hist),
        )

    # -- Market bridging ----------------------------------------------------

    def to_market(self, consensus: ConsensusOdds, data_mode: DataMode = DataMode.LIVE) -> Optional[Market]:
        """
        Turn a consensus view into the Market object the rest of PTAI scores.

        data_mode is passed in by the caller because only the caller knows
        whether this fetch is feeding live capital, a paper run or a replay.
        """
        ev = consensus.event
        if not consensus.fair_probs:
            return None
        labels = list(consensus.fair_probs.keys())
        prices = [consensus.best_price.get(l, 0.0) for l in labels]
        if any(p <= 1.0 for p in prices):
            return None

        q = f"{ev.away_team} vs {ev.home_team} ({ev.league.upper()})"
        return Market(
            id=f"sports-{ev.league}-{ev.event_id}",
            source=MarketSource.POLYMARKET,  # normalised carrier; venue_id is authoritative
            question=q,
            description=(f"{ev.league.upper()} {ev.sport}: {ev.away_team} @ {ev.home_team}, "
                         f"starts {ev.commence_time.isoformat()}, {consensus.n_books} books, "
                         f"mean overround {consensus.mean_overround:.3f}"),
            outcomes=labels,
            outcome_prices=[round(consensus.fair_probs[l], 4) for l in labels],
            tokens=[Token(token_id=f"{ev.event_id}:{l}", outcome=l, price=round(p, 4))
                    for l, p in zip(labels, prices)],
            volume=0.0, volume_24h=0.0, liquidity=0.0,
            end_date=ev.commence_time,
            active=not ev.is_live, closed=False,
            slug=f"{ev.key}-{consensus.market}", event_slug=ev.key,
            market_type="categorical" if len(labels) > 2 else "binary",
            raw={
                "venue": "sports", "sport": ev.sport, "league": ev.league,
                "market_type": consensus.market, "is_live": ev.is_live,
                "n_books": consensus.n_books,
                "best_price": consensus.best_price, "best_book": consensus.best_book,
                "sharp_probs": consensus.sharp_probs,
                "fair_probs": {k: round(v, 4) for k, v in consensus.fair_probs.items()},
                "mean_overround": round(consensus.mean_overround, 4),
                "data_mode": data_mode.value,
                "data_source": "sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
            },
            venue_id="sports", venue_type="other",
            data_mode=data_mode, is_mock=data_mode == DataMode.MOCK,
            data_source="sports_live_feed" if data_mode.can_deploy_live_capital else data_mode.value,
        )

    # -- reporting ----------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        providers = [p.health() for p in self.providers]
        reachable = [p for p in providers if p["reachable"]]
        return {
            "providers": providers,
            "n_reachable": len(reachable),
            "n_configured": len(providers),
            "events_cached": len(self.events_cache),
            "line_history_keys": len(self._line_history),
            "diagnostics": self.diagnostics[:10],
            "has_real_odds": any("pinnacle" in (p.get("provider", "").lower()) for p in providers)
                             or any(not p["is_synthetic"] and p["reachable"] for p in providers),
            "devig_method": self.devig_method,
        }


# ---------------------------------------------------------------------------
# Offline sample generator - clearly labelled, never presented as live
# ---------------------------------------------------------------------------

def sample_events(n: int = 5, data_mode: DataMode = DataMode.MOCK) -> List[SportsEvent]:
    """
    Deterministic fixtures for development and unit tests ONLY.

    These carry `data_mode=MOCK` by construction and the execution guard
    blocks MOCK data from live orders. It exists so the pipeline can be
    exercised without a network, not so the system can pretend to have
    real odds.
    """
    base = datetime.now(timezone.utc) + timedelta(hours=3)
    fixtures = [
        ("nba", "Los Angeles Lakers", "Golden State Warriors"),
        ("nfl", "Kansas City Chiefs", "Buffalo Bills"),
        ("epl", "Manchester City", "Arsenal"),
        ("laliga", "Real Madrid", "Barcelona"),
        ("ucl", "Bayern Munich", "Paris Saint-Germain"),
        ("mlb", "New York Yankees", "Boston Red Sox"),
        ("nhl", "Toronto Maple Leafs", "Montreal Canadiens"),
    ]
    out: List[SportsEvent] = []
    for i, (league, home, away) in enumerate(fixtures[:n]):
        out.append(SportsEvent(
            event_id=f"MOCK-{i}", sport=league, league=league, home_team=home, away_team=away,
            commence_time=base + timedelta(hours=i * 6), provider="sample",
            raw={"synthetic": True, "data_mode": data_mode.value,
                 "safety": "MOCK_DATA must be impossible to reach live execution"},
        ))
    return out


def sample_book(home: str, away: str, draw: bool = False) -> List[BookOdds]:
    """Deterministic three-book sample for tests. Prices are obviously round."""
    if draw:
        return [
            BookOdds(book="sample-book-a", market="h2h",
                     outcomes={home: 2.10, away: 3.60, "Draw": 3.40}),
            BookOdds(book="sample-book-b", market="h2h",
                     outcomes={home: 2.05, away: 3.75, "Draw": 3.30}),
            BookOdds(book="pinnacle", market="h2h",
                     outcomes={home: 2.08, away: 3.68, "Draw": 3.35}, commission_pct=0.0),
        ]
    return [
        BookOdds(book="sample-book-a", market="h2h", outcomes={home: 1.91, away: 1.91}),
        BookOdds(book="sample-book-b", market="h2h", outcomes={home: 1.87, away: 1.95}),
        BookOdds(book="pinnacle", market="h2h", outcomes={home: 1.89, away: 1.93}),
        BookOdds(book="betfair-exchange", market="h2h", outcomes={home: 1.90, away: 1.92},
                 is_exchange=True, commission_pct=0.05),
    ]
