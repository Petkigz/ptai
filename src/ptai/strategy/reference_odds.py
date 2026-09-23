"""
Cross-venue reference odds - a fair value anchor from prices that exist.

The point of a reference is that it is INDEPENDENT. A reference derived from
the market's own price is not an anchor, it is the market agreeing with
itself, and the gap between them is whatever offset you chose to add.

That is exactly what this module used to do. Four of its five sources
fabricated a number:

    pinnacle:   best_price + 0.05        # "5% edge mock"
    deribit:    assumed BTC is $90,000, then 0.5 - distance * 1.5
    fed_funds:  0.35 / 0.60 / 0.55 picked by which word appeared in the question
    polling:    0.52 / 0.48 hardcoded per candidate

and each came back with a confidence of 0.65-0.85 plus a `should_trade` flag
that the alpha engine reads to apply a 20% score boost. So a market was being
boosted because it disagreed with a number generated from itself. This is the
single most dangerous bug of its kind in the codebase: it does not merely fail
to find edge, it manufactures the appearance of it in the module whose whole
job is to be the honest counterweight.

What it does now
----------------
Every source either returns a price it actually obtained, or returns None and
records why. There is no synthetic path and no hardcoded probability.

    pinnacle / betfair  real odds via TheOddsApiProvider (needs a key)
    deribit             real HTTP call to the public options summary API
    kalshi              real cross-venue price from fetched Kalshi markets
    fed_funds           unavailable - CME has no public auth-free API
    polling             unavailable - no free aggregator API
    manifold            real HTTP call to the public markets API
    metaculus           real HTTP call to the public API

Every returned reference carries `provenance`, so a caller can tell a live
fetched price from a cross-venue comparison, and `is_synthetic` is always
False - if a source cannot be reached it contributes nothing rather than
contributing something fake.

The unavailable sources are kept in the source list on purpose. Silently
dropping them would hide the fact that this project has no Fed funds or
polling reference, which is itself information a user is entitled to see.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..markets.base import Market

# Provenance kinds. A reference is only allowed to be one of these; there is
# deliberately no "estimated" or "modelled" value, because that is how the
# fabricated numbers got in.
PROV_LIVE_API = "live_api"          # fetched from the source's own API
PROV_CROSS_VENUE = "cross_venue"    # a real price on another prediction venue
PROV_UNAVAILABLE = "unavailable"    # no data; the reference contributes nothing

# Sources with no public auth-free API. Kept visible so the absence is
# reported rather than hidden behind a fabricated number.
UNAVAILABLE_SOURCES: Dict[str, str] = {
    "fed_funds": ("CME FedWatch has no public auth-free API. A Fed funds "
                  "reference must come from a licensed CME data feed."),
    "polling": ("No free polling-aggregator API is available. A polling "
                "reference must come from a licensed aggregator."),
}

# Per-source trade thresholds. These are unchanged from the previous version
# so downstream behaviour on REAL references is preserved - the change is
# that fabricated references no longer exist to clear them.
SOURCE_RULES: Dict[str, Dict[str, float]] = {
    "deribit":  {"min_edge": 0.07, "min_confidence": 0.70, "category": "crypto"},
    "fed_funds": {"min_edge": 0.05, "min_confidence": 0.80, "category": "economics"},
    "pinnacle": {"min_edge": 0.06, "min_confidence": 0.75, "category": "sports"},
    "kalshi":   {"min_edge": 0.05, "min_confidence": 0.60, "category": None},
    "polling":  {"min_edge": 0.06, "min_confidence": 0.60, "category": "politics"},
    "manifold": {"min_edge": 0.10, "min_confidence": 0.50, "category": None},
    "metaculus": {"min_edge": 0.10, "min_confidence": 0.50, "category": None},
}


@dataclass
class ReferenceOdds:
    """One independent reference price for one market."""
    source: str
    market_id: str
    reference_price: float
    polymarket_price: float
    edge: float                 # reference - market; positive = market underpriced
    confidence: float
    reasoning: str
    should_trade: bool
    category: str
    provenance: str = PROV_LIVE_API
    is_synthetic: bool = False
    fetched_at: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)


@dataclass
class UnavailableReference:
    """
    A source that could not produce a price, with the reason.

    Returned alongside real references so the caller can see what is missing
    instead of silently working with fewer anchors than it thinks it has.
    """
    source: str
    reason: str
    category: str = ""


class ReferenceOddsEngine:
    """
    Cross-venue reference odds.

    Sources are attempted in order and each is independently optional. The
    engine never substitutes a model estimate for a missing source - if only
    one of seven sources is reachable, the ensemble is built from that one
    and the report says so.
    """

    def __init__(self, odds_api_key: str = "", odds_regions: str = "eu,uk",
                 timeout: float = 12.0, enabled_sources: Optional[List[str]] = None,
                 http_get=None):
        """
        `http_get` is injectable for tests: an async or sync callable taking a
        URL and returning parsed JSON or None. Nothing else about the engine
        changes.
        """
        self.sources = ["deribit", "kalshi", "pinnacle", "fed_funds", "polling",
                        "manifold", "metaculus"]
        # `enabled_sources=[]` means "disable everything" and must not fall
        # through to the default - an empty list is falsy, so `or` would
        # silently re-enable every source.
        self.enabled_sources = (list(self.sources) if enabled_sources is None
                                else list(enabled_sources))
        self.odds_api_key = odds_api_key
        self.odds_regions = odds_regions
        self.timeout = timeout
        self._http_get = http_get
        # per-run diagnostics: source -> reason it produced nothing
        self.unavailable: Dict[str, str] = {}
        self.last_run_at: Optional[str] = None
        self.fetch_counts: Dict[str, int] = {s: 0 for s in self.sources}

    # ------------------------------------------------------------------
    # Category detection
    # ------------------------------------------------------------------

    def _detect_category(self, market: Market) -> str:
        q = market.question.lower()
        if any(k in q for k in ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana"]):
            return "crypto"
        if any(k in q for k in ["trump", "biden", "election", "republican", "democrat",
                                "senate", "congress"]):
            return "politics"
        if any(k in q for k in ["nfl", "nba", "mlb", "soccer", "football", "sports",
                                "team", "game"]):
            return "sports"
        if any(k in q for k in ["fed", "cpi", "inflation", "interest rate", "fomc",
                                "gdp", "nfp", "jobs"]):
            return "economics"
        if any(k in q for k in ["weather", "hurricane", "temperature"]):
            return "weather"
        return "general"

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Fetch JSON or return None. Never raises, never invents a payload."""
        if self._http_get is not None:
            try:
                return self._http_get(url, params)
            except Exception as e:
                logger.debug(f"[reference] injected http_get failed: {type(e).__name__}")
                return None
        try:
            import requests
            r = requests.get(url, params=params or {}, timeout=self.timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            logger.debug(f"[reference] GET {url} failed: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------
    # Deribit - real options implied probability
    # ------------------------------------------------------------------

    def get_deribit_implied_prob(self, market: Market) -> Optional[Tuple[float, float, str]]:
        """
        Risk-neutral probability from the live Deribit options summary.

        The old version assumed BTC was at $90,000 and applied a made-up
        decay to the target. It now reads the actual mark price and IV from
        the public API and prices the target under a lognormal model, which
        is the standard risk-neutral approximation for a digital payout.

        Returns None when the API is unreachable, so the reference simply
        does not exist rather than being guessed.
        """
        if "deribit" not in self.enabled_sources:
            return None
        category = self._detect_category(market)
        if category != "crypto":
            return None

        target = self._extract_price_target(market.question)
        if target is None:
            self.unavailable["deribit"] = "no dollar target found in the question"
            return None

        coin = "BTC" if any(k in market.question.lower() for k in ("btc", "bitcoin")) else "ETH"
        data = self._get_json(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
            {"currency": coin, "kind": "option"})
        if not data or not isinstance(data.get("result"), list) or not data["result"]:
            self.unavailable["deribit"] = f"{coin} options summary unreachable"
            return None

        spot, iv = self._deribit_spot_and_iv(data["result"], target)
        if spot is None or spot <= 0:
            self.unavailable["deribit"] = f"no {coin} mark price in the options summary"
            return None

        days = self._days_to_expiry(market)
        prob = lognormal_digital_prob(spot, target, iv, days)
        if prob is None:
            self.unavailable["deribit"] = "could not price the target (bad expiry or IV)"
            return None

        # Confidence comes from the quality of the data actually returned, not
        # from a constant: how many strikes we saw and how close the IV sits to
        # the target strike.
        confidence = _deribit_confidence(len(data["result"]), iv)
        self.fetch_counts["deribit"] += 1
        reasoning = (f"Deribit {coin} options: mark ${spot:,.0f}, ATM IV {iv*100:.1f}%, "
                     f"{days:.0f}d to expiry -> P(> ${target:,.0f}) = {prob:.3f} "
                     f"(lognormal risk-neutral)")
        return round(prob, 4), round(confidence, 3), reasoning

    def _deribit_spot_and_iv(self, rows: List[Dict[str, Any]],
                             target: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Best available mark price and the IV of the strike nearest the target.

        Deribit's summary rows carry `mark_price` as a fraction of the index
        and `strike`. Using the nearest-strike IV rather than a blanket ATM
        figure matters, because IV is not flat across strikes and the target
        is usually well out of the money.
        """
        marks = [r.get("mark_price") for r in rows
                 if isinstance(r.get("mark_price"), (int, float)) and r["mark_price"] > 0]
        underlying = None
        for r in rows:
            idx = r.get("underlying_price")
            if isinstance(idx, (int, float)) and idx > 0:
                underlying = float(idx)
                break
        if underlying is None and marks:
            # mark_price is a fraction of spot, so it cannot recover spot alone
            return None, None

        best_iv = None
        best_dist = None
        for r in rows:
            strike = r.get("strike")
            iv = r.get("implied_volatility")
            if not isinstance(strike, (int, float)) or not isinstance(iv, (int, float)):
                continue
            if iv <= 0:
                continue
            dist = abs(float(strike) - target)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_iv = float(iv)
        return underlying, best_iv

    @staticmethod
    def _extract_price_target(question: str) -> Optional[float]:
        """Parse a dollar target out of a question. Returns None if absent."""
        m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kKmM]?)", question)
        if not m:
            return None
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            return None
        suffix = m.group(2).lower()
        return value * {"k": 1_000, "m": 1_000_000}.get(suffix, 1)

    @staticmethod
    def _days_to_expiry(market: Market) -> float:
        """Days until the market resolves, floored at one day."""
        end = getattr(market, "end_date", None)
        if isinstance(end, datetime):
            try:
                now = datetime.now(timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                return max(1.0, (end - now).total_seconds() / 86400.0)
            except Exception:
                pass
        return 30.0

    def get_deribit_reference(self, market: Market) -> Optional[Any]:
        """Compatibility wrapper returning an object with reference_price."""
        result = self.get_deribit_implied_prob(market)
        if result is None:
            return None
        prob, conf, reasoning = result
        return SimpleRef(source="deribit", reference_price=prob,
                         confidence=conf, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Pinnacle / Betfair - real sharp bookmaker odds
    # ------------------------------------------------------------------

    def get_pinnacle_implied_prob(self, market: Market) -> Optional[Tuple[float, float, str]]:
        """
        Sharp sportsbook reference from real bookmaker odds.

        The previous implementation computed `best_price + 0.05` - an edge
        invented from the market's own price, dressed up as the sharpest
        sportsbook in the world. There is no way to recover an honest number
        from that, so it is gone.

        This now delegates to TheOddsApiProvider, which carries genuine
        Pinnacle and Betfair Exchange prices. Without an API key it returns
        None, and the reference simply does not exist.
        """
        if "pinnacle" not in self.enabled_sources:
            return None
        category = self._detect_category(market)
        if category != "sports":
            return None
        if not self.odds_api_key:
            self.unavailable["pinnacle"] = (
                "no THE_ODDS_API_KEY configured - refusing to invent a sharp price "
                "from the market's own odds")
            return None

        try:
            from ..betting.sports_data import TheOddsApiProvider, sample_events
        except Exception as e:
            self.unavailable["pinnacle"] = f"sports data module unavailable: {e}"
            return None

        provider = TheOddsApiProvider(api_key=self.odds_api_key, regions=self.odds_regions)
        league = self._guess_league(market.question)
        if not league:
            self.unavailable["pinnacle"] = "could not identify a league from the question"
            return None

        try:
            import asyncio
            events = asyncio.get_event_loop().run_until_complete(provider.events([league]))
        except RuntimeError:
            events = asyncio.new_event_loop().run_until_complete(provider.events([league]))
        except Exception as e:
            self.unavailable["pinnacle"] = f"odds fetch failed: {type(e).__name__}"
            return None

        sharp = self._sharp_price_from_events(events, market, provider)
        if sharp is None:
            self.unavailable["pinnacle"] = (
                provider.last_error or f"no Pinnacle/exchange price for '{league}'")
            return None

        prob, book_name, devigged = sharp
        self.fetch_counts["pinnacle"] += 1
        reasoning = (f"{book_name} de-vigged probability {prob:.3f} for "
                     f"'{market.question[:50]}' (raw {devigged:.3f} before vig removal)")
        # Pinnacle is the sharpest public price; confidence reflects that, but
        # only because a real price was actually returned.
        return round(prob, 4), 0.80, reasoning

    @staticmethod
    def _sharp_price_from_events(events, market: Market, provider) -> Optional[Tuple[float, str, float]]:
        """De-vig the sharpest book's price for this market. None if absent."""
        SHARP = ("pinnacle", "betfair", "exchange")
        q = market.question.lower()
        for ev in events or []:
            if ev.home_team and ev.home_team.lower() not in q and ev.away_team.lower() not in q:
                continue
            for bo in provider.odds_for_event(ev):
                if not any(s in bo.book.lower() for s in SHARP):
                    continue
                prices = [p for p in bo.outcomes.values() if isinstance(p, (int, float)) and p > 1.0]
                if len(prices) < 2:
                    continue
                # remove the overround so the reference is a probability, not a book price
                inv_sum = sum(1.0 / p for p in prices)
                devigged = [1.0 / (p * inv_sum) for p in prices]
                # the market's outcome, matched by name where possible
                idx = _best_outcome_index(bo.outcomes, market)
                if idx is None:
                    continue
                return devigged[idx], bo.book, devigged[idx]
        return None

    @staticmethod
    def _guess_league(question: str) -> Optional[str]:
        q = question.lower()
        mapping = {"nba": "nba", "nfl": "nfl", "mlb": "mlb", "nhl": "nhl",
                   "premier league": "epl", "epl": "epl", "la liga": "laliga",
                   "serie a": "seriea", "bundesliga": "bundesliga",
                   "champions league": "ucl"}
        for key, league in mapping.items():
            if key in q:
                return league
        return None

    # ------------------------------------------------------------------
    # Kalshi - real cross-venue price
    # ------------------------------------------------------------------

    def get_kalshi_reference(self, market: Market,
                             kalshi_markets: Optional[List[Market]] = None
                             ) -> Optional[Tuple[float, float, str]]:
        """
        Kalshi as an independent anchor for a Polymarket market.

        This one was already honest: it compares against an actual price on
        another regulated venue. The only change is that the similarity
        threshold is enforced and the match is reported, so a weak match
        cannot masquerade as a reference.
        """
        if "kalshi" not in self.enabled_sources:
            return None
        if not kalshi_markets:
            self.unavailable["kalshi"] = "no Kalshi markets supplied to compare against"
            return None

        try:
            from .arbitrage import ArbitrageEngine
            arb = ArbitrageEngine(min_spread=0.01, min_confidence_same_event=0.6)
            score_of = arb._same_event_score
        except Exception as e:
            self.unavailable["kalshi"] = f"matching engine unavailable: {e}"
            return None

        best_match, best_score = None, 0.0
        for km in kalshi_markets:
            try:
                score = score_of(market, km)
            except Exception:
                continue
            if score > best_score:
                best_score, best_match = score, km

        if best_match is None or best_score <= 0.6:
            self.unavailable["kalshi"] = (
                "no Kalshi market matched above the 0.6 similarity threshold")
            return None

        price = getattr(best_match, "best_price", None)
        if not isinstance(price, (int, float)) or not (0 < price < 1):
            self.unavailable["kalshi"] = "matched Kalshi market has no usable price"
            return None

        self.fetch_counts["kalshi"] += 1
        confidence = round(best_score * 0.8, 3)
        reasoning = (f"Kalshi '{best_match.question[:50]}' at {price:.3f} "
                     f"(similarity {best_score:.2f}) vs market {market.best_price:.3f}")
        return round(float(price), 4), confidence, reasoning

    # ------------------------------------------------------------------
    # Manifold / Metaculus - real public prediction markets
    # ------------------------------------------------------------------

    def get_manifold_reference(self, market: Market) -> Optional[Tuple[float, float, str]]:
        """
        Manifold is a play-money market, so its prices are weak evidence and
        are weighted accordingly. Still a real price on another venue, which
        is more than a hardcoded number ever was.
        """
        if "manifold" not in self.enabled_sources:
            return None
        data = self._get_json("https://manifold.markets/api/markets",
                              {"term": market.question[:60], "limit": 5})
        if not data or not isinstance(data, list):
            self.unavailable["manifold"] = "Manifold API unreachable or empty"
            return None

        best, best_score = None, 0.0
        for m in data:
            if not isinstance(m, dict):
                continue
            score = _text_similarity(market.question, str(m.get("question", "")))
            if score > best_score:
                best_score, best = score, m
        if best is None or best_score < 0.55:
            self.unavailable["manifold"] = "no Manifold market matched above 0.55 similarity"
            return None

        prob = best.get("probability")
        if not isinstance(prob, (int, float)) or not (0 < prob < 1):
            self.unavailable["manifold"] = "matched Manifold market has no usable probability"
            return None

        self.fetch_counts["manifold"] += 1
        # Play money: real price, low information content.
        confidence = round(min(0.45, best_score * 0.5), 3)
        reasoning = (f"Manifold '{str(best.get('question',''))[:50]}' at {prob:.3f} "
                     f"(similarity {best_score:.2f}); play-money market, weak anchor")
        return round(float(prob), 4), confidence, reasoning

    def get_metaculus_reference(self, market: Market) -> Optional[Tuple[float, float, str]]:
        """Metaculus community forecast as a long-horizon anchor."""
        if "metaculus" not in self.enabled_sources:
            return None
        data = self._get_json("https://www.metaculus.com/api2/questions/",
                              {"search": market.question[:60], "page_size": 5})
        results = (data or {}).get("results") if isinstance(data, dict) else None
        if not results:
            self.unavailable["metaculus"] = "Metaculus API unreachable or empty"
            return None

        best, best_score = None, 0.0
        for q in results:
            if not isinstance(q, dict):
                continue
            score = _text_similarity(market.question, str(q.get("title", "")))
            if score > best_score:
                best_score, best = score, q
        if best is None or best_score < 0.55:
            self.unavailable["metaculus"] = "no Metaculus question matched above 0.55"
            return None

        forecast = ((best.get("prediction_timeseries") or [{}])[-1] or {}).get("c")
        if not isinstance(forecast, (int, float)) or not (0 < forecast < 1):
            self.unavailable["metaculus"] = "matched Metaculus question has no usable forecast"
            return None

        self.fetch_counts["metaculus"] += 1
        confidence = round(min(0.5, best_score * 0.55), 3)
        reasoning = (f"Metaculus '{str(best.get('title',''))[:50]}' at {forecast:.3f} "
                     f"(similarity {best_score:.2f})")
        return round(float(forecast), 4), confidence, reasoning

    # ------------------------------------------------------------------
    # Sources with no public API
    # ------------------------------------------------------------------

    def get_fed_funds_implied_prob(self, market: Market) -> None:
        """
        CME FedWatch implied probability.

        Deliberately always None. There is no public auth-free CME API, and
        the previous version returned 0.35/0.60/0.55 chosen by which word
        appeared in the question while claiming 85% confidence and calling
        CME "gold standard". A Fed decision anchor that is invented is worse
        than no anchor, because it is believed.

        Wired to nothing on purpose. When a licensed feed is available, this
        is the single method to implement.
        """
        if "fed_funds" not in self.enabled_sources:
            return None
        if self._detect_category(market) != "economics":
            return None
        self.unavailable["fed_funds"] = UNAVAILABLE_SOURCES["fed_funds"]
        return None

    def get_polling_reference(self, market: Market) -> None:
        """
        Polling aggregator reference.

        Deliberately always None. The previous version returned a hardcoded
        0.52 for Trump and 0.48 for Biden regardless of the actual race, the
        year, or whether the market was even about those candidates - then
        called it a polling aggregator with 65% confidence.
        """
        if "polling" not in self.enabled_sources:
            return None
        if self._detect_category(market) != "politics":
            return None
        self.unavailable["polling"] = UNAVAILABLE_SOURCES["polling"]
        return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_all_reference_odds(self, market: Market,
                               kalshi_markets: Optional[List[Market]] = None
                               ) -> List[ReferenceOdds]:
        """
        Every independent reference that actually produced a price.

        Sources that produced nothing are recorded in `self.unavailable` with
        a reason, and never appear here. An empty list means no anchor exists
        for this market - which is a real answer, not a failure to answer.
        """
        self.unavailable = {}
        self.last_run_at = datetime.now(timezone.utc).isoformat()
        category = self._detect_category(market)
        price = float(market.best_price)

        candidates = [
            ("deribit", lambda: self.get_deribit_implied_prob(market)),
            ("pinnacle", lambda: self.get_pinnacle_implied_prob(market)),
            ("kalshi", lambda: self.get_kalshi_reference(market, kalshi_markets)),
            ("manifold", lambda: self.get_manifold_reference(market)),
            ("metaculus", lambda: self.get_metaculus_reference(market)),
            # these two always return None by design, but calling them records
            # the reason in self.unavailable so the gap is visible
            ("fed_funds", lambda: self.get_fed_funds_implied_prob(market)),
            ("polling", lambda: self.get_polling_reference(market)),
        ]

        references: List[ReferenceOdds] = []
        for source, fn in candidates:
            try:
                result = fn()
            except Exception as e:
                self.unavailable[source] = f"{type(e).__name__}: {e}"
                logger.debug(f"[reference] {source} raised: {type(e).__name__}: {e}")
                continue
            if not result:
                continue

            ref_price, confidence, reasoning = result
            if not (0.0 < ref_price < 1.0):
                self.unavailable[source] = f"returned an out-of-range price {ref_price}"
                continue

            rules = SOURCE_RULES.get(source, {})
            min_edge = rules.get("min_edge", 0.06)
            min_conf = rules.get("min_confidence", 0.6)
            edge = ref_price - price
            references.append(ReferenceOdds(
                source=source, market_id=market.id, reference_price=ref_price,
                polymarket_price=price, edge=round(edge, 4), confidence=confidence,
                reasoning=reasoning,
                should_trade=(abs(edge) > min_edge and confidence > min_conf),
                category=rules.get("category") or category,
                provenance=(PROV_CROSS_VENUE if source in ("kalshi", "manifold", "metaculus")
                            else PROV_LIVE_API),
                is_synthetic=False, fetched_at=self.last_run_at,
            ))
        return references

    def get_unavailable_references(self, market: Market) -> List[UnavailableReference]:
        """Which sources produced nothing, and why. Call after get_all_reference_odds."""
        category = self._detect_category(market)
        return [UnavailableReference(source=s, reason=r, category=category)
                for s, r in sorted(self.unavailable.items())]

    def get_ensemble_reference(self, market: Market,
                               kalshi_markets: Optional[List[Market]] = None
                               ) -> Optional[Tuple[float, float, str, List[ReferenceOdds]]]:
        """
        Confidence-weighted ensemble across the references that exist.

        Returns None when there are none. It does not fall back to the market
        price, because an ensemble equal to the market price would report zero
        edge while implying that several independent sources confirmed it.
        """
        references = self.get_all_reference_odds(market, kalshi_markets)
        if not references:
            return None

        total_weight = sum(r.confidence for r in references)
        if total_weight <= 0:
            return None

        ensemble_price = sum(r.reference_price * r.confidence for r in references) / total_weight
        # Averaging independent sources raises confidence, but only up to the
        # point where a single source dominates - hence weighting by count.
        ensemble_confidence = min(0.95, total_weight / len(references)
                                  * (1.0 + 0.1 * (len(references) - 1)))
        edge = ensemble_price - float(market.best_price)

        reasoning = (
            f"Ensemble of {len(references)} real source(s): "
            f"{', '.join(f'{r.source} {r.reference_price:.3f} (conf {r.confidence:.2f})' for r in references)} | "
            f"Weighted {ensemble_price:.3f} vs market {market.best_price:.3f} = edge {edge*100:.1f}% | "
            f"Confidence {ensemble_confidence:.2f}"
        )
        missing = sorted(self.unavailable)
        if missing:
            reasoning += f" | no anchor from: {', '.join(missing)}"

        return round(ensemble_price, 4), round(ensemble_confidence, 3), reasoning, references

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """
        Honest capability report.

        This used to advertise seven sources at 65-85% confidence, four of
        which were hardcoded numbers. It now states plainly which sources can
        produce a real price and which cannot, because a user deciding whether
        to trust the anchor needs to know that.
        """
        return {
            "engine": "Cross-Venue Reference Odds",
            "principle": ("a reference must be independent of the market's own price; "
                          "a source that cannot be reached contributes nothing rather "
                          "than contributing a number"),
            "sources": {
                "deribit": {
                    "status": "real" if not self.odds_api_key else "real",
                    "how": "public options summary API; lognormal risk-neutral digital "
                           "probability at the nearest-strike IV",
                    "requires": "outbound HTTPS",
                },
                "pinnacle": {
                    "status": "real if configured",
                    "how": "de-vigged Pinnacle / Betfair Exchange prices via The Odds API",
                    "requires": "THE_ODDS_API_KEY",
                    "configured": bool(self.odds_api_key),
                },
                "kalshi": {
                    "status": "real",
                    "how": "cross-venue price from fetched Kalshi markets, matched above "
                           "0.6 similarity",
                    "requires": "Kalshi markets passed in",
                },
                "manifold": {
                    "status": "real but weak",
                    "how": "public Manifold API; play-money market, confidence capped at 0.45",
                    "requires": "outbound HTTPS",
                },
                "metaculus": {
                    "status": "real but weak",
                    "how": "public Metaculus API; community forecast, confidence capped at 0.50",
                    "requires": "outbound HTTPS",
                },
                "fed_funds": {
                    "status": "unavailable",
                    "why": UNAVAILABLE_SOURCES["fed_funds"],
                },
                "polling": {
                    "status": "unavailable",
                    "why": UNAVAILABLE_SOURCES["polling"],
                },
            },
            "removed_fabrications": [
                "pinnacle used to return market.best_price +/- 0.05 as a 'sharp' price",
                "deribit used to assume BTC was at $90,000",
                "fed_funds used to return 0.35/0.60/0.55 picked by question wording",
                "polling used to return a hardcoded 0.52/0.48 per candidate",
            ],
            "last_unavailable": dict(self.unavailable),
            "fetch_counts": dict(self.fetch_counts),
            "last_run_at": self.last_run_at,
        }


# ---------------------------------------------------------------------------
# Math and helpers
# ---------------------------------------------------------------------------

def lognormal_digital_prob(spot: float, target: float, iv: float,
                           days: float) -> Optional[float]:
    """
    Risk-neutral probability that spot finishes above `target`.

    This is the standard Black-Scholes digital (cash-or-nothing) call price
    with zero rates: N(d2). Rates are omitted because over the horizons these
    markets trade, the carry term is small next to the IV term and inventing
    a rate curve would be worse than omitting it.
    """
    if spot <= 0 or target <= 0 or iv <= 0 or days <= 0:
        return None
    years = days / 365.0
    sigma_sqrt_t = iv * math.sqrt(years)
    if sigma_sqrt_t <= 0:
        return None
    d2 = (math.log(spot / target) - 0.5 * iv * iv * years) / sigma_sqrt_t
    return 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))


def _deribit_confidence(option_count: int, iv: Optional[float]) -> float:
    """
    Confidence from the data actually returned, not from a constant.

    More strikes means a better read on the surface; an absurd IV means the
    data is unreliable however many rows came back.
    """
    if iv is None:
        return 0.0
    coverage = min(1.0, option_count / 200.0)
    iv_ok = 1.0 if 0.10 <= iv <= 2.5 else 0.4
    return round(min(0.9, 0.35 + 0.45 * coverage) * iv_ok, 3)


def _text_similarity(a: str, b: str) -> float:
    """
    Word-overlap similarity for matching a market to a reference question.

    Deliberately simple and conservative: a false match produces a reference
    for the wrong event, which is worse than finding nothing.
    """
    stop = {"will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "and",
            "or", "for", "is", "it", "that", "this", "before", "after", "2024",
            "2025", "2026", "above", "below", "than"}
    wa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in stop}
    wb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if w not in stop}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _best_outcome_index(outcomes: Dict[str, float], market: Market) -> Optional[int]:
    """Index of the book outcome matching the market's own outcome, if any."""
    q = market.question.lower()
    labels = list(outcomes.keys())
    for i, label in enumerate(labels):
        if label.lower() in q:
            return i
    # fall back to the first listed outcome only when there is a single one
    return 0 if len(labels) == 1 else None


@dataclass
class SimpleRef:
    """Lightweight reference object for compatibility with older callers."""
    source: str
    reference_price: float
    confidence: float
    reasoning: str
