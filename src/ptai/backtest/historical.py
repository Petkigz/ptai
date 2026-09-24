"""
Historical market data - the input the backtest gate actually requires.

The V9 gate refuses to backtest on synthetic markets, which is correct: a
random-walk market tells you nothing about whether a strategy works. But
refusing synthetic data without providing a way to obtain real data just meant
the backtester could never run at all. The CLI command raised ValueError on
every invocation, and two tests have been failing since the gate landed.

This module supplies the missing half: real resolved markets, fetched from
Polymarket's public API, with the resolution outcome and the price at which the
market last traded.

What a backtest honestly needs
------------------------------
A backtest is three things: real prices, real outcomes, and a signal. This
module provides the first two. The third cannot be conjured - without recorded
model outputs from the time, there is no way to know what the strategy believed
then.

So `build_dataset` is explicit about it. With no signal supplied it produces a
NO-SKILL BASELINE where fair_value equals the market price, giving zero edge
and therefore zero trades. That is the correct result: a strategy with no
information should not appear to make money. A baseline that lost to fees by
exactly the fee rate would be the meaningful check, and a fabricated edge that
produced profitable trades would be a lie.

To backtest a real strategy, pass either recorded fair values or a callable
that produces one. The dataset records which mode was used, and a baseline run
is never marked production-grade.

Determinism
-----------
Results are cached to disk keyed by the fetch parameters, so a backtest run
twice on the same data gives the same answer. The engine's latency-drift
simulation is separately seeded; see BacktestEngine.run(seed=...).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

GAMMA_API = "https://gamma-api.polymarket.com"

# A resolved market with no volume is not evidence about anything - nobody
# traded it, so there is no price to have been wrong.
MIN_VOLUME_USD = 1000.0

# Below this the order book was too thin for the recorded price to be a price
# anyone could actually have transacted at.
MIN_LIQUIDITY_USD = 500.0


@dataclass
class ResolvedMarket:
    """
    One market that has resolved.

    `entry_price` and `outcome` are deliberately separate. A closed market's
    `outcomePrices` is its SETTLEMENT - exactly 1 or 0 - which is not a price
    anyone could have transacted at. Treating settlement as the entry price
    (the first version of this module did) means "buy YES at $1.00", which is
    not a trade and made every market fail validation.

    The entry price comes from the last traded price or the final book, which
    is what a backtest can honestly fill against.
    """
    market_id: str
    question: str
    entry_price: float          # last traded YES price before close, 0-1
    outcome: int                # 1 = YES resolved true, 0 = NO
    volume_usd: float
    liquidity_usd: float
    settlement_price: float = 1.0   # from outcomePrices; 1.0 or 0.0
    resolved_at: Optional[datetime] = None
    category: str = ""
    venue_id: str = "polymarket"
    # Cost model. Defaults are Polymarket's published terms; the real spread
    # and depth are unknown historically, so they are marked as assumed rather
    # than silently treated as measured.
    fee_pct: float = 0.0
    spread: Optional[float] = None
    depth_usd: Optional[float] = None
    cost_fields_assumed: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """
        Usable means there was a real price to trade at.

        An entry price of 0 or 1 is not tradeable - it is the settlement
        leaking through, which happens when the feed has no last-trade record
        and only the resolution.
        """
        return (self.volume_usd >= MIN_VOLUME_USD
                and 0.0 < self.entry_price < 1.0
                and self.settlement_price in (0.0, 1.0))


@dataclass
class HistoricalDataset:
    """A dataset ready to feed BacktestEngine.run()."""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    fetched_at: Optional[str] = None
    from_cache: bool = False
    signal_mode: str = "none"       # none | recorded | callable
    is_baseline: bool = True
    warnings: List[str] = field(default_factory=list)
    rejected: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.rows)


class HistoricalDataProvider:
    """
    Fetches and caches resolved markets.

    Fetching is injectable so the whole path is testable without a network:
    pass `fetch` as a callable (url, params) -> parsed JSON or None.
    """

    def __init__(self, base_url: str = GAMMA_API, cache_dir: str = "./data/historical",
                 timeout: float = 20.0, fetch: Optional[Callable] = None,
                 min_volume: float = MIN_VOLUME_USD, min_liquidity: float = MIN_LIQUIDITY_USD):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self._fetch = fetch
        self.min_volume = min_volume
        self.min_liquidity = min_liquidity
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        url = f"{self.base_url}{path}"
        if self._fetch is not None:
            try:
                return self._fetch(url, params)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return None
        try:
            import requests
            r = requests.get(url, params=params or {}, timeout=self.timeout)
            if r.status_code != 200:
                self.last_error = f"{path} returned HTTP {r.status_code}"
                return None
            return r.json()
        except Exception as e:
            self.last_error = f"{path}: {type(e).__name__}: {e}"
            logger.debug(f"[historical] {self.last_error}")
            return None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"resolved_{digest}.json")

    def _load_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
            return payload.get("markets")
        except Exception as e:
            logger.debug(f"[historical] cache read failed: {e}")
            return None

    def _save_cache(self, key: str, markets: List[Dict[str, Any]]) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._cache_path(key), "w") as f:
                json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(),
                           "markets": markets}, f)
        except Exception as e:
            logger.debug(f"[historical] cache write failed: {e}")

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_resolved_markets(self, limit: int = 500, days_back: int = 90,
                               use_cache: bool = True) -> List[ResolvedMarket]:
        """
        Resolved markets from the public API, oldest-resolution filtered.

        Returns [] with a reason recorded when the API is unreachable. It does
        not substitute generated markets - that is precisely what the backtest
        gate exists to prevent.
        """
        self.last_error = ""
        cache_key = f"limit={limit}&days_back={days_back}"
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"[historical] {len(cached)} markets from cache")
                return [self._parse(m) for m in cached if self._parse(m)]

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        data = self._get_json("/markets", {
            "closed": "true", "limit": min(limit, 500),
            "end_date_min": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "endDate", "ascending": "false",
        })
        if not isinstance(data, list):
            if not self.last_error:
                self.last_error = "market list returned no array"
            return []

        out: List[ResolvedMarket] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            parsed = self._parse(raw)
            if parsed:
                out.append(parsed)

        if out and use_cache:
            self._save_cache(cache_key, [r for r in data if isinstance(r, dict)])
        logger.info(f"[historical] fetched {len(out)} resolved markets")
        return out

    def _parse(self, raw: Dict[str, Any]) -> Optional[ResolvedMarket]:
        """
        Map one API row onto a ResolvedMarket.

        Returns None for anything that cannot be used. Gamma encodes
        `outcomes` and `outcomePrices` as JSON strings rather than arrays, so
        both need decoding before they mean anything.
        """
        if not raw.get("closed"):
            return None

        outcomes = _decode(raw.get("outcomes"))
        prices = _decode(raw.get("outcomePrices"))
        if not outcomes or not prices or len(outcomes) != len(prices):
            return None

        # The YES outcome's SETTLEMENT price, and therefore whether YES won.
        try:
            yes_idx = next(i for i, o in enumerate(outcomes)
                           if str(o).strip().lower() in ("yes", "true", "up"))
        except StopIteration:
            yes_idx = 0
        try:
            settlement = float(prices[yes_idx])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= settlement <= 1.0):
            return None

        # A resolved market settles at 1 or 0. Anything else means it voided or
        # the feed has not settled it, and it must not be treated as an outcome.
        if not (settlement >= 0.95 or settlement <= 0.05):
            return None
        outcome = 1 if settlement >= 0.5 else 0

        # The ENTRY price is a different number. Prefer the last traded price;
        # fall back to the final book midpoint. Never use the settlement, which
        # is 1 or 0 and cannot be filled against.
        entry = _float_or_none(raw.get("lastTradePrice"))
        if entry is None:
            bid = _float_or_none(raw.get("bestBid"))
            ask = _float_or_none(raw.get("bestAsk"))
            if bid is not None and ask is not None and ask > bid:
                entry = (bid + ask) / 2.0
        if entry is None:
            return None
        if not (0.0 < entry < 1.0):
            return None

        volume = _float(raw.get("volume") or raw.get("volumeNum"))
        liquidity = _float(raw.get("liquidity") or raw.get("liquidityNum"))

        resolved_at = None
        end = raw.get("endDate") or raw.get("closedTime")
        if isinstance(end, str):
            try:
                resolved_at = datetime.fromisoformat(end.replace("Z", "+00:00"))
            except ValueError:
                resolved_at = None

        assumed: List[str] = []
        fee = raw.get("makerFee")
        fee_pct = _float(fee) if fee is not None else 0.0
        if fee is None:
            assumed.append("fee_pct (Polymarket charges no maker/taker fee on most markets)")

        return ResolvedMarket(
            market_id=str(raw.get("id") or raw.get("conditionId") or ""),
            question=str(raw.get("question") or ""),
            entry_price=entry, outcome=outcome, settlement_price=float(settlement),
            volume_usd=volume, liquidity_usd=liquidity,
            resolved_at=resolved_at,
            category=str(raw.get("category") or ""),
            fee_pct=fee_pct,
            cost_fields_assumed=assumed,
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(self, markets: Optional[List[ResolvedMarket]] = None,
                      signal: Optional[Callable[[ResolvedMarket], Optional[float]]] = None,
                      recorded_fair_values: Optional[Dict[str, float]] = None,
                      limit: int = 500, days_back: int = 90,
                      assume_spread: float = 0.02) -> HistoricalDataset:
        """
        Turn resolved markets into rows for BacktestEngine.run().

        The signal is the part that cannot be invented. Three modes:

          recorded  - fair values captured when the markets were live. The only
                      mode that produces a production-grade backtest.
          callable  - a function scoring each market. Only meaningful if it
                      uses information available before resolution.
          none      - fair_value = entry_price, so edge is exactly zero and no
                      trade is taken. This is a no-skill baseline, and it is
                      labelled as one.

        Passing no signal yields the baseline rather than a fabricated edge,
        because a dataset that produced profitable trades from invented skill
        would be worse than no backtest at all.
        """
        warnings: List[str] = []
        if markets is None:
            markets = self.fetch_resolved_markets(limit=limit, days_back=days_back)
            if not markets:
                return HistoricalDataset(source="polymarket_gamma",
                                         fetched_at=datetime.now(timezone.utc).isoformat(),
                                         warnings=[self.last_error or "no resolved markets available"])

        if recorded_fair_values:
            signal_mode = "recorded"
            is_baseline = False
        elif signal is not None:
            signal_mode = "callable"
            is_baseline = False
        else:
            signal_mode = "none"
            is_baseline = True
            warnings.append(
                "no signal supplied: fair_value is set to the market price, so edge is "
                "zero and no trades are taken. This is a no-skill baseline, not a "
                "strategy evaluation. Supply recorded fair values to backtest a strategy.")

        rows: List[Dict[str, Any]] = []
        rejected = 0
        # Ordered oldest-first so the equity curve runs forward in time.
        ordered = sorted(markets, key=lambda m: (m.resolved_at or datetime.min.replace(tzinfo=timezone.utc)))

        for day, m in enumerate(ordered):
            if not m.is_usable:
                rejected += 1
                continue
            if m.liquidity_usd < self.min_liquidity:
                rejected += 1
                continue

            if signal_mode == "recorded":
                fair = (recorded_fair_values or {}).get(m.market_id)
                if fair is None:
                    rejected += 1
                    continue
            elif signal_mode == "callable":
                try:
                    fair = signal(m)
                except Exception as e:
                    warnings.append(f"signal raised {type(e).__name__}: {e}")
                    fair = None
                if fair is None or not (0.0 < float(fair) < 1.0):
                    rejected += 1
                    continue
            else:
                fair = m.entry_price

            fair = float(fair)
            spread = m.spread if m.spread is not None else assume_spread
            if m.spread is None and "spread" not in m.cost_fields_assumed:
                m.cost_fields_assumed.append("spread")

            rows.append({
                "day": day,
                "market_id": m.market_id,
                "question": m.question,
                "market_price": m.entry_price,
                "fair_value": fair,
                "edge": round(fair - m.entry_price, 6),
                "settlement_price": m.settlement_price,
                "actual_outcome": m.outcome,
                # Confidence is not knowable historically. It is set to the
                # engine's own pass threshold so the gate is not the thing
                # deciding the result - and that is recorded, not hidden.
                "confidence": 0.6,
                "bid": round(max(0.01, m.entry_price - spread / 2), 4),
                "ask": round(min(0.99, m.entry_price + spread / 2), 4),
                "spread": round(spread, 4),
                "depth": m.liquidity_usd,
                "fee_pct": m.fee_pct,
                "venue_id": m.venue_id,
                "category": m.category,
                "data_mode": "historical",
                "is_synthetic": False,
                "cost_fields_assumed": list(m.cost_fields_assumed),
            })

        if rejected:
            warnings.append(f"{rejected} market(s) rejected: unresolvable price, below the "
                            f"${self.min_volume:.0f} volume floor, or below the "
                            f"${self.min_liquidity:.0f} liquidity floor")
        # True in every mode. Recorded fair values make the SIGNAL real; they
        # say nothing about execution cost, which is still assumed.
        warnings.append("spread, depth and latency are not recoverable from historical "
                        "data and are assumed; real execution costs would differ")

        return HistoricalDataset(
            rows=rows, source="polymarket_gamma",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            signal_mode=signal_mode, is_baseline=is_baseline,
            warnings=warnings, rejected=rejected)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        return {
            "provider": "Historical Market Data",
            "source": f"{self.base_url}/markets?closed=true",
            "floors": {"min_volume_usd": self.min_volume,
                       "min_liquidity_usd": self.min_liquidity},
            "cache_dir": self.cache_dir,
            "signal_modes": {
                "recorded": "fair values captured while the market was live - the only "
                            "mode that yields a production-grade backtest",
                "callable": "a scoring function; only valid if it uses pre-resolution "
                            "information",
                "none": "fair_value = market price, edge zero, no trades. A no-skill "
                        "baseline, never marked production-grade",
            },
            "not_recoverable": ("historical order book depth and spread are not published, "
                                "so execution costs in a backtest are assumed and labelled"),
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(value: Any) -> Optional[List[Any]]:
    """Gamma returns outcomes/outcomePrices as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else None
        except (ValueError, TypeError):
            return None
    return None


def _float_or_none(value: Any) -> Optional[float]:
    """Parse a float, returning None (not 0.0) when absent - 0.0 is a real price."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
