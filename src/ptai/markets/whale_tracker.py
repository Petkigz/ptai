"""
Whale tracking - real Polymarket wallet activity, and honest P&L scoring.

Polymarket settles on Polygon and publishes a public activity feed, so wallet
behaviour is observable without an API key. That makes this one of the few
"alternative data" sources the system can actually read rather than pretend to
read.

What was wrong
--------------
The tracker never queried anything. `mock_whale_data()` returned three
handwritten wallets - addresses like `0x1234...smart1`, which is not even a
valid hex address - with invented P&L and win rates, and `get_report()` said so
in a field literally named "mock". That data was then fed into
`alpha_engine.run_full_scan()` and the dashboard, so a scan reported smart and
dumb whales that did not exist.

There was also a second, subtler fabrication inside the signal itself:

    edge_estimate = abs(wallet.score) * 0.05   # smart score 0.8 => 4% edge

A wallet's historical win rate does not imply a percentage edge on a specific
market. Multiplying a score by a chosen constant to produce an "edge estimate"
invents the very number the rest of the system needs to be honest about. The
edge now comes from the price the whale paid versus the price available now,
which is the only edge a copy trade can actually capture - and it is usually
negative, because you are following.

What it does now
----------------
Real HTTP against Polymarket's public data API. No key required. When the API
is unreachable the tracker returns an empty result with a reason and never
substitutes sample wallets.

P&L is computed properly: a prediction-market position's payoff depends on
whether it resolved, so a trade that is still open has no realised P&L and must
not be counted as a win or a loss. The previous version summed `pnl_usd` over
whatever was passed in and treated a zero as a loss, which biases every win
rate downward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Polymarket's public data API. No auth, no key.
DATA_API = "https://data-api.polymarket.com"

# A trade with no known outcome contributes nothing to a win rate. Counting it
# as a loss - which is what treating pnl == 0 as a loss does - systematically
# understates every wallet's skill and makes good wallets look neutral.
RESOLVED_WIN = "win"
RESOLVED_LOSS = "loss"
UNRESOLVED = "open"


@dataclass
class WhaleWallet:
    """Historical record of one wallet, computed from its actual trades."""
    address: str
    total_volume: float
    total_trades: int
    pnl_usd: float
    win_rate: float
    avg_position: float
    is_smart: bool
    is_dumb: bool
    score: float                  # -1 (fade) to +1 (copy)
    recent_trades: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    resolved_trades: int = 0      # how many trades actually have an outcome
    sample_size_ok: bool = False  # enough resolved trades to judge at all
    data_source: str = ""
    is_synthetic: bool = False


@dataclass
class WhaleSignal:
    """One actionable read on one whale trade in one market."""
    market_id: str
    whale_address: str
    whale_score: float
    side: str
    amount_usd: float
    market_price: float
    whale_entry_price: float
    signal_type: str              # copy_smart | fade_dumb
    edge_estimate: float
    should_trade: bool
    reasoning: str
    provenance: str = "live_api"
    is_synthetic: bool = False


@dataclass
class WhaleFeed:
    """Result of a whale scan, including why it may be empty."""
    wallets: List[WhaleWallet] = field(default_factory=list)
    market_trades: Dict[str, List[Dict]] = field(default_factory=dict)
    wallets_by_address: Dict[str, WhaleWallet] = field(default_factory=dict)
    last_error: str = ""
    source: str = ""
    fetched_at: Optional[str] = None
    is_synthetic: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.wallets)


class WhaleTracker:
    """
    Tracks Polymarket whales from the public activity feed.

    Every method degrades to an empty result with a reason. There is no
    synthetic path: a tracker that cannot reach the feed has nothing to say,
    and inventing wallets is what it used to do.
    """

    def __init__(self, min_whale_volume: float = 10000, smart_threshold: float = 0.6,
                 dumb_threshold: float = 0.4, min_resolved_trades: int = 20,
                 timeout: float = 12.0, base_url: str = DATA_API, http_get=None):
        """
        `min_resolved_trades` is the floor for judging a wallet at all. A
        wallet with three resolved trades has a win rate that is noise, and
        copying noise is how you lose money to a small sample.
        """
        self.min_whale_volume = min_whale_volume
        self.smart_threshold = smart_threshold
        self.dumb_threshold = dumb_threshold
        self.min_resolved_trades = min_resolved_trades
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._http_get = http_get
        self.last_error: str = ""
        self.fetch_counts: Dict[str, int] = {"activity": 0, "wallet": 0}

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Fetch JSON or None. Never raises, never invents a payload."""
        url = f"{self.base_url}{path}"
        if self._http_get is not None:
            try:
                return self._http_get(url, params)
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
            logger.debug(f"[whale] {self.last_error}")
            return None

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_recent_activity(self, limit: int = 500,
                              side: str = "") -> List[Dict[str, Any]]:
        """
        Recent public trades across all wallets.

        This is the real feed the previous version claimed to read. Each row
        carries the wallet, the market, the side, the price and the size,
        which is everything needed to find large traders and what they did.
        """
        params: Dict[str, Any] = {"limit": min(limit, 500), "type": "TRADE"}
        if side:
            params["side"] = side
        data = self._get_json("/activity", params)
        if not isinstance(data, list):
            if not self.last_error:
                self.last_error = "activity feed returned no list"
            return []
        self.fetch_counts["activity"] += 1
        return [r for r in data if isinstance(r, dict)]

    def fetch_wallet_activity(self, address: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Full recent trade history for one wallet."""
        if not _is_address(address):
            self.last_error = f"'{address}' is not a valid wallet address"
            return []
        data = self._get_json("/activity", {"user": address, "limit": min(limit, 500),
                                            "type": "TRADE"})
        if not isinstance(data, list):
            if not self.last_error:
                self.last_error = f"no activity returned for {address[:10]}"
            return []
        self.fetch_counts["wallet"] += 1
        return [r for r in data if isinstance(r, dict)]

    def discover_whales(self, limit: int = 500,
                        min_trades: int = 3) -> Dict[str, float]:
        """
        Wallets with the largest traded notional in the recent feed.

        Discovery is by realised notional, not by trade count: one large
        position is a whale, fifty dust trades are not.
        """
        rows = self.fetch_recent_activity(limit=limit)
        volume: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in rows:
            address = str(r.get("proxyWallet") or r.get("user") or r.get("address") or "")
            if not _is_address(address):
                continue
            usd = _usd_of(r)
            if usd <= 0:
                continue
            volume[address] = volume.get(address, 0.0) + usd
            counts[address] = counts.get(address, 0) + 1
        return {a: v for a, v in sorted(volume.items(), key=lambda kv: -kv[1])
                if counts.get(a, 0) >= min_trades}

    def load_whales(self, max_wallets: int = 10, limit: int = 500) -> WhaleFeed:
        """
        Discover large wallets and build a real record for each.

        Returns an empty feed with a reason when the API is unreachable. It
        does not fall back to sample wallets - that is the bug being removed.
        """
        self.last_error = ""
        discovered = self.discover_whales(limit=limit)
        if not discovered:
            return WhaleFeed(last_error=self.last_error or "no whale activity found",
                             source="polymarket_data_api",
                             fetched_at=datetime.now(timezone.utc).isoformat())

        wallets: List[WhaleWallet] = []
        for address in list(discovered)[:max_wallets]:
            rows = self.fetch_wallet_activity(address)
            trades = [normalise_trade(r) for r in rows]
            trades = [t for t in trades if t]
            # pass the discovered notional in so the volume gate sees it
            wallets.append(self.analyze_wallet(address, trades,
                                               known_volume=discovered[address]))

        # Group the recent feed by market so signals can be read per market
        market_trades: Dict[str, List[Dict]] = {}
        for r in self.fetch_recent_activity(limit=limit):
            t = normalise_trade(r)
            if t and t.get("whale_address") in discovered:
                market_trades.setdefault(t["market_id"], []).append(t)

        return WhaleFeed(
            wallets=wallets,
            market_trades=market_trades,
            wallets_by_address={w.address: w for w in wallets},
            source="polymarket_data_api",
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def analyze_wallet(self, address: str, trades: List[Dict],
                       known_volume: Optional[float] = None) -> WhaleWallet:
        """
        Score one wallet from its own trade history.

        Only RESOLVED trades count toward the win rate. A position that has not
        resolved yet has no outcome, and counting it as a loss would drag every
        active wallet's win rate toward zero - which is exactly the kind of
        quiet bias that makes good wallets look mediocre.
        """
        if not trades:
            return WhaleWallet(address=address, total_volume=0.0, total_trades=0,
                               pnl_usd=0.0, win_rate=0.5, avg_position=0.0,
                               is_smart=False, is_dumb=False, score=0.0,
                               recent_trades=[], resolved_trades=0,
                               sample_size_ok=False,
                               reasoning="no trades found for this wallet")

        # `known_volume` is the notional measured across the recent feed. It
        # must be used for the whale gate BEFORE scoring: a wallet's fetched
        # history is a page, not a lifetime, so the history notional alone
        # understates a real whale and would gate them out.
        history_volume = sum(float(t.get("amount_usd", 0.0) or 0.0) for t in trades)
        total_volume = max(history_volume, float(known_volume or 0.0))
        total_trades = len(trades)
        avg_position = history_volume / total_trades if total_trades else 0.0

        resolved = [t for t in trades if t.get("outcome") in (RESOLVED_WIN, RESOLVED_LOSS)]
        wins = sum(1 for t in resolved if t["outcome"] == RESOLVED_WIN)
        resolved_n = len(resolved)
        win_rate = wins / resolved_n if resolved_n else 0.5
        pnl = sum(float(t.get("pnl_usd", 0.0) or 0.0) for t in resolved)

        sample_ok = resolved_n >= self.min_resolved_trades
        volume_ok = total_volume >= self.min_whale_volume

        # A wallet is only called smart or dumb with a real sample behind it.
        is_smart = sample_ok and volume_ok and win_rate >= self.smart_threshold and pnl > 0
        is_dumb = sample_ok and volume_ok and win_rate <= self.dumb_threshold and pnl < 0

        score = self._score(win_rate, pnl, resolved_n)

        reasoning = (
            f"{address[:10]}.. ${total_volume:,.0f} over {total_trades} trades "
            f"({resolved_n} resolved) | PnL ${pnl:,.2f} | win {win_rate*100:.1f}% | "
            f"score {score:+.2f} | "
            + ("sample too small to judge" if not sample_ok else
               ("smart - copy" if is_smart else "dumb - fade" if is_dumb else "neutral")))

        return WhaleWallet(
            address=address, total_volume=round(total_volume, 2),
            total_trades=total_trades, pnl_usd=round(pnl, 2),
            win_rate=round(win_rate, 4), avg_position=round(avg_position, 2),
            is_smart=is_smart, is_dumb=is_dumb, score=round(score, 4),
            recent_trades=trades[-10:], resolved_trades=resolved_n,
            sample_size_ok=sample_ok, data_source="polymarket_data_api",
            reasoning=reasoning)

    def _score(self, win_rate: float, pnl: float, resolved_n: int) -> float:
        """
        Score in [-1, +1].

        Shrunk toward zero when the sample is small, because a 70% win rate
        over five trades is worth far less than 55% over two hundred. Without
        the shrink a lucky newcomer outranks a proven wallet.
        """
        if resolved_n <= 0:
            return 0.0
        # Bayesian-style shrink toward 0.5 win rate using the minimum sample as
        # the strength of the prior.
        shrunk = (win_rate * resolved_n + 0.5 * self.min_resolved_trades) / (
            resolved_n + self.min_resolved_trades)
        base = (shrunk - 0.5) * 2.0
        # P&L adds a bounded tilt so a high-volume winner is not equal to a
        # high-volume loser with the same hit rate.
        pnl_tilt = max(-0.25, min(0.25, pnl / 40000.0))
        return max(-1.0, min(1.0, base + pnl_tilt))

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def get_whale_signals(self, market_id: str, market_price: float,
                          whale_trades: List[Dict],
                          whale_wallets: Dict[str, WhaleWallet],
                          min_amount: float = 100.0,
                          min_score: float = 0.30) -> List[WhaleSignal]:
        """
        Read whale activity on one market.

        The edge is the gap between what the whale paid and what the market
        costs now. That is the only edge a copy trade can capture, and it is
        usually negative once you are following - which is why most signals
        here should NOT be traded. The previous version computed
        `abs(score) * 0.05`, inventing a 4% edge from a win rate.

        `is_smart` / `is_dumb` is the qualification gate - it already requires
        a real sample, real size, and a win rate past the threshold.
        `min_score` is only a small floor to exclude a wallet that barely
        cleared the win-rate gate; it is deliberately NOT a second strict
        threshold, because the score is shrunk toward zero for small samples
        and a high bar there would exclude genuinely good wallets. The score
        sizes the conviction, the edge decides whether to trade.
        """
        signals: List[WhaleSignal] = []

        for trade in whale_trades:
            address = trade.get("whale_address", "")
            wallet = whale_wallets.get(address)
            if wallet is None:
                continue
            # Never act on a wallet whose record is too thin to judge.
            if not wallet.sample_size_ok:
                continue

            side = trade.get("side", "BUY")
            amount = float(trade.get("amount_usd", 0.0) or 0.0)
            entry = float(trade.get("price", 0.0) or 0.0)
            if amount < min_amount or not (0 < entry < 1) or not (0 < market_price < 1):
                continue

            signal_type = None
            if wallet.is_smart and wallet.score >= min_score:
                signal_type = "copy_smart"
            elif wallet.is_dumb and wallet.score <= -min_score:
                signal_type = "fade_dumb"
            if signal_type is None:
                continue

            # Copying: you pay today's price for what the whale bought earlier.
            # Fading: you take the opposite side at today's price.
            if signal_type == "copy_smart":
                edge = entry - market_price      # positive only if you are still cheaper
            else:
                edge = market_price - entry      # fading profits if price has moved up

            should_trade = edge > 0.01
            reasoning = (
                f"{signal_type}: {address[:10]}.. score {wallet.score:+.2f} "
                f"({wallet.win_rate*100:.0f}% over {wallet.resolved_trades} resolved) "
                f"{side} ${amount:,.0f} at {entry:.3f}, market now {market_price:.3f} -> "
                f"captureable edge {edge*100:+.1f}% "
                f"{'(traded)' if should_trade else '(no edge left after the move)'}")

            signals.append(WhaleSignal(
                market_id=market_id, whale_address=address, whale_score=wallet.score,
                side=side, amount_usd=amount, market_price=market_price,
                whale_entry_price=entry, signal_type=signal_type,
                edge_estimate=round(edge, 4), should_trade=should_trade,
                reasoning=reasoning,
                provenance=wallet.data_source or "live_api",
                is_synthetic=wallet.is_synthetic))

        signals.sort(key=lambda s: s.edge_estimate, reverse=True)
        return signals

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """
        Capability report. The old one had a field named "mock" admitting the
        data was invented while the alpha engine consumed it as real.
        """
        return {
            "tracker": "Whale Tracking",
            "source": f"{self.base_url}/activity (public, no key)",
            "method": ("discover wallets by realised notional, score on resolved trades "
                       "only, copy smart and fade dumb only when a captureable edge remains"),
            "thresholds": {
                "min_whale_volume": self.min_whale_volume,
                "smart_win_rate": self.smart_threshold,
                "dumb_win_rate": self.dumb_threshold,
                "min_resolved_trades": self.min_resolved_trades,
            },
            "edge_definition": ("whale entry price vs current market price - the only edge a "
                                "copy trade can capture. NOT a multiple of the wallet score."),
            "caveats": [
                "following a whale means paying after the move, so the edge is usually gone",
                "unresolved trades are excluded, so an active wallet is judged on less data",
                "the public feed is paginated and does not give a complete lifetime history",
            ],
            "removed_fabrication": (
                "mock_whale_data() returned three handwritten wallets with invented P&L; "
                "get_whale_signals computed edge as abs(score)*0.05"),
            "last_error": self.last_error,
            "fetch_counts": dict(self.fetch_counts),
        }


# ---------------------------------------------------------------------------
# Normalisation and helpers
# ---------------------------------------------------------------------------

def normalise_trade(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map one raw activity row onto the internal trade shape.

    Returns None for rows that cannot be used, rather than filling gaps with
    defaults - a trade with no price cannot contribute to an edge calculation.
    """
    address = str(row.get("proxyWallet") or row.get("user") or row.get("address") or "")
    if not _is_address(address):
        return None

    market_id = str(row.get("conditionId") or row.get("market") or row.get("slug") or "")
    price = row.get("price")
    if not isinstance(price, (int, float)) or not (0 < float(price) < 1):
        return None

    size = row.get("size")
    usd = row.get("usdcSize")
    if isinstance(usd, (int, float)) and usd > 0:
        amount = float(usd)
    elif isinstance(size, (int, float)):
        amount = float(size) * float(price)
    else:
        return None
    if amount <= 0:
        return None

    raw_side = str(row.get("side") or "BUY").upper()
    outcome = _outcome_of(row)
    pnl = row.get("pnl") if isinstance(row.get("pnl"), (int, float)) else None

    return {
        "whale_address": address,
        "market_id": market_id,
        "title": str(row.get("title") or ""),
        "side": raw_side,
        "outcome_side": str(row.get("outcome") or ""),
        "amount_usd": round(amount, 2),
        "price": round(float(price), 4),
        "outcome": outcome,
        "pnl_usd": float(pnl) if pnl is not None else None,
        "timestamp": row.get("timestamp"),
        "is_synthetic": False,
    }


def _outcome_of(row: Dict[str, Any]) -> str:
    """
    Whether a trade has resolved, and how.

    Derived from realised P&L when the feed provides it. A row with no P&L
    field is UNRESOLVED, not a loss - conflating the two is the bias that made
    every active wallet look mediocre.
    """
    if "pnl" not in row:
        return UNRESOLVED
    pnl = row.get("pnl")
    if not isinstance(pnl, (int, float)):
        return UNRESOLVED
    if pnl > 0:
        return RESOLVED_WIN
    if pnl < 0:
        return RESOLVED_LOSS
    return UNRESOLVED


def _usd_of(row: Dict[str, Any]) -> float:
    usd = row.get("usdcSize")
    if isinstance(usd, (int, float)) and usd > 0:
        return float(usd)
    size, price = row.get("size"), row.get("price")
    if isinstance(size, (int, float)) and isinstance(price, (int, float)):
        return float(size) * float(price)
    return 0.0


def _is_address(value: str) -> bool:
    """A Polygon/EVM address is 0x plus 40 hex characters."""
    if not isinstance(value, str) or len(value) != 42 or not value.startswith("0x"):
        return False
    try:
        int(value[2:], 16)
        return True
    except ValueError:
        return False
