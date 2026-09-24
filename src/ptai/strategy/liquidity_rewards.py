"""
Market making and liquidity rewards - quoted from the real book, not assumed.

What was wrong
--------------
The spread was invented in two places:

    spread = 0.02  # assume 2% spread

even though `OrderbookAnalyzer` already computes a real spread from the
market's bid and ask. So the one number that determines whether market making
is profitable was a constant, and it was a constant that happened to make the
strategy look good.

The reward side was worse. `mock_rewards` asserted a rate of 0.1% per day -
36.5% APR - with no source, and the gate that was supposed to filter bad
opportunities tested `reward_apr > 0.1`. With the rate hardcoded to 0.365 that
condition could never fail, so the gate was decorative. `get_report()` even
advertised the whole thing under a key named "mock".

`mid_price = market.best_price` was also wrong in a way that matters for
quoting: `best_price` is one side of the book, not the midpoint. Quoting around
a one-sided price skews every quote by half the spread in a direction that
depends on which side happened to be stored.

And `spread_capture = spread / 2 - maker_fee` presented half the spread as
income with certainty. Capturing the spread requires being filled on both
sides; a maker quoting into informed flow is adversely selected, so the
realised capture is below the gross figure. This module now labels that number
as a gross upper bound rather than presenting it as expected income.

What it does now
----------------
Spread, midpoint and depth come from the market's actual orderbook. When there
is no book, there is no quote - the engine returns `should_quote=False` with
the reason, because a two-sided quote priced around an invented midpoint is how
a market maker gets picked off.

Reward parameters must be supplied by the caller. There are no defaults, so a
deployment that has not configured a reward rate gets `reward_available=False`
and a strategy that can only earn spread, rather than a fabricated 36.5% APR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..markets.base import Market

# A spread this wide cannot be captured profitably after fees and adverse
# selection, whatever the reward rate.
MAX_QUOTABLE_SPREAD = 0.10

# Minimum top-of-book size for a quote to be worth placing. Below this the
# queue position is meaningless.
MIN_TOP_OF_BOOK_USD = 25.0

# Fraction of bankroll at risk in a single market. Market making concentrates
# inventory in one outcome, so the cap is tighter than for directional trades.
DEFAULT_INVENTORY_FRACTION = 0.10


@dataclass
class RewardConfig:
    """
    Liquidity reward parameters for a venue.

    Deliberately has no default rate. The previous version asserted 0.1% per
    day with no source; a deployment that has not measured its venue's reward
    schedule should not be told it is earning 36.5% APR.
    """
    reward_rate_per_day: Optional[float] = None
    maker_fee: float = 0.0
    min_spread_to_qualify: float = 0.02
    min_size_to_qualify: float = 10.0
    active: bool = True
    source: str = ""

    @property
    def reward_available(self) -> bool:
        return bool(self.active and self.reward_rate_per_day is not None
                    and self.reward_rate_per_day > 0)

    @property
    def reward_apr(self) -> float:
        if not self.reward_available:
            return 0.0
        return float(self.reward_rate_per_day) * 365.0


@dataclass
class BookQuote:
    """What the orderbook actually says, or why it could not be read."""
    has_book: bool = False
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    bid_size_usd: float = 0.0
    ask_size_usd: float = 0.0
    source: str = "none"
    warnings: List[str] = field(default_factory=list)


@dataclass
class LiquidityRewardEstimate:
    market_id: str
    reward_per_day_usd: float
    reward_apr: float
    maker_fee: float
    spread_capture_per_trade: float
    inventory_limit: float
    should_provide_liquidity: bool
    reasoning: str
    # New: what the estimate was actually built from.
    reward_available: bool = False
    spread_source: str = "none"
    spread: Optional[float] = None
    gross_capture_is_upper_bound: bool = True
    blockers: List[str] = field(default_factory=list)


@dataclass
class MarketMakingQuote:
    market_id: str
    bid_price: float
    ask_price: float
    bid_size_usd: float
    ask_size_usd: float
    spread: float
    mid_price: float
    inventory: float
    inventory_limit: float
    reward_estimate: float
    should_quote: bool
    reasoning: str
    mid_source: str = "none"
    blockers: List[str] = field(default_factory=list)


class LiquidityRewardsEngine:
    """
    Two-sided quoting sized against the real book.

    Every decision that used to rest on an assumed constant now rests on data
    from the market, and the absence of that data produces a refusal with a
    reason rather than a confident number.
    """

    def __init__(self, bankroll: float = 50.0,
                 rewards: Optional[RewardConfig] = None,
                 inventory_fraction: float = DEFAULT_INVENTORY_FRACTION,
                 max_quotable_spread: float = MAX_QUOTABLE_SPREAD):
        self.bankroll = bankroll
        self.rewards = rewards or RewardConfig()
        self.inventory_fraction = inventory_fraction
        self.max_quotable_spread = max_quotable_spread
        self.inventory_limit = bankroll * inventory_fraction

    # ------------------------------------------------------------------
    # Reading the book
    # ------------------------------------------------------------------

    @staticmethod
    def read_book(market: Market, orderbook: Optional[Dict] = None) -> BookQuote:
        """
        Extract bid, ask, mid and spread from the market's real orderbook.

        Accepts the book either as an explicit argument or from
        `market.raw['orderbook']`. Returns `has_book=False` with a reason when
        neither side can be read, because quoting around an invented midpoint
        is how a maker gets adversely selected.
        """
        book = orderbook
        if book is None:
            raw = getattr(market, "raw", None)
            book = raw.get("orderbook") if isinstance(raw, dict) else None
        if not isinstance(book, dict) or not book:
            return BookQuote(warnings=["no orderbook available for this market"])

        bid = _first_price(book.get("bids") or book.get("bid"))
        ask = _first_price(book.get("asks") or book.get("ask"))
        warnings: List[str] = []

        # A precomputed spread is acceptable only alongside real sides; on its
        # own it is exactly the assumed constant this rewrite removes.
        if bid is None or ask is None:
            fallback_spread = book.get("spread")
            if isinstance(fallback_spread, (int, float)):
                warnings.append("book has a spread figure but no bid/ask sides, "
                                "so no midpoint can be derived")
            else:
                warnings.append("book has no readable bid/ask")
            return BookQuote(has_book=False, spread=(float(fallback_spread)
                                                     if isinstance(fallback_spread, (int, float))
                                                     else None),
                             source="spread_only", warnings=warnings)

        if ask <= bid:
            return BookQuote(has_book=False, bid=bid, ask=ask, source="crossed",
                             warnings=[f"book is crossed (bid {bid} >= ask {ask})"])

        mid = (bid + ask) / 2.0
        spread = ask - bid
        return BookQuote(
            has_book=True, bid=bid, ask=ask, mid=round(mid, 6),
            spread=round(spread, 6), spread_pct=round(spread / mid, 6) if mid > 0 else None,
            bid_size_usd=_top_size_usd(book.get("bids") or book.get("bid"), bid),
            ask_size_usd=_top_size_usd(book.get("asks") or book.get("ask"), ask),
            source="orderbook", warnings=warnings)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def estimate_rewards(self, market: Market, amount_usd: float,
                         orderbook: Optional[Dict] = None) -> LiquidityRewardEstimate:
        """
        What providing liquidity here would earn, and whether it is worth it.

        The spread comes from the book. The reward rate comes from configured
        venue parameters. If either is missing the estimate says so and
        `should_provide_liquidity` is False - it does not fall back to an
        assumed spread or a default APR.
        """
        book = self.read_book(market, orderbook)
        blockers: List[str] = list(book.warnings)

        reward_available = self.rewards.reward_available
        rate = float(self.rewards.reward_rate_per_day or 0.0)
        reward_per_day = amount_usd * rate if reward_available else 0.0
        apr = self.rewards.reward_apr
        if not reward_available:
            blockers.append("no liquidity reward rate configured for this venue - "
                            "spread capture is the only return")

        maker_fee = self.rewards.maker_fee

        if book.has_book and book.spread is not None:
            spread = book.spread
            spread_source = book.source
            # Gross capture assumes BOTH sides fill. A maker quoting into
            # informed flow is adversely selected, so realised capture is below
            # this. It is reported as an upper bound, not as income.
            gross_capture = max(0.0, spread / 2.0 - maker_fee)
        else:
            spread = None
            spread_source = "none"
            gross_capture = 0.0
            blockers.append("no real spread available, so spread capture cannot be estimated")

        # Qualification uses the measured spread, not an assumed one.
        spread_ok = (spread is not None
                     and 0 < spread <= self.max_quotable_spread)
        size_ok = amount_usd >= self.rewards.min_size_to_qualify
        liquid_ok = market.liquidity >= 5000 and market.volume_24h >= 10000

        if not spread_ok and spread is not None:
            blockers.append(f"spread {spread:.4f} is outside the quotable range "
                            f"(0, {self.max_quotable_spread}]")
        if not size_ok:
            blockers.append(f"${amount_usd:.2f} is below the ${self.rewards.min_size_to_qualify:.2f} "
                            "minimum to qualify for rewards")
        if not liquid_ok:
            blockers.append(f"market too thin (liquidity ${market.liquidity:.0f}, "
                            f"24h volume ${market.volume_24h:.0f})")

        # A real return must exist. Earning a fabricated APR was the bug, so
        # without either a configured reward or a measurable spread there is
        # nothing to earn and the answer is no.
        has_return = (reward_available and apr > 0.1) or gross_capture > maker_fee
        if not has_return:
            blockers.append("no positive expected return once fees are deducted")

        should_provide = bool(spread_ok and size_ok and liquid_ok and has_return
                              and amount_usd <= self.inventory_limit)
        if amount_usd > self.inventory_limit:
            blockers.append(f"${amount_usd:.2f} exceeds the ${self.inventory_limit:.2f} "
                            "inventory limit")

        reasoning = (
            f"{market.id}: spread {spread:.4f} from {spread_source} | gross capture "
            f"{gross_capture*100:.2f}%/round trip (upper bound, assumes both sides fill) | "
            f"maker fee {maker_fee*100:.2f}% | rewards "
            f"{apr*100:.1f}% APR" if spread is not None else
            f"{market.id}: no measurable spread") + (
            f" | ${reward_per_day:.4f}/day on ${amount_usd:.2f}" if reward_available
            else " | no reward rate configured") + (
            f" | inventory limit ${self.inventory_limit:.2f} | provide={should_provide}"
            + (f" | blockers: {'; '.join(blockers)}" if blockers else ""))

        return LiquidityRewardEstimate(
            market_id=market.id,
            reward_per_day_usd=round(reward_per_day, 6),
            reward_apr=round(apr, 4),
            maker_fee=maker_fee,
            spread_capture_per_trade=round(gross_capture, 6),
            inventory_limit=round(self.inventory_limit, 2),
            should_provide_liquidity=should_provide,
            reasoning=reasoning,
            reward_available=reward_available,
            spread_source=spread_source,
            spread=(round(spread, 6) if spread is not None else None),
            blockers=blockers)

    # ------------------------------------------------------------------
    # Quoting
    # ------------------------------------------------------------------

    def create_quotes(self, market: Market, inventory: float = 0.0,
                      amount_usd: float = 5.0,
                      orderbook: Optional[Dict] = None) -> MarketMakingQuote:
        """
        Build two-sided quotes around the REAL midpoint.

        Quotes are placed inside the existing book rather than at a fixed
        target spread: quoting wider than the current best bid/ask earns
        nothing, and quoting tighter than the tick guarantees a loss to the
        existing queue.
        """
        book = self.read_book(market, orderbook)
        blockers: List[str] = list(book.warnings)

        if not book.has_book or book.mid is None or book.spread is None:
            # No midpoint means no honest quote. The previous version used
            # market.best_price, which is one side of the book, so every quote
            # was skewed by half a spread in an arbitrary direction.
            fallback_mid = float(getattr(market, "best_price", 0.0) or 0.0)
            blockers.append("no bid/ask to quote around - quote suppressed")
            return MarketMakingQuote(
                market_id=market.id, bid_price=0.0, ask_price=0.0,
                bid_size_usd=0.0, ask_size_usd=0.0, spread=0.0,
                mid_price=fallback_mid, inventory=inventory,
                inventory_limit=self.inventory_limit, reward_estimate=0.0,
                should_quote=False, mid_source="none", blockers=blockers,
                reasoning=(f"{market.id}: cannot quote - no two-sided book. "
                           f"best_price {fallback_mid:.3f} is one side only and is "
                           "not a midpoint."))

        mid = book.mid
        spread = book.spread
        blockers.extend(book.warnings)

        # Quote inside the current book, at a fraction of the observed spread.
        quote_spread = max(spread * 0.5, self.rewards.min_spread_to_qualify * 0.5)
        if quote_spread > self.max_quotable_spread:
            blockers.append(f"required quote spread {quote_spread:.4f} exceeds the "
                            f"{self.max_quotable_spread:.2f} maximum")

        # Inventory skew: lean the quotes toward reducing the position.
        skew = self._inventory_skew(inventory, spread)
        bid_price = mid - quote_spread / 2.0 - skew
        ask_price = mid + quote_spread / 2.0 - skew
        bid_price = round(max(0.01, min(0.99, bid_price)), 4)
        ask_price = round(max(0.01, min(0.99, ask_price)), 4)

        if ask_price <= bid_price:
            blockers.append(f"quotes collapsed (bid {bid_price} >= ask {ask_price}) "
                            "after clamping to the 0.01-0.99 range")

        bid_size, ask_size = self._sizes(inventory, amount_usd)

        reward_est = self.estimate_rewards(market, amount_usd, orderbook)

        if abs(inventory) >= self.inventory_limit:
            blockers.append(f"inventory {inventory:.2f} is at or over the "
                            f"${self.inventory_limit:.2f} limit")
        if market.liquidity < 5000:
            blockers.append(f"liquidity ${market.liquidity:.0f} is below the $5000 minimum")
        if book.bid_size_usd < MIN_TOP_OF_BOOK_USD or book.ask_size_usd < MIN_TOP_OF_BOOK_USD:
            blockers.append(f"top of book too thin (bid ${book.bid_size_usd:.0f}, "
                            f"ask ${book.ask_size_usd:.0f})")

        should_quote = not blockers and spread <= self.max_quotable_spread

        reasoning = (
            f"{market.id}: mid {mid:.4f} (real book, spread {spread:.4f}) | "
            f"bid {bid_price:.4f} ${bid_size:.2f} / ask {ask_price:.4f} ${ask_size:.2f} "
            f"quote spread {quote_spread:.4f} | inventory {inventory:.2f} skew {skew*100:.2f}% "
            f"limit ${self.inventory_limit:.2f} | rewards {reward_est.reward_apr*100:.1f}% APR "
            f"| quote={should_quote}"
            + (f" | blockers: {'; '.join(blockers)}" if blockers else ""))

        return MarketMakingQuote(
            market_id=market.id, bid_price=bid_price, ask_price=ask_price,
            bid_size_usd=round(bid_size, 2), ask_size_usd=round(ask_size, 2),
            spread=round(quote_spread, 6), mid_price=mid, inventory=inventory,
            inventory_limit=round(self.inventory_limit, 2),
            reward_estimate=reward_est.reward_per_day_usd,
            should_quote=should_quote, mid_source=book.source,
            blockers=blockers, reasoning=reasoning)

    def _inventory_skew(self, inventory: float, spread: float) -> float:
        """
        Shift both quotes in the direction that reduces the position.

        Scaled to the observed spread rather than a fixed percentage, so the
        skew is meaningful in a wide book and harmless in a tight one.
        """
        if self.inventory_limit <= 0:
            return 0.0
        ratio = max(-1.0, min(1.0, inventory / self.inventory_limit))
        return ratio * spread * 0.5

    def _sizes(self, inventory: float, amount_usd: float) -> Tuple[float, float]:
        """
        Size each side to unwind the position.

        Long inventory means a smaller bid and a larger ask. Capped so a large
        position cannot produce a zero-sized quote that silently stops trading.
        """
        if self.inventory_limit <= 0:
            return amount_usd, amount_usd
        ratio = max(-1.0, min(1.0, inventory / self.inventory_limit))
        bid_size = amount_usd * max(0.3, 1.0 - ratio)
        ask_size = amount_usd * max(0.3, 1.0 + ratio)
        return bid_size, ask_size

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        return {
            "engine": "Market Making / Liquidity Rewards",
            "spread_source": ("the market's own orderbook (bid/ask). A market with no "
                              "two-sided book is not quoted, because there is no honest "
                              "midpoint to quote around."),
            "mid_source": "(bid + ask) / 2 from the real book, never market.best_price, "
                          "which is one side and would skew every quote by half a spread",
            "rewards": {
                "configured": self.rewards.reward_available,
                "rate_per_day": self.rewards.reward_rate_per_day,
                "apr": round(self.rewards.reward_apr, 4),
                "source": self.rewards.source or "not configured",
                "note": ("no default rate is assumed. Without a configured rate the "
                         "engine reports reward_available=False and the strategy can "
                         "only earn spread."),
            },
            "spread_capture": ("spread/2 minus maker fee, reported as a GROSS UPPER BOUND. "
                               "It assumes both sides fill and ignores adverse selection, "
                               "so realised capture is lower."),
            "inventory": f"{self.inventory_fraction*100:.0f}% of bankroll "
                         f"(${self.inventory_limit:.2f}); quotes skew to unwind",
            "quotable_spread_range": f"(0, {self.max_quotable_spread}]",
            "removed_assumptions": [
                "spread was hardcoded to 0.02 in two places ('# assume 2% spread')",
                "rewards asserted 0.1%/day (36.5% APR) with no source",
                "the reward_apr > 0.1 gate could never fail against that constant",
                "mid_price used market.best_price, which is one side of the book",
                "spread/2 was presented as income rather than a gross upper bound",
            ],
        }


# ---------------------------------------------------------------------------
# Book parsing helpers
# ---------------------------------------------------------------------------

def _first_price(levels: Any) -> Optional[float]:
    """
    Best price from a level list, a [price, size] pair, or a scalar.

    Handles the shapes the venue adapters actually emit rather than assuming
    one, since a misread book silently produces a wrong midpoint.
    """
    if levels is None:
        return None
    if isinstance(levels, (int, float)):
        return float(levels)
    if isinstance(levels, dict):
        for key in ("price", "p"):
            if isinstance(levels.get(key), (int, float)):
                return float(levels[key])
        return None
    if isinstance(levels, (list, tuple)):
        if not levels:
            return None
        first = levels[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, dict):
            for key in ("price", "p"):
                if isinstance(first.get(key), (int, float)):
                    return float(first[key])
            return None
        if isinstance(first, (list, tuple)) and first:
            return float(first[0]) if isinstance(first[0], (int, float)) else None
    return None


def _top_size_usd(levels: Any, price: Optional[float]) -> float:
    """Notional available at the top of the book, in dollars."""
    if levels is None or price is None:
        return 0.0
    entry = None
    if isinstance(levels, (list, tuple)) and levels:
        entry = levels[0]
    elif isinstance(levels, dict):
        entry = levels

    size = None
    if isinstance(entry, dict):
        for key in ("size", "s", "amount"):
            if isinstance(entry.get(key), (int, float)):
                size = float(entry[key])
                break
    elif isinstance(entry, (list, tuple)) and len(entry) > 1:
        if isinstance(entry[1], (int, float)):
            size = float(entry[1])
    if size is None:
        return 0.0
    # A size above 1 is shares, not a probability, so multiply by price to get
    # notional. A size at or below 1 is already a fraction of the outcome.
    return size * price if size > 1.0 else size
