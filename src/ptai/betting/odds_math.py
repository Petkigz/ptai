"""
Odds Math - real betting mathematics, no placeholders.

Everything the betting subsystem needs to convert between odds formats,
strip bookmaker vig, size stakes, and detect arbitrage. Pure functions,
fully deterministic, no network, no invented numbers.

Odds formats
------------
decimal  2.50   stake 10 -> returns 25 (profit 15)
fractional 3/2  stake 10 -> profit 15
american +150 / -200

The vig
-------
A two-way book quoted at 1.91/1.91 has implied probs summing to 1.0472,
i.e. 4.72% overround. To get a fair probability you must remove that
overround. Which removal method you use matters: proportional is the
textbook default but is wrong for longshots, the power (multiplicative
odds power) method is the best general default, and Shin corrects for
insider trading which is material in low-liquidity books.

Sources for the de-vig methods:
- Proportional (multiplicative probability): standard.
- Power: solving sum(p_i^k) = 1 for k, Shin (1991) / Clarke et al.
- Shin: assumes a fraction z of the book is insiders.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Odds format conversion
# ---------------------------------------------------------------------------

def decimal_to_fractional(decimal_odds: float) -> str:
    """2.50 -> '3/2', 1.01 -> '1/100'."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    frac = Fraction(decimal_odds - 1.0).limit_denominator(1000)
    return f"{frac.numerator}/{frac.denominator}"


def fractional_to_decimal(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("fractional denominator cannot be 0")
    return round(1.0 + numerator / denominator, 4)


def parse_fractional(fractional: str) -> float:
    """'3/2' -> 2.5"""
    num, _, den = fractional.partition("/")
    return fractional_to_decimal(float(num), float(den or 1))


def american_to_decimal(american: float) -> float:
    if american == 0:
        raise ValueError("american odds cannot be 0")
    if american > 0:
        return round(1.0 + american / 100.0, 4)
    return round(1.0 + 100.0 / abs(american), 4)


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds < 1.01:
        raise ValueError(f"decimal odds must be >= 1.01, got {decimal_odds}")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def parse_american(american: str) -> float:
    """'+150' / '-200' -> decimal."""
    american = american.strip().replace("+", "")
    return american_to_decimal(float(american))


def parse_odds(value: str | float) -> float:
    """
    Best-effort parse of any odds format into decimal.

    Accepts '2.50', '3/2', '+150', '-200', 2.5. American is detected by the
    presence of a sign or by magnitude >= 100 (nobody quotes decimal odds
    of 150.0).
    """
    if isinstance(value, (int, float)):
        v = float(value)
        if abs(v) >= 100:
            return american_to_decimal(v)
        return v
    s = str(value).strip()
    if "/" in s:
        return parse_fractional(s)
    if s.startswith("+") or s.startswith("-"):
        return parse_american(s)
    v = float(s)
    if abs(v) >= 100:
        return american_to_decimal(v)
    return v


def implied_probability(decimal_odds: float) -> float:
    """Gross implied probability, vig included."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def probability_to_decimal(probability: float) -> float:
    if probability <= 0:
        raise ValueError("probability must be > 0")
    return 1.0 / probability


def breakeven_probability(decimal_odds: float) -> float:
    """Same as implied probability - the prob at which EV is exactly 0."""
    return implied_probability(decimal_odds)


def expected_value(probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV per unit staked. p*odds - 1. Positive = value bet."""
    return probability * decimal_odds - 1.0


# ---------------------------------------------------------------------------
# Book overround / margin
# ---------------------------------------------------------------------------

@dataclass
class BookMetrics:
    """Diagnostics for a whole book of prices."""
    prices: List[float]
    gross_implied: List[float]
    overround: float          # sum of implied probs, 1.0 = fair book
    margin_pct: float         # (overround - 1) * 100, the bookmaker's cut
    theoretical_hold_pct: float  # what the book earns on balanced action
    is_arbitrage: bool
    arb_edge_pct: float

    @property
    def is_fair_book(self) -> bool:
        return abs(self.overround - 1.0) < 1e-6


def analyse_book(prices: Sequence[float]) -> BookMetrics:
    """
    Full diagnostics for one book (all outcomes of one event).

    overround  = sum(1/price_i). A 1.06 book on a two-way market means the
                 book collects $106 of implied stake for $100 of exposure.
    margin     = overround - 1, i.e. 6%.
    Hold       = margin / overround, the % of handle the book keeps when
                 stakes are balanced. For 1.06 that is 5.66%, NOT 6%.
    """
    prices = [float(p) for p in prices]
    if len(prices) < 2:
        raise ValueError("a book needs at least 2 prices")
    gross = [1.0 / p for p in prices]
    overround = sum(gross)
    margin = overround - 1.0
    hold = margin / overround if overround else 0.0
    is_arb = overround < 1.0
    return BookMetrics(
        prices=prices,
        gross_implied=gross,
        overround=overround,
        margin_pct=margin * 100.0,
        theoretical_hold_pct=hold * 100.0,
        is_arbitrage=is_arb,
        arb_edge_pct=(1.0 - overround) * 100.0 if is_arb else 0.0,
    )


# ---------------------------------------------------------------------------
# De-vigging
# ---------------------------------------------------------------------------

def devig_proportional(prices: Sequence[float]) -> List[float]:
    """
    Proportional / multiplicative removal: divide each implied prob by the
    overround. Simple, but systematically over-inflates longshot prices
    because it assumes vig is spread evenly across outcomes.
    """
    prices = [float(p) for p in prices]
    overround = sum(1.0 / p for p in prices)
    if overround <= 0:
        raise ValueError("overround must be positive")
    return [(1.0 / p) / overround for p in prices]


def devig_power(prices: Sequence[float], max_iter: int = 200, tol: float = 1e-10) -> List[float]:
    """
    Power method: find exponent k such that sum((1/p_i)^k) = 1, then
    fair_i = (1/p_i)^k.

    This is the best general-purpose de-vig. It removes more vig from
    favourites than from longshots, which matches observed behaviour
    (books shade longshots harder). k > 1 for a normal overround book.
    """
    prices = [float(p) for p in prices]
    gross = [1.0 / p for p in prices]
    overround = sum(gross)

    if abs(overround - 1.0) < 1e-12:
        return list(gross)

    # k > 1 shrinks a book whose overround is > 1; k < 1 grows an arb book.
    lo, hi = 0.01, 20.0

    def f(k: float) -> float:
        return sum(g ** k for g in gross) - 1.0

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # No bracket - fall back to proportional rather than returning garbage.
        return devig_proportional(prices)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol:
            lo = hi = mid
            break
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    fair = [g ** hi for g in gross]
    total = sum(fair)
    if total <= 0:
        return devig_proportional(prices)
    # Bisection already lands on sum==1, but renormalise to kill float drift.
    return [x / total for x in fair]


def devig_shin(prices: Sequence[float], max_iter: int = 200, tol: float = 1e-10) -> Tuple[List[float], float]:
    """
    Shin (1991) de-vig. Assumes a fraction z of bettors are insiders who
    only bet when they know the outcome; the book shades prices to defend
    against them. Returns (fair_probs, z).

    p_i^fair = (sqrt(z^2 + 4(1-z) * p_i^2 / overround) - z) / (2(1-z))

    Best when the book is thin or has a known insider problem. If z cannot
    be solved, falls back to the power method with z=0.
    """
    prices = [float(p) for p in prices]
    gross = [1.0 / p for p in prices]
    overround = sum(gross)
    n = len(prices)

    def fair_for_z(z: float) -> List[float]:
        if z >= 1.0:
            return devig_proportional(prices)
        out = []
        for g in gross:
            disc = z * z + 4.0 * (1.0 - z) * (g * g) / overround
            disc = max(disc, 0.0)
            out.append((math.sqrt(disc) - z) / (2.0 * (1.0 - z)))
        return out

    # Solve for z such that sum(fair) == 1. sum(fair) decreases as z rises.
    lo, hi = 0.0, 0.99

    def total(z: float) -> float:
        return sum(fair_for_z(z))

    if total(0.0) <= 1.0:
        # No insider component needed; power method is the right answer.
        return devig_power(prices), 0.0
    if total(hi) >= 1.0:
        return devig_power(prices), 0.0

    z = 0.0
    for _ in range(max_iter):
        z = (lo + hi) / 2.0
        t = total(z)
        if abs(t - 1.0) < tol:
            break
        if t > 1.0:
            lo = z
        else:
            hi = z

    fair = fair_for_z(z)
    s = sum(fair)
    if s > 0:
        fair = [x / s for x in fair]
    return fair, z


def devig_additive(prices: Sequence[float]) -> List[float]:
    """
    Additive removal: subtract the same absolute amount from every implied
    prob. Tends to produce nonsense (negative probs) on lopsided books;
    included because some books genuinely price this way. Clamped to > 0.
    """
    prices = [float(p) for p in prices]
    gross = [1.0 / p for p in prices]
    overround = sum(gross)
    n = len(prices)
    excess = (overround - 1.0) / n
    fair = [max(g - excess, 1e-6) for g in gross]
    s = sum(fair)
    return [x / s for x in fair]


DEVIG_METHODS = {
    "proportional": devig_proportional,
    "power": devig_power,
    "additive": devig_additive,
}


def devig(prices: Sequence[float], method: str = "power", want_shin_z: bool = False):
    """
    Single entry point. method in {proportional, power, additive, shin}.
    Returns list of fair probabilities (and z if want_shin_z).
    """
    if method == "shin":
        fair, z = devig_shin(prices)
        return (fair, z) if want_shin_z else fair
    fn = DEVIG_METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown devig method {method!r}; use one of {list(DEVIG_METHODS) + ['shin']}")
    return fn(prices)


def fair_decimal_odds(prices: Sequence[float], method: str = "power") -> List[float]:
    """De-vigged prices expressed back as decimal odds."""
    return [probability_to_decimal(p) for p in devig(prices, method)]


# ---------------------------------------------------------------------------
# Commission-aware pricing (betting exchanges)
# ---------------------------------------------------------------------------

def commission_adjusted_decimal(decimal_odds: float, commission_pct: float, side: str = "back") -> float:
    """
    Exchanges (Betfair/Betdaq/Smarkets) charge commission on NET WINNINGS,
    not on stake. A back of 2.00 at 5% commission pays 0.95 net per unit,
    so the effective decimal odds are 1 + (odds-1)*(1-c) = 1.95.

    A lay is the mirror image: the effective lay odds rise, because you keep
    less of the profit you make when the selection loses.
    """
    if not 0.0 <= commission_pct < 1.0:
        raise ValueError("commission must be in [0, 1)")
    side = side.lower()
    if side == "back":
        return 1.0 + (decimal_odds - 1.0) * (1.0 - commission_pct)
    if side == "lay":
        if decimal_odds <= 1.0:
            raise ValueError("lay odds must be > 1.0")
        return 1.0 + (decimal_odds - 1.0) / (1.0 - commission_pct)
    raise ValueError("side must be 'back' or 'lay'")


def breakeven_with_commission(probability: float, commission_pct: float) -> float:
    """Minimum decimal odds needed to break even after commission on winnings."""
    if probability <= 0:
        raise ValueError("probability must be > 0")
    net_needed = 1.0 / (1.0 - commission_pct)
    return 1.0 + (net_needed - 1.0) / probability


# ---------------------------------------------------------------------------
# Betfair price ladder
# ---------------------------------------------------------------------------

# (upper_bound_inclusive, tick_size)
_BETFAIR_LADDER: List[Tuple[float, float]] = [
    (2.0, 0.01),
    (3.0, 0.02),
    (4.0, 0.05),
    (6.0, 0.10),
    (10.0, 0.20),
    (20.0, 0.50),
    (30.0, 1.00),
    (50.0, 2.00),
    (100.0, 5.00),
    (1000.0, 10.00),
]


def betfair_tick_size(price: float) -> float:
    """Betfair increments prices on a non-uniform ladder."""
    for upper, tick in _BETFAIR_LADDER:
        if price <= upper + 1e-9:
            return tick
    return 10.00


def _build_betfair_ladder() -> List[float]:
    """
    Materialise the real Betfair price ladder.

    It cannot be derived from tick_size alone, because the tick changes at
    each band boundary: 1.99 -> 2.00 is a 0.01 step, but 2.00 -> 2.02 is a
    0.02 step, so 2.01 is not a legal price at all. Walking the ladder by
    adding the current tick produces prices you cannot actually order at.
    """
    ladder: List[float] = []
    bands = [(1.01, 2.00, 0.01), (2.00, 3.00, 0.02), (3.00, 4.00, 0.05),
             (4.00, 6.00, 0.10), (6.00, 10.00, 0.20), (10.00, 20.00, 0.50),
             (20.00, 30.00, 1.00), (30.00, 50.00, 2.00), (50.00, 100.00, 5.00),
             (100.00, 1000.00, 10.00)]
    for lo, hi, tick in bands:
        p = lo
        steps = int(round((hi - lo) / tick))
        for i in range(steps + 1):
            v = round(lo + i * tick, 4)
            if not ladder or v > ladder[-1] + 1e-9:
                ladder.append(v)
    return ladder


_BETFAIR_LADDER_PRICES: List[float] = _build_betfair_ladder()


def snap_to_ladder(price: float, direction: str = "nearest", ladder: str = "betfair") -> float:
    """
    Snap a model price to the nearest legal price you could actually order at.

    direction 'down' is the conservative choice when quoting a back (better
    to be paid less than to cross the spread), 'up' for lays.
    """
    if ladder != "betfair":
        raise ValueError(f"unknown ladder {ladder!r}")
    prices = _BETFAIR_LADDER_PRICES
    price = max(prices[0], min(price, prices[-1]))
    idx = bisect_left(prices, price)
    if idx >= len(prices):
        return prices[-1]
    if abs(prices[idx] - price) < 1e-9:
        return prices[idx]
    if direction == "down":
        return prices[max(0, idx - 1)]
    if direction == "up":
        return prices[min(len(prices) - 1, idx)]
    # nearest
    lower = prices[max(0, idx - 1)]
    upper = prices[min(len(prices) - 1, idx)]
    return lower if (price - lower) <= (upper - price) else upper


def next_ladder_step(price: float, direction: str = "up") -> float:
    """One legal price better (up) or worse (down). 2.00 -> 2.02, not 2.01."""
    prices = _BETFAIR_LADDER_PRICES
    snapped = snap_to_ladder(price, "nearest")
    idx = prices.index(snapped)
    if direction == "up":
        return prices[min(len(prices) - 1, idx + 1)]
    return prices[max(0, idx - 1)]


def ladder_walk(price: float, ticks: int) -> float:
    """Move N legal prices up (positive) or down (negative)."""
    p = price
    step = "up" if ticks >= 0 else "down"
    for _ in range(abs(int(ticks))):
        p = next_ladder_step(p, step)
    return p


# ---------------------------------------------------------------------------
# Back / lay liability
# ---------------------------------------------------------------------------

def back_liability(stake: float, decimal_odds: float) -> float:
    """Back: you risk exactly your stake."""
    return stake


def back_profit(stake: float, decimal_odds: float, commission_pct: float = 0.0) -> float:
    win = stake * (decimal_odds - 1.0)
    return win * (1.0 - commission_pct)


def lay_liability(stake: float, decimal_odds: float) -> float:
    """
    Lay: you act as the bookmaker. If the selection WINS you pay
    stake * (odds - 1). That liability, not the stake, is the money at risk.
    """
    return stake * (decimal_odds - 1.0)


def lay_stake_for_liability(liability: float, decimal_odds: float) -> float:
    """Inverse of lay_liability - size a lay by the money you are willing to lose."""
    if decimal_odds <= 1.0:
        raise ValueError("lay odds must be > 1.0")
    return liability / (decimal_odds - 1.0)


def lay_profit(stake: float, commission_pct: float = 0.0) -> float:
    """If the laid selection loses you keep the stake, less commission."""
    return stake * (1.0 - commission_pct)


def green_up_stake(back_stake: float, back_odds: float, lay_odds: float,
                   commission_pct: float = 0.0, target_profit_pct: float = 0.0) -> Optional[float]:
    """
    Cash-out ("green up") a matched back by laying the same selection.

    Exchange semantics, spelled out because getting them wrong loses money:
    a lay's stake is what the layer WINS if the selection loses, and the
    layer's liability is stake*(odds-1) if it wins. Commission is charged
    on the layer's winnings only.

        selection wins  -> S*(B-1) - L*(X-1)
        selection loses -> L*(1-c) - S

    Equalising those gives L = S*B / (X - c). That L exists with positive
    profit only when B > 1 + (X-1)/(1-c), i.e. only when the back price
    beat the commission-adjusted lay price. Otherwise this returns None:
    there is no green-up, and pretending otherwise invents a profit.

    `target_profit_pct` gives back a fraction of the equalised profit to
    shrink the lay, because lay liability - not stake - is what the account
    reserves.
    """
    if lay_odds <= 1.0 or back_odds <= 1.0 or back_stake <= 0:
        return None
    c = commission_pct
    denom = lay_odds - c
    if denom <= 0:
        return None

    equal_lay = back_stake * back_odds / denom
    profit_if_wins = back_stake * (back_odds - 1.0) - equal_lay * (lay_odds - 1.0)
    profit_if_loses = equal_lay * (1.0 - c) - back_stake
    equal_profit = min(profit_if_wins, profit_if_loses)
    if equal_profit <= 0:
        # The lay would cost more than the position is worth: no green-up.
        return None

    if target_profit_pct <= 0:
        return round(equal_lay, 4)

    # Shrinking L raises the win-case and lowers the lose-case, so the
    # binding constraint becomes the lose side.
    target_profit = equal_profit * (1.0 - max(0.0, min(1.0, target_profit_pct)))
    reduced = (target_profit + back_stake) / (1.0 - c)
    return round(min(equal_lay, reduced), 4)


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(probability: float, decimal_odds: float, fraction: float = 1.0,
                   cap: float = 0.06) -> float:
    """
    Kelly for decimal odds: f* = (p*b - q) / b where b = odds - 1.

    `fraction` scales for fractional Kelly (0.25 quarter-Kelly is standard
    practice - full Kelly is ruinously volatile with estimation error).
    `cap` is the hard ceiling on a single bet, matching the project's 6%
    max-position rule. Returns 0 when the bet has no edge.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    f = (probability * b - q) / b
    if f <= 0:
        return 0.0
    f *= max(0.0, fraction)
    return min(f, cap)


def kelly_stake(bankroll: float, probability: float, decimal_odds: float,
                fraction: float = 1.0, cap: float = 0.06) -> float:
    return bankroll * kelly_fraction(probability, decimal_odds, fraction, cap)


# ---------------------------------------------------------------------------
# Dutching (split a stake across several outcomes for equal profit)
# ---------------------------------------------------------------------------

@dataclass
class DutchResult:
    stakes: Dict[str, float]
    total_staked: float
    guaranteed_return: float   # same payout whichever selected outcome hits
    guaranteed_profit: float
    return_pct: float
    is_arbitrage: bool
    overround: float


def dutch(odds_by_outcome: Dict[str, float], total_stake: float,
          commission_pct: float = 0.0) -> Optional[DutchResult]:
    """
    Split total_stake across all listed outcomes so the return is identical
    whichever one wins. If the book overround is below 1.0 that identical
    return exceeds the stake and the bet is a risk-free arbitrage.
    """
    if not odds_by_outcome or total_stake <= 0:
        return None
    prices = [commission_adjusted_decimal(o, commission_pct, "back") for o in odds_by_outcome.values()]
    metrics = analyse_book(prices)
    if metrics.overround <= 0:
        return None

    stakes: Dict[str, float] = {}
    for (name, raw), adj in zip(odds_by_outcome.items(), prices):
        implied = 1.0 / adj
        stakes[name] = round(total_stake * implied / metrics.overround, 4)

    guaranteed_return = 0.0
    for (name, raw) in odds_by_outcome.items():
        guaranteed_return = stakes[name] * commission_adjusted_decimal(raw, commission_pct, "back")
        break
    profit = guaranteed_return - sum(stakes.values())
    return DutchResult(
        stakes=stakes,
        total_staked=round(sum(stakes.values()), 4),
        guaranteed_return=round(guaranteed_return, 4),
        guaranteed_profit=round(profit, 4),
        return_pct=round(profit / max(sum(stakes.values()), 1e-9) * 100.0, 4),
        is_arbitrage=profit > 0,
        overround=round(metrics.overround, 6),
    )


# ---------------------------------------------------------------------------
# Arbitrage
# ---------------------------------------------------------------------------

@dataclass
class ArbitrageLeg:
    outcome: str
    venue: str
    decimal_odds: float
    side: str = "back"          # back | lay
    stake: float = 0.0
    liability: float = 0.0
    max_stake: Optional[float] = None   # bookmaker limit, None = unknown
    commission_pct: float = 0.0
    is_synthetic: bool = False


@dataclass
class ArbitrageOpportunity:
    event_key: str
    legs: List[ArbitrageLeg]
    overround: float
    edge_pct: float                 # (1 - overround) * 100
    stakes: Dict[str, float]
    total_staked: float
    guaranteed_profit: float
    return_pct: float
    limiting_leg: Optional[str] = None      # which leg's max_stake caps the arb
    executable: bool = False
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def find_arbitrage(event_key: str, legs: Sequence[ArbitrageLeg], total_bankroll: float,
                   max_total_exposure_pct: float = 0.20) -> Optional[ArbitrageOpportunity]:
    """
    Detect and size a risk-free arbitrage across venues.

    Sizing rule: stake each leg in proportion to 1/effective_odds, which
    equalises the return. The arb is only as big as its smallest leg, so
    we scale down to the most restrictive bookmaker limit. Total exposure
    is capped as a fraction of bankroll because an arb ties up capital in
    every leg until settlement.

    Note on 'side': on an exchange a lay at odds X is economically a back
    at odds X/(X-1) of the opposite outcome. Callers should already have
    expressed each leg as covering a distinct outcome of the same event.
    """
    if len(legs) < 2:
        return None

    effective: List[float] = []
    blockers: List[str] = []
    warnings: List[str] = []
    for leg in legs:
        if leg.decimal_odds <= 1.0:
            blockers.append(f"{leg.outcome}@{leg.venue}: odds {leg.decimal_odds} <= 1.0, not bettable")
            continue
        eff = commission_adjusted_decimal(leg.decimal_odds, leg.commission_pct, "back")
        effective.append(eff)
        if leg.is_synthetic:
            blockers.append(f"{leg.outcome}@{leg.venue}: synthetic price, arb not real")
        if leg.side not in ("back", "lay"):
            blockers.append(f"{leg.outcome}@{leg.venue}: unknown side {leg.side}")

    if blockers:
        return ArbitrageOpportunity(
            event_key=event_key, legs=list(legs), overround=float("nan"), edge_pct=0.0,
            stakes={}, total_staked=0.0, guaranteed_profit=0.0, return_pct=0.0,
            executable=False, blockers=blockers, warnings=warnings,
        )

    metrics = analyse_book(effective)
    if not metrics.is_arbitrage:
        return ArbitrageOpportunity(
            event_key=event_key, legs=list(legs), overround=round(metrics.overround, 6),
            edge_pct=0.0, stakes={}, total_staked=0.0, guaranteed_profit=0.0, return_pct=0.0,
            executable=False, blockers=[f"no arb: overround {metrics.overround:.4f} >= 1.0"],
            warnings=warnings,
        )

    inv = [1.0 / e for e in effective]
    unit_total = sum(inv)
    # unit arb: stake inv_i per leg, total unit_total, guaranteed return 1.0
    unit_profit = 1.0 - unit_total
    unit_return_pct = unit_profit / unit_total * 100.0

    # Scale up to bankroll exposure cap, then down to the tightest leg limit.
    exposure_cap = total_bankroll * max_total_exposure_pct
    scale = exposure_cap / unit_total

    limiting_leg: Optional[str] = None
    for leg, w in zip(legs, inv):
        if leg.max_stake is None:
            warnings.append(f"{leg.outcome}@{leg.venue}: no max stake known, sizing on exposure cap only")
            continue
        allowed_scale = leg.max_stake / w if w > 0 else float("inf")
        if allowed_scale < scale:
            scale = allowed_scale
            limiting_leg = f"{leg.outcome}@{leg.venue} max ${leg.max_stake:.2f}"

    if scale <= 0:
        return ArbitrageOpportunity(
            event_key=event_key, legs=list(legs), overround=round(metrics.overround, 6),
            edge_pct=round(metrics.arb_edge_pct, 4), stakes={}, total_staked=0.0,
            guaranteed_profit=0.0, return_pct=0.0, executable=False,
            blockers=["scale is zero - a leg limit is zero"], warnings=warnings,
        )

    stakes: Dict[str, float] = {}
    for leg, w in zip(legs, inv):
        stakes[f"{leg.outcome}@{leg.venue}"] = round(w * scale, 4)

    total = sum(stakes.values())
    profit = unit_profit * scale

    if total > exposure_cap + 1e-6:
        warnings.append(f"sizing clipped: ${total:.2f} vs cap ${exposure_cap:.2f}")
    if profit < 0.10:
        warnings.append(f"profit ${profit:.2f} below $0.10, likely not worth the fees and account risk")

    return ArbitrageOpportunity(
        event_key=event_key,
        legs=list(legs),
        overround=round(metrics.overround, 6),
        edge_pct=round(metrics.arb_edge_pct, 4),
        stakes=stakes,
        total_staked=round(total, 4),
        guaranteed_profit=round(profit, 4),
        return_pct=round(unit_return_pct, 4),
        limiting_leg=limiting_leg,
        executable=profit > 0 and total <= exposure_cap + 1e-6,
        blockers=blockers,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Closing line value
# ---------------------------------------------------------------------------

def closing_line_value(entry_odds: float, closing_odds: float) -> float:
    """
    CLV = the EV of your bet, priced at the closing line.

        CLV = p_close * entry_odds - 1 = entry_odds / closing_odds - 1

    Backed at 2.20 that closed at 2.00 gives +10%: you took a price 10%
    better than the market's final view. This is the standard definition
    and the sign matters - a shorter close means the market moved toward
    you.

    Positive CLV is the strongest short-sample evidence of real edge,
    because closing prices are the market's best estimate. A process that
    beats the close consistently profits long run even when short-run
    results are negative, and vice versa.
    """
    if entry_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0
    return entry_odds / closing_odds - 1.0


def is_sharp_process(clv_samples: Sequence[float], min_samples: int = 20) -> Tuple[bool, float, int]:
    """
    Judge a process on CLV, not on realised PnL.

    Returns (is_sharp, mean_clv_pct, n). Requires min_samples before it
    will say anything - 5 bets of good luck is not a process.
    """
    samples = [s for s in clv_samples if s is not None]
    n = len(samples)
    if n < min_samples:
        return False, 0.0, n
    mean_clv = sum(samples) / n * 100.0
    return mean_clv > 0.5, round(mean_clv, 4), n


def no_vig_edge(model_prob: float, book_prices: Sequence[float], method: str = "power") -> float:
    """
    Edge of a model probability against a de-vigged book.

    This is the honest version of 'reference odds edge': it compares the
    model to the market's vig-free consensus rather than to a fabricated
    sharp price.
    """
    fair = devig(book_prices, method)
    if not fair:
        return 0.0
    market_fair = fair[0]
    return model_prob - market_fair
