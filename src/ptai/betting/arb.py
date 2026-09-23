"""
Sports Arbitrage - cross-book and book-vs-exchange surebets, plus the
account-risk reality that makes most textbook arbs untradeable.

Two arb shapes:

1. Book vs book. Two bookmakers disagree enough that the sum of the best
   inverse prices across all outcomes is below 1. Rare, small, and killed
   by stake limits.

2. Book vs exchange (the practical one). Back an outcome at a soft book's
   inflated price and LAY the same outcome on Betfair. The exchange price
   is sharp, the book price is soft, and the gap is where the money is.
   The exchange leg carries commission, so the arithmetic must use
   commission-adjusted lay odds or the 'arb' evaporates.

Account risk is not a footnote. Books limit winning accounts, so arbing at
maximum size gets you limited fast. This module sizes for longevity:
rounded stakes, no obvious max-stake patterns, and a per-book exposure
budget. An arb that costs you the account is a bad arb.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .odds_math import (
    ArbitrageLeg,
    ArbitrageOpportunity,
    analyse_book,
    commission_adjusted_decimal,
    dutch,
    find_arbitrage,
    lay_stake_for_liability,
)
from .sports_data import BookOdds, ConsensusOdds, SportsEvent


# ---------------------------------------------------------------------------
# Event matching across venues
# ---------------------------------------------------------------------------

_TEAM_ALIASES: Dict[str, List[str]] = {
    "man utd": ["manchester united", "man united", "man u", "mnu"],
    "man city": ["manchester city", "mci"],
    "spurs": ["tottenham", "tottenham hotspur"],
    "wolves": ["wolverhampton"],
    "psg": ["paris saint-germain", "paris sg"],
    "bayern": ["bayern munich", "bayern munchen", "fc bayern"],
    "inter": ["inter milan", "internazionale"],
    "atletico": ["atletico madrid", "atl madrid"],
    "barca": ["barcelona", "fc barcelona"],
    "lakers": ["la lakers", "los angeles lakers"],
    "warriors": ["golden state warriors", "gsw"],
    "celtics": ["boston celtics"],
    "niners": ["san francisco 49ers", "49ers"],
    "pats": ["new england patriots"],
    "jags": ["jacksonville jaguars"],
    "newcastle": ["newcastle united"],
    "west ham": ["west ham united"],
}


def normalise_team(name: str) -> str:
    """
    Canonical team name so the same fixture can be matched across feeds.

    Bookmakers and data providers disagree on names constantly ('Man Utd',
    'Manchester United FC', 'MNU'), and a failed match silently turns a real
    arb into two unrelated bets on unrelated events.
    """
    s = name.strip().lower()
    for token in (" fc", " cf", " sc", " ac", "afc", " club", " de ", " bk", " if"):
        s = s.replace(token, " ")
    s = " ".join(s.split())
    for canon, aliases in _TEAM_ALIASES.items():
        if s in aliases or s == canon:
            return canon
    return s


def event_similarity(a: SportsEvent, b: SportsEvent) -> float:
    """0-1 similarity between two fixtures from different providers."""
    if a.league != b.league and a.sport != b.sport:
        return 0.0
    ha, ab = normalise_team(a.home_team), normalise_team(a.away_team)
    hb, bb = normalise_team(b.home_team), normalise_team(b.away_team)
    direct = (SequenceMatcher(None, ha, hb).ratio() + SequenceMatcher(None, ab, bb).ratio()) / 2.0
    swapped = (SequenceMatcher(None, ha, bb).ratio() + SequenceMatcher(None, ab, hb).ratio()) / 2.0
    # swapped home/away is still the same fixture, just listed the other way
    score = max(direct, swapped)
    # kickoff within 6h required, else it is a different fixture
    dt = abs((a.commence_time - b.commence_time).total_seconds()) / 3600.0
    if dt > 6.0:
        score *= 0.3
    return round(score, 4)


def match_events(primary: Sequence[SportsEvent], secondary: Sequence[SportsEvent],
                 threshold: float = 0.85) -> List[Tuple[SportsEvent, SportsEvent, float]]:
    """Greedy best-match pairing between two feeds."""
    pairs: List[Tuple[SportsEvent, SportsEvent, float]] = []
    used_secondary = set()
    for a in primary:
        best, best_score = None, 0.0
        for j, b in enumerate(secondary):
            if j in used_secondary:
                continue
            s = event_similarity(a, b)
            if s > best_score:
                best, best_score = b, s
        if best is not None and best_score >= threshold:
            pairs.append((a, best, best_score))
            used_secondary.add(secondary.index(best))
    return pairs


# ---------------------------------------------------------------------------
# Account risk
# ---------------------------------------------------------------------------

@dataclass
class BookProfile:
    """What we know about one bookmaker's behaviour toward this account."""
    book: str
    is_exchange: bool = False
    commission_pct: float = 0.0
    max_stake_usd: Optional[float] = None
    remaining_limit_usd: Optional[float] = None
    limited: bool = False
    n_arb_bets: int = 0
    n_total_bets: int = 0
    total_staked: float = 0.0
    pnl: float = 0.0
    notes: str = ""

    @property
    def arb_ratio(self) -> float:
        return self.n_arb_bets / self.n_total_bets if self.n_total_bets else 0.0


class BookmakerRiskManager:
    """
    Books close or limit accounts that arb. The manager's job is to keep
    the account alive, which means:

    - Never bet the exact maximum stake repeatedly (the classic tell).
    - Round stakes to non-obvious amounts.
    - Keep the arb share of total bets below a threshold.
    - Cap per-book exposure per day.
    - Prefer exchange legs, which do not limit winners (they take
      commission instead).

    This deliberately leaves money on the table. That is the point.
    """

    def __init__(self, max_arb_ratio: float = 0.35, daily_book_exposure_pct: float = 0.15,
                 round_to: float = 1.0, avoid_round_stakes: bool = True,
                 min_stake_jitter_pct: float = 3.0):
        self.max_arb_ratio = max_arb_ratio
        self.daily_book_exposure_pct = daily_book_exposure_pct
        self.round_to = round_to
        self.avoid_round_stakes = avoid_round_stakes
        self.min_stake_jitter_pct = min_stake_jitter_pct
        self.books: Dict[str, BookProfile] = {}
        self.day_key: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_staked: Dict[str, float] = {}

    def profile(self, book: str) -> BookProfile:
        if book not in self.books:
            self.books[book] = BookProfile(book=book)
        return self.books[book]

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day_key:
            self.day_key = today
            self.daily_staked.clear()

    def can_bet(self, book: str, stake: float, bankroll: float, is_arb: bool = True) -> Tuple[bool, str]:
        self._roll_day()
        p = self.profile(book)
        if p.limited:
            return False, f"{book} is limited/closed to us"
        if p.remaining_limit_usd is not None and stake > p.remaining_limit_usd:
            return False, (f"{book} stake ${stake:.2f} exceeds remaining limit "
                           f"${p.remaining_limit_usd:.2f}")
        if p.max_stake_usd is not None and stake > p.max_stake_usd:
            return False, f"{book} stake ${stake:.2f} exceeds max ${p.max_stake_usd:.2f}"
        if p.n_total_bets >= 8 and p.arb_ratio > self.max_arb_ratio and is_arb:
            return False, (f"{book} arb ratio {p.arb_ratio:.0%} > {self.max_arb_ratio:.0%} - "
                           f"pause arbing here to protect the account")
        day_cap = bankroll * self.daily_book_exposure_pct
        already = self.daily_staked.get(book, 0.0)
        if already + stake > day_cap:
            return False, (f"{book} daily exposure ${already + stake:.2f} would exceed "
                           f"cap ${day_cap:.2f} ({self.daily_book_exposure_pct:.0%} of bankroll)")
        return True, ""

    def round_stake(self, stake: float) -> float:
        """
        Round to a non-tell amount. Repeatedly betting exactly $100.00 is
        the single most obvious arbing signature in a book's risk system.
        """
        if stake <= 0:
            return 0.0
        rounded = round(stake / self.round_to) * self.round_to
        if self.avoid_round_stakes and rounded >= 10:
            # nudge off the round number deterministically by stake size
            nudge = (int(rounded) % 7) - 3
            if nudge == 0:
                nudge = 1
            rounded = max(self.round_to, rounded + nudge)
        return round(rounded, 2)

    def record_bet(self, book: str, stake: float, is_arb: bool = True, pnl: float = 0.0) -> None:
        self._roll_day()
        p = self.profile(book)
        p.n_total_bets += 1
        if is_arb:
            p.n_arb_bets += 1
        p.total_staked += stake
        p.pnl += pnl
        self.daily_staked[book] = self.daily_staked.get(book, 0.0) + stake

    def report(self) -> Dict:
        return {
            "books": {
                b.book: {"limited": b.limited, "arb_ratio": round(b.arb_ratio, 3),
                         "n_bets": b.n_total_bets, "staked": round(b.total_staked, 2),
                         "pnl": round(b.pnl, 2), "max_stake": b.max_stake_usd,
                         "is_exchange": b.is_exchange}
                for b in self.books.values()
            },
            "daily_exposure": {k: round(v, 2) for k, v in self.daily_staked.items()},
            "day": self.day_key,
            "policy": {
                "max_arb_ratio": self.max_arb_ratio,
                "daily_book_exposure_pct": self.daily_book_exposure_pct,
                "avoid_round_stakes": self.avoid_round_stakes,
            },
        }


# ---------------------------------------------------------------------------
# Arb construction
# ---------------------------------------------------------------------------

@dataclass
class ExchangeLayArb:
    event_key: str
    outcome: str
    back_book: str
    back_price: float
    exchange_book: str
    lay_price: float
    back_stake: float
    lay_stake: float
    lay_liability: float
    commission_pct: float
    guaranteed_profit: float
    return_pct: float
    total_outlay: float
    executable: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def book_vs_exchange_arb(event: SportsEvent, consensus: ConsensusOdds,
                         exchange_book: BookOdds, bankroll: float,
                         max_exposure_pct: float = 0.20,
                         min_profit_usd: float = 0.50) -> List[ExchangeLayArb]:
    """
    Back at a soft book, lay the same selection on the exchange.

    Profit if the selection wins:  back_stake*(back_price-1) - lay_stake
    Profit if it loses:            lay_stake*(lay_price-1)*(1-c) - back_stake

    Setting those equal gives the lay stake. If the resulting profit is
    positive at both outcomes, the position is risk-free (before commission
    and before the risk of one leg being voided).
    """
    results: List[ExchangeLayArb] = []
    ex_price_by_outcome = exchange_book.outcomes
    c = exchange_book.commission_pct

    for outcome, ex_lay_price in ex_price_by_outcome.items():
        blockers: List[str] = []
        warnings: List[str] = []

        best_soft_price = 0.0
        best_soft_book = ""
        for b in consensus.books:
            if b is exchange_book or b.is_exchange:
                continue
            price = b.outcomes.get(outcome)
            if price and price > best_soft_price:
                best_soft_price = price
                best_soft_book = b.book

        if best_soft_price <= 1.0:
            continue
        if ex_lay_price <= 1.0:
            blockers.append(f"exchange lay price {ex_lay_price} invalid")

        if not blockers:
            # arb exists when the soft back price beats the de-commissioned lay
            ex_lay_eff = commission_adjusted_decimal(ex_lay_price, c, "lay")
            if best_soft_price <= ex_lay_eff:
                blockers.append(
                    f"no arb: back {best_soft_price:.3f} <= commission-adjusted lay "
                    f"{ex_lay_eff:.3f} at {ex_lay_price:.2f} w/ {c:.0%} commission")

        if blockers:
            results.append(ExchangeLayArb(
                event_key=event.key, outcome=outcome, back_book=best_soft_book,
                back_price=best_soft_price, exchange_book=exchange_book.book,
                lay_price=ex_lay_price, back_stake=0.0, lay_stake=0.0, lay_liability=0.0,
                commission_pct=c, guaranteed_profit=0.0, return_pct=0.0, total_outlay=0.0,
                executable=False, blockers=blockers, warnings=warnings))
            continue

        # Equalise profit across both outcomes, using exchange semantics:
        # a lay's stake is what the layer WINS if the selection loses, and
        # the layer's liability is L*(X-1) if it wins. Commission is charged
        # on the layer's winnings only.
        #
        #   selection wins  -> S*(B-1) - L*(X-1)
        #   selection loses -> L*(1-c) - S
        #
        # Setting them equal gives L = S*B/(X-c).
        B, X = best_soft_price, ex_lay_price
        unit_back = 1.0
        unit_lay = B / (X - c)
        profit_per_unit = unit_back * (B - 1.0) - unit_lay * (X - 1.0)

        if profit_per_unit <= 0:
            blockers.append("equalised profit is not positive after commission")
            results.append(ExchangeLayArb(
                event_key=event.key, outcome=outcome, back_book=best_soft_book,
                back_price=B, exchange_book=exchange_book.book, lay_price=X,
                back_stake=0.0, lay_stake=0.0, lay_liability=0.0, commission_pct=c,
                guaranteed_profit=0.0, return_pct=0.0, total_outlay=0.0,
                executable=False, blockers=blockers, warnings=warnings))
            continue

        # Money at risk is the larger of the back stake and the lay liability.
        liability_per_unit = unit_lay * (X - 1.0)
        risk_per_unit = max(unit_back, liability_per_unit)
        exposure_cap = bankroll * max_exposure_pct
        scale = exposure_cap / risk_per_unit if risk_per_unit > 0 else 0.0

        back_stake = round(unit_back * scale, 2)
        lay_stake = round(unit_lay * scale, 2)
        lay_liab = round(lay_stake * (X - 1.0), 2)
        profit = round(profit_per_unit * scale, 2)
        outlay = round(back_stake + lay_liab, 2)
        return_pct = round(profit / outlay * 100.0, 3) if outlay > 0 else 0.0

        if profit < min_profit_usd:
            warnings.append(f"profit ${profit:.2f} below ${min_profit_usd:.2f} minimum - "
                            f"not worth the settlement and void risk")
        if lay_stake < 2.0:
            warnings.append(f"lay stake ${lay_stake:.2f} below exchange minimum $2.00")

        results.append(ExchangeLayArb(
            event_key=event.key, outcome=outcome, back_book=best_soft_book, back_price=round(B, 4),
            exchange_book=exchange_book.book, lay_price=round(X, 4),
            back_stake=back_stake, lay_stake=lay_stake, lay_liability=lay_liab,
            commission_pct=c, guaranteed_profit=profit, return_pct=return_pct,
            total_outlay=outlay,
            executable=profit >= min_profit_usd and lay_stake >= 2.0 and not blockers,
            blockers=blockers, warnings=warnings))

    return sorted(results, key=lambda r: r.guaranteed_profit, reverse=True)


def multi_book_arb(event: SportsEvent, consensus: ConsensusOdds, bankroll: float,
                   max_exposure_pct: float = 0.20,
                   limits: Optional[Dict[str, float]] = None) -> Optional[ArbitrageOpportunity]:
    """
    Classic cross-book arb using the best price on each outcome, wherever
    that price lives. Sizing respects each book's stake limit.
    """
    limits = limits or {}
    if len(consensus.best_price) < 2:
        return None
    legs: List[ArbitrageLeg] = []
    for outcome, price in consensus.best_price.items():
        book = consensus.best_book.get(outcome, "unknown")
        commission = 0.0
        is_ex = False
        for b in consensus.books:
            if b.book == book:
                commission = b.commission_pct
                is_ex = b.is_exchange
        legs.append(ArbitrageLeg(
            outcome=outcome, venue=book, decimal_odds=price, side="lay" if is_ex else "back",
            max_stake=limits.get(book), commission_pct=commission,
            is_synthetic=consensus.is_synthetic,
        ))
    return find_arbitrage(event.key, legs, bankroll, max_exposure_pct)


# ---------------------------------------------------------------------------
# Hedging an existing position
# ---------------------------------------------------------------------------

def hedge_to_green(position_stake: float, position_odds: float, current_lay_odds: float,
                   commission_pct: float = 0.05) -> Dict:
    """
    Close an open back position by laying it out, locking equal profit on
    either outcome. Used when the price has drifted in your favour, or when
    new information invalidates the original thesis and you would rather
    take a small certain loss than ride it.
    """
    if position_stake <= 0 or position_odds <= 1.0 or current_lay_odds <= 1.0:
        return {"executable": False, "reason": "invalid inputs", "lay_stake": 0.0}

    # L = S*B/(X-c) equalises the two outcomes (see book_vs_exchange_arb).
    denom = current_lay_odds - commission_pct
    if denom <= 0:
        return {"executable": False, "reason": "commission exceeds lay price", "lay_stake": 0.0}
    lay_stake = position_stake * position_odds / denom
    profit_if_wins = position_stake * (position_odds - 1.0) - lay_stake * (current_lay_odds - 1.0)
    profit_if_loses = lay_stake * (1.0 - commission_pct) - position_stake
    return {
        "executable": True,
        "lay_stake": round(lay_stake, 2),
        "profit_if_wins": round(profit_if_wins, 2),
        "profit_if_loses": round(profit_if_loses, 2),
        "equalised": abs(profit_if_wins - profit_if_loses) < 0.01,
        "locked_pnl": round(min(profit_if_wins, profit_if_loses), 2),
        "reason": "" if min(profit_if_wins, profit_if_loses) >= 0 else
                  "hedge locks a loss, but smaller than the open risk",
    }


def dutch_outcome_group(books_by_outcome: Dict[str, Dict[str, float]], total_stake: float,
                        commission_pct: float = 0.0) -> Dict:
    """
    Spread a stake over several outcomes of the same event so the return is
    identical whichever wins. If the best available prices across books sum
    to an overround below 1, that identical return exceeds the stake.

    books_by_outcome = {'TeamA': {'book1': 2.10, 'book2': 2.05}, ...}
    """
    best = {o: max(p.values()) for o, p in books_by_outcome.items() if p}
    result = dutch(best, total_stake, commission_pct)
    if result is None:
        return {"executable": False, "reason": "no odds supplied"}
    return {
        "executable": result.is_arbitrage,
        "stakes": result.stakes,
        "best_prices": best,
        "total_staked": result.total_staked,
        "guaranteed_return": result.guaranteed_return,
        "guaranteed_profit": result.guaranteed_profit,
        "return_pct": result.return_pct,
        "overround": result.overround,
        "reason": "" if result.is_arbitrage else
                  f"overround {result.overround:.4f} >= 1.0, dutching would guarantee a loss",
    }
