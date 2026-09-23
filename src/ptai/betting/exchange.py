"""
Betting Exchange Engine - back/lay execution semantics, positions, market making.

An exchange is NOT a bookmaker. You match against other users, so you can
take either side:
  BACK = bet the outcome happens   (risk stake, win stake*(odds-1))
  LAY  = bet the outcome does not  (risk stake*(odds-1), win stake)

That asymmetry is the whole point: lay capability is what makes arbitrage,
market making and in-play hedging possible, none of which exist at a
traditional sportsbook. This module models the money correctly so nothing
downstream can size a lay as if it were a back.

Orders are only matched to available liquidity, so partial fills are the
normal case and every method here tracks matched vs unmatched separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .odds_math import (
    back_profit,
    betfair_tick_size,
    commission_adjusted_decimal,
    green_up_stake,
    kelly_fraction,
    kelly_stake,
    lay_liability,
    lay_stake_for_liability,
    next_ladder_step,
    snap_to_ladder,
)


class BetSide(str, Enum):
    BACK = "back"
    LAY = "lay"


class OrderStatus(str, Enum):
    PENDING = "pending"          # submitted, nothing matched yet
    PARTIALLY_MATCHED = "partially_matched"
    MATCHED = "matched"          # fully matched
    UNMATCHED_CANCELLED = "unmatched_cancelled"
    REJECTED = "rejected"
    VOIDED = "voided"            # market voided (abandoned event etc)


@dataclass
class PriceLevel:
    price: float
    size: float


@dataclass
class ExchangeBook:
    """Order book for one exchange market. back_to_bet = available to back (i.e. others laying)."""
    market_id: str
    back_levels: List[PriceLevel] = field(default_factory=list)   # descending price
    lay_levels: List[PriceLevel] = field(default_factory=list)    # ascending price
    last_updated: Optional[datetime] = None
    is_real: bool = False

    @property
    def best_back(self) -> Optional[float]:
        return self.back_levels[0].price if self.back_levels else None

    @property
    def best_lay(self) -> Optional[float]:
        return self.lay_levels[0].price if self.lay_levels else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_back is None or self.best_lay is None:
            return None
        return self.best_lay - self.best_back

    def available_to_back(self, max_price: Optional[float] = None) -> float:
        return sum(l.size for l in self.back_levels if max_price is None or l.price >= max_price)

    def available_to_lay(self, min_price: Optional[float] = None) -> float:
        return sum(l.size for l in self.lay_levels if min_price is None or l.price <= min_price)

    def walk(self, side: BetSide, target_size: float) -> Tuple[float, float]:
        """
        Consume liquidity down the book until target_size is filled.

        Returns (filled_size, average_price). This is the honest fill model:
        a $500 bet into a book with $120 at the top price does not get the
        top price, it gets a weighted average further down the ladder.
        """
        levels = self.back_levels if side == BetSide.BACK else self.lay_levels
        filled = 0.0
        cost = 0.0
        for level in levels:
            if filled >= target_size:
                break
            take = min(level.size, target_size - filled)
            if take <= 0:
                continue
            filled += take
            cost += take * level.price
        if filled <= 0:
            return 0.0, 0.0
        return filled, cost / filled


@dataclass
class ExchangeBet:
    bet_id: str
    market_id: str
    selection: str
    side: BetSide
    requested_price: float
    requested_size: float
    matched_size: float = 0.0
    average_matched_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    commission_pct: float = 0.05
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    matched_at: Optional[datetime] = None
    void_reason: str = ""

    @property
    def unmatched_size(self) -> float:
        return max(0.0, self.requested_size - self.matched_size)

    @property
    def exposure(self) -> float:
        """
        Money at risk if this bet LOSES. For a lay that is the liability,
        which is larger than the stake whenever odds > 2.
        """
        if self.matched_size <= 0:
            return 0.0
        if self.side == BetSide.BACK:
            return self.matched_size
        return lay_liability(self.matched_size, self.average_matched_price)

    @property
    def net_win(self) -> float:
        """Profit if this bet wins, after commission."""
        if self.matched_size <= 0:
            return 0.0
        if self.side == BetSide.BACK:
            return back_profit(self.matched_size, self.average_matched_price, self.commission_pct)
        return self.matched_size * (1.0 - self.commission_pct)


@dataclass
class ExchangePosition:
    """Net position in one selection, aggregated across all bets."""
    market_id: str
    selection: str
    back_matched: float = 0.0
    lay_matched: float = 0.0
    back_avg_price: float = 0.0
    lay_avg_price: float = 0.0
    commission_pct: float = 0.05

    @property
    def net_back(self) -> float:
        """Positive = net long, negative = net short (net lay)."""
        return self.back_matched - self.lay_matched

    def profit_if_wins(self) -> float:
        if self.back_matched > 0:
            return back_profit(self.back_matched, self.back_avg_price, self.commission_pct)
        return 0.0

    def loss_if_wins(self) -> float:
        if self.lay_matched > 0:
            return lay_liability(self.lay_matched, self.lay_avg_price)
        return 0.0

    def pnl_if(self, selection_wins: bool) -> float:
        """Net P&L of this position for one binary resolution."""
        if selection_wins:
            return self.profit_if_wins() - self.loss_if_wins()
        # selection loses: back loses its stake, lay keeps the stake less commission
        lost_back = self.back_matched
        won_lay = self.lay_matched * (1.0 - self.commission_pct)
        return won_lay - lost_back

    def worst_case(self) -> float:
        return min(self.pnl_if(True), self.pnl_if(False))

    def is_hedged(self) -> bool:
        return self.worst_case() >= 0.0


class ExchangeAccount:
    """
    Tracks an exchange wallet: balance, matched/unmatched exposure, and the
    liability that a lay consumes. Exchanges reserve lay liability against
    your available balance, so you cannot size lays as if they cost their
    stake.
    """

    def __init__(self, venue_id: str, balance: float, commission_pct: float = 0.05,
                 min_bet: float = 2.0, currency: str = "USD"):
        self.venue_id = venue_id
        self.starting_balance = balance
        self.balance = balance
        self.commission_pct = commission_pct
        self.min_bet = min_bet
        self.currency = currency
        self.bets: List[ExchangeBet] = []
        self.positions: Dict[str, ExchangePosition] = {}

    @property
    def reserved_liability(self) -> float:
        """Total exposure across all unmatched-and-live bets."""
        return sum(b.exposure for b in self.bets if b.status in (
            OrderStatus.MATCHED, OrderStatus.PARTIALLY_MATCHED, OrderStatus.PENDING))

    @property
    def available_balance(self) -> float:
        return max(0.0, self.balance - self.reserved_liability)

    def _position(self, market_id: str, selection: str) -> ExchangePosition:
        key = f"{market_id}:{selection}"
        if key not in self.positions:
            self.positions[key] = ExchangePosition(
                market_id=market_id, selection=selection, commission_pct=self.commission_pct)
        return self.positions[key]

    def _apply_fill(self, bet: ExchangeBet, filled: float, avg_price: float) -> None:
        if filled <= 0:
            return
        pos = self._position(bet.market_id, bet.selection)
        prev_matched = pos.back_matched if bet.side == BetSide.BACK else pos.lay_matched
        prev_avg = pos.back_avg_price if bet.side == BetSide.BACK else pos.lay_avg_price
        total = prev_matched + filled
        if total > 0:
            new_avg = (prev_matched * prev_avg + filled * avg_price) / total
        else:
            new_avg = avg_price
        if bet.side == BetSide.BACK:
            pos.back_matched, pos.back_avg_price = total, new_avg
        else:
            pos.lay_matched, pos.lay_avg_price = total, new_avg

    def submit(self, bet_id: str, market_id: str, selection: str, side: BetSide,
               price: float, stake: float, book: Optional[ExchangeBook] = None,
               persist_unmatched: bool = True) -> ExchangeBet:
        """
        Submit an order and match it against the book.

        A BACK consumes lay_levels (someone else laying at that price or
        better); a LAY consumes back_levels. If the book is not supplied the
        order stays PENDING (unmatched) rather than pretending to fill - a
        simulated fill with no liquidity behind it is exactly the kind of
        invented data that must never reach a live decision.
        """
        price = snap_to_ladder(price, "nearest")
        bet = ExchangeBet(
            bet_id=bet_id, market_id=market_id, selection=selection, side=side,
            requested_price=price, requested_size=stake, commission_pct=self.commission_pct,
        )
        self.bets.append(bet)

        if stake < self.min_bet:
            bet.status = OrderStatus.REJECTED
            bet.void_reason = f"stake {stake} below exchange minimum {self.min_bet}"
            logger.warning(f"[{self.venue_id}] rejected {side.value} {selection}: {bet.void_reason}")
            return bet

        if side == BetSide.LAY:
            liability = lay_liability(stake, price)
            if liability > self.available_balance:
                bet.status = OrderStatus.REJECTED
                bet.void_reason = (f"lay liability ${liability:.2f} exceeds available "
                                   f"${self.available_balance:.2f} (balance {self.balance:.2f} - "
                                   f"reserved {self.reserved_liability:.2f})")
                logger.warning(f"[{self.venue_id}] rejected lay {selection}: {bet.void_reason}")
                return bet
        elif stake > self.available_balance:
            bet.status = OrderStatus.REJECTED
            bet.void_reason = (f"back stake ${stake:.2f} exceeds available "
                               f"${self.available_balance:.2f}")
            logger.warning(f"[{self.venue_id}] rejected back {selection}: {bet.void_reason}")
            return bet

        if book is None:
            bet.status = OrderStatus.PENDING
            return bet

        filled, avg_price = book.walk(side, stake)
        if filled <= 0:
            bet.status = OrderStatus.PENDING
            return bet

        bet.matched_size = round(filled, 4)
        bet.average_matched_price = round(avg_price, 4)
        bet.matched_at = datetime.now(timezone.utc)
        bet.status = OrderStatus.MATCHED if filled >= stake - 1e-9 else OrderStatus.PARTIALLY_MATCHED
        if not persist_unmatched and bet.status == OrderStatus.PARTIALLY_MATCHED:
            bet.requested_size = bet.matched_size
            bet.status = OrderStatus.MATCHED
        self._apply_fill(bet, bet.matched_size, bet.average_matched_price)
        logger.info(f"[{self.venue_id}] {side.value} {selection} matched "
                    f"{bet.matched_size:.2f}@{bet.average_matched_price:.2f} "
                    f"(asked {stake:.2f}@{price:.2f})")
        return bet

    def settle(self, market_id: str, winner: Optional[str]) -> float:
        """
        Settle every position in a market. winner=None means the market was
        voided and all stakes/liabilities are returned. Returns net P&L.
        """
        pnl = 0.0
        for key, pos in list(self.positions.items()):
            if pos.market_id != market_id:
                continue
            if winner is None:
                for bet in self.bets:
                    if bet.market_id == market_id and bet.matched_size > 0:
                        bet.status = OrderStatus.VOIDED
                        bet.void_reason = "market voided"
                del self.positions[key]
                continue
            won = pos.selection == winner
            pnl += pos.pnl_if(won)
            del self.positions[key]
        self.balance += pnl
        logger.info(f"[{self.venue_id}] settled {market_id} winner={winner} pnl={pnl:+.2f} "
                    f"balance={self.balance:.2f}")
        return pnl

    def green_up(self, market_id: str, selection: str, current_lay_price: float,
                 book: Optional[ExchangeBook] = None,
                 bet_id: str = "", target_profit_pct: float = 0.0) -> Optional[ExchangeBet]:
        """
        Hedge a net back position by laying at the current price. Used to
        lock profit in-play when the selection's price has drifted in your
        favour, or to close out a position you no longer believe in.
        """
        pos = self.positions.get(f"{market_id}:{selection}")
        if pos is None or pos.net_back <= 0:
            return None
        lay_stake = green_up_stake(
            back_stake=pos.back_matched, back_odds=pos.back_avg_price,
            lay_odds=current_lay_price, commission_pct=self.commission_pct,
            target_profit_pct=target_profit_pct)
        if lay_stake is None:
            logger.info(f"[{self.venue_id}] no profitable green-up for {selection} "
                        f"at lay {current_lay_price}")
            return None
        return self.submit(bet_id or f"gu-{market_id}-{selection}", market_id, selection,
                           BetSide.LAY, current_lay_price, round(lay_stake, 2), book=book)

    def report(self) -> Dict:
        return {
            "venue": self.venue_id,
            "balance": round(self.balance, 2),
            "available": round(self.available_balance, 2),
            "reserved_liability": round(self.reserved_liability, 2),
            "open_positions": len(self.positions),
            "bets": len(self.bets),
            "worst_case_pnl": round(sum(p.worst_case() for p in self.positions.values()), 2),
            "hedged_positions": sum(1 for p in self.positions.values() if p.is_hedged()),
        }


# ---------------------------------------------------------------------------
# Market making
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    selection: str
    back_price: float
    back_size: float
    lay_price: float
    lay_size: float
    mid_price: float
    spread: float
    expected_profit_per_round_trip: float
    skipped: bool = False
    skip_reason: str = ""


class ExchangeMarketMaker:
    """
    Two-sided quoting around a model fair price.

    You only make money on an exchange if your fair price is better than
    the market's, and if you manage inventory - otherwise you accumulate a
    large one-sided position exactly when the market is about to move
    against you. Skewing the quote by inventory is the standard defence.

    Deliberately refuses to quote into thin books, wide spreads, or when
    the commission eats the whole spread.
    """

    def __init__(self, commission_pct: float = 0.05, min_edge_pct: float = 1.0,
                 max_inventory_exposure: float = 50.0, quote_size: float = 10.0,
                 inventory_skew_ticks: float = 1.0):
        self.commission_pct = commission_pct
        self.min_edge_pct = min_edge_pct
        self.max_inventory_exposure = max_inventory_exposure
        self.quote_size = quote_size
        self.inventory_skew_ticks = inventory_skew_ticks

    def quote(self, selection: str, fair_prob: float, book: ExchangeBook,
              inventory_exposure: float = 0.0) -> Quote:
        fair_price = 1.0 / fair_prob if fair_prob > 0 else 0.0
        best_back, best_lay = book.best_back, book.best_lay

        if fair_price <= 1.01:
            return Quote(selection=selection, back_price=0.0, back_size=0.0, lay_price=0.0,
                     lay_size=0.0, mid_price=0.0, spread=0.0,
                     expected_profit_per_round_trip=0.0, skipped=True,
                     skip_reason="fair probability invalid")
        if best_back is None or best_lay is None:
            return Quote(selection=selection, back_price=0.0, back_size=0.0, lay_price=0.0,
                     lay_size=0.0, mid_price=0.0, spread=0.0,
                     expected_profit_per_round_trip=0.0, skipped=True,
                     skip_reason="book is one-sided or empty, cannot quote both ways")

        mid = (best_back + best_lay) / 2.0
        spread = best_lay - best_back

        # Inventory skew: long inventory -> quote more aggressively on the
        # lay side (sell down) and less on the back side.
        skew = 0.0
        if self.max_inventory_exposure > 0:
            skew = self.inventory_skew_ticks * (inventory_exposure / self.max_inventory_exposure)

        back_price = next_ladder_step(mid, "down")
        lay_price = next_ladder_step(mid, "up")
        for _ in range(int(abs(skew))):
            if skew > 0:
                back_price = next_ladder_step(back_price, "down")
                lay_price = next_ladder_step(lay_price, "down")
            else:
                back_price = next_ladder_step(back_price, "up")
                lay_price = next_ladder_step(lay_price, "up")

        # Only quote where we have genuine edge on BOTH sides.
        back_edge_pct = (fair_prob - 1.0 / back_price) * 100.0 if back_price > 0 else 0.0
        lay_edge_pct = (1.0 / lay_price - fair_prob) * 100.0 if lay_price > 0 else 0.0
        if back_edge_pct < self.min_edge_pct and lay_edge_pct < self.min_edge_pct:
            return Quote(selection=selection, back_price=back_price, back_size=0.0,
                         lay_price=lay_price, lay_size=0.0, mid_price=round(mid, 4),
                         spread=round(spread, 4), expected_profit_per_round_trip=0.0,
                         skipped=True,
                         skip_reason=f"no edge either side (back {back_edge_pct:+.2f}%, "
                                     f"lay {lay_edge_pct:+.2f}%, need {self.min_edge_pct}%)")

        round_trip = spread - betfair_tick_size(mid)
        eff_back = commission_adjusted_decimal(back_price, self.commission_pct, "back")
        expected = self.quote_size * max(0.0, (fair_prob * eff_back - 1.0))

        back_size = self.quote_size if back_edge_pct >= self.min_edge_pct else 0.0
        lay_size = self.quote_size if lay_edge_pct >= self.min_edge_pct else 0.0

        # Don't add inventory beyond the cap.
        if inventory_exposure + back_size > self.max_inventory_exposure:
            back_size = 0.0
        if inventory_exposure - lay_liability(lay_size, lay_price) < -self.max_inventory_exposure:
            lay_size = 0.0

        if back_size == 0 and lay_size == 0:
            return Quote(selection=selection, back_price=back_price, back_size=0.0,
                         lay_price=lay_price, lay_size=0.0, mid_price=round(mid, 4),
                         spread=round(spread, 4), expected_profit_per_round_trip=0.0,
                         skipped=True, skip_reason="inventory cap reached on both sides")

        return Quote(selection=selection, back_price=back_price, back_size=round(back_size, 2),
                     lay_price=lay_price, lay_size=round(lay_size, 2), mid_price=round(mid, 4),
                     spread=round(spread, 4),
                     expected_profit_per_round_trip=round(expected, 4))


# ---------------------------------------------------------------------------
# Sizing helpers that respect exchange semantics
# ---------------------------------------------------------------------------

def size_back(bankroll: float, model_prob: float, price: float, kelly_frac: float = 0.25,
              cap: float = 0.06, commission_pct: float = 0.05, min_bet: float = 2.0) -> Dict:
    """
    Kelly size for a BACK, on commission-adjusted odds.

    Using the raw price overstates edge because commission on winnings
    reduces the effective payout.
    """
    eff = commission_adjusted_decimal(price, commission_pct, "back")
    edge_pct = (model_prob * eff - 1.0) * 100.0
    f = kelly_fraction(model_prob, eff, kelly_frac, cap)
    stake = bankroll * f
    if stake < min_bet:
        return {"stake": 0.0, "edge_pct": round(edge_pct, 3), "kelly": round(f, 6),
                "reason": f"size ${stake:.2f} below minimum ${min_bet}"}
    return {"stake": round(stake, 2), "edge_pct": round(edge_pct, 3),
            "kelly": round(f, 6), "effective_odds": round(eff, 4), "reason": ""}


def size_lay(bankroll: float, model_prob: float, price: float, kelly_frac: float = 0.25,
             cap: float = 0.06, commission_pct: float = 0.05, min_bet: float = 2.0) -> Dict:
    """
    Kelly size for a LAY, expressed as STAKE. Liability is reported too
    because liability is what the account actually reserves.

    A lay wins when the selection loses, so the winning probability is
    (1 - model_prob) and the payout is the stake itself.
    """
    p_win = 1.0 - model_prob
    # Equivalent decimal odds for "selection loses" priced at lay odds.
    eff_odds = commission_adjusted_decimal(price, commission_pct, "lay")
    eq_price = eff_odds / (eff_odds - 1.0) if eff_odds > 1.0 else 0.0
    if eq_price <= 1.0:
        return {"stake": 0.0, "edge_pct": 0.0, "kelly": 0.0, "reason": "lay price not bettable"}
    edge_pct = (p_win * eq_price - 1.0) * 100.0
    f = kelly_fraction(p_win, eq_price, kelly_frac, cap)
    liability = bankroll * f
    if liability <= 0:
        return {"stake": 0.0, "edge_pct": round(edge_pct, 3), "kelly": 0.0, "reason": "no edge"}
    stake = lay_stake_for_liability(liability, price)
    if stake < min_bet:
        return {"stake": 0.0, "edge_pct": round(edge_pct, 3), "kelly": round(f, 6),
                "reason": f"stake ${stake:.2f} below minimum ${min_bet}"}
    return {"stake": round(stake, 2), "liability": round(liability, 2),
            "edge_pct": round(edge_pct, 3), "kelly": round(f, 6), "reason": ""}
