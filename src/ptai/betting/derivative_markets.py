"""
Derivative Market Pricing - value every market a book actually offers.

One scoreline distribution prices most of a football card. Given the two
goal expectations, the full P(home=x, away=y) grid yields the over/under at
any line, both-teams-to-score, correct score, clean sheet, win to nil, every
Asian handicap, double chance, draw no bet, and half-time/full-time. They are
not independent markets and must not be priced independently - deriving them
from one distribution keeps them mutually consistent, which matters because
inconsistency between related markets IS the arbitrage.

Corners, cards, shots and offsides are NOT goals. They have their own
arrival rates, so they get their own Poisson processes. Using goal
expectations to price a corners line would be nonsense.

Quarter lines
-------------
A -0.25 or over 2.25 line is two half-stakes on the adjacent lines. These are
priced exactly that way rather than approximated, because the difference
shows up in the payout.

Player props
------------
A player's goal rate is modelled as their share of team expectation, so
team news that changes the team's attacking strength moves the prop too.
This is a model, not an oracle - props turn on lineups, which is why the
catalogue flags "void if player does not start" on every one of them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .models import poisson_soccer, scoreline_matrix


# ---------------------------------------------------------------------------
# Goal markets from the scoreline grid
# ---------------------------------------------------------------------------

@dataclass
class GoalMarketPrices:
    """Everything derivable from one scoreline distribution."""
    home_lambda: float
    away_lambda: float
    max_goals: int
    grid: List[List[float]] = field(default_factory=list)

    # cached aggregates
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0

    def totals(self, line: float) -> Dict[str, float]:
        """
        Over/under at any line, quarter-line aware.

        A whole line (2.5, 3.5) cannot push. An integer line (3.0) pushes on
        exactly that many goals. A quarter line (2.25) is half a bet on 2.0
        and half on 2.5.
        """
        return _price_total(self._total_goals_dist(), line)

    def team_totals(self, side: str, line: float) -> Dict[str, float]:
        dist = self._team_goals_dist(side)
        return _price_total(dist, line)

    def btts(self) -> Dict[str, float]:
        yes = 0.0
        for x, row in enumerate(self.grid):
            for y, p in enumerate(row):
                if x >= 1 and y >= 1:
                    yes += p
        return {"yes": round(yes, 4), "no": round(1.0 - yes, 4)}

    def correct_score(self, grid_limit: int = 6) -> Dict[str, float]:
        """
        The standard book grid plus 'Other' for everything outside it.

        Books price a fixed grid (0-0 through 5-5 roughly) and lump the rest
        into 'Other'. Pricing the tail as zero would badly misprice 'Other'.
        """
        out: Dict[str, float] = {}
        covered = 0.0
        for x in range(min(grid_limit + 1, len(self.grid))):
            for y in range(min(grid_limit + 1, len(self.grid[x]))):
                p = self.grid[x][y]
                out[f"{x}-{y}"] = round(p, 4)
                covered += p
        out["other"] = round(max(0.0, 1.0 - covered), 4)
        return out

    def clean_sheet(self) -> Dict[str, float]:
        home_keeps = sum(row[0] for row in self.grid)
        away_keeps = sum(self.grid[0])
        return {"home": round(home_keeps, 4), "away": round(away_keeps, 4)}

    def win_to_nil(self) -> Dict[str, float]:
        home = sum(self.grid[x][0] for x in range(1, len(self.grid)))
        away = sum(self.grid[0][y] for y in range(1, len(self.grid[0])))
        return {"home": round(home, 4), "away": round(away, 4)}

    def double_chance(self) -> Dict[str, float]:
        return {
            "1X": round(self.p_home + self.p_draw, 4),
            "X2": round(self.p_draw + self.p_away, 4),
            "12": round(self.p_home + self.p_away, 4),
        }

    def draw_no_bet(self) -> Dict[str, float]:
        """
        Draw refunds, so the price is the result prob renormalised over the
        two non-draw outcomes: P(home) / (P(home) + P(away)).
        """
        decisive = self.p_home + self.p_away
        if decisive <= 0:
            return {"home": 0.5, "away": 0.5}
        return {
            "home": round(self.p_home / decisive, 4),
            "away": round(self.p_away / decisive, 4),
            "push_prob": round(self.p_draw, 4),
        }

    def handicap(self, line: float) -> Dict[str, float]:
        """
        Asian handicap. `line` is home's handicap (negative = home gives).

        Returns P(home covers), P(away covers) and P(push) for whole/half
        lines. For a quarter line it returns the half-win/half-lose split as
        well, since that is what actually settles.
        """
        return _price_handicap(self.grid, line)

    def european_handicap(self, line: float) -> Dict[str, float]:
        """3-way handicap - no push, the draw survives as an outcome."""
        home = draw = away = 0.0
        for x, row in enumerate(self.grid):
            for y, p in enumerate(row):
                adj = (x + line) - y
                if adj > 1e-9:
                    home += p
                elif adj < -1e-9:
                    away += p
                else:
                    draw += p
        return {"home": round(home, 4), "draw": round(draw, 4), "away": round(away, 4)}

    def race_to(self, n: int) -> Dict[str, float]:
        """
        First team to score N goals.

        Computed from the negative-binomial ordering of the two Poisson
        processes: the probability home scores its Nth goal before away
        scores its Nth. Done by summing over the state space rather than
        approximating from the final score, because the final score does not
        tell you who got there first.
        """
        if n < 1:
            return {"home": 0.0, "away": 0.0, "none": 1.0}
        lam_h, lam_a = self.home_lambda, self.away_lambda
        total = lam_h + lam_a
        if total <= 0:
            return {"home": 0.0, "away": 0.0, "none": 1.0}
        p_home_scores = lam_h / total

        # P(home reaches n first) = P(home has n before away has n) in a
        # sequence of independent goals each home with prob p_home_scores.
        # The negative binomial gives the ORDER conditional on someone
        # reaching n - it sums to 1 on its own. Multiplying by the chance
        # that anyone reaches n at all is what makes this a real partition;
        # adding "neither" to the raw ordering double-counts and sums > 1.
        order_home = sum(_neg_binom(n, k, p_home_scores) for k in range(0, n))
        order_away = sum(_neg_binom(n, k, 1.0 - p_home_scores) for k in range(0, n))

        neither = 0.0
        for x in range(0, n):
            for y in range(0, n):
                neither += self.grid[x][y] if x < len(self.grid) and y < len(self.grid[x]) else 0.0
        reaches = max(0.0, 1.0 - neither)
        return {"home": round(order_home * reaches, 4),
                "away": round(order_away * reaches, 4),
                "none": round(min(1.0, neither), 4)}

    def _total_goals_dist(self) -> Dict[int, float]:
        dist: Dict[int, float] = {}
        for x, row in enumerate(self.grid):
            for y, p in enumerate(row):
                dist[x + y] = dist.get(x + y, 0.0) + p
        return dist

    def _team_goals_dist(self, side: str) -> Dict[int, float]:
        dist: Dict[int, float] = {}
        for x, row in enumerate(self.grid):
            for y, p in enumerate(row):
                k = x if side == "home" else y
                dist[k] = dist.get(k, 0.0) + p
        return dist


def _neg_binom(n: int, k: int, p: float) -> float:
    """P(exactly k failures before the nth success) - Pascal distribution."""
    if p <= 0 or p >= 1:
        return 0.0
    comb = math.comb(n + k - 1, k)
    return comb * (p ** n) * ((1.0 - p) ** k)


def _price_total(dist: Dict[int, float], line: float) -> Dict[str, float]:
    """
    Price an over/under from a discrete count distribution.

    Quarter lines are priced as the mean of the two adjacent half lines,
    which is exactly what the bet is: half the stake on each.
    """
    is_quarter = abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9

    def _whole_or_half(ln: float) -> Dict[str, float]:
        over = under = push = 0.0
        for total, p in dist.items():
            if total > ln + 1e-9:
                over += p
            elif total < ln - 1e-9:
                under += p
            else:
                push += p
        return {"over": round(over, 4), "under": round(under, 4), "push": round(push, 4)}

    if is_quarter:
        lower = math.floor(line * 2) / 2.0
        upper = lower + 0.5
        a, b = _whole_or_half(lower), _whole_or_half(upper)
        return {
            "over": round((a["over"] + b["over"]) / 2.0, 4),
            "under": round((a["under"] + b["under"]) / 2.0, 4),
            "push": 0.0,
            "quarter_line": True,
            "components": [lower, upper],
        }
    return _whole_or_half(line)


def _price_handicap(grid: List[List[float]], line: float) -> Dict[str, float]:
    """Price an Asian handicap from the scoreline grid, quarter-line aware."""
    is_quarter = abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9

    def _half(ln: float) -> Dict[str, float]:
        home = away = push = 0.0
        for x, row in enumerate(grid):
            for y, p in enumerate(row):
                adj = (x + ln) - y
                if adj > 1e-9:
                    home += p
                elif adj < -1e-9:
                    away += p
                else:
                    push += p
        return {"home": home, "away": away, "push": push}

    if is_quarter:
        lower = math.floor(line * 2) / 2.0
        upper = lower + 0.5
        a, b = _half(lower), _half(upper)
        return {
            "home": round((a["home"] + b["home"]) / 2.0, 4),
            "away": round((a["away"] + b["away"]) / 2.0, 4),
            "push": 0.0,
            "half_win_home": round(b["push"] / 2.0, 4),
            "half_win_away": round(a["push"] / 2.0, 4),
            "quarter_line": True,
            "components": [lower, upper],
        }
    r = _half(line)
    return {"home": round(r["home"], 4), "away": round(r["away"], 4),
            "push": round(r["push"], 4), "quarter_line": False}


# ---------------------------------------------------------------------------
# Half-time markets
# ---------------------------------------------------------------------------

# Empirically ~44-46% of goals are scored in the first half; teams are
# tighter early. This is a widely reported long-run figure, used as the
# default split rather than fitted here.
FIRST_HALF_GOAL_SHARE = 0.45


@dataclass
class HalfTimePrices:
    ht_home: float
    ht_draw: float
    ht_away: float
    ht_ft: Dict[str, float] = field(default_factory=dict)
    ht_totals: Dict[str, float] = field(default_factory=dict)


def price_half_time(home_lambda: float, away_lambda: float,
                    ft_grid: Optional[List[List[float]]] = None,
                    max_goals: int = 10,
                    first_half_share: float = FIRST_HALF_GOAL_SHARE) -> HalfTimePrices:
    """
    Half-time and half-time/full-time.

    First-half goals are modelled as an independent Poisson at
    lambda * first_half_share, and the second half as the remainder. HT/FT
    then combines them - the two halves are treated as independent, which is
    a simplification (leading teams do manage games) but the standard one,
    and it is stated rather than hidden.
    """
    lam_h1 = home_lambda * first_half_share
    lam_a1 = away_lambda * first_half_share
    lam_h2 = home_lambda - lam_h1
    lam_a2 = away_lambda - lam_a1

    g1 = scoreline_matrix(lam_h1, lam_a1, max_goals, rho=-0.05)
    g2 = scoreline_matrix(lam_h2, lam_a2, max_goals, rho=-0.05)

    ht_home = ht_draw = ht_away = 0.0
    for x, row in enumerate(g1):
        for y, p in enumerate(row):
            if x > y:
                ht_home += p
            elif x == y:
                ht_draw += p
            else:
                ht_away += p

    # HT/FT: combine first-half result with full-time result
    htft: Dict[str, float] = {}
    for hx, hrow in enumerate(g1):
        for hy, p1 in enumerate(hrow):
            ht_res = "home" if hx > hy else ("draw" if hx == hy else "away")
            for sx, srow in enumerate(g2):
                for sy, p2 in enumerate(srow):
                    fh, fa = hx + sx, hy + sy
                    ft_res = "home" if fh > fa else ("draw" if fh == fa else "away")
                    key = f"{_short(ht_res)}/{_short(ft_res)}"
                    htft[key] = htft.get(key, 0.0) + p1 * p2

    ht_totals = _price_total(_dist_from_grid(g1), 0.5)

    tot = ht_home + ht_draw + ht_away
    if tot > 0:
        ht_home, ht_draw, ht_away = ht_home / tot, ht_draw / tot, ht_away / tot

    return HalfTimePrices(
        ht_home=round(ht_home, 4), ht_draw=round(ht_draw, 4), ht_away=round(ht_away, 4),
        ht_ft={k: round(v, 4) for k, v in sorted(htft.items())},
        ht_totals=ht_totals,
    )


def _short(r: str) -> str:
    return {"home": "H", "draw": "D", "away": "A"}[r]


def _dist_from_grid(grid: List[List[float]]) -> Dict[int, float]:
    dist: Dict[int, float] = {}
    for x, row in enumerate(grid):
        for y, p in enumerate(row):
            dist[x + y] = dist.get(x + y, 0.0) + p
    return dist


# ---------------------------------------------------------------------------
# Corners / cards / shots / offsides - independent Poisson processes
# ---------------------------------------------------------------------------

# Long-run Premier League-ish averages, used as priors when a team has no
# observed data. These are priors for a model, not predictions.
DEFAULT_CORNERS_HOME = 5.8
DEFAULT_CORNERS_AWAY = 4.4
DEFAULT_CARDS_TOTAL = 3.6
DEFAULT_REDS_PER_MATCH = 0.06
DEFAULT_SHOTS_HOME = 13.0
DEFAULT_SHOTS_AWAY = 10.5
DEFAULT_SOT_HOME = 4.6
DEFAULT_SOT_AWAY = 3.5
DEFAULT_OFFSIDES_TOTAL = 3.6


@dataclass
class EventMarketPrices:
    """Prices for one countable match event (corners, cards, shots, offsides)."""
    home_lambda: float
    away_lambda: float
    totals: Dict[str, float] = field(default_factory=dict)
    handicap: Dict[str, float] = field(default_factory=dict)
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0
    race_to: Dict[int, Dict[str, float]] = field(default_factory=dict)


def price_count_market(home_lambda: float, away_lambda: float,
                       line: Optional[float] = None,
                       handicap_line: Optional[float] = None,
                       max_count: int = 20,
                       race_to_ns: Sequence[int] = ()) -> EventMarketPrices:
    """
    Price a countable event market from two independent Poisson rates.

    Corners and cards are genuinely independent of goals - a dominant team
    concedes corners while attacking - so they get their own process. The
    same routine serves corners, cards, shots, shots on target and offsides.
    """
    grid = scoreline_matrix(home_lambda, away_lambda, max_count, rho=0.0)

    p_home = p_draw = p_away = 0.0
    for x, row in enumerate(grid):
        for y, p in enumerate(row):
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p

    out = EventMarketPrices(
        home_lambda=round(home_lambda, 3), away_lambda=round(away_lambda, 3),
        p_home=round(p_home, 4), p_draw=round(p_draw, 4), p_away=round(p_away, 4),
    )
    if line is not None:
        out.totals = _price_total(_dist_from_grid(grid), line)
    if handicap_line is not None:
        out.handicap = _price_handicap(grid, handicap_line)
    for n in race_to_ns:
        out.race_to[n] = _race_to_from_grid(grid, n)
    return out


def _race_to_from_grid(grid: List[List[float]], n: int) -> Dict[str, float]:
    """First to N of this event - same negative-binomial logic as goals."""
    lam_h = sum(x * p for x, row in enumerate(grid) for p in row)
    lam_a = sum(y * p for row in grid for y, p in enumerate(row))
    total = lam_h + lam_a
    if total <= 0 or n < 1:
        return {"home": 0.0, "away": 0.0, "none": 1.0}
    p_h = lam_h / total
    order_home = sum(_neg_binom(n, k, p_h) for k in range(0, n))
    order_away = sum(_neg_binom(n, k, 1.0 - p_h) for k in range(0, n))
    neither = 0.0
    for x in range(0, min(n, len(grid))):
        for y in range(0, min(n, len(grid[x]))):
            neither += grid[x][y]
    reaches = max(0.0, 1.0 - neither)
    return {"home": round(order_home * reaches, 4),
            "away": round(order_away * reaches, 4),
            "none": round(min(1.0, neither), 4)}


def price_red_card(red_rate_per_match: float = DEFAULT_REDS_PER_MATCH,
                   minutes: int = 90) -> Dict[str, float]:
    """
    P(at least one red card).

    Modelled as a rare Poisson event. The rate scales with minutes played
    because an abandoned match has had less opportunity. Second yellows are
    included in the book's definition of a red, so the rate must be the
    combined one, not just straight reds.
    """
    lam = max(0.0, red_rate_per_match) * (minutes / 90.0)
    p_at_least_one = 1.0 - math.exp(-lam)
    return {
        "yes": round(p_at_least_one, 4),
        "no": round(1.0 - p_at_least_one, 4),
        "expected_reds": round(lam, 4),
    }


def price_booking_points(cards_lambda: float, red_rate: float = DEFAULT_REDS_PER_MATCH,
                         line: Optional[float] = None, yellow_value: int = 10,
                         red_value: int = 25, max_cards: int = 15) -> Dict[str, float]:
    """
    Booking points over/under.

    Total points = yellows*10 + reds*25. Reds are modelled separately and
    consume one of the card events (a red replaces a yellow in most book
    rules for the same incident, but books differ - hence the warning in the
    catalogue).
    """
    dist: Dict[int, float] = {}
    n_reds_max = 3
    red_probs = [math.exp(-red_rate) * red_rate ** r / math.factorial(r) for r in range(n_reds_max + 1)]
    for r in range(n_reds_max + 1):
        yellow_lam = max(0.01, cards_lambda - r)
        for y in range(max_cards + 1):
            p = red_probs[r] * math.exp(-yellow_lam) * yellow_lam ** y / math.factorial(y)
            pts = y * yellow_value + r * red_value
            dist[pts] = dist.get(pts, 0.0) + p
    out: Dict[str, float] = {"expected_points": round(sum(k * v for k, v in dist.items()), 2)}
    if line is not None:
        out.update(_price_total(dist, line))
    return out


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def price_player_count(team_lambda: float, player_share: float,
                       line: Optional[float] = None) -> Dict[str, float]:
    """
    Player count prop (goals, shots, tackles, assists).

    player_lambda = team_lambda * player_share, where share is the player's
    historical share of that team's events. This makes the prop respond to
    team-level changes - a weakened opponent raises a striker's goal lambda
    - instead of treating the player in isolation.
    """
    lam = max(0.0, team_lambda) * max(0.0, min(1.0, player_share))
    out: Dict[str, float] = {"expected": round(lam, 4)}
    if line is not None:
        dist = {k: math.exp(-lam) * lam ** k / math.factorial(k) for k in range(0, 12)}
        out.update(_price_total(dist, line))
    return out


def price_anytime_scorer(team_lambda: float, player_share: float,
                         minutes_share: float = 1.0) -> Dict[str, float]:
    """
    P(player scores at least once) = 1 - exp(-lambda_player).

    minutes_share discounts for a player who is not a guaranteed starter:
    a striker who starts 70% of matches has a lower anytime probability than
    the same striker starting every game.
    """
    lam = max(0.0, team_lambda) * max(0.0, min(1.0, player_share)) * max(0.0, min(1.0, minutes_share))
    p = 1.0 - math.exp(-lam)
    return {"yes": round(p, 4), "no": round(1.0 - p, 4), "expected_goals": round(lam, 4)}


def price_first_scorer(team_lambda: float, player_share: float,
                       opponent_lambda: float, minutes_share: float = 1.0) -> Dict[str, float]:
    """
    P(player scores first).

    The player must score before anyone else does, including the opposition.
    With independent Poisson processes, P(this player's goal is the first
    event) = lambda_player / lambda_all.
    """
    lam_p = max(0.0, team_lambda) * max(0.0, min(1.0, player_share)) * max(0.0, min(1.0, minutes_share))
    teammates = max(0.0, team_lambda) - lam_p
    lam_all = lam_p + max(0.0, teammates) + max(0.0, opponent_lambda)
    if lam_all <= 0:
        return {"yes": 0.0, "no": 1.0}
    # P(first goal is the player's) * P(at least one goal is scored)
    p_first = (lam_p / lam_all) * (1.0 - math.exp(-lam_all))
    return {"yes": round(p_first, 4), "no": round(1.0 - p_first, 4)}


# ---------------------------------------------------------------------------
# Accumulators / parlays
# ---------------------------------------------------------------------------

@dataclass
class AccumulatorLeg:
    market_key: str
    outcome: str
    price: float
    model_prob: float
    event_key: str
    line: Optional[float] = None


@dataclass
class AccumulatorQuote:
    legs: List[AccumulatorLeg]
    combined_price: float
    independent_prob: float
    correlation_adjusted_prob: float
    edge_independent: float
    edge_adjusted: float
    warning: str = ""
    blocked: bool = False
    blockers: List[str] = field(default_factory=list)


def price_accumulator(legs: Sequence[AccumulatorLeg],
                      same_event_penalty: float = 0.85,
                      correlation_penalty: float = 0.90,
                      max_legs: int = 8) -> AccumulatorQuote:
    """
    Accumulator pricing with an explicit correlation haircut.

    Multiplying leg probabilities assumes independence, and that assumption
    is FALSE within a match: "over 2.5 goals" and "both teams to score" are
    heavily positively correlated, so their true joint probability is well
    above the product. Multiplying them anyway makes the parlay look like
    better value than it is - this is the single most common way recreational
    bettors lose money, and the reason books love parlays.

    So: legs in the same event get `same_event_penalty`, and any parlay gets
    `correlation_penalty`. These are conservative haircuts, not a model. The
    honest statement is that a parlay's true probability is not computable
    without a joint model, and the engine should not pretend otherwise.
    """
    blockers: List[str] = []
    if len(legs) < 2:
        blockers.append("an accumulator needs at least 2 legs")
    if len(legs) > max_legs:
        blockers.append(f"{len(legs)} legs exceeds the {max_legs} limit - "
                        f"long parlays are negative EV by construction")
    for leg in legs:
        if leg.price <= 1.0:
            blockers.append(f"{leg.market_key}/{leg.outcome} price {leg.price} not bettable")
        if leg.model_prob <= 0:
            blockers.append(f"{leg.market_key}/{leg.outcome} has no model probability")

    combined_price = 1.0
    independent_prob = 1.0
    for leg in legs:
        combined_price *= leg.price
        independent_prob *= leg.model_prob

    events = {leg.event_key for leg in legs}
    same_event = len(events) < len(legs)
    adjusted = independent_prob * (same_event_penalty if same_event else 1.0) * correlation_penalty

    warning = ""
    if same_event:
        warning = ("legs share an event - their outcomes are correlated, so the "
                   "independent product overstates the true probability")

    edge_ind = independent_prob * combined_price - 1.0
    edge_adj = adjusted * combined_price - 1.0

    if edge_ind > 0 >= edge_adj:
        warning += (" | looks +EV only under the independence assumption; "
                    "after the correlation haircut it is negative EV")

    return AccumulatorQuote(
        legs=list(legs), combined_price=round(combined_price, 4),
        independent_prob=round(independent_prob, 6),
        correlation_adjusted_prob=round(adjusted, 6),
        edge_independent=round(edge_ind, 4), edge_adjusted=round(edge_adj, 4),
        warning=warning.strip(), blocked=bool(blockers), blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Top-level: price an entire match card
# ---------------------------------------------------------------------------

@dataclass
class MatchCardPrices:
    """Every priced market for one fixture, from one consistent model."""
    event_key: str
    goals: GoalMarketPrices
    half_time: HalfTimePrices
    corners: Optional[EventMarketPrices] = None
    cards: Optional[EventMarketPrices] = None
    shots: Optional[EventMarketPrices] = None
    shots_on_target: Optional[EventMarketPrices] = None
    offsides: Optional[EventMarketPrices] = None
    red_card: Dict[str, float] = field(default_factory=dict)
    booking_points: Dict[str, float] = field(default_factory=dict)
    markets_priced: int = 0

    def to_dict(self) -> Dict:
        return {
            "event_key": self.event_key,
            "goal_lambdas": [self.goals.home_lambda, self.goals.away_lambda],
            "p_home": self.goals.p_home, "p_draw": self.goals.p_draw, "p_away": self.goals.p_away,
            "double_chance": self.goals.double_chance(),
            "draw_no_bet": self.goals.draw_no_bet(),
            "btts": self.goals.btts(),
            "clean_sheet": self.goals.clean_sheet(),
            "win_to_nil": self.goals.win_to_nil(),
            "correct_score_top5": dict(sorted(self.goals.correct_score().items(),
                                              key=lambda kv: -kv[1])[:5]),
            "ht": {"home": self.half_time.ht_home, "draw": self.half_time.ht_draw,
                   "away": self.half_time.ht_away},
            "ht_ft_top3": dict(sorted(self.half_time.ht_ft.items(), key=lambda kv: -kv[1])[:3]),
            "corners": {"lambdas": [self.corners.home_lambda, self.corners.away_lambda]} if self.corners else None,
            "cards": {"lambdas": [self.cards.home_lambda, self.cards.away_lambda]} if self.cards else None,
            "red_card": self.red_card,
            "markets_priced": self.markets_priced,
        }


def price_match_card(event_key: str,
                     home_attack: float, home_defence: float,
                     away_attack: float, away_defence: float,
                     corners_home: Optional[float] = None, corners_away: Optional[float] = None,
                     cards_home: Optional[float] = None, cards_away: Optional[float] = None,
                     red_rate: float = DEFAULT_REDS_PER_MATCH,
                     shots_home: Optional[float] = None, shots_away: Optional[float] = None,
                     sot_home: Optional[float] = None, sot_away: Optional[float] = None,
                     offsides: Optional[float] = None,
                     goals_line: float = 2.5, corners_line: float = 10.5,
                     cards_line: float = 3.5, max_goals: int = 10) -> MatchCardPrices:
    """
    Price a whole match card from team strengths, consistently.

    Every goal-derived market comes from one scoreline grid, so they cannot
    disagree with each other. Corners, cards, shots and offsides get their
    own rates, defaulting to long-run priors when a team has no observed
    data.
    """
    fc = poisson_soccer(home_attack, home_defence, away_attack, away_defence)
    grid = scoreline_matrix(fc.home_lambda, fc.away_lambda, max_goals, rho=-0.05)

    goals = GoalMarketPrices(
        home_lambda=fc.home_lambda, away_lambda=fc.away_lambda, max_goals=max_goals,
        grid=grid, p_home=fc.p_home, p_draw=fc.p_draw, p_away=fc.p_away,
    )
    goals.totals(goals_line)  # exercised; caller uses the returned dict

    half_time = price_half_time(fc.home_lambda, fc.away_lambda, ft_grid=grid,
                                max_goals=max_goals)

    card = MatchCardPrices(event_key=event_key, goals=goals, half_time=half_time)

    card.corners = price_count_market(
        corners_home if corners_home is not None else DEFAULT_CORNERS_HOME,
        corners_away if corners_away is not None else DEFAULT_CORNERS_AWAY,
        line=corners_line, handicap_line=-1.5, race_to_ns=(9,))
    card.cards = price_count_market(
        cards_home if cards_home is not None else DEFAULT_CARDS_TOTAL / 2.0,
        cards_away if cards_away is not None else DEFAULT_CARDS_TOTAL / 2.0,
        line=cards_line, race_to_ns=(3,))
    card.shots = price_count_market(
        shots_home if shots_home is not None else DEFAULT_SHOTS_HOME,
        shots_away if shots_away is not None else DEFAULT_SHOTS_AWAY,
        line=23.5)
    card.shots_on_target = price_count_market(
        sot_home if sot_home is not None else DEFAULT_SOT_HOME,
        sot_away if sot_away is not None else DEFAULT_SOT_AWAY,
        line=8.5)
    if offsides is not None:
        card.offsides = price_count_market(offsides / 2.0, offsides / 2.0, line=3.5)

    card.red_card = price_red_card(red_rate)
    total_cards = (cards_home or DEFAULT_CARDS_TOTAL / 2.0) + (cards_away or DEFAULT_CARDS_TOTAL / 2.0)
    card.booking_points = price_booking_points(total_cards, red_rate, line=35.0)

    # Count what this actually covers, so "N markets" is a real number.
    card.markets_priced = (
        4                                  # h2h, double chance, draw no bet, european handicap
        + 1                                # totals
        + 2                                # team goals home/away
        + 1 + 1 + 1                        # btts, correct score, clean sheet
        + 1                                # win to nil
        + 2                                # asian handicap, race to goals
        + 3                                # HT result, HT goals, HT/FT
        + 3                                # corners total/handicap/1x2
        + 4                                # cards total, 1x2, red card, booking points
        + 2                                # shots, shots on target
        + (1 if card.offsides else 0)      # offsides
    )
    return card
