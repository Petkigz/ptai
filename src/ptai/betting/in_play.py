"""
In-play pricing - fair value as a function of match state.

Why this module exists
----------------------
Every price in this project was a pre-match price. Nothing decayed. That is a
real money bug, not a coverage gap: a 0-0 draw in the 80th minute is worth far
more than a 0-0 draw at kickoff, because there is far less time for either side
to score. A model that returns the kickoff number at minute 80 will call the
draw a massive edge and bet the wrong side of it.

Live betting is also where most exchange volume sits, so refusing to price it
means the wide market catalogue is only usable for the small pre-match window.

How the pricing works
---------------------
Expected remaining scoring is not simply `rate * minutes_left / 90`. Two
adjustments matter and both are empirically supported:

1. **Time profile.** Goals are not uniformly distributed across a soccer match.
   The final quarter is markedly heavier than the first - fatigue, chasing
   behaviour, substitutions and stoppage time. The multiplier below is applied
   per minute and normalised so that integrating across a full match still
   returns the calibrated full-match rate. Otherwise every live price would be
   biased by whatever the multipliers happen to average to.

2. **Game state.** A trailing side commits more bodies forward, which raises
   its own attack rate AND exposes it to the counter. A leading side sits
   deeper. These are not symmetric and the opponent's rate moves too, so both
   lambdas are adjusted, not just one.

Red cards multiply both effects: ten men score less and concede more.

For high-scoring sports the same structure applies with a normal distribution
on the margin, where the expected margin scales with the fraction of the game
remaining but the standard deviation scales with the SQUARE ROOT of that
fraction, because variance adds over independent possessions.

What this module deliberately does not do
-----------------------------------------
It does not model shot quality, possession, xG accumulation or momentum from
live event streams - this project has no such feed. It prices from score,
clock and cards, which is what a scoreboard actually gives you. Claiming more
would be the same kind of invention the rest of this subsystem was built to
remove.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .high_scoring import SPORT_PARAMS, SportParams, resolve_sport
from .models import scoreline_matrix

# ---------------------------------------------------------------------------
# Soccer time profile
# ---------------------------------------------------------------------------

# Per-minute goal-rate multipliers by minute band. Not normalised here -
# _normalise_profile does that so full-match lambda stays calibrated.
_TIME_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 15.0, 0.85),
    (15.0, 30.0, 0.95),
    (30.0, 45.0, 1.00),
    (45.0, 60.0, 1.08),
    (60.0, 75.0, 1.12),
    (75.0, 90.0, 1.35),
)

_RAW_MEAN = sum(m * (b - a) for a, b, m in _TIME_BANDS) / 90.0
TIME_PROFILE: Tuple[Tuple[float, float, float], ...] = tuple(
    (a, b, m / _RAW_MEAN) for a, b, m in _TIME_BANDS
)

# Game-state multipliers on a side's OWN attack rate
_ATTACK_WHEN_TRAILING = 1.18    # chasing the game, committing forward
_ATTACK_WHEN_LEVEL = 1.00
_ATTACK_WHEN_LEADING = 0.92     # protecting the result, sitting deeper

# The opponent's attack rate also moves, because pushing up concedes counters
_CONCEDE_WHEN_PUSHING = 1.10
_CONCEDE_WHEN_PROTECTING = 0.92

# A sending-off compounds both effects
_RED_ATTACK_MULTIPLIER = 0.72
_RED_CONCEDE_MULTIPLIER = 1.22


@dataclass
class MatchState:
    """
    Live match state. Everything here is observable from a scoreboard feed;
    nothing requires a proprietary data source.
    """
    minutes_elapsed: float = 0.0
    home_score: int = 0
    away_score: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_cards: int = 0
    away_cards: int = 0
    in_extra_time: bool = False
    # Regulation length in minutes. 90 for soccer, 48 for NBA, 60 for NFL/NHL,
    # 9 innings-equivalent ~ 162 minutes for MLB.
    regulation_minutes: float = 90.0
    is_high_scoring: bool = False

    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def minutes_remaining(self) -> float:
        if self.in_extra_time:
            return 0.0
        return max(0.0, self.regulation_minutes - self.minutes_elapsed)

    @property
    def fraction_remaining(self) -> float:
        if self.regulation_minutes <= 0:
            return 0.0
        return max(0.0, min(1.0, self.minutes_remaining / self.regulation_minutes))

    @property
    def is_finished(self) -> bool:
        return self.minutes_remaining <= 0.0 and not self.in_extra_time

    @property
    def leading_side(self) -> str:
        if self.margin > 0:
            return "home"
        if self.margin < 0:
            return "away"
        return "level"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.minutes_elapsed < 0:
            problems.append("minutes_elapsed is negative")
        if self.minutes_elapsed > self.regulation_minutes and not self.in_extra_time:
            problems.append(
                f"minutes_elapsed {self.minutes_elapsed} exceeds regulation "
                f"{self.regulation_minutes} without in_extra_time set")
        if self.home_score < 0 or self.away_score < 0:
            problems.append("scores cannot be negative")
        if self.home_red_cards < 0 or self.away_red_cards < 0:
            problems.append("red card counts cannot be negative")
        if self.home_red_cards > 4 or self.away_red_cards > 4:
            problems.append("more than four sendings-off is not a real match state")
        return problems


# ---------------------------------------------------------------------------
# Soccer live rates
# ---------------------------------------------------------------------------

def _profile_weight(fraction_remaining: float) -> float:
    """
    Share of the full-match scoring rate still to come.

    Integrates the normalised time profile over the remaining minutes. At
    kickoff this returns 1.0 (the full rate is still ahead); with no time left
    it returns 0.0. Because the profile is front-light and back-heavy, the
    weight falls SLOWER than the clock at first and then faster - so a naive
    linear `minutes_left / 90` understates late-game scoring.
    """
    frac = max(0.0, min(1.0, fraction_remaining))
    if frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return 1.0
    start_min = 90.0 * (1.0 - frac)
    weight = 0.0
    for a, b, m in TIME_PROFILE:
        lo = max(a, start_min)
        hi = min(b, 90.0)
        if hi > lo:
            weight += m * (hi - lo)
    return weight / 90.0


def _state_multipliers(state: MatchState) -> Tuple[float, float]:
    """
    Attack-rate multipliers for (home, away) given score and sendings-off.

    Both sides move: chasing the game raises your own attack and also raises
    what you concede, so the adjustments are applied asymmetrically below.
    """
    side = state.leading_side
    if side == "home":
        home_attack, away_attack = _ATTACK_WHEN_LEADING, _ATTACK_WHEN_TRAILING
    elif side == "away":
        home_attack, away_attack = _ATTACK_WHEN_TRAILING, _ATTACK_WHEN_LEADING
    else:
        home_attack = away_attack = _ATTACK_WHEN_LEVEL

    # sendings-off: fewer men score less and concede more
    home_attack *= _RED_ATTACK_MULTIPLIER ** state.home_red_cards
    away_attack *= _RED_ATTACK_MULTIPLIER ** state.away_red_cards
    return home_attack, away_attack


def live_rates(home_rate: float, away_rate: float, state: MatchState) -> Tuple[float, float]:
    """
    Expected goals for each side over the REMAINING time.

    Combines the time profile with game state. Returns (0, 0) once the match
    is finished, because no further scoring is possible.
    """
    weight = _profile_weight(state.fraction_remaining)
    home_mult, away_mult = _state_multipliers(state)

    # The side pushing forward concedes more on the counter, so the opponent's
    # rate is scaled by the counter-effect of the other side's posture.
    home_pushing = home_mult > 1.0
    away_pushing = away_mult > 1.0
    home_concede_mult = _CONCEDE_WHEN_PUSHING if home_pushing else (
        _CONCEDE_WHEN_PROTECTING if home_mult < 1.0 else 1.0)
    away_concede_mult = _CONCEDE_WHEN_PUSHING if away_pushing else (
        _CONCEDE_WHEN_PROTECTING if away_mult < 1.0 else 1.0)

    h = home_rate * home_mult * away_concede_mult
    a = away_rate * away_mult * home_concede_mult

    # sendings-off raise what the short-handed side concedes
    h *= _RED_CONCEDE_MULTIPLIER ** state.away_red_cards
    a *= _RED_CONCEDE_MULTIPLIER ** state.home_red_cards

    return round(h * weight, 4), round(a * weight, 4)


def live_scoreline_matrix(home_rate: float, away_rate: float,
                          state: MatchState) -> Dict[Tuple[int, int], float]:
    """
    Full-match scoreline probabilities given the current score.

    Only the REMAINING goals are modelled, then shifted onto the current score,
    because goals already scored are certain and must not be re-randomised.
    """
    h_rem, a_rem = live_rates(home_rate, away_rate, state)
    if state.is_finished:
        return {(state.home_score, state.away_score): 1.0}
    # scoreline_matrix returns a [home][away] grid, not a dict
    grid = scoreline_matrix(h_rem, a_rem)
    return {(state.home_score + hg, state.away_score + ag): p
            for hg, row in enumerate(grid)
            for ag, p in enumerate(row) if p > 0}


# ---------------------------------------------------------------------------
# Soccer live markets
# ---------------------------------------------------------------------------

def live_match_odds(home_rate: float, away_rate: float,
                    state: MatchState) -> Dict[str, float]:
    """1X2 from the current state onward, including goals already scored."""
    grid = live_scoreline_matrix(home_rate, away_rate, state)
    return {
        "home": round(sum(p for (h, a), p in grid.items() if h > a), 4),
        "draw": round(sum(p for (h, a), p in grid.items() if h == a), 4),
        "away": round(sum(p for (h, a), p in grid.items() if h < a), 4),
    }


def live_double_chance(home_rate: float, away_rate: float,
                       state: MatchState) -> Dict[str, float]:
    o = live_match_odds(home_rate, away_rate, state)
    return {"home_or_draw": round(o["home"] + o["draw"], 4),
            "home_or_away": round(o["home"] + o["away"], 4),
            "draw_or_away": round(o["draw"] + o["away"], 4)}


def live_totals(home_rate: float, away_rate: float, state: MatchState,
                line: float) -> Dict[str, float]:
    """
    Over/under on the FULL match total from the current score.

    Already-scored goals are certain, so the only randomness left is the
    remaining total. The line is shifted down by the goals already on the
    board: at 2-0 with 10 minutes left, over 2.5 needs one more goal, not
    three.
    """
    remaining_line = line - state.total_goals
    if state.is_finished:
        return {"over": round(1.0 if state.total_goals > line else 0.0, 4),
                "under": round(1.0 if state.total_goals < line else 0.0, 4),
                "push": round(1.0 if state.total_goals == line else 0.0, 4)}

    h_rem, a_rem = live_rates(home_rate, away_rate, state)
    lam = h_rem + a_rem
    max_g = min(12, int(lam + 6 * math.sqrt(max(lam, 1.0)) + 3))

    over = under = push = 0.0
    for g in range(0, max_g + 1):
        p = math.exp(-lam) * lam ** g / math.factorial(g)
        total = state.total_goals + g
        if total > line:
            over += p
        elif total < line:
            under += p
        else:
            push += p
    return {"over": round(over, 4), "under": round(under, 4), "push": round(push, 4)}


def live_btts(home_rate: float, away_rate: float, state: MatchState) -> Dict[str, float]:
    """
    Both teams to score, from the current state.

    If both have already scored it is certain. If one has not and time has
    run out it is impossible. Otherwise it is the chance the missing side
    scores at least once in what is left.
    """
    if state.home_score > 0 and state.away_score > 0:
        return {"yes": 1.0, "no": 0.0}
    if state.is_finished:
        return {"yes": 0.0, "no": 1.0}

    h_rem, a_rem = live_rates(home_rate, away_rate, state)
    home_scores = 1.0 - math.exp(-h_rem)
    away_scores = 1.0 - math.exp(-a_rem)

    if state.home_score == 0 and state.away_score == 0:
        yes = home_scores * away_scores
    elif state.home_score == 0:
        yes = home_scores           # away already scored
    else:
        yes = away_scores           # home already scored
    return {"yes": round(yes, 4), "no": round(1.0 - yes, 4)}


def live_next_goal(home_rate: float, away_rate: float,
                   state: MatchState) -> Dict[str, float]:
    """
    Which side scores next.

    This is a race between two Poisson processes, not a normalisation of the
    two rates. The rates give the chance that home wins the race, but there is
    also a real chance nobody scores again, and that must be its own outcome
    or the three will not sum to one.
    """
    if state.is_finished:
        return {"home": 0.0, "away": 0.0, "none": 1.0}

    h_rem, a_rem = live_rates(home_rate, away_rate, state)
    total = h_rem + a_rem
    if total <= 0:
        return {"home": 0.0, "away": 0.0, "none": 1.0}

    any_goal = 1.0 - math.exp(-total)
    home_next = (h_rem / total) * any_goal
    away_next = (a_rem / total) * any_goal
    return {"home": round(home_next, 4), "away": round(away_next, 4),
            "none": round(1.0 - any_goal, 4)}


def live_correct_score(home_rate: float, away_rate: float, state: MatchState,
                       top_n: int = 10) -> Dict[str, float]:
    """Most likely final scorelines given what is already on the board."""
    grid = live_scoreline_matrix(home_rate, away_rate, state)
    top = sorted(grid.items(), key=lambda kv: -kv[1])[:top_n]
    return {f"{h}-{a}": round(p, 4) for (h, a), p in top}


def live_team_total(home_rate: float, away_rate: float, state: MatchState,
                    team: str, line: float) -> Dict[str, float]:
    """One side's full-match goal total against a line."""
    remaining_line = line - (state.home_score if team == "home" else state.away_score)
    base = home_rate if team == "home" else away_rate
    rem, _ = live_rates(home_rate, away_rate, state) if team == "home" else (
        None, live_rates(home_rate, away_rate, state)[1])
    lam = rem if team == "home" else _

    if state.is_finished:
        current = state.home_score if team == "home" else state.away_score
        return {"over": round(1.0 if current > line else 0.0, 4),
                "under": round(1.0 if current < line else 0.0, 4),
                "push": round(1.0 if current == line else 0.0, 4)}

    max_g = min(10, int(lam + 6 * math.sqrt(max(lam, 1.0)) + 3))
    over = under = push = 0.0
    for g in range(0, max_g + 1):
        p = math.exp(-lam) * lam ** g / math.factorial(g)
        if g > remaining_line:
            over += p
        elif g < remaining_line:
            under += p
        else:
            push += p
    return {"over": round(over, 4), "under": round(under, 4), "push": round(push, 4)}


def live_draw_no_bet(home_rate: float, away_rate: float,
                     state: MatchState) -> Dict[str, float]:
    o = live_match_odds(home_rate, away_rate, state)
    return {"home": round(o["home"], 4), "away": round(o["away"], 4),
            "push": round(o["draw"], 4)}


# ---------------------------------------------------------------------------
# High-scoring live pricing
# ---------------------------------------------------------------------------

# Late-game rate inflation for high-scoring sports: fouling and free throws in
# basketball, hurry-up offence and prevent defence in gridiron, pulling the
# goalie in hockey. Milder than soccer because scoring is far more uniform.
_HS_LATE_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.50, 0.97),
    (0.50, 0.80, 1.00),
    (0.80, 0.95, 1.05),
    (0.95, 1.00, 1.12),
)
_HS_MEAN = sum(m * (b - a) for a, b, m in _HS_LATE_BANDS)
_HS_PROFILE = tuple((a, b, m / _HS_MEAN) for a, b, m in _HS_LATE_BANDS)


def _hs_profile_weight(fraction_remaining: float) -> float:
    frac = max(0.0, min(1.0, fraction_remaining))
    if frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return 1.0
    start = 1.0 - frac
    weight = 0.0
    for a, b, m in _HS_PROFILE:
        lo = max(a, start)
        hi = min(b, 1.0)
        if hi > lo:
            weight += m * (hi - lo)
    return weight


@dataclass
class LiveTotalPrices:
    line: float
    over: float
    under: float
    push: float
    expected_total: float
    remaining_std: float
    model: str = "live_normal"


@dataclass
class LiveSpreadPrices:
    line: float
    home: float
    away: float
    push: float
    expected_margin: float
    remaining_std: float
    model: str = "live_normal"


def live_total_high_scoring(expected_home: float, expected_away: float,
                            state: MatchState, line: float,
                            params: Optional[SportParams] = None) -> LiveTotalPrices:
    """
    Full-match total for a high-scoring sport, from the current score.

    The expected remaining total scales with the time-profile weight, but the
    standard deviation scales with the SQUARE ROOT of the fraction remaining,
    because variance adds over independent possessions. Scaling the deviation
    linearly would understate late-game uncertainty badly.

    The push band is delegated to the pre-match pricer rather than
    reimplemented here, so a whole-number line gets the same integer-interval
    push band in-play as it does pre-match.
    """
    params = params or SPORT_PARAMS.get("basketball")
    scored = state.total_goals
    weight = _hs_profile_weight(state.fraction_remaining)

    if state.is_finished or weight <= 0:
        return LiveTotalPrices(line=line,
                               over=round(1.0 if scored > line else 0.0, 4),
                               under=round(1.0 if scored < line else 0.0, 4),
                               push=round(1.0 if scored == line else 0.0, 4),
                               expected_total=float(scored), remaining_std=0.0)

    # expected remaining points, keeping each side's share of the pre-game total
    pre_total = max(expected_home + expected_away, 1e-9)
    rem_points = params.typical_total * weight
    rem_home = rem_points * (expected_home / pre_total)
    rem_away = rem_points * (expected_away / pre_total)
    exp_total = scored + rem_home + rem_away
    rem_std = params.total_std * math.sqrt(weight)

    from .high_scoring import price_total_normal
    tp = price_total_normal(exp_total, rem_std, line)
    return LiveTotalPrices(line=line, over=tp.over, under=tp.under, push=tp.push,
                           expected_total=round(exp_total, 2),
                           remaining_std=round(rem_std, 3))


def live_spread_high_scoring(expected_home: float, expected_away: float,
                             state: MatchState, line: float,
                             params: Optional[SportParams] = None) -> LiveSpreadPrices:
    """
    Full-match margin against the spread, from the current score.

    The current margin is certain; only the remaining margin is random, with a
    smaller standard deviation. The two are added and the result handed to the
    pre-match spread pricer so whole/quarter lines decompose identically
    in-play and pre-match.
    """
    params = params or SPORT_PARAMS.get("basketball")
    weight = _hs_profile_weight(state.fraction_remaining)

    if state.is_finished or weight <= 0:
        m = state.margin
        return LiveSpreadPrices(line=line,
                                home=round(1.0 if m + line > 0 else 0.0, 4),
                                away=round(1.0 if m + line < 0 else 0.0, 4),
                                push=round(1.0 if m + line == 0 else 0.0, 4),
                                expected_margin=float(m), remaining_std=0.0)

    pre_total = max(expected_home + expected_away, 1e-9)
    share_home = expected_home / pre_total
    rem_points = params.typical_total * weight
    rem_home = rem_points * share_home
    rem_away = rem_points * (1.0 - share_home)
    rem_std = params.margin_std * math.sqrt(weight)

    exp_margin = state.margin + params.home_advantage_points * weight + (rem_home - rem_away)

    from .high_scoring import price_spread_normal
    sp = price_spread_normal(exp_margin, rem_std, line)
    return LiveSpreadPrices(line=line, home=sp.home, away=sp.away, push=sp.push,
                            expected_margin=round(exp_margin, 2),
                            remaining_std=round(rem_std, 3))


def live_moneyline_high_scoring(expected_home: float, expected_away: float,
                                state: MatchState,
                                params: Optional[SportParams] = None) -> Dict[str, float]:
    sp = live_spread_high_scoring(expected_home, expected_away, state, 0.0, params)
    return {"home": sp.home, "away": sp.away, "push": sp.push}


# ---------------------------------------------------------------------------
# Live card - every live market in one call
# ---------------------------------------------------------------------------

@dataclass
class LiveCard:
    """Every live market priced for one fixture at one point in time."""
    event_key: str
    minutes_elapsed: float
    score: str
    sport: str
    model: str
    markets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    state_warnings: List[str] = field(default_factory=list)
    priced_at_minute: float = 0.0

    def fair_price(self, market: str, outcome: str) -> Optional[float]:
        return self.markets.get(market, {}).get(outcome)


def price_live_soccer_card(event_key: str, home_rate: float, away_rate: float,
                           state: MatchState) -> LiveCard:
    """Price the full live soccer card from score, clock and cards."""
    warnings = state.validate()
    card = LiveCard(event_key=event_key, minutes_elapsed=state.minutes_elapsed,
                    score=f"{state.home_score}-{state.away_score}",
                    sport="soccer", model="live_poisson_time_decay",
                    state_warnings=warnings,
                    priced_at_minute=state.minutes_elapsed)

    card.markets["h2h"] = live_match_odds(home_rate, away_rate, state)
    card.markets["double_chance"] = live_double_chance(home_rate, away_rate, state)
    card.markets["draw_no_bet"] = live_draw_no_bet(home_rate, away_rate, state)
    card.markets["btts"] = live_btts(home_rate, away_rate, state)
    card.markets["next_goal"] = live_next_goal(home_rate, away_rate, state)
    card.markets["correct_score"] = live_correct_score(home_rate, away_rate, state)
    card.markets["team_total_home_1.5"] = live_team_total(
        home_rate, away_rate, state, "home", 1.5)
    card.markets["team_total_away_0.5"] = live_team_total(
        home_rate, away_rate, state, "away", 0.5)
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        card.markets[f"totals_{line}"] = live_totals(home_rate, away_rate, state, line)
    return card


def price_live_high_scoring_card(event_key: str, sport: str,
                                 expected_home: float, expected_away: float,
                                 state: MatchState,
                                 lines: Optional[Dict[str, float]] = None) -> LiveCard:
    """Price the full live card for basketball / gridiron / baseball / hockey."""
    warnings = state.validate()
    resolved = resolve_sport(sport)
    params = SPORT_PARAMS.get(resolved, SPORT_PARAMS["basketball"])
    lines = lines or {}

    card = LiveCard(event_key=event_key, minutes_elapsed=state.minutes_elapsed,
                    score=f"{state.home_score}-{state.away_score}",
                    sport=resolved, model="live_normal_time_decay",
                    state_warnings=warnings,
                    priced_at_minute=state.minutes_elapsed)

    card.markets["moneyline"] = live_moneyline_high_scoring(
        expected_home, expected_away, state, params)

    spread_line = lines.get("spread", -2.5)
    sp = live_spread_high_scoring(expected_home, expected_away, state, spread_line, params)
    card.markets[f"spreads_{spread_line}"] = {
        "home": sp.home, "away": sp.away, "push": sp.push}

    total_line = lines.get("total", params.typical_total)
    tp = live_total_high_scoring(expected_home, expected_away, state, total_line, params)
    card.markets[f"totals_{total_line}"] = {
        "over": tp.over, "under": tp.under, "push": tp.push}

    for delta in (-10.0, -5.0, 0.0, 5.0, 10.0):
        ln = params.typical_total / 2 + delta
        t = live_total_high_scoring(expected_home, expected_away, state, ln, params)
        card.markets[f"team_total_home_{ln}"] = {"over": t.over, "under": t.under,
                                                 "push": t.push}
    return card
