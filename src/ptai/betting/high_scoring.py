"""
High-Scoring Sport Models - normal margin distribution, not Poisson.

Basketball, gridiron and hockey were being priced with the soccer Poisson
scoreline grid. That is wrong in a way that costs money rather than just
being inelegant:

    A Poisson total has variance equal to its mean. An NBA game totalling
    224 points implies a total standard deviation of sqrt(2*112) = 15.0.
    The real standard deviation of an NBA total is about 20-22.

So Poisson understates the spread by roughly 30%, which makes every line
near the total look further away in probability terms than it is - and an
over/under that looks like a 5% edge is often a 2% edge. NFL is worse:
Poisson implies a total std of 6.7 against a real 13.5.

The right model for these sports is a normal distribution on the total and
on the margin, because scoring is many near-independent small events rather
than rare ones. The standard deviations below are long-run league figures,
used as priors and overridable per fixture.

Push handling
-------------
A whole-number line can land exactly. The normal is continuous, so the push
probability is computed over the integer interval around the line rather
than treated as zero - otherwise every integer line is priced as if a push
were impossible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# League priors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SportParams:
    """Long-run scoring parameters for one sport."""
    sport: str
    typical_total: float
    total_std: float
    margin_std: float
    home_advantage_points: float
    scoring_unit: float          # 1 for goals/points, used for push width
    is_integer_score: bool
    notes: str = ""


SPORT_PARAMS: Dict[str, SportParams] = {
    "basketball": SportParams(
        "basketball", typical_total=225.0, total_std=20.5, margin_std=12.0,
        home_advantage_points=2.5, scoring_unit=1.0, is_integer_score=True,
        notes="NBA. Poisson would imply a total std of ~15 vs ~20.5 actual.",
    ),
    "football": SportParams(
        "football", typical_total=45.0, total_std=13.5, margin_std=13.5,
        home_advantage_points=2.0, scoring_unit=1.0, is_integer_score=True,
        notes="NFL. Margin std and total std are close because scores are low-scoring "
              "relative to their variance.",
    ),
    "baseball": SportParams(
        "baseball", typical_total=8.6, total_std=4.3, margin_std=4.4,
        home_advantage_points=0.35, scoring_unit=1.0, is_integer_score=True,
        notes="MLB. Baseball is the borderline case: low enough scoring that Poisson "
              "is not unreasonable, but the normal fits the total better.",
    ),
    "hockey": SportParams(
        "hockey", typical_total=6.1, total_std=2.7, margin_std=2.9,
        home_advantage_points=0.30, scoring_unit=1.0, is_integer_score=True,
        notes="NHL. Regulation only - overtime/shootout changes the moneyline "
              "settlement and must be handled separately.",
    ),
}


# Feeds and the rest of PTAI label fixtures by LEAGUE ("nba", "nfl"), while
# the model parameters are keyed by SPORT ("basketball"). Without this map an
# NBA fixture silently falls through to the soccer Poisson grid and gets
# priced with goal rates of 1.65 - plausible-looking and completely wrong.
LEAGUE_TO_SPORT: Dict[str, str] = {
    "nba": "basketball", "wnba": "basketball", "ncaam": "basketball",
    "euroleague": "basketball", "acb": "basketball",
    "nfl": "football", "ncaaf": "football", "cfl": "football",
    "mlb": "baseball", "npb": "baseball", "kbo": "baseball",
    "nhl": "hockey", "khl": "hockey", "shl": "hockey", "ahl": "hockey",
    "epl": "soccer", "laliga": "soccer", "seriea": "soccer",
    "bundesliga": "soccer", "ucl": "soccer", "mls": "soccer", "ligue1": "soccer",
    "atp": "tennis", "wta": "tennis",
}


def resolve_sport(sport_or_league: str) -> str:
    """Map a league code to its sport, passing through anything already a sport."""
    key = (sport_or_league or "").strip().lower()
    return LEAGUE_TO_SPORT.get(key, key)


def params_for(sport: str) -> SportParams:
    return SPORT_PARAMS.get(resolve_sport(sport), SPORT_PARAMS["basketball"])


def is_high_scoring(sport: str) -> bool:
    """
    Whether this sport should use the normal model rather than Poisson.

    Accepts either a sport ("basketball") or a league code ("nba").
    Hockey and baseball are borderline; they are included because the total
    is what gets bet most and the normal fits it better. Soccer stays on
    Poisson, where it is genuinely the right model.
    """
    return resolve_sport(sport) in SPORT_PARAMS


# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def prob_above(value: float, mu: float, sigma: float) -> float:
    return 1.0 - normal_cdf(value, mu, sigma)


def prob_between(lo: float, hi: float, mu: float, sigma: float) -> float:
    return max(0.0, normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma))


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

@dataclass
class TotalPrices:
    line: float
    over: float
    under: float
    push: float
    mean: float
    std: float
    z_score: float
    quarter_line: bool = False
    components: List[float] = field(default_factory=list)


def price_total_normal(expected_total: float, total_std: float, line: float,
                       is_integer_score: bool = True) -> TotalPrices:
    """
    Over/under on a normally distributed total.

    For a whole-number line the push band is the integer interval around it
    (e.g. line 225 pushes if the total is exactly 225, i.e. in [224.5, 225.5]
    for integer scoring). For a half line there is no push.
    """
    if total_std <= 0:
        raise ValueError("total_std must be positive")

    is_whole = abs(line - round(line)) < 1e-9
    is_quarter = (abs(line * 4 - round(line * 4)) < 1e-9
                  and abs(line * 2 - round(line * 2)) > 1e-9)

    if is_quarter:
        lo_line = math.floor(line * 2) / 2.0
        hi_line = lo_line + 0.5
        a = price_total_normal(expected_total, total_std, lo_line, is_integer_score)
        b = price_total_normal(expected_total, total_std, hi_line, is_integer_score)
        return TotalPrices(
            line=line,
            over=round((a.over + b.over) / 2.0, 4),
            under=round((a.under + b.under) / 2.0, 4),
            push=0.0, quarter_line=True, components=[lo_line, hi_line],
            mean=round(expected_total, 3), std=round(total_std, 3),
            z_score=round((expected_total - line) / total_std, 4),
        )

    if is_whole and is_integer_score:
        push = prob_between(line - 0.5, line + 0.5, expected_total, total_std)
        over = prob_above(line + 0.5, expected_total, total_std)
        under = normal_cdf(line - 0.5, expected_total, total_std)
    else:
        push = 0.0
        over = prob_above(line, expected_total, total_std)
        under = 1.0 - over

    return TotalPrices(
        line=line, over=round(over, 4), under=round(under, 4), push=round(push, 4),
        mean=round(expected_total, 3), std=round(total_std, 3),
        z_score=round((expected_total - line) / total_std, 4),
        quarter_line=False,
    )


# ---------------------------------------------------------------------------
# Spreads / handicaps
# ---------------------------------------------------------------------------

@dataclass
class SpreadPrices:
    line: float
    home: float
    away: float
    push: float
    mean_margin: float
    margin_std: float
    z_score: float
    quarter_line: bool = False
    components: List[float] = field(default_factory=list)


def price_spread_normal(expected_margin: float, margin_std: float, line: float,
                        is_integer_score: bool = True) -> SpreadPrices:
    """
    Point spread from a normal margin distribution.

    `expected_margin` is home minus away, positive = home favoured. `line` is
    the home handicap, so home covers when margin + line > 0.
    """
    if margin_std <= 0:
        raise ValueError("margin_std must be positive")

    is_whole = abs(line - round(line)) < 1e-9
    is_quarter = (abs(line * 4 - round(line * 4)) < 1e-9
                  and abs(line * 2 - round(line * 2)) > 1e-9)

    if is_quarter:
        lo = math.floor(line * 2) / 2.0
        hi = lo + 0.5
        a = price_spread_normal(expected_margin, margin_std, lo, is_integer_score)
        b = price_spread_normal(expected_margin, margin_std, hi, is_integer_score)
        return SpreadPrices(
            line=line,
            home=round((a.home + b.home) / 2.0, 4),
            away=round((a.away + b.away) / 2.0, 4),
            push=0.0, quarter_line=True, components=[lo, hi],
            mean_margin=round(expected_margin, 3), margin_std=round(margin_std, 3),
            z_score=round((expected_margin + line) / margin_std, 4),
        )

    # margin + line > 0 means home covers
    threshold = -line
    if is_whole and is_integer_score:
        push = prob_between(threshold - 0.5, threshold + 0.5, expected_margin, margin_std)
        home = prob_above(threshold + 0.5, expected_margin, margin_std)
        away = normal_cdf(threshold - 0.5, expected_margin, margin_std)
    else:
        push = 0.0
        home = prob_above(threshold, expected_margin, margin_std)
        away = 1.0 - home

    return SpreadPrices(
        line=line, home=round(home, 4), away=round(away, 4), push=round(push, 4),
        mean_margin=round(expected_margin, 3), margin_std=round(margin_std, 3),
        z_score=round((expected_margin + line) / margin_std, 4), quarter_line=False,
    )


def price_moneyline_normal(expected_margin: float, margin_std: float,
                           draw_possible: bool = False,
                           ot_resolves: bool = True) -> Dict[str, float]:
    """
    Two-way moneyline from the margin distribution.

    For sports with no draw the away probability is simply margin < 0.
    `ot_resolves` matters for hockey: a regulation draw goes to overtime, so
    the moneyline resolves on the eventual winner, not on the regulation
    score.
    """
    home = prob_above(0.0, expected_margin, margin_std)
    if draw_possible:
        return {"home": round(home, 4), "away": round(1.0 - home, 4),
                "note": "draw pushed to overtime" if ot_resolves else "regulation result"}
    return {"home": round(home, 4), "away": round(1.0 - home, 4)}


# ---------------------------------------------------------------------------
# Margin of victory / winning margin bands
# ---------------------------------------------------------------------------

def price_margin_band(expected_margin: float, margin_std: float,
                      lo: int, hi: int) -> float:
    """
    P(home wins by between lo and hi points inclusive).

    Bands like "1-5", "6-10", "11+" are standard NFL/NBA markets and cannot
    be derived from a Poisson scoreline grid at these scoring levels.
    """
    return round(prob_between(lo - 0.5, hi + 0.5, expected_margin, margin_std), 4)


def price_margin_bands(expected_margin: float, margin_std: float,
                       bands: List[Tuple[int, int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lo, hi in bands:
        out[f"{lo}-{hi}"] = price_margin_band(expected_margin, margin_std, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Team totals and periods
# ---------------------------------------------------------------------------

def price_team_total_normal(team_expected: float, team_std: float,
                            line: float) -> TotalPrices:
    """One team's points over/under."""
    return price_total_normal(team_expected, team_std, line)


def price_period_total(expected_full_total: float, total_std: float,
                       line: float, period_fraction: float) -> TotalPrices:
    """
    Quarter/half total.

    The mean scales linearly with the fraction of the game; the standard
    deviation scales with the SQUARE ROOT of the fraction, because variances
    add over independent periods. Scaling the std linearly too would
    badly overstate first-quarter variance.
    """
    frac = max(0.0, min(1.0, period_fraction))
    return price_total_normal(expected_full_total * frac,
                              total_std * math.sqrt(frac), line)


def price_period_spread(expected_full_margin: float, margin_std: float,
                        line: float, period_fraction: float) -> SpreadPrices:
    frac = max(0.0, min(1.0, period_fraction))
    return price_spread_normal(expected_full_margin * frac,
                               margin_std * math.sqrt(frac), line)


# ---------------------------------------------------------------------------
# Race to N points
# ---------------------------------------------------------------------------

def price_race_to_points(expected_home: float, expected_away: float, n: int,
                         home_std: Optional[float] = None,
                         away_std: Optional[float] = None) -> Dict[str, float]:
    """
    First team to N points.

    Approximated from the ordering of two scoring processes: each scoring
    event belongs to the home team with probability proportional to its
    scoring rate. This is an approximation and is labelled as such - an exact
    answer needs the joint time distribution, which these priors do not carry.
    """
    if n < 1:
        return {"home": 0.0, "away": 0.0, "none": 1.0}
    total = expected_home + expected_away
    if total <= 0:
        return {"home": 0.0, "away": 0.0, "none": 1.0}
    p_home = expected_home / total

    # Negative binomial: P(home scores its nth before away scores its nth),
    # scaled by the chance that anyone reaches n at all.
    hs = home_std if home_std is not None else math.sqrt(max(expected_home, 1.0))
    as_ = away_std if away_std is not None else math.sqrt(max(expected_away, 1.0))

    p_home_lt_n = normal_cdf(n - 0.5, expected_home, hs)
    p_away_lt_n = normal_cdf(n - 0.5, expected_away, as_)
    neither = p_home_lt_n * p_away_lt_n
    reaches = max(0.0, 1.0 - neither)

    order_home = sum(_neg_binom(n, k, p_home) for k in range(0, n))
    order_away = sum(_neg_binom(n, k, 1.0 - p_home) for k in range(0, n))
    return {
        "home": round(order_home * reaches, 4),
        "away": round(order_away * reaches, 4),
        "none": round(min(1.0, neither), 4),
        "method": "negative_binomial_ordering (approximation)",
    }


def _neg_binom(n: int, k: int, p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return math.comb(n + k - 1, k) * (p ** n) * ((1.0 - p) ** k)


# ---------------------------------------------------------------------------
# Whole card for a high-scoring fixture
# ---------------------------------------------------------------------------

@dataclass
class HighScoringCard:
    sport: str
    event_key: str
    expected_home: float
    expected_away: float
    expected_total: float
    expected_margin: float
    total_std: float
    margin_std: float
    moneyline: Dict[str, float] = field(default_factory=dict)
    totals: Dict[float, TotalPrices] = field(default_factory=dict)
    spreads: Dict[float, SpreadPrices] = field(default_factory=dict)
    margin_bands: Dict[str, float] = field(default_factory=dict)
    team_totals: Dict[str, TotalPrices] = field(default_factory=dict)
    periods: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model: str = "normal_margin"

    def to_dict(self) -> Dict:
        return {
            "sport": self.sport, "event_key": self.event_key,
            "expected": {"home": self.expected_home, "away": self.expected_away,
                         "total": self.expected_total, "margin": self.expected_margin},
            "std": {"total": self.total_std, "margin": self.margin_std},
            "moneyline": self.moneyline,
            "totals": {str(k): {"over": v.over, "under": v.under, "push": v.push,
                                "z": v.z_score} for k, v in self.totals.items()},
            "spreads": {str(k): {"home": v.home, "away": v.away, "push": v.push,
                                 "z": v.z_score} for k, v in self.spreads.items()},
            "margin_bands": self.margin_bands,
            "team_totals": {k: {"over": v.over, "under": v.under, "push": v.push}
                            for k, v in self.team_totals.items()},
            "model": self.model,
        }


def price_high_scoring_card(event_key: str, sport: str,
                            expected_home: float, expected_away: float,
                            total_std: Optional[float] = None,
                            margin_std: Optional[float] = None,
                            total_lines: Optional[List[float]] = None,
                            spread_lines: Optional[List[float]] = None,
                            team_total_lines: Optional[List[float]] = None,
                            margin_bands: Optional[List[Tuple[int, int]]] = None,
                            periods: Optional[Dict[str, float]] = None) -> HighScoringCard:
    """
    Price a full card for basketball / gridiron / baseball / hockey.

    Uses the normal margin model rather than the soccer Poisson grid, with
    league-standard deviations unless overridden.
    """
    sport = resolve_sport(sport)
    p = params_for(sport)
    t_std = total_std if total_std is not None else p.total_std
    m_std = margin_std if margin_std is not None else p.margin_std

    total = expected_home + expected_away
    margin = expected_home - expected_away

    card = HighScoringCard(
        sport=sport, event_key=event_key,
        expected_home=round(expected_home, 2), expected_away=round(expected_away, 2),
        expected_total=round(total, 2), expected_margin=round(margin, 2),
        total_std=t_std, margin_std=m_std,
        moneyline=price_moneyline_normal(margin, m_std,
                                         draw_possible=(sport == "hockey")),
    )

    for ln in (total_lines or [round(p.typical_total), p.typical_total + 0.5,
                               p.typical_total - 0.5]):
        card.totals[ln] = price_total_normal(total, t_std, ln, p.is_integer_score)

    for ln in (spread_lines or [0.0, -1.5, 1.5, -3.5, 3.5, -6.5, 6.5]):
        card.spreads[ln] = price_spread_normal(margin, m_std, ln, p.is_integer_score)

    if margin_bands:
        card.margin_bands = price_margin_bands(margin, m_std, margin_bands)
    elif sport in ("football", "basketball"):
        card.margin_bands = price_margin_bands(
            margin, m_std, [(1, 3), (4, 6), (7, 10), (11, 15), (16, 25)])

    team_std = t_std / 2.0
    for ln in (team_total_lines or []):
        card.team_totals[f"home_{ln}"] = price_team_total_normal(expected_home, team_std, ln)
        card.team_totals[f"away_{ln}"] = price_team_total_normal(expected_away, team_std, ln)

    for name, frac in (periods or {}).items():
        ft = price_period_total(total, t_std, round(total * frac), frac)
        card.periods[name] = {"fraction": frac, "expected": round(total * frac, 2),
                              "over": ft.over, "under": ft.under}

    return card
