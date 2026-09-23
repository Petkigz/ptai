"""
Sports Models - real probabilistic models, not vibes.

Three model families, each appropriate to the scoring structure of the
sport it is applied to:

Poisson / Dixon-Coles   soccer and other low-scoring sports. Goals are
                        approximately independent rare events, so the
                        scoreline distribution follows from two rates.
                        Dixon-Coles adds a correction because 0-0, 1-0,
                        0-1 and 1-1 are empirically more (or less) common
                        than independence predicts.

Elo                     basketball, gridiron, tennis. Ratings difference
                        maps to expected score through a logistic curve,
                        with home advantage and margin-of-victory scaling.

Base rate               the prior that keeps small samples honest. With
                        3 games of history, the league average matters far
                        more than the observed form.

The engine blends model output with the de-vigged market consensus, and
reports an honest uncertainty band. It will NOT emit a confident number
from a handful of games - that is precisely how fake edge gets invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .sports_data import BookOdds, ConsensusOdds, SportsEvent


# ---------------------------------------------------------------------------
# Poisson / Dixon-Coles
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon & Coles (1997) low-score correction.

    Empirically, draws at 0-0 and 1-1 are more common than a plain Poisson
    model predicts, and 1-0 / 0-1 slightly less so. rho is typically in
    [-0.13, -0.03]; a positive rho would push probabilities outside [0,1]
    for some inputs so it is clipped.
    """
    rho = max(-0.2, min(0.0, rho))
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def scoreline_matrix(home_lambda: float, away_lambda: float, max_goals: int = 10,
                     rho: float = -0.05) -> List[List[float]]:
    """Joint P(home=x, away=y), normalised to sum to 1."""
    home_lambda = max(0.05, home_lambda)
    away_lambda = max(0.05, away_lambda)
    grid = []
    for x in range(max_goals + 1):
        row = []
        for y in range(max_goals + 1):
            p = poisson_pmf(x, home_lambda) * poisson_pmf(y, away_lambda)
            p *= max(0.0, dixon_coles_tau(x, y, home_lambda, away_lambda, rho))
            row.append(p)
        grid.append(row)
    total = sum(sum(r) for r in grid)
    if total <= 0:
        return grid
    return [[p / total for p in row] for row in grid]


@dataclass
class SoccerForecast:
    home_lambda: float
    away_lambda: float
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    p_over_2_5: float
    p_under_2_5: float
    p_btts: float            # both teams to score
    most_likely_score: str
    most_likely_score_p: float
    over_2_5_curve: Dict[str, float] = field(default_factory=dict)

    def outcome_probs(self) -> Dict[str, float]:
        return {"home": self.p_home, "draw": self.p_draw, "away": self.p_away}


def poisson_soccer(home_attack: float, home_defence: float,
                   away_attack: float, away_defence: float,
                   league_home_goals: float = 1.50, league_away_goals: float = 1.20,
                   home_advantage: float = 1.10, rho: float = -0.05,
                   max_goals: int = 10) -> SoccerForecast:
    """
    Classic Maher (1982) attack/defence Poisson model.

    Strengths are ratios relative to league average:
      home_lambda = league_home_goals * home_attack * away_defence * home_adv
      away_lambda = league_away_goals * away_attack * home_defence

    A team with attack 1.3 and a league-average opponent defence concedes
    30% more than league average. Everything is a ratio, so the model is
    portable across leagues with different scoring levels.
    """
    home_lambda = league_home_goals * max(0.2, home_attack) * max(0.2, away_defence) * home_advantage
    away_lambda = league_away_goals * max(0.2, away_attack) * max(0.2, home_defence)

    grid = scoreline_matrix(home_lambda, away_lambda, max_goals, rho)

    p_home = p_draw = p_away = 0.0
    p_over = p_under = p_btts = 0.0
    curve: Dict[str, float] = {}
    best_p, best_score = -1.0, ""
    for x, row in enumerate(grid):
        for y, p in enumerate(row):
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p
            total = x + y
            if total > 2.5:
                p_over += p
            else:
                p_under += p
            if x >= 1 and y >= 1:
                p_btts += p
            key = str(total)
            curve[key] = round(curve.get(key, 0.0) + p, 4)
            if p > best_p:
                best_p, best_score = p, f"{x}-{y}"

    return SoccerForecast(
        home_lambda=round(home_lambda, 4), away_lambda=round(away_lambda, 4),
        p_home=round(p_home, 4), p_draw=round(p_draw, 4), p_away=round(p_away, 4),
        expected_home_goals=round(home_lambda, 3), expected_away_goals=round(away_lambda, 3),
        p_over_2_5=round(p_over, 4), p_under_2_5=round(p_under, 4), p_btts=round(p_btts, 4),
        most_likely_score=best_score, most_likely_score_p=round(best_p, 4),
        over_2_5_curve=curve,
    )


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

@dataclass
class EloForecast:
    home_rating: float
    away_rating: float
    p_home: float
    p_draw: float
    p_away: float
    elo_diff: float
    home_advantage: float
    reliability: float          # 0-1, driven by games played

    def outcome_probs(self) -> Dict[str, float]:
        return {"home": self.p_home, "draw": self.p_draw, "away": self.p_away}


def elo_expected(rating_a: float, rating_b: float, home_advantage: float = 0.0,
                 scale: float = 400.0) -> float:
    """Logistic Elo expectation for a, including an optional home advantage."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a - home_advantage) / scale))


def elo_forecast(home_rating: float, away_rating: float, home_advantage: float = 60.0,
                 draw_width: float = 25.0, games_played_home: int = 10,
                 games_played_away: int = 10, scale: float = 400.0) -> EloForecast:
    """
    Elo with a draw band, for sports where draws are possible.

    The draw is modelled as a band around parity: the closer the two
    ratings, the more likely a draw. For draw-free sports pass
    draw_width=0 and the mass is split between home and away in proportion
    to their raw expectations.

    `reliability` shrinks the confidence of the forecast when either side
    has played few games - a rating built on 2 matches is mostly noise.
    """
    eff_home = home_rating + home_advantage
    diff = eff_home - away_rating
    p_home_raw = elo_expected(home_rating, away_rating, home_advantage, scale)

    # Draw mass peaks at parity, decays with the square of the rating gap.
    if draw_width > 0:
        p_draw = math.exp(-0.5 * (diff / draw_width) ** 2) * 0.30
        p_draw = max(0.0, min(0.35, p_draw))
    else:
        p_draw = 0.0

    remaining = 1.0 - p_draw
    p_home = remaining * p_home_raw
    p_away = remaining * (1.0 - p_home_raw)

    # Round then renormalise: rounding each leg to 4dp independently drifts
    # the trio off 1.0, and downstream code treats this as a distribution.
    p_home, p_draw, p_away = (round(p_home, 4), round(p_draw, 4), round(p_away, 4))
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = (round(p_home / total, 4), round(p_draw / total, 4),
                                  round(p_away / total, 4))
        drift = round(1.0 - (p_home + p_draw + p_away), 4)
        if abs(drift) > 0:
            p_home = round(p_home + drift, 4)

    games = min(games_played_home, games_played_away)
    reliability = games / (games + 10.0)  # 10 games -> 0.50, 40 games -> 0.80

    return EloForecast(
        home_rating=round(home_rating, 1), away_rating=round(away_rating, 1),
        p_home=round(p_home, 4), p_draw=round(p_draw, 4), p_away=round(p_away, 4),
        elo_diff=round(diff, 1), home_advantage=home_advantage,
        reliability=round(reliability, 3),
    )


def margin_of_victory_multiplier(score_diff: int) -> float:
    """
    FiveThirtyEight-style MOV scaling for Elo updates.

    A 30-point blowout carries more information than a 3-point win, but
    with strongly diminishing returns, otherwise one lopsided game swings a
    rating more than ten competitive ones.
    """
    d = abs(int(score_diff))
    if d <= 1:
        return 1.0
    return math.log(d + 1.0) * (2.2 / ((abs(score_diff) * 0.001) + 2.2))


def elo_update(rating: float, opponent: float, actual: float, k: float = 20.0,
               score_diff: int = 1, home_advantage: float = 0.0,
               scale: float = 400.0) -> float:
    """actual: 1.0 win, 0.5 draw, 0.0 loss."""
    expected = elo_expected(rating, opponent, home_advantage, scale)
    return rating + k * margin_of_victory_multiplier(score_diff) * (actual - expected)


def decay_toward(rating: float, target: float, half_life_days: float, elapsed_days: float) -> float:
    """
    Regression of a rating toward a baseline over time. Teams change
    personnel; a rating from two seasons ago should not be trusted as-is.
    """
    if half_life_days <= 0:
        return rating
    factor = 0.5 ** (elapsed_days / half_life_days)
    return target + (rating - target) * factor


# ---------------------------------------------------------------------------
# Base rates and priors
# ---------------------------------------------------------------------------

# Empirical long-run outcome frequencies. These are the priors that keep a
# small sample from producing a 90% favourite. Values are widely published
# long-run league averages, not fitted to this codebase.
BASE_RATES: Dict[str, Dict[str, float]] = {
    "soccer": {"home": 0.45, "draw": 0.26, "away": 0.29},
    "basketball": {"home": 0.60, "away": 0.40},
    "football": {"home": 0.57, "away": 0.43},
    "baseball": {"home": 0.54, "away": 0.46},
    "hockey": {"home": 0.55, "away": 0.45},
    "tennis": {"home": 0.63, "away": 0.37},
    "default": {"home": 0.55, "away": 0.45},
}


def base_rate(sport: str) -> Dict[str, float]:
    return dict(BASE_RATES.get(sport, BASE_RATES["default"]))


def shrink_to_base_rate(probs: Dict[str, float], n_games: int, half_life: float = 10.0) -> Dict[str, float]:
    """
    James-Stein style shrinkage: weight = n / (n + half_life).

    With 0 games you get the pure prior. With 30 games the data dominates.
    Without this, a team that won its first two games reads as a near
    certainty and the sizing layer bets accordingly.
    """
    if n_games <= 0:
        return dict(probs)
    w = n_games / (n_games + half_life)
    keys = list(probs.keys())
    prior_keys = {k: 1.0 / len(keys) for k in keys}
    # try to use the sport prior if the caller passed one in probs['_prior']
    out = {}
    for k in keys:
        out[k] = w * probs[k] + (1.0 - w) * prior_keys[k]
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else dict(probs)


# ---------------------------------------------------------------------------
# Model engine
# ---------------------------------------------------------------------------

@dataclass
class SportsForecast:
    event_key: str
    sport: str
    probs: Dict[str, float]
    confidence: float
    uncertainty: float
    model_used: str
    reasoning: str
    sources: List[str]
    n_games_used: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_synthetic: bool = False


class SportsModelEngine:
    """
    Produces a model probability per fixture, blended with the market and
    reported with an honest uncertainty band.

    The blend weight is driven by how much real information exists. A model
    built on 40 games of ratings deserves weight; one built on defaults
    does not, and in that case the engine says so instead of inventing an
    opinion.
    """

    def __init__(self, max_model_weight: float = 0.55, min_games_for_weight: int = 10):
        self.max_model_weight = max_model_weight
        self.min_games_for_weight = min_games_for_weight

    # -- individual models --------------------------------------------------

    def forecast_soccer(self, event: SportsEvent, strengths: Dict[str, Dict[str, float]],
                        league_home_goals: float = 1.50,
                        league_away_goals: float = 1.20) -> SportsForecast:
        """
        strengths = {'home': {'attack':1.2,'defence':0.9,'games':14},
                     'away': {...}}
        defence is expressed as a ratio where >1 means concedes more.
        """
        home = strengths.get("home") or {}
        away = strengths.get("away") or {}
        n = int(min(home.get("games", 0), away.get("games", 0)))

        fc = poisson_soccer(
            home_attack=float(home.get("attack", 1.0)),
            home_defence=float(home.get("defence", 1.0)),
            away_attack=float(away.get("attack", 1.0)),
            away_defence=float(away.get("defence", 1.0)),
            league_home_goals=league_home_goals, league_away_goals=league_away_goals,
        )
        raw = {"home": fc.p_home, "draw": fc.p_draw, "away": fc.p_away}
        probs = shrink_to_base_rate(raw, n, half_life=10.0)

        uncertainty = self._uncertainty(n, spread=max(probs.values()) - min(probs.values()))
        confidence = max(0.15, min(0.85, 1.0 - uncertainty))

        return SportsForecast(
            event_key=event.key, sport=event.sport,
            probs={k: round(v, 4) for k, v in probs.items()},
            confidence=round(confidence, 3), uncertainty=round(uncertainty, 3),
            model_used="dixon_coles_poisson",
            reasoning=(f"Poisson lams {fc.expected_home_goals} vs {fc.expected_away_goals}, "
                       f"likely score {fc.most_likely_score} ({fc.most_likely_score_p:.0%}), "
                       f"o2.5 {fc.p_over_2_5:.0%}, btts {fc.p_btts:.0%}, "
                       f"shrunk toward base rate on {n} games"),
            sources=["poisson", "dixon_coles"],
            n_games_used=n,
        )

    def forecast_elo(self, event: SportsEvent, ratings: Dict[str, Dict[str, float]],
                     draw_width: float = 0.0, home_advantage: float = 60.0) -> SportsForecast:
        home = ratings.get("home") or {}
        away = ratings.get("away") or {}
        hr = float(home.get("rating", 1500.0))
        ar = float(away.get("rating", 1500.0))
        n = int(min(home.get("games", 0), away.get("games", 0)))

        fc = elo_forecast(hr, ar, home_advantage=home_advantage, draw_width=draw_width,
                          games_played_home=int(home.get("games", 0)),
                          games_played_away=int(away.get("games", 0)))
        raw = {k: v for k, v in fc.outcome_probs().items() if v > 0}
        probs = shrink_to_base_rate(raw, n, half_life=10.0)
        uncertainty = self._uncertainty(n, spread=max(probs.values()) - min(probs.values()))
        # Elo reliability directly widens the band
        uncertainty = min(0.6, uncertainty + (1.0 - fc.reliability) * 0.15)
        confidence = max(0.15, min(0.85, fc.reliability * (1.0 - uncertainty)))

        return SportsForecast(
            event_key=event.key, sport=event.sport,
            probs={k: round(v, 4) for k, v in probs.items()},
            confidence=round(confidence, 3), uncertainty=round(uncertainty, 3),
            model_used="elo",
            reasoning=(f"Elo {hr:.0f} vs {ar:.0f} (diff {fc.elo_diff:+.0f}, "
                       f"HA {home_advantage}), reliability {fc.reliability:.2f} on {n} games"),
            sources=["elo"],
            n_games_used=n,
        )

    # -- blending -----------------------------------------------------------

    def blend_with_market(self, forecast: SportsForecast, consensus: Optional[ConsensusOdds],
                          label_map: Optional[Dict[str, str]] = None) -> SportsForecast:
        """
        Combine model and market fair values.

        Weighting rules:
        - The model gets at most max_model_weight, and less than that when
          it is built on few games.
        - If a SHARP book (Pinnacle / exchange) is present its de-vigged
          price is preferred over the plain multi-book average, because the
          sharp book is where informed money actually trades.
        - The band widens when model and market disagree a lot. Honest
          disagreement is information about uncertainty, not a free edge.
        """
        if consensus is None or not consensus.fair_probs:
            forecast.reasoning += " | no market data, model alone"
            forecast.is_synthetic = forecast.is_synthetic
            return forecast

        label_map = label_map or {}
        market_probs: Dict[str, float] = {}
        for label, p in consensus.fair_probs.items():
            key = label_map.get(label, label)
            market_probs[key] = p
        if consensus.sharp_probs:
            for label, p in consensus.sharp_probs.items():
                key = label_map.get(label, label)
                market_probs[key] = p  # sharp overrides the plain average

        w = self._model_weight(forecast.n_games_used)
        blended: Dict[str, float] = {}
        max_disagreement = 0.0
        for key, mp in forecast.probs.items():
            mkt = market_probs.get(key)
            if mkt is None:
                blended[key] = mp
                continue
            blended[key] = w * mp + (1.0 - w) * mkt
            max_disagreement = max(max_disagreement, abs(mp - mkt))

        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total, 4) for k, v in blended.items()}

        new_uncertainty = min(0.75, forecast.uncertainty + max_disagreement * 0.5)
        return SportsForecast(
            event_key=forecast.event_key, sport=forecast.sport,
            probs=blended,
            confidence=round(max(0.1, forecast.confidence - max_disagreement * 0.4), 3),
            uncertainty=round(new_uncertainty, 3),
            model_used=f"{forecast.model_used}+market_blend",
            reasoning=(f"{forecast.reasoning} | blended with market at model weight {w:.2f}"
                       f" (sharp={'yes' if consensus.sharp_probs else 'no'}, "
                       f"books={consensus.n_books}, max disagreement {max_disagreement:.3f})"),
            sources=forecast.sources + [f"market:{consensus.n_books}books"],
            n_games_used=forecast.n_games_used,
        )

    # -- edge ---------------------------------------------------------------

    def value_edges(self, forecast: SportsForecast, consensus: ConsensusOdds,
                    label_map: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Where does the model beat the best available price?

        edge = model_prob * decimal_odds - 1. Reported per outcome with the
        book that offers the price, so it can be verified against the feed.
        """
        label_map = label_map or {}
        out: List[Dict] = []
        for key, prob in forecast.probs.items():
            label = next((l for l, k in label_map.items() if k == key), key)
            price = consensus.best_price.get(label)
            book = consensus.best_book.get(label, "")
            if not price or price <= 1.0:
                continue
            commission = 0.0
            for b in consensus.books:
                if b.book == book:
                    commission = b.commission_pct
            eff = price * (1 - commission) if commission else price
            eff = 1.0 + (price - 1.0) * (1.0 - commission) if commission else price
            edge = prob * eff - 1.0
            out.append({
                "outcome": label, "model_prob": round(prob, 4),
                "price": round(price, 4), "book": book,
                "commission_pct": commission, "effective_odds": round(eff, 4),
                "edge": round(edge, 4), "edge_pct": round(edge * 100.0, 3),
                "is_value": edge > 0,
                "conservative_prob": round(max(0.0, prob - forecast.uncertainty), 4),
                "conservative_edge": round(max(0.0, prob - forecast.uncertainty) * eff - 1.0, 4),
            })
        return sorted(out, key=lambda d: d["edge"], reverse=True)

    # -- internals ----------------------------------------------------------

    def _model_weight(self, n_games: int) -> float:
        if n_games <= 0:
            return 0.10
        if n_games < self.min_games_for_weight:
            return 0.10 + 0.25 * (n_games / self.min_games_for_weight)
        return self.max_model_weight

    @staticmethod
    def _uncertainty(n_games: int, spread: float) -> float:
        """
        Uncertainty band from sample size and how lopsided the model is.

        Few games -> wide. A model that is nearly 50/50 is genuinely less
        informative than one that separates the teams, but it is not more
        uncertain, so spread contributes only mildly.
        """
        sample_term = 1.0 / (1.0 + max(0, n_games) / 8.0)   # 0 games -> 1.0, 40 -> 0.167
        spread_term = 0.10 * (1.0 - min(1.0, spread))
        return round(min(0.7, 0.55 * sample_term + spread_term + 0.05), 3)

    def get_report(self) -> Dict:
        return {
            "models": ["dixon_coles_poisson", "elo", "base_rate_prior", "market_blend"],
            "max_model_weight": self.max_model_weight,
            "min_games_for_weight": self.min_games_for_weight,
            "shrinkage": "w = n/(n+10), prior = base rate",
            "sharp_preference": "Pinnacle/exchange de-vigged price overrides multi-book average",
            "no_fabrication": "returns empty/low-confidence rather than inventing odds",
        }
