"""
Does the forecast beat the price? Measured, with uncertainty.

Borrowed from the sibling project (`avt-bot`) — not its domain, its discipline.
There, a model is only deployed if it beats the best simple estimator
out-of-sample, with a bootstrap confidence interval on the skill and a binomial
test on the entries it would actually have taken. The point of that protocol is
one sentence: *a number is not evidence until you say what it beat*.

PTAI had no such comparison. `forecast_skill` was `max(0, 1 - brier*2)` — a
rescaled Brier score, which is a number about nothing: in a market priced at
0.50, predicting 0.50 forever earns a Brier of 0.25 and a "skill" of 0.5. The
qualification gate read `brier <= 0.25` and `skill >= 0.6` as if they measured
edge. Neither of them ever looked at the price.

The market is PTAI's opponent and the only null that matters here: the claim
under test is "my probability is better than the price". So:

  * `model_brier`      — squared error of what the agent believed
  * `market_brier`     — squared error of what the PRICE said, on the same
                         trades, resolved the same way
  * `improvement`      — mean(market error - model error), in Brier units per
                         trade. Positive means the agent beat the price.
  * `ci_low/ci_high`   — paired bootstrap over the trades. The interval is on
                         `improvement`, so `ci_low > 0` is the whole claim:
                         "on these trades, and with this much uncertainty, the
                         forecast is still better than the price."
  * `p_value`          — the entries the agent TOOK, against the null that the
                         odds it paid were right: the trades are counted on the
                         side that was bought, at the price that side cost.
                         Not a plain binomial — each trade had its own
                         break-even price, so the null is Poisson-binomial and
                         is computed exactly.

Both halves are reported and both are required before live capital: better
probabilities than the price, AND entries that clear the odds they paid. They
are different questions — a forecast can be better calibrated than the market
while every trade still loses money by paying too much for the side — and the
first is not a licence to trade without the second.

Nothing here decides policy. `qualification.py` reads this and fails closed —
an unmeasured venue is not a skilled one, and "no evidence" is reported as
"no evidence" rather than as a pass.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Below this many paired trades there is nothing to say. A CI over 12 trades is
# a restatement of those 12 trades, not an estimate of anything.
MIN_PAIRED_TRADES = 30
# Bootstrap resamples. 600 gives a stable 95% interval at these sample sizes and
# keeps a full cycle (every venue, every cycle) cheap.
DEFAULT_ITERATIONS = 600
# One-sided 95%, the same confidence the winner selection uses.
ALPHA = 0.05

# About the FORECASTS: how they scored against the price.
FORECAST_BEATS_PRICE = "forecast_beats_price"
FORECAST_BEHIND_PRICE = "forecast_behind_price"
NO_EVIDENCE = "no_evidence"
INSUFFICIENT = "insufficient"
UNMEASURED = "unmeasured"


def _clip(p: float) -> float:
    """A probability that is a probability. 0 and 1 have no Brier scale."""
    return min(max(float(p), 1e-6), 1.0 - 1e-6)


@dataclass
class SkillEvidence:
    """The forecast against the price, trade by trade."""

    n: int = 0
    model_brier: Optional[float] = None
    market_brier: Optional[float] = None
    skill: Optional[float] = None          # relative: 1 - model/market
    improvement: Optional[float] = None    # Brier units per trade
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    # The entries taken, counted on the side that was bought and priced at what
    # that side cost. This is the money question, and it is not the same as the
    # Brier comparison above.
    hits: int = 0
    expected_hits: Optional[float] = None  # the null's own expectation
    p_value: Optional[float] = None
    verdict: str = UNMEASURED
    reason: str = "no paired forecast/price/outcome rows"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_skill_n": self.n,
            "market_skill_model_brier": _round(self.model_brier, 5),
            "market_skill_market_brier": _round(self.market_brier, 5),
            "market_skill": _round(self.skill, 4),
            "market_improvement": _round(self.improvement, 5),
            "market_skill_ci_low": _round(self.ci_low, 5),
            "market_skill_ci_high": _round(self.ci_high, 5),
            "market_skill_hits": self.hits,
            "market_skill_expected_hits": _round(self.expected_hits, 3),
            "market_skill_p_value": _round(self.p_value, 4),
            "market_skill_verdict": self.verdict,
            "market_skill_reason": self.reason,
        }

    @property
    def forecast_beats_price(self) -> bool:
        """Better probabilities than the price, beyond sampling doubt."""
        return (self.verdict == FORECAST_BEATS_PRICE
                and self.ci_low is not None and self.ci_low > 0.0)

    @property
    def entries_clear_the_odds(self) -> bool:
        """
        The trades taken won more often than the prices they paid implied.

        The null is the price itself: if the sides bought were priced right, the
        wins are a Poisson-binomial draw with those prices as probabilities, and
        a right tail means the agent has been picking sides the market got
        wrong. Fail closed on an unmeasured p — "not measured" is not "fine".
        """
        return (self.p_value is not None and self.p_value < ALPHA
                and self.expected_hits is not None
                and self.hits > self.expected_hits)

    @property
    def beats_market(self) -> bool:
        """
        Both halves of the claim, cleared. This is the deployable state.

        Either half alone is a trap: forecasts better than the price with losing
        entries means the agent is right about the world and wrong about the
        price it pays; winning entries with worse forecasts means luck with the
        market's odds.
        """
        return self.forecast_beats_price and self.entries_clear_the_odds

    @property
    def behind_market(self) -> bool:
        """The price beat the forecast, beyond doubt. Drift, not noise."""
        return (self.ci_high is not None and self.ci_high < 0.0)


def _round(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def pairs_from_rows(rows: Iterable[Any]) -> List[Tuple[float, float, float]]:
    """
    `(p_model, p_market, outcome)` for every row that has all three.

    All three on the SAME scale — the YES outcome — so there is no side
    arithmetic to get wrong. `forecast_prob` is the agent's probability for YES
    and `actual_outcome` is whether YES happened, both already stored that way;
    the market's own probability for YES is the price at entry, which older rows
    never recorded, so those rows are dropped rather than guessed at. Dropping
    is safe: coverage is reported, and a gate that fails closed cannot be fooled
    by rows that are missing.
    """
    pairs: List[Tuple[float, float, float]] = []
    for row in rows:
        try:
            forecast = row["forecast_prob"]
            price = row["yes_price"]
            outcome = row["actual_outcome"]
        except (KeyError, IndexError, TypeError):
            # A row shaped differently is not evidence; skip it rather than
            # inventing a price for it.
            continue
        if forecast is None or price is None or outcome is None:
            continue
        try:
            p_model = _clip(float(forecast))
            p_market = _clip(float(price))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(p_model) and math.isfinite(p_market)):
            continue
        pairs.append((p_model, p_market, 1.0 if float(outcome) > 0.5 else 0.0))
    return pairs


def entries_from_rows(rows: Iterable[Any]) -> Tuple[int, List[float]]:
    """
    `(wins, prices paid)` for the side each trade actually bought.

    The Brier comparison above lives on the YES scale, where it is exactly
    symmetric — (p_model - y)^2 for the NO side equals (1-p_model - (1-y))^2, so
    the same number. The MONEY question is not symmetric: a NO trade wins when
    YES does not, and it paid `1 - yes_price` for that side. Counting its win
    against the YES price would score a 0.90 NO buy as a 0.10 coin and credit
    the agent with an edge it was never offered.

    Rows without a side are treated as YES: that is what the old records meant,
    and every stored row from the single-opportunity path carries one.
    """
    wins = 0
    prices: List[float] = []
    for row in rows:
        try:
            price = row["yes_price"]
            outcome = row["actual_outcome"]
            side = row["side"]
        except (KeyError, IndexError, TypeError):
            continue
        if price is None or outcome is None:
            continue
        try:
            yes_price = _clip(float(price))
            yes_outcome = 1.0 if float(outcome) > 0.5 else 0.0
        except (TypeError, ValueError):
            continue
        bought_yes = str(side or "YES").strip().upper() in (
            "YES", "1", "LONG", "BUY", "TRUE", "")
        if bought_yes:
            prices.append(yes_price)
            wins += int(yes_outcome == 1.0)
        else:
            prices.append(1.0 - yes_price)
            wins += int(yes_outcome == 0.0)
    return wins, prices


def brier(pairs: Sequence[Tuple[float, float, float]], which: int) -> float:
    """Mean squared error of column `which` (0 = model, 1 = market)."""
    if not pairs:
        return float("nan")
    return sum((p[which] - p[2]) ** 2 for p in pairs) / len(pairs)


def _improvements(pairs: Sequence[Tuple[float, float, float]]) -> List[float]:
    """Per-trade (market error - model error). Positive = the agent was better."""
    return [((p[1] - p[2]) ** 2) - ((p[0] - p[2]) ** 2) for p in pairs]


def bootstrap_ci(values: Sequence[float], *, iterations: int = DEFAULT_ITERATIONS,
                 seed: int = 20260926) -> Tuple[float, float]:
    """
    Percentile bootstrap of the mean. Deterministic: same values, same interval.

    Determinism is not a nicety here — the qualification gate reads this, and a
    verdict that flips between two identical cycles would be a bug with a
    statistical costume on.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(int(iterations)):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[max(0, int(math.floor(ALPHA / 2 * len(means))))]
    hi = means[min(len(means) - 1, int(math.ceil((1 - ALPHA / 2) * len(means))) - 1)]
    return (lo, hi)


def poisson_binomial_tail(hits: int, probabilities: Sequence[float]) -> float:
    """
    P(X >= hits) when X is the sum of independent coins with different biases.

    Each trade had its own break-even price, so the null is not one binomial
    with one p — it is the sum of 150 binomials with 150 different p's. The
    plain binomial in the sibling project is an approximation; this is exact,
    and exact is the same cost at these sizes.

    The probabilities are the prices paid, so this answers precisely: "if the
    prices I paid were the true probabilities, how often would I have won this
    many or more, by luck?"
    """
    n = len(probabilities)
    if n == 0:
        return 1.0
    hits = int(hits)
    # "At least zero wins" is certain; "at least more wins than there were
    # trades" is impossible. Clamping the second case to n would answer 1.0 for
    # a question whose answer is 0.
    if hits <= 0:
        return 1.0
    if hits > n:
        return 0.0
    # Distribution over the number of wins, built coin by coin.
    dist = [0.0] * (n + 1)
    dist[0] = 1.0
    for i, p in enumerate(probabilities, start=1):
        p = min(max(float(p), 0.0), 1.0)
        for k in range(i, 0, -1):
            dist[k] = dist[k] * (1 - p) + dist[k - 1] * p
        dist[0] *= (1 - p)
    return min(1.0, float(sum(dist[hits:])))


def market_skill(rows: Iterable[Any], *, iterations: int = DEFAULT_ITERATIONS,
                 min_pairs: int = MIN_PAIRED_TRADES) -> SkillEvidence:
    """
    The forecast measured against the price it had to beat.

    Returns a verdict, never a bare number: `unmeasured` (nothing recorded),
    `insufficient` (too few paired trades to say), `beats_market`,
    `behind_market`, or `no_evidence` (the interval still contains zero — the
    honest reading of a small edge, and not a licence to deploy).
    """
    pairs = pairs_from_rows(rows)
    if not pairs:
        return SkillEvidence()
    if len(pairs) < int(min_pairs):
        return SkillEvidence(
            n=len(pairs),
            model_brier=brier(pairs, 0),
            market_brier=brier(pairs, 1),
            verdict=INSUFFICIENT,
            reason=(f"{len(pairs)} paired trade(s): fewer than {int(min_pairs)} "
                    f"is not enough to compare against the price"))

    model_b = brier(pairs, 0)
    market_b = brier(pairs, 1)
    improvements = _improvements(pairs)
    mean_improvement = sum(improvements) / len(improvements)
    lo, hi = bootstrap_ci(improvements, iterations=iterations)
    # The money test reads the SIDES, not the YES scale: what was bought, at
    # what it cost.
    hits, side_prices = entries_from_rows(rows if isinstance(rows, (list, tuple))
                                          else list(rows))
    if hits == 0 and not side_prices:
        hits = int(sum(1 for p in pairs if p[2] > 0.5))
        side_prices = [p[1] for p in pairs]
    p_value = poisson_binomial_tail(hits, side_prices)
    relative = (1.0 - model_b / market_b) if market_b > 0 else None

    evidence = SkillEvidence(
        n=len(pairs),
        model_brier=model_b,
        market_brier=market_b,
        skill=relative,
        improvement=mean_improvement,
        ci_low=lo,
        ci_high=hi,
        hits=hits,
        expected_hits=sum(side_prices),
        p_value=p_value,
    )
    if lo > 0.0:
        evidence.verdict = FORECAST_BEATS_PRICE
    elif hi < 0.0:
        evidence.verdict = FORECAST_BEHIND_PRICE
    else:
        evidence.verdict = NO_EVIDENCE

    forecast_part = {
        FORECAST_BEATS_PRICE: (
            f"forecast beats the price on {len(pairs)} trade(s): Brier "
            f"{model_b:.4f} vs market {market_b:.4f}, improvement "
            f"{mean_improvement:+.4f} per trade, 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]"),
        FORECAST_BEHIND_PRICE: (
            f"the PRICE beat the forecast on {len(pairs)} trade(s): Brier "
            f"{model_b:.4f} vs market {market_b:.4f}, improvement "
            f"{mean_improvement:+.4f} per trade, 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] - entirely negative"),
        NO_EVIDENCE: (
            f"no measurable edge over the price on {len(pairs)} trade(s): "
            f"improvement {mean_improvement:+.4f} per trade, 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] still contains zero"),
    }[evidence.verdict]
    entries_part = (
        f"entries: {hits} win(s) against {sum(side_prices):.1f} the prices "
        f"implied (p={p_value:.4f}) - "
        + ("clear the odds they paid" if evidence.entries_clear_the_odds
           else "do NOT clear the odds they paid"))
    evidence.reason = f"{forecast_part}; {entries_part}"
    return evidence


def recent_window(rows: Sequence[Any], size: int) -> List[Any]:
    """
    The newest `size` rows, in the order given (callers pass time order).

    Used for drift: a venue is qualified on the whole record, but live capital
    is at risk on the NEXT trade, so what matters is whether the edge is still
    there now. An edge that has gone is a reason to stop, not a reason to
    average it with the months it worked.
    """
    if size <= 0:
        return []
    return list(rows[-size:])


def drift(rows: Sequence[Any], *, window: int = 60,
          iterations: int = DEFAULT_ITERATIONS,
          min_pairs: int = MIN_PAIRED_TRADES) -> SkillEvidence:
    """
    The same question asked of the recent record only.

    `behind_market` here means the interval over the newest trades is entirely
    below zero: the agent is worse than the price, on evidence, recently. That
    is the condition a venue must lose its live qualification on — not "its
    all-time average dipped", which lets a dead edge sit in the average for
    months.
    """
    return market_skill(recent_window(rows, window), iterations=iterations,
                        min_pairs=min_pairs)
