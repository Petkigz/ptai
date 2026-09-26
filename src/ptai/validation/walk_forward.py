"""
Out-of-sample validation: does the record survive being judged on trades the
model never saw?

The qualification gate in `venues/qualification.py` judges a venue on its
realised outcomes. That is the right basis for qualifying it, and it has a hole
that no amount of care in the thresholds closes: **every one of those outcomes was
produced by a rule that was chosen while looking at them.** A threshold tuned
against the same trades it is measured on will fit noise, and a $50 account
cannot afford to discover that with real money.

This module is the sibling avt-bot project's `walk-forward.js`, rebuilt on PTAI's
own record. It splits the settled outcomes into consecutive folds:

    [ ...... discovery ...... ][ ... holdout ... ]
                 significant here?        does it still hold here?

and asks two separate questions of each candidate rule:

  1. **statistical** - does it beat the base rate on the discovery folds, with
     the interval, after correcting for having tested several rules at once
     (Holm-Bonferroni, family-wise error)?
  2. **economic** - does it clear the BREAK-EVEN the entries actually paid? A
     rule can be significantly better than random and still lose money.

The verdict is written next to the data and read by the agent, the console and
`ptai validate`.

WHAT THIS DELIBERATELY CANNOT DO
--------------------------------
It cannot qualify anything. A walk-forward verdict can REFUSE a rule - "no
out-of-sample edge on this record" - and it can never open the gate that lets
real capital move, because the standing rule in this project is that a backtest
figure must never trigger live capital. Historical replay is evidence about the
past; qualification stays on money that actually settled. Stating that in code
rather than in a comment is the point: `may_qualify()` below returns False
always, and the tests pin it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

# State key the agent and the console read. One key, one verdict.
VERDICTS_KEY = "validation.walk_forward"

# What a verdict can say. `signal_confirmed` is the strongest thing this module
# will ever claim, and it still only means "worth re-testing on fresh data".
NO_SIGNAL = "no_signal"
UNCONFIRMED = "unconfirmed"
CONFIRMED_UNECONOMIC = "confirmed_uneconomic"
CONFIRMED_ECONOMIC = "confirmed_economic"
INSUFFICIENT = "insufficient"

# A fold needs enough entries for a proportion to mean anything.
MIN_ENTRIES_PER_FOLD = 10
DEFAULT_FOLDS = 5
DEFAULT_HOLDOUT_FRACTION = 0.34
ALPHA = 0.05


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def norm_cdf(z: float) -> float:
    """Standard normal CDF. `math.erf` is exact enough here and has no table."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def one_sided_p(z: float) -> float:
    """P(observing something at least this good) - for "is it ABOVE the bar"."""
    return 1.0 - norm_cdf(z)


def holm_bonferroni(p_values: Sequence[Optional[float]],
                    alpha: float = ALPHA) -> List[bool]:
    """
    Which hypotheses survive, controlling the family-wise error rate.

    Testing five rules and reporting the best one is how a research process
    manufactures a discovery out of noise: at alpha 0.05, one in twenty random
    rules "works". This is the correction for having looked more than once.
    Null p-values (not enough data to test) stay False - untested is not passed.
    """
    tested = [(p, i) for i, p in enumerate(p_values) if p is not None]
    keep = [False] * len(p_values)
    if not tested:
        return keep
    tested.sort()
    m = len(tested)
    for rank, (p, index) in enumerate(tested):
        if p <= alpha / (m - rank):
            keep[index] = True
        else:
            # Step-down: once one fails, every weaker one fails too.
            break
    return keep


@dataclass
class Row:
    """One settled outcome, as the validator needs it."""

    resolved_at: str
    pnl: float
    amount_usd: float
    side: str
    forecast_prob: Optional[float]
    yes_price: Optional[float]
    actual_outcome: Optional[float]
    venue_id: str = ""
    strategy: str = ""
    category: str = ""
    execution_mode: str = ""

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class Candidate:
    """A rule, expressed as "take this trade or not" - nothing more."""

    name: str
    description: str
    take: Callable[[Row], bool]
    # There is deliberately no payout field: for a binary contract the price paid
    # IS the break-even, and it comes from the recorded row (`break_even_for`)
    # rather than from anything a rule declares about itself. A rule that could
    # describe its own payoff could describe a flattering one.


def default_candidates() -> List[Candidate]:
    """
    The rules that are actually choosing trades today.

    These are not research ideas - each one is a gate already in the loop, so
    "does it help out of sample" is a question about the shipping system.
    """
    return [
        Candidate("all", "every resolved trade (the baseline)",
                  lambda r: True),
        Candidate("edge_8pct", "entered with at least an 8% modelled edge",
                  lambda r: _edge(r) is not None and _edge(r) >= 0.08),
        Candidate("edge_15pct", "entered with at least a 15% modelled edge",
                  lambda r: _edge(r) is not None and _edge(r) >= 0.15),
        Candidate("forecast_above_price",
                  "the forecast was above the price paid (a real mispricing)",
                  lambda r: (r.forecast_prob is not None and r.yes_price is not None
                             and _forecast_yes(r) > r.yes_price)),
        Candidate("confident",
                  "the forecast was at least 10 points clear of the price",
                  lambda r: (r.forecast_prob is not None and r.yes_price is not None
                             and _forecast_yes(r) - r.yes_price >= 0.10)),
    ]


def _edge(row: Row) -> Optional[float]:
    """Modelled edge, recomputed from the recorded pair rather than trusted."""
    if row.forecast_prob is None or row.yes_price is None:
        return None
    if str(row.side).upper() == "NO":
        return (1.0 - row.forecast_prob) - (1.0 - row.yes_price)
    return row.forecast_prob - row.yes_price


def _forecast_yes(row: Row) -> float:
    """The forecast expressed as P(YES), whichever side was taken."""
    prob = float(row.forecast_prob or 0.0)
    return prob if str(row.side).upper() != "NO" else 1.0 - prob


def rows_from_storage(storage, *, venue_id: Optional[str] = None,
                      strategy: Optional[str] = None,
                      execution_mode: Optional[str] = None,
                      resolved_only: bool = True) -> List[Row]:
    """
    The settled record, oldest first.

    Order matters: a walk-forward split is only out-of-sample if the folds are
    consecutive in TIME. Sorting by anything else - P&L, venue - would let the
    holdout see the future.
    """
    if storage is None:
        return []
    clauses: List[str] = []
    params: List[Any] = []
    if resolved_only:
        clauses.append("resolved_at IS NOT NULL")
    if venue_id:
        clauses.append("venue_id = ?")
        params.append(venue_id)
    if strategy:
        clauses.append("strategy = ?")
        params.append(strategy)
    if execution_mode:
        clauses.append("execution_mode = ?")
        params.append(execution_mode)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = ("SELECT resolved_at, COALESCE(pnl, 0.0), COALESCE(amount_usd, 0.0), "
           "COALESCE(side, ''), forecast_prob, yes_price, actual_outcome, "
           "COALESCE(venue_id, ''), COALESCE(strategy, ''), "
           "COALESCE(category, ''), COALESCE(execution_mode, '') "
           "FROM trade_outcomes" + where + " ORDER BY resolved_at ASC")
    try:
        cursor = storage.conn.execute(sql, params)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Walk-forward could not read the outcome log: "
                     f"{type(e).__name__}: {e}")
        return []
    return [Row(*row) for row in cursor.fetchall()]


@dataclass
class FoldResult:
    index: int
    entries: int
    wins: int
    pnl: float
    held_out: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"fold": self.index, "entries": self.entries, "wins": self.wins,
                "pnl": _round(self.pnl, 2), "held_out": self.held_out}


@dataclass
class CandidateVerdict:
    name: str
    description: str
    entries: int = 0
    wins: int = 0
    hit_rate: Optional[float] = None
    lift: Optional[float] = None
    p_value: Optional[float] = None
    significant: bool = False
    holdout_entries: int = 0
    holdout_hit_rate: Optional[float] = None
    holdout_lift: Optional[float] = None
    confirmed: bool = False
    break_even: Optional[float] = None
    economic_p_value: Optional[float] = None
    economically_viable: bool = False
    pnl: float = 0.0
    pnl_per_entry: Optional[float] = None
    max_drawdown_usd: float = 0.0
    folds: List[FoldResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "entries": self.entries, "wins": self.wins,
            "hit_rate": _round(self.hit_rate),
            "lift": _round(self.lift),
            "p_value": _round(self.p_value, 6),
            "significant": self.significant,
            "holdout_entries": self.holdout_entries,
            "holdout_hit_rate": _round(self.holdout_hit_rate),
            "holdout_lift": _round(self.holdout_lift),
            "confirmed": self.confirmed,
            "break_even": _round(self.break_even),
            "economic_p_value": _round(self.economic_p_value, 6),
            "economically_viable": self.economically_viable,
            "pnl": _round(self.pnl, 2),
            "pnl_per_entry": _round(self.pnl_per_entry, 4),
            "max_drawdown_usd": _round(self.max_drawdown_usd, 2),
            "folds": [f.to_dict() for f in self.folds],
        }


@dataclass
class WalkForwardReport:
    rows: int = 0
    folds: int = 0
    holdout_from_fold: int = 0
    verdict: str = INSUFFICIENT
    summary: str = ""
    base_rate: Optional[float] = None
    holdout_base_rate: Optional[float] = None
    results: List[CandidateVerdict] = field(default_factory=list)
    generated_at: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.rows > 0,
            "rows": self.rows, "folds": self.folds,
            "holdout_from_fold": self.holdout_from_fold,
            "verdict": self.verdict, "summary": self.summary,
            "base_rate": _round(self.base_rate),
            "holdout_base_rate": _round(self.holdout_base_rate),
            "results": [r.to_dict() for r in self.results],
            "generated_at": self.generated_at, "scope": self.scope,
            "may_qualify": False,
            "may_refuse": True,
            "rule": ("out-of-sample validation can refuse a strategy and can "
                     "never qualify one: qualification stays on settled money"),
        }

    @property
    def refuses(self) -> bool:
        """Are any rules contradicted by this record?"""
        return bool(self.refused_names())

    def refused_names(self) -> List[str]:
        """
        The rules this record says NOT to size capital by.

        Only two things earn a refusal, and "the record is quiet" is not one of
        them:

          * a rule that looked good on the discovery folds and FAILED the fresh
            holdout - the signature of a false positive, and the one case where
            acting on it is an error the record already caught; and
          * a rule with a real lift whose entries still lose money at the prices
            they paid.

        `no_signal` - nothing beat simply taking every trade - is a statement
        about the FILTERS, not a claim that they are harmful, and it is not a
        reason to stop: an agent whose filters add nothing still has to trade the
        opportunities its other gates approve. Saying "refuse everything" there
        would be the same overstatement in the opposite direction.
        """
        if self.verdict == UNCONFIRMED:
            return [r.name for r in self.results
                    if r.name != "all" and r.significant and not r.confirmed]
        if self.verdict == CONFIRMED_UNECONOMIC:
            return [r.name for r in self.results
                    if r.name != "all" and r.confirmed
                    and not r.economically_viable]
        return []


def may_qualify(_report: Optional[WalkForwardReport] = None) -> bool:
    """
    Can a walk-forward result qualify a venue for real capital?

    No. Never. A backtest figure must not trigger live capital - that is a
    standing rule of this project, and the honest spelling of it is a function
    that always answers no, rather than a comment somebody can miss.
    """
    return False


def _max_drawdown(pnls: Iterable[float]) -> float:
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def run_walk_forward(rows: Sequence[Row], *,
                     candidates: Optional[Sequence[Candidate]] = None,
                     folds: int = DEFAULT_FOLDS,
                     holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
                     alpha: float = ALPHA,
                     min_entries_per_fold: int = MIN_ENTRIES_PER_FOLD,
                     scope: Optional[Dict[str, Any]] = None) -> WalkForwardReport:
    """
    Slide a discovery window forward and score every rule on the next fold.

    Each fold: the rule is evaluated on trades up to the fold boundary, then
    every trade IN the fold is checked against the rule - with the rule's own
    per-fold hit rate compared with that fold's unconditional rate, so a fold
    where everything won is not mistaken for a good rule.
    """
    candidates = list(candidates or default_candidates())
    ordered = sorted(rows, key=lambda r: str(r.resolved_at or ""))
    report = WalkForwardReport(
        rows=len(ordered), scope=dict(scope or {}),
        generated_at=datetime.now(timezone.utc).isoformat())

    if len(ordered) < min_entries_per_fold * folds:
        report.summary = (
            f"{len(ordered)} settled trade(s) is not enough for out-of-sample "
            f"validation: it needs {min_entries_per_fold * folds}+ so that each "
            f"fold can fail on its own. Nothing is claimed about the strategy "
            f"until then.")
        return report

    size = len(ordered) // folds
    windows: List[Tuple[int, int]] = []
    for i in range(folds):
        start = i * size
        end = len(ordered) if i == folds - 1 else (i + 1) * size
        windows.append((start, end))

    holdout_from = max(1, int(round(folds * (1.0 - holdout_fraction))))
    report.holdout_from_fold = holdout_from
    report.folds = folds

    verdicts: List[CandidateVerdict] = []
    for candidate in candidates:
        verdict = CandidateVerdict(name=candidate.name,
                                   description=candidate.description)
        discovery_entries = 0
        discovery_wins = 0
        holdout_entries = 0
        holdout_wins = 0
        all_pnls: List[float] = []
        break_evens: List[float] = []

        for index, (start, end) in enumerate(windows):
            held_out = index >= holdout_from
            fold_entries = 0
            fold_wins = 0
            fold_pnl = 0.0
            for row in ordered[start:end]:
                if not candidate.take(row):
                    continue
                fold_entries += 1
                if row.won:
                    fold_wins += 1
                fold_pnl += float(row.pnl or 0.0)
                all_pnls.append(float(row.pnl or 0.0))
                price = break_even_for(row)
                if price is not None:
                    break_evens.append(price)
                if held_out:
                    holdout_entries += 1
                    if row.won:
                        holdout_wins += 1
                else:
                    discovery_entries += 1
                    if row.won:
                        discovery_wins += 1
            verdict.folds.append(FoldResult(
                index=index, entries=fold_entries, wins=fold_wins,
                pnl=fold_pnl, held_out=held_out))

        # The unconditional rate over the fold entries this rule actually saw.
        # Comparing against the whole record's win rate would let a rule look
        # clever merely for trading in a stretch that was easy for everything.
        verdict.entries = discovery_entries + holdout_entries
        verdict.wins = discovery_wins + holdout_wins
        verdict.hit_rate = (verdict.wins / verdict.entries) if verdict.entries else None
        base_rate = _base_rate(ordered)
        verdict.lift = (None if verdict.hit_rate is None or base_rate is None
                        else verdict.hit_rate - base_rate)
        verdict.pnl = sum(all_pnls)
        verdict.pnl_per_entry = (verdict.pnl / verdict.entries
                                 if verdict.entries else None)
        verdict.max_drawdown_usd = _max_drawdown(all_pnls)

        if base_rate is not None and discovery_entries >= min_entries_per_fold:
            discovery_rate = discovery_wins / discovery_entries
            se = math.sqrt(max(base_rate * (1.0 - base_rate) / discovery_entries,
                               1e-12))
            verdict.p_value = two_sided_p((discovery_rate - base_rate) / se)

        # Break-even from the prices the entries actually paid. A binary contract
        # bought at p pays 1 per share, so it breaks even at a hit rate of exactly
        # p - which makes the average price paid the bar the hit rate has to clear,
        # and it is measured from the recorded price, not assumed.
        if break_evens:
            verdict.break_even = sum(break_evens) / len(break_evens)
        if (verdict.break_even is not None
                and verdict.break_even not in (0.0, 1.0)
                and discovery_entries >= min_entries_per_fold):
            discovery_rate = discovery_wins / discovery_entries
            be = verdict.break_even
            se_be = math.sqrt(max(be * (1.0 - be) / discovery_entries, 1e-12))
            verdict.economic_p_value = one_sided_p((discovery_rate - be) / se_be)
            verdict.economically_viable = bool(
                verdict.economic_p_value < alpha and discovery_rate > be)

        verdict.holdout_entries = holdout_entries
        verdict.holdout_hit_rate = (holdout_wins / holdout_entries
                                    if holdout_entries else None)
        if holdout_entries and verdict.holdout_hit_rate is not None and base_rate is not None:
            verdict.holdout_lift = verdict.holdout_hit_rate - base_rate
        verdicts.append(verdict)

    corrected = holm_bonferroni([v.p_value for v in verdicts], alpha=alpha)
    for verdict, survives in zip(verdicts, corrected):
        verdict.significant = bool(survives and (verdict.lift or 0.0) > 0)
        verdict.confirmed = bool(
            verdict.significant
            and verdict.holdout_entries >= min_entries_per_fold
            and (verdict.holdout_lift or 0.0) > 0)

    report.results = verdicts
    report.base_rate = _base_rate(ordered)
    report.holdout_base_rate = _base_rate(
        [row for i, (s, e) in enumerate(windows) if i >= holdout_from
         for row in ordered[s:e]])

    any_confirmed = any(v.confirmed for v in verdicts)
    any_significant = any(v.significant for v in verdicts)
    any_economic = any(v.economically_viable for v in verdicts if v.confirmed)
    rule_names = ", ".join(v.name for v in verdicts if v.name != "all")

    if any_confirmed and any_economic:
        report.verdict = CONFIRMED_ECONOMIC
        report.summary = (
            f"a rule ({rule_names}) beat the base rate on the discovery folds, "
            f"survived the untouched holdout, and clears the break-even its "
            f"entries paid. This is a candidate worth re-testing on fresh data - "
            f"it does NOT qualify anything: only settled money can do that.")
    elif any_confirmed:
        report.verdict = CONFIRMED_UNECONOMIC
        report.summary = (
            f"a rule beat the base rate and survived the holdout, but the hit "
            f"rate does not clear the break-even the entries paid, so betting it "
            f"still loses money. The gate stays shut.")
    elif any_significant:
        report.verdict = UNCONFIRMED
        report.summary = (
            f"a rule looked significant on the discovery folds and did not hold "
            f"on the fresh holdout - the signature of a false positive. Treat it "
            f"as noise and keep the gate shut.")
    else:
        report.verdict = NO_SIGNAL
        report.summary = (
            f"no rule beat the base rate out of sample on {len(ordered)} settled "
            f"trade(s), after correcting for testing {len(verdicts)} at once. "
            f"There is no out-of-sample edge in this record to act on; the "
            f"discipline runs without one.")

    if len(ordered) < min_entries_per_fold * folds * 2:
        report.summary += (" The record is still small, so treat this as a first "
                           "reading rather than a settled answer.")
    return report


def break_even_for(row: Row) -> Optional[float]:
    """
    The hit rate this entry had to beat to be worth taking.

    For a binary contract bought at a price p, a win pays 1 per share, so the
    break-even probability IS the price paid. For a NO trade the price paid is
    `1 - yes_price` - the YES price is what the record stores, and the side is
    what says which token was bought. None when the price was not recorded: an
    unknown break-even is not a break-even of zero.
    """
    if row.yes_price is None:
        return None
    price = float(row.yes_price)
    if str(row.side).upper() == "NO":
        price = 1.0 - price
    return price if 0.0 < price < 1.0 else None


def _base_rate(ordered: Sequence[Row]) -> Optional[float]:
    """
    The unconditional win rate of the record - the honest null.

    A rule is only worth something if it beats simply taking every trade. If it
    cannot, the rule is not selecting anything; it is adding a label.
    """
    if not ordered:
        return None
    wins = sum(1 for row in ordered if row.won)
    return wins / len(ordered)


def verdict_path(data_dir: Any, scope: str = "all") -> Any:
    from pathlib import Path

    safe = "".join(c if (c.isalnum() or c in "-._") else "-" for c in str(scope))
    return Path(data_dir) / f"validation-{safe}.json"


def record_verdict(storage, report: WalkForwardReport,
                   scope: str = "all") -> bool:
    """Persist the verdict for the console and the agent to read."""
    if storage is None:
        return False
    try:
        payload = report.to_dict()
        payload["scope_name"] = scope
        storage.set_state(VERDICTS_KEY,
                          json.dumps({scope: payload}))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not record the walk-forward verdict: "
                       f"{type(e).__name__}: {e}")
        return False


def load_verdicts(storage) -> Dict[str, Any]:
    """Every recorded verdict, or an explicit "not run yet"."""
    block: Dict[str, Any] = {"available": False, "scopes": {},
                             "source": f"state key {VERDICTS_KEY}"}
    if storage is None:
        block["reason"] = "no storage"
        return block
    try:
        raw = storage.get_state(VERDICTS_KEY)
    except Exception as e:  # noqa: BLE001
        block["reason"] = f"{type(e).__name__}: {e}"
        return block
    if not raw:
        block["reason"] = ("out-of-sample validation has not been run on this "
                           "record yet - run `python main.py validate`")
        return block
    try:
        scopes = json.loads(raw)
    except (TypeError, ValueError) as e:
        block["reason"] = f"unreadable verdict ({e})"
        return block
    block["scopes"] = scopes if isinstance(scopes, dict) else {}
    block["available"] = bool(block["scopes"])
    if not block["available"]:
        block["reason"] = "no verdict recorded yet"
    return block


def verdict_line(verdicts: Dict[str, Any]) -> str:
    """One sentence for the CLI and the panel."""
    if not verdicts.get("available"):
        return f"out-of-sample: {verdicts.get('reason') or 'not run'}"
    scopes = verdicts.get("scopes") or {}
    parts = []
    for name, payload in scopes.items():
        parts.append(f"{name}: {payload.get('verdict')} ({payload.get('rows')} rows)")
    return "out-of-sample validation - " + "; ".join(parts)
