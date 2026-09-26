"""
The bench: a rule that failed its own holdout does not authorise real money.

`validation/walk_forward.py` works out, out of sample, which of the rules that
are choosing trades today are contradicted by the record. Until this module,
nothing but a log line read that answer: the agent recorded "edge_8pct failed
its holdout" and went on sizing live capital by the 8% edge floor, because a
warning is not a policy.

The sibling avt-bot project settles this at the pattern level: a pattern that
stops working is BENCHED - taken out of the decision - rather than argued with,
and its signal lifecycle walks a candidate from shadow to micro to drifting to
retired. That is the piece PTAI was missing, and it is the right shape here too,
with one difference that matters:

    a refused rule does not become an inverted rule or a wider filter.

The bench removes AUTHORISATION, it never changes the entry filters. An entry
still has to clear every gate it cleared before - the 8% floor, the opportunity
score, the risk chain - and it runs in paper exactly as before. What it loses is
the right to spend real money while the only rule that can vouch for it has been
refused out of sample. Paper keeps learning; live stops being the thing that
pays for the lesson.

So the decision is per ENTRY, not per rule:

  * an entry is carried by the STRONGEST rule it satisfies - the same rule
    objects the validator tested, evaluated on the entry's own recorded inputs
    (forecast, price paid, side). A weaker rule that the stronger one contains
    carries nothing: an entry with a 16% edge is also above the 8% floor and
    above the price, and quoting the 8% floor for it would let the weakest claim
    on the sheet vouch for money the strongest claim has lost;
  * if at least one of those strongest rules still stands, the entry may risk
    real money and the reason names the rule that carried it;
  * if every one of them has been refused, the entry is benched from real money
    and the reason names each refusal.

An entry no tested rule carries is not this module's business: the bench only
subtracts, and the entry gates own what gets in.

Why subtract rather than require: the bench refuses an entry only when the record
CONTRADICTS every rule that would justify it, and never demands that a rule have
been affirmatively confirmed out of sample. Making confirmation a condition of
live money would turn this module into the gate that walks the walk-forward
result back in through the side door - and a small, quiet record would stop an
unattended agent from trading at all, which is the same overstatement in the
opposite direction. Qualification stays on settled money; the bench only takes
backwhat the record has disproved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .walk_forward import (Row, default_candidates, load_verdicts,
                           refusals_from_results)

# The rule every entry satisfies by definition is not a filter and cannot carry
# an entry on its own, so it is not consulted here either.
BASELINE_RULE = "all"

# Rules that describe the YES side of a market: "the forecast is above the price
# paid", "the forecast is ten points clear of it". For a NO entry neither is a
# reason to own the NO token - a forecast above the YES price is evidence for
# the OTHER side - so they cannot carry a NO entry, and taking them at face value
# would let a rule that argues against the trade vouch for it.
YES_SCALE_RULES = ("forecast_above_price", "confident")

# Which rules a rule CONTAINS. An entry clearing the 15% floor has also cleared
# the 8% floor, and a YES entry clearing either has also cleared "forecast above
# the price", so once the stronger rule is refused, quoting the weaker one for
# that same entry is quoting the half of the claim that has already been
# contradicted: a rule with no demonstrated edge would be vouching for real
# money. Declared here, verified against the predicates in the tests over every
# forecast/price pair, so it cannot drift away from what the rules do.
CONTAINS = {
    "edge_15pct": ("edge_8pct", "forecast_above_price"),
    "edge_8pct": ("forecast_above_price",),
    "confident": ("forecast_above_price",),
    "forecast_above_price": (),
}


@dataclass(frozen=True)
class EntryClaim:
    """
    What an entry says about itself, in the same terms the record stores.

    The fields are named after `trade_outcomes` on purpose: the claim is built
    from the values the trade will be RECORDED with, so the bench judges the
    entry the validator will later read, not a parallel description of it.
    """

    forecast_prob: Optional[float]
    yes_price: Optional[float]
    side: str = "YES"

    def satisfied(self) -> List[str]:
        """Names of the tested rules this entry satisfies, cheapest rule first."""
        row = Row(resolved_at="", pnl=0.0, amount_usd=0.0,
                  side=self.side or "YES",
                  forecast_prob=self.forecast_prob,
                  yes_price=self.yes_price,
                  actual_outcome=None)
        side = str(self.side or "YES").upper()
        names: List[str] = []
        for candidate in default_candidates():
            if candidate.name == BASELINE_RULE:
                continue
            if side == "NO" and candidate.name in YES_SCALE_RULES:
                continue
            try:
                if candidate.take(row):
                    names.append(candidate.name)
            except Exception as e:  # noqa: BLE001
                # A rule that cannot be evaluated against an entry does not get
                # a vote on it. Failing "taken" would let a bug in one rule
                # bench every entry; failing "not taken" keeps the bench honest
                # about what it actually knows.
                logger.debug(f"Rule {candidate.name} could not judge the entry "
                             f"({type(e).__name__}: {e}) - not counting it")
        return names

    def strongest(self) -> List[str]:
        """
        The rules this entry can stand on: the ones no other satisfied rule
        contains.

        These are what live money answers to. A rule that is merely implied by a
        stronger one is not a separate reason to believe the entry is good, and
        it must not become a loophole through which a refused rule still gets to
        size a real position.
        """
        satisfied = self.satisfied()
        return [name for name in satisfied
                if not any(name in CONTAINS.get(other, ())
                           for other in satisfied if other != name)]


def claim_of(opportunity) -> EntryClaim:
    """The claim an opportunity makes, from the fields its trade row will carry."""
    try:
        yes_price = getattr(opportunity, "market_price", None)
        return EntryClaim(
            forecast_prob=getattr(opportunity, "estimated_fair", None),
            yes_price=float(yes_price) if yes_price is not None else None,
            side=str(getattr(opportunity, "side", "YES") or "YES"))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not build the entry claim: {type(e).__name__}: {e}")
        return EntryClaim(forecast_prob=None, yes_price=None)


def benched_rules(storage) -> Dict[str, Dict[str, Any]]:
    """
    Rules the recorded verdicts refuse, and why.

    Reads the persisted verdicts - it never recomputes them - because the bench
    has to act on the same answer the console and the operator are shown. An
    unreadable or absent record benches NOTHING: a refusal is a claim about a
    record, and with no record there is no claim to make. (Storage that cannot
    be read at all is the money guard's problem, and that one fails closed for
    real money on its own.)
    """
    benched: Dict[str, Dict[str, Any]] = {}
    try:
        verdicts = load_verdicts(storage)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read the out-of-sample verdicts: "
                       f"{type(e).__name__}: {e}")
        return benched
    for scope, payload in (verdicts.get("scopes") or {}).items():
        if not isinstance(payload, dict):
            continue
        # A payload recorded before the refusal list was written into it still
        # answers to the documented policy, evaluated by the validator's own
        # function rather than a copy of it. Verdicts recorded today carry the
        # list outright.
        refused = payload.get("refused")
        if refused is None:
            refused = refusals_from_results(payload.get("results") or [],
                                            payload.get("verdict"))
        for name in refused or []:
            if not name or name == BASELINE_RULE:
                continue
            benched[str(name)] = {
                "scope": scope,
                "verdict": payload.get("verdict"),
                "why": payload.get("summary") or "refused out of sample",
            }
    return benched


def bench_decision(claim: EntryClaim,
                   benched: Dict[str, Dict[str, Any]]) -> Tuple[bool, str, List[str]]:
    """
    May this entry risk REAL money? -> (allowed, reason, carrying rules).

    The question is asked of the STRONGEST claims the entry can make, not of
    every test it happens to pass: a 16% edge also passes an 8% floor, and
    letting the floor vouch for it after the floor was refused is how a bench
    would quietly stop benching anything.

    Fail-open on an empty bench is deliberate and is not the same as failing
    open on a bad record: nothing is benched until a verdict says so, and a
    verdict only refuses a rule for one of the two reasons the validator will
    name. Fail-open on an UNREADABLE verdict store is deliberate too, for the
    reason in `benched_rules`.
    """
    strongest = claim.strongest()
    if not strongest:
        return (True,
                "no tested rule is the strongest claim of this entry, so there "
                "is nothing for the bench to withdraw", [])
    standing = [name for name in strongest if name not in benched]
    if standing:
        return (True, f"carried by {', '.join(standing)}", standing)
    reasons = "; ".join(
        f"{name} ({benched[name].get('why')})" for name in strongest)
    return (False,
            f"every rule that would justify this entry has been refused out of "
            f"sample: {reasons}", [])


def bench_live_authorisation(opportunity, benched: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    """Convenience wrapper for the loop: (may spend real money, why)."""
    allowed, reason, _carrying = bench_decision(claim_of(opportunity), benched)
    return allowed, reason


def validation_block(storage) -> Dict[str, Any]:
    """
    The verdicts plus the bench, in one payload for the operator surfaces.

    Both readers (console, `ptai status`) showed the verdicts and neither showed
    what the agent DOES about them, which left the operator to infer that a
    refused rule was actually stopped. It is now stated.
    """
    try:
        block = dict(load_verdicts(storage))
    except Exception as e:  # noqa: BLE001
        return {"available": False, "scopes": {}, "benched": {},
                "reason": f"{type(e).__name__}: {e}"}
    block["benched"] = benched_rules(storage)
    if block["benched"]:
        names = ", ".join(sorted(block["benched"]))
        block["bench_line"] = (
            f"benched from real money out of sample: {names} - entries they "
            f"alone carry still run in paper")
    else:
        block["bench_line"] = ("nothing is benched: no rule has been refused "
                               "out of sample on this record")
    return block
