"""
The bench: a rule that failed its own holdout must not authorise real money.

These tests are about a gap that existed between two modules that were each
correct on their own. `validation/walk_forward.py` worked out, out of sample,
which rules the record contradicts. `agent/v3_loop.py` sized live capital by
those same rules. Nothing joined the two, so the finding was logged and the
money moved anyway.

What is pinned here:

  * a refusal benched a rule for REAL money and only for real money;
  * the bench never becomes a second entry filter - an entry carried by a rule
    that still stands is untouched, and an entry no tested rule carries is not
    the bench's business at all;
  * `no_signal` benches nothing: "no filter beat taking every trade" is not
    evidence that a filter loses;
  * an absent or unreadable record benches nothing, while a recorded refusal
    benches every entry the refused rule alone would carry;
  * the two reasons a rule can be refused are the validator's two reasons,
    read through the validator's own function rather than re-implemented.
"""

from __future__ import annotations

import json

import pytest

from ptai.storage.db import Storage
from ptai.validation import rule_bench
from ptai.validation.walk_forward import (CONFIRMED_UNECONOMIC, INSUFFICIENT,
                                          NO_SIGNAL, UNCONFIRMED, VERDICTS_KEY,
                                          WalkForwardReport, record_verdict,
                                          refusals_from_results)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _storage(tmp_path) -> Storage:
    return Storage(db_path=str(tmp_path / "ptai.db"))


def _refused_result(name: str, *, significant=True, confirmed=False,
                    viable=False) -> dict:
    return {"name": name, "significant": significant, "confirmed": confirmed,
            "economically_viable": viable, "entries": 120, "holdout_entries": 40,
            "hit_rate": 0.82, "p_value": 0.00001}


def _store(results, verdict, scope="all", *, storage, summary="looked good, "
           "then did not hold on the fresh holdout") -> None:
    payload = {"available": True, "rows": 400, "folds": 5, "verdict": verdict,
               "summary": summary, "results": results,
               "refused": refusals_from_results(results, verdict),
               "may_qualify": False, "may_refuse": True}
    storage.set_state(VERDICTS_KEY, json.dumps({scope: payload}))


OFF = rule_bench.EntryClaim(forecast_prob=None, yes_price=None)


# --------------------------------------------------------------------------
# the claim, evaluated by the validator's own rules
# --------------------------------------------------------------------------

def test_an_8pct_edge_satisfies_the_8pct_rule_and_nothing_stronger():
    claim = rule_bench.EntryClaim(forecast_prob=0.59, yes_price=0.50)
    names = claim.satisfied()
    assert "edge_8pct" in names
    assert "edge_15pct" not in names
    # forecast 0.59 vs price 0.50 is 9 points clear, so not "confident" (10+).
    assert "confident" not in names
    assert "forecast_above_price" in names
    assert "all" not in names, "the baseline is not a rule an entry can cite"


def test_a_big_edge_is_carried_by_more_than_one_rule():
    claim = rule_bench.EntryClaim(forecast_prob=0.68, yes_price=0.50)
    names = claim.satisfied()
    assert {"edge_8pct", "edge_15pct", "forecast_above_price", "confident"} <= set(names)


def test_a_no_side_entry_is_judged_on_the_side_it_took():
    # Forecast 0.30 on NO at 0.50 is a 20-point edge on the NO token.
    claim = rule_bench.EntryClaim(forecast_prob=0.30, yes_price=0.50, side="NO")
    assert "edge_8pct" in claim.satisfied()
    assert "edge_15pct" in claim.satisfied()
    # ...and the YES-scale rules cannot vouch for it: "the forecast is above the
    # YES price" is an argument for the other side of this trade.
    assert "forecast_above_price" not in claim.satisfied()
    assert "confident" not in claim.satisfied()


def test_the_claim_comes_from_the_fields_the_trade_row_will_carry():
    class Opp:
        market_price = 0.50
        estimated_fair = 0.60
        side = "YES"

    claim = rule_bench.claim_of(Opp())
    assert (claim.forecast_prob, claim.yes_price, claim.side) == (0.60, 0.50, "YES")


def test_an_opportunity_missing_its_numbers_claims_nothing():
    class Bare:
        pass

    claim = rule_bench.claim_of(Bare())
    assert claim.forecast_prob is None and claim.yes_price is None


# --------------------------------------------------------------------------
# the bench, read from the record
# --------------------------------------------------------------------------

def test_an_absent_record_benches_nothing(tmp_path):
    storage = _storage(tmp_path)
    try:
        assert rule_bench.benched_rules(storage) == {}
    finally:
        storage.close()


def test_an_unreadable_record_benches_nothing_and_says_why(tmp_path):
    storage = _storage(tmp_path)
    try:
        storage.set_state(VERDICTS_KEY, "not json at all")
        assert rule_bench.benched_rules(storage) == {}
        block = rule_bench.validation_block(storage)
        assert block["available"] is False
        assert "unreadable" in block["reason"]
    finally:
        storage.close()


def test_a_significant_rule_that_failed_its_holdout_is_benched(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct"), _refused_result("all",
                                                              significant=False)],
               UNCONFIRMED, storage=storage)
        benched = rule_bench.benched_rules(storage)
        assert "edge_8pct" in benched
        assert "all" not in benched, "the baseline is never benched"
        assert benched["edge_8pct"]["scope"] == "all"
        assert "holdout" in benched["edge_8pct"]["why"]
    finally:
        storage.close()


def test_a_confirmed_but_uneconomic_rule_is_benched(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_15pct", significant=True, confirmed=True,
                                viable=False)],
               CONFIRMED_UNECONOMIC, storage=storage)
        assert "edge_15pct" in rule_bench.benched_rules(storage)
    finally:
        storage.close()


def test_no_signal_benches_nothing(tmp_path):
    storage = _storage(tmp_path)
    try:
        # Every rule looks quiet. "Nothing beat taking every trade" is a
        # statement about the filters, not evidence they lose money, and the
        # bench must not turn it into a stop.
        _store([_refused_result("edge_8pct", significant=False),
                _refused_result("confident", significant=False)],
               NO_SIGNAL, storage=storage)
        assert rule_bench.benched_rules(storage) == {}
    finally:
        storage.close()


def test_a_record_too_small_to_say_anything_benches_nothing(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([], INSUFFICIENT, storage=storage, summary="0 settled trades")
        assert rule_bench.benched_rules(storage) == {}
    finally:
        storage.close()


def test_a_verdict_recorded_before_the_refusal_list_existed_still_benches(tmp_path):
    # Payloads written by the previous version have no "refused" key. They must
    # answer to the same policy, through the validator's own function.
    storage = _storage(tmp_path)
    try:
        payload = {"available": True, "verdict": UNCONFIRMED, "summary": "s",
                   "results": [_refused_result("edge_8pct")]}
        storage.set_state(VERDICTS_KEY, json.dumps({"all": payload}))
        assert "edge_8pct" in rule_bench.benched_rules(storage)
    finally:
        storage.close()


def test_recording_one_scope_does_not_erase_another(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct")], UNCONFIRMED, scope="all",
               storage=storage)
        report = WalkForwardReport(rows=400, folds=5, verdict=NO_SIGNAL,
                                   summary="quiet", results=[])
        assert record_verdict(storage, report, scope="polymarket") is True
        scopes = json.loads(storage.get_state(VERDICTS_KEY))
        assert set(scopes) == {"all", "polymarket"}
    finally:
        storage.close()


# --------------------------------------------------------------------------
# the decision: live money only
# --------------------------------------------------------------------------

def test_a_weaker_rule_does_not_rescue_an_entry_from_a_refused_stronger_one():
    # The loophole this policy exists to close: a 9-point edge satisfies
    # edge_8pct AND forecast_above_price, and on the YES side the second is
    # contained by the first. Letting "forecast is above the price" vouch for
    # real money after the edge rule lost its holdout is how a bench stops
    # benching anything.
    benched = {"edge_8pct": {"why": "failed its holdout", "scope": "all"}}
    claim = rule_bench.EntryClaim(forecast_prob=0.59, yes_price=0.50)
    assert claim.strongest() == ["edge_8pct"]
    allowed, reason, carrying = rule_bench.bench_decision(claim, benched)
    assert allowed is False
    assert carrying == []
    assert "edge_8pct" in reason


def test_a_no_entry_rests_on_its_edge_rules_alone():
    # Same numbers, taken on the other side: the only rules left are the edge
    # rules, so a refused 8% floor benched the entry even though a YES-scale
    # rule would have been satisfied.
    benched = {"edge_15pct": {"why": "failed its holdout", "scope": "all"}}
    # Forecast 0.30 on a 0.50 YES price is a 20-point edge on the NO token; the
    # YES-scale rules are true here too (the forecast is well below the price)
    # and are excluded, so the entry stands on the edge floors alone.
    claim = rule_bench.EntryClaim(forecast_prob=0.30, yes_price=0.50, side="NO")
    assert claim.strongest() == ["edge_15pct"]
    allowed, _reason, carrying = rule_bench.bench_decision(claim, benched)
    assert allowed is False and carrying == []


def test_a_stronger_standing_rule_still_carries_the_entry():
    # 18 points clear: the entry stands on edge_15pct and on `confident`, and
    # neither has been refused, so the refused 8% floor is irrelevant to it.
    benched = {"edge_8pct": {"why": "failed its holdout", "scope": "all"}}
    claim = rule_bench.EntryClaim(forecast_prob=0.68, yes_price=0.50)
    assert set(claim.strongest()) == {"edge_15pct", "confident"}
    allowed, reason, carrying = rule_bench.bench_decision(claim, benched)
    assert allowed is True
    assert set(carrying) == {"edge_15pct", "confident"}


def test_the_containment_map_matches_the_rules_themselves():
    # The map is declared by hand, so it is checked against the predicates it
    # claims to describe, over every forecast/price pair on the YES side:
    # wherever a stronger rule takes a trade, every rule it contains takes that
    # same trade. If a threshold is ever changed in walk_forward.py without this
    # map, this fails rather than silently letting a weaker rule vouch for a
    # refused stronger one.
    from ptai.validation.walk_forward import default_candidates

    candidates = {c.name: c for c in default_candidates()}
    yes_rows = [
        walk_row(forecast=round(f / 100, 2), price=round(p / 100, 2), side="YES")
        for f in range(0, 101)
        for p in range(0, 101, 2)
    ]
    for strong, weak_names in rule_bench.CONTAINS.items():
        for weak in weak_names:
            for row in yes_rows:
                if candidates[strong].take(row):
                    assert candidates[weak].take(row), (
                        f"{strong} took a trade that {weak} did not: "
                        f"0.{int(row.forecast_prob*100)} vs 0.5 {row.side}")

    # Both edge floors are on the entry's own side, so that containment holds
    # whichever side was taken.
    no_rows = [
        walk_row(forecast=round(f / 100, 2), price=round(p / 100, 2), side="NO")
        for f in range(0, 101)
        for p in range(0, 101, 2)
    ]
    for row in no_rows:
        if candidates["edge_15pct"].take(row):
            assert candidates["edge_8pct"].take(row), row


def walk_row(forecast: float, price: float, side: str = "YES"):
    from ptai.validation.walk_forward import Row

    return Row(resolved_at="", pnl=0.0, amount_usd=0.0, side=side,
               forecast_prob=forecast, yes_price=price, actual_outcome=None)


def test_a_benched_rule_still_carries_an_entry_that_another_rule_backs():
    benched = {"edge_8pct": {"why": "failed its holdout", "scope": "all"}}
    claim = rule_bench.EntryClaim(forecast_prob=0.68, yes_price=0.50)
    allowed, reason, carrying = rule_bench.bench_decision(claim, benched)
    assert allowed is True
    assert "edge_15pct" in carrying
    assert "carried by" in reason


def test_an_entry_carried_only_by_a_benched_rule_is_refused(tmp_path):
    # 0.59 against 0.50 is a 9-point mispricing, so the only rules that can
    # vouch for this entry are edge_8pct and forecast_above_price. Bench both
    # and the entry has nothing left standing.
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct"),
                _refused_result("forecast_above_price")],
               UNCONFIRMED, storage=storage)
        benched = rule_bench.benched_rules(storage)

        class Opp:
            market_price = 0.50
            estimated_fair = 0.59
            side = "YES"

        allowed, reason = rule_bench.bench_live_authorisation(Opp(), benched)
        assert allowed is False
        assert "edge_8pct" in reason
        assert "refused out of sample" in reason
    finally:
        storage.close()


def test_the_bench_refuses_real_money_and_not_the_trade(tmp_path):
    # The refusal is expressed as a lane decision. The order still runs; what it
    # loses is the right to spend real money. Nothing here may touch the entry
    # filters - that is the property the loop relies on.
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct"),
                _refused_result("forecast_above_price"),
                _refused_result("confident")], UNCONFIRMED, storage=storage)
        benched = rule_bench.benched_rules(storage)

        class Opp:
            market_price = 0.50
            estimated_fair = 0.59
            side = "YES"

        allowed, reason = rule_bench.bench_live_authorisation(Opp(), benched)
        assert allowed is False
        # ...and with none of the rules benched, nothing is refused at all.
        allowed_again, _ = rule_bench.bench_live_authorisation(Opp(), {})
        assert allowed_again is True
    finally:
        storage.close()


def test_an_entry_no_tested_rule_carries_is_not_the_benchs_business():
    # e.g. a forecast below the price the agent is buying. Whether that entry
    # should exist is the entry gates' question; the bench only subtracts, and
    # inventing a refusal here would make it a second filter.
    claim = rule_bench.EntryClaim(forecast_prob=0.40, yes_price=0.50, side="YES")
    assert claim.strongest() == []
    allowed, reason, carrying = rule_bench.bench_decision(
        claim, {"edge_8pct": {"why": "x"}})
    assert allowed is True
    assert carrying == []
    assert "nothing for the bench to withdraw" in reason


def test_failing_to_evaluate_a_rule_does_not_bench_every_entry(monkeypatch):
    # A rule that raises when judging an entry does not get a vote. Benching is a
    # statement about evidence; a bug in one rule is not evidence.
    class Broken:
        name = "edge_8pct"
        description = "broken on purpose"

        def take(self, row):
            raise RuntimeError("boom")

    monkeypatch.setattr(rule_bench, "default_candidates", lambda: [Broken()])
    claim = rule_bench.EntryClaim(forecast_prob=0.59, yes_price=0.50)
    assert claim.satisfied() == []


# --------------------------------------------------------------------------
# the surfaces
# --------------------------------------------------------------------------

def test_the_validation_block_carries_both_the_verdict_and_the_bench(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct")], UNCONFIRMED, storage=storage)
        block = rule_bench.validation_block(storage)
        assert block["available"] is True
        assert "edge_8pct" in block["benched"]
        assert "benched from real money" in block["bench_line"]
        assert block["scopes"]["all"]["verdict"] == UNCONFIRMED
    finally:
        storage.close()


def test_the_block_says_plainly_when_nothing_is_benched(tmp_path):
    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct", significant=False)], NO_SIGNAL,
               storage=storage)
        block = rule_bench.validation_block(storage)
        assert block["benched"] == {}
        assert "nothing is benched" in block["bench_line"]
    finally:
        storage.close()


def test_operator_view_shows_the_bench(tmp_path):
    from ptai.operator_view import _validation

    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct")], UNCONFIRMED, storage=storage)
        block = _validation(storage)
        assert "edge_8pct" in block["benched"]
    finally:
        storage.close()


def test_the_loop_consults_the_bench_before_real_money():
    # Source-level, because the order path needs credentials and a funded venue
    # to run: the bench call must sit between the lane decision and the guard,
    # so no live order can be dispatched past a refused rule.
    from pathlib import Path

    source = Path("src/ptai/agent/v3_loop.py").read_text()
    flat = " ".join(source.split())
    assert "bench_live_authorisation" in flat
    assert "OUT-OF-SAMPLE BENCH refuses real money" in flat
    assert "self._cycle_benched = benched_rules(self.storage)" in flat
    # ...and the lane and the guard can both refuse: neither reason is lost.
    assert "Both are kept." in flat


def test_the_bench_never_touches_qualification(tmp_path):
    # Qualification is about settled money and stays that way. If the bench could
    # reach it, an out-of-sample verdict would be able to move the gate - the one
    # thing walk-forward is forbidden to do.
    from pathlib import Path

    gate = Path("src/ptai/venues/qualification.py").read_text()
    assert "rule_bench" not in gate and "validation" not in gate

    bench = Path("src/ptai/validation/rule_bench.py").read_text()
    # The word may appear in prose about the boundary; no code may cross it.
    for forbidden in ("venues.qualification", "qualification_engine",
                      "VenueQualificationEngine", "may_qualify"):
        assert forbidden not in bench, (
            f"the bench must not reach into qualification ({forbidden})")


def test_the_doctor_asks_the_bench(tmp_path):
    from ptai.doctor import PASS, WARN, _bench_check

    storage = _storage(tmp_path)
    db = storage.db_path
    storage.close()
    assert _bench_check(db).status == PASS

    storage = _storage(tmp_path)
    try:
        _store([_refused_result("edge_8pct")], UNCONFIRMED, storage=storage)
    finally:
        storage.close()
    check = _bench_check(db)
    assert check.status == WARN
    assert "edge_8pct" in check.detail
