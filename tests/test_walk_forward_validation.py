"""
Out-of-sample validation, and the line it must never cross.

The qualification gate judges a venue on realised outcomes - every one of them
produced by rules chosen while looking at them. That is not a flaw in the gate's
thresholds; it is a hole no threshold closes. Five rules tested against the same
record, and the best one reported, is how a research process manufactures a
discovery out of noise: at alpha 0.05, one in twenty random rules "works".

So this harness borrows the sibling avt-bot project's walk-forward design:

  * consecutive folds, with the newest third untouched during discovery,
  * Holm-Bonferroni correction for having tested several rules at once,
  * statistical lift and economic viability tested SEPARATELY (a rule can be
    significantly better than random and still lose money at the prices it paid),
  * and the honest answer when nothing survives - "no out-of-sample edge" -
    printed as a verdict rather than buried.

The line it must never cross, pinned in code as well as here: a walk-forward
verdict can REFUSE a rule and can never qualify a venue, because a backtest figure
must never trigger live capital.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.storage.db import Storage
from src.ptai.validation.walk_forward import (
    CONFIRMED_ECONOMIC,
    CONFIRMED_UNECONOMIC,
    INSUFFICIENT,
    NO_SIGNAL,
    UNCONFIRMED,
    Candidate,
    Row,
    break_even_for,
    holm_bonferroni,
    load_verdicts,
    may_qualify,
    one_sided_p,
    record_verdict,
    rows_from_storage,
    run_walk_forward,
    two_sided_p,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
PRICE = 0.50


def _row(i: int, prob: float, won: bool, *, side: str = "YES",
         price: float = PRICE) -> Row:
    return Row(
        resolved_at=(BASE + timedelta(days=i)).isoformat(),
        pnl=(1.0 / price - 1.0) if won else -1.0,
        amount_usd=1.0, side=side, forecast_prob=prob, yes_price=price,
        actual_outcome=1.0 if won else 0.0,
    )


def _clean(n: int = 400, *, seed: int = 3, big_until: int | None = None,
           real: bool = True) -> list:
    """
    A record where half the trades carry a large edge and half a small one.

    The selection effect is strong on purpose: a validator that cannot DETECT an
    edge is no better than one that never refuses, and both failure modes are
    tested below. `big_until` stops the edge at a row index, which is how a
    strategy that stopped working is modelled.
    """
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        big = (i % 2 == 1) and (big_until is None or i < big_until)
        edge = (0.35 if big else 0.05) if real else 0.20
        prob = 0.5 + edge
        rows.append(_row(i, prob, rnd.random() < prob))
    return rows


def _noise(n: int = 400, *, seed: int = 11) -> list:
    """A record where the forecast has no relationship with the outcome."""
    rnd = random.Random(seed)
    return [_row(i, 0.5 + rnd.random() * 0.35, rnd.random() < 0.5)
            for i in range(n)]


# ---------------------------------------------------------------------------
# 1. the statistics underneath
# ---------------------------------------------------------------------------

class TestTheCorrectionForLookingMoreThanOnce:
    def test_a_single_strong_result_survives(self):
        assert holm_bonferroni([0.0001, 0.4, 0.6, None]) == [True, False,
                                                             False, False]

    def test_four_identical_marginal_results_do_not(self):
        """
        The heart of it. Four rules each "significant" at p = 0.02 would each pass
        on their own; tested together, none does. This is the correction that
        stops a research process finding a winner in noise.
        """
        assert holm_bonferroni([0.02, 0.02, 0.02, 0.02]) == [False] * 4

    def test_an_untested_rule_is_not_a_passed_rule(self):
        assert holm_bonferroni([None, None]) == [False, False]

    def test_the_correction_is_step_down(self):
        """
        The smallest p is compared against alpha/m, the next against alpha/(m-1),
        and so on - and the walk stops at the first failure, because once a
        hypothesis fails at its own threshold every weaker one fails too.
        """
        # 0.001 <= 0.05/3; 0.02 <= 0.05/2; 0.9 > 0.05 - two survive.
        assert holm_bonferroni([0.001, 0.02, 0.9]) == [True, True, False]
        # 0.03 misses 0.05/2 = 0.025, and the step-down stops there.
        assert holm_bonferroni([0.001, 0.03, 0.9]) == [True, False, False]
        # Order does not matter: the p-values are ranked, not read in place.
        assert holm_bonferroni([0.9, 0.001, 0.02]) == [False, True, True]

    def test_the_tail_probabilities_agree_with_known_values(self):
        assert two_sided_p(0.0) == pytest.approx(1.0)
        assert two_sided_p(1.96) == pytest.approx(0.05, abs=0.001)
        assert one_sided_p(1.645) == pytest.approx(0.05, abs=0.001)
        assert one_sided_p(0.0) == pytest.approx(0.5)


class TestTheBreakEvenIsThePricePaid:
    def test_a_yes_entry_breaks_even_at_its_price(self):
        row = _row(0, 0.6, True, price=0.30)
        assert break_even_for(row) == pytest.approx(0.30)

    def test_a_no_entry_breaks_even_at_the_other_token_s_price(self):
        """
        The record stores the YES price. A NO trade bought at 0.70 of the NO
        token breaks even at 0.70 - reading the YES price here would ask the NO
        side to clear a bar set for the other token.
        """
        row = Row("2026-01-01T00:00:00+00:00", 1.0, 1.0, "NO", 0.25, 0.30, 1.0)
        assert break_even_for(row) == pytest.approx(0.70)

    def test_an_unrecorded_price_is_not_a_zero_bar(self):
        row = Row("2026-01-01T00:00:00+00:00", 1.0, 1.0, "YES", 0.6, None, 1.0)
        assert break_even_for(row) is None


# ---------------------------------------------------------------------------
# 2. what the harness will and will not conclude
# ---------------------------------------------------------------------------

class TestTheVerdicts:
    def test_an_edge_that_holds_is_confirmed_and_economic(self):
        report = run_walk_forward(_clean())
        assert report.verdict == CONFIRMED_ECONOMIC
        assert report.refuses is False

        best = [r for r in report.results if r.name == "edge_15pct"][0]
        assert best.significant is True
        assert best.confirmed is True, "it must survive the fresh holdout"
        assert best.economically_viable is True
        assert best.holdout_entries > 0
        assert best.lift > 0.10

    def test_no_relationship_is_no_signal(self):
        report = run_walk_forward(_noise())
        assert report.verdict == NO_SIGNAL
        assert "no rule beat the base rate" in report.summary
        assert not any(r.significant for r in report.results), (
            "noise must not produce a significant rule")

    def test_an_edge_that_stops_is_not_confirmed(self):
        """
        The false positive the holdout exists to catch.

        The rule looks strong on the discovery folds - it really was strong - and
        the newest third of the record says it stopped. Reporting it as working
        would be reporting the past as though it were the present.
        """
        report = run_walk_forward(_clean(big_until=240))
        assert report.verdict == UNCONFIRMED
        assert report.refuses is True
        assert set(report.refused_names()) >= {"edge_8pct", "edge_15pct"}
        assert "false positive" in report.summary
        for name in report.refused_names():
            row = [r for r in report.results if r.name == name][0]
            assert row.significant is True
            assert row.confirmed is False

    def test_a_real_lift_that_still_loses_money(self):
        """
        Significantly better than random, and still a losing strategy.

        The base rate is 30%: most trades lose. A rule selecting trades that win
        45% of the time beats that convincingly. At a 0.50 break-even it still
        loses money on every entry - which is the case a hit-rate-only harness
        would call a discovery.
        """
        rnd = random.Random(5)
        rows = []
        for i in range(400):
            strong = (i % 3 == 0)
            prob = 0.42 if strong else 0.24
            rows.append(_row(i, 0.90 if strong else 0.10, rnd.random() < prob))

        report = run_walk_forward(rows)
        assert report.verdict == CONFIRMED_UNECONOMIC
        assert report.refuses is True
        assert "still loses money" in report.summary
        confident = [r for r in report.results if r.name == "confident"][0]
        assert confident.significant and confident.confirmed
        assert confident.economically_viable is False

    def test_a_quiet_record_refuses_nothing(self):
        """
        "No filter beat taking every trade" is a statement about the filters.

        It is not evidence that the filters are HARMFUL, and turning it into a
        blanket refusal would stop an agent whose other gates had approved an
        opportunity for reasons this record cannot see.
        """
        report = run_walk_forward(_noise())
        assert report.verdict == NO_SIGNAL
        assert report.refused_names() == []
        assert report.refuses is False

    def test_too_little_evidence_says_so(self):
        report = run_walk_forward(_clean(40))
        assert report.verdict == INSUFFICIENT
        assert "not enough" in report.summary
        assert all(r.entries == 0 for r in report.results)

    def test_every_rule_is_reported_including_the_ones_that_failed(self):
        report = run_walk_forward(_clean())
        names = {r.name for r in report.results}
        assert {"all", "edge_8pct", "edge_15pct", "forecast_above_price",
                "confident"} <= names, (
            "a harness that only prints the winner is a sales document")
        assert report.results[0].name == "all", "the baseline comes first"

    def test_the_folds_are_consecutive_and_the_holdout_is_the_newest(self):
        report = run_walk_forward(_clean())
        for result in report.results:
            folds = result.folds
            assert [f.index for f in folds] == list(range(report.folds))
            held = [f.held_out for f in folds]
            assert held == sorted(held), (
                "the holdout must be the newest folds; anything else lets the "
                "discovery see the future")
            assert sum(1 for x in held if x) >= 1
            assert sum(1 for x in held if not x) >= 1


# ---------------------------------------------------------------------------
# 3. the line that must never be crossed
# ---------------------------------------------------------------------------

class TestABacktestCannotOpenTheGate:
    def test_may_qualify_is_always_false(self):
        report = run_walk_forward(_clean())
        assert report.verdict == CONFIRMED_ECONOMIC
        assert may_qualify(report) is False, (
            "a good backtest must never be able to qualify a venue for real "
            "capital; qualification stays on money that actually settled")
        assert may_qualify(None) is False

    def test_the_report_says_so_itself(self):
        payload = run_walk_forward(_clean()).to_dict()
        assert payload["may_qualify"] is False
        assert payload["may_refuse"] is True
        assert "never qualify" in payload["rule"]

    def test_the_validated_rules_are_not_a_qualification_input(self):
        """
        The gate reads realised outcomes. If a walk-forward verdict ever reached
        it, a backtest figure could move real capital - the one thing this module
        exists to prevent.
        """
        from pathlib import Path

        source = Path("src/ptai/venues/qualification.py").read_text()
        for term in ("walk_forward", "walk-forward", "validation"):
            assert term not in source.lower(), (
                f"{term} must not appear in the qualification gate")

    def test_the_agent_records_the_verdict_without_acting_on_it(self):
        from pathlib import Path

        source = Path("src/ptai/agent/v3_loop.py").read_text()
        flat = " ".join(source.split())
        assert "record_verdict(self.storage, _wf, scope=\"all\")" in flat
        assert "_wf.refused_names()" in flat, "a refusal is logged, not silent"


# ---------------------------------------------------------------------------
# 4. reading the record, writing the verdict
# ---------------------------------------------------------------------------

class TestItReadsTheRealRecord:
    @pytest.fixture()
    def storage(self, tmp_path):
        store = Storage(db_path=str(tmp_path / "wf.db"))
        yield store
        store.close()

    def _settle(self, storage, *, venue="polymarket", strategy="value",
                lane="paper", pnl=1.0, resolved_at=None, i=0):
        from src.ptai.learning.trade_outcomes import TradeOutcomeTracker

        tracker = TradeOutcomeTracker(storage=storage)
        trade_id = f"{venue}-{strategy}-{i}"
        tracker.record_trade(
            trade_id=trade_id, market_id=f"M{i}", venue_id=venue,
            strategy=strategy, category="politics", forecast_prob=0.6,
            market_price=0.5, edge=0.1, side="YES", amount_usd=1.0,
            execution_mode=lane, yes_price=0.5)
        tracker.record_resolution(trade_id, 1.0 if pnl > 0 else 0.0, pnl)
        if resolved_at:
            storage.conn.execute(
                "UPDATE trade_outcomes SET resolved_at = ? WHERE trade_id = ?",
                (resolved_at, trade_id))
            storage.conn.commit()

    def test_rows_come_back_in_time_order(self, storage):
        for i, when in enumerate(["2026-03-02", "2026-01-05", "2026-02-01"]):
            self._settle(storage, i=i, resolved_at=f"{when}T00:00:00+00:00")

        rows = rows_from_storage(storage)
        assert [r.resolved_at[:10] for r in rows] == [
            "2026-01-05", "2026-02-01", "2026-03-02"], (
            "a walk-forward split is only out-of-sample if the folds are "
            "consecutive in time")

    def test_a_filter_selects_the_record_it_names(self, storage):
        for i in range(4):
            self._settle(storage, i=i, venue="kalshi",
                         resolved_at=f"2026-01-0{i + 1}T00:00:00+00:00")
        self._settle(storage, i=9, venue="polymarket",
                     resolved_at="2026-01-09T00:00:00+00:00")

        assert len(rows_from_storage(storage, venue_id="kalshi")) == 4
        assert len(rows_from_storage(storage, venue_id="polymarket")) == 1
        assert len(rows_from_storage(storage, execution_mode="live")) == 0

    def test_an_unreadable_record_is_an_empty_record_not_a_crash(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "broken.db"))
        try:
            storage.conn.execute("DROP TABLE trade_outcomes")
            storage.conn.commit()
            assert rows_from_storage(storage) == []
        finally:
            storage.close()

    def test_the_verdict_round_trips_through_storage(self, storage):
        report = run_walk_forward(_clean())
        assert record_verdict(storage, report, scope="all") is True

        loaded = load_verdicts(storage)
        assert loaded["available"] is True
        assert loaded["scopes"]["all"]["verdict"] == CONFIRMED_ECONOMIC
        assert loaded["scopes"]["all"]["rows"] == 400
        assert loaded["scopes"]["all"]["may_qualify"] is False

    def test_no_verdict_yet_says_how_to_get_one(self, storage):
        loaded = load_verdicts(storage)
        assert loaded["available"] is False
        assert "python main.py validate" in loaded["reason"]

    def test_a_corrupt_verdict_is_reported_not_swallowed(self, storage):
        storage.set_state("validation.walk_forward", "[not a dict]")
        loaded = load_verdicts(storage)
        assert loaded["available"] is False or loaded["scopes"] == {}
        storage.set_state("validation.walk_forward", "{{broken")
        assert load_verdicts(storage)["available"] is False
        assert "unreadable" in load_verdicts(storage)["reason"]

    def test_a_custom_rule_is_just_a_predicate(self):
        """
        The harness is not tied to the shipped candidates - but a rule can only
        say WHICH trades it takes, never what it pays, because a rule that could
        describe its own payoff could describe a flattering one.
        """
        candidates = [Candidate("losing_half", "odd rows",
                                lambda r: True)]
        report = run_walk_forward([_row(i, 0.6, i % 2 == 0) for i in range(400)],
                                  candidates=candidates)
        assert report.results[0].name == "losing_half"
        assert report.verdict == NO_SIGNAL


# ---------------------------------------------------------------------------
# 5. the operator surfaces
# ---------------------------------------------------------------------------

class TestTheOperatorCanSeeTheVerdict:
    @pytest.fixture()
    def storage(self, tmp_path):
        store = Storage(db_path=str(tmp_path / "surf.db"))
        store.set_bankroll(50.0)
        yield store
        store.close()

    def test_the_snapshot_carries_it_and_the_cli_lines_name_it(self, storage):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        record_verdict(storage, run_walk_forward(_clean()), scope="all")
        snapshot = operator_snapshot(storage)
        assert snapshot["validation"]["available"] is True

        lines = describe_snapshot(snapshot)
        assert any(line.startswith("Out-of-sample: all:") for line in lines)
        # ...and it stays out of the venue ranking, where it could read as a
        # reason to deploy capital.
        assert "validation" not in (snapshot["venues"] or {})

    def test_without_a_verdict_the_lines_say_nothing_rather_than_zero(
            self, storage):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        lines = describe_snapshot(operator_snapshot(storage))
        assert not [line for line in lines if line.startswith("Out-of-sample:")]

    def test_the_console_panel_renders_the_folds(self):
        from src.ptai.ui import console as console_module

        html = console_module.CONSOLE_HTML
        assert 'id="validation"' in html
        assert "Out-of-sample validation" in html
        script = html.split("<script>")[1]
        assert "body.validation" in script
        assert "failed holdout" in script
        assert "It can never qualify a venue" in script

    def test_the_route_sends_it(self, tmp_path, monkeypatch):
        import importlib

        from fastapi.testclient import TestClient

        monkeypatch.setenv("PTAI_DB", str(tmp_path / "route.db"))
        module = importlib.import_module("src.ptai.ui.console")

        store = Storage(db_path=str(tmp_path / "route.db"))
        try:
            store.set_bankroll(50.0)
            record_verdict(store, run_walk_forward(_clean()), scope="all")
        finally:
            store.close()

        body = TestClient(module.app).get("/api/console/venue").json()
        assert body["validation"]["available"] is True
        assert body["validation"]["scopes"]["all"]["verdict"] == CONFIRMED_ECONOMIC
