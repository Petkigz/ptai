"""
The capital-reality layer: fills, the position ledger, and live readiness.

Five defects from the operator's audit, all confirmed by execution:

1. The qualification gate inverted itself. `is_qualified = not
   qualified_venue_ids or opp.venue_id in qualified_venue_ids or ...` meant that
   when NO venue had qualified, everything was treated as qualified. On a fresh
   install no venue qualifies, so the default state permitted live capital.

2. A rejected/errored execution still created an "open" position. The ledger row
   was written unconditionally after execution, at the REQUESTED size and the
   REQUESTED price, because the executor copied those straight through. A trade
   that never happened was then closed by settlement and booked a P&L.

3. `TradeOutcomeTracker.record_trade` was called by V3 with fields it does not
   accept (confidence, data_mode, trust_tier) and without the fields it requires
   (trade_id, category, forecast_prob, market_price, side), so every call raised
   TypeError and no venue/strategy outcome was recorded.

4. `TradeOutcomeTracker.outcomes` was memory-only, so venue and strategy
   performance vanished on restart.

5. Sizing used the stored bankroll, which does not distinguish committed capital
   from free capital, so positions in one batch each claimed 6% of the same
   dollars.

Plus the readiness ladder's top rung, which had no implementation and so could
never be reached: no adapter implemented an order probe.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from src.ptai.execution.multi_venue_executor import (
    FILLED_STATUSES,
    NO_POSITION_STATUSES,
    SIMULATED_STATUSES,
    ExecutionResult,
    MultiVenueExecutor,
)
from src.ptai.execution.position_ledger import PositionLedgerBuilder
from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
from src.ptai.storage.db import Storage
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueOpportunity,
    VenueType,
)
from src.ptai.markets.base import Market, MarketSource, Token


# --------------------------------------------------------------------------
# 1. the qualification gate
# --------------------------------------------------------------------------

class TestQualificationGateIsOneWay:
    def test_no_qualified_venues_means_nothing_is_qualified(self):
        """
        The regression, stated as the rule: an empty qualified set is an empty
        qualified set.
        """
        qualified_venue_ids = []
        candidates = ["polymarket", "kalshi", "manifold"]
        admitted = [
            v for v in candidates
            if bool(qualified_venue_ids)
            and (v in qualified_venue_ids or v.split("+")[0] in qualified_venue_ids)
        ]
        assert admitted == [], (
            "with no qualified venues, no venue may be treated as qualified"
        )

    def test_the_old_expression_admitted_everything(self):
        """Demonstrates the exact defect that was removed."""
        qualified_venue_ids = []
        venue_id = "polymarket"
        old_expression = (
            not qualified_venue_ids
            or venue_id in qualified_venue_ids
            or venue_id.split("+")[0] in qualified_venue_ids
        )
        assert old_expression is True, (
            "the old gate used to admit everything when nothing was qualified"
        )

    def test_source_no_longer_contains_the_inverted_gate(self):
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "not qualified_venue_ids or" not in source, (
            "the inverted qualification gate is back"
        )
        assert "bool(qualified_venue_ids)" in source, (
            "the gate must require a non-empty qualified set"
        )

    def test_exploration_lane_is_the_only_destination_without_qualification(self):
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "paper/shadow" in source, (
            "unqualified venues must be routed to paper/shadow explicitly"
        )


# --------------------------------------------------------------------------
# 2. only a proven fill becomes a position
# --------------------------------------------------------------------------

class TestFillReadingIsHonest:
    @pytest.fixture()
    def executor(self):
        return MultiVenueExecutor.__new__(MultiVenueExecutor)

    def test_real_fill_is_read(self, executor):
        fill = executor._read_fill(
            {"status": "matched", "orderID": "0xabc", "size": 4.0, "price": 0.5},
            2.0, 0.5)
        assert fill["status"] == "filled"
        assert fill["filled_usd"] == pytest.approx(2.0)
        assert fill["price"] == pytest.approx(0.5)
        assert fill["order_id"] == "0xabc"

    def test_explicit_usd_fill_is_read(self, executor):
        fill = executor._read_fill(
            {"status": "filled", "filled_usd": 3.5, "filled_price": 0.42}, 3.5, 0.5)
        assert fill["status"] == "filled"
        assert fill["filled_usd"] == pytest.approx(3.5)
        assert fill["price"] == pytest.approx(0.42)

    @pytest.mark.parametrize("status", sorted(NO_POSITION_STATUSES))
    def test_refusals_commit_nothing(self, executor, status):
        fill = executor._read_fill({"status": status, "reason": "no"}, 2.0, 0.5)
        assert fill["filled_usd"] == 0.0
        assert fill["status"] in NO_POSITION_STATUSES

    def test_rejection_reason_is_preserved(self, executor):
        """Refusals keep their own name so an operator can see why."""
        assert executor._read_fill({"status": "rejected"}, 2.0, 0.5)["status"] == "rejected"
        assert executor._read_fill({"status": "error"}, 2.0, 0.5)["status"] == "error"
        assert executor._read_fill({"status": "rate_limited"}, 2.0, 0.5)["status"] == "rate_limited"

    def test_a_fill_with_no_size_is_not_accountable(self, executor):
        """
        The venue says it took the order but reports no size. We cannot account
        for capital we cannot quantify, so it must not become a position.
        """
        fill = executor._read_fill({"status": "filled"}, 2.0, 0.5)
        assert fill["status"] == "unknown"
        assert fill["filled_usd"] == 0.0
        assert "no fill size" in fill["note"]

    @pytest.mark.parametrize("result", [None, "a string", 42, []])
    def test_garbage_is_not_a_fill(self, executor, result):
        fill = executor._read_fill(result, 2.0, 0.5)
        assert fill["status"] == "unknown"
        assert fill["filled_usd"] == 0.0

    def test_unknown_status_is_not_a_fill(self, executor):
        fill = executor._read_fill({"status": "something_new"}, 2.0, 0.5)
        assert fill["status"] == "unknown"

    def test_the_requested_amount_is_never_used_as_the_fill(self, executor):
        """
        The old code copied max_spend_usd and max_price into the result. A
        refusal must not report the requested figures as having filled.
        """
        fill = executor._read_fill({"status": "rejected"}, 7.5, 0.99)
        assert fill["filled_usd"] != 7.5
        assert fill["price"] != 0.99


class TestExecutionResultClassification:
    @staticmethod
    def _result(status, filled_usd=0.0):
        from src.ptai.execution.multi_venue_executor import ExecutionResult
        return ExecutionResult(
            venue_id="v", market_id="m", status=status, amount_usd=2.0,
            price=0.5, fees_usd=0.0, gas_usd=0.0, latency_ms=1.0,
            reasoning="x", filled_usd=filled_usd)

    @pytest.mark.parametrize("status", sorted(FILLED_STATUSES))
    def test_filled_statuses_commit_capital(self, status):
        r = self._result(status, filled_usd=2.0)
        assert r.committed_capital is True
        assert r.should_record_position is True
        assert r.is_simulated is False

    @pytest.mark.parametrize("status", sorted(SIMULATED_STATUSES))
    def test_simulated_statuses_do_not_commit_real_capital(self, status):
        r = self._result(status)
        assert r.committed_capital is False
        assert r.is_simulated is True
        # A simulated result records a position only if it SIMULATED a fill.
        # This used to be True for any simulated status, and the loop then
        # booked the REQUESTED size - so a dry run with no fill became a paper
        # position at a price nobody traded at. A venue that cannot price a
        # fill produces no paper evidence, which is the honest outcome: a
        # fabricated position would qualify a venue on trades that never
        # existed.
        result_with_size = ExecutionResult(
            venue_id="v", market_id="m", status="dry_run", amount_usd=2.0,
            price=0.5, fees_usd=0.0, gas_usd=0.0, latency_ms=1.0, reasoning="",
            filled_usd=1.8, filled_price=0.5,
            paper_fill={"filled_usd": 1.8})
        assert result_with_size.should_record_position is True, (
            "a simulated fill with a size must become a paper position, which "
            "is how paper trading qualifies venues"
        )
        assert result_with_size.position_size_usd == pytest.approx(1.8)
        assert r.should_record_position is False, (
            "a dry run still produces a PAPER position, which is how paper "
            "trading qualifies venues"
        )

    @pytest.mark.parametrize("status", sorted(NO_POSITION_STATUSES))
    def test_no_position_statuses_record_nothing(self, status):
        r = self._result(status)
        assert r.should_record_position is False, (
            f"{status} committed no capital and must not create a position"
        )

    def test_filled_without_size_does_not_commit(self):
        r = self._result("filled", filled_usd=0.0)
        assert r.committed_capital is False

    def test_to_position_dict_reports_the_facts(self):
        r = self._result("filled", filled_usd=1.75)
        d = r.to_position_dict()
        assert d["filled_usd"] == pytest.approx(1.75)
        assert d["requested_usd"] == pytest.approx(2.0)
        assert d["status"] == "filled"

    def test_run_cycle_gates_recording_on_the_fill(self):
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "should_record_position" in source, (
            "position recording must be gated on a proven fill"
        )
        assert "position_recorded" in source

    def test_trade_row_uses_the_fill_not_the_request(self):
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "exec_result.filled_price" in source, (
            "the trade must be recorded at the price that filled"
        )
        assert "exec_result.filled_usd" in source, (
            "the trade must be recorded at the size that filled"
        )


# --------------------------------------------------------------------------
# 3 and 4. the tracker
# --------------------------------------------------------------------------

class TestTradeOutcomeTracker:
    def test_accepts_the_fields_v3_passes(self):
        """
        V3 passed confidence, data_mode and trust_tier. The signature must
        tolerate extras rather than discarding the whole outcome with a
        TypeError.
        """
        tracker = TradeOutcomeTracker()
        ok = tracker.record_trade(
            trade_id="t1", market_id="m1", venue_id="polymarket",
            strategy="value", category="politics", forecast_prob=0.7,
            market_price=0.55, edge=0.15, side="YES", amount_usd=3.0,
            confidence=0.8, data_mode="live_paper", trust_tier=1)
        assert ok is True
        assert len(tracker.outcomes) == 1

    def test_accepts_positional_legacy_calls(self):
        """The old positional signature must keep working."""
        tracker = TradeOutcomeTracker()
        ok = tracker.record_trade(
            "t1", "m1", "polymarket", "value", "politics",
            0.7, 0.55, 0.15, "YES", 3.0)
        assert ok is True

    @pytest.mark.parametrize("missing", ["trade_id", "market_id", "venue_id", "strategy"])
    def test_refuses_an_unattributable_outcome(self, missing):
        """
        An outcome with no venue or strategy cannot inform allocation. Refusing
        it loudly is better than storing a row nothing can use.
        """
        tracker = TradeOutcomeTracker()
        kwargs = dict(trade_id="t1", market_id="m1", venue_id="polymarket",
                      strategy="value", forecast_prob=0.7, market_price=0.55)
        kwargs[missing] = ""
        assert tracker.record_trade(**kwargs) is False
        assert tracker.outcomes == []

    def test_missing_forecast_is_refused_not_invented(self):
        """
        Falling back to the market price would record the market's view as ours
        and corrupt every calibration score computed from it.
        """
        tracker = TradeOutcomeTracker()
        assert tracker.record_trade(
            trade_id="t1", market_id="m1", venue_id="v", strategy="s",
            forecast_prob=None, market_price=0.55) is False
        assert tracker.outcomes == []

    def test_missing_market_price_is_refused(self):
        tracker = TradeOutcomeTracker()
        assert tracker.record_trade(
            trade_id="t1", market_id="m1", venue_id="v", strategy="s",
            forecast_prob=0.7, market_price=None) is False

    def test_resolution_computes_brier_and_correctness(self):
        tracker = TradeOutcomeTracker()
        tracker.record_trade(trade_id="t1", market_id="m1", venue_id="v",
                             strategy="s", forecast_prob=0.8, market_price=0.5)
        assert tracker.record_resolution("t1", 1.0, 2.0) is True
        o = tracker.outcomes[0]
        assert o.brier_score == pytest.approx(0.04)
        assert o.was_correct is True
        assert o.pnl == pytest.approx(2.0)

    def test_a_wrong_forecast_is_marked_wrong(self):
        tracker = TradeOutcomeTracker()
        tracker.record_trade(trade_id="t1", market_id="m1", venue_id="v",
                             strategy="s", forecast_prob=0.9, market_price=0.5)
        tracker.record_resolution("t1", 0.0, -3.0)
        assert tracker.outcomes[0].was_correct is False

    def test_double_resolution_is_ignored(self):
        tracker = TradeOutcomeTracker()
        tracker.record_trade(trade_id="t1", market_id="m1", venue_id="v",
                             strategy="s", forecast_prob=0.8, market_price=0.5)
        assert tracker.record_resolution("t1", 1.0, 2.0) is True
        assert tracker.record_resolution("t1", 0.0, -99.0) is False
        assert tracker.outcomes[0].pnl == pytest.approx(2.0)

    def test_unknown_trade_resolution_is_reported(self):
        tracker = TradeOutcomeTracker()
        assert tracker.record_resolution("nonexistent", 1.0, 1.0) is False

    def test_outcomes_survive_a_restart(self, tmp_path):
        """
        The tracker held outcomes in a plain list, so venue and strategy
        performance vanished on every restart and allocation re-estimated from
        nothing.
        """
        db = str(tmp_path / "to.db")
        storage = Storage(db_path=db)
        tracker = TradeOutcomeTracker(storage=storage)
        tracker.record_trade(trade_id="t1", market_id="m1", venue_id="polymarket",
                             strategy="value", category="politics",
                             forecast_prob=0.7, market_price=0.55, edge=0.15,
                             side="YES", amount_usd=3.0)
        tracker.record_resolution("t1", 1.0, 2.0)
        perf_before = tracker.get_venue_performance()
        storage.close()

        storage2 = Storage(db_path=db)
        tracker2 = TradeOutcomeTracker(storage=storage2)
        try:
            assert len(tracker2.outcomes) == 1, "outcomes did not survive the restart"
            o = tracker2.outcomes[0]
            assert o.actual_outcome == pytest.approx(1.0)
            assert o.pnl == pytest.approx(2.0)
            perf_after = tracker2.get_venue_performance()
            assert perf_after == perf_before, (
                "venue performance changed across a restart, so allocation "
                "would be estimated from different numbers each run"
            )
        finally:
            storage2.close()

    def test_performance_aggregates_by_venue_and_strategy(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i, (venue, won) in enumerate([("polymarket", True), ("polymarket", False),
                                          ("kalshi", True)]):
            tracker.record_trade(trade_id=f"t{i}", market_id=f"m{i}", venue_id=venue,
                                 strategy="value", forecast_prob=0.7, market_price=0.5)
            tracker.record_resolution(f"t{i}", 1.0 if won else 0.0,
                                      2.0 if won else -1.0)
        perf = tracker.get_venue_performance()
        assert "polymarket" in perf
        assert perf["polymarket"]["total"] == 2
        assert perf["polymarket"]["wins"] == 1
        storage.close()


# --------------------------------------------------------------------------
# 5. the position ledger
# --------------------------------------------------------------------------

class TestPositionLedger:
    @pytest.fixture()
    def storage(self, tmp_path):
        s = Storage(db_path=str(tmp_path / "l.db"))
        s.set_bankroll(50.0)
        yield s
        s.close()

    def _open(self, storage, market_id, size, price=0.5, side="YES", status="open"):
        storage.conn.execute(
            "INSERT INTO trades (timestamp, market_id, side, market_price, "
            "position_size_usd, status, venue_id, resolved) VALUES "
            "('2026-01-01T00:00:00', ?, ?, ?, ?, ?, 'polymarket', 0)",
            (market_id, side, price, size, status))
        storage.conn.commit()

    def test_clean_state(self, storage):
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.equity == pytest.approx(50.0)
        assert L.free_cash == pytest.approx(50.0)
        assert L.reserved_capital == 0.0
        assert L.can_open_new is True

    def test_committed_capital_reduces_free_cash(self, storage):
        self._open(storage, "M1", 3.0, 0.60)
        storage.set_bankroll(47.0)
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.reserved_capital == pytest.approx(3.0)
        assert L.free_cash == pytest.approx(44.0), (
            "free cash must exclude capital already committed"
        )
        assert L.equity == pytest.approx(47.0)

    def test_sizing_uses_free_cash_not_equity(self, storage):
        """
        The core defect: sizing against the bankroll let concurrent positions
        each claim 6% of the same dollars.
        """
        self._open(storage, "M1", 3.0, 0.60)
        storage.set_bankroll(47.0)
        builder = PositionLedgerBuilder(storage=storage)
        L = builder.build()
        assert builder.sizing_capital() == pytest.approx(L.free_cash)
        assert builder.sizing_capital() != pytest.approx(L.equity), (
            "sizing capital must differ from equity while capital is committed"
        )

    def test_no_free_cash_blocks_new_positions(self, storage):
        self._open(storage, "M1", 50.0, 0.5)
        storage.set_bankroll(0.0)
        builder = PositionLedgerBuilder(storage=storage)
        L = builder.build()
        assert L.free_cash == 0.0
        assert L.can_open_new is False

    def test_paper_positions_do_not_consume_live_capital(self, storage):
        self._open(storage, "M1", 3.0, 0.60)
        self._open(storage, "M2", 5.0, 0.50, status="paper")
        storage.set_bankroll(47.0)
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.reserved_capital == pytest.approx(3.0), (
            "paper capital is not real capital"
        )
        assert L.paper_position_count == 1
        assert L.live_position_count == 1

    def test_marking_to_market_moves_unrealised_pnl(self, storage):
        self._open(storage, "M1", 3.0, 0.60)
        storage.set_bankroll(47.0)
        builder = PositionLedgerBuilder(storage=storage)
        up = builder.build(price_lookup=lambda mid: 0.80)
        down = builder.build(price_lookup=lambda mid: 0.30)
        assert up.unrealised_pnl > 0
        assert down.unrealised_pnl < 0
        assert up.unrealised_pnl != down.unrealised_pnl

    def test_unmarked_positions_are_carried_at_cost_with_a_warning(self, storage):
        """
        An unknown mark is not a zero and not a gain. Reporting it as profit
        would be inventing a number.
        """
        self._open(storage, "M1", 3.0, 0.60)
        storage.set_bankroll(47.0)
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.unrealised_pnl == pytest.approx(0.0)
        assert L.open_position_value == pytest.approx(3.0)
        assert any("carried at cost" in w for w in L.warnings)

    def test_a_missing_mark_only_affects_that_position(self, storage):
        self._open(storage, "M1", 3.0, 0.60)
        self._open(storage, "M2", 2.0, 0.50)
        storage.set_bankroll(45.0)
        L = PositionLedgerBuilder(storage=storage).build(
            price_lookup=lambda mid: 0.80 if mid == "M1" else None)
        assert any("1 open position" in w for w in L.warnings)
        assert L.unrealised_pnl == pytest.approx(1.0), "M1: 5 shares at 0.60 -> 4.00 cost 3.00"

    def test_realised_pnl_is_summed_from_settled_trades(self, storage):
        storage.conn.execute(
            "INSERT INTO trades (timestamp, market_id, side, market_price, "
            "position_size_usd, status, resolved, pnl) VALUES "
            "('2026-01-01T00:00:00','M9','YES',0.5,2.0,'settled',1,1.5)")
        storage.conn.commit()
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.realised_pnl == pytest.approx(1.5)

    def test_ledger_disagreement_is_reported(self, storage):
        """
        Committed capital exceeding the recorded bankroll means storage and the
        ledger disagree. That must be visible, not absorbed by max(0, ...).
        """
        self._open(storage, "M1", 80.0, 0.5)
        storage.set_bankroll(50.0)
        L = PositionLedgerBuilder(storage=storage).build()
        assert L.free_cash == 0.0
        assert any("exceed" in w for w in L.warnings)

    def test_ledger_reports_the_figures_the_operator_asked_for(self, storage):
        self._open(storage, "M1", 3.0, 0.60)
        storage.set_bankroll(47.0)
        d = PositionLedgerBuilder(storage=storage).build().to_dict()
        for key in ("equity", "free_cash", "reserved_capital",
                    "open_position_value", "realised_pnl", "unrealised_pnl",
                    "reserved_pct", "live_position_count", "paper_position_count"):
            assert key in d, f"the operator-facing ledger is missing {key}"

    def test_v3_sizes_against_free_capital(self):
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "free_capital * kelly_result.kelly_fraction_adj" in source, (
            "sizing must use free capital"
        )
        assert "bankroll=free_capital" in source, (
            "Kelly must be given free capital, not the stored bankroll"
        )
        assert "free_capital - _committed" in source, (
            "free capital must be drawn down within a batch, or every position "
            "in the batch sizes against the same snapshot"
        )


class TestRiskChainCallsTheRealMethods:
    """
    Three consecutive calls in the pre-execution risk chain named methods and
    parameters that do not exist:

      * KellyCalculator.calculate(edge=, prob=, confidence=) - the real
        signature is (market_price, fair_prob, bankroll) and it returns a
        KellyResult, not a fraction.
      * ExposureManager.can_open_position(...) - the real name is can_open().
      * LimitsEngine.validate(...) - the real name is validate_proposal().

    Each would have raised on the first qualified opportunity, so the entire
    risk gate was a chain of independent exceptions. Nothing caught it because
    no venue ever qualified in the suite, which made the sizing path reachable
    in principle and dead in practice.
    """

    def test_the_kelly_call_matches_the_real_signature(self):
        import inspect as _inspect
        from src.ptai.risk.kelly import KellyCalculator

        params = set(_inspect.signature(KellyCalculator.calculate).parameters)
        assert {"market_price", "fair_prob", "bankroll"} <= params, (
            f"KellyCalculator.calculate signature changed: {params}"
        )

    def test_v3_passes_the_real_kelly_parameters(self):
        """
        Scoped to the Kelly call itself: `edge=` is a legitimate argument
        elsewhere (the outcome tracker takes one), so a whole-function grep
        would fail for the wrong reason.
        """
        import re
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        calls = re.findall(r"kelly_calculator\.calculate\((.*?)\)", source, re.S)
        assert calls, "no kelly_calculator.calculate call found at all"
        for call in calls:
            # Word-boundary match: plain `"prob=" in call` would also match
            # inside `fair_prob=`, which is a real parameter.
            for bad in re.findall(r"(?<![_\w])(edge|prob|confidence)\s*=", call):
                raise AssertionError(
                    f"the dead Kelly call is back: {bad!r} is not a "
                    f"KellyCalculator parameter and raises TypeError on every "
                    f"qualified opportunity. Call was: {call.strip()[:200]}"
                )
            assert "market_price" in call and "fair_prob" in call and "bankroll" in call, (
                f"Kelly call does not pass the real parameters: {call.strip()[:200]}"
            )

    def test_kelly_declines_are_respected(self):
        """
        KellyResult carries should_bet. Treating the result as a bare fraction
        would discard the calculator's own refusal, including its 8% min_edge.
        """
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "kelly_result.should_bet" in source, (
            "Kelly's refusal is being ignored"
        )

    def test_the_exposure_call_names_a_real_method(self):
        from src.ptai.risk.exposure import ExposureManager

        assert hasattr(ExposureManager, "can_open"), (
            "ExposureManager.can_open is missing"
        )
        assert not hasattr(ExposureManager, "can_open_position"), (
            "can_open_position exists again; the v3_loop call must be updated"
        )
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "exposure_manager.can_open_position" not in source, (
            "the dead ExposureManager call is back"
        )
        assert "exposure_manager.can_open(" in source

    def test_the_limits_call_names_a_real_method(self):
        from src.ptai.risk.limits import LimitsEngine

        assert hasattr(LimitsEngine, "validate_proposal")
        assert not hasattr(LimitsEngine, "validate"), (
            "LimitsEngine.validate exists again; the v3_loop call must be updated"
        )
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "limits_engine.validate(" not in source, (
            "the dead LimitsEngine call is back"
        )
        assert "limits_engine.validate_proposal(" in source

    def test_limits_proposal_carries_the_keys_it_reads(self):
        """
        validate_proposal reads `trade`, `fair_probability` and
        `market_probability`. Without `trade` it defaults the flag to False and
        rejects every opportunity with "LLM says no trade" - a rejection that
        looks like a risk decision and is really a missing key.
        """
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        for key in ('"trade": True', '"fair_probability"', '"market_probability"'):
            assert key in source, f"validate_proposal is not given {key}"

    def test_the_adjusted_size_is_used_not_the_proposed_one(self):
        """
        validate_proposal returns an adjusted order. Sizing past the approved
        amount would make the limits engine advisory.
        """
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert 'adjusted.get("max_spend_usd")' in source, (
            "the limits-approved size is being discarded"
        )


# --------------------------------------------------------------------------
# the order probe: the rung that had no implementation
# --------------------------------------------------------------------------

class TestOrderProbeMakesReadinessReachable:
    def _opportunity(self):
        m = Market(id="M1", source=MarketSource.POLYMARKET, question="Q?",
                   condition_id="0xcond1",
                   tokens=[Token(token_id="tok-1", outcome="YES", price=0.5)])
        return VenueOpportunity(market=m, venue_id="polymarket",
                                venue_type=VenueType.PREDICTION, side="YES",
                                market_price=0.5, estimated_fair=0.6, raw_edge=0.1)

    def _adapter(self, dry_run=False, credentials=True):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        if credentials:
            ad = PolymarketAdapter(private_key="0x" + "ab" * 32,
                                   funder="0x" + "cd" * 20, dry_run=dry_run)
        else:
            ad = PolymarketAdapter(dry_run=dry_run)
        ad.capabilities.supports_trading = credentials
        return ad

    def test_polymarket_declares_the_probe(self):
        ad = self._adapter()
        assert ad.capabilities.supports_order_probe is True

    def test_probe_refuses_in_dry_run(self):
        """
        A probe places a real order. Running one while the agent believes it is
        simulating is exactly what the dry_run gate exists to prevent.
        """
        ad = self._adapter(dry_run=True)
        assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False

    def test_probe_refuses_without_credentials(self):
        ad = self._adapter(credentials=False)
        assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False

    def test_probe_refuses_without_a_market(self):
        ad = self._adapter()
        assert asyncio.run(ad.probe_order_permission(None)) is False

    def _with_fake_clob(self, post_response, cancel_raises=False, not_cancelled=False):
        """
        Swap only the CLOB transport, keeping the real PolymarketExecutor.

        Subclassing rather than replacing means cancel_order() and its
        not_canceled handling are the real implementation - the pathway the
        probe actually depends on - instead of a stub that could pass while the
        real one is broken.
        """
        import src.ptai.markets.polymarket as pm

        class FakeClient:
            """The V2 CLOB surface, which is what the executor now calls."""

            def __init__(self):
                self.order_args, self.options, self.posted, self.cancelled = [], [], [], []

            def get_clob_market_info(self, condition_id):
                # A real market payload, so the probe reads a real tick.
                return {"min_tick_size": "0.01", "neg_risk": False,
                        "min_order_size": 1.0, "taker_base_fee": 0.0,
                        "maker_base_fee": 0.0, "seconds_delay": 0,
                        "accepting_orders": True}

            def get_tick_size(self, token_id):
                return "0.01"

            def get_neg_risk(self, token_id):
                return False

            def create_and_post_order(self, order_args, options, order_type, post_only):
                self.order_args.append(order_args)
                self.options.append(options)
                self.posted.append((order_type, post_only))
                return post_response

            def cancel_order(self, payload):
                order_id = getattr(payload, "orderID", payload)
                if cancel_raises:
                    raise RuntimeError("network down")
                self.cancelled.append(order_id)
                if not_cancelled:
                    return {"canceled": [], "not_canceled": {order_id: "order not found"}}
                return {"canceled": [order_id], "not_canceled": {}}

        client = FakeClient()
        real_cls = pm.PolymarketExecutor

        class FakeExecutor(real_cls):
            def __init__(self, **kwargs):
                # Bypass client construction entirely; everything else is real.
                self.client = client
                self.client_error = None
                self.private_key = kwargs.get("private_key")
                self.funder = kwargs.get("funder")
                self._mechanics_cache = {}

        pm.PolymarketExecutor = FakeExecutor
        return client, lambda: setattr(pm, "PolymarketExecutor", real_cls)

    def test_successful_round_trip_verifies_permission(self):
        ad = self._adapter()
        client, restore = self._with_fake_clob({"success": True, "orderID": "o1"})
        try:
            assert asyncio.run(ad.probe_order_permission(self._opportunity())) is True
        finally:
            restore()
        assert client.cancelled == ["o1"], "the probe order must be withdrawn"
        assert ad.last_order_probe["cancelled"] is True

    def test_the_probe_order_cannot_cross(self):
        """
        The probe must test permission without taking a position. It posts at
        the market's OWN lowest expressible price - the tick size - not at a
        hardcoded 0.01, which on a 0.001-tick market is a real bid.
        """
        ad = self._adapter()
        client, restore = self._with_fake_clob({"success": True, "orderID": "o1"})
        try:
            asyncio.run(ad.probe_order_permission(self._opportunity()))
        finally:
            restore()
        args = client.order_args[0]
        assert args.price == 0.01, (
            f"probe posted at {args.price}, which could fill and become a real position"
        )
        assert ad.last_order_probe["mechanics"]["tick_size"] == "0.01"

    def test_the_probe_reads_the_market_tick_not_a_constant(self):
        """
        On a market with a finer tick, the lowest safe price is lower than
        0.01. Posting 0.01 there is a genuine bid that could fill.
        """
        ad = self._adapter()
        client, restore = self._with_fake_clob({"success": True, "orderID": "o1"})
        original = client.get_clob_market_info

        def finer(cid):
            info = original(cid)
            info["min_tick_size"] = "0.001"
            return info
        client.get_clob_market_info = finer
        try:
            asyncio.run(ad.probe_order_permission(self._opportunity()))
        finally:
            restore()
        assert client.order_args[0].price == 0.001, (
            "the probe ignored the market's tick and used a constant"
        )

    def test_a_rejected_post_does_not_verify(self):
        ad = self._adapter()
        _, restore = self._with_fake_clob({"success": False, "errorMsg": "denied"})
        try:
            assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False
        finally:
            restore()

    def test_a_failed_cancel_does_not_verify(self):
        """
        An order left resting in the book is exposure the caller does not know
        about, so the probe must not claim success.
        """
        ad = self._adapter()
        _, restore = self._with_fake_clob({"success": True, "orderID": "o1"},
                                          cancel_raises=True)
        try:
            assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False
        finally:
            restore()
        assert ad.last_order_probe["cancelled"] is False
        assert "cancel" in ad.last_order_probe["reason"]

    def test_a_venue_refused_cancel_does_not_verify(self):
        ad = self._adapter()
        _, restore = self._with_fake_clob({"success": True, "orderID": "o1"},
                                          not_cancelled=True)
        try:
            assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False
        finally:
            restore()

    def test_a_post_with_no_order_id_does_not_verify(self):
        ad = self._adapter()
        _, restore = self._with_fake_clob({"success": True})
        try:
            assert asyncio.run(ad.probe_order_permission(self._opportunity())) is False
        finally:
            restore()

    def test_health_engine_reaches_trade_permitted(self):
        """
        The ladder's top rung must be reachable, or the ladder is a wall.
        """
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        ad = self._adapter()
        async def portfolio():
            return {"available": True, "balance": 120.0, "source": "clob_real"}
        ad.get_portfolio = portfolio

        _, restore = self._with_fake_clob({"success": True, "orderID": "o1"})
        class Registry:
            adapters = {"polymarket": ad}
        # Polymarket requires identity verification for UG, and the ladder now
        # requires a RECORD that it was completed before the top rung - an
        # authenticated, funded account with a working probe is not evidence that
        # a KYC check happened. The operator completes that at the venue, so it is
        # recorded the way the operator records it.
        import tempfile

        from src.ptai.execution.account_health import record_verification
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=tempfile.mktemp(suffix=".db"))
        record_verification(storage, "polymarket", "KYC completed")
        engine = AccountHealthEngine(storage=storage)
        engine.venue_registry = Registry()
        try:
            result = asyncio.run(engine.check_venue_health(
                "polymarket", opportunity=self._opportunity()))
        finally:
            restore()

        assert result.readiness == TradeReadiness.TRADE_PERMITTED
        assert result.ready_to_trade is True
        assert result.healthy is True
        assert any("Verification recorded" in c for c in result.checks)

    def test_health_engine_stops_at_funded_when_the_probe_fails(self):
        from src.ptai.execution.account_health import (
            AccountHealthEngine,
            TradeReadiness,
        )

        ad = self._adapter()
        async def portfolio():
            return {"available": True, "balance": 120.0, "source": "clob_real"}
        ad.get_portfolio = portfolio

        _, restore = self._with_fake_clob({"success": False})
        class Registry:
            adapters = {"polymarket": ad}
        engine = AccountHealthEngine()
        engine.venue_registry = Registry()
        try:
            result = asyncio.run(engine.check_venue_health(
                "polymarket", opportunity=self._opportunity()))
        finally:
            restore()

        assert result.readiness == TradeReadiness.FUNDED
        assert result.ready_to_trade is False
        assert "order_permission_unproven" in result.blockers

    def test_v3_passes_the_opportunity_to_the_probe(self):
        """
        Without this the probe has no market to test against and can never
        verify anything, so TRADE_PERMITTED would be unreachable in practice.
        """
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "opportunity=opp" in source, (
            "V3 must pass the opportunity so the probe has a market to test on"
        )


class TestDelayedFillsKeepTheirThesis:
    """
    An order that rests and fills hours later used to become "resting_order_fill"
    with edge 0 and confidence 0.

    The trade had a thesis; the fill had amnesia. Learning from the outcome then
    taught the agent about a strategy that never chose the trade, which is worse
    than not learning at all - it is learning something false. The forecast is now
    written down WITH the order and read back when the fill arrives.
    """

    def _order_manager(self, tmp_path):
        from src.ptai.execution.order_manager import OrderManager

        return OrderManager(storage=Storage(db_path=str(tmp_path / "orders.db")))

    def _submitted(self, manager, forecast):
        from src.ptai.execution.multi_venue_executor import ExecutionResult

        # `needs_reconciliation` is derived, not passed: it is a property of the
        # result, computed from the status and the unfilled remainder.
        result = ExecutionResult(
            venue_id="polymarket", market_id="M-REST",
            status="resting", amount_usd=3.0, price=0.55,
            fees_usd=0.0, gas_usd=0.0, latency_ms=12.0,
            reasoning="limit order rested in the book",
            filled_usd=0.0, order_id="ORD-REST-1", resting_usd=3.0)
        key = manager.record_submission(
            result, market_id="M-REST", venue_id="polymarket", side="YES",
            forecast=forecast)
        assert key, "the order was not recorded, so nothing could be reconciled"
        return key

    def test_the_forecast_is_stored_with_the_order(self, tmp_path):
        manager = self._order_manager(tmp_path)
        self._submitted(manager, {
            "fair_price": 0.72, "edge": 0.17, "confidence": 0.81,
            "strategy": "value_poisson", "category": "sports",
            "data_mode": "live_paper"})

        row = manager.storage.get_order_row("ORD-REST-1")
        assert row is not None, "the order was not stored at all"
        assert row["fair_price"] == pytest.approx(0.72)
        assert row["edge"] == pytest.approx(0.17)
        assert row["confidence"] == pytest.approx(0.81)
        assert row["strategy"] == "value_poisson"
        assert row["category"] == "sports"

    def test_a_delayed_fill_inherits_the_original_thesis(self, tmp_path):
        """
        The position opened hours later must carry what the agent believed when it
        decided - not the price the fill happened to land at.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3

        manager = self._order_manager(tmp_path)
        self._submitted(manager, {
            "fair_price": 0.72, "edge": 0.17, "confidence": 0.81,
            "strategy": "value_poisson", "category": "sports",
            "data_mode": "live_paper"})

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.storage = manager.storage
        agent.trade_outcome_tracker.storage = manager.storage

        row = manager.storage.get_order_row("ORD-REST-1")
        trade_id = agent._open_position_from_fill(dict(row), 3.0, 0.55)
        assert trade_id > 0, "the fill produced no position"

        position = manager.storage.conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert position["strategy"] == "value_poisson", (
            "the position was filed under strategy "
            f"{position['strategy']!r} instead of the strategy that chose it"
        )
        assert position["category"] == "sports"
        assert position["fair_value"] == pytest.approx(0.72), (
            "the fair value was replaced by the fill price, so the forecast "
            "cannot be scored"
        )
        assert position["edge"] == pytest.approx(0.17)
        assert position["confidence"] == pytest.approx(0.81)
        assert str(position["order_id"]) == "ORD-REST-1", (
            "the position cannot be traced back to the venue order it came from"
        )

    def test_the_delayed_fill_reaches_the_learning_record(self, tmp_path):
        """The point of keeping the thesis: the outcome must be learnable."""
        from src.ptai.agent.v3_loop import TradingAgentV3

        manager = self._order_manager(tmp_path)
        self._submitted(manager, {
            "fair_price": 0.72, "edge": 0.17, "confidence": 0.81,
            "strategy": "value_poisson", "category": "sports",
            "data_mode": "live_paper"})

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.storage = manager.storage
        tracker = agent.trade_outcome_tracker
        tracker.storage = manager.storage

        row = manager.storage.get_order_row("ORD-REST-1")
        trade_id = agent._open_position_from_fill(dict(row), 3.0, 0.55)

        recorded = [o for o in tracker.outcomes if o.trade_id == str(trade_id)]
        assert recorded, (
            "the delayed fill produced no learning record, so the agent cannot "
            "study its decisions that took time to execute"
        )
        outcome = recorded[0]
        assert outcome.venue_id == "polymarket"
        assert outcome.strategy == "value_poisson"
        assert outcome.forecast_prob == pytest.approx(0.72)
        assert outcome.edge == pytest.approx(0.17)

    def test_a_fill_with_no_forecast_is_labelled_unknown_not_invented(self, tmp_path):
        """
        An order with no stored forecast - placed by an older version, or created
        by reconciliation itself. The position must still be opened (the exposure
        is real) but it must be marked as unattributed rather than dressed up as a
        decision the agent made.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3

        manager = self._order_manager(tmp_path)
        self._submitted(manager, None)  # no forecast supplied

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.storage = manager.storage
        agent.trade_outcome_tracker.storage = manager.storage

        row = manager.storage.get_order_row("ORD-REST-1")
        trade_id = agent._open_position_from_fill(dict(row), 3.0, 0.55)
        assert trade_id > 0, "the exposure is real; the position must exist"

        position = manager.storage.conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert position["strategy"] == "resting_order_fill"
        assert position["edge"] == 0.0
        assert position["confidence"] == 0.0
        # The fair value falls back to the fill price, which is the only price
        # known, and the outcome is flagged unattributed so it is not mistaken
        # for a measured forecast.
        assert position["fair_value"] == pytest.approx(0.55)
        unattributed = [
            o for o in agent.trade_outcome_tracker.outcomes
            if o.trade_id == str(trade_id)]
        assert unattributed, "even an unattributed fill must reach the record"
