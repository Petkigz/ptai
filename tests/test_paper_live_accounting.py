"""
What the account DID, versus what the simulation said it did.

The remaining danger in this codebase is no longer missing infrastructure - it is
accounting semantics. A simulation that can move real money, or a NO trade that
settles on the wrong side of the spread, makes the agent believe it earned
something the account never saw. Every test here is about one of those four
distinctions:

    paper   vs live          (whose bankroll moved)
    YES     vs NO            (which price was paid)
    data    vs execution     (which mode a trade ran in)
    edge    vs expected EV   (which number the gate is reading)
"""

from __future__ import annotations

import pytest

from src.ptai.execution.position_ledger import PositionLedgerBuilder
from src.ptai.execution.settlement import compute_pnl
from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker, qualification_stats_from_outcomes,
)
from src.ptai.storage.db import Storage, _execution_mode


# ---------------------------------------------------------------------------
# paper P&L must never reach the real bankroll
# ---------------------------------------------------------------------------

class TestPaperPnlNeverTouchesRealCapital:
    """
    `SettlementEngine` resolved every trade the same way, and
    `resolve_trade` banked the P&L into the bankroll. So this was possible:

        paper exploration trade -> paper P&L -> real bankroll += paper P&L

    The agent could manufacture or destroy real capital through simulated
    trades, which breaks the capital-growth mission at its foundation.
    """

    def _paper_trade(self, storage, pnl=1.20, size=3.0):
        trade_id = storage.log_trade({
            "market_id": "M-PAPER", "side": "YES", "position_size_usd": size,
            "market_price": 0.5, "yes_price_at_entry": 0.5,
            "token_price_at_entry": 0.5, "status": "paper",
            "execution_mode": "paper"})
        storage.resolve_trade(trade_id, outcome=1.0, pnl=pnl)
        return trade_id

    def _live_trade(self, storage, pnl=1.20, size=3.0):
        trade_id = storage.log_trade({
            "market_id": "M-LIVE", "side": "YES", "position_size_usd": size,
            "market_price": 0.5, "yes_price_at_entry": 0.5,
            "token_price_at_entry": 0.5, "status": "open",
            "execution_mode": "live"})
        storage.resolve_trade(trade_id, outcome=1.0, pnl=pnl)
        return trade_id

    def test_a_winning_paper_trade_does_not_change_the_real_bankroll(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p1.db"))
        before = storage.get_bankroll()
        self._paper_trade(storage, pnl=1.20)

        assert storage.get_bankroll() == pytest.approx(before), (
            "a simulated win moved real capital"
        )
        assert storage.get_paper_bankroll() == pytest.approx(before + 1.20), (
            "the paper account did not record the result, so the simulation "
            "learns nothing"
        )

    def test_a_losing_paper_trade_does_not_change_the_real_bankroll(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p2.db"))
        before = storage.get_bankroll()
        self._paper_trade(storage, pnl=-3.00)
        assert storage.get_bankroll() == pytest.approx(before)
        assert storage.get_paper_bankroll() == pytest.approx(before - 3.00)

    def test_a_live_trade_still_moves_the_real_bankroll(self, tmp_path):
        """
        The control. A fix that made NOTHING bank would pass the tests above and
        break the agent.
        """
        storage = Storage(db_path=str(tmp_path / "p3.db"))
        before = storage.get_bankroll()
        self._live_trade(storage, pnl=1.20)
        assert storage.get_bankroll() == pytest.approx(before + 1.20)

    def test_the_mode_is_read_from_the_row_not_from_the_caller(self, tmp_path):
        """
        `resolve_trade` decides from the stored row, so a caller cannot make a
        paper trade bank by asking for it - and cannot stop a live trade banking
        either.
        """
        storage = Storage(db_path=str(tmp_path / "p4.db"))
        before = storage.get_bankroll()
        trade_id = storage.log_trade({
            "market_id": "M-PAPER", "side": "YES", "position_size_usd": 3.0,
            "market_price": 0.5, "status": "paper", "execution_mode": "paper"})
        # A caller with a bug, or an enthusiasm for real money:
        storage.resolve_trade(trade_id, outcome=1.0, pnl=99.0)
        assert storage.get_bankroll() == pytest.approx(before), (
            "asking loudly enough made a simulation pay out real capital"
        )

    def test_realised_pnl_and_win_rate_ignore_paper(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p5.db"))
        self._paper_trade(storage, pnl=5.0)
        self._live_trade(storage, pnl=-1.0)

        summary = storage.get_performance_summary()
        assert summary["win_rate"] == 0.0, (
            "a paper win was counted in the operator's win rate"
        )
        live_only = storage.conn.execute(
            "SELECT SUM(pnl) AS s FROM trades WHERE resolved=1 "
            "AND execution_mode='live'").fetchone()["s"]
        assert live_only == pytest.approx(-1.0)

        paper = summary["paper"]
        assert paper["settled_trades"] == 1
        assert paper["net_pnl"] == pytest.approx(5.0), (
            "the paper record has to be kept, it is what step 6 runs on"
        )

    def test_the_ledger_realised_pnl_counts_live_settlements_only(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p6.db"))
        self._paper_trade(storage, pnl=7.5)
        ledger = PositionLedgerBuilder(storage=storage).build()
        assert ledger.realised_pnl == pytest.approx(0.0), (
            "simulated profit appeared in the operator's realised P&L"
        )

    def test_a_paper_position_reserves_no_real_cash(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "p7.db"))
        storage.log_trade({
            "market_id": "M-PAPER", "side": "YES", "position_size_usd": 3.0,
            "market_price": 0.5, "status": "paper", "execution_mode": "paper"})
        ledger = PositionLedgerBuilder(storage=storage).build()
        assert ledger.paper_position_count == 1
        assert ledger.live_position_count == 0
        assert ledger.reserved_capital == pytest.approx(0.0)


def test_the_mode_normaliser_fails_closed_but_not_against_live():
    """
    Unknown becomes paper, because the alternative is treating an unknown trade
    as one that moved real money. `status='open'` is the one inference that must
    NOT be paper: that is what the execution path writes for a real position, and
    calling it paper would leave real P&L unbanked forever.
    """
    assert _execution_mode("paper", None) == "paper"
    assert _execution_mode("simulated", None) == "paper"
    assert _execution_mode(None, "paper") == "paper"
    assert _execution_mode("live", None) == "live"
    assert _execution_mode(None, "open") == "live"
    assert _execution_mode(None, "executed") == "live"
    assert _execution_mode(None, None) == "paper"


# ---------------------------------------------------------------------------
# NO settlement must price the token that was bought
# ---------------------------------------------------------------------------

class TestNoSettlementPricesTheTokenBought:
    """
    YES 0.70, so the NO token costs 0.30. A $3 stake buys 10 NO shares; if NO
    wins they settle at $1 each, so the profit is $7.00.

    Settlement read the stored price as a YES price and computed 1 - 0.30 = 0.70
    per share, paying out as though the shares had cost 0.70 - $1.29 instead of
    $7.00 on the same trade.
    """

    def test_a_winning_no_is_paid_at_the_price_it_paid(self):
        pnl = compute_pnl("NO", 0.30, 3.0, 0.0, price_is_token_price=True)
        assert pnl == pytest.approx(7.0), f"a winning NO paid ${pnl:.2f}, not $7.00"

    def test_a_losing_no_loses_exactly_the_stake(self):
        pnl = compute_pnl("NO", 0.30, 3.0, 1.0, price_is_token_price=True)
        assert pnl == pytest.approx(-3.0)

    def test_the_two_conventions_agree_when_each_is_given_its_own_number(self):
        """
        A token price of 0.30 and a YES price of 0.70 describe the same trade,
        and must produce the same P&L. The bug was mixing them.
        """
        token = compute_pnl("NO", 0.30, 3.0, 0.0, price_is_token_price=True)
        yes_scale = compute_pnl("NO", 0.70, 3.0, 0.0, price_is_token_price=False)
        assert token == pytest.approx(yes_scale) == pytest.approx(7.0)

    def test_mixing_them_is_the_1_29_that_was_paid(self):
        """
        The regression itself, pinned: reading a 0.30 TOKEN price as a YES price.
        """
        wrong = compute_pnl("NO", 0.30, 3.0, 0.0, price_is_token_price=False)
        assert wrong == pytest.approx(1.2857, abs=0.001), (
            "this is the number that was being banked for a winning NO"
        )

    def test_settlement_uses_the_stored_token_price(self, tmp_path):
        """
        Through `Storage` and the row the execution path writes: the trade must
        settle on `token_price_at_entry`, not on a re-derivation.
        """
        storage = Storage(db_path=str(tmp_path / "n1.db"))
        trade_id = storage.log_trade({
            "market_id": "M-NO", "side": "NO", "position_size_usd": 3.0,
            "market_price": 0.70, "yes_price_at_entry": 0.70,
            "token_price_at_entry": 0.30, "status": "open",
            "execution_mode": "live"})
        row = storage.conn.execute(
            "SELECT market_price, yes_price_at_entry, token_price_at_entry "
            "FROM trades WHERE id=?", (trade_id,)).fetchone()
        assert row["market_price"] == pytest.approx(0.70), "market_price is the YES price"
        assert row["yes_price_at_entry"] == pytest.approx(0.70)
        assert row["token_price_at_entry"] == pytest.approx(0.30)

        pnl = compute_pnl("NO", row["token_price_at_entry"], 3.0, 0.0,
                          price_is_token_price=True)
        assert pnl == pytest.approx(7.0)

    def test_a_legacy_row_without_a_token_price_still_settles(self, tmp_path):
        """
        Rows written before the column existed put the FILL price in
        `market_price`, and the fill is on the token bought - so the same reading
        is the right one for them. It must not crash, and it must not silently
        return no P&L.
        """
        storage = Storage(db_path=str(tmp_path / "n2.db"))
        trade_id = storage.log_trade({
            "market_id": "M-LEGACY", "side": "NO", "position_size_usd": 3.0,
            "market_price": 0.30, "status": "open", "execution_mode": "live"})
        row = storage.conn.execute(
            "SELECT token_price_at_entry, market_price FROM trades WHERE id=?",
            (trade_id,)).fetchone()
        assert row["token_price_at_entry"] is None
        entry = row["token_price_at_entry"] or row["market_price"]
        assert compute_pnl("NO", entry, 3.0, 0.0,
                           price_is_token_price=True) == pytest.approx(7.0)

    def test_an_unpriceable_row_returns_none_rather_than_a_number(self):
        assert compute_pnl("NO", None, 3.0, 0.0, price_is_token_price=True) is None
        assert compute_pnl("NO", 0.0, 3.0, 0.0, price_is_token_price=True) is None
        assert compute_pnl("NO", 0.30, 3.0, None, price_is_token_price=True) is None


# ---------------------------------------------------------------------------
# execution mode is not data mode
# ---------------------------------------------------------------------------

class TestExecutionModeIsNotDataMode:
    """
    An exploration trade runs on paper against live prices. That is
    data_mode="live" and execution_mode="paper" - and the qualification split was
    computed from data_mode, so a simulated trade could be counted as a live
    outcome and carry a venue toward real capital.
    """

    def _record(self, tracker, storage, i, execution_mode, data_mode):
        trade_id = storage.log_trade({
            "market_id": f"M{i}", "side": "YES", "position_size_usd": 3.0,
            "market_price": 0.5, "status": "paper" if execution_mode == "paper" else "open",
            "execution_mode": execution_mode})
        tracker.record_trade(
            trade_id=str(trade_id), market_id=f"M{i}", venue_id="polymarket",
            strategy="value", forecast_prob=0.6, market_price=0.5, edge=0.10,
            side="YES", amount_usd=3.0,
            data_mode=data_mode, execution_mode=execution_mode)
        tracker.record_resolution(str(trade_id), actual_outcome=1.0, pnl=1.0)

    def test_live_data_with_paper_execution_counts_as_paper(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "e1.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(12):
            self._record(tracker, storage, i, "paper", "live")

        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["paper_trades"] == 12
        assert stats["live_trades"] == 0, (
            "simulated trades running against live prices were counted as live "
            "outcomes - this is what could carry a venue to real capital"
        )
        assert stats["profit_live"] == 0.0
        assert stats["profit_paper"] == pytest.approx(12.0)

    def test_live_execution_counts_as_live(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "e2.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(12):
            self._record(tracker, storage, i, "live", "live")
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["live_trades"] == 12
        assert stats["paper_trades"] == 0

    def test_an_unlabelled_outcome_is_neither_live_nor_paper(self, tmp_path):
        """
        Unknown must not be promoted to live, and must not be quietly filed as
        paper either. It is reported in its own count so the gap is visible.
        """
        storage = Storage(db_path=str(tmp_path / "e3.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(5):
            self._record(tracker, storage, i, "", "live")
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["live_trades"] == 0
        assert stats["paper_trades"] == 0
        assert stats["unclassified_trades"] == 5


# ---------------------------------------------------------------------------
# expected value means expected value
# ---------------------------------------------------------------------------

class TestQualificationReadsRecordedExpectedEv:
    """
    `expected_value` was `sum(edges)/n` - the mean recorded EFFECTIVE EDGE, a
    probability-scale number - compared against "EV > 1% per trade". Mean edge
    and expected monetary value are different quantities, and a venue could clear
    the EV bar on edge while losing money to fees.
    """

    def test_the_recorded_expected_net_ev_is_what_is_reported(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "q1.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(20):
            trade_id = storage.log_trade({
                "market_id": f"Q{i}", "side": "YES", "position_size_usd": 3.0,
                "market_price": 0.5, "execution_mode": "paper",
                "status": "paper"})
            tracker.record_trade(
                trade_id=str(trade_id), market_id=f"Q{i}",
                venue_id="polymarket", strategy="value", forecast_prob=0.6,
                market_price=0.5,
                # A LARGE edge that the costs would have eaten: this is the
                # number that must NOT be read as expected value.
                edge=0.20, side="YES", amount_usd=3.0,
                execution_mode="paper", data_mode="live",
                # The recorded prediction: 1% of a $3 stake.
                expected_net_ev=0.03, expected_net_ev_pct=0.01)
            tracker.record_resolution(str(trade_id), actual_outcome=1.0, pnl=0.03)

        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["avg_edge"] == pytest.approx(0.20), "the edge is still reported"
        assert stats["expected_value"] == pytest.approx(0.01, abs=1e-9), (
            "expected_value reported the mean edge, not the expected net EV"
        )
        assert stats["expected_value_usd"] == pytest.approx(0.03, abs=1e-9)
        assert stats["expected_value_samples"] == 20
        assert stats["expected_value_coverage"] == pytest.approx(1.0)

    def test_it_is_none_when_nothing_was_measured(self, tmp_path):
        """
        None, not 0.0: "never measured" is not "predicted no profit", and the
        gate must fail on it rather than treat it as a value.
        """
        storage = Storage(db_path=str(tmp_path / "q2.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(5):
            trade_id = storage.log_trade({
                "market_id": f"Q{i}", "side": "YES", "position_size_usd": 3.0,
                "market_price": 0.5, "execution_mode": "paper",
                "status": "paper"})
            tracker.record_trade(
                trade_id=str(trade_id), market_id=f"Q{i}",
                venue_id="polymarket", strategy="value", forecast_prob=0.6,
                market_price=0.5, edge=0.20, side="YES", amount_usd=3.0,
                execution_mode="paper", data_mode="live")
            tracker.record_resolution(str(trade_id), actual_outcome=1.0, pnl=0.0)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["expected_value"] is None
        assert stats["expected_value_samples"] == 0
        assert stats["expected_value_coverage"] == 0.0

    def test_the_gate_reports_which_number_it_used(self, tmp_path):
        """
        The reasoning line has to say whether the EV came from recorded
        predictions or from the mean-edge fallback, or a reader cannot tell.
        """
        from src.ptai.venues.qualification import VenueQualificationEngine
        storage = Storage(db_path=str(tmp_path / "q3.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        for i in range(20):
            trade_id = storage.log_trade({
                "market_id": f"Q{i}", "side": "YES", "position_size_usd": 3.0,
                "market_price": 0.5, "execution_mode": "paper",
                "status": "paper"})
            tracker.record_trade(
                trade_id=str(trade_id), market_id=f"Q{i}",
                venue_id="polymarket", strategy="value", forecast_prob=0.75,
                market_price=0.5, edge=0.15, side="YES", amount_usd=3.0,
                execution_mode="paper", data_mode="live",
                expected_net_ev=0.09, expected_net_ev_pct=0.03)
            tracker.record_resolution(str(trade_id), actual_outcome=1.0, pnl=0.09)

        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        result = engine.evaluate_qualification("polymarket",
                                               performance_stats=stats)
        assert "recorded expected net EV" in result.reasoning
        assert result.expected_value == pytest.approx(0.03, abs=1e-9)


# ---------------------------------------------------------------------------
# the venue's account, not ours
# ---------------------------------------------------------------------------

class TestVenueOnlyPositionsAreReserved:
    """
    The balance was venue-verified while the positions and orders beside it were
    read from PTAI's SQLite. A position the venue holds and PTAI does not know
    about is exposure the risk system cannot see - and free cash it may spend
    twice.
    """

    def test_a_venue_position_the_local_log_never_saw_is_reserved(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "v1.db"))
        ledger = PositionLedgerBuilder(storage=storage).build(
            venue_positions=[{"market_id": "EXT-1", "amount_usd": 12.0}],
            venue_state_complete=True)
        assert ledger.open_position_cost == pytest.approx(12.0)
        assert ledger.live_position_count == 1
        assert ledger.free_cash == pytest.approx(50.0 - 12.0)
        assert any("venue-only" in w for w in ledger.warnings), (
            "reserving foreign capital silently is how it gets double-spent later"
        )

    def test_an_unreadable_venue_account_blocks_new_sizing(self, tmp_path):
        """
        Fail closed, exactly like an unreadable working order: "I could not ask"
        and "there is nothing there" are different answers.
        """
        storage = Storage(db_path=str(tmp_path / "v2.db"))
        ledger = PositionLedgerBuilder(storage=storage).build(
            venue_state_complete=False)
        assert ledger.reservations_unknown is True
        assert ledger.can_open_new is False, (
            "sizing is allowed against an account that could not be read"
        )
        assert any("could not be read" in w for w in ledger.warnings)

    def test_a_readable_empty_account_does_not_block_anything(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "v3.db"))
        ledger = PositionLedgerBuilder(storage=storage).build(
            venue_positions=[], venue_state_complete=True)
        assert ledger.reservations_unknown is False
        assert ledger.can_open_new is True

    def test_an_unvalued_venue_position_is_treated_as_unknown(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "v4.db"))
        ledger = PositionLedgerBuilder(storage=storage).build(
            venue_positions=[{"market_id": "EXT-2", "amount_usd": 0}],
            venue_state_complete=True)
        assert ledger.reservations_unknown is True, (
            "a position with no value is still exposure; sizing must stop"
        )


# ---------------------------------------------------------------------------
# the LLM receives the shape V3 actually builds
# ---------------------------------------------------------------------------

class TestTheBrainReadsTheShapeV3Builds:
    """
    V3 puts a DICT in `context["sentiment"]`; `Brain._build_prompt` read
    attributes. On the real V3 path the LLM branch raised AttributeError before
    the provider was called, and the caller logged it as "LLM forecast failed" -
    which is why the component looked wired and produced nothing for every
    market. The research V3 collected was dropped on the same line.
    """

    def _market(self):
        from src.ptai.markets.base import Market, MarketSource
        return Market(id="M1", source=MarketSource.POLYMARKET,
                      question="Will the stub happen?",
                      outcome_prices=[0.70, 0.30], raw={})

    _V3_SENTIMENT = {
        "score": 0.42, "credibility": 0.8, "novelty": 0.5,
        "time_decay": 0.5, "corroborated": True, "reasoning": "X posts bullish",
    }

    def _router(self, captured):
        from types import SimpleNamespace

        class FakeRouter:
            def get_provider_name(self):
                return "fake-llm"

            def chat(self, prompt=None, system=None, **kw):
                captured["user"] = prompt or ""
                captured["system"] = system or ""
                return SimpleNamespace(
                    provider="fake-llm", content="",
                    parsed_json={"fair_value": 0.8, "confidence": 0.8,
                                 "reasoning": "ok", "should_trade": True})
        return FakeRouter()

    def test_a_dict_sentiment_reaches_the_prompt_instead_of_raising(self):
        from src.ptai.agent.brain import Brain
        captured = {}
        brain = Brain(llm_router=self._router(captured))
        result = brain.estimate_fair_value(
            market=self._market(), sentiment=dict(self._V3_SENTIMENT))
        assert result.fair_value == pytest.approx(0.8)
        assert "0.42" in captured["user"], (
            "the sentiment score V3 computed never reached the prompt"
        )

    def test_the_research_v3_collected_reaches_the_prompt(self):
        from src.ptai.agent.brain import Brain
        captured = {}
        brain = Brain(llm_router=self._router(captured))
        brain.estimate_fair_value(
            market=self._market(), sentiment=dict(self._V3_SENTIMENT),
            research_text="RESEARCH_MARKER: base rate 0.8 over 400 cases")
        assert "RESEARCH_MARKER" in captured["user"], (
            "web research was fetched and thrown away before the LLM call"
        )

    def test_the_no_llm_fallback_survives_the_dict_too(self):
        """
        The second reader of the same value: `_fallback_heuristic` reads
        `.confidence`. Normalising only inside `_build_prompt` fixed the prompt
        and failed here, which is the same bug one step along.
        """
        from src.ptai.agent.brain import Brain
        result = Brain().estimate_fair_value(
            market=self._market(), sentiment=dict(self._V3_SENTIMENT))
        assert result.llm_provider == "heuristic"
        assert result.fair_value > 0.0

    def test_the_ensemble_ends_up_with_an_llm_component(self):
        """
        The whole point: the 0.30-weight component must exist when a router does.
        """
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        captured = {}
        ctx = {"category": "politics", "sentiment": dict(self._V3_SENTIMENT),
               "tweets": [{"text": "yes"}], "news": "headline",
               "research": "RESEARCH_MARKER",
               "orderbook": {"is_real": True, "spread": 0.02, "depth": 5000}}
        forecaster = EnsembleForecaster(llm_router=self._router(captured))
        result = forecaster.forecast_market(self._market(), context=ctx)
        names = [f.model_name for f in result.model_forecasts]
        assert "llm_reasoning" in names, (
            f"the LLM component is missing from the ensemble: {names}"
        )
        llm = [f for f in result.model_forecasts if f.model_name == "llm_reasoning"][0]
        assert llm.confidence == pytest.approx(0.8)
        assert "RESEARCH_MARKER" in captured["user"]


# ---------------------------------------------------------------------------
# the report the operator reads
# ---------------------------------------------------------------------------

class TestTheSettlementReportSaysWhoseMoneyMoved:
    """
    `realised_pnl_usd` summed paper and live together, so a settlement report
    reading "+$9.20" could be $8.00 of simulation and $1.20 of real money with
    nothing to tell them apart.
    """

    def test_the_report_splits_paper_from_realised(self):
        from src.ptai.execution.settlement import SettlementReport

        report = SettlementReport()
        report.realised_pnl_usd += 1.20
        report.live_settled += 1
        report.paper_pnl_usd += 8.00
        report.paper_settled += 1

        payload = report.to_dict()
        assert payload["realised_pnl_usd"] == pytest.approx(1.20), (
            "the realised figure must be real money only"
        )
        assert payload["paper_pnl_usd"] == pytest.approx(8.00)
        assert payload["live_settled"] == 1
        assert payload["paper_settled"] == 1

    def test_the_summary_the_dashboard_reads_keeps_them_apart(self, tmp_path):
        """
        `total_pnl` is bankroll minus initial, so it is real money by
        construction now that paper P&L cannot reach the bankroll. The paper
        record rides alongside it rather than inside it.
        """
        storage = Storage(db_path=str(tmp_path / "r1.db"))
        paper_id = storage.log_trade({
            "market_id": "M-P", "side": "YES", "position_size_usd": 3.0,
            "market_price": 0.5, "status": "paper", "execution_mode": "paper"})
        storage.resolve_trade(paper_id, outcome=1.0, pnl=4.0)
        live_id = storage.log_trade({
            "market_id": "M-L", "side": "YES", "position_size_usd": 3.0,
            "market_price": 0.5, "status": "open", "execution_mode": "live"})
        storage.resolve_trade(live_id, outcome=1.0, pnl=0.5)

        summary = storage.get_performance_summary()
        assert summary["total_pnl"] == pytest.approx(0.5), (
            "the operator's P&L included a simulated result"
        )
        assert summary["paper"]["net_pnl"] == pytest.approx(4.0), (
            "the paper record is missing, so step 6 has nothing to read"
        )
        assert summary["paper"]["win_rate"] == pytest.approx(100.0)
        assert summary["win_rate"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# a measurement has to cover the sample it is judging
# ---------------------------------------------------------------------------

class TestTheGateRequiresCostCoverage:
    """
    The gate read an average execution quality, not how much of the record it
    covered - so a venue whose costs were measured on one trade out of 150 was
    judged on that one measurement, and 0.5 could pass the 0.5 requirement on a
    single lucky fill.
    """

    def _engine(self, tmp_path):
        from src.ptai.venues.qualification import VenueQualificationEngine
        return VenueQualificationEngine(data_dir=str(tmp_path))

    def _record(self, storage, tracker, n, measured):
        """`measured` of `n` trades carry cost figures; the rest carry none."""
        for i in range(n):
            trade_id = storage.log_trade({
                "market_id": f"CV{i}", "side": "YES", "position_size_usd": 3.0,
                "market_price": 0.5, "status": "paper",
                "execution_mode": "paper"})
            won = (i % 10) < 8
            kwargs = ({"fees_usd": 0.06, "slippage_bps": 10.0,
                       "execution_quality": 0.98}
                      if i < measured else {})
            tracker.record_trade(
                trade_id=str(trade_id), market_id=f"CV{i}",
                venue_id="polymarket", strategy="value", forecast_prob=0.75,
                market_price=0.5, edge=0.15, side="YES", amount_usd=3.0,
                execution_mode="paper", data_mode="live", **kwargs)
            tracker.record_resolution(str(trade_id), actual_outcome=1.0 if won else 0.0,
                                      pnl=1.2 if won else -1.8)

    def test_coverage_is_reported_with_the_totals(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "cv1.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        self._record(storage, tracker, 20, measured=5)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["costs_measured"] == 5
        assert stats["cost_coverage"] == pytest.approx(0.25)
        assert stats["execution_quality_coverage"] == pytest.approx(0.25)

    def test_a_thin_measurement_does_not_qualify_a_venue(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "cv2.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        # 150 trades, a good record, but costs measured on 10 of them - and those
        # ten look excellent.
        self._record(storage, tracker, 150, measured=10)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        result = self._engine(tmp_path).evaluate_qualification("polymarket", stats)
        assert result.is_qualified is False, (
            "a venue was qualified on a cost measurement covering 7% of its "
            f"trades; reasoning: {result.reasoning[-200:]}"
        )
        assert "exec_quality" in result.reasoning

    def test_a_complete_measurement_is_not_penalised(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "cv3.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        self._record(storage, tracker, 150, measured=150)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["cost_coverage"] == pytest.approx(1.0)
        # The same record that qualifies elsewhere in the suite still does: the
        # check is passable, so the coverage bar cannot be read as "never trade".
        assert stats["execution_quality_avg"] > 0.5
