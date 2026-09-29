"""
A round ends with a number, and it does more than three things at once.

The operator's request, in their words:

    "for a round to be complete its supposed to do research on the market find
     the suitable predictions research more about them on which ones to bet on
     and then bet on them using the fake currency in the paper mode and produce
     results. A complete run should take some time and when its done its supposed
     to produce results in form of either a bankroll with a negative or a
     positive. i dont know how many events it can manage to be involved with at
     the same time but it would be better if it would do more at once."

So there are three claims to prove here, and one of them is a claim about the
accounting:

  1. A round has two ends and a net. `equity_end - equity_start`, realised and
     marked kept apart, and NO RESULT rather than a flat $50 when nothing could
     be priced.
  2. The paper account is marked to market, or a paper round can only ever
     report the figure it started with. (Carrying positions at cost made paper
     equity a constant between settlements.)
  3. "More at once" is bounded by the RULES, not by the number of ideas: a round
     fills every position slot the limits leave open, and the exposure ceilings
     it is checked against can actually see the book that already exists.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.ptai.execution.position_ledger import PositionLedgerBuilder
from src.ptai.execution.round import (
    RoundReport,
    load_rounds,
    record_round,
    round_history_summary,
)
from src.ptai.storage.db import Storage


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _storage(tmp_path) -> Storage:
    return Storage(db_path=str(tmp_path / "round.db"))


def _paper_position(storage, market_id="M-PAPER", size=3.0, entry=0.5,
                    side="YES", status="paper", category="politics"):
    """A recorded paper position, the way the execution loop writes one."""
    return storage.log_trade({
        "market_id": market_id,
        "market_question": f"will {market_id} happen?",
        "side": side,
        "market_price": entry,
        "token_price_at_entry": entry,
        "yes_price_at_entry": entry,
        "position_size_usd": size,
        "status": status,
        "execution_mode": "paper",
        "category": category,
        "correlation_group": category,
        "venue_id": "polymarket",
    })


def _ledger(storage, price_lookup=None):
    return PositionLedgerBuilder(storage=storage).build(price_lookup=price_lookup)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    """An agent whose storage is this test's database, never the workspace one."""
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
    return tmp_path


# ===========================================================================
# 1. the arithmetic: two ends and a net, or no number at all
# ===========================================================================

class TestARoundHasTwoEndsAndANet:
    def test_the_net_is_the_difference_between_the_two_ends(self):
        report = RoundReport(number=3, equity_start=50.0, equity_end=51.25)
        assert report.net_usd == pytest.approx(1.25)
        assert report.verdict == "up"

    def test_a_fall_is_down(self):
        report = RoundReport(number=4, equity_start=50.0, equity_end=47.5)
        assert report.net_usd == pytest.approx(-2.5)
        assert report.verdict == "down"

    def test_a_move_smaller_than_a_cent_is_flat(self):
        """
        Half a cent is rounding noise on a 4-decimal account, and calling it a
        gain would make every quiet round look like a win.
        """
        report = RoundReport(equity_start=50.0, equity_end=50.004)
        assert report.verdict == "flat"

    def test_no_end_value_is_unknown_not_flat(self):
        """
        The bug this replaces: `net = end - start` with a missing end reports a
        flat account. Unknown and unchanged are opposite facts.
        """
        report = RoundReport(equity_start=50.0, equity_end=None)
        assert report.net_usd is None
        assert report.verdict == "unknown"

    def test_no_start_value_is_unknown_too(self):
        report = RoundReport(equity_start=None, equity_end=52.0)
        assert report.net_usd is None
        assert report.verdict == "unknown"

    def test_the_report_keeps_realised_and_marked_apart(self):
        report = RoundReport(equity_start=50.0, equity_end=51.0,
                             realised_pnl=4.0, unrealised_pnl=-3.0)
        body = report.to_dict()
        assert body["realised_pnl"] == 4.0
        assert body["unrealised_pnl"] == -3.0
        assert body["net_usd"] == 1.0
        assert body["open_positions_marked"] == 0

    def test_the_headline_carries_both_ends_and_the_net(self):
        report = RoundReport(number=7, equity_start=50.0, equity_end=51.2,
                             unrealised_pnl=1.2, positions_opened=6,
                             positions_held=6, duration_seconds=95.0)
        headline = report.headline()
        assert "Round 7" in headline
        assert "$50.00 -> $51.20" in headline
        assert "+$1.20" in headline
        assert "marked, not settled" in headline, (
            "a marked gain is not a settlement and the log must not say it is"
        )

    def test_a_flat_round_is_reported_as_zero_not_plus_zero(self):
        report = RoundReport(number=2, equity_start=50.0, equity_end=50.0)
        assert "= $0.00" in report.headline()
        assert "+$0.00" not in report.headline()

    def test_a_round_with_no_figure_says_so_in_the_headline(self):
        report = RoundReport(number=9, equity_start=50.0, equity_end=None)
        headline = report.headline()
        assert "no result" in headline
        assert "$50.00" not in headline, (
            "a round that could not be priced must not print the starting "
            "figure as if it were the result"
        )


# ===========================================================================
# 2. the record: rounds accumulate, and the running score is the answer
# ===========================================================================

class TestTheRoundsAreKeptAndScored:
    def _round(self, number, start=50.0, end=50.0):
        return RoundReport(number=number, equity_start=start, equity_end=end)

    def test_rounds_are_read_newest_first(self, tmp_path):
        storage = _storage(tmp_path)
        for n, (start, end) in enumerate([(50, 51), (51, 49), (49, 50)], start=1):
            record_round(storage, self._round(n, start, end))
        rows = load_rounds(storage)
        assert [r["number"] for r in rows] == [3, 2, 1]

    def test_the_history_is_capped_and_keeps_the_newest(self, tmp_path):
        storage = _storage(tmp_path)
        for n in range(1, 60):
            record_round(storage, self._round(n, 50.0, 50.0 + n / 100))
        rows = load_rounds(storage, limit=9999)
        assert len(rows) == 50
        assert rows[0]["number"] == 59, "the newest round must survive the cap"

    def test_the_summary_scores_up_down_and_flat_and_totals_them(self, tmp_path):
        storage = _storage(tmp_path)
        record_round(storage, self._round(1, 50.0, 51.0))
        record_round(storage, self._round(2, 51.0, 49.5))
        record_round(storage, self._round(3, 49.5, 49.5))
        summary = round_history_summary(load_rounds(storage))
        assert summary["scored"] == 3
        assert summary["up"] == 1 and summary["down"] == 1 and summary["flat"] == 1
        assert summary["net_usd"] == pytest.approx(-0.5)
        assert summary["best_round"] == pytest.approx(1.0)
        assert summary["worst_round"] == pytest.approx(-1.5)

    def test_a_round_with_no_figure_is_kept_but_not_scored(self, tmp_path):
        """
        The record keeps every round - "it ran and could not be priced" is
        history - but the score is computed only over rounds that produced a
        number, and says how many that was.
        """
        storage = _storage(tmp_path)
        record_round(storage, self._round(1, 50.0, 51.0))
        record_round(storage, RoundReport(number=2, equity_start=51.0,
                                          equity_end=None))
        summary = round_history_summary(load_rounds(storage))
        assert summary["rounds"] == 2
        assert summary["scored"] == 1
        assert summary["net_usd"] == pytest.approx(1.0)

    def test_no_scored_round_reports_no_number_rather_than_zero(self):
        summary = round_history_summary([{"number": 1, "net_usd": None}])
        assert summary["net_usd"] is None
        assert "no completed round has produced a result yet" in summary["note"]

    def test_a_round_without_storage_is_not_recorded_and_does_not_raise(self):
        assert record_round(None, RoundReport(number=1)) is False

    def test_an_unreadable_record_reports_nothing_rather_than_inventing_one(
            self, tmp_path):
        storage = _storage(tmp_path)
        storage.set_state("agent.rounds", "{not json")
        assert load_rounds(storage) == []


# ===========================================================================
# 2b. the paper account is a real account: marked, with its own cash
# ===========================================================================

class TestThePaperBookIsMarkedToMarket:
    def test_a_paper_position_is_worth_what_it_would_sell_for(self, tmp_path):
        """
        $3 at 0.50 is 6 shares. At 0.60 those shares are worth $3.60, and that
        is what the paper account holds - otherwise paper equity is a constant
        and no paper round can ever report a result.
        """
        storage = _storage(tmp_path)
        _paper_position(storage, size=3.0, entry=0.50)
        ledger = _ledger(storage, price_lookup=lambda mid: 0.60)
        assert ledger.paper_position_value == pytest.approx(3.6, abs=0.01)
        assert ledger.paper_unrealised_pnl == pytest.approx(0.6, abs=0.01)
        assert ledger.paper_equity == pytest.approx(50.6, abs=0.01)
        assert ledger.paper_marked == 1 and ledger.paper_unmarked == 0

    def test_a_no_position_is_marked_on_its_own_side(self, tmp_path):
        """
        A NO position gains when YES falls. Marking it on the YES price would
        report the loss as a gain.
        """
        storage = _storage(tmp_path)
        _paper_position(storage, side="NO", size=3.0, entry=0.50)
        ledger = _ledger(storage, price_lookup=lambda mid: 0.40)
        # NO entry 0.50 -> 6 shares; NO mark = 1 - 0.40 = 0.60 -> $3.60
        assert ledger.paper_position_value == pytest.approx(3.6, abs=0.01)
        assert ledger.paper_unrealised_pnl == pytest.approx(0.6, abs=0.01)

    def test_an_unpriced_position_is_carried_at_cost_and_counted(self, tmp_path):
        """An unknown mark is not a zero and not a gain."""
        storage = _storage(tmp_path)
        _paper_position(storage, size=3.0, entry=0.50)
        ledger = _ledger(storage, price_lookup=lambda mid: None)
        assert ledger.paper_position_value == pytest.approx(3.0)
        assert ledger.paper_unrealised_pnl == pytest.approx(0.0)
        assert ledger.paper_unmarked == 1
        assert any("could not be marked" in w for w in ledger.warnings)

    def test_a_paper_mark_never_moves_the_live_equity(self, tmp_path):
        storage = _storage(tmp_path)
        _paper_position(storage, size=3.0, entry=0.50)
        ledger = _ledger(storage, price_lookup=lambda mid: 0.60)
        assert ledger.unrealised_pnl == pytest.approx(0.0)
        assert ledger.live_position_count == 0
        assert ledger.equity == pytest.approx(50.0)

    def test_the_report_exposes_the_paper_split(self, tmp_path):
        storage = _storage(tmp_path)
        _paper_position(storage, size=3.0, entry=0.50)
        body = _ledger(storage, price_lookup=lambda mid: 0.60).to_dict()
        assert body["paper_unrealised_pnl"] == pytest.approx(0.6)
        assert body["paper_marked"] == 1
        assert body["paper_unmarked"] == 0
        assert body["paper_equity"] == pytest.approx(50.6)


# ===========================================================================
# 3. closing a round: the agent scores the account it is actually trading
# ===========================================================================

class TestTheAgentScoresTheRound:
    def _agent_with(self, monkeypatch, tmp_path, dry_run=True):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
        from src.ptai.agent.v3_loop import TradingAgentV3

        return TradingAgentV3(country_code="UG", dry_run=dry_run)

    def test_a_paper_round_that_lost_money_reports_a_negative(self, monkeypatch,
                                                             tmp_path):
        """
        The whole request in one test: buy $3 at 0.50, the price falls to 0.30,
        and the round ends DOWN by $1.20 on the paper account.
        """
        agent = self._agent_with(monkeypatch, tmp_path)
        _paper_position(agent.storage, market_id="M-FALL", size=3.0, entry=0.50)
        agent._round_open_ledger = agent._round_ledger(lambda mid: 0.50)
        report = RoundReport(number=1, mode="paper", account="paper",
                             started_at="2026-09-28T00:00:00+00:00")
        monkeypatch.setattr(agent, "_current_marks", lambda markets: {"M-FALL": 0.30})

        agent._score_round(report, markets=[], scan_result=None,
                           execution_results=[], discovered=12)

        assert report.equity_start == pytest.approx(50.0)
        assert report.equity_end == pytest.approx(48.8)
        assert report.net_usd == pytest.approx(-1.2)
        assert report.verdict == "down"
        assert "-$1.20" in report.headline()

    def test_a_paper_round_that_gained_reports_a_positive(self, monkeypatch,
                                                          tmp_path):
        agent = self._agent_with(monkeypatch, tmp_path)
        _paper_position(agent.storage, market_id="M-RISE", size=3.0, entry=0.50)
        agent._round_open_ledger = agent._round_ledger(lambda mid: 0.50)
        report = RoundReport(number=2, mode="paper", account="paper")
        monkeypatch.setattr(agent, "_current_marks", lambda markets: {"M-RISE": 0.70})

        agent._score_round(report, markets=[], scan_result=None,
                           execution_results=[], discovered=8)

        assert report.net_usd == pytest.approx(1.2)
        assert report.verdict == "up"

    def test_the_round_counts_the_stages_the_operator_named(self, monkeypatch,
                                                            tmp_path):
        """
        research -> suitable predictions -> more research -> the bets -> the
        result. The report carries a number for each stage, so "it ran" can be
        told apart from "it did the work".
        """
        agent = self._agent_with(monkeypatch, tmp_path)
        agent._round_open_ledger = agent._round_ledger(lambda mid: None)
        agent._screen = {"considered": 41, "shortlist": [], "screened_out": 33}
        agent._deep_market_ids = {"a", "b", "c"}
        agent._round_settled = 2
        report = RoundReport(number=3, mode="paper", account="paper")
        monkeypatch.setattr(agent, "_current_marks", lambda markets: {})

        class _Scan:
            total_scanned = 17
            total_candidates = 5

        agent._score_round(report, markets=[], scan_result=_Scan(),
                           execution_results=[
                               {"position_recorded": True, "amount_usd": 3.0},
                               {"position_recorded": True, "amount_usd": 2.0},
                               {"position_recorded": False},
                           ], discovered=64)

        assert report.markets_discovered == 64
        assert report.markets_screened == 41
        assert report.markets_researched == 3
        assert report.markets_priced == 17
        assert report.candidates == 5
        assert report.positions_opened == 2
        assert report.staked_usd == pytest.approx(5.0)
        assert report.positions_settled == 2

    def test_the_marks_are_stored_for_the_next_rounds_opening_book(self, monkeypatch,
                                                                   tmp_path):
        agent = self._agent_with(monkeypatch, tmp_path)
        agent._round_open_ledger = agent._round_ledger(lambda mid: None)
        report = RoundReport(number=4, mode="paper", account="paper")
        monkeypatch.setattr(agent, "_current_marks",
                            lambda markets: {"M-A": 0.44, "M-B": 0.61})
        agent._score_round(report, markets=[], scan_result=None,
                           execution_results=[], discovered=1)
        assert agent._round_marks() == {"M-A": 0.44, "M-B": 0.61}

    def test_a_quiet_round_keeps_the_last_prices_it_saw(self, monkeypatch,
                                                        tmp_path):
        """
        No venue answered this round. The book is still marked at the last prices
        this agent saw, so the round reports a flat account - not a phantom loss
        caused by the position suddenly having no mark at all.
        """
        agent = self._agent_with(monkeypatch, tmp_path)
        _paper_position(agent.storage, market_id="M-QUIET", size=3.0, entry=0.50)
        agent.storage.set_state("agent.position_marks", json.dumps({"M-QUIET": 0.60}))
        agent._round_open_ledger = agent._round_ledger(lambda mid: 0.60)
        report = RoundReport(number=5, mode="paper", account="paper")
        monkeypatch.setattr(agent, "_current_marks", lambda markets: {})

        agent._score_round(report, markets=[], scan_result=None,
                           execution_results=[], discovered=0)

        assert report.equity_start == pytest.approx(50.6)
        assert report.equity_end == pytest.approx(50.6)
        assert report.net_usd == pytest.approx(0.0)
        assert report.verdict == "flat"
        assert report.open_positions_marked == 1
        assert report.open_positions_unmarked == 0

    def test_current_marks_refuse_a_book_that_failed_validation(self, monkeypatch,
                                                                tmp_path):
        """
        A book the scan refused must not be used to mark a position. Those
        positions are counted as unmarked instead, and carried at cost.
        """
        agent = self._agent_with(monkeypatch, tmp_path)
        agent._cycle_books = {
            "bad": {"validated": False, "midpoint": 0.10},
            "good": {"validated": True, "midpoint": 0.42},
        }

        class _Market:
            def __init__(self, mid, price):
                self.id, self.yes_price = mid, price

        marks = agent._current_marks([_Market("bad", 0.10), _Market("good", 0.42)])
        assert marks == {"good": 0.42}


# ===========================================================================
# 4. more at once - bounded by the rules, not by the ideas
# ===========================================================================

class TestARoundFillsTheSlotsTheRulesLeaveOpen:
    def _agent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
        from src.ptai.agent.v3_loop import TradingAgentV3

        return TradingAgentV3(country_code="UG", dry_run=True)

    def test_it_asks_for_every_free_slot(self, monkeypatch, tmp_path):
        agent = self._agent(monkeypatch, tmp_path)
        for i in range(2):
            _paper_position(agent.storage, market_id=f"M-{i}", size=3.0)
        called = {}

        async def _fake_cycle(target_per_venue=200, max_trades=3):
            called["target_per_venue"] = target_per_venue
            called["max_trades"] = max_trades
            return {"status": "stub"}

        monkeypatch.setattr(agent, "run_cycle", _fake_cycle)
        import asyncio

        result = asyncio.run(agent.run_round())
        assert result == {"status": "stub"}
        assert called["max_trades"] == agent.limits_engine.limits.max_open_positions - 2
        assert called["max_trades"] >= 1

    def test_a_full_book_is_not_widened_to_keep_the_round_busy(self, monkeypatch,
                                                               tmp_path):
        agent = self._agent(monkeypatch, tmp_path)
        limit = int(agent.limits_engine.limits.max_open_positions)
        for i in range(limit):
            _paper_position(agent.storage, market_id=f"M-full-{i}", size=1.0)
        called = {}

        async def _fake_cycle(target_per_venue=200, max_trades=3):
            called["max_trades"] = max_trades
            return {}

        monkeypatch.setattr(agent, "run_cycle", _fake_cycle)
        import asyncio

        asyncio.run(agent.run_round())
        assert called["max_trades"] == 0, (
            "every slot the rules allow is already held - opening another "
            "position is exactly what the ceiling exists to prevent"
        )

    def test_the_exposure_caps_are_seeded_with_the_paper_book(self, monkeypatch,
                                                              tmp_path):
        """
        `ExposureManager.can_open` enforces the position count and the category,
        correlated and total ceilings. It was consulted with an empty list every
        cycle, so those ceilings were enforced against zero - survivable while a
        cycle opened three trades, not survivable when a round fills the book.
        """
        agent = self._agent(monkeypatch, tmp_path)
        for i in range(3):
            _paper_position(agent.storage, market_id=f"M-{i}", size=3.0,
                            category="politics")
        _paper_position(agent.storage, market_id="M-live", size=3.0,
                        status="open", category="politics")
        # The live one above is a paper row unless the mode says otherwise.
        agent.storage.conn.execute(
            "UPDATE trades SET execution_mode='live' WHERE market_id='M-live'")
        agent.storage.conn.commit()

        seed = agent._seed_exposure_manager(_ledger(agent.storage))

        assert seed["account"] == "paper" and seed["readable"] is True
        assert seed["positions"] == 3, "the live position is not the paper book"
        assert seed["book_value_usd"] == pytest.approx(9.0)
        assert len(agent.exposure_manager.positions) == 3
        assert agent.exposure_manager.bankroll == pytest.approx(
            agent.storage.get_paper_bankroll())

    def test_the_caps_can_refuse_once_the_book_is_seeded(self, monkeypatch,
                                                        tmp_path):
        """
        $9 already held in one category on a $50 account is 18% > the 15%
        ceiling. Before the seed, `can_open` saw nothing and allowed it.
        """
        agent = self._agent(monkeypatch, tmp_path)
        for i in range(3):
            _paper_position(agent.storage, market_id=f"M-{i}", size=3.0,
                            category="politics")
        agent._seed_exposure_manager(_ledger(agent.storage))

        allowed, reason = agent.exposure_manager.can_open(
            market_id="M-new", amount_usd=3.0, category="politics",
            correlation_group="politics", venue="polymarket")
        assert allowed is False and "cap" in reason.lower()

    def test_a_position_with_no_recorded_category_is_counted_and_disclosed(
            self, monkeypatch, tmp_path):
        """
        Membership cannot be reconstructed for a row written before the column
        existed. It counts against the count and total ceilings, not against a
        category it may not belong to, and the report says how many those are.
        """
        agent = self._agent(monkeypatch, tmp_path)
        _paper_position(agent.storage, market_id="M-old", size=3.0,
                        category="politics")
        agent.storage.conn.execute(
            "UPDATE trades SET category=NULL, correlation_group=NULL "
            "WHERE market_id='M-old'")
        agent.storage.conn.commit()

        seed = agent._seed_exposure_manager(_ledger(agent.storage))
        assert seed["positions"] == 1
        assert seed["unknown_category"] == 1
        assert agent.exposure_manager.positions[0]["category"] == "unknown:M-old"

    def test_an_unreadable_book_is_reported_not_treated_as_empty(self, monkeypatch,
                                                                 tmp_path):
        agent = self._agent(monkeypatch, tmp_path)

        def _boom():
            raise RuntimeError("database is locked")

        monkeypatch.setattr(agent.storage, "get_open_positions", _boom)
        seed = agent._seed_exposure_manager(_ledger(agent.storage))
        assert seed["readable"] is False
        assert "could not be read" in seed["warning"]
        assert agent._exposure_account is None, (
            "nothing may be added to an exposure view that was never loaded"
        )

    def test_the_trade_row_remembers_which_ceiling_it_counts_against(self, tmp_path):
        """
        `category` was stored; the correlated group was not. Without it the 20%
        ceiling cannot be re-checked on any later round.
        """
        storage = _storage(tmp_path)
        _paper_position(storage, market_id="M-cat", category="sports",
                        )
        row = storage.conn.execute(
            "SELECT category, correlation_group FROM trades WHERE market_id='M-cat'"
        ).fetchone()
        assert row["category"] == "sports"
        assert row["correlation_group"] == "sports"


# ===========================================================================
# 5. the console: the button runs a round and shows the bankroll
# ===========================================================================

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    import importlib
    import src.ptai.ui.console as console
    importlib.reload(console)
    return TestClient(console.app)


class TestTheConsoleShowsTheRound:
    """
    The round card, through the routes that actually paint it.

    V50 changed WHERE a pressed round runs. It used to run inside the web handler
    (the handler built its own engine to do it), and the button's HTTP reply
    carried the result. Now the console HOSTS the agent: the button wakes that one
    loop, the round runs there, and the page gets the figures from the rounds
    route - the same route the card polls. These tests follow the data to where
    the page now reads it, and pin the button's reply at the same time.
    """

    @pytest.fixture()
    def hosted(self, client, monkeypatch, tmp_path):
        """The console as `run_ptai.bat` leaves it: this process runs the agent."""
        import importlib
        from src.ptai.agent.v3_loop import TradingAgentV3
        import src.ptai.ui.console as console
        console = importlib.import_module("src.ptai.ui.console")
        engine = TradingAgentV3(country_code="UG", dry_run=True)
        monkeypatch.setattr(console, "_agent", lambda: engine)
        monkeypatch.setattr(console, "_controller_agent_started", lambda: True)
        monkeypatch.setattr(engine, "request_immediate_cycle",
                            lambda: True, raising=False)
        return client, engine

    @staticmethod
    def _run_a_round(engine):
        import asyncio
        return asyncio.run(engine.run_round())

    def test_a_fresh_install_reports_no_round_rather_than_a_flat_one(self, client):
        body = client.get("/api/console/rounds").json()
        assert body["rounds"] == []
        assert body["summary"]["net_usd"] is None
        assert "no completed round" in body["summary"]["note"]

    def test_the_button_asks_the_running_agent_instead_of_building_one(self,
                                                                      hosted):
        """
        One press, one engine. The reply says what happened (a round was asked
        for) and does NOT contain a round it never ran.
        """
        client, engine = hosted
        beaten = {"n": 0}

        def _wake():
            beaten["n"] += 1
            return True

        import src.ptai.ui.console as console
        console._agent().request_immediate_cycle = _wake
        response = client.post("/api/console/run-cycle", json={"mode": "paper"})
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "requested"
        assert beaten["n"] == 1
        assert "round" not in body
        assert "watch the log" in body["message"]

    def test_the_round_card_gets_the_bankroll_from_the_rounds_route(self, hosted):
        """
        The operator's checklist, on the payload the card renders: a round with an
        account at both ends and a net that equals the difference.
        """
        client, engine = hosted
        self._run_a_round(engine)
        round_ = client.get("/api/console/rounds").json()["rounds"][0]
        assert round_["number"] >= 1
        assert round_["account"] == "paper"
        assert round_["equity_start"] is not None
        assert round_["equity_end"] is not None
        assert round_["net_usd"] == pytest.approx(
            round_["equity_end"] - round_["equity_start"], abs=0.011)
        assert round_["verdict"] in ("up", "down", "flat")
        assert "duration_seconds" in round_ and round_["duration_seconds"] >= 0
        for key in ("markets_discovered", "markets_screened", "markets_researched",
                    "markets_priced", "positions_opened", "positions_held",
                    "staked_usd"):
            assert key in round_, key

    def test_the_rounds_route_also_answers_with_the_history(self, hosted):
        client, engine = hosted
        self._run_a_round(engine)
        body = client.get("/api/console/rounds").json()
        assert len(body["rounds"]) >= 1
        assert body["summary"]["scored"] >= 1
        assert body["rounds"][0]["number"] == body["rounds"][0]["number"]

    def test_the_completed_round_survives_the_next_poll(self, hosted):
        client, engine = hosted
        self._run_a_round(engine)
        first = client.get("/api/console/rounds").json()
        polled = client.get("/api/console/rounds").json()
        assert polled["rounds"][0]["number"] == first["rounds"][0]["number"]
        assert polled["summary"]["scored"] == 1

    def test_the_page_has_a_round_card_and_a_button_that_asks_for_a_round(
            self, client):
        html = client.get("/").text
        assert "Run a round now" in html
        assert 'id="round"' in html
        assert "what the bankroll did" in html
        assert "loadRounds()" in html
        assert "showRound(" in html
        assert "no figure" in html, (
            "a round with no result must be rendered as no figure, not as zero"
        )

    def test_the_button_never_claims_a_round_it_did_not_run(self, hosted):
        """
        The reply to a press is about the REQUEST. The round's own mode and figures
        arrive on the rounds route, from the engine that ran it - so nothing can
        report a mode that was never executed.
        """
        client, engine = hosted
        body = client.post("/api/console/run-cycle",
                           json={"mode": "paper"}).json()
        assert "round" not in body and "mode" not in body
        self._run_a_round(engine)
        round_ = client.get("/api/console/rounds").json()["rounds"][0]
        assert round_["mode"] == "paper"


# ===========================================================================
# 6. the whole round on a stub venue: research, pick, bet, bankroll
# ===========================================================================

class TestAWholeRoundEndsWithABankroll:
    """
    The operator's checklist, end to end, on the real production path.

    Only the DISCOVERY of the edge is injected (the same stub harness the
    full-cycle tests use); screening, sizing, the risk gates, the paper fill and
    the marked result are the code that runs on their machine. The stub's book is
    0.54 / 0.56, so a paper buy at the ask is immediately worth the 0.55 mid -
    a small modelled loss, which is exactly the kind of figure the round now has
    to be able to report.
    """

    def _fill_in_paper(self, adapter, monkeypatch):
        """
        Fill the way the real adapters fill in paper: by walking the book.

        The stub's own dry-run answer is a refusal, which is right for a venue
        with no order path - but Polymarket and Kalshi both simulate a fill
        against the real ladder, and that is what makes a paper round measurable.
        Same PaperBroker, same book, no order sent.
        """
        async def _place(opportunity, max_spend_usd, max_price):
            from src.ptai.execution.paper_broker import PaperBroker
            from src.ptai.markets.mechanics import MarketMechanics

            mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0,
                                        source="clob_market_info", is_real=True)
            book = await adapter.get_orderbook(opportunity.market)
            limit = round(float(max_price), 2)
            fill = PaperBroker(mechanics=mechanics).simulate(
                book, "BUY", float(max_spend_usd), limit_price=limit,
                mechanics=mechanics, book_source="orderbook")
            return {
                "status": "paper", "is_real": False, "simulated": True,
                "venue_id": adapter.venue_id,
                "market_id": opportunity.market.id,
                "price": fill.avg_price or limit,
                "size": fill.filled_shares,
                "simulated_filled_usd": fill.filled_usd,
                "filled_usd": fill.filled_usd,
                "filled_price": fill.avg_price,
                "fees_usd": fill.fee_usd,
                "paper_fill": fill.to_dict(),
                "reason": fill.reason,
            }

        monkeypatch.setattr(adapter, "place_order", _place, raising=False)

    def _run(self, tmp_path, monkeypatch):
        from tests.test_full_cycle_from_discovery_to_allocation import (
            _force_qualified, _inject_opportunity, build_agent)

        agent, adapter = build_agent(tmp_path, dry_run=True)
        self._fill_in_paper(adapter, monkeypatch)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        import asyncio
        return agent, adapter, asyncio.run(agent.run_round(target_per_venue=5))

    def test_the_round_opens_the_position_and_ends_with_a_number(self, tmp_path,
                                                                 monkeypatch):
        agent, adapter, result = self._run(tmp_path, monkeypatch)
        round_ = result["round"]

        assert result["execution"], "the round proposed nothing to execute"
        assert result["execution"][0]["position_recorded"] is True, (
            result["execution"][0].get("position_reason"))
        assert round_["positions_opened"] == 1
        assert round_["staked_usd"] > 0
        assert round_["equity_start"] == pytest.approx(50.0, abs=0.01)
        assert round_["equity_end"] is not None
        assert round_["net_usd"] is not None, (
            "a completed round must end with a bankroll figure")
        assert round_["verdict"] in ("up", "down", "flat")
        assert round_["duration_seconds"] >= 0
        # A paper buy at the ask against a 0.54/0.56 book is worth less than it
        # paid the moment it fills. If the round reported a rise here, the mark
        # would be the request price rather than the market's.
        assert round_["equity_end"] <= round_["equity_start"] + 0.01
        assert round_["realised_pnl"] == pytest.approx(0.0)

    def test_the_round_says_which_half_of_the_move_it_is(self, tmp_path,
                                                        monkeypatch):
        _agent, _adapter, result = self._run(tmp_path, monkeypatch)
        round_ = result["round"]
        assert round_["open_positions_marked"] == 1, (
            "the position was not marked to the book the round read")
        assert round_["open_positions_unmarked"] == 0
        assert round_["positions_held"] == 1

    def test_the_round_counts_each_stage_it_went_through(self, tmp_path,
                                                         monkeypatch):
        _agent, _adapter, result = self._run(tmp_path, monkeypatch)
        round_ = result["round"]
        assert round_["markets_discovered"] >= 1
        assert round_["markets_screened"] >= 1
        assert round_["markets_priced"] >= 1
        assert round_["candidates"] >= 1

    def test_the_exposure_view_grew_with_the_position_it_opened(self, tmp_path,
                                                               monkeypatch):
        """
        The caps must see the positions THIS round opened, not only the ones it
        started with. Without this the next trade in the same round is sized as
        though the previous one did not exist.
        """
        agent, _adapter, result = self._run(tmp_path, monkeypatch)
        assert result["exposure"]["readable"] is True
        assert len(agent.exposure_manager.positions) == 1
        assert agent.exposure_manager.positions[0]["market_id"] == "STUB-M1"

    def test_the_round_is_written_to_the_record_the_console_reads(self, tmp_path,
                                                                  monkeypatch):
        agent, _adapter, result = self._run(tmp_path, monkeypatch)
        rows = load_rounds(agent.storage)
        assert rows and rows[0]["number"] == result["round"]["number"]
        assert rows[0]["net_usd"] == result["round"]["net_usd"]

    def test_the_next_round_measures_the_book_against_the_mark_it_left(self,
                                                                      tmp_path,
                                                                      monkeypatch):
        """
        The opening book of round 2 uses the marks stored at the end of round 1,
        so an unrealised gain already reported is not reported again. Without
        that, every round would re-earn the same move.
        """
        agent, _adapter, result = self._run(tmp_path, monkeypatch)
        marks = agent._round_marks()
        assert marks, "the round stored no marks, so the next opening book is blind"
        assert agent._round_report is not None
        assert agent._round_open_ledger is not None
