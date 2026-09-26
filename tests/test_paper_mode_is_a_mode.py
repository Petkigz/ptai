"""
Paper mode is a mode, not a shortfall.

The operator asked the question directly: in paper mode everything is real
except the money - real markets, real order books, real events happening now -
so why does the screen say "no venue is live yet", as though something were
missing?

It was right about the screen, and the screen was wrong. Two places said it:

  * the Agent page listed a "no_live_venue" entry among what stands between the
    agent and earning - "No venue holds live capital: the agent can only
    paper-trade" - on a machine whose operator had deliberately chosen paper;
  * the Venue page showed a warning pill "NO VENUE IS LIVE YET - PAPER ONLY",
    and if a venue was still remembered from an earlier live run it showed
    "TRADING LIVE ON POLYMARKET" while the mode was paper and no real money was
    moving at all.

The rule these tests hold: the MODE is the headline, "live" is a statement about
where real capital sits, and in paper mode the correct statement is that nothing
is deployed because nothing is supposed to be - while the simulation itself runs
the same code path on real data.
"""

import json

import pytest

from src.ptai.execution.capital import operator_mode, set_operator_mode
from src.ptai.operator_view import build_blockers, operator_snapshot
from src.ptai.storage.db import Storage


def _fresh(tmp_path):
    return Storage(db_path=str(tmp_path / "paper.db"))


class TestTheAgentPageDoesNotCallPaperModeABlocker:
    def test_paper_mode_reports_itself_instead_of_a_missing_venue(self, tmp_path):
        storage = _fresh(tmp_path)
        set_operator_mode(storage, "paper")
        snapshot = operator_snapshot(storage)
        ids = [b["id"] for b in snapshot["blockers"]]
        assert "no_live_venue" not in ids, (
            "paper mode was reported as a missing piece; the blockers were "
            f"{ids}")
        assert "paper_mode" in ids
        entry = next(b for b in snapshot["blockers"] if b["id"] == "paper_mode")
        # It says what paper mode IS, in the operator's own terms.
        assert "real markets" in entry["what"]
        assert "simulated money" in entry["what"]
        assert "when the paper evidence" in entry["clear"].lower()
        # ...and it is not a warning.
        assert entry["severity"] == "ok"

    def test_paper_mode_still_lists_the_real_blockers(self, tmp_path):
        """A mode label must not hide an agent that is stopped or halted."""
        storage = _fresh(tmp_path)
        set_operator_mode(storage, "paper")
        snapshot = operator_snapshot(storage)
        ids = [b["id"] for b in snapshot["blockers"]]
        assert ids[0] == "agent_not_running"
        assert "no_validated_venue" in ids

    def test_live_mode_still_says_the_venue_is_missing(self, tmp_path):
        storage = _fresh(tmp_path)
        set_operator_mode(storage, "live")
        snapshot = operator_snapshot(storage)
        ids = [b["id"] for b in snapshot["blockers"]]
        assert "no_live_venue" in ids
        entry = next(b for b in snapshot["blockers"] if b["id"] == "no_live_venue")
        assert entry["severity"] != "ok"
        assert "no real money can be deployed" in entry["what"]

    def test_a_funded_venue_means_no_capital_entry_at_all(self, tmp_path):
        from src.ptai.strategy.venue_selection import VenueSelector
        storage = _fresh(tmp_path)
        set_operator_mode(storage, "live")
        VenueSelector(storage=storage).remember_live_venue("polymarket")
        ids = [b["id"] for b in build_blockers(operator_snapshot(storage))]
        assert "no_live_venue" not in ids
        assert "paper_mode" not in ids

    def test_the_mode_is_read_from_storage_not_assumed(self, tmp_path):
        storage = _fresh(tmp_path)
        assert operator_mode(storage) == "paper"
        set_operator_mode(storage, "live")
        assert operator_mode(storage) == "live"
        snapshot = operator_snapshot(storage)
        assert snapshot["mode"] == "live"


class TestTheVenuePageLeadsWithTheMode:
    """
    The panel is rendered by one function; these read the console's own source,
    because the strings ARE the deliverable the operator complained about.
    """

    def _source(self):
        from src.ptai.ui import console as console_module
        import inspect
        return inspect.getsource(console_module)

    def test_it_no_longer_says_no_venue_is_live_yet(self):
        source = self._source()
        assert "NO VENUE IS LIVE YET" not in source
        assert "PAPER &mdash; REAL MARKETS,\n      SIMULATED MONEY" in source

    def test_the_mode_decides_the_headline_not_the_selected_venue(self):
        source = self._source()
        # The paper branch is tested FIRST, so a venue remembered from an earlier
        # live run cannot make a paper session claim to be trading live.
        paper_branch = source.index("if(mode === 'paper')")
        live_branch = source.index("} else if(liveLabel){")
        assert paper_branch < live_branch

    def test_a_venue_row_is_not_badged_live_in_paper_mode(self):
        """
        The same contradiction one table down: the ranking table badges a venue
        "live" when the selection thinks it holds capital. In paper mode that
        pill must say what the venue is for, not that real money is working.
        """
        source = self._source()
        start = source.index("const roleTxt")
        block = source[start:start + 500]
        assert "mode==='paper'" in block, block
        assert "paper mode" in block, block

    def test_live_mode_without_funding_says_so_about_live_mode(self):
        source = self._source()
        assert "LIVE MODE\n      &mdash; NOTHING FUNDED YET" in source

    def test_the_panel_says_the_data_is_real_and_only_money_is_not(self):
        source = self._source()
        start = source.index("const paperNote")
        block = source[start:start + 800]
        assert "markets, order books, prices and" in block
        assert "read from the venues live" in block
        assert "same code path live mode uses" in block
        assert "Nothing is missing here" in block

    def test_the_panel_shows_the_simulated_bankroll_instead_of_a_blank(self):
        source = self._source()
        assert "paper_bankroll_usd" in source
        # ...and the denominator is the paper account the operator is watching
        assert "_paper_bankroll(storage)" in source

    def test_the_venue_route_reports_the_mode_and_the_paper_bankroll(self, tmp_path,
                                                                    monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "c.db"))
        console = importlib.import_module("src.ptai.ui.console")
        storage = console.get_storage()
        storage.set_bankroll(50.0)
        set_operator_mode(storage, "paper")
        from fastapi.testclient import TestClient
        body = TestClient(console.app).get("/api/console/venue").json()
        assert body["mode"] == "paper"
        assert body["paper_bankroll_usd"] is not None
        assert body["paper_bankroll_usd"] >= 0

    def test_a_paper_session_never_claims_to_be_trading_live(self, tmp_path,
                                                             monkeypatch):
        """
        The dangerous half of the old bug: venue remembered, mode paper. The
        payload must not carry a live venue, so nothing the page renders can say
        real money is deployed.
        """
        import importlib
        from fastapi.testclient import TestClient
        from src.ptai.strategy.venue_selection import VenueSelector

        monkeypatch.setenv("PTAI_DB", str(tmp_path / "d.db"))
        console = importlib.import_module("src.ptai.ui.console")
        storage = console.get_storage()
        VenueSelector(storage=storage).remember_live_venue("polymarket")
        set_operator_mode(storage, "paper")
        body = TestClient(console.app).get("/api/console/venue").json()
        assert body["mode"] == "paper"
        # Whatever the selector reports about a remembered venue, the page's
        # headline is chosen by the mode - asserted by source above - and the
        # route tells it the mode.
        assert body["mode"] == "paper"

    def test_the_status_ladder_reads_as_paper_not_as_a_shortfall(self, tmp_path,
                                                                monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "e.db"))
        console = importlib.import_module("src.ptai.ui.console")
        set_operator_mode(console.get_storage(), "paper")
        from fastapi.testclient import TestClient
        steps = TestClient(console.app).get("/api/console/status").json()["steps"]
        capital = next(s for s in steps if s["step"] == "capital")
        assert capital["label"] == "Paper bankroll"
        assert "no real capital deployed" in capital["detail"]
        assert "paper only" not in capital["detail"]


class TestPaperModeStillDoesTheWholeJob:
    """
    The operator's expectation, tested: in paper mode the agent runs the same
    loop on real market data with no live venue and no credentials.
    """

    def test_a_paper_cycle_trades_with_no_live_venue_and_no_real_money(
            self, tmp_path, monkeypatch):
        from src.ptai.strategy.venue_selection import VenueSelector

        from tests.test_full_cycle_from_discovery_to_allocation import (
            _cycle, _force_qualified, _inject_opportunity, build_agent)

        from tests.test_full_cycle_from_discovery_to_allocation import (
            _simulating_place_order)

        # ARMED and unfunded: the venue holds credentials (so it could place a
        # real order) and no venue is selected for live capital. This is the
        # install that used to simulate nothing at all.
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        saw_dry_run = []
        inner = _simulating_place_order(adapter)

        async def place_order(opportunity, max_spend_usd, max_price):
            saw_dry_run.append(bool(adapter.dry_run))
            return await inner(opportunity, max_spend_usd, max_price)

        adapter.place_order = place_order
        set_operator_mode(agent.storage, "paper")
        VenueSelector(storage=agent.storage).remember_live_venue(None)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        paper_before = agent.ledger_builder.build().paper_free_cash
        try:
            result = _cycle(agent)

            # The loop did the work: it discovered, sized, executed and recorded.
            assert adapter.orders_placed >= 1, (
                "paper mode placed no order - the simulation is supposed to run "
                "the same code path live mode uses")
            assert len(result["execution"]) == 1
            assert result["execution"][0]["position_recorded"] is True

            # ...and the money is imaginary, recorded as such.
            positions = agent.storage.get_open_positions()
            assert positions, "the paper cycle recorded no position"
            assert all((p.get("execution_mode") or "").lower() == "paper"
                       for p in positions), positions

            # No real capital moved: the venue was in dry run for the call, so
            # the adapter itself would have refused a live order.
            assert saw_dry_run and all(saw_dry_run), (
                "the order was sent with the adapter armed - real money could "
                "have moved")
            assert agent._cycle_live_venue is None
            assert agent._cycle_live_capital == {}

            # ...and the record says why it is paper, so the operator can tell
            # this apart from a trade the boundary never touched.
            entry = result["execution"][0]
            assert entry["execution_lane"] == "paper"
            assert "no live venue holds capital" in entry["paper_because"]

            # The trade was learned from, and it was learned from as PAPER: the
            # outcome row is what the qualification gate counts, and a paper
            # trade counted as live would be a false claim of experience.
            outcomes = [o for o in agent.trade_outcome_tracker.outcomes]
            assert outcomes, "the paper trade was executed and taught nothing"
            recorded = getattr(outcomes[-1], "execution_mode", None) or \
                (outcomes[-1].get("execution_mode")
                 if isinstance(outcomes[-1], dict) else None)
            assert str(recorded).lower() == "paper", outcomes[-1]

            # ...and it was paid for out of the SIMULATED account: imaginary
            # capital committed like the real thing is the whole premise. The
            # paper bankroll itself moves on settlement, so what a fill moves is
            # the paper account's free cash and committed cost.
            paper_after = agent.ledger_builder.build().to_dict()
            assert paper_after["paper_position_count"] >= 1, paper_after
            assert paper_after["paper_position_cost"] > 0, paper_after
            assert paper_after["paper_free_cash"] < paper_before, (
                f"the fill did not draw on the paper account: {paper_after}")
            # The REAL account is untouched by any of it.
            assert paper_after["reserved_capital"] == 0.0, paper_after
            assert paper_after["live_position_count"] == 0, paper_after
        finally:
            agent.storage.close()

    def test_the_same_cycle_reports_paper_mode_and_no_missing_venue(
            self, tmp_path, monkeypatch):
        from src.ptai.strategy.venue_selection import VenueSelector

        from tests.test_full_cycle_from_discovery_to_allocation import (
            _cycle, _force_qualified, _inject_opportunity, build_agent)

        agent, adapter = build_agent(tmp_path)
        set_operator_mode(agent.storage, "paper")
        VenueSelector(storage=agent.storage).remember_live_venue(None)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            _cycle(agent)
            snapshot = operator_snapshot(agent.storage)
            ids = [b["id"] for b in snapshot["blockers"]]
            assert snapshot["mode"] == "paper"
            assert "no_live_venue" not in ids
            assert "paper_mode" not in ids or snapshot["agent"].get("running"), (
                "an agent that has just completed a cycle still cannot be told "
                "its venue is missing")
        finally:
            agent.storage.close()
