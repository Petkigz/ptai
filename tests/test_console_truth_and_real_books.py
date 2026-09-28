"""
The console must not say STOPPED while the round it just started is running.

The operator's report, 2026-09-28: "when i press run a round it still says
styopped while its showing that its running ; logz". The pasted log was the bug
report, and it contained three separate faults, each of which produced that same
sentence on the screen:

  1. `parse_price_size` was called on four lines of the CLOB book reader and
     defined nowhere, so EVERY real book raised NameError, the fallback caught
     it, and 200 markets in a row were reported as
     "ESTIMATED ... NOT REAL". Real depth was in hand and the system said it had
     an estimate - the one thing an estimate must never be able to do.
  2. The liveness verdict read a heartbeat that is written ONCE at the start of
     a cycle and ignored `agent.phase`, which is rewritten for every market the
     cycle touches. Any cycle longer than the window was reported STOPPED while
     the page printed the live phase line underneath it.
  3. The run-cycle reply carried the cycle's own result under "detail", which
     holds objects (`CombinatorialGroup`, `Market`, enums). FastAPI raised
     inside the response, so a round that RAN returned HTTP 500: the page got
     an error box instead of a bankroll, and never re-read the agent state.

Two more undefined names came out of the same sweep - `logger` inside the
settlement guard, and `MarketMechanics` used as a return annotation while only
imported inside the method body - and they are pinned here too, because a settler
that raises must return UNSETTLEABLE rather than raise NameError over a stake.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.storage.db import Storage
from src.ptai.venues.polymarket_adapter import normalise_book, parse_price_size


def _when(**delta) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


# ---------------------------------------------------------------------------
# 1. a real book is read as a real book
# ---------------------------------------------------------------------------

class TestOneParserForBookLevels:
    def test_both_shapes_the_venue_uses_are_read(self):
        assert parse_price_size({"price": "0.42", "size": "120"}) == (0.42, 120.0)
        assert parse_price_size(["0.55", "7"]) == (0.55, 7.0)
        assert parse_price_size((0.3, 2)) == (0.3, 2.0)

    def test_a_level_that_is_not_a_quote_reads_as_zero_and_never_raises(self):
        """
        Total on purpose. This function used to not exist, and the NameError it
        raised was caught by the fallback's `except` - which is exactly how a
        measured book became an estimate. A parser that raises here cannot be
        allowed to exist again.
        """
        for junk in ("junk", None, {}, [], [0.5], {"price": None, "size": None},
                     {"size": 5}, float("nan")):
            price, size = parse_price_size(junk)
            assert isinstance(price, float) and isinstance(size, float)
            assert price == price, "a NaN price must not survive the parser"

    def test_normalise_book_reads_the_same_levels_the_depth_reader_does(self):
        book = normalise_book(
            {"asset_id": "T1", "market": "M1",
             "bids": [{"price": "0.41", "size": "100"},
                      {"price": "0.43", "size": "50"}],
             "asks": [[0.45, 10], {"price": 0.47, "size": 5}]},
            expected_token_id="T1", expected_condition_id="M1")

        assert book["validated"] is True
        # Best is the HIGHEST bid and the LOWEST ask, whatever order they came in.
        assert book["best_bid"] == pytest.approx(0.43)
        assert book["best_ask"] == pytest.approx(0.45)
        assert book["bids"][0] == (0.43, 50.0)
        assert book["asks"][0] == (0.45, 10.0)


class TestTheRealClobBookIsNotReportedAsAnEstimate:
    """
    The four broken lines live in the adapter's real read path, which no test
    exercised: every other test stubs `get_orderbook` itself, so 2120 passing
    tests could not see a NameError inside it. This one goes through the real
    adapter with a fake CLIENT, one layer lower.
    """

    def _adapter(self, raw_book):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        class _Client:
            def __init__(self, raw):
                self.raw = raw
                self.asked = []

            def get_orderbook(self, token_id):
                self.asked.append(token_id)
                return self.raw

        adapter = PolymarketAdapter(dry_run=True)
        adapter.client = _Client(raw_book)
        return adapter

    def _market(self):
        from src.ptai.markets.base import Market, MarketSource, Token

        return Market(
            id="MKT-1", source=MarketSource.POLYMARKET,
            question="Will the real book be read?", outcomes=["YES", "NO"],
            outcome_prices=[0.44, 0.56],
            tokens=[Token(token_id="T1", outcome="YES", price=0.44),
                    Token(token_id="T2", outcome="NO", price=0.56)],
            volume=100_000.0, volume_24h=50_000.0, liquidity=50_000.0,
            active=True, closed=False, slug="mkt-1", condition_id="M1",
            raw={"conditionId": "M1"})

    def test_a_real_book_is_used_as_a_real_book(self):
        adapter = self._adapter({
            "asset_id": "T1", "market": "M1",
            "bids": [{"price": "0.42", "size": "300"},
                     {"price": "0.43", "size": "200"}],
            "asks": [{"price": "0.46", "size": "150"},
                     {"price": "0.45", "size": "100"}],
        })

        book = asyncio.run(adapter.get_orderbook(self._market()))

        assert book["is_real"] is True, (
            "a real CLOB book was in hand and was reported as an estimate")
        assert book["source"] == "clob_real"
        assert book["bid"] == pytest.approx(0.43)
        assert book["ask"] == pytest.approx(0.45)
        # The sizes ARE the four lines that used to raise NameError: top-5 on
        # each side, and the top-10 totals the imbalance is computed from.
        assert book["bid_size"] == pytest.approx(500.0)
        assert book["ask_size"] == pytest.approx(250.0)
        assert book["depth"] == pytest.approx(750.0)
        assert book["total_bid_depth"] == pytest.approx(500.0)
        assert book["total_ask_depth"] == pytest.approx(250.0)
        assert book["imbalance"] == pytest.approx((500 - 250) / 750, abs=1e-9)
        assert book["executable_price_yes"] == pytest.approx(0.45)
        assert book["executable_price_no"] == pytest.approx(0.57)

    def test_a_crossed_book_is_still_refused_rather_than_priced(self):
        """
        The validation is untouched by this round: a book that fails a check is
        not this market's book, and the reply says so with `is_real: False`
        rather than quietly estimating from it.
        """
        adapter = self._adapter({
            "asset_id": "T1", "market": "M1",
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [{"price": "0.55", "size": "100"}],
        })

        book = asyncio.run(adapter.get_orderbook(self._market()))

        assert book["is_real"] is False
        assert book["validated"] is False
        assert book["spread"] is None
        assert book["executable"] is False
        assert "validation" in book["warning"].lower()


# ---------------------------------------------------------------------------
# 2. the pill: a fresh phase is evidence of life
# ---------------------------------------------------------------------------

class TestStoppedWhileRunning:
    def _store(self, tmp_path):
        return Storage(db_path=str(tmp_path / "truth.db"))

    def test_a_long_cycle_is_working_not_stopped(self, tmp_path):
        """
        The heartbeat is written once, at cycle start. On a cycle that runs
        longer than the window - the operator's own log has a 200-market scan -
        it ages out while the phase keeps being rewritten. The old verdict read
        only the heartbeat and said STOPPED, under a live phase line.
        """
        from src.ptai.operator_view import agent_state

        store = self._store(tmp_path)
        store.set_state("agent.heartbeat", json.dumps(
            {"at": _when(minutes=30), "status": "cycle_start"}))
        store.set_state("agent.phase", json.dumps(
            {"at": _when(seconds=4), "phase": "evaluating",
             "label": "pricing what it found",
             "detail": "market 3 of 25: Will the thing happen?"}))

        state = agent_state(store)

        assert state["running"] is True
        assert state["state"] == "working", (
            "a phase written 4s ago must not be reported as STOPPED")
        assert state["phase_is_live_evidence"] is True
        assert "market 3 of 25" in state["doing"]
        assert "written 4s ago" in state["evidence"]
        assert "longer than the heartbeat window" in state["evidence"]

    @pytest.mark.parametrize("phase", ["scanning", "screening", "evaluating",
                                       "executing"])
    def test_every_working_phase_counts(self, tmp_path, phase):
        from src.ptai.operator_view import agent_state

        store = self._store(tmp_path)
        store.set_state("agent.phase", json.dumps(
            {"at": _when(seconds=2), "phase": phase, "label": phase,
             "detail": f"doing {phase}"}))

        assert agent_state(store)["state"] == "working"

    @pytest.mark.parametrize("phase", ["cycle_complete", "no_markets",
                                       "blocked", "sleeping"])
    def test_a_resting_phase_is_not_evidence_of_work(self, tmp_path, phase):
        """
        A process that wrote "sleeping" and then died must not look alive for
        another window. Resting phases are covered by the scan row a completed
        cycle writes - not by pretending the note itself is a pulse.
        """
        from src.ptai.operator_view import agent_state

        store = self._store(tmp_path)
        store.set_state("agent.heartbeat", json.dumps(
            {"at": _when(minutes=40), "status": "cycle_start"}))
        store.set_state("agent.phase", json.dumps(
            {"at": _when(seconds=3), "phase": phase, "label": phase,
             "detail": phase}))

        state = agent_state(store)

        assert state["running"] is False
        assert state["state"] == "not_running"
        # ...and the screen explains the apparent contradiction instead of
        # leaving "STOPPED" sitting above a three-second-old phase line.
        assert "not work in progress" in state["evidence"], (
            "a fresh resting phase must be named as such, not left to look "
            "like the agent is being called dead for no reason")

    def test_a_stale_phase_does_not_pretend_the_agent_is_alive(self, tmp_path):
        """
        The same rule that made the heartbeat expire applies to the phase: it
        buys a working cycle time, not immortality.
        """
        from src.ptai.operator_view import agent_state

        store = self._store(tmp_path)
        store.set_state("agent.heartbeat", json.dumps(
            {"at": _when(hours=3), "status": "cycle_start"}))
        store.set_state("agent.phase", json.dumps(
            {"at": _when(hours=2), "phase": "evaluating",
             "label": "pricing what it found", "detail": "market 3 of 25"}))

        state = agent_state(store)

        assert state["running"] is False
        assert state["state"] == "not_running"
        assert state["phase_is_live_evidence"] is False

    def test_a_fresh_scan_row_still_wins(self, tmp_path):
        """The scan row is the strongest evidence and its meaning is unchanged."""
        from src.ptai.operator_view import agent_state

        store = self._store(tmp_path)
        store.log_scan(markets_scanned=200, opportunities_found=3, avg_edge=0.04,
                       execution_time=37.0, bankroll=50.0)
        store.conn.execute("UPDATE market_scans SET timestamp=?", (_when(seconds=5),))
        store.conn.commit()

        state = agent_state(store)

        assert state["state"] == "running"
        assert "last completed cycle" in state["evidence"]


# ---------------------------------------------------------------------------
# 3. a settler that raises still settles
# ---------------------------------------------------------------------------

class TestASettlerThatRaises:
    def test_a_raising_settler_returns_unsettleable_not_a_nameerror(self,
                                                                    monkeypatch):
        """
        The guard existed to turn an exception into UNSETTLEABLE (the stake
        comes back). Without its logger import the guard itself raised
        NameError, so the exception escaped mid-settlement and the position had
        no verdict at all.
        """
        from src.ptai.betting import market_types as mt

        spec = mt.get_spec("h2h")
        assert spec is not None and spec.settle is not None

        def _boom(facts, outcome, line):
            raise RuntimeError("a malformed fact, as a feed would send")

        monkeypatch.setattr(spec, "settle", _boom)
        facts = mt.MatchFacts(home_team="A", away_team="B",
                              home_goals=2, away_goals=1)

        verdict = mt.settle_market("h2h", facts, "home")

        assert verdict == mt.Settlement.UNSETTLEABLE

    def test_the_ordinary_path_still_settles(self):
        """The guard must not have swallowed the working case."""
        from src.ptai.betting import market_types as mt

        facts = mt.MatchFacts(home_team="A", away_team="B",
                              home_goals=2, away_goals=1)

        assert mt.settle_market("h2h", facts, "home") == mt.Settlement.WIN
        assert mt.settle_market("h2h", facts, "away") == mt.Settlement.LOSE


# ---------------------------------------------------------------------------
# 4. the round's reply can always be sent
# ---------------------------------------------------------------------------

class TestTheRoundReplyIsAlwaysSendable:
    def test_objects_in_the_cycle_result_no_longer_500_the_reply(self):
        """
        The exact object whose presence turned a completed round into HTTP 500
        on the operator's machine.
        """
        from src.ptai.strategy.combinatorial import CombinatorialGroup

        group = CombinatorialGroup(
            group_id="g1", event_slug="mece", markets=[], sum_yes=2.55,
            is_exhaustive=True, is_exclusive=True,
            arbitrage_type="sell_all_yes_buy_all_no",
            estimated_profit_pct=0.389, cost=3.45, payout=5.0,
            should_trade=True, reasoning="6 markets sum YES 2.550")

        from src.ptai.ui.console import _json_safe

        payload = _json_safe({"detail": {"alpha_scan": {"opps": [group]}},
                              "round": {"net_usd": 0.32}})

        sent = json.dumps(payload)  # what JSONResponse does
        assert "sell_all_yes_buy_all_no" in sent
        assert payload["round"]["net_usd"] == 0.32

    def test_the_page_is_never_sent_a_nan(self):
        """
        json.dumps allows NaN by default, but the page parses with JSON.parse,
        which does not - one NaN in a payload and the whole console stops
        updating, which is the same failure by another route.
        """
        from src.ptai.ui.console import _json_safe

        payload = _json_safe({"price": float("nan"), "spread": float("inf"),
                              "ok": 0.25})

        assert payload == {"price": None, "spread": None, "ok": 0.25}
        assert "NaN" not in json.dumps(payload)

    def test_the_endpoint_answers_a_round_that_ran(self, tmp_path, monkeypatch):
        """
        End to end: a round whose result contains live objects must come back
        200 with the round's bankroll on it. This is the operator pressing the
        button and getting an answer instead of an error box.
        """
        import importlib

        from fastapi.testclient import TestClient

        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        console = importlib.import_module("src.ptai.ui.console")
        storage = Storage(db_path=str(tmp_path / "console.db"))
        storage.set_bankroll(50.0)
        storage.set_paper_bankroll(50.0)

        from src.ptai.strategy.combinatorial import CombinatorialGroup

        class _Agent:
            dry_run = True

            async def run_round(self, *a, **k):
                return {
                    "status": "complete",
                    "execution": [{"position_recorded": True}],
                    "alpha_scan": {"combinatorial": {"opps": [CombinatorialGroup(
                        group_id="g", event_slug="e", markets=[], sum_yes=2.5,
                        is_exhaustive=True, is_exclusive=True,
                        arbitrage_type="sell_all_yes_buy_all_no",
                        estimated_profit_pct=0.3, cost=3.0, payout=5.0,
                        should_trade=True, reasoning="why not")]}},
                    "round": {"number": 1, "net_usd": 0.32, "verdict": "up",
                              "account": {"start_usd": 50.0, "end_usd": 50.32}},
                }

            def round_history(self, limit=10):
                return {"rounds": [], "summary": {}}

        monkeypatch.setattr(console, "_agent", lambda: _Agent())
        monkeypatch.setattr(console, "get_storage", lambda: storage)

        client = TestClient(console.app)
        response = client.post("/api/console/run-cycle", json={"mode": "paper"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["round"]["net_usd"] == 0.32
        assert body["executed"] == 1
        # The objects are still described, as text, rather than dropped.
        assert "sell_all_yes_buy_all_no" in json.dumps(body)
