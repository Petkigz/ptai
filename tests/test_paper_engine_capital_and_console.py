"""
Paper mode, capital accounting, and the operator console.

The three things this file has to prove, because each one is a way the system
lies to its operator:

  1. Paper mode can LOSE. A simulation that fills every order at the price the
     agent wanted is not a simulation, and its equity curve is fiction.
  2. Capital cannot be counted twice. Positions and reservations are different
     things; a resting order has bought nothing and still holds the cash.
  3. The console refuses rather than warns. Live mode with nothing ready, a
     budget larger than the account, a balance that was never read - all refused.
"""

import asyncio
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.ptai.execution.capital import (
    CapitalLedger,
    FUNDING_ROUTES,
    VenueAccount,
    plan_for_budget,
)
from src.ptai.execution.paper_broker import PaperBroker, simulate_paper_fill
from src.ptai.execution.multi_venue_executor import ExecutionResult
from src.ptai.markets.mechanics import MarketMechanics
from src.ptai.storage.db import Storage


def _mechanics(tick="0.01", min_size=1.0, is_real=True):
    return MarketMechanics(tick_size=tick, min_order_size=min_size,
                           source="clob_market_info" if is_real else "assumed_default",
                           is_real=is_real)


BOOK = {
    "bids": [{"price": "0.54", "size": "300"}, {"price": "0.53", "size": "500"}],
    "asks": [{"price": "0.56", "size": "20"}, {"price": "0.57", "size": "40"},
             {"price": "0.60", "size": "500"}],
}
THIN = {"asks": [{"price": "0.56", "size": "5"}, {"price": "0.57", "size": "40"}]}


# ==========================================================================
# 1. paper mode is a simulation, not a wish
# ==========================================================================

class TestPaperModePricesAgainstTheBook:
    def test_a_market_buy_walks_the_ladder_and_pays_slippage(self):
        """
        $5 cannot fit in the $11.20 offered at the touch. Walking to the second
        level is the cost the old paper mode never charged.
        """
        fill = PaperBroker().simulate(THIN, "BUY", 5.0, mechanics=_mechanics())
        assert fill.filled_usd == pytest.approx(5.0, abs=0.01)
        assert fill.levels_consumed == 2
        assert fill.avg_price > 0.56
        assert fill.slippage_bps > 0, "walking the book must cost something"
        assert fill.worst_price == 0.57

    def test_depth_is_finite_so_a_fill_can_be_partial(self):
        """
        A $40 order against a book holding $14.05. It gets $14.05 and the rest
        rests. The old code filled the whole $40.
        """
        shallow = {"asks": [{"price": "0.56", "size": "20"},
                            {"price": "0.57", "size": "5"}]}
        fill = PaperBroker().simulate(shallow, "BUY", 40.0, mechanics=_mechanics())
        assert fill.filled_usd == pytest.approx(14.05, abs=0.01)
        assert fill.is_partial
        assert fill.unfilled_usd == pytest.approx(25.95, abs=0.01)
        assert fill.would_rest

    def test_a_deep_book_fills_the_whole_order(self):
        """The other half of the same rule: depth is not a permanent penalty."""
        fill = PaperBroker().simulate(BOOK, "BUY", 40.0, mechanics=_mechanics())
        assert fill.filled_usd == pytest.approx(40.0, abs=0.01)
        assert fill.is_partial is False

    def test_a_limit_the_market_does_not_cross_fills_nothing(self):
        """
        The most important line in the file. A passive order does not fill on
        submission, and a simulator that fills it manufactures an edge.
        """
        fill = PaperBroker().simulate(BOOK, "BUY", 3.0, limit_price=0.40,
                                      mechanics=_mechanics())
        assert fill.filled_shares == 0.0
        assert fill.filled_usd == 0.0
        assert fill.would_rest is True
        assert fill.unfilled_usd == pytest.approx(3.0), (
            "an unfilled resting order must still account for the whole request"
        )

    def test_a_resting_order_fills_only_when_the_market_comes_to_it(self):
        later = {"asks": [{"price": "0.39", "size": "50"}],
                 "bids": [{"price": "0.38", "size": "100"}]}
        fill = PaperBroker().simulate_resting(0.40, "BUY", 3.0, later,
                                             mechanics=_mechanics())
        assert fill.filled_usd > 0
        assert fill.avg_price == pytest.approx(0.40), (
            "a passive order fills at its own price"
        )
        assert fill.optimistic is True, (
            "a resting fill without a queue model must be labelled optimistic"
        )

    def test_a_resting_order_the_market_never_reaches_stays_unfilled(self):
        fill = PaperBroker().simulate_resting(0.40, "BUY", 3.0, BOOK,
                                             mechanics=_mechanics())
        assert fill.filled_shares == 0.0
        assert fill.would_rest is True

    def test_fees_are_charged_on_the_notional(self):
        fill = PaperBroker(taker_fee_rate=0.02).simulate(BOOK, "BUY", 3.0,
                                                         mechanics=_mechanics())
        assert fill.fee_usd == pytest.approx(fill.filled_usd * 0.02, abs=1e-6)
        assert fill.total_cost_usd == pytest.approx(fill.filled_usd * 1.02, abs=1e-6)

    def test_a_fill_with_no_book_is_not_evidence(self):
        """
        The honesty rule. A fill priced from an assumed book is a guess wearing
        a number, and it says so.
        """
        fill = PaperBroker().simulate(None, "BUY", 3.0, mechanics=_mechanics(),
                                      book_source="assumed_default")
        assert fill.is_real is False
        assert fill.warnings, "an unbacked simulation must carry its warning"

    def test_a_fill_below_the_venue_minimum_is_flagged(self):
        """
        The venue would reject this order. A paper engine that accepts it books
        a trade that cannot exist.
        """
        fill = PaperBroker().simulate({"asks": [{"price": "0.56", "size": "1"}]},
                                      "BUY", 0.5, mechanics=_mechanics(min_size=5.0))
        assert any("minimum" in w for w in fill.warnings)

    def test_the_executor_keeps_the_simulated_size(self):
        """
        A simulated fill that reaches the ledger as zeros sends the loop back to
        the REQUESTED size and price - which is exactly how paper mode filled
        every order at the price the agent wanted.
        """
        # Deliberately shallow, so the simulated fill is a PARTIAL one and
        # cannot be confused with the requested amount.
        executor_book = {"asks": [{"price": "0.56", "size": "5"},
                                  {"price": "0.57", "size": "2"}]}
        fill = PaperBroker().simulate(executor_book, "BUY", 5.0,
                                      mechanics=_mechanics())
        assert fill.is_partial
        result = {
            "status": "paper", "is_real": False, "simulated": True,
            "filled_usd": fill.filled_usd, "simulated_filled_usd": fill.filled_usd,
            "filled_price": fill.avg_price, "price": fill.avg_price,
            "size_matched": fill.filled_shares, "paper_fill": fill.to_dict(),
        }
        from src.ptai.execution.multi_venue_executor import MultiVenueExecutor
        read = MultiVenueExecutor(registry=None, bankroll=50.0)._read_fill(
            result, requested_usd=5.0, requested_price=0.56)

        assert read["status"] == "dry_run", (
            "a simulated result must stay in the simulated family, so it is "
            "recorded as paper rather than as a live position"
        )
        assert read["filled_usd"] == pytest.approx(fill.filled_usd, abs=0.01)
        assert read["price"] == pytest.approx(fill.avg_price, abs=1e-6)

        execution = ExecutionResult(
            venue_id="polymarket", market_id="M1", status="dry_run",
            amount_usd=5.0, price=read["price"], fees_usd=0.0, gas_usd=0.0,
            latency_ms=1.0, reasoning="", filled_usd=read["filled_usd"],
            filled_price=read["price"], paper_fill=result["paper_fill"])
        assert execution.is_simulated
        assert execution.position_size_usd == pytest.approx(fill.filled_usd, abs=0.01)
        assert execution.position_size_usd != 5.0, (
            "the ledger was charged the REQUESTED amount for a partial paper fill"
        )


# ==========================================================================
# 2. capital is not counted twice
# ==========================================================================

def _storage():
    path = os.path.join(tempfile.mkdtemp(), "cap.db")
    return Storage(db_path=path)


class TestCapitalAccounting:
    def test_a_resting_order_reserves_cash_without_being_a_position(self):
        """
        The gap nothing else covers. An unfilled order is invisible to a position
        ledger, so without its own reservation the agent sizes the next trade
        against money the venue has already locked.
        """
        storage = _storage()
        storage.log_trade({"market_id": "M1", "venue_id": "polymarket",
                           "side": "YES", "position_size_usd": 3.0,
                           "market_price": 0.5, "data_mode": "live"})
        storage.upsert_order({"order_id": "0xr", "market_id": "M2",
                              "venue_id": "polymarket", "status": "submitted",
                              "requested_usd": 3.0, "matched_usd": 0.0})
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={"polymarket": 50.0},
            balances={"polymarket": {"available": True, "balance": 50.0}})
        account = plan.accounts[0]
        assert account.in_positions_usd == pytest.approx(3.0)
        assert account.reserved_usd == pytest.approx(3.0)
        assert account.available_usd == pytest.approx(44.0), (
            "both the position cost and the reservation must come out of the "
            "same pool, exactly once each"
        )
        assert account.committed_usd == pytest.approx(6.0)

    def test_paper_positions_do_not_consume_live_capital(self):
        """
        Two pools. A paper position must not eat live budget, and a live
        position must not hide behind paper ones.
        """
        storage = _storage()
        storage.log_trade({"market_id": "P1", "venue_id": "polymarket",
                           "side": "YES", "position_size_usd": 9.0,
                           "market_price": 0.5, "status": "paper",
                           "data_mode": "live_paper"})
        storage.log_trade({"market_id": "L1", "venue_id": "polymarket",
                           "side": "YES", "position_size_usd": 2.0,
                           "market_price": 0.5, "status": "open",
                           "data_mode": "live"})
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={"polymarket": 50.0},
            balances={"polymarket": {"available": True, "balance": 50.0}})
        account = plan.accounts[0]
        assert account.in_positions_usd == pytest.approx(2.0), (
            "the $9 paper position was charged to live capital"
        )
        assert account.available_usd == pytest.approx(48.0)

    def test_available_never_goes_negative(self):
        account = VenueAccount(venue_id="v", deposited_usd=10.0,
                               in_positions_usd=8.0, reserved_usd=5.0)
        assert account.available_usd == 0.0
        assert account.over_committed is True

    def test_an_unreadable_balance_is_not_funded(self):
        """
        The rule that stops an agent believing it has money because someone
        typed a number into a form.
        """
        storage = _storage()
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={"polymarket": 50.0}, balances={})
        account = plan.accounts[0]
        assert account.balance_is_real is False
        assert account.funded is False
        assert account.can_deploy_live is False
        assert plan.is_deployable is False
        assert any("balance could not be read" in w for w in account.warnings)

    def test_a_budget_above_the_real_balance_sizes_on_the_smaller_figure(self):
        storage = _storage()
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={"polymarket": 50.0},
            balances={"polymarket": {"available": True, "balance": 12.0}})
        account = plan.accounts[0]
        assert account.deposited_usd == pytest.approx(12.0)
        assert any("exceeds the venue's reported balance" in w
                   for w in account.warnings)

    def test_money_at_a_venue_with_no_budget_is_not_deployable(self):
        storage = _storage()
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={},
            balances={"polymarket": {"available": True, "balance": 80.0}})
        assert len(plan.accounts) == 1
        assert plan.accounts[0].can_deploy_live is False
        assert plan.accounts[0].available_usd == 0.0

    def test_live_mode_with_no_ready_venue_is_not_deployable(self):
        storage = _storage()
        plan = CapitalLedger(storage=storage).build(
            mode="live", budgets={"polymarket": 50.0}, balances={})
        assert plan.is_deployable is False
        assert plan.warnings, "a live plan that cannot deploy must say why"

    def test_paper_mode_is_deployable_without_any_money(self):
        """
        Paper needs no capital and no credentials. That is the whole point of it
        existing as a mode.
        """
        storage = _storage()
        plan = CapitalLedger(storage=storage).build(mode="paper", budgets={},
                                                    balances={})
        assert plan.accounts == []
        assert plan.live_venues == []


# ==========================================================================
# 3. how money gets in
# ==========================================================================

class TestFundingARoute:
    def test_the_route_names_the_network_because_that_is_how_people_lose_money(self):
        route = FUNDING_ROUTES["polymarket"]
        text = " ".join(route["deposit_steps"]).lower()
        assert "polygon" in text
        assert "wrong network" in text

    def test_an_empty_budget_sends_you_to_paper(self):
        result = plan_for_budget(0, mode="live")
        assert result["allocations"] == {}
        assert any("paper mode" in w for w in result["warnings"])

    def test_below_the_deposit_minimum_is_refused_with_its_reason(self):
        result = plan_for_budget(0.5, mode="live", preferred="kalshi")
        assert result["allocations"] == {}
        assert any("minimum" in w for w in result["warnings"])

    def test_a_small_budget_goes_to_one_venue_not_three(self):
        """
        Splitting $50 three ways is three accounts that each cannot meet a
        minimum order size. One funded venue beats three starved ones.
        """
        result = plan_for_budget(50, mode="live")
        assert result["allocations"] == {"polymarket": 50.0}
        assert "one venue" in " ".join(result["notes"]).lower()

    def test_the_plan_does_not_pretend_to_transfer_money(self):
        result = plan_for_budget(50, mode="live")
        notes = " ".join(result["notes"]).lower()
        assert "has no account of its own" in notes, (
            "the funding explanation must be explicit that money goes to the "
            "venue account, not to the agent"
        )

    def test_venues_that_cannot_hold_a_small_budget_are_named(self):
        result = plan_for_budget(50, mode="live")
        assert "manifold" in result["unfundable"]
        assert "play-money" in result["unfundable"]["manifold"]


# ==========================================================================
# 4. the console refuses rather than warns
# ==========================================================================

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    import importlib
    import src.ptai.ui.console as console
    importlib.reload(console)
    return TestClient(console.app)


class TestTheConsoleRefuses:
    def test_it_opens_in_paper_mode(self, client):
        body = client.get("/api/console/status").json()
        assert body["mode"] == "paper", (
            "a fresh install must not open armed"
        )

    def test_live_mode_is_refused_when_nothing_is_ready(self, client):
        response = client.post("/api/console/mode", json={"mode": "live"})
        assert response.status_code == 409
        assert response.json()["blockers"]
        assert "paper" in response.json()["note"].lower()

    def test_paper_mode_is_always_available(self, client):
        response = client.post("/api/console/mode", json={"mode": "paper"})
        assert response.status_code == 200

    def test_a_budget_without_a_readable_balance_is_refused(self, client):
        response = client.post("/api/console/budget",
                               json={"venue": "polymarket", "amount": 50})
        assert response.status_code == 409
        body = response.json()
        assert "not read" in body["error"]
        assert "not a way to put money in" in body["note"]

    def test_a_venue_with_no_funding_route_is_refused(self, client):
        response = client.post("/api/console/budget",
                               json={"venue": "manifold", "amount": 10})
        assert response.status_code == 400

    def test_a_negative_budget_is_refused(self, client):
        response = client.post("/api/console/budget",
                               json={"venue": "polymarket", "amount": -5})
        assert response.status_code == 400

    def test_a_bad_mode_is_refused(self, client):
        response = client.post("/api/console/mode", json={"mode": "yolo"})
        assert response.status_code == 400

    def test_a_live_cycle_is_refused_when_nothing_is_ready(self, client):
        response = client.post("/api/console/run-cycle", json={"mode": "live"})
        assert response.status_code == 409

    def test_results_report_unavailable_rather_than_zero(self, client):
        """
        0.0% return and 0.0% drawdown are claims about performance. On an account
        with no resolved history they are undefined, and saying so is the
        difference between a dashboard and a sales pitch.
        """
        body = client.get("/api/console/results").json()
        assert "net_return_30d_pct" in body["unavailable"]
        assert "max_drawdown_pct" in body["unavailable"]
        assert "best_validated_venue" in body["unavailable"]
        assert body["figures"].get("realised_pnl_usd") is None or \
            "realised_pnl_usd" in body["figures"]

    def test_the_funding_panel_explains_that_the_agent_holds_nothing(self, client):
        body = client.get("/api/console/funding?total=50").json()
        assert "no account to send money to" in body["principle"].lower()
        assert body["routes"]["polymarket"]["deposit_steps"]

    def test_a_paper_cycle_runs_from_the_console(self, client):
        """
        Paper mode has to be runnable, or the operator has to launch a second
        process to see the thing they asked to see. It is safe to construct here
        precisely because dry_run propagates to every adapter.
        """
        response = client.post("/api/console/run-cycle", json={"mode": "paper"})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "paper"
        assert "status" in body and "settlement" in body

    def test_a_live_cycle_is_never_run_from_the_console(self, client):
        """
        A live engine must be the supervised process. If the web handler could
        construct one, then loading the page and pressing a button would both be
        enough to trade real money.
        """
        response = client.post("/api/console/run-cycle", json={"mode": "live"})
        assert response.status_code in (409, 503)
        assert "live" in response.json()["error"].lower()

    def test_a_venue_that_does_not_answer_is_not_a_zero_balance(self, client):
        """
        "Asked and did not answer" is not "has no money", and the difference is
        what stops the operator chasing a funding problem that does not exist.
        """
        body = client.get("/api/console/status").json()
        data = [s for s in body["steps"] if s["step"] == "data"][0]
        assert "reported a balance" in data["detail"] or \
            "no venue was reachable" in data["detail"]

    def test_the_page_loads_and_shows_the_mode(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "PAPER" in response.text
        assert "How capital gets in" in response.text
