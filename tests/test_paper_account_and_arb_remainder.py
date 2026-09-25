"""
The paper account is an account, and the arb pair leaves no remainder behind.

The operator's plan is to run PTAI in paper mode for a while before switching
to live. For that trial to be evidence, paper must have the SAME execution as
live, and two gaps stood between it and that:

  1. The paper pool was not charged by paper trades. The paper bankroll
     exists in storage and paper settlements bank into it, but paper
     POSITIONS never reduced the free capital the loop sizes against - so a
     paper run could open position after position against a $50 bankroll
     that never shrank, and the trial would measure a strategy on capital it
     did not have. Live has no such freedom: every live position reduces the
     live free cash. Now the ledger splits the positions and the reservations
     between the two accounts, the loop sizes each trade from - and draws it
     down against - the pool that trade will actually spend, and the
     operator view shows the paper account while the operator runs paper.

  2. An arbitrage leg that partially filled left its remainder resting in the
     book after the hedge. The hedge is sized in SHARES on what filled, so a
     later fill of the remainder is a one-sided position - exactly the
     exposure the pair exists to avoid. Live and paper both now cancel the
     remainder (venue first where there is a venue order, then the local
     row), and a venue cancel that is not confirmed leaves the reservation
     open until reconciliation proves the order is gone.

Along the way: cancel_order now persists to the orders table (it only ever
touched the in-memory cache, so a restart lost the cancel and the kill
switch could not release the reservations), and cancel_all walks storage as
well as the cache.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptai.storage.db import Storage
from ptai.execution.position_ledger import PositionLedgerBuilder
from ptai.execution.order_manager import OrderManager


# ------------------------------------------------------------- the split

class TestLedgerSplitsThePools:
    def _storage(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "pools.db"))
        storage.set_bankroll(50.0)
        storage.set_paper_bankroll(40.0)
        storage.log_trade({
            "market_id": "POOL-LIVE", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 10.0, "market_price": 0.50,
            "execution_mode": "live", "status": "open",
        })
        storage.log_trade({
            "market_id": "POOL-PAPER", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 12.0, "market_price": 0.50,
            "execution_mode": "paper", "status": "paper",
        })
        storage.upsert_order({
            "order_id": "live-order-1", "market_id": "POOL-LIVE",
            "side": "YES", "limit_price": 0.50, "requested_usd": 8.0,
            "status": "submitted", "venue_id": "polymarket",
            "original_size": 16.0, "size_matched": 2.0, "matched_usd": 1.0,
            "execution_mode": "live",
        })
        storage.upsert_order({
            "order_id": "local-paper-1", "market_id": "POOL-PAPER",
            "side": "YES", "limit_price": 0.50, "requested_usd": 5.0,
            "status": "dry_run", "venue_id": "polymarket",
            "original_size": 10.0, "size_matched": 0.0, "matched_usd": 0.0,
            "execution_mode": "paper",
        })
        return storage

    def test_reservations_split_by_owner(self, tmp_path):
        storage = self._storage(tmp_path)
        assert storage.resting_capital_usd() == pytest.approx(12.0)
        assert storage.resting_capital_usd(execution_mode="live") == \
            pytest.approx(7.0)
        assert storage.resting_capital_usd(execution_mode="paper") == \
            pytest.approx(5.0)

    def test_each_pool_is_charged_the_way_live_is(self, tmp_path):
        storage = self._storage(tmp_path)
        ledger = PositionLedgerBuilder(storage=storage).build()
        # live: 50 bankroll - 10 live position - 7 live resting
        assert ledger.free_cash == pytest.approx(33.0)
        assert ledger.open_position_cost == pytest.approx(10.0)
        assert ledger.resting_order_cost == pytest.approx(7.0)
        # paper: 40 bankroll - 12 paper position - 5 paper resting
        assert ledger.paper_free_cash == pytest.approx(23.0)
        assert ledger.paper_position_cost == pytest.approx(12.0)
        assert ledger.paper_resting_order_cost == pytest.approx(5.0)
        # equity of each pool is its whole account
        assert ledger.equity == pytest.approx(50.0)
        assert ledger.paper_equity == pytest.approx(40.0)
        assert ledger.paper_bankroll == pytest.approx(40.0)
        # the pools do not leak into each other
        assert ledger.live_position_count == 1
        assert ledger.paper_position_count == 1


# ------------------------------------------------- sizing from the right pool

class TestPaperSizingUsesThePaperPool:
    def test_a_paper_run_cannot_spend_more_than_its_pool(self, tmp_path,
                                                         monkeypatch):
        """
        The paper bankroll is $20. A paper trade may therefore size at most
        6% of $20 = $1.20. If the loop sized against the live pool ($50) the
        same trade would be ~$3.00 - capital the paper run does not have.
        """
        import tests.test_full_cycle_from_discovery_to_allocation as base
        from src.ptai.execution.position_ledger import PositionLedgerBuilder

        agent, adapter = base.TestPaperModeSimulatesTheWholeCycle()._build(
            tmp_path, book={"asks": [{"price": "0.56", "size": "5000"}]})
        agent.storage.set_paper_bankroll(20.0)
        base._force_qualified(agent, monkeypatch)
        base._inject_opportunity(agent, adapter._market(), monkeypatch)

        result = base._cycle(agent)
        assert result["execution"], "the cycle recorded no execution at all"
        fill = result["execution"][0].get("fill") or {}
        assert fill.get("simulated") is True, "the result was recorded as live"

        positions = agent.storage.get_open_positions()
        assert len(positions) == 1
        size = positions[0]["position_size_usd"]
        assert size <= 1.25, (
            f"paper trade sized ${size:.4f} against a $20 paper bankroll - "
            f"the loop must size from the paper pool, not the live one")
        assert size >= 1.0, "the min order size is $1, so it must have sized"

        # Rebuild from storage: last_ledger is the pre-trade snapshot.
        ledger = PositionLedgerBuilder(storage=agent.storage).build()
        # the live pool is untouched by a paper trade
        assert ledger.free_cash == pytest.approx(50.0, abs=0.02)
        # the paper pool is drawn down by the position
        assert ledger.paper_free_cash == pytest.approx(
            20.0 - size, abs=0.02)
        # and the real bankroll is what it was
        assert agent.storage.get_bankroll() == pytest.approx(50.0)

    def test_the_sizing_pool_follows_the_adapter_not_the_agent(self,
                                                               tmp_path,
                                                               monkeypatch):
        """
        The agent is NOT dry-run, but the venue can only simulate, so the
        trade runs in paper and must draw the paper pool: the pool is a
        property of the order, not of the process.
        """
        import tests.test_arbitrage_lane_executes as arb
        import tests.test_full_cycle_from_discovery_to_allocation as base

        agent, venues = arb._build(tmp_path, monkeypatch)
        # Make the two pools different numbers, so the pool the trade drew
        # from is visible in the size: 6% of $20 is $1.20, of $50 it is $3.00.
        agent.storage.set_paper_bankroll(20.0)
        # One single-path opportunity on the paper venue - the arb lane gets
        # an empty scan, so the cycle trades exactly this one order.
        base._force_qualified(agent, monkeypatch)
        base._inject_opportunity(agent, venues[arb.VENUE_A]._market(),
                                 monkeypatch)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=1))
        assert report["execution"], "the cycle traded nothing"

        # Rebuild from storage: last_ledger is the pre-trade snapshot.
        from src.ptai.execution.position_ledger import PositionLedgerBuilder
        ledger = PositionLedgerBuilder(storage=agent.storage).build()
        paper_committed = (ledger.paper_position_cost
                           + ledger.paper_resting_order_cost)
        assert paper_committed > 0, "no paper trade was recorded"
        # SIZED from the paper pool even though the agent process is not
        # dry-run: the pool follows the adapter, not the flag.
        assert paper_committed <= 1.25, (
            f"the paper trade sized ${paper_committed:.4f} against a $20 "
            f"paper bankroll - it drew the live pool ($50 -> ~$3.00)")
        assert ledger.paper_free_cash == pytest.approx(
            20.0 - paper_committed, abs=0.02), (
            "the paper pool must equal its bankroll minus what paper trades "
            "committed, whatever mode the agent process is in")
        # and the live pool is untouched
        assert ledger.free_cash == pytest.approx(50.0, abs=0.02)


# ----------------------------------------------------- the operator's screen

class TestOperatorViewShowsThePaperAccount:
    def test_paper_mode_reads_the_paper_account(self, tmp_path):
        from ptai.operator_view import _capital
        from ptai.execution.capital import set_operator_mode

        storage = Storage(db_path=str(tmp_path / "opview.db"))
        storage.set_bankroll(50.0)
        storage.set_paper_bankroll(45.0)
        storage.log_trade({
            "market_id": "OPV", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 5.0, "market_price": 0.50,
            "execution_mode": "paper", "status": "paper",
        })
        set_operator_mode(storage, "paper")
        capital = _capital(storage)
        assert capital["available"] is True
        assert capital["account"] == "paper"
        assert capital["paper_positions"] == 1
        # 45 - 5 position = 40 free; the live $50 is not on screen
        assert capital["free_cash_usd"] == pytest.approx(40.0)
        assert capital["equity_usd"] == pytest.approx(45.0)
        assert capital["reserved_capital_usd"] == pytest.approx(5.0)

    def test_live_mode_reads_the_live_account(self, tmp_path):
        from ptai.operator_view import _capital
        from ptai.execution.capital import set_operator_mode

        storage = Storage(db_path=str(tmp_path / "opview.db"))
        storage.set_bankroll(50.0)
        storage.log_trade({
            "market_id": "OPV", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 5.0, "market_price": 0.50,
            "execution_mode": "live", "status": "open",
        })
        set_operator_mode(storage, "live")
        capital = _capital(storage)
        assert capital["account"] == "live"
        assert capital["live_positions"] == 1
        assert capital["free_cash_usd"] == pytest.approx(45.0)
        assert capital["equity_usd"] == pytest.approx(50.0)


# --------------------------------------------------- the arb remainder

class TestArbRemainderIsCancelled:
    def test_a_partial_leg_is_hedged_and_its_remainder_cancelled(
            self, tmp_path, monkeypatch):
        """
        Leg A's book holds 2 shares; the leg is asked for $1.50. It fills 2
        shares and $0.58 would rest. The hedge is sized on the 2 filled
        shares, and the remainder must be cancelled - if it is left to fill
        later it is one-sided exposure, the one thing the pair exists to
        avoid.
        """
        import tests.test_arbitrage_lane_executes as arb

        agent, venues = arb._build(
            tmp_path, monkeypatch, ask_sizes={arb.VENUE_A: "2"})
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        lane = report["arbitrage"]["execution"]
        assert lane["attempted"] == 1
        assert lane["legs_filled"] == 2, lane

        # The pair is balanced in SHARES: both legs hold the same number of
        # shares, and that number is what leg A's 2-share book actually
        # filled (leg B's hedge is sized on leg A's filled shares).
        positions = agent.storage.get_open_positions()
        assert len(positions) == 2, positions

        def _shares(p):
            tp = p.get("token_price_at_entry")
            if tp:
                return p["position_size_usd"] / tp
            return p.get("filled_shares")

        shares = sorted(_shares(p) for p in positions)
        assert shares[0] == pytest.approx(shares[1], abs=0.01), (
            f"the legs hold different share counts - the pair is NOT "
            f"balanced: {shares}")
        assert shares[1] == pytest.approx(2.0, abs=0.01), (
            f"the hedge is sized on leg A's 2 filled shares, but the legs "
            f"hold {shares}")

        # The remainder is gone: nothing is still working in the book, so the
        # reservation is released and the pair holds nothing unhedged.
        assert agent.storage.resting_capital_usd() == pytest.approx(0.0)
        open_rows = agent.storage.get_open_orders()
        assert open_rows == [], (
            f"the resting remainder was left in the book: {open_rows}")
        cancelled = agent.storage.conn.execute(
            "SELECT id, status, terminal_reason FROM orders "
            "WHERE status = 'cancelled'").fetchall()
        assert cancelled, (
            "the remainder order was never cancelled - it would fill later "
            "as one-sided exposure")
        assert "one-sided" in str(cancelled[0]["terminal_reason"])

        # The report says the cancel happened.
        arb_results = report["arbitrage"]["execution"]["results"]
        assert any(r.get("remainder_cancelled") for r in arb_results), (
            f"no slot recorded the remainder cancel: {arb_results}")

    def test_a_fully_filled_pair_leaves_nothing_to_cancel(
            self, tmp_path, monkeypatch):
        """
        Deep books: both legs fill completely, no remainder exists, no order
        is recorded, and the cancel path is a no-op that leaves the pair
        intact.
        """
        import tests.test_arbitrage_lane_executes as arb

        agent, venues = arb._build(tmp_path, monkeypatch)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))
        lane = report["arbitrage"]["execution"]
        assert lane["legs_filled"] == 2
        assert len(agent.storage.get_open_positions()) == 2
        assert agent.storage.resting_capital_usd() == pytest.approx(0.0)
        cancelled = agent.storage.conn.execute(
            "SELECT id FROM orders WHERE status = 'cancelled'").fetchall()
        assert cancelled == []


# -------------------------------------------------- the cancel itself

class TestCancelPersistsToStorage:
    def test_cancel_releases_the_reservation(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "cancel.db"))
        om = OrderManager(storage=storage)
        storage.upsert_order({
            "order_id": "local-cancel-1", "market_id": "C1", "side": "YES",
            "limit_price": 0.50, "requested_usd": 3.0, "status": "dry_run",
            "venue_id": "polymarket", "original_size": 6.0,
            "size_matched": 2.0, "matched_usd": 1.0,
            "execution_mode": "paper",
        })
        assert storage.resting_capital_usd() == pytest.approx(2.0)
        assert om.cancel_order("local-cancel-1", reason="test") is True
        row = storage.get_order_row("local-cancel-1")
        assert row["status"] == "cancelled"
        assert "test" in str(row["terminal_reason"])
        assert storage.resting_capital_usd() == pytest.approx(0.0)

    def test_cancel_all_walks_storage_after_a_restart(self, tmp_path):
        """
        A restarted process has an empty in-memory cache and the orders in
        storage. cancel_all must reach them, or the kill switch releases
        nothing.
        """
        storage = Storage(db_path=str(tmp_path / "cancel.db"))
        for i, status in enumerate(("dry_run", "submitted")):
            storage.upsert_order({
                "order_id": f"local-restart-{i}", "market_id": f"R{i}",
                "side": "YES", "limit_price": 0.50, "requested_usd": 2.0,
                "status": status, "venue_id": "polymarket",
                "original_size": 4.0, "size_matched": 0.0,
                "matched_usd": 0.0, "execution_mode": "paper",
            })
        assert storage.resting_capital_usd() == pytest.approx(4.0)
        fresh = OrderManager(storage=storage)  # empty in-memory cache
        assert fresh.cancel_all(reason="kill switch") == 2
        assert storage.resting_capital_usd() == pytest.approx(0.0)
