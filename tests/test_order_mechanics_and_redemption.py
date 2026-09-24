"""
The four things this batch had to make true.

  1. an order can be priced on the venue's tick and minimum size
  2. neg_risk selects the correct exchange contract
  3. resting orders are visible and reconciled into the ledger
  4. a settled winning position can be redeemed

Every test here drives the real object. Where a transport has to be faked it is
faked at the venue boundary - the CLOB client, the relayer, the Data API - so
the logic under test is the logic that will run.
"""

import asyncio
import inspect
import json

import pytest

from src.ptai.execution.multi_venue_executor import ExecutionResult, MultiVenueExecutor
from src.ptai.execution.order_manager import OrderManager
from src.ptai.execution.redemption import (
    COLLATERAL_ADDRESS,
    CTF_ADDRESS,
    CTF_COLLATERAL_ADAPTER,
    LEGACY_NEG_RISK_ADAPTER,
    NEG_RISK_COLLATERAL_ADAPTER,
    RedeemablePosition,
    Redeemer,
    _as_bytes32,
    redeem_calldata,
)
from src.ptai.markets.mechanics import (
    MIN_TICK_SIZE,
    SIZE_DECIMALS,
    TICK_DECIMALS,
    TICK_SIZES,
    MarketMechanics,
    is_on_tick,
    normalise_tick,
    round_price_to_tick,
    round_size_to_step,
    tick_rounding_cost,
)
from src.ptai.storage.db import Storage


# ==========================================================================
# 1. the tick and the minimum size
# ==========================================================================

class TestPricedOnTheVenueTick:
    def test_the_decimals_match_the_venue_rounding_config(self):
        """
        The venue rounds prices with ROUNDING_CONFIG[tick].price. If our decimal
        count differs we round onto a grid the venue does not share, and the
        venue silently re-rounds what we signed.
        """
        expected = {
            "0.1": 1, "0.01": 2, "0.005": 3, "0.0025": 4, "0.001": 3, "0.0001": 4,
        }
        assert TICK_DECIMALS == expected, (
            "our tick decimals disagree with the venue's ROUNDING_CONFIG"
        )
        assert SIZE_DECIMALS == 2, "the venue rounds size to 2 decimals"

    def test_the_tick_literal_includes_the_fractional_ticks(self):
        """
        Read from the TickSize literal, not from a verbal list: 0.005 and 0.0025
        are real ticks and a hardcoded {0.1, 0.01, 0.001, 0.0001} would silently
        treat them as invalid.
        """
        for tick in ("0.1", "0.01", "0.005", "0.0025", "0.001", "0.0001"):
            assert tick in TICK_SIZES
            # normalise_tick returns the canonical string form, and parses a
            # float or a string to the same one.
            assert normalise_tick(tick) == tick
            assert normalise_tick(float(tick)) == tick

    def test_a_buy_never_rounds_above_the_modelled_price(self):
        """
        Rounding a buy UP pays more than the model decided was worth paying, and
        the venue's own round_normal does exactly that: 0.567 becomes 0.57. Our
        rounding must be at or below the modelled price for every tick.
        """
        from py_clob_client_v2.order_builder.helpers import round_normal
        from py_clob_client_v2.order_builder.builder import ROUNDING_CONFIG

        for tick in TICK_SIZES:
            decimals = TICK_DECIMALS[tick]
            assert ROUNDING_CONFIG[tick].price == decimals, (
                f"tick {tick}: our decimals differ from the venue's"
            )
            for price in (0.567, 0.1234, 0.9999, 0.0301):
                if not (float(tick) <= price <= 1 - float(tick)):
                    continue
                ours = round_price_to_tick(price, tick, "BUY")
                venue = round_normal(price, decimals)
                assert ours <= price + 1e-12, (
                    f"buy rounded {price} up to {ours} on tick {tick}"
                )
                assert ours <= venue + 1e-12, (
                    f"tick {tick}: we round {price} to {ours} but the venue signs "
                    f"{venue}, so the signed price is worse than the modelled one"
                )
                assert is_on_tick(ours, tick), f"{ours} is not on the {tick} grid"

    def test_a_sell_rounds_down_so_it_never_undersells(self):
        for tick in TICK_SIZES:
            price = 0.5678
            if not (float(tick) <= price <= 1 - float(tick)):
                continue
            rounded = round_price_to_tick(price, tick, "SELL")
            assert rounded >= price, (
                f"a sell rounded {price} down to {rounded}, selling for less than "
                f"the modelled price"
            )
            assert is_on_tick(rounded, tick)

    def test_an_off_tick_order_is_corrected_locally_before_signing(self):
        """
        The venue's price_valid is BOUNDS ONLY - it accepts 0.567 on a 0.01 tick
        and then silently re-rounds it upward when signing. So tick conformance
        cannot be delegated to the venue; it has to happen at decision time.
        """
        mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0,
                                    source="test", is_real=True)
        signed = mechanics.round_price(0.567, "BUY")
        assert signed == 0.56
        assert mechanics.validate_order(0.567, 5.0)[0] is False, (
            "the model must not be allowed to sign an off-tick price"
        )
        ok, reason = mechanics.validate_order(0.567, 5.0)
        assert "tick" in reason.lower()

    def test_the_minimum_size_is_enforced_locally_and_named(self):
        """
        A refusal must come from our own check, with the venue's rule in the
        reason, not from the venue rejecting an order we already signed.
        """
        mechanics = MarketMechanics(tick_size="0.01", min_order_size=5.0,
                                    source="clob_market_info", is_real=True)
        ok, reason = mechanics.validate_order(0.57, 3.0)
        assert ok is False
        assert "5" in reason and "minimum" in reason

    def test_size_rounds_down_so_the_spend_never_exceeds_the_budget(self):
        """
        A size rounded up spends more than the approved amount, which would make
        the risk limits advisory.
        """
        mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0)
        shares = mechanics.shares_for_usd(3.0, 0.57, "BUY")
        assert shares * 0.57 <= 3.0 + 1e-9, (
            f"{shares} shares at 0.57 costs more than the $3.00 approved"
        )
        assert round_size_to_step(5.456) == 5.45

    def test_a_longshot_rounding_cost_is_visible_not_hidden(self):
        """
        On a 0.01 tick an 0.011 model price is off-grid, and rounding down moves
        it ~9%. The cost is reported so a strategy cannot silently trade a
        materially different price than it modelled.
        """
        cost = tick_rounding_cost(0.011, "0.01", "BUY")
        assert abs(cost) > 0.08, (
            f"rounding 0.011 to the 0.01 grid moves the price {abs(cost):.1%}, "
            f"which a strategy must be told about"
        )
        assert cost < 0, "a buy rounds down, so the modelled price falls"

    def test_assumed_mechanics_are_labelled_and_never_claim_to_be_real(self):
        """
        Guessing a tick is sometimes unavoidable; claiming it is the venue's is
        not. An assumption carries is_real=False and says so.
        """
        assumed = MarketMechanics.assumed("venue did not answer")
        assert assumed.is_real is False
        assert assumed.source == "assumed_default"
        assert assumed.warnings, "an assumption must carry its reason"
        assert assumed.tick_size == MIN_TICK_SIZE


# ==========================================================================
# 2. neg_risk selects the exchange contract
# ==========================================================================

class _FakeV2Client:
    """The V2 CLOB surface the executor calls, recording every argument."""

    def __init__(self, market_info=None):
        self.posted = []
        self.cancelled = []
        self.market_info = market_info or {
            "min_tick_size": "0.01", "neg_risk": True, "min_order_size": 1.0,
            "taker_base_fee": 0.0, "maker_base_fee": 0.0,
            "seconds_delay": 0, "accepting_orders": True,
        }

    def get_clob_market_info(self, condition_id):
        return self.market_info

    def get_tick_size(self, token_id):
        return self.market_info.get("min_tick_size", "0.01")

    def get_neg_risk(self, token_id):
        return self.market_info.get("neg_risk", False)

    def create_and_post_order(self, order_args, options, order_type, post_only):
        self.posted.append({"args": order_args, "options": options,
                            "order_type": order_type, "post_only": post_only})
        return {"success": True, "orderID": "0xorder", "status": "live",
                "size_matched": "0", "original_size": "5.45"}

    def cancel_order(self, payload):
        self.cancelled.append(getattr(payload, "orderID", payload))
        return {"canceled": [getattr(payload, "orderID", payload)], "not_canceled": {}}

    def get_open_orders(self, params):
        return []

    def get_order(self, order_id):
        return {"status": "live", "size_matched": "0", "original_size": "5.45",
                "price": "0.57"}

    def get_trades(self, params):
        return []

    def get_balance_allowance(self, params):
        return {"balance": "0", "allowance": "0"}


def _executor_with(client):
    import src.ptai.markets.polymarket as pm

    executor = pm.PolymarketExecutor()
    executor.client = client
    executor.client_error = None
    return executor


class TestNegRiskSelectsTheContract:
    def test_neg_risk_is_passed_into_the_order_options(self):
        """
        The exchange contract a market settles against depends on neg_risk, and
        the SDK takes it per order in CreateOrderOptions. Omitting it routes a
        neg-risk order at the standard exchange.
        """
        client = _FakeV2Client()
        executor = _executor_with(client)
        mechanics = executor.get_mechanics("tok", condition_id="0xc1")
        assert mechanics.neg_risk is True
        assert mechanics.is_real is True

        result = executor.place_order("tok", price=0.57, size=5.0, side="BUY",
                                      mechanics=mechanics, dry_run=False)
        assert result["status"] != "failed", result
        options = client.posted[-1]["options"]
        assert options.neg_risk is True, (
            "neg_risk was dropped, so the order would go to the wrong exchange"
        )
        assert str(options.tick_size) == "0.01"

    def test_a_standard_market_does_not_set_neg_risk(self):
        client = _FakeV2Client(market_info={
            "min_tick_size": "0.001", "neg_risk": False, "min_order_size": 1.0})
        executor = _executor_with(client)
        mechanics = executor.get_mechanics("tok", condition_id="0xc2")
        assert mechanics.neg_risk is False
        executor.place_order("tok", price=0.5, size=5.0, side="BUY",
                             mechanics=mechanics, dry_run=False)
        assert client.posted[-1]["options"].neg_risk is False

    def test_the_order_that_is_signed_is_the_order_that_was_modelled(self):
        """
        The venue re-rounds silently, so a rejected-looking price can still be
        signed at a different one. What gets signed is reported back.
        """
        client = _FakeV2Client()
        executor = _executor_with(client)
        mechanics = executor.get_mechanics("tok", condition_id="0xc3")

        result = executor.place_order("tok", price=0.567, size=5.456, side="BUY",
                                      mechanics=mechanics, dry_run=True)
        assert result["status"] == "dry_run"
        assert result["signed_price"] == 0.56
        assert result["requested_price"] == 0.567
        assert result["mechanics_adjustment"]["price"] == pytest.approx(-0.007)
        assert client.posted == [], "a dry run must not reach the venue"

    def test_a_venue_illegal_price_is_refused_rather_than_signed(self):
        client = _FakeV2Client()
        executor = _executor_with(client)
        mechanics = executor.get_mechanics("tok", condition_id="0xc4")
        result = executor.place_order("tok", price=1.0, size=5.0, side="BUY",
                                      mechanics=mechanics, dry_run=False)
        assert result["status"] == "rejected", result
        assert "tick" in result["reason"].lower() or "price" in result["reason"].lower()
        assert client.posted == [], "an illegal price must not be signed"


# ==========================================================================
# 3. resting orders are visible and reconciled
# ==========================================================================

class TestRestingOrdersAreReconciled:
    def test_an_order_resting_in_the_book_is_not_a_position(self):
        """
        A live order has committed capital but bought nothing. Booking it as a
        position invents shares; ignoring it lets the cash be spent twice.
        """
        executor = MultiVenueExecutor(registry=None, bankroll=50.0)
        fill = executor._read_fill({"status": "live", "orderID": "0x1",
                                    "size_matched": "0", "original_size": "5.45",
                                    "price": "0.57"}, requested_usd=3.1,
                                   requested_price=0.57)
        assert fill["status"] == "submitted"
        assert fill["filled_usd"] == 0.0
        assert fill["size_matched"] == 0.0 and fill["original_size"] == 5.45

    def test_a_partial_fill_books_only_the_matched_part(self):
        """
        `size_matched` below `original_size` means the remainder is still
        working. Treating it as complete overstates the position by the unfilled
        remainder.
        """
        executor = MultiVenueExecutor(registry=None, bankroll=50.0)
        fill = executor._read_fill({"status": "matched", "orderID": "0x2",
                                    "size_matched": "2.0", "original_size": "5.45",
                                    "price": "0.57"}, requested_usd=3.1,
                                   requested_price=0.57)
        assert fill["status"] == "partial", (
            "a part-filled order was booked as a complete fill"
        )
        assert fill["filled_usd"] == pytest.approx(1.14)

    def test_a_matched_order_that_is_whole_is_a_fill(self):
        executor = MultiVenueExecutor(registry=None, bankroll=50.0)
        fill = executor._read_fill({"status": "matched", "orderID": "0x3",
                                    "size_matched": "5.45", "original_size": "5.45",
                                    "price": "0.57"}, requested_usd=3.1,
                                   requested_price=0.57)
        assert fill["status"] == "filled"
        assert fill["filled_usd"] == pytest.approx(3.1065, abs=1e-6)

    def test_a_failed_send_is_not_final(self):
        """
        A send that failed with no venue confirmation may still be resting: the
        response can be what was lost, not the request. It is reported as
        unconfirmed and must be asked about.
        """
        executor = MultiVenueExecutor(registry=None, bankroll=50.0)
        fill = executor._read_fill(
            {"status": "failed", "unconfirmed_send": True, "error": "timeout"},
            requested_usd=3.0, requested_price=0.57)
        assert fill["status"] == "failed"
        assert fill["unconfirmed_send"] is True

    def test_the_result_separates_committed_from_reserved_capital(self):
        resting = ExecutionResult(
            venue_id="polymarket", market_id="M1", status="submitted",
            amount_usd=3.0, price=0.57, fees_usd=0, gas_usd=0, latency_ms=1,
            reasoning="", filled_usd=0.0, filled_price=0.0, order_id="0x9",
            size_matched=0.0, original_size=5.26, resting_usd=3.0)
        assert resting.committed_capital is False, "a resting order bought nothing"
        assert resting.reserves_capital is True, "its cash is locked at the venue"
        assert resting.should_record_position is False
        assert resting.needs_reconciliation is True

        partial = ExecutionResult(
            venue_id="polymarket", market_id="M2", status="partial",
            amount_usd=3.0, price=0.57, fees_usd=0, gas_usd=0, latency_ms=1,
            reasoning="", filled_usd=1.14, filled_price=0.57, order_id="0x8",
            size_matched=2.0, original_size=5.26)
        assert partial.committed_capital is True
        assert partial.should_record_position is True
        assert partial.unfilled_shares == pytest.approx(3.26)


class _StubAdapter:
    """The two reads reconciliation needs, in the shape the adapter provides."""

    def __init__(self, orders=None, listing_raises=False, order_raises=False):
        self._orders = orders or {}
        self._listing = list(self._orders)
        self._listing_raises = listing_raises
        self._order_raises = order_raises

    async def get_open_orders(self, *args, **kwargs):
        if self._listing_raises:
            raise RuntimeError("venue unreachable")
        return {"available": True, "is_real": True,
                "orders": [{"id": oid} for oid in self._listing]}

    async def get_order(self, order_id):
        if self._order_raises:
            raise RuntimeError("venue unreachable")
        if order_id not in self._orders:
            return {"available": False, "reason": "not found"}
        return {"available": True, "is_real": True, **self._orders[order_id]}


class TestTheLedgerLearnsWhatTheVenueDid:
    def _manager(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "recon.db"))
        return OrderManager(storage=storage), storage

    def _seed(self, storage, **overrides):
        row = {
            "order_id": "0xord1", "market_id": "M1", "token_id": "tok",
            "side": "BUY", "limit_price": 0.57, "requested_usd": 3.0,
            "status": "submitted", "venue_id": "polymarket",
            "original_size": 5.26, "size_matched": 0.0, "matched_usd": 0.0,
            "trade_id": None,
        }
        row.update(overrides)
        assert storage.upsert_order(row)
        return row["order_id"]

    def test_a_fill_that_grows_adds_to_the_existing_position(self, tmp_path):
        """
        One position row per market. Settlement looks an open trade up by market
        id, so a second row for a later fill would never be closed and its P&L
        would never be realised.
        """
        manager, storage = self._manager(tmp_path)
        trade_id = storage.log_trade({
            "market_id": "M1", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 1.0, "market_price": 0.5, "fair_price": 0.6,
            "edge": 0.1, "confidence": 0.7, "strategy": "test",
            "data_mode": "live"})
        self._seed(storage, trade_id=trade_id)

        adapter = _StubAdapter({"0xord1": {"status": "matched",
                                           "size_matched": 5.26,
                                           "original_size": 5.26,
                                           "price": 0.57}})
        report = asyncio.run(manager.reconcile(adapter, venue_id="polymarket"))

        assert report.filled_more == 1
        assert report.grew_usd == pytest.approx(5.26 * 0.57, abs=1e-6)
        assert report.now_complete == 1

        row = storage.conn.execute(
            "SELECT position_size_usd, market_price FROM trades WHERE id = ?",
            (trade_id,)).fetchone()
        assert row["position_size_usd"] == pytest.approx(1.0 + 5.26 * 0.57, abs=1e-6)
        # The entry price is the weighted average of what was actually paid.
        expected = (1.0 * 0.5 + 5.26 * 0.57 * 0.57) / (1.0 + 5.26 * 0.57)
        assert row["market_price"] == pytest.approx(expected, abs=1e-6)

        open_orders = storage.get_open_orders(venue_id="polymarket")
        assert open_orders == [], "a fully matched order must stop being open"

    def test_an_unresolved_fill_is_never_added_to_a_closed_position(self, tmp_path):
        manager, storage = self._manager(tmp_path)
        trade_id = storage.log_trade({
            "market_id": "M1", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 1.0, "market_price": 0.5, "fair_price": 0.6,
            "edge": 0.1, "confidence": 0.7, "strategy": "test",
            "data_mode": "live"})
        storage.conn.execute("UPDATE trades SET resolved = 1 WHERE id = ?", (trade_id,))
        storage.conn.commit()
        assert storage.add_to_position(trade_id, 2.0, 0.6) is False
        row = storage.conn.execute(
            "SELECT position_size_usd FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert row["position_size_usd"] == 1.0

    def test_an_order_gone_from_the_book_releases_its_capital(self, tmp_path):
        """
        Cancelled or expired. Either way it no longer holds capital at the venue,
        so it must stop being tracked - otherwise the agent reserves cash
        forever for an order that does not exist.
        """
        manager, storage = self._manager(tmp_path)
        self._seed(storage)
        assert storage.resting_capital_usd("polymarket") == pytest.approx(3.0)

        adapter = _StubAdapter({})  # not in the venue's working set
        report = asyncio.run(manager.reconcile(adapter, venue_id="polymarket"))

        assert report.released == 1
        assert report.released_usd == pytest.approx(3.0)
        assert storage.get_open_orders(venue_id="polymarket") == []
        assert storage.resting_capital_usd("polymarket") == 0.0

    def test_an_unanswerable_venue_leaves_the_order_open(self, tmp_path):
        """
        The failure that matters: a venue that will not say must not be read as
        "finished". An order of unknown fate keeps holding its capital.
        """
        manager, storage = self._manager(tmp_path)
        self._seed(storage)

        adapter = _StubAdapter({"0xord1": {}}, listing_raises=True, order_raises=True)
        report = asyncio.run(manager.reconcile(adapter, venue_id="polymarket"))

        assert report.unreconciled == 1
        assert report.released == 0
        assert storage.resting_capital_usd("polymarket") == pytest.approx(3.0)
        assert len(storage.get_open_orders(venue_id="polymarket")) == 1

    def test_no_adapter_leaves_orders_open_and_says_so(self, tmp_path):
        manager, storage = self._manager(tmp_path)
        self._seed(storage)
        report = asyncio.run(manager.reconcile(None, venue_id="polymarket"))
        assert report.unreconciled == 1
        assert storage.resting_capital_usd("polymarket") == pytest.approx(3.0)

    def test_an_unconfirmed_send_is_assumed_to_hold_its_whole_request(self, tmp_path):
        """
        The conservative direction. If the send's fate is unknown the cash may be
        locked, and under-counting locked cash is how a second trade gets sized
        against money that is already committed.
        """
        manager, storage = self._manager(tmp_path)
        self._seed(storage, order_id="0xunsent", status="unconfirmed_send")
        assert storage.resting_capital_usd("polymarket") == pytest.approx(3.0)

    def test_resting_capital_excludes_booked_position_cost(self, tmp_path):
        """
        A booked position's cost is in the trades table; an unfilled order's
        reservation is in the orders table. Adding both is right; counting either
        twice is not.
        """
        manager, storage = self._manager(tmp_path)
        storage.log_trade({
            "market_id": "M9", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 4.0, "market_price": 0.5, "fair_price": 0.6,
            "edge": 0.1, "confidence": 0.7, "strategy": "test",
            "data_mode": "live"})
        self._seed(storage, order_id="0xresting", matched_usd=0.0)
        assert storage.resting_capital_usd("polymarket") == pytest.approx(3.0)

    def test_a_partly_filled_resting_order_reserves_only_the_remainder(self, tmp_path):
        manager, storage = self._manager(tmp_path)
        self._seed(storage, matched_usd=1.14)
        assert storage.resting_capital_usd("polymarket") == pytest.approx(1.86)

    def test_orders_survive_a_restart(self, tmp_path):
        """
        The venue still holds the order across a process restart. If the agent
        forgets it, its capital is invisible and its fill is never booked.
        """
        path = str(tmp_path / "restart.db")
        storage = Storage(db_path=path)
        self._seed(storage)

        reopened = Storage(db_path=path)
        open_orders = reopened.get_open_orders(venue_id="polymarket")
        assert len(open_orders) == 1
        assert open_orders[0]["order_id"] == "0xord1"
        assert reopened.resting_capital_usd("polymarket") == pytest.approx(3.0)

    def test_a_fill_on_a_resting_order_opens_the_position_it_created(self, tmp_path):
        """
        The end of the resting-order story. The order rested, so no position was
        booked when it was submitted; hours later it fills. Without an opener the
        fill has nowhere to go and the exposure is invisible - which is the gap
        this whole batch exists to close.
        """
        manager, storage = self._manager(tmp_path)
        self._seed(storage, side="YES")
        adapter = _StubAdapter({"0xord1": {"status": "live", "size_matched": 2.0,
                                          "original_size": 5.26, "price": 0.57}})

        def opener(order, add_usd, price):
            assert order["side"] == "YES", "the opener needs the outcome side"
            return storage.log_trade({
                "market_id": order["market_id"], "venue_id": "polymarket",
                "side": order["side"], "position_size_usd": add_usd,
                "market_price": price, "fair_price": price, "edge": 0.0,
                "confidence": 0.0, "strategy": "resting_order_fill",
                "data_mode": "live"})

        report = asyncio.run(manager.reconcile(adapter, venue_id="polymarket",
                                              position_opener=opener))
        assert report.opened == 1
        assert report.grew_usd == pytest.approx(2.0 * 0.57, abs=1e-6)
        positions = storage.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["position_size_usd"] == pytest.approx(1.14, abs=1e-6)

    def test_a_fill_with_nowhere_to_go_is_refused_loudly(self, tmp_path):
        """
        With no opener the fill must not vanish and must not be invented. It is
        reported as unreconciled so the missing exposure is visible.
        """
        manager, storage = self._manager(tmp_path)
        self._seed(storage)
        adapter = _StubAdapter({"0xord1": {"status": "live", "size_matched": 5.26,
                                          "original_size": 5.26, "price": 0.57}})
        report = asyncio.run(manager.reconcile(adapter, venue_id="polymarket"))
        assert report.opened == 0
        assert report.unreconciled == 1
        assert storage.get_open_positions() == []

    def test_an_order_with_no_outcome_side_does_not_invent_a_position(self, tmp_path):
        """
        A position that cannot be settled is worse than no position: it holds
        capital forever. The opener the loop supplies refuses rather than
        guessing the side, and BUY/SELL is not a side a market can settle on.
        """
        import src.ptai.agent.v3_loop as v3

        manager, storage = self._manager(tmp_path)

        class Shim:
            pass

        shim = Shim()
        shim.storage = storage
        shim.data_mode = "live"
        opener = v3.TradingAgentV3._open_position_from_fill.__get__(shim, Shim)

        # A buy/sell direction is not an outcome side.
        assert opener({"market_id": "M1", "side": "BUY", "order_id": "0x1"},
                      1.0, 0.5) == 0
        assert opener({"market_id": "", "side": "YES"}, 1.0, 0.5) == 0
        assert opener({"market_id": "M1", "side": "YES"}, 0.0, 0.5) == 0
        assert storage.get_open_positions() == [], (
            "an unbookable fill became a position"
        )

        # The same opener books a real outcome side.
        trade_id = opener({"market_id": "M1", "side": "YES", "venue_id": "polymarket",
                           "order_id": "0x1"}, 2.0, 0.5)
        assert trade_id > 0
        positions = storage.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["side"] == "YES"
        assert positions[0]["position_size_usd"] == pytest.approx(2.0)

    def test_the_loop_reconciles_and_redeems_every_cycle(self):
        """
        Reachable in principle and dead in practice is how the last three of
        these gaps survived. Assert both steps are actually called from the cycle.
        """
        import src.ptai.agent.v3_loop as v3

        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "await self._reconcile_working_orders()" in source
        assert "await self._redeem_settled_wins()" in source
        assert "self.redeemer = Redeemer(" in inspect.getsource(v3.TradingAgentV3.__init__)
        # And the credentials must actually reach it. They were local variables
        # in __init__, so `getattr(self, "funder", None)` was always None and the
        # redeemer could never have run.
        init = inspect.getsource(v3.TradingAgentV3.__init__)
        assert "self.private_key = pk" in init and "self.funder = funder" in init

    def test_submissions_are_recorded_from_the_execution_path(self):
        """
        The recorder existing is not the same as the loop calling it. Without
        both, no order is ever tracked and nothing can be reconciled.
        """
        import src.ptai.agent.v3_loop as v3
        source = inspect.getsource(v3.TradingAgentV3.run_cycle)
        assert "self.order_manager.record_submission(" in source
        reconcile_source = inspect.getsource(
            v3.TradingAgentV3._reconcile_working_orders)
        assert "position_opener=self._open_position_from_fill" in reconcile_source


# ==========================================================================
# 4. a settled win can be redeemed
# ==========================================================================

class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return _FakeResponse(self.payload)


class _FakeRelayer:
    def __init__(self, state="STATE_CONFIRMED"):
        self.state = state
        self.executed = []

    def execute(self, transactions, metadata=None):
        self.executed.append((transactions, metadata))

        class Response:
            transaction_id = "tx-1"
        return Response()

    def poll_until_state(self, transaction_id, states=None, fail_state=None,
                         max_polls=None, poll_frequency=None):
        if self.state == "STATE_CONFIRMED":
            return {"id": transaction_id, "state": "STATE_CONFIRMED"}
        return None


class TestASettledWinCanBeRedeemed:
    def test_the_calldata_selector_is_the_venues_own(self):
        """
        0x01b7037c is the selector the venue publishes for
        NegRiskCtfCollateralAdapter.redeemPositions. If ours disagrees, the call
        hits a different function - or nothing.
        """
        data = redeem_calldata("0x" + "ab" * 32)
        assert data.startswith("0x01b7037c"), data[:10]
        # The four-argument form: selector + 4 head words + 2 tail words.
        assert len(data) == 2 + 8 + 32 * 7 * 2

    def test_the_legacy_two_argument_form_is_different(self):
        """
        Markets created before the migration keep the two-argument signature.
        Sending them the four-argument call would not redeem them.
        """
        legacy = redeem_calldata("0x" + "ab" * 32, legacy=True)
        assert not legacy.startswith("0x01b7037c")
        assert legacy != redeem_calldata("0x" + "ab" * 32)

    def test_index_sets_cover_both_outcomes(self):
        """
        Redeeming both outcomes is safe - the losing side pays zero - and avoids
        having to know which side we hold.
        """
        data = redeem_calldata("0x" + "ab" * 32, index_sets=(1, 2))
        assert data.rstrip("0").endswith("2")
        assert "2" in data[-64:]

    def test_a_condition_id_must_be_32_bytes(self):
        with pytest.raises(ValueError):
            _as_bytes32("")
        assert len(_as_bytes32("0x" + "cd" * 32)) == 32
        assert _as_bytes32("0x" + "0a" * 20) == b"\x00" * 12 + bytes.fromhex("0a" * 20)

    def test_neg_risk_routes_to_the_neg_risk_adapter(self):
        """
        The exchange contract a market settles against depends on neg_risk, and
        redemption has to follow the same routing or the claim goes to a contract
        that cannot pay it.
        """
        relayer = _FakeRelayer()
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = relayer

        neg = RedeemablePosition(condition_id="0x" + "11" * 32, asset="a",
                                 outcome="YES", size=10, current_value_usd=10,
                                 neg_risk=True)
        standard = RedeemablePosition(condition_id="0x" + "22" * 32, asset="b",
                                     outcome="YES", size=10, current_value_usd=10,
                                     neg_risk=False)
        report = redeemer.redeem([neg, standard])

        assert report.claimed == 2
        targets = [tx.to for batch, _ in relayer.executed for tx in batch]
        assert targets == [NEG_RISK_COLLATERAL_ADAPTER, CTF_COLLATERAL_ADAPTER]
        assert CTF_ADDRESS not in targets, (
            "V2 redeems through the collateral adapter that wraps into pUSD, "
            "not the raw CTF"
        )

    def test_a_legacy_neg_risk_market_uses_the_legacy_adapter(self):
        relayer = _FakeRelayer()
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = relayer
        position = RedeemablePosition(condition_id="0x" + "33" * 32, asset="a",
                                      outcome="YES", size=1, current_value_usd=1,
                                      neg_risk=True,
                                      raw={"legacyNegRisk": True})
        redeemer.redeem([position])
        assert relayer.executed[0][0][0].to == LEGACY_NEG_RISK_ADAPTER

    def test_no_signer_means_everything_is_unclaimed(self):
        """
        Fail closed. A redemption nobody performed is not a redemption, and the
        report must say the collateral is still locked.
        """
        redeemer = Redeemer(funder="0xf", dry_run=False)  # no private key
        assert redeemer.can_redeem is False
        position = RedeemablePosition(condition_id="0x" + "44" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5)
        report = redeemer.redeem([position])
        assert report.available is False
        assert report.claimed == 0
        assert report.unclaimed_value_usd == pytest.approx(5.0)
        assert any("signer" in u["reason"] for u in report.unclaimed)

    def test_a_dry_run_never_claims(self):
        relayer = _FakeRelayer()
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=True)
        redeemer._relayer = relayer
        position = RedeemablePosition(condition_id="0x" + "55" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5)
        report = redeemer.redeem([position])
        assert report.claimed == 0
        assert relayer.executed == [], "a dry run submitted a claim"
        assert report.unclaimed_value_usd == pytest.approx(5.0)

    def test_a_claim_that_does_not_confirm_is_not_claimed(self):
        """
        Submitted is not redeemed. An unconfirmed transaction is reported
        unclaimed so the agent does not book locked collateral as spendable.
        """
        relayer = _FakeRelayer(state="STATE_NEW")
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = relayer
        position = RedeemablePosition(condition_id="0x" + "66" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5)
        report = redeemer.redeem([position])
        assert report.claimed == 0
        assert report.failed == 1
        assert report.unclaimed_value_usd == pytest.approx(5.0)
        assert "confirm" in report.unclaimed[0]["reason"]

    def test_a_position_the_venue_does_not_mark_redeemable_is_skipped(self):
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = _FakeRelayer()
        position = RedeemablePosition(condition_id="0x" + "77" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5,
                                      redeemable=False)
        report = redeemer.redeem([position])
        assert report.skipped == 1 and report.attempted == 0
        assert redeemer._relayer.executed == []

    def test_redeemable_positions_are_read_from_the_venue(self):
        payload = [{"conditionId": "0xcond", "asset": "tok1", "outcome": "Yes",
                    "size": 12.0, "currentValue": 12.0, "negativeRisk": True,
                    "title": "Will it happen?"}]
        session = _FakeSession(payload)
        redeemer = Redeemer(funder="0xwallet", session=session)
        positions, available, reason = redeemer.read_redeemable()

        assert available is True and reason == ""
        assert len(positions) == 1
        assert positions[0].neg_risk is True
        assert positions[0].current_value_usd == 12.0
        url, params = session.calls[0]
        assert params["user"] == "0xwallet" and params["redeemable"] == "true"
        assert params["sizeThreshold"] == 0, "dust positions must be included"

    def test_an_unreachable_api_is_reported_as_unavailable(self):
        """
        "Nothing to claim" and "could not find out" are different facts. Only one
        of them justifies leaving the collateral alone.
        """
        class Failing:
            def get(self, *a, **k):
                raise RuntimeError("connection refused")

        redeemer = Redeemer(funder="0xwallet", session=Failing())
        positions, available, reason = redeemer.read_redeemable()
        assert positions == []
        assert available is False
        assert "connection refused" in reason

    def test_a_claimed_redemption_is_recorded_once(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "redeem.db"))
        relayer = _FakeRelayer()
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = relayer
        position = RedeemablePosition(condition_id="0x" + "88" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5)
        report = redeemer.redeem([position])
        redeemer.record(storage, report, [position])

        assert redeemer.already_redeemed(storage, position.condition_id) is True
        assert redeemer.already_redeemed(storage, "0xnope") is False
        row = storage.conn.execute(
            "SELECT value_usd, transaction_id FROM redemptions").fetchone()
        assert row["value_usd"] == pytest.approx(5.0)
        assert row["transaction_id"] == "tx-1"

    def test_a_failed_claim_is_retried_rather_than_recorded(self, tmp_path):
        """
        Redeeming is idempotent at the venue, so a failed claim must stay
        outstanding. Recording it would leave the winnings locked forever.
        """
        storage = Storage(db_path=str(tmp_path / "retry.db"))
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = _FakeRelayer(state="STATE_NEW")
        position = RedeemablePosition(condition_id="0x" + "99" * 32, asset="a",
                                      outcome="YES", size=5, current_value_usd=5)
        report = redeemer.redeem([position])
        redeemer.record(storage, report, [position])
        assert redeemer.already_redeemed(storage, position.condition_id) is False

    def test_every_position_is_either_claimed_or_explained(self):
        """
        The accounting identity of this module: nothing may simply vanish.
        """
        redeemer = Redeemer(funder="0xf", private_key="0xk", dry_run=False)
        redeemer._relayer = _FakeRelayer()
        positions = [
            RedeemablePosition(condition_id="0x" + "aa" * 32, asset="a",
                               outcome="YES", size=1, current_value_usd=1.0),
            RedeemablePosition(condition_id="0x" + "bb" * 32, asset="b",
                               outcome="NO", size=1, current_value_usd=2.0,
                               redeemable=False),
            RedeemablePosition(condition_id="0x" + "cc" * 32, asset="c",
                               outcome="YES", size=1, current_value_usd=3.0,
                               neg_risk=True),
        ]
        report = redeemer.redeem(positions)
        accounted = {i["condition_id"] for i in report.items}
        assert accounted == {p.condition_id for p in positions}
        assert report.claimed + report.failed + report.skipped == len(positions)
        assert report.claimed_value_usd + report.unclaimed_value_usd == pytest.approx(
            report.claimable_value_usd)
