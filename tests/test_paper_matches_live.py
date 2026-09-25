"""
Paper mode must decide the way live mode decides.

Seventh report, item 2: paper realism at least as rich as the live executor
before qualification trusts it. The probe (see /tmp/probe_paper_vs_live.py)
proved three gaps on the one venue that has a live path at all:

  1. an order the venue's own rules reject (minimum size / notional) was
     REFUSED by the live path and FILLED by the paper path - paper was
     trading executions that cannot exist;
  2. the live path books the venue's declared taker fee, the simulated fill
     showed 0 - the evidence was kinder than the ledger;
  3. a live partial reports size_matched / original_size and locks the
     remainder in the order table; a paper partial dropped the remainder -
     the same live order locks cash the paper order left free.

The third fix gives paper orders the live lifecycle: recorded in the order
table, reserved, re-simulated against the current book each cycle, filled
when the market crosses, released when final. `simulate_resting` gets its
first production caller.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptai.markets.base import Market, MarketSource, Token
from ptai.markets.mechanics import MarketMechanics
from ptai.venues.adapter import VenueOpportunity, VenueType
from ptai.venues.polymarket_adapter import PolymarketAdapter
from ptai.execution.multi_venue_executor import MultiVenueExecutor
from ptai.execution.order_manager import OrderManager
from ptai.storage.db import Storage
from ptai.agent.v3_loop import _side_token_id


# ----------------------------------------------------------------- fixtures

def _market():
    return Market(
        id="PARITY-1", source=MarketSource.POLYMARKET,
        question="Will the parity probe fill?", outcomes=["YES", "NO"],
        outcome_prices=[0.60, 0.40],
        tokens=[Token(token_id="T-YES", outcome="YES", price=0.60),
                Token(token_id="T-NO", outcome="NO", price=0.40)],
        volume=200_000.0, volume_24h=100_000.0, liquidity=50_000.0,
        active=True, closed=False, slug="parity-1",
        raw={"venue": "polymarket"},
    )


def _opp(market, side="YES"):
    price = 0.60 if side == "YES" else 0.40
    return VenueOpportunity(
        market=market, venue_id="polymarket",
        venue_type=VenueType.PREDICTION, side=side, market_price=price,
        estimated_fair=0.65 if side == "YES" else 0.35, raw_edge=0.05,
        effective_edge=0.05, confidence=0.8, liquidity_score=0.9,
        execution_quality=0.9, category="test", score=0.05,
        should_trade=True, market_id=market.id, data_mode="live",
        raw={"strategy": "value"},
    )


def _book(ask="0.60", ask_size="10000", bid="0.59"):
    return {"bids": [{"price": bid, "size": "10000"}],
            "asks": [{"price": ask, "size": ask_size}],
            "spread": round(float(ask) - float(bid), 4),
            "is_real": True, "source": "probe"}


def _adapter(book=None, mechanics=None):
    a = PolymarketAdapter()  # dry_run, no credentials: the paper path
    a.client.get_orderbook = lambda token_id: dict(book or _book())
    if mechanics is not None:
        a.get_mechanics = lambda opportunity=None, token_id=None: mechanics
    return a


def _executor(adapter):
    class _Reg:
        adapters = {"polymarket": adapter}
        def get_adapter_for_venue_id(self, venue_id):
            return self.adapters.get(venue_id)
    return MultiVenueExecutor(registry=_Reg(), bankroll=50.0)


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------- 1. same decision boundary

class TestPaperRefusesWhatLiveRefuses:
    def test_min_size_refusal_matches_live(self):
        m = MarketMechanics(tick_size="0.01", min_order_size=12.0,
                            min_order_notional_usd=1.0,
                            source="clob_market_info", is_real=True)
        limit = m.round_price(0.60, "BUY")
        size = m.shares_for_usd(3.0, limit, "BUY")
        ok, live_reason = m.validate_order(limit, size)
        assert not ok, "fixture: the live path must refuse this order"
        result = _run(_adapter(mechanics=m).place_paper_order(
            _opp(_market()), 3.0, 0.60))
        assert result["status"] == "rejected"
        assert result["reason"] == live_reason

    def test_min_notional_refusal_matches_live(self):
        m = MarketMechanics(tick_size="0.01", min_order_size=0.0,
                            min_order_notional_usd=1.0,
                            source="clob_market_info", is_real=True)
        limit = m.round_price(0.60, "BUY")
        size = m.shares_for_usd(0.5, limit, "BUY")
        ok, live_reason = m.validate_order(limit, size)
        assert not ok, "fixture: $0.50 is below the venue notional floor"
        result = _run(_adapter(mechanics=m).place_paper_order(
            _opp(_market()), 0.5, 0.60))
        assert result["status"] == "rejected"
        assert result["reason"] == live_reason

    def test_an_order_live_accepts_still_fills_in_paper(self):
        m = MarketMechanics(tick_size="0.01", min_order_size=5.0,
                            min_order_notional_usd=1.0,
                            source="clob_market_info", is_real=True)
        limit = m.round_price(0.60, "BUY")
        ok, _ = m.validate_order(limit, m.shares_for_usd(3.0, limit, "BUY"))
        assert ok, "fixture: the live path must accept this order"
        result = _run(_adapter(mechanics=m).place_paper_order(
            _opp(_market()), 3.0, 0.60))
        assert result["status"] == "paper"
        assert float(result["filled_usd"]) > 0


# ------------------------------------------------------- 2. the same fee

class TestPaperChargesTheFeeLiveBooks:
    def test_declared_rate_when_the_venue_declares_no_real_rate(self):
        # Mechanics without venue fee data: the rate in capabilities - the
        # same one calculate_fees() books for the live fill.
        a = _adapter()
        result = _run(a.place_paper_order(_opp(_market()), 3.0, 0.60))
        declared = float(a.capabilities.fee_taker_pct)
        assert declared > 0, "fixture: the venue must declare a rate"
        assert float(result["fees_usd"]) == pytest.approx(
            float(result["filled_usd"]) * declared, rel=1e-6)

    def test_a_real_venue_rate_wins(self):
        m = MarketMechanics(tick_size="0.01", taker_fee_rate=0.01,
                            source="clob_market_info", is_real=True)
        result = _run(_adapter(mechanics=m).place_paper_order(
            _opp(_market()), 3.0, 0.60))
        assert float(result["fees_usd"]) == pytest.approx(
            float(result["filled_usd"]) * 0.01, rel=1e-6)


# -------------------------------------------------- 3. the same remainder

class TestPaperPartialReportsTheRemainder:
    def test_partial_fill_carries_the_live_numbers(self):
        a = _adapter(book=_book(ask_size="2"))  # $1.20 of size at the touch
        result = _run(a.place_paper_order(_opp(_market()), 3.0, 0.60))
        assert result["status"] == "paper"
        assert float(result["filled_usd"]) == pytest.approx(1.20)
        assert float(result["size_matched"]) == pytest.approx(2.0)
        assert float(result["original_size"]) == pytest.approx(5.0)
        assert float(result["resting_usd"]) == pytest.approx(1.80)
        assert float(result["resting_limit_price"]) == pytest.approx(0.60)

    def test_execution_result_reserves_the_remainder(self):
        ex = _executor(_adapter(book=_book(ask_size="2")))
        result = _run(ex.execute_single(_opp(_market()), 3.0, 0.60))
        assert result.status == "dry_run"
        assert result.filled_usd == pytest.approx(1.20)
        assert result.committed_capital is False
        assert result.unfilled_shares == pytest.approx(5.0 - 2.0, abs=1e-6)
        assert result.reserves_capital is True
        assert result.resting_usd == pytest.approx(1.80)
        assert result.limit_price == pytest.approx(0.60)


# ------------------------------------------- 4. the resting-order lifecycle

class _BookReader:
    def __init__(self, book):
        self.book = book

    async def get_token_orderbook(self, token_id):
        return {"available": True, "book": dict(self.book), "is_real": True}


class TestPaperRestingOrderLifecycle:
    def _place_resting_order(self, storage, om, book):
        ex = _executor(_adapter(book=book))
        er = _run(ex.execute_single(_opp(_market()), 3.0, 0.50))
        assert er.status == "dry_run"
        assert er.filled_usd == 0.0, "0.50 does not cross a book at 0.60"
        assert er.reserves_capital is True
        key = om.record_submission(
            er, market_id="PARITY-1", token_id="T-YES",
            venue_id="polymarket", side="YES",
            forecast={"execution_mode": "paper", "fair_price": 0.55,
                      "strategy": "value"})
        assert key
        return key

    def test_rests_reserves_then_crosses_then_releases(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "parity.db"))
        om = OrderManager(storage=storage)
        key = self._place_resting_order(storage, om, book=_book(ask="0.60"))
        assert storage.resting_capital_usd() == pytest.approx(3.0)

        # The market has not come to 0.50: it stays open, still reserved.
        report = _run(om.reconcile(_BookReader(_book(ask="0.60")),
                                   venue_id="polymarket"))
        assert report.unreconciled == 0
        assert report.released == 0
        assert storage.resting_capital_usd() == pytest.approx(3.0)

        # The market crosses to 0.50: the order fills at its limit, the
        # position it was for is opened, the capital is released.
        opened = {}
        def opener(order, add_usd, price):
            opened["usd"] = add_usd
            opened["price"] = price
            return 999
        report = _run(om.reconcile(_BookReader(_book(ask="0.50", ask_size="100")),
                                   venue_id="polymarket",
                                   position_opener=opener))
        assert opened["usd"] == pytest.approx(3.0)
        assert opened["price"] == pytest.approx(0.50)
        assert report.filled_more == 1
        assert storage.resting_capital_usd() == 0.0
        row = storage.get_order_row(key)
        assert row["status"] == "filled"
        assert row["trade_id"] == 999

    def test_an_unreadable_book_keeps_the_reservation(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "parity.db"))
        om = OrderManager(storage=storage)
        key = self._place_resting_order(storage, om, book=_book(ask="0.60"))

        class _DeadVenue:
            async def get_token_orderbook(self, token_id):
                return {"available": False, "reason": "boom"}

        report = _run(om.reconcile(_DeadVenue(), venue_id="polymarket"))
        assert report.unreconciled == 1
        assert report.released == 0
        assert storage.resting_capital_usd() == pytest.approx(3.0)

    def test_a_live_venue_id_order_is_not_routed_to_paper(self, tmp_path):
        # execution_mode "live" (or absent) must never be re-simulated
        # locally: the venue is asked.
        storage = Storage(db_path=str(tmp_path / "parity.db"))
        om = OrderManager(storage=storage)
        storage.upsert_order({
            "order_id": "venue-order-1", "market_id": "PARITY-1",
            "token_id": "T-YES", "side": "YES", "limit_price": 0.50,
            "requested_usd": 3.0, "status": "submitted",
            "venue_id": "polymarket", "original_size": 6.0,
            "size_matched": 0.0, "matched_usd": 0.0,
            "execution_mode": "live",
        })

        class _Venue:
            async def get_order(self, order_id):
                return {"available": True, "status": "cancelled"}
            async def get_open_orders(self, *a, **k):
                return {"available": True, "orders": []}

        report = _run(om.reconcile(_Venue(), venue_id="polymarket"))
        row = storage.get_order_row("venue-order-1")
        assert row["status"] == "cancelled"
        assert report.released == 1


# ------------------------------------------- end to end, through the loop

class TestPaperPartialEndToEnd:
    def test_partial_paper_fill_splits_position_and_resting_order(
            self, tmp_path, monkeypatch):
        """
        $3.00 requested against $1.20 of reachable depth. A live partial fill
        produces a position for the filled part and an order reserving the
        rest; the paper partial must produce the same split, or the ledger
        shows more free capital than the same live trade would leave free.
        """
        import tests.test_full_cycle_from_discovery_to_allocation as base
        from src.ptai.execution.position_ledger import PositionLedgerBuilder

        agent, adapter = base.TestPaperModeSimulatesTheWholeCycle()._build(
            tmp_path, book={"asks": [{"price": "0.56", "size": "2"}]})
        base._force_qualified(agent, monkeypatch)
        base._inject_opportunity(agent, adapter._market(), monkeypatch)

        result = base._cycle(agent)
        assert result["execution"], "the cycle recorded no execution at all"
        fill = result["execution"][0].get("fill") or {}
        requested, filled = fill["requested_usd"], fill["filled_usd"]
        assert requested == pytest.approx(3.00)
        assert filled == pytest.approx(1.12), "2 shares at 0.56 = $1.12"
        assert 0 < filled < requested, "the fill must be genuinely short"

        # The position holds what filled.
        positions = agent.storage.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["position_size_usd"] == pytest.approx(1.12, abs=0.01)

        # The order holds the rest, and the capital ledger sees it. Before
        # this fix the remainder was invisible: the same request would have
        # left free cash $1.88 higher than the live venue leaves it.
        #
        # (Paper POSITIONS are an evidence pool the ledger deliberately does
        # not charge against live free cash - the separate paper bankroll is
        # part of the capital-allocation layer, not this one. The working
        # order is not paper decoration: it is cash promised to an order, and
        # it counts.)
        assert agent.storage.resting_capital_usd() == pytest.approx(1.88, abs=0.01)
        ledger = PositionLedgerBuilder(storage=agent.storage).build()
        assert ledger.resting_order_cost == pytest.approx(1.88, abs=0.01)
        assert ledger.paper_position_count == 1
        assert ledger.free_cash == pytest.approx(50.0 - 1.88, abs=0.02)


# -------------------------------------------------- the token of the side

class TestOrderCarriesTheSideToken:
    def test_side_tokens(self):
        assert _side_token_id(_market(), "YES") == "T-YES"
        assert _side_token_id(_market(), "NO") == "T-NO"
        assert _side_token_id(_market(), "no") == "T-NO"

    def test_falls_back_to_the_first_token(self):
        m = _market()
        m.tokens = [Token(token_id="T-X", outcome="", price=0.5)]
        assert _side_token_id(m, "NO") == "T-X"
