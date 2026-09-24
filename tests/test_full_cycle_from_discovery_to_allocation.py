"""
The whole loop in one run: discovery -> sizing -> fill -> position -> settlement
-> P&L -> learning -> allocation.

The five link tests each prove one joint. This proves the CHAIN, because the
failure mode the operator is protecting against is not a broken joint - it is a
joint that works in isolation and is never actually reached by the loop. That is
exactly why the qualification gate could silently invert itself, why the
executor could report a refusal as a fill, and why the loop could record a
position that nothing ever settled.

Note what this test does NOT claim. Cards fill and settlement computes P&L, so
this proves the loop ACCOUNTS for a trade correctly. It proves nothing about
whether the trade had an edge. Accounting for a losing trade correctly is a
success for this test and a failure for the strategy; only out-of-sample
expectancy, measured in real paper trading, can speak to that.

The venue is a stub, so "real data" here means the shape, units, timing and
pricing of a real feed, not a live socket. Step 6 of the operator's sequence -
running this same chain against a real venue's live feed - is what turns this
from a wiring proof into an evidence proof.
"""

from __future__ import annotations

import asyncio
import pytest

from src.ptai.agent.v3_loop import TradingAgentV3
from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.storage.db import Storage
from src.ptai.venues.registry import VenueRegistry
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueType,
)

VENUE = "polymarket"
MARKET_ID = "STUB-M1"


class StubVenue(MarketAdapter):
    """
    A venue with the shape of a real one: it discovers a market, quotes a book,
    accepts an order and reports a fill with a size and a price, and can report
    settlement once the market closes.

    It is deliberately NOT optimistic. `fill` says what actually happened, and
    the test drives the unhappy paths (refusal, unquantified fill, unsettled
    market) to prove the loop reacts to them.
    """

    def __init__(self, dry_run: bool = True, fill_response=None):
        # Registered under the real Polymarket venue id, and its markets carry
        # MarketSource.POLYMARKET, because the executor requires the adapter's
        # venue id and the market's source to agree. A stub that violates that
        # invariant gets correctly aborted by the executor, which is a useful
        # safety property but tests nothing about the accounting chain.
        super().__init__(venue_id=VENUE, venue_type=VenueType.PREDICTION,
                         dry_run=dry_run)
        self.capabilities = AdapterCapability(
            supports_market_discovery=True,
            supports_orderbook=True,
            supports_trading=True,
            supports_portfolio=True,
            supports_history=True,
            # No order probe: this venue cannot really place a live order,
            # so it must not claim it can prove permission.
            supports_order_probe=False,
            implementation_status="live",
        )
        self.fill_response = fill_response or {
            "status": "matched", "orderID": "stub-order-1",
            "size": 2.50, "price": 0.55,
        }
        self.settled_outcome = None
        self.orders_placed = 0
        if not dry_run:
            # Credential-shaped attributes, so the health engine sees a
            # configured venue instead of a paper-only one. The value is
            # irrelevant to the stub and no network call is made.
            self.private_key = "0x" + "ab" * 32
            self.funder = "0x" + "cd" * 20

    def check_eligibility(self, country_code: str = "UG") -> EligibilityStatus:
        return EligibilityStatus.ELIGIBLE

    async def get_portfolio(self):
        return {"available": True, "balance": 50.0, "source": "stub_live"}

    async def discover_markets(self, target_count: int = 500, filters=None):
        return [self._market()]

    def _market(self) -> Market:
        return Market(
            id=MARKET_ID,
            source=MarketSource.POLYMARKET,
            question="Will the stub event happen?",
            outcomes=["YES", "NO"],
            outcome_prices=[0.55, 0.45],
            tokens=[Token(token_id="STUB-YES", outcome="YES", price=0.55),
                    Token(token_id="STUB-NO", outcome="NO", price=0.45)],
            volume=250_000.0, volume_24h=120_000.0, liquidity=90_000.0,
            active=True, closed=False, slug="stub-m1",
            raw={"venue": VENUE},
        )

    async def get_orderbook(self, market: Market):
        return {"bids": [{"price": "0.54", "size": "5000"}],
                "asks": [{"price": "0.56", "size": "5000"}],
                "market": market.id}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        self.orders_placed += 1
        if self.dry_run:
            # A real adapter refuses to send a live order while in dry run, and
            # reports the standard refusal so the caller can record and learn
            # from a PAPER position. A stub that returned a "matched" fill while
            # in dry run would let the loop record simulated money as real.
            return self.real_order_refusal(max_spend_usd, max_price,
                                           opportunity.side, opportunity.market.id)
        return dict(self.fill_response)

    async def get_settlement(self, market_id: str):
        # `is_real` is how an adapter certifies that this verdict came from the
        # venue rather than from a guess. Settlement refuses anything without it,
        # so a stub that omits it is correctly treated as unreadable.
        if self.settled_outcome is None:
            return {"settled": False, "is_real": True,
                    "reason": "market still open"}
        return {"settled": True, "is_real": True,
                "outcome": self.settled_outcome, "source": "stub"}

    def calculate_fees(self, market, amount_usd): return 0.0
    def estimate_slippage(self, market, amount_usd): return 0.0
    def estimate_spread(self, orderbook): return 0.02


def stub_registry(adapter):
    """
    A real VenueRegistry holding one stub adapter.

    Using the real registry rather than a stand-in keeps routing, eligibility
    and adapter lookup on the code path that production uses - so a bug in
    routing cannot hide behind the test double.
    """
    registry = VenueRegistry(country_code="UG")
    registry.register(adapter)
    return registry


def _repoint_storage(agent, storage):
    """Point every storage-holding component at the same database."""
    for name in ("ledger_builder", "settlement_engine", "trade_outcome_tracker",
                 "calibration_engine"):
        component = getattr(agent, name, None)
        if component is not None and hasattr(component, "storage"):
            component.storage = storage
    # These components cache what they loaded at construction time and would
    # otherwise keep serving the repo database's contents.
    tracker = getattr(agent, "trade_outcome_tracker", None)
    if tracker is not None:
        tracker.outcomes = []
        if hasattr(tracker, "_load"):
            tracker._load()
    calibration = getattr(agent, "calibration_engine", None)
    if calibration is not None:
        calibration.points = []
        if hasattr(calibration, "load_from_storage"):
            calibration.load_from_storage()


def build_agent(tmp_path, fill_response=None, dry_run=False, bankroll=50.0):
    adapter = StubVenue(dry_run=dry_run, fill_response=fill_response)
    agent = TradingAgentV3(country_code="UG", dry_run=dry_run)
    agent.storage = Storage(db_path=str(tmp_path / "cycle.db"))
    agent.storage.set_bankroll(bankroll)
    # The registry must be swapped EVERYWHERE it was captured. V3 passes its
    # registry into the executor, the health engine and the capability engine at
    # construction time, so replacing only agent.venue_registry leaves those
    # holding the original - and the cycle then runs against the real adapters
    # while appearing to use the stub.
    # Storage must be swapped EVERYWHERE it was captured, or the cycle writes
    # positions to the test database while the tracker, calibration store and
    # ledger read the repo's data/ptai.db - and a test then sees another test's
    # outcomes. That is not a hypothetical: it is how a refused order appeared
    # to produce a learning record.
    _repoint_storage(agent, agent.storage)

    agent.venue_registry = stub_registry(adapter)
    agent.multi_venue_executor.registry = agent.venue_registry
    agent.account_health_engine.venue_registry = agent.venue_registry
    agent.capability_engine.venue_registry = agent.venue_registry
    agent.settlement_engine.venue_registry = agent.venue_registry
    # The qualification engine needs 100+ real paper trades before it will
    # qualify anything. That gate is what we WANT in production; here we inject
    # the qualified set so the downstream chain can be exercised at all.
    agent._qualified_venue_ids = [VENUE]
    return agent, adapter


def _force_qualified(agent, monkeypatch):
    """
    Present the stub venue as already qualified, without weakening the gate.

    The gate itself is verified separately; leaving the engine's real 100-trade
    requirement in place here would mean never reaching the execution chain.
    """
    async def evaluate_all_venues(self, target_per_venue=20, **kwargs):
        return _report(qualified=[VENUE])
    monkeypatch.setattr(
        type(agent.capability_engine), "evaluate_all_venues",
        evaluate_all_venues, raising=True)


def _report(qualified):
    from src.ptai.venues.capability_engine import QualificationEngineReport
    return QualificationEngineReport(
        total_venues=1,
        qualified_venues=len(qualified),
        data_only_venues=0,
        restricted_venues=0,
        untested_venues=0,
        venue_reports=[],
        strategy_reports=[],
        qualified_venue_ids=list(qualified),
        recommended_venues=list(qualified),
        execution_time=0.0,
        reasoning=("stub venue pre-qualified for the chain test" if qualified
                   else "nothing qualified"),
    )


def _inject_opportunity(agent, market, monkeypatch, edge=0.15):
    """
    Feed the chain one opportunity, exactly as the strategy engine would.

    Only the DISCOVERY of the edge is stubbed. Everything the operator is
    auditing - sizing, the risk gates, execution, fill reading, position
    recording, settlement, P&L, learning, allocation - runs as production code.
    Finding the edge is what step 6 (real data, out of sample) has to prove, and
    this test deliberately does not claim to have done it: the edge here is
    asserted, not discovered.
    """
    from src.ptai.strategy.strategy_engine import MultiVenueScanResult
    from src.ptai.venues.adapter import VenueOpportunity

    def make():
        return VenueOpportunity(
            market=market, venue_id=VENUE, venue_type=VenueType.PREDICTION,
            side="YES", market_price=0.55, estimated_fair=0.55 + edge,
            raw_edge=edge, effective_edge=edge, confidence=0.75,
            liquidity_score=0.9, execution_quality=0.9,
            category="politics", score=edge, should_trade=True,
            market_id=market.id, data_mode="live",
            raw={"strategy": "value", "venue": VENUE},
        )
    result = MultiVenueScanResult(
        total_scanned=1, total_after_cheap=1, total_after_liquidity=1,
        total_candidates=1, total_tradeable=1,
        venue_reports=[], all_opportunities=[make()],
        final_selected=[make()], best_opportunity=make(),
        reasoning="injected opportunity",
    )

    async def scan_all_venues(self, markets_by_venue, context_provider=None,
                              max_final_trades=3):
        return result

    monkeypatch.setattr(type(agent.strategy_engine_v3), "scan_all_venues",
                        scan_all_venues, raising=True)


def _execute(agent):
    return agent._last_execution


def _cycle(agent, **kwargs):
    return asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=1, **kwargs))


class TestFullCycle:
    def test_discovery_through_settlement_to_pnl(self, tmp_path, monkeypatch):
        """
        One real venue through the full cycle: discover -> size -> fill ->
        position -> settle -> net P&L -> store permanently.
        """
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            # --- cycle 1: discover, size, execute, record -----------------
            r1 = _cycle(agent)

            assert r1["core_objective_execution"]["step1_qualified_venues"] == [VENUE]
            assert adapter.orders_placed >= 1, "no order reached the venue"
            assert len(r1["execution"]) == 1, (
                f"expected one execution entry, got {len(r1['execution'])}; "
                f"reasoning: {r1.get('reasoning')}"
            )
            assert r1["execution"][0]["position_recorded"] is True

            # the position exists, at the FILL, not at the request
            open_positions = agent.storage.get_open_positions()
            assert len(open_positions) == 1, "the position was not recorded"
            pos = open_positions[0]
            assert pos["market_id"] == MARKET_ID
            fill = r1["execution"][0]["fill"]
            assert fill["status"] == "filled"
            # 2.50 shares at 0.55 = $1.375 committed, NOT the $3.00 requested.
            # The requested figure must never appear as the fill.
            assert fill["filled_usd"] == pytest.approx(1.375)
            assert fill["requested_usd"] == pytest.approx(3.0)
            assert fill["filled_usd"] != pytest.approx(fill["requested_usd"])
            assert fill["filled_price"] == pytest.approx(0.55)
            assert fill["order_id"] == "stub-order-1"

            # sizing came from free capital under the 6% cap
            # Live capital is reserved, and free cash falls by what was
            # actually committed - not by what was requested.
            after = agent.ledger_builder.build()
            assert after.live_position_count == 1
            assert after.paper_position_count == 0
            assert after.reserved_capital == pytest.approx(1.375), (
                "the ledger did not reserve the committed capital"
            )
            assert after.reserved_capital <= 50.0 * 0.06 + 1e-9, (
                f"reserved {after.reserved_capital} exceeds the 6% cap"
            )
            assert after.free_cash == pytest.approx(50.0 - 1.375), (
                "free capital did not fall by the committed amount"
            )

            # learning recorded the outcome, with the venue attached
            outcomes = agent.trade_outcome_tracker.outcomes
            assert len(outcomes) == 1, (
                "execution did not produce a learning outcome, so future "
                "allocation has nothing to learn from"
            )
            assert outcomes[0].venue_id == VENUE
            assert outcomes[0].strategy, "the outcome is not attributable to a strategy"
            assert outcomes[0].actual_outcome is None, "settled before settlement ran"

            # The FILL, not the request, is what the trade row carries.
            row = agent.storage.conn.execute(
                "SELECT market_price, position_size_usd FROM trades WHERE market_id=?",
                (MARKET_ID,)).fetchone()
            assert row["market_price"] == pytest.approx(0.55), (
                "the trade was recorded at the requested price, not the fill"
            )

            # --- cycle 2: the venue reports the market resolved YES -------
            adapter.settled_outcome = 1.0
            r2 = _cycle(agent)

            settlement = r2["settlement"]
            assert settlement["settled"] == 1, (
                f"settlement closed {settlement['settled']} positions: "
                f"{settlement}"
            )

            # the position is closed and P&L is real
            assert agent.storage.get_open_positions() == []
            assert agent.storage.count_open_positions() == 0
            resolved = agent.storage.conn.execute(
                "SELECT pnl, resolved FROM trades WHERE status='settled'").fetchall()
            assert len(resolved) == 1
            pnl = resolved[0][0]
            # $1.375 committed at 0.55 buys exactly 2.5 shares. A YES win pays
            # 2.5, so the profit is 2.5 - 1.375 = 1.125. Computed from the FILL,
            # not the request: had the row carried the requested $3.00, the
            # profit would have been overstated by more than double.
            assert pnl == pytest.approx(1.125, abs=0.01), f"unexpected P&L {pnl}"

            # --- the outcome was used to alter future decisions -----------
            assert outcomes[0].actual_outcome == pytest.approx(1.0)
            assert outcomes[0].pnl == pytest.approx(pnl, abs=0.02)
            assert outcomes[0].brier_score is not None
            assert r2["capital"]["realised_pnl"] == pytest.approx(pnl, abs=0.02)
            assert r2["capital"]["reserved_capital"] == 0.0, (
                "a resolved position still reserves capital"
            )

            # --- and it survives a restart --------------------------------
            agent.storage.close()
            reopened = Storage(db_path=str(tmp_path / "cycle.db"))
            try:
                from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
                tracker = TradeOutcomeTracker(storage=reopened)
                assert len(tracker.outcomes) == 1
                assert tracker.outcomes[0].actual_outcome == pytest.approx(1.0)
                assert tracker.get_venue_performance()[VENUE]["total"] == 1
                assert reopened.count_open_positions() == 0
            finally:
                reopened.close()
        finally:
            agent.storage.close()

    def test_a_refused_order_never_becomes_a_position(self, tmp_path, monkeypatch):
        """
        The defect: a rejected order still created an open position at the
        requested size, which settlement later closed against a real outcome and
        booked a P&L for a trade that never happened.
        """
        agent, adapter = build_agent(
            tmp_path, fill_response={"status": "rejected", "reason": "blocked"})
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            r = _cycle(agent)
            assert r["execution"][0]["position_recorded"] is False
            assert "committed no capital" in r["execution"][0]["position_reason"]
            assert agent.storage.get_open_positions() == [], (
                "a refused order created a position"
            )
            assert agent.trade_outcome_tracker.outcomes == [], (
                "a refused order produced a learning outcome"
            )
        finally:
            agent.storage.close()

    def test_an_unquantified_fill_never_becomes_a_position(self, tmp_path, monkeypatch):
        """
        The venue says it took the order but reports no size. Capital we cannot
        quantify must not be treated as capital we committed.
        """
        agent, adapter = build_agent(
            tmp_path, fill_response={"status": "filled", "orderID": "o1"})
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            r = _cycle(agent)
            entry = r["execution"][0]
            assert entry["executor_result"]["status"] == "unknown"
            assert entry["position_recorded"] is False, (
                "an unquantified fill created a position"
            )
            assert agent.storage.get_open_positions() == []
        finally:
            agent.storage.close()

    def test_an_unsettled_market_keeps_the_position_open(self, tmp_path, monkeypatch):
        """
        Only the venue can report settlement. An unreadable or still-open market
        must leave the position open rather than guessing an outcome, or the
        agent books a P&L it invented.
        """
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            _cycle(agent)
            assert len(agent.storage.get_open_positions()) == 1

            adapter.settled_outcome = None  # still open
            r = _cycle(agent)
            assert r["settlement"]["settled"] == 0
            assert len(agent.storage.get_open_positions()) == 1, (
                "the position closed on a market that has not settled"
            )
            assert agent.trade_outcome_tracker.outcomes[0].actual_outcome is None
        finally:
            agent.storage.close()

    def test_a_losing_trade_is_accounted_for_honestly(self, tmp_path, monkeypatch):
        """
        A correct loss is recorded as a loss. An accounting layer that only
        handles wins looks identical to a profitable one until the money is
        already gone.
        """
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            _cycle(agent)
            adapter.settled_outcome = 0.0
            r = _cycle(agent)
            assert r["settlement"]["settled"] == 1
            pnl = agent.storage.conn.execute(
                "SELECT pnl FROM trades WHERE status='settled'").fetchone()[0]
            assert pnl < 0, "a losing position was not recorded as a loss"
            assert agent.trade_outcome_tracker.outcomes[0].was_correct is False
        finally:
            agent.storage.close()

    def test_equity_reflects_settlement_across_cycles(self, tmp_path, monkeypatch):
        """
        Bankroll must track reality: after a win, the next cycle sizes against
        more capital; after a loss, against less.
        """
        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            before = _cycle(agent)["capital"]["equity"]
            adapter.settled_outcome = 1.0
            after = _cycle(agent)["capital"]["equity"]
            assert after > before, (
                f"equity did not rise after a settled win ({before} -> {after}), "
                "so the agent is sizing against a stale bankroll"
            )
        finally:
            agent.storage.close()

    def test_does_nothing_when_the_venue_is_not_qualified(self, tmp_path, monkeypatch):
        """
        Fail closed. The original gate inverted itself here, treating an empty
        qualified set as 'everything is qualified'.
        """
        agent, adapter = build_agent(tmp_path)
        async def no_qualifications(self, target_per_venue=20, **kwargs):
            return _report(qualified=[])
        monkeypatch.setattr(type(agent.capability_engine),
                            "evaluate_all_venues", no_qualifications)
        try:
            r = _cycle(agent)
            assert r["core_objective_execution"]["step1_qualified_venues"] == []
            assert r["execution"] == [], (
                "capital was deployed with no qualified venue"
            )
            listed = agent.storage.conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status != 'paper'").fetchone()[0]
            assert listed == 0, "a live trade was recorded with nothing qualified"
        finally:
            agent.storage.close()

    def test_paper_positions_do_not_consume_live_capital(self, tmp_path, monkeypatch):
        """
        Paper trading is how a venue earns qualification. It must build the
        track record without touching the money.

        Note what changed: the dry-run venue now SIMULATES a fill against its
        real book before a paper position exists. It used to return a bare
        refusal, and the loop booked a paper position at the REQUESTED size for
        it - so the track record this test protects was built on fills that were
        never priced. The qualification gate would then have been fed positions
        the venue never produced.
        """
        agent, adapter = build_agent(tmp_path, dry_run=True)
        adapter.place_order = _simulating_place_order(adapter)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        try:
            r = _cycle(agent)
            assert r["execution"][0]["position_recorded"] is True
            status = agent.storage.conn.execute(
                "SELECT status FROM trades WHERE market_id=?", (MARKET_ID,)).fetchone()[0]
            assert status == "paper", (
                f"a dry-run execution was recorded as '{status}', so simulated "
                "trades would count as real P&L"
            )
            # `r["capital"]` is the snapshot taken BEFORE the trade, so the
            # fresh ledger is what shows the recorded position.
            after = agent.ledger_builder.build()
            assert after.paper_position_count == 1
            assert after.live_position_count == 0
            assert after.reserved_capital == 0.0
            assert after.equity == pytest.approx(50.0), (
                "a paper position moved live equity"
            )
        finally:
            agent.storage.close()


class TestSettlementDoesNotDependOnCalibration:
    """
    Settlement used to be a side effect of the CALIBRATION store rather than a
    responsibility of the position ledger:

      * `if not pending: return report` - with no unresolved forecasts, NO trade
        was settled. A fresh install, or any run after the calibration store was
        cleared, left every open position open forever.
      * the loop iterated pending forecasts only, so an open position whose
        market had no forecast entry was invisible to settlement.

    Either way a position could be opened, never closed, and never contribute a
    P&L: the agent would keep accounting for capital it had already lost.
    """

    def test_a_position_with_no_forecast_still_settles(self, tmp_path):
        from src.ptai.execution.settlement import SettlementEngine

        agent, adapter = build_agent(tmp_path)
        try:
            # A position and nothing else: no forecast was ever recorded.
            trade_id = agent.storage.log_trade({
                "market_id": MARKET_ID, "side": "YES", "market_price": 0.55,
                "position_size_usd": 1.375, "status": "open",
                "venue_id": VENUE, "market_question": "Will the stub event happen?",
            })
            assert agent.storage.get_open_positions(), "the fixture position is missing"
            assert agent.calibration_engine.pending_forecasts() == [], (
                "this test is meaningless with a pending forecast present"
            )

            adapter.settled_outcome = 1.0
            engine = SettlementEngine(
                venue_registry=agent.venue_registry,
                storage=agent.storage,
                calibration_engine=agent.calibration_engine,
                trade_outcome_tracker=agent.trade_outcome_tracker,
            )
            report = asyncio.run(engine.settle_pending())

            assert report.settled == 1, (
                f"a position with no forecast was not settled: "
                f"{report.to_dict()}"
            )
            assert report.checked_trades == 1
            assert report.realised_pnl_usd == pytest.approx(1.125, abs=0.01)
            assert agent.storage.get_open_positions() == []
            row = agent.storage.conn.execute(
                "SELECT resolved, pnl FROM trades WHERE id=?", (trade_id,)).fetchone()
            assert row["resolved"] == 1
            assert row["pnl"] == pytest.approx(1.125, abs=0.01)
        finally:
            agent.storage.close()

    def test_an_empty_calibration_store_does_not_block_settlement(self, tmp_path):
        """
        The early return, stated as a rule: an empty calibration store is not a
        reason to leave positions open.
        """
        import inspect as _inspect
        from src.ptai.execution import settlement as settlement_module

        source = _inspect.getsource(settlement_module.SettlementEngine.settle_pending)
        # The docstring quotes the removed line, so match the CODE shape (the
        # keyword followed by a newline) rather than the bare phrase.
        assert "if not pending:\n" not in source, (
            "the early return is back: no pending forecasts means no trade is "
            "ever settled"
        )
        assert "market_order" in source, (
            "settlement must iterate the union of forecast markets and open "
            "trade markets"
        )

    def test_positions_settle_without_a_calibration_engine_at_all(self, tmp_path):
        """
        The engine declares calibration_engine optional. With it absent, only
        the forecast half of the work is impossible - not the position half.
        """
        from src.ptai.execution.settlement import SettlementEngine

        agent, adapter = build_agent(tmp_path)
        try:
            agent.storage.log_trade({
                "market_id": MARKET_ID, "side": "YES", "market_price": 0.55,
                "position_size_usd": 1.375, "status": "open",
                "venue_id": VENUE, "market_question": "Will the stub event happen?",
            })
            adapter.settled_outcome = 0.0
            engine = SettlementEngine(
                venue_registry=agent.venue_registry,
                storage=agent.storage,
                calibration_engine=None,
            )
            report = asyncio.run(engine.settle_pending())
            assert report.settled == 1, (
                f"a position went unsettled because the calibration engine was "
                f"absent: {report.to_dict()}"
            )
            assert report.realised_pnl_usd == pytest.approx(-1.375), (
                "a total loss on a $1.375 stake must realise -1.375"
            )
        finally:
            agent.storage.close()


# ==========================================================================
# Paper mode: the whole cycle, simulated, against a real book
# ==========================================================================

def _simulating_place_order(adapter, book=None):
    """
    A dry-run venue that simulates its fill, the way the real adapter now does.

    A stub that refuses without pricing anything is still a valid venue - it just
    produces no paper evidence, which is the correct outcome.
    """
    from src.ptai.execution.paper_broker import PaperBroker
    from src.ptai.markets.mechanics import MarketMechanics

    book = book if book is not None else {
        "bids": [{"price": "0.54", "size": "5000"}],
        "asks": [{"price": "0.56", "size": "5000"}]}

    async def place_order(opportunity, max_spend_usd, max_price):
        adapter.orders_placed += 1
        mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0,
                                    source="clob_market_info", is_real=True)
        fill = PaperBroker().simulate(book, "BUY", float(max_spend_usd),
                                      limit_price=float(max_price),
                                      mechanics=mechanics, book_source="orderbook")
        return {
            "status": "paper", "is_real": False, "simulated": True,
            "venue_id": VENUE, "market_id": opportunity.market.id,
            "simulated_filled_usd": fill.filled_usd,
            "filled_usd": fill.filled_usd, "filled_price": fill.avg_price,
            "price": fill.avg_price or float(max_price),
            "size": fill.filled_shares, "fees_usd": fill.fee_usd,
            "paper_fill": fill.to_dict(), "reason": fill.reason,
        }
    return place_order


class TestPaperModeSimulatesTheWholeCycle:
    """
    Paper mode has to be the real system with the order replaced by a
    simulation. Not a stub that reports success, and not a refusal that reports
    nothing.

    The venue here is in dry-run, so nothing can be sent. What is asserted is
    that a trade still comes out the other end - priced by walking the book, at
    the size the book would actually have filled, recorded as a paper position,
    consuming paper capital and not live capital.
    """

    def _paper_venue(self, tmp_path, book):
        """
        A dry-run venue that simulates its fill the way the real adapter now
        does: walk the real ladder, report the size it found, charge the fee.
        """
        from src.ptai.execution.paper_broker import PaperBroker
        from src.ptai.markets.mechanics import MarketMechanics

        adapter = StubVenue(dry_run=True)

        async def get_orderbook(market):
            adapter.last_book_requested = market.id
            return dict(book)

        async def place_order(opportunity, max_spend_usd, max_price):
            adapter.orders_placed += 1
            mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0,
                                        source="clob_market_info", is_real=True)
            fill = PaperBroker(taker_fee_rate=0.0).simulate(
                book, "BUY", float(max_spend_usd),
                limit_price=float(max_price), mechanics=mechanics,
                book_source="orderbook")
            return {
                "status": "paper", "is_real": False, "simulated": True,
                "venue_id": VENUE, "market_id": opportunity.market.id,
                "simulated_filled_usd": fill.filled_usd,
                "filled_usd": fill.filled_usd,
                "filled_price": fill.avg_price,
                "price": fill.avg_price or float(max_price),
                "size": fill.filled_shares,
                "fees_usd": fill.fee_usd,
                "paper_fill": fill.to_dict(),
                "reason": fill.reason,
            }

        adapter.get_orderbook = get_orderbook
        adapter.place_order = place_order
        return adapter

    def _build(self, tmp_path, book):
        adapter = self._paper_venue(tmp_path, book)
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.storage = Storage(db_path=str(tmp_path / "paper.db"))
        agent.storage.set_bankroll(50.0)
        _repoint_storage(agent, agent.storage)
        agent.venue_registry = stub_registry(adapter)
        agent.multi_venue_executor.registry = agent.venue_registry
        agent.account_health_engine.venue_registry = agent.venue_registry
        agent.capability_engine.venue_registry = agent.venue_registry
        agent.settlement_engine.venue_registry = agent.venue_registry
        agent._qualified_venue_ids = [VENUE]
        return agent, adapter

    def test_a_paper_trade_flows_through_the_whole_cycle(self, tmp_path, monkeypatch):
        """
        Depth-limited book: $3.00 requested, only $2.26 reachable, so the
        simulated fill is a fraction of the request. If paper mode books the
        REQUEST as the position, the ledger is fiction - so that is exactly what
        this asserts against.
        """
        # $1.12 at 0.56 and $1.14 at 0.57: $2.26 reachable inside a $3 guard
        # cap, so the simulated fill is genuinely short AND the average price
        # comes from walking two levels.
        book = {"asks": [{"price": "0.56", "size": "2"},
                         {"price": "0.57", "size": "2"}]}
        agent, adapter = self._build(tmp_path, book)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)

        result = _cycle(agent)
        assert adapter.orders_placed >= 1, "the paper venue was never asked"
        assert result["execution"], "the cycle recorded no execution at all"

        entry = result["execution"][0]
        fill = entry.get("fill") or {}
        assert entry.get("position_recorded") is True, (
            f"a simulated fill did not become a paper position: "
            f"{entry.get('position_reason')}"
        )
        assert fill.get("simulated") is True, "the result was recorded as live"

        requested = fill["requested_usd"]
        filled = fill["filled_usd"]
        assert 0 < filled < requested, (
            f"paper mode filled ${filled:.4f} of a ${requested:.4f} request "
            f"against a book holding only $14.05 - depth was ignored"
        )

        positions = agent.storage.get_open_positions()
        assert len(positions) == 1
        position = positions[0]
        assert position["status"] == "paper", (
            "a simulated fill must not be recorded as a live position holding "
            "real capital"
        )
        assert position["position_size_usd"] == pytest.approx(filled, abs=0.01), (
            f"the ledger was charged ${position['position_size_usd']:.4f} for a "
            f"${filled:.4f} simulated fill"
        )
        # The average price came from walking the ladder, so it is above the
        # touch the strategy was looking at.
        assert position["market_price"] > 0.56

    def test_a_paper_fill_does_not_consume_live_capital(self, tmp_path, monkeypatch):
        """
        Two pools. The paper engine must be able to trade all day without
        touching the budget reserve for real money - that is what makes it safe
        to leave running.
        """
        from src.ptai.execution.capital import CapitalLedger

        book = {"asks": [{"price": "0.56", "size": "5000"}]}
        agent, adapter = self._build(tmp_path, book)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        _cycle(agent)

        plan = CapitalLedger(storage=agent.storage).build(
            mode="live", budgets={"polymarket": 50.0},
            balances={"polymarket": {"available": True, "balance": 50.0}})
        account = plan.accounts[0]
        assert account.in_positions_usd == 0.0, (
            "a paper position was charged to live capital"
        )
        assert account.available_usd == pytest.approx(50.0)

    def test_paper_mode_with_no_book_records_nothing(self, tmp_path, monkeypatch):
        """
        The other half of the honesty rule. With no book there is no fill to
        simulate, so the cycle must record no position - not a position at the
        requested price, which is what the old fallback did.
        """
        agent, adapter = self._build(tmp_path, {})

        async def no_book(market):
            return {}
        adapter.get_orderbook = no_book
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)

        result = _cycle(agent)
        positions = agent.storage.get_open_positions()
        assert positions == [], (
            "a paper position was opened with no orderbook to price it against"
        )
        if result["execution"]:
            assert result["execution"][0].get("position_recorded") is not True
