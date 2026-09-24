"""
The connection defects that made the pipeline look smarter than it ran.

Five things were built and never reached:

1. `StrategyEngine` calls `context_provider.get_context(market)`. The agent only
   had `get_context_for_market()`. Every call raised AttributeError into a bare
   `except:`, which substituted a fabricated orderbook. News, X sentiment, web
   research and the real book existed and never reached a single forecast.

2. `EdgeCalculator.calculate` computed `fair_prob - market_price` unconditionally.
   A market the agent thought was OVERPRICED produced a negative edge and was
   discarded, even though the same belief is a positive edge on the other side.
   Only "the market is too low" could ever be traded.

3. The venue's authenticated balance endpoint existed and the adapter never
   called it, so the readiness ladder could not pass CONFIGURED.

4. Free cash ignored capital committed to working orders, so a resting GTC order
   let the next cycle spend money already promised to it.

5. Qualification "considered" fees, slippage, execution quality and a paper/live
   split. None of the four was recorded, and the placeholder execution quality
   happened to equal the threshold, so every venue passed that check by
   coincidence.
"""

from __future__ import annotations

import asyncio

import pytest

from src.ptai.execution.account_health import (
    AccountHealthEngine,
    TradeReadiness,
    is_venue_side_provenance,
    read_balance,
    record_verification,
    verification_status,
)
from src.ptai.execution.position_ledger import PositionLedgerBuilder
from src.ptai.learning.trade_outcomes import (
    TradeOutcomeTracker,
    qualification_stats_from_outcomes,
)
from src.ptai.markets.base import Market, MarketSource
from src.ptai.storage.db import Storage
from src.ptai.strategy.edge import EdgeCalculator
from src.ptai.venues.adapter import EligibilityStatus


def _market(price: float = 0.70, liquidity: float = 50000.0) -> Market:
    return Market(id="CTX-1", question="Will X happen?",
                  source=MarketSource.POLYMARKET, liquidity=liquidity,
                  outcome_prices=[price, 1 - price])


# ----------------------------------------------------------------------
# 1. the context connection
# ----------------------------------------------------------------------

def test_the_agent_has_the_method_the_strategy_engine_calls():
    """
    The exact mismatch. Asserted as a callable, because the failure it caused was
    that the call raised and nobody heard it.
    """
    from src.ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    assert callable(getattr(agent, "get_context", None)), (
        "StrategyEngine calls context_provider.get_context(market); without it "
        "every market falls back to a fabricated orderbook"
    )
    assert callable(getattr(agent, "get_context_for_market", None)), (
        "the original name must keep working for any existing caller"
    )


def test_get_context_returns_the_real_pipeline_output():
    """
    And it returns the rich context, not the placeholder. This calls the agent's
    method directly and checks it is the real implementation rather than a stub
    that satisfies the name.
    """
    from src.ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    context = asyncio.run(agent.get_context(_market()))

    assert isinstance(context, dict)
    # The real method always sets these keys; the fallback sets "degraded".
    assert "orderbook" in context
    assert "sources" in context
    assert context.get("degraded") is not True, (
        "get_context returned the degraded placeholder - the connection is "
        "still not reaching the real intelligence pipeline"
    )


def test_the_fallback_context_is_marked_degraded_not_silent():
    """
    The bare `except:` is gone. When the provider fails, the substitution is
    logged and labelled, because a silent fake is how this went unnoticed.
    """
    from pathlib import Path

    source = Path("src/ptai/strategy/strategy_engine.py").read_text()
    assert "except Exception as e:" in source
    assert '"degraded": True' in source, (
        "the fallback context must be marked, or nothing downstream can tell it "
        "from a researched forecast"
    )
    # The specific bug: a bare except swallowing an AttributeError.
    assert "except:\n                    context = {" not in source


# ----------------------------------------------------------------------
# 2. symmetric YES/NO edge
# ----------------------------------------------------------------------

class TestSymmetricEdge:
    def test_a_market_that_is_too_high_is_a_positive_edge_on_no(self):
        """
        fair 0.55 against a 0.70 market.

        This is not a loss - it is a NO trade. It used to be discarded.
        """
        calc = EdgeCalculator()
        yes = calc.calculate(market=_market(0.70), fair_prob=0.55, side="YES")
        no = calc.calculate(market=_market(0.70), fair_prob=0.55, side="NO")

        assert yes.raw_edge < 0, "buying YES here is indeed a loss"
        assert no.raw_edge > 0, (
            "buying NO at 0.30 with fair value 0.45 is the trade, and it must "
            "have a positive edge"
        )
        assert no.raw_edge == pytest.approx(1 - 0.55 - (1 - 0.70))
        assert no.raw_edge == pytest.approx(-yes.raw_edge)

    def test_the_edge_is_measured_on_the_side_being_bought(self):
        """The price and the fair value must both be on the traded side."""
        calc = EdgeCalculator()
        no = calc.calculate(market=_market(0.70), fair_prob=0.55, side="NO")
        assert no.market_price == pytest.approx(0.30), (
            "the NO side is bought at 1 - market_yes"
        )
        assert "[NO]" in no.reasoning, "the log line must name the side"

    def test_a_market_that_is_too_low_is_a_positive_edge_on_yes(self):
        """The mirror, which is what worked before - and still does."""
        calc = EdgeCalculator()
        yes = calc.calculate(market=_market(0.70), fair_prob=0.85, side="YES")
        no = calc.calculate(market=_market(0.70), fair_prob=0.85, side="NO")
        assert yes.raw_edge > 0
        assert no.raw_edge < 0

    def test_the_two_sides_of_a_genuine_mispricing_are_symmetric(self):
        """
        Raw edge must mirror exactly. It need not be identical after deductions -
        Polymarket's 0.06*p*(1-p) fee is a bigger percentage of a 0.30 notional
        than of a 0.70 one - but the raw asymmetry must be exact.
        """
        calc = EdgeCalculator()
        a = calc.calculate(market=_market(0.70), fair_prob=0.55, side="NO")
        b = calc.calculate(market=_market(0.70), fair_prob=0.85, side="YES")
        assert a.raw_edge == pytest.approx(b.raw_edge)

    def test_the_default_side_is_yes_so_existing_callers_are_unchanged(self):
        calc = EdgeCalculator()
        explicit = calc.calculate(market=_market(0.70), fair_prob=0.85, side="YES")
        default = calc.calculate(market=_market(0.70), fair_prob=0.85)
        assert default.raw_edge == pytest.approx(explicit.raw_edge)


# ----------------------------------------------------------------------
# 3. the venue balance reaches the readiness ladder
# ----------------------------------------------------------------------

class _FakeExecutor:
    def __init__(self, balance=42.5, source="clob_balance_allowance", real=True):
        self._balance, self._source, self._real = balance, source, real

    def get_balance_allowance(self, token_id=None, asset_type=None):
        return {"available": self._real, "is_real": self._real,
                "source": self._source, "balance": self._balance,
                "allowance": "999999999"}


def _adapter_with_executor(executor):
    from src.ptai.venues.polymarket_adapter import PolymarketAdapter

    adapter = PolymarketAdapter(private_key="0xKEY", funder="0xFUNDER")
    adapter._get_executor = lambda: executor
    return adapter


def test_the_balance_allowance_source_is_recognised_as_venue_side():
    """
    Two halves that both existed and never met: the executor emits
    "clob_balance_allowance" and the allowlist only knew "clob_real", so a
    genuinely venue-confirmed balance was refused as unverified.
    """
    assert is_venue_side_provenance("clob_balance_allowance"), (
        "this is the provenance PolymarketExecutor.get_balance_allowance() "
        "actually emits after an authenticated CLOB read"
    )


def test_the_adapter_uses_the_venue_balance_not_the_local_one():
    adapter = _adapter_with_executor(_FakeExecutor(balance=42.5))
    portfolio = asyncio.run(adapter.get_portfolio())

    assert portfolio["balance"] == pytest.approx(42.5), (
        "the venue said 42.50; the local bankroll is an estimate of it"
    )
    assert portfolio["venue_confirmed_balance"] == pytest.approx(42.5)
    balance, real, provenance = read_balance(portfolio)
    assert real is True
    assert is_venue_side_provenance(provenance), (
        f"provenance {provenance!r} does not prove Polymarket answered"
    )


def test_a_silent_venue_is_still_not_proof():
    """
    The other half, which must not regress: when the venue does not answer, the
    local balance must not be dressed up as a verified one.
    """
    adapter = _adapter_with_executor(
        _FakeExecutor(balance=None, real=False, source="no_clob_client"))
    portfolio = asyncio.run(adapter.get_portfolio())
    balance, real, provenance = read_balance(portfolio)
    assert balance is None, "an unverified balance must read as unknown"
    assert real is False
    assert not is_venue_side_provenance(provenance)


def test_the_ladder_can_reach_trade_permitted_with_venue_evidence():
    """The blocker: this was unreachable for a correctly configured account."""
    adapter = _adapter_with_executor(_FakeExecutor(balance=42.5))

    async def probe(token_id=None):
        return {"verified": True, "reason": "round trip ok"}

    adapter.probe_order_permission = probe

    class _Registry:
        adapters = {"polymarket": adapter}

    engine = AccountHealthEngine()
    engine.venue_registry = _Registry()
    result = asyncio.run(engine.check_venue_health("polymarket"))
    assert result.readiness in (TradeReadiness.TRADE_PERMITTED,
                                TradeReadiness.FUNDED), result.reason
    assert result.evidence["balance_provenance"] == "clob_balance_allowance"


# ----------------------------------------------------------------------
# 4. required verification cannot silently become live-ready
# ----------------------------------------------------------------------

class TestVerificationIsEvidenced:
    def _engine(self, storage):
        adapter = _adapter_with_executor(_FakeExecutor(balance=42.5))

        async def probe(token_id=None):
            return {"verified": True, "reason": "round trip ok"}

        adapter.probe_order_permission = probe

        class _Registry:
            adapters = {"polymarket": adapter}

        engine = AccountHealthEngine(storage=storage)
        engine.venue_registry = _Registry()
        return engine

    def test_requires_verification_with_no_record_stops_below_live(self, tmp_path):
        """
        Uganda is REQUIRES_VERIFICATION for Polymarket. Authenticated, funded and
        an order probe is NOT evidence the check was completed - the old code only
        blocked RESTRICTED, so "requires verification" slid into TRADE_PERMITTED
        with nothing but the word "requires" distinguishing it from READY.
        """
        storage = Storage(db_path=str(tmp_path / "v.db"))
        result = asyncio.run(self._engine(storage).check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.FUNDED
        assert result.ready_to_trade is False
        assert "verification_unproven" in result.blockers
        assert "requires_verification" in result.reason or "verification" in result.reason

    def test_recording_the_verification_lets_it_through(self, tmp_path):
        """The operator completes KYC at the venue; the agent cannot see it."""
        storage = Storage(db_path=str(tmp_path / "v2.db"))
        assert verification_status(storage, "polymarket") == "not recorded"
        record_verification(storage, "polymarket", "KYC completed 2026-09")

        result = asyncio.run(self._engine(storage).check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.TRADE_PERMITTED
        assert result.ready_to_trade is True
        assert "verification_unproven" not in result.blockers
        # And the claim travels with the evidence, so it can be audited.
        assert any("Verification recorded" in c for c in result.checks)

    def test_an_adapter_that_reports_verification_itself_is_also_evidence(self):
        """If a venue ever exposes the state, that counts too - and is preferred."""
        adapter = _adapter_with_executor(_FakeExecutor(balance=42.5))
        adapter.verification_evidence = lambda: "venue reports KYC verified"

        async def probe(token_id=None):
            return {"verified": True, "reason": "ok"}

        adapter.probe_order_permission = probe

        class _Registry:
            adapters = {"polymarket": adapter}

        engine = AccountHealthEngine()
        engine.venue_registry = _Registry()
        result = asyncio.run(engine.check_venue_health("polymarket"))
        assert result.readiness == TradeReadiness.TRADE_PERMITTED


# ----------------------------------------------------------------------
# 5. free cash includes capital promised to working orders
# ----------------------------------------------------------------------

class TestWorkingOrdersReserveCash:
    def _storage_with(self, tmp_path, position_usd=3.0, resting_usd=3.0):
        storage = Storage(db_path=str(tmp_path / "cap.db"))
        storage.set_bankroll(50.0)
        if position_usd:
            storage.log_trade({
                "market_id": "M1", "venue_id": "polymarket", "side": "YES",
                "position_size_usd": position_usd, "market_price": 0.5,
                "fair_value": 0.6, "edge": 0.1, "confidence": 0.7,
                "strategy": "value"})
        if resting_usd:
            storage.upsert_order({
                "order_id": "ORD-R", "market_id": "M2", "venue_id": "polymarket",
                "side": "YES", "status": "open", "requested_usd": resting_usd,
                "matched_usd": 0.0, "limit_price": 0.5})
        return storage

    def test_a_resting_order_reduces_free_cash(self, tmp_path):
        """
        $50, a $3 position and a $3 order working in the book: $44 is free, not
        $47. The old ledger spent the difference next cycle.
        """
        ledger = PositionLedgerBuilder(
            storage=self._storage_with(tmp_path)).build()
        assert ledger.resting_order_cost == pytest.approx(3.0)
        assert ledger.reserved_capital == pytest.approx(3.0)
        assert ledger.free_cash == pytest.approx(44.0), (
            f"free cash {ledger.free_cash} ignores the working order"
        )

    def test_no_working_orders_means_the_old_answer_still_holds(self, tmp_path):
        ledger = PositionLedgerBuilder(storage=self._storage_with(
            tmp_path, resting_usd=0.0)).build()
        assert ledger.resting_order_cost == 0.0
        assert ledger.free_cash == pytest.approx(47.0)

    def test_an_unreadable_reservation_fails_closed(self, tmp_path):
        """
        Unknown committed capital and zero committed capital are opposite facts.
        Treating the first as the second is the overcommit this guards.
        """
        storage = self._storage_with(tmp_path)

        def boom(*args, **kwargs):
            raise RuntimeError("orders table unreadable")

        storage.resting_capital_usd = boom
        ledger = PositionLedgerBuilder(storage=storage).build()
        assert ledger.reservations_unknown is True
        assert ledger.can_open_new is False, (
            "with the working orders unreadable, nothing may be sized"
        )
        assert any("could not be read" in w for w in ledger.warnings)


# ----------------------------------------------------------------------
# 6. qualification metrics are measured, not assumed
# ----------------------------------------------------------------------

def _record_costed_outcomes(storage, tracker, n=150, with_costs=True):
    for i in range(n):
        won = (i % 10) < 8
        tid = storage.log_trade({
            "market_id": f"C{i}", "venue_id": "polymarket", "side": "YES",
            "position_size_usd": 3.0, "market_price": 0.5, "fair_value": 0.75,
            "edge": 0.15, "confidence": 0.7, "strategy": "value"})
        pnl = 1.2 if won else -1.8
        storage.resolve_trade(tid, outcome=1.0 if won else 0.0, pnl=pnl)
        kwargs = {"fees_usd": 0.06, "slippage_bps": 10.0,
                  "execution_quality": 0.98, "data_mode": "live_paper"} \
            if with_costs else {}
        tracker.record_trade(
            trade_id=str(tid), market_id=f"C{i}", venue_id="polymarket",
            strategy="value", category="politics", forecast_prob=0.75,
            market_price=0.5, edge=0.15, side="YES", amount_usd=3.0, **kwargs)
        tracker.record_resolution(str(tid), actual_outcome=1.0 if won else 0.0,
                                  pnl=pnl)


class TestQualificationMetricsAreMeasured:
    def test_execution_quality_is_not_the_threshold_by_default(self, tmp_path):
        """
        0.5 was both the default and the requirement, so every venue passed an
        execution-quality check that had never been performed.
        """
        storage = Storage(db_path=str(tmp_path / "q1.db"))
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["execution_quality_avg"] == 0.0, (
            "unmeasured execution quality must fail the 0.5 requirement, not "
            "equal it"
        )
        assert stats["costs_measured"] == 0

    def test_costs_are_aggregated_when_recorded(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "q2.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        _record_costed_outcomes(storage, tracker, n=150)
        stats = qualification_stats_from_outcomes(storage, "polymarket")

        assert stats["costs_measured"] == 150
        assert stats["fees_total"] == pytest.approx(150 * 0.06)
        assert stats["slippage_total"] > 0, "slippage must be aggregated"
        assert stats["execution_quality_avg"] > 0.9

    def test_paper_and_live_profit_are_separated(self, tmp_path):
        """
        With no mode recorded they were one pool, so a paper result could carry a
        venue toward live capital.
        """
        storage = Storage(db_path=str(tmp_path / "q3.db"))
        tracker = TradeOutcomeTracker(storage=storage)
        _record_costed_outcomes(storage, tracker, n=20)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        assert stats["live_trades"] == 0
        assert stats["paper_trades"] == 20
        assert stats["profit_live"] == 0.0, (
            "nothing here was executed live, so live profit is zero"
        )
        assert stats["profit_paper"] != 0.0

    def test_drawdown_starts_from_the_initial_bankroll_not_the_current_one(
            self, tmp_path):
        """
        Historical drawdown was replayed from today's balance, so it described a
        past that never happened and changed whenever the balance did.
        """
        storage = Storage(db_path=str(tmp_path / "q4.db"))
        storage.set_bankroll(1000.0)   # current, wildly inflated
        tracker = TradeOutcomeTracker(storage=storage)
        _record_costed_outcomes(storage, tracker, n=30)
        stats = qualification_stats_from_outcomes(storage, "polymarket")
        # With a $1000 starting point a $1.80 loss is 0.18% - the drawdown would
        # look trivial. Anchored at the recorded initial bankroll it does not.
        assert stats["drawdown_max"] > 0.001
