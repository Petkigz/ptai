"""
What a cost IS, not just what it is worth.

The EV engine's output decides which trades happen, so a number in it that was
never measured is a decision made on an invention. Three specific inventions
were in there:

  * `opportunity.fees_pct or 0.02` - a falsy test, so a venue that declares a
    0.0% fee (Manifold, Kalshi) was charged the 2% default, and a venue whose
    real schedule is 0.1% was charged 2%. The default was doing the deciding.
  * `gas_usd = 0.05` for Polymarket and AFX DEX - a literal with no source, on a
    $3 stake that is 1.7% of the position, charged to venues whose order flow is
    relayed rather than sent on-chain.
  * the `_spread_is_real` provenance from `read_spread` was assigned and never
    used, so a default spread and a measured one were indistinguishable.

The tests below pin the contract: a cost is MEASURED, DECLARED or ASSUMED, the
assumptions are named, and a trade that only clears because of an assumption
does not clear.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.strategy.expected_ev import ExpectedNetEVEngine
from src.ptai.venues.adapter import VenueOpportunity, VenueType


def _market(price: float = 0.60, liquidity: float = 50_000.0) -> Market:
    return Market(
        id="M-EV", source=MarketSource.POLYMARKET,
        question="Will the cost accounting be honest?",
        outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
        tokens=[Token(token_id="M-EV-Y", outcome="YES", price=price)],
        volume=200_000.0, volume_24h=100_000.0, liquidity=liquidity,
        active=True, closed=False, slug="m-ev",
        raw={"venue": "polymarket"},
    )


def _opp(*, fees_pct: Optional[float], slippage_pct: Optional[float] = None,
         spread_pct: Optional[float] = None, side: str = "YES",
         fair: float = 0.70, price: float = 0.60,
         liquidity: float = 50_000.0, execution_quality: float = 0.9,
         order_gas_usd: Optional[float] = 0.0):
    """`order_gas_usd=0.0` is what the Polymarket adapter declares."""
    return VenueOpportunity(
        market=_market(price, liquidity),
        venue_id="polymarket", venue_type=VenueType.PREDICTION, side=side,
        market_price=price, estimated_fair=fair, raw_edge=0.10,
        effective_edge=0.08, confidence=0.8, uncertainty=0.1,
        liquidity_score=0.9, execution_quality=execution_quality,
        should_trade=True,
        fees_pct=fees_pct, slippage_pct=slippage_pct, spread_pct=spread_pct,
        order_gas_usd=order_gas_usd,
    )


def _real_book(**overrides):
    book = {"bids": [{"price": "0.59", "size": "5000"}],
            "asks": [{"price": "0.61", "size": "5000"}],
            "spread": 0.02, "is_real": True, "source": "clob_real"}
    book.update(overrides)
    return book


class TestTheVenuesOwnFeeScheduleIsUsed:
    def test_a_fee_free_venue_is_not_charged_the_default_two_percent(self):
        """
        The bug, exactly: `fees_pct or 0.02` on a venue that declares 0.0.
        """
        engine = ExpectedNetEVEngine()
        free = engine.calculate(_opp(fees_pct=0.0), 3.0, _real_book())
        unknown = engine.calculate(_opp(fees_pct=None), 3.0, _real_book())

        assert free.fees_usd == pytest.approx(0.0), (
            "a venue declaring no fees was charged them anyway")
        assert free.cost_provenance["fees"].startswith("declared")
        # The unknown case is still charged the conservative default, and says so.
        assert unknown.fees_usd == pytest.approx(0.06)
        assert "fees" in unknown.assumed_costs

        # And the difference is real money on the same trade.
        assert free.net_ev_usd > unknown.net_ev_usd

    def test_a_cheap_venue_is_charged_its_own_rate_not_the_default(self):
        """WhiteBIT declares 0.1%; the default charged it 2%."""
        engine = ExpectedNetEVEngine()
        result = engine.calculate(_opp(fees_pct=0.001), 3.0, _real_book())
        assert result.fees_usd == pytest.approx(0.003)
        assert "0.10%" in result.cost_provenance["fees"], (
            "the provenance must name the rate that was actually charged")


class TestGasIsNoLongerAFabricatedLiteral:
    def test_polymarket_is_not_charged_the_old_five_cents(self):
        """
        $0.05 on a $3 order is 1.7% of the stake.

        Polymarket relays its orders, so the operator pays no gas to trade -
        gas is spent funding the account and redeeming a settled position. The
        old constant charged every Polymarket trade as though each one went
        on-chain, which on a 3% net-EV threshold decides the trade.
        """
        engine = ExpectedNetEVEngine()
        result = engine.calculate(_opp(fees_pct=0.0), 3.0, _real_book())
        assert result.gas_usd == pytest.approx(0.0), (
            f"gas is still being charged to a relayed venue: ${result.gas_usd:.4f}")
        assert result.cost_provenance["gas"].startswith("not applicable")
        assert "0.05" not in result.cost_provenance["gas"]

    def test_an_undeclared_venue_falls_back_to_the_gas_model_and_says_so(self):
        """
        A venue that does not declare its mechanism is ESTIMATED, and the
        estimate is labelled - it is not quietly assumed to be free, and it is
        not a literal in the EV engine either.
        """
        engine = ExpectedNetEVEngine()
        opp = _opp(fees_pct=0.0, order_gas_usd=None)   # nothing declared
        result = engine.calculate(opp, 3.0, _real_book())
        assert result.gas_usd > 0, (
            "an undeclared venue with on-chain mechanics was costed at zero gas")
        assert result.cost_provenance["gas"].startswith("assumed")
        assert "gas" in result.assumed_costs
        assert "GasModel" in result.cost_provenance["gas"]

    def test_a_venue_with_no_gas_is_reported_as_not_applicable(self):
        """Off-chain venues do not have a gas cost, and saying so is not zero-by-accident."""
        engine = ExpectedNetEVEngine()
        opp = _opp(fees_pct=0.0)
        opp.venue_id = "manifold"
        result = engine.calculate(opp, 3.0, _real_book())
        assert result.gas_usd == 0.0
        assert result.cost_provenance["gas"].startswith("not applicable")


class TestAssumptionsAreNamedAndCarryWeight:
    def test_every_cost_says_where_it_came_from(self):
        engine = ExpectedNetEVEngine()
        result = engine.calculate(_opp(fees_pct=0.02), 3.0, _real_book())
        for cost in ("fees", "spread", "slippage", "gas", "execution_loss"):
            assert cost in result.cost_provenance, f"{cost} has no provenance"
            assert result.cost_provenance[cost].strip()
        # With a real book and a declared fee, nothing is assumed about the
        # spread or the execution - those are read.
        assert result.cost_provenance["spread"] == "measured from the orderbook"
        assert result.cost_provenance["execution_loss"].startswith("measured")
        assert "spread" not in result.assumed_costs

    def test_a_trade_whose_edge_RUNS_THROUGH_a_guess_does_not_trade(self):
        """
        The rule that gives an assumption teeth.

        No fee schedule, no real book: every cost in the decision is ours. A
        trade that clears on those numbers is not evidence of an edge, so it is
        refused - and the refusal names the assumptions.
        """
        engine = ExpectedNetEVEngine()
        # A 10c mispricing, no fee schedule and no book: the costs are ours.
        #
        # The edge is sized deliberately: it must clear BOTH hard thresholds on
        # the assumption-based numbers (net EV > 0 and >= 3% of stake), so that
        # the ONLY thing refusing it is the cost-stress clause. A thinner edge
        # would be refused by the 3% floor and this test would pass with the
        # clause deleted - which is exactly what the first version of it did.
        opp = _opp(fees_pct=None, fair=0.68, price=0.58, order_gas_usd=None)
        assumed = engine.calculate(opp, 3.0, {})
        assert assumed.assumed_costs, "the fixture must rely on assumptions"
        assert assumed.net_ev_usd > 0 and assumed.net_ev_pct >= 0.03, (
            "the fixture must clear the ordinary thresholds, or this test "
            "measures a different gate")
        assert assumed.should_trade is False, (
            "a trade that only clears because the costs were guessed was "
            "allowed to trade")
        assert "Assumed" in assumed.reasoning
        assert assumed.stressed_net_ev_usd <= 0

    def test_the_same_trade_with_measured_costs_is_allowed(self):
        """
        The other direction. This is what makes the rule a cost rule and not a
        blanket refusal: measure the costs and the same opportunity trades.
        """
        engine = ExpectedNetEVEngine()
        opp = _opp(fees_pct=0.02, fair=0.70, price=0.60)
        measured = engine.calculate(opp, 3.0, _real_book(slippage_estimate=0.004))
        assert measured.should_trade is True, (
            f"a well-measured opportunity was refused: {measured.reasoning}")
        assert measured.material_assumptions(3.0) == []
        assert measured.costs_are_measured is True

    def test_the_stress_only_doubles_what_was_assumed(self):
        """
        Measured costs are measurements. Doubling them would be inventing
        pessimism, so the stress figure must leave them alone.
        """
        engine = ExpectedNetEVEngine()
        measured_costs = engine.calculate(
            _opp(fees_pct=0.02), 3.0, _real_book(slippage_estimate=0.004))
        with_assumption = engine.calculate(_opp(fees_pct=None), 3.0, _real_book())
        # Declared fee, declared gas, measured spread and slippage: nothing assumed.
        assert measured_costs.assumed_cost_usd == 0.0
        assert measured_costs.stressed_net_ev_usd == pytest.approx(
            measured_costs.net_ev_usd), (
            "nothing was assumed, so the stressed figure must not move")
        # One assumption, and the stress figure moves by exactly that assumption.
        assert with_assumption.assumed_cost_usd > 0.0
        assert with_assumption.stressed_net_ev_usd == pytest.approx(
            with_assumption.net_ev_usd - with_assumption.assumed_cost_usd)


class TestTheDataclassContract:
    def test_zero_and_unknown_are_different_values(self):
        """
        `0.0` used to mean both "this venue is free" and "nobody looked", which
        is the same defect as a column holding two different numbers.
        """
        opp = _opp(fees_pct=None)
        assert opp.fees_pct is None, "unknown must not be represented as 0.0"
        assert _opp(fees_pct=0.0).fees_pct == 0.0
        # An unmeasured cost still SCORES as though it were expensive, because
        # the alternative is scoring it as free.
        assert opp._cost_or_default(opp.fees_pct, 0.02) == 0.02
        assert opp._cost_or_default(0.0, 0.02) == 0.0

    def test_the_result_dict_carries_the_provenance_forward(self):
        """The learning record and the console both read to_dict()."""
        engine = ExpectedNetEVEngine()
        payload = engine.calculate(_opp(fees_pct=None), 3.0, {}).to_dict()
        assert payload["cost_provenance"]["fees"].startswith("assumed")
        assert "fees" in payload["assumed_costs"]
        assert payload["assumed_cost_usd"] > 0
        assert payload["costs_are_measured"] is False
    def test_a_measured_decision_reports_itself_as_measured(self):
        engine = ExpectedNetEVEngine()
        payload = engine.calculate(
            _opp(fees_pct=0.02), 3.0, _real_book(slippage_estimate=0.004)).to_dict()
        assert payload["costs_are_measured"] is True
        assert payload["assumed_costs"] == []
        assert payload["assumed_cost_pct"] == 0.0


class TestTheVenueFeeReachesTheScan:
    def test_the_adapter_declares_what_the_engine_charges(self):
        """
        The wiring, not the engine: the venue's declared taker fee has to travel
        from the adapter through the context to the opportunity, or the default
        decides and the engine's honesty about defaults is doing all the work.
        """
        from src.ptai.markets.base import Market, MarketSource, Token  # noqa: F401
        import asyncio
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        try:
            market = _market()
            adapter = agent.venue_registry.adapters.get("polymarket")
            assert adapter is not None
            declared = float(adapter.capabilities.fee_taker_pct)
            context = asyncio.run(agent.get_context_for_market(market))
            assert context.get("fee_taker_pct") == pytest.approx(declared), (
                "the venue's own fee schedule did not reach the context, so the "
                "EV stage is back on its 2% default")
        finally:
            agent.storage.close()
            try:
                agent.memory.close()
            except Exception:
                pass
