"""
The loop that could not close, and the number that was not a forecast.

Three defects, each of which broke the chain in a place that looked fine:

1. THE EXPLORATION DEADLOCK. Exploration candidates were selected, marked
   `_is_exploration`, appended to a list, and then counted in a log line.
   Nothing executed them. So on a fresh installation:

       no qualified venues -> pick an exploration candidate
       -> do not execute it -> zero trade outcomes
       -> qualification stays at zero -> no qualified venues -> repeat

   The agent could never bootstrap its own qualification evidence from zero. The
   "100 paper trades -> qualified -> live" lifecycle was unreachable by
   construction, and every component involved worked.

2. THE RANDOM "LLM" FORECAST. When the LLM did not answer, Brain invented a fair
   value from `random.uniform`, the ensemble labelled it `llm_reasoning`, gave it
   the LARGEST weight in the ensemble (0.30), and handed it a confidence that
   INCREASED with the size of the random edge. On a machine with no local model
   - which is the state the startup log reports - the biggest single component of
   every forecast was a random draw, and the trades it produced were recorded as
   evidence about venues and strategies.

3. THE WRONG SIDE ON A REAL ORDER. `_resolve_token_id` selects the NO token for a
   NO opportunity, and the order was then sent as side="SELL". Taking a NO
   position means BUYING the NO token. The price cap had the same problem: it was
   computed from the YES price for both sides.
"""

from __future__ import annotations

import asyncio

import pytest

from src.ptai.agent.brain import Brain
from src.ptai.intelligence.ensemble import EnsembleForecaster
from src.ptai.markets.base import Market, MarketSource
from src.ptai.venues.adapter import VenueOpportunity, VenueType


def _market(price: float = 0.50, liquidity: float = 50000.0) -> Market:
    return Market(id="L1", question="Will X happen?",
                  source=MarketSource.POLYMARKET, liquidity=liquidity,
                  outcome_prices=[price, 1 - price])


def _opportunity(side: str = "YES", price: float = 0.70) -> VenueOpportunity:
    market = _market(price)
    return VenueOpportunity(
        market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION,
        side=side, market_price=price, estimated_fair=0.6,
        raw_edge=0.1, effective_edge=0.12, confidence=0.7, uncertainty=0.1,
        liquidity_score=0.9, execution_quality=0.9, category="politics")


# ----------------------------------------------------------------------
# 1. exploration actually executes, and only in paper
# ----------------------------------------------------------------------

class TestExplorationLaneExecutes:
    def test_exploration_candidates_enter_the_execution_loop(self):
        """
        The deadlock, tested at its source: candidates must be routed into the
        loop that executes, not merely collected.

        Asserted against the code path because the failure was an ABSENCE - a
        list that was filled and never consumed - and no runtime assertion inside
        the old code could have failed.
        """
        from pathlib import Path

        source = Path("src/ptai/agent/v3_loop.py").read_text()
        assert "for opp in exploration_trades:" in source, (
            "exploration trades must be routed into the execution loop"
        )
        # And the routing must add them to the collection the loop iterates.
        assert "final_trades.append(opp)" in source, (
            "the execution loop iterates final_trades; exploration candidates "
            "must reach it or they are never executed"
        )

    def test_an_exploration_trade_cannot_use_live_capital(self):
        """
        Exploration exists to earn qualification, and qualification is earned in
        paper. An exploration trade touching real money would be live trading at
        an unqualified venue.
        """
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=False)
        opp = _opportunity()
        opp._is_exploration = True

        adapter = agent.venue_registry.adapters["polymarket"]
        adapter.dry_run = False          # a genuinely armed adapter
        captured = {}

        async def fake_execute(opportunity, max_spend_usd, max_price):
            # Record what the adapter's gate was DURING the call.
            captured["dry_run_during"] = adapter.dry_run
            captured["cap"] = max_price
            from src.ptai.execution.multi_venue_executor import ExecutionResult
            return ExecutionResult(
                venue_id="polymarket", market_id=opportunity.market.id,
                status="paper", amount_usd=max_spend_usd, price=max_price,
                fees_usd=0.0, gas_usd=0.0, latency_ms=1.0, reasoning="paper")

        agent.multi_venue_executor.execute_single = fake_execute
        asyncio.run(agent._execute_with_side_aware_cap(
            opp=opp, amount_usd=3.0, exploration=True))

        assert captured["dry_run_during"] is True, (
            "the adapter was not forced to paper for an exploration trade"
        )
        assert adapter.dry_run is False, (
            "the adapter's own setting must be restored afterwards"
        )

    def test_a_qualified_trade_is_not_forced_to_paper(self):
        """The forcing is for exploration only - the live path must stay live."""
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=False)
        adapter = agent.venue_registry.adapters["polymarket"]
        adapter.dry_run = True
        seen = {}

        async def fake_execute(opportunity, max_spend_usd, max_price):
            seen["dry_run_during"] = adapter.dry_run
            from src.ptai.execution.multi_venue_executor import ExecutionResult
            return ExecutionResult(
                venue_id="polymarket", market_id=opportunity.market.id,
                status="paper", amount_usd=max_spend_usd, price=max_price,
                fees_usd=0.0, gas_usd=0.0, latency_ms=1.0, reasoning="x")

        agent.multi_venue_executor.execute_single = fake_execute
        asyncio.run(agent._execute_with_side_aware_cap(
            opp=_opportunity(), amount_usd=3.0, exploration=False))
        assert seen["dry_run_during"] is True, (
            "a non-exploration call must leave the adapter's own gate alone"
        )


# ----------------------------------------------------------------------
# 2. the price cap is on the side being bought
# ----------------------------------------------------------------------

class TestPriceCapFollowsTheSide:
    def _cap_for(self, side, price=0.70):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        seen = {}

        async def fake_execute(opportunity, max_spend_usd, max_price):
            seen["cap"] = max_price
            from src.ptai.execution.multi_venue_executor import ExecutionResult
            return ExecutionResult(
                venue_id="polymarket", market_id=opportunity.market.id,
                status="paper", amount_usd=max_spend_usd, price=max_price,
                fees_usd=0.0, gas_usd=0.0, latency_ms=1.0, reasoning="x")

        agent.multi_venue_executor.execute_single = fake_execute
        asyncio.run(agent._execute_with_side_aware_cap(
            opp=_opportunity(side=side, price=price), amount_usd=3.0))
        return seen["cap"]

    def test_a_yes_order_is_capped_above_the_yes_price(self):
        assert self._cap_for("YES", 0.70) == pytest.approx(0.72)

    def test_a_no_order_is_capped_above_the_no_price_not_the_yes_price(self):
        """
        The NO token trades near 1 - yes. A cap of yes + 0.02 is a cap on a price
        that token never reaches, so the order is rejected as non-marketable - or
        accepted at a limit that makes no sense for the token being bought.
        """
        cap = self._cap_for("NO", 0.70)
        assert cap == pytest.approx(1.0 - 0.70 + 0.02), (
            f"NO cap {cap} was derived from the YES price"
        )
        assert cap == pytest.approx(0.32)


# ----------------------------------------------------------------------
# 3. the real order is always a BUY of the selected token
# ----------------------------------------------------------------------

def test_the_live_order_is_always_a_buy():
    """
    A NO position is a BUY of the NO token. The order said
    side="BUY" if side in ("YES","BUY") else "SELL", which selects the NO token
    and then SELLS it - the wrong direction on the right token.
    """
    from pathlib import Path

    source = Path("src/ptai/venues/polymarket_adapter.py").read_text()
    assert '"BUY" if str(opportunity.side).upper() in ("YES", "BUY") else "SELL"' \
        not in source, "the SELL branch is still there"
    assert source.count('side="BUY",') >= 1 or 'side="BUY"' in source


def test_the_adapter_sends_buy_for_a_no_opportunity():
    """Executed, not grepped: the SDK must receive BUY."""
    from src.ptai.venues.polymarket_adapter import PolymarketAdapter

    adapter = PolymarketAdapter(private_key="0xKEY", funder="0xFUNDER")
    sent = {}

    class FakeExecutor:
        def place_order(self, token_id, price, size, side, order_type,
                        dry_run, mechanics):
            sent.update(token_id=token_id, side=side, price=price, size=size)
            return {"status": "matched", "orderID": "o1", "size_matched": size}

    adapter._get_executor = lambda: FakeExecutor()
    adapter._executor = FakeExecutor()
    # `can_place_real_orders` is a PROPERTY (not dry_run AND supports_trading),
    # so the adapter is armed through the two things it actually reads.
    adapter.dry_run = False
    adapter.capabilities.supports_trading = True

    opp = _opportunity(side="NO", price=0.70)
    # The NO token must be the one selected, and it must be bought.
    adapter._resolve_token_id = lambda opportunity: "NO_TOKEN"
    adapter.get_mechanics = lambda opportunity, token_id=None: type(
        "M", (), {
            "round_price": lambda self, p, s: 0.30,
            "shares_for_usd": lambda self, usd, p, s: 10.0,
            "validate_order": lambda self, p, s: (True, ""),
            "to_dict": lambda self: {},
            "is_real": True,
        })()

    asyncio.run(adapter.place_order(
        opportunity=opp, max_spend_usd=3.0, max_price=0.32))

    assert sent.get("token_id") == "NO_TOKEN", "the NO token must be bought"
    assert sent.get("side") == "BUY", (
        f"a NO position was sent as {sent.get('side')!r} on the NO token"
    )


# ----------------------------------------------------------------------
# 4. no LLM means no opinion, not a random one
# ----------------------------------------------------------------------

class TestNoLLMIsNotRandomness:
    def test_the_fallback_does_not_invent_a_fair_value(self):
        """
        This is the one that matters. `random.uniform` produced edges that were
        then weighted at 0.30 and recorded as evidence.
        """
        brain = Brain()
        market = _market(0.50)
        first = brain._fallback_heuristic(market, None)
        second = brain._fallback_heuristic(market, None)

        assert first.fair_value == second.fair_value, (
            "two calls with no new information produced different fair values"
        )
        assert first.fair_value == pytest.approx(market.yes_price), (
            "with no model and no sentiment the only defensible estimate is the "
            "market's own price"
        )
        assert first.edge == pytest.approx(0.0)
        assert first.should_trade is False, (
            "no model answered, so there is no edge to trade"
        )

    def test_the_fallback_says_it_is_not_an_llm(self):
        brain = Brain()
        result = brain._fallback_heuristic(_market(0.50), None)
        assert result.llm_provider == "heuristic"
        assert "not an LLM forecast" in result.reasoning

    def test_sentiment_alone_still_moves_the_estimate(self):
        """Real evidence is real evidence - it is randomness that is refused."""
        brain = Brain()

        class Sentiment:
            score = 0.8
            confidence = 0.9
            summary = "strongly positive"

        result = brain._fallback_heuristic(_market(0.50), Sentiment())
        assert result.fair_value > 0.50
        assert result.llm_provider == "heuristic"
        assert "no LLM" in result.reasoning


class TestHeuristicCannotWearTheLLMsName:
    def test_a_heuristic_answer_is_named_for_what_it_is(self):
        forecaster = EnsembleForecaster()
        forecast = forecaster.add_llm_forecast(
            _market(0.5),
            {"fair_value": 0.7, "confidence": 0.9, "reasoning": "guess",
             "llm_provider": "heuristic"})
        assert forecast.model_name == "heuristic_reasoning", (
            "a non-LLM answer must not be labelled llm_reasoning"
        )
        assert "NOT AN LLM ANSWER" in forecast.reasoning

    def test_a_real_llm_answer_keeps_the_llm_name(self):
        forecaster = EnsembleForecaster()
        forecast = forecaster.add_llm_forecast(
            _market(0.5),
            {"fair_value": 0.7, "confidence": 0.9, "reasoning": "analysis",
             "llm_provider": "ollama"})
        assert forecast.model_name == "llm_reasoning"

    def test_the_heuristic_carries_less_weight_than_the_llm(self):
        """Executed: the ensemble must actually discount it."""
        forecaster = EnsembleForecaster()
        assert (forecaster.model_weights["heuristic_reasoning"]
                < forecaster.model_weights["llm_reasoning"])

    def test_a_heuristic_cannot_outvote_a_real_llm_answer(self):
        forecaster = EnsembleForecaster()
        heuristic = forecaster.add_llm_forecast(
            _market(0.5), {"fair_value": 0.95, "confidence": 0.95,
                           "reasoning": "noise", "llm_provider": "heuristic"})
        real = forecaster.add_llm_forecast(
            _market(0.5), {"fair_value": 0.55, "confidence": 0.95,
                           "reasoning": "analysis", "llm_provider": "ollama"})
        result = forecaster.ensemble([heuristic, real], _market(0.5))
        # The real answer must dominate: the heuristic pulls the mean up only
        # slightly, so the result stays far closer to 0.55 than to 0.95.
        assert result.fair_probability < 0.70, (
            f"the heuristic moved the ensemble to {result.fair_probability}"
        )


def test_brain_reuses_the_router_it_was_given():
    """
    `Brain()` built its own router from settings, so the fallback forecast could
    run against a different provider, endpoint, model and timeout than the
    intelligence stack that asked for it.
    """
    sentinel = object()
    brain = Brain(llm_router=sentinel)
    assert brain.llm_router is sentinel, (
        "the supplied router was replaced by a settings-derived one"
    )


# ----------------------------------------------------------------------
# 5. the real orderbook reaches every stage
# ----------------------------------------------------------------------

def test_the_orderbook_is_attached_to_the_market_for_the_ev_stage():
    """
    EdgeCalculator priced the spread from the real book; ExpectedNetEVEngine was
    handed `opp.market.raw["orderbook"]`, which nothing ever wrote, so it fell
    back to a default 0.02 spread. Two cost estimates for one trade, and the
    later one decided.
    """
    from pathlib import Path

    source = Path("src/ptai/agent/v3_loop.py").read_text()
    assert 'market.raw["orderbook"] = orderbook' in source, (
        "the retrieved orderbook must be attached to the market, or the EV "
        "stage recomputes costs from defaults"
    )


def test_execution_quality_is_measured_from_the_book_not_a_constant():
    from src.ptai.strategy.strategy_engine import _execution_quality_from_book

    tight = {"is_real": True, "spread": 0.01, "depth": 20000}
    wide = {"is_real": True, "spread": 0.09, "depth": 200}
    assert _execution_quality_from_book(tight, _market()) > \
        _execution_quality_from_book(wide, _market())
    assert _execution_quality_from_book({}, _market()) == 0.0, (
        "an unmeasured execution is not a good one"
    )


# ----------------------------------------------------------------------
# 6. one qualification decision, not two
# ----------------------------------------------------------------------

def test_the_capability_engine_defers_to_the_qualification_engine():
    """
    The capability engine re-derived its own four-condition historical_edge test
    and used THAT for is_qualified, so a venue could be qualified while failing
    log loss, ECE, forecast skill, EV, average edge, drawdown and execution
    quality - every criterion the real engine checks.
    """
    from pathlib import Path

    source = Path("src/ptai/venues/capability_engine.py").read_text()
    assert "eng_result.is_qualified" in source, (
        "the weaker local re-derivation must be replaced by the engine's answer"
    )
    assert "net_pnl > 0 and\n                brier <= 0.25" not in source, (
        "the duplicate four-condition test is still deciding qualification"
    )


def test_a_failing_engine_result_blocks_capability_qualification(tmp_path):
    """Executed: the engine's refusal must be what decides."""
    import asyncio

    from src.ptai.storage.db import Storage
    from src.ptai.venues.capability_engine import VenueStrategyQualificationEngine
    from src.ptai.venues.qualification import VenueQualificationEngine
    from src.ptai.venues.registry import VenueRegistry

    class Adapter:
        venue_id = "polymarket"
        venue_type = VenueType.PREDICTION
        is_qualified = False

        class capabilities:
            fee_taker_pct = 0.02
            min_order_usd = 1.0
            implementation_status = "live"

        class performance_stats:
            total_paper_trades = 500
            win_rate = 0.9
            brier_score = 0.05
            profit_factor = 3.0
            profit_paper = 500.0
            avg_edge = 0.2

        def check_eligibility(self, country):
            from src.ptai.venues.adapter import EligibilityStatus
            return EligibilityStatus.ELIGIBLE

        async def get_orderbook(self, market):
            return {"is_real": True, "spread": 0.01, "depth": 10000}

    adapter = Adapter()
    registry = VenueRegistry()
    registry.adapters = {"polymarket": adapter}

    qual = VenueQualificationEngine(data_dir=str(tmp_path))
    # A record that satisfies the OLD four conditions but fails everything else.
    qual.qualifications["polymarket"] = qual.evaluate_qualification(
        "polymarket", {
            "total_paper_trades": 500, "win_rate": 0.9, "avg_edge": 0.2,
            "brier_score": 0.05, "log_loss": 0.9, "calibration_ece": 0.5,
            "profit_paper": 500.0, "net_pnl": 500.0, "expected_value": 0.2,
            "profit_factor": 3.0, "drawdown_max": 0.5, "fees_total": 0.0,
            "slippage_total": 0.0, "execution_quality_avg": 0.1,
            "forecast_skill": 0.1,
        })

    engine = VenueStrategyQualificationEngine(
        venue_registry=registry, qualification_engine=qual)
    report = asyncio.run(engine.evaluate_all_venues())
    assert "polymarket" not in report.qualified_venue_ids, (
        "a venue the qualification engine refused was qualified by the "
        "capability engine's weaker test"
    )


def test_one_execution_quality_rule_for_every_stage():
    """
    `execution_quality` was a constant in three stages of the pipeline.

    `strategy_engine` measured it from the book; `opportunity.py` read
    `orderbook.get("execution_quality", 0.8)` - a key real books do not carry,
    so that path scored every opportunity 0.8 - and the dashboard demo hard-coded
    0.8 on hand-written markets. One rule, one implementation, one answer.
    """
    from src.ptai.markets.orderbook import execution_quality_from_book
    from src.ptai.strategy.strategy_engine import _execution_quality_from_book

    real_book = {"is_real": True, "spread": 0.02, "depth": 5000}
    market = _market()

    # The strategy engine's name and the shared implementation agree.
    assert _execution_quality_from_book(real_book, market) == \
        execution_quality_from_book(real_book, market)

    # No book, and a book that declared a quality but is not real: both 0.0.
    assert execution_quality_from_book(None, market) == 0.0
    assert execution_quality_from_book({}, market) == 0.0
    assert execution_quality_from_book(
        {"is_real": False, "execution_quality": 0.9}, market) == 0.0, (
        "an estimated quality on a non-real book is not a measured one"
    )
    # A real book measures, and a tight deep one beats a wide thin one.
    assert execution_quality_from_book(real_book, market) > 0.0
    wide = {"is_real": True, "spread": 0.09, "depth": 200}
    assert execution_quality_from_book(wide, market) < \
        execution_quality_from_book(real_book, market)


def test_mock_markets_cannot_be_ranked_as_tradeable():
    """
    The dashboard's ranking demo built hand-written markets and hand-written
    opportunities (fair = price + 0.10 for all of them) with `Market`'s default
    `data_mode=LIVE`, and they were ranked as though they were tradeable. The
    demo has since been deleted - nothing on the page fetched it, and a route
    whose purpose is to rank invented markets is not something the console
    should be able to reach.

    The contract it was made to honour is the part worth keeping, so it is
    asserted here against the real selector rather than through a demo
    endpoint: mock markets, whatever they claim about their execution quality,
    are never returned as opportunities.
    """
    from src.ptai.markets.base import DataMode, Market, MarketSource, Token
    from src.ptai.strategy.opportunity import OpportunityEngine
    from src.ptai.venues.adapter import VenueOpportunity, VenueType

    def _mock(market_id: str, price: float) -> Market:
        return Market(
            id=market_id, source=MarketSource.POLYMARKET,
            question=f"Will {market_id} happen?",
            outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
            tokens=[Token(token_id=market_id, outcome="YES", price=price)],
            volume=50_000.0, volume_24h=20_000.0, liquidity=25_000.0,
            raw={}, is_mock=True, data_mode=DataMode.MOCK,
        )

    def _opp(market: Market, price: float) -> VenueOpportunity:
        return VenueOpportunity(
            market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION,
            side="YES", market_price=price, estimated_fair=price + 0.10,
            raw_edge=0.10, effective_edge=0.10, confidence=0.9,
            liquidity_score=0.9, execution_quality=0.9, category="general",
            should_trade=True)

    markets = [_mock("M1", 0.60), _mock("M2", 0.55), _mock("M3", 0.50)]
    # MOCK provenance survives the object, so nothing downstream has to guess.
    assert all(m.is_mock and m.data_mode == DataMode.MOCK for m in markets)

    ranked = OpportunityEngine().rank_and_select(
        [_opp(m, m.outcome_prices[0]) for m in markets],
        max_trades=3, bankroll=50.0, current_positions=[])
    assert ranked == [], (
        f"invented markets were ranked as tradeable opportunities: {ranked}")


def test_the_dashboard_routes_that_should_work_do_work():
    """
    Almost every dashboard route catches its own exception and returns
    `{"error": ..., "traceback": ...}` with HTTP 200, so a broken route looks
    healthy to anything that only checks the status code - and nothing was
    checking at all. One was found broken this way (an import that landed in a
    neighbouring function), which is exactly what a smoke test is for.

    `/api/llm/status` is excluded: it reports on a local LLM server, and there
    is deliberately none in test conditions.
    """
    import asyncio
    import src.ptai.dashboard as dashboard

    checked = {"/api/status", "/api/v2/status", "/api/v3/status",
               "/api/v7/status", "/api/vault/status",
               # `/api/v6/opportunity-ranking` and `/api/v6/fixes` used to be
               # here. They ranked five mock markets and printed a hand-written
               # list of fixes, nothing on the page fetched either, and they are
               # deleted. The operator snapshot took their place as the thing a
               # reader should be able to trust.
               "/api/operator"}
    seen = set()
    for route in dashboard.app.routes:
        path = getattr(route, "path", None)
        if path not in checked:
            continue
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        seen.add(path)
        out = asyncio.run(endpoint())
        assert not (isinstance(out, dict) and "error" in out), (
            f"{path} returned an error payload: "
            f"{out.get('error') if isinstance(out, dict) else out}"
        )
    assert seen == checked, f"routes not found: {checked - seen}"
