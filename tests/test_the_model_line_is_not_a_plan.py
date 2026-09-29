"""
"SENT TO THE MODEL" IS A COUNT OF SENDS, NOT A SHORTLIST.

The operator's 2026-09-29 log carried this line, in a cycle that spent 155
seconds and made no model calls at all:

    Local model: qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp (...) - auto:
    first loaded model is qwen3.8-27b-... | NOT USED THIS CYCLE: 8 market(s)
    were sent to the model but no answer was recorded

The router's own counter in that same cycle read 0 calls. Both numbers were
printed from the same dict, and they contradicted each other: 8 was the deep
shortlist - a PLAN the screen writes before pricing - and the sentence claimed
it was a record of sends. A market counts as sent here only where the call is
made, and the line now says which of three different facts is true:

  * the model was asked and answered;
  * the model was asked and nothing came back (and why);
  * the shortlist never reached the pricing stage, so the model was never asked.

It also pins the cause of the starvation: the screen must only choose markets
the pricing stage will actually price, or the cycle's model budget is spent on
markets that are dropped before any model time happens.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pytest

from src.ptai.intelligence.ensemble import EnsembleForecaster
from src.ptai.llm.provider import LLMResponse


# ----------------------------------------------------------------------
# a router that counts what it was asked, and an ensemble to ask it
# ----------------------------------------------------------------------

class _StubRouter:
    def __init__(self, *, answer: bool = True):
        self.answer = answer
        self.seen: List[str] = []
        self.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                      "models": {}, "last_model": "", "last_error": ""}
        self.active_model = "stub-model"
        self.active_model_reason = "auto: first loaded model"

    def describe(self) -> str:
        return "stub-model (StubProvider at http://localhost:1234/v1) - auto"

    def get_provider_name(self) -> str:
        return "StubProvider"

    def usage_report(self) -> Dict[str, Any]:
        return dict(self.usage)

    def reset_usage(self) -> None:
        self.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                      "models": {}, "last_model": "", "last_error": ""}

    def chat(self, prompt: str, system: str = "") -> Optional[LLMResponse]:
        self.seen.append(prompt[:40])
        self.usage["calls"] += 1
        if not self.answer:
            self.usage["failed"] += 1
            return None
        self.usage["answered"] += 1
        self.usage["models"]["stub-model"] = self.usage["models"].get(
            "stub-model", 0) + 1
        return LLMResponse(content='{"fair_value": 0.5, "confidence": 0.6}',
                           model="stub-model", provider="lm_studio",
                           parsed_json={"fair_value": 0.5, "confidence": 0.6})


class _Provider:
    base_url = "http://localhost:1234/v1"
    model = "stub-model"


def _router(**kwargs) -> _StubRouter:
    r = _StubRouter(**kwargs)
    r.provider = _Provider()
    return r



class _Capture:
    """Collect loguru messages for one block of code."""

    def __init__(self):
        self.lines = []

    def __enter__(self):
        from loguru import logger
        self._sink = logger.add(
            lambda m: self.lines.append((m.record["level"].name,
                                         m.record["message"])),
            level="DEBUG")
        return self

    def __exit__(self, *exc):
        from loguru import logger
        logger.remove(self._sink)
        return False

    def messages(self, level=None):
        return [msg for lvl, msg in self.lines
                if level is None or lvl == level]


class _Agent:
    """The two methods under test, over a synthetic cycle state."""

    dry_run = True

    def __init__(self, router=None, *, shortlist: int = 0, considered: int = 0,
                 screened_out: int = 0, priced: int = 0, deep_priced: int = 0,
                 asked: int = 0, answered: int = 0, problems=(), last_model=""):
        self.llm_router = router
        self._screen = {"considered": considered,
                        "shortlist": [{}] * shortlist,
                        "screened_out": screened_out}
        self._cycle_context_calls = priced
        self._cycle_deep_context_calls = deep_priced
        self._last_cycle_model = last_model
        self._accounting = {"asked": asked, "answered": answered,
                            "problems": list(problems)}

        class _Ensemble:
            def __init__(self, accounting):
                self._accounting = accounting

            def llm_accounting(self):
                return dict(self._accounting)

        self.ensemble_forecaster = _Ensemble(self._accounting)


def _status_agent(**kwargs):
    from src.ptai.agent.v3_loop import TradingAgentV3
    agent = _Agent(**kwargs)
    agent._local_model_status = TradingAgentV3._local_model_status.__get__(agent)
    agent._local_model_line = TradingAgentV3._local_model_line.__get__(agent)
    return agent


# ----------------------------------------------------------------------
# the operator's exact line
# ----------------------------------------------------------------------

class TestTheOperatorsLine:
    def test_a_shortlist_that_was_never_priced_is_not_a_send(self):
        router = _router()
        agent = _status_agent(router=router, shortlist=8, considered=899,
                              screened_out=891, priced=0, deep_priced=0,
                              asked=0, answered=0)
        line = agent._local_model_line()
        assert "8 market(s) were chosen for deep analysis but none of them " \
               "reached the pricing stage" in line
        assert "the model was never asked" in line
        # the lie is gone
        assert "were sent to the model" not in line

    def test_the_reason_names_the_filters_that_dropped_them(self):
        agent = _status_agent(router=_router(), shortlist=8, considered=899,
                              priced=0, deep_priced=0)
        reason = agent._local_model_status()["not_used_reason"]
        assert "volume, liquidity or an unvalidated book" in reason

    def test_the_status_dict_separates_asked_from_shortlisted(self):
        agent = _status_agent(router=_router(), shortlist=8, considered=899,
                              priced=0, deep_priced=0)
        status = agent._local_model_status()
        assert status["deep_shortlist"] == 8
        assert status["asked"] == 0
        assert status["priced"] == 0
        assert status["used"] is False


# ----------------------------------------------------------------------
# asked, and answered
# ----------------------------------------------------------------------

class TestAskedAndAnswered:
    def test_two_markets_asked_and_answered_are_reported_as_such(self):
        router = _router()
        router.usage = {"calls": 2, "answered": 2, "failed": 0, "seconds": 12.0,
                        "models": {"stub-model": 2}, "last_model": "stub-model",
                        "last_error": ""}
        agent = _status_agent(router=router, shortlist=8, considered=899,
                              priced=2, deep_priced=2, asked=2, answered=2)
        line = agent._local_model_line()
        assert "markets asked: 2" in line
        assert "2 of 2 call(s) returned" in line
        assert "NOT USED" not in line

    def test_asked_and_nothing_came_back_names_the_problem(self):
        router = _router()
        router.usage = {"calls": 2, "answered": 0, "failed": 2, "seconds": 90.0,
                        "models": {}, "last_model": "",
                        "last_error": "APITimeoutError: timed out"}
        agent = _status_agent(router=router, shortlist=2, considered=40,
                              priced=2, deep_priced=2, asked=2, answered=0,
                              problems=["APITimeoutError: timed out"])
        line = agent._local_model_line()
        assert "2 market(s) reached the model step and none was answered" in line
        assert "APITimeoutError: timed out" in line

    def test_a_call_that_reached_the_ensemble_but_not_the_router_is_named(self):
        """The counter disagreement itself must be visible, not smoothed over."""
        router = _router()
        router.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                        "models": {}, "last_model": "", "last_error": ""}
        agent = _status_agent(router=router, shortlist=1, considered=10,
                              priced=1, deep_priced=1, asked=1, answered=0)
        reason = agent._local_model_status()["not_used_reason"]
        assert "the ask never reached the provider" in reason


# ----------------------------------------------------------------------
# a model that changed between cycles
# ----------------------------------------------------------------------

class TestTheModelIsNamedConsistently:
    def test_a_changed_model_is_announced_with_the_previous_name(self):
        router = _router()
        agent = _status_agent(router=router, shortlist=0,
                              last_model="qwen/qwen3-14b")
        line = agent._local_model_line()
        assert "CHANGED since the last cycle" in line
        assert "it was qwen/qwen3-14b, and this cycle it is stub-model" in line

    def test_the_same_model_twice_is_not_a_change(self):
        router = _router()
        agent = _status_agent(router=router, shortlist=0,
                              last_model="stub-model")
        line = agent._local_model_line()
        assert "CHANGED" not in line


# ----------------------------------------------------------------------
# the counting itself
# ----------------------------------------------------------------------

class TestTheEnsembleCountsWhatItWasAsked:
    def test_the_ensemble_counts_a_market_it_was_asked_about(self, monkeypatch):
        ensemble = EnsembleForecaster(llm_router=_router())
        assert ensemble.llm_accounting() == {"asked": 0, "answered": 0,
                                             "forecasts_run": 0,
                                             "problems": []}
        from src.ptai.markets.base import Market, MarketSource, DataMode
        market = Market(id="m1", source=MarketSource.POLYMARKET,
                        question="Will it rain?", outcome_prices=[0.5],
                        volume_24h=1000, liquidity=1000,
                        venue_id="polymarket", data_mode=DataMode.LIVE)
        ensemble.forecast_market(market, context={"deep_analysis": True,
                                                  "orderbook": {"is_real": True,
                                                                "spread": 0.02}})
        accounting = ensemble.llm_accounting()
        assert accounting["asked"] == 1
        assert accounting["answered"] == 1

    def test_a_reset_clears_the_cycle_count(self):
        ensemble = EnsembleForecaster(llm_router=_router())
        ensemble.llm_asked = 5
        ensemble.llm_answered = 3
        ensemble.reset_llm_accounting()
        assert ensemble.llm_accounting() == {"asked": 0, "answered": 0,
                                             "forecasts_run": 0,
                                             "problems": []}

    def test_a_fallback_answer_is_not_counted_as_an_answer(self):
        ensemble = EnsembleForecaster(llm_router=_router(answer=False))
        from src.ptai.markets.base import Market, MarketSource, DataMode
        market = Market(id="m2", source=MarketSource.POLYMARKET,
                        question="Will it snow?", outcome_prices=[0.5],
                        volume_24h=1000, liquidity=1000,
                        venue_id="polymarket", data_mode=DataMode.LIVE)
        result = ensemble.forecast_market(market, context={"deep_analysis": True,
                                                          "orderbook": {"is_real": True}})
        accounting = ensemble.llm_accounting()
        assert accounting["asked"] == 1
        assert accounting["answered"] == 0
        assert accounting["problems"], "why no answer came back must be recorded"
        # the ensemble fell back to the heuristic, so no LLM component was added
        llm_rows = [c for c in result.components
                    if "llm" in str(getattr(c, "name", "") or "").lower()]
        assert llm_rows == []


# ----------------------------------------------------------------------
# the wiring, so the count cannot be missing from the sentence again
# ----------------------------------------------------------------------

class TestOneEnsemble:
    def test_the_agent_prices_with_the_ensemble_it_reports_on(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.__init__)
        assert "self.fair_value_engine.ensemble_forecaster = self.ensemble_forecaster" \
            in source, ("the pricing engine must share the agent's ensemble, or "
                        "the cycle counts calls in one object and reports on another")

    def test_the_cycle_resets_the_count_it_reports(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "reset_llm_accounting" in source

    def test_the_forecast_engine_does_not_build_a_second_ensemble(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.__init__)
        assert source.count("EnsembleForecaster(") == 1


# ----------------------------------------------------------------------
# the starvation itself: the screen must not shortlist what the scan drops
# ----------------------------------------------------------------------

class TestTheScreenOnlyShortlistsWhatWillBePriced:
    def _screen(self, market, book):
        from src.ptai.agent.v3_loop import TradingAgentV3

        class _Engine:
            min_volume_24h = 500
            min_liquidity = 100
            max_spread = 0.10

        class _Self:
            strategy_engine_v3 = _Engine()

        return TradingAgentV3._screen_score(_Self(), market, book)

    def _market(self, **kw):
        from src.ptai.markets.base import Market, MarketSource, DataMode
        base = dict(id="m", source=MarketSource.POLYMARKET,
                    question="Will it be sunny?", outcome_prices=[0.4],
                    volume_24h=50_000, liquidity=40_000,
                    venue_id="polymarket", data_mode=DataMode.LIVE)
        base.update(kw)
        return Market(**base)

    def test_a_validated_book_with_volume_and_liquidity_is_shortlisted(self):
        score, why = self._screen(self._market(),
                                  {"validated": True, "spread": 0.02,
                                   "depth": 5000})
        assert score > 0, why

    def test_a_market_below_the_scan_volume_floor_is_not_shortlisted(self):
        score, why = self._screen(self._market(volume_24h=10),
                                  {"validated": True, "spread": 0.02,
                                   "depth": 5000})
        assert score == -1.0
        assert "below the 500 the scan will trade" in why

    def test_a_market_below_the_scan_liquidity_floor_is_not_shortlisted(self):
        score, why = self._screen(self._market(liquidity=5),
                                  {"validated": True, "spread": 0.02,
                                   "depth": 5000})
        assert score == -1.0
        assert "below the 100 the scan will trade" in why

    def test_an_inactive_market_is_not_shortlisted(self):
        score, why = self._screen(self._market(active=False),
                                  {"validated": True, "spread": 0.02,
                                   "depth": 5000})
        assert score == -1.0
        assert "not active" in why

    def test_the_floors_are_read_from_the_scan_engine_not_hardcoded(self):
        """A screen with its own idea of the floors is how they drifted apart."""
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._screen_score)
        assert "engine.min_volume_24h" in source
        assert "engine.min_liquidity" in source


# ----------------------------------------------------------------------
# the starvation itself, proved with the real screen and the real scan
# ----------------------------------------------------------------------

class TestTheShortlistReachesTheScan:
    """A market the scan will not price must not be chosen for model time.

    This is the operator's cycle, rebuilt: one venue where the top-100 by
    volume have thin books and the deepest books belong to eight low-volume
    markets. The screen ranked the eight on book depth, the scan dropped all
    eight at its own per-venue cap, and the cycle spent exactly zero model time
    while printing a line about markets being sent to the model.
    """

    @pytest.fixture()
    def agent_and_router(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "align.db"))
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.markets.base import DataMode, Market, MarketSource

        router = _router()
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.llm_router = router
        agent.ensemble_forecaster.llm_router = router
        agent.fair_value_engine.llm_router = router
        agent.venue_registry = _StubRegistry()
        agent._cycle_books = {}

        markets = []
        for i in range(158):
            lottery = i >= 150
            markets.append(Market(
                id=f"lottery-{i - 150}" if lottery else str(4_000_000 + i),
                source=MarketSource.POLYMARKET,
                question=f"Will market {i} resolve YES?",
                outcome_prices=[0.001 if lottery else 0.35],
                volume_24h=120 if lottery else 40_000 + i,
                liquidity=60 if lottery else 20_000,
                venue_id="polymarket", data_mode=DataMode.LIVE,
                raw={"venue_id": "polymarket"}))
        return agent, router, markets

    @pytest.mark.asyncio
    async def test_every_shortlisted_market_is_one_the_scan_will_price(
            self, agent_and_router):
        agent, router, markets = agent_and_router
        screen = await agent._prescan_and_rank(markets)
        shortlisted = {row["market_id"] for row in screen["shortlist"]}
        assert shortlisted, "a reachable shortlist must exist"
        evaluated = {m.id for m in agent.strategy_engine_v3
                     .markets_that_will_be_evaluated(markets)}
        assert shortlisted <= evaluated, (
            "the screen chose markets the scan's own per-venue cap will drop: "
            "model time would be spent on nothing")

    @pytest.mark.asyncio
    async def test_the_cycle_then_asks_the_model_for_every_shortlisted_market(
            self, agent_and_router):
        agent, router, markets = agent_and_router
        screen = await agent._prescan_and_rank(markets)
        agent.ensemble_forecaster.reset_llm_accounting()
        agent._cycle_context_calls = 0
        agent._cycle_deep_context_calls = 0
        for market in markets:
            if market.id in {row["market_id"] for row in screen["shortlist"]}:
                context = await agent.get_context_for_market(market)
                agent.strategy_engine_v3.evaluate_market_with_all_strategies(
                    market, context=context)
        status = agent._local_model_status()
        assert status["deep_shortlist"] > 0
        assert status["asked"] == status["deep_shortlist"]
        assert status["used"] is True
        assert "markets asked" in agent._local_model_line()

    @pytest.mark.asyncio
    async def test_the_screen_says_how_many_the_cap_will_drop(
            self, agent_and_router):
        agent, _router_, markets = agent_and_router
        screen = await agent._prescan_and_rank(markets)
        assert screen["dropped_by_scan"] == 50

    def test_the_scan_and_the_screen_share_one_definition(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3
        source = inspect.getsource(StrategyEngineV3.scan_venue)
        assert "self.evaluate_limit" in source
        assert "markets_that_will_be_evaluated" in inspect.getsource(
            StrategyEngineV3)

    def test_the_per_market_line_does_not_claim_a_rank_it_never_computed(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._deep_analysis_for)
        assert "ranked below it" not in source
        assert "market(s) in total were priced on their measured book " in source


class _StubAdapter:
    venue_id = "polymarket"

    class capabilities:
        fee_taker_pct = 0.0
        order_gas_usd = 0.0

    async def get_orderbook(self, market):
        lottery = market.id.startswith("lottery")
        return {"is_real": True, "validated": True, "source": "clob_real",
                "spread": 0.001 if lottery else 0.02,
                "depth": 900_000 if lottery else 3_000,
                "asks": [[0.002 if lottery else 0.36, 900]],
                "bids": [[0.001 if lottery else 0.34, 900]],
                "executable": True}


class _StubRegistry:
    def __init__(self):
        self.adapter = _StubAdapter()

    def get_adapter_for_market(self, market):
        return self.adapter


# ----------------------------------------------------------------------
# the second half of the same defect: a plan is not a price
# ----------------------------------------------------------------------

class TestAPlanIsNotAPrice:
    """The 16:53 cycle, where the model line was wrong again - in a new way.

    It read:

        NOT USED THIS CYCLE: 1 market(s) were priced in full without the model:
        this process has no model router wired to the forecast engine

    while the same cycle showed `Priced 4949306 in 0.0s: no trade (fair 0.595
    vs market 0.595) - Blocked by resolution risk`. The router was wired; the
    one deep market was refused by a gate BEFORE the forecast, so the model was
    never reached. And the line's tail said "199 priced on their measured book
    alone" in a cycle where the venues priced one market - the 199 was the
    screen's plan.
    """

    def test_a_deep_market_refused_before_the_forecast_says_so(self):
        agent = _status_agent(router=_router(), shortlist=1, considered=200,
                              priced=1, deep_priced=1, asked=0, answered=0)
        status = agent._local_model_status()
        status["forecasts_run"] = 0
        agent._accounting["forecasts_run"] = 0
        reason = agent._local_model_status()["not_used_reason"]
        assert "refused before a forecast was built" in reason
        assert "the model was never asked" in reason
        assert "no model router wired" not in reason

    def test_a_forecast_that_ran_without_a_router_still_says_that(self):
        agent = _status_agent(router=_router(), shortlist=1, considered=200,
                              priced=1, deep_priced=1)
        agent._accounting["forecasts_run"] = 3
        reason = agent._local_model_status()["not_used_reason"]
        assert "no model router wired to the forecast engine" in reason

    def test_the_tail_reports_the_venues_own_counts(self):
        agent = _status_agent(router=_router(), shortlist=1, considered=200,
                              priced=1, deep_priced=1)
        agent._cycle_scan_counts = {"venues": 19, "evaluated": 1,
                                    "skipped_no_book": 99, "beyond_cap": 100,
                                    "discovered": 200}
        line = agent._local_model_line()
        assert ("venue scan: priced 1 market(s) of the 200 read (99 had no "
                "usable book, 100 beyond their venue's cap)") in line
        assert "priced on their measured book alone" not in line

    def test_the_screen_line_does_not_call_a_plan_a_price(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._prescan_and_rank)
        assert "none of them has been priced yet" in source
        assert "priced on the cheap context only" not in source
        assert "they cannot be priced" in source

    def test_the_scan_report_carries_what_it_actually_did(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        class _Ctx:
            async def get_context(self, market):
                return {"deep_analysis": True}

        engine = StrategyEngineV3()
        markets = [self._market(f"m{i}") for i in range(3)]

        async def _scan():
            return await engine.scan_venue("polymarket", markets,
                                           context_provider=_Ctx(),
                                           book_lookup=lambda mid: {
                                               "is_real": True, "validated": True,
                                               "spread": 0.02, "depth": 5000})

        import asyncio
        report, _opps = asyncio.run(_scan())
        assert report.evaluated == 3
        assert report.skipped_no_book == 0
        assert report.beyond_cap == 0

    def test_a_market_with_no_usable_book_is_counted_not_priced(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        class _Ctx:
            async def get_context(self, market):
                return {"deep_analysis": True}

        engine = StrategyEngineV3()
        markets = [self._market("a"), self._market("b")]

        async def _scan():
            return await engine.scan_venue(
                "polymarket", markets, context_provider=_Ctx(),
                book_lookup=lambda mid: {"is_real": False, "validated": False})

        import asyncio
        report, _opps = asyncio.run(_scan())
        assert report.evaluated == 0
        assert report.skipped_no_book == 2

    def test_markets_beyond_the_per_venue_cap_are_counted(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        class _Ctx:
            async def get_context(self, market):
                return {"deep_analysis": True}

        engine = StrategyEngineV3()
        engine.evaluate_limit = 5
        markets = [self._market(f"m{i}", volume_24h=40_000 + i)
                   for i in range(12)]

        async def _scan():
            return await engine.scan_venue(
                "polymarket", markets, context_provider=_Ctx(),
                book_lookup=lambda mid: {"is_real": True, "validated": True,
                                         "spread": 0.02, "depth": 5000})

        import asyncio
        report, _opps = asyncio.run(_scan())
        assert report.evaluated == 5
        assert report.beyond_cap == 7

    def _market(self, market_id, **overrides):
        from src.ptai.markets.base import Market, MarketSource, DataMode
        fields = dict(id=market_id, source=MarketSource.POLYMARKET,
                      question="Will it happen?", outcome_prices=[0.42],
                      volume_24h=50_000, liquidity=5_000,
                      venue_id="polymarket", data_mode=DataMode.LIVE,
                      raw={"venue_id": "polymarket"})
        fields.update(overrides)
        return Market(**fields)


# ----------------------------------------------------------------------
# the counters themselves
# ----------------------------------------------------------------------

class TestTheCountersCount:
    def test_a_forecast_is_counted_where_it_runs(self):
        """forecasts_run separates "reached pricing" from "built a forecast".

        The 16:53 cycle had one deep market, refused by the resolution gate
        before the ensemble ran. Without this counter the line could only see
        "1 priced" and blamed the missing router.
        """
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        from src.ptai.markets.base import Market, MarketSource, DataMode
        ens = EnsembleForecaster()
        assert ens.llm_accounting()["forecasts_run"] == 0
        market = Market(id="m1", source=MarketSource.POLYMARKET,
                        question="Will it happen?", outcome_prices=[0.4],
                        volume_24h=50_000, liquidity=5_000, venue_id="polymarket",
                        data_mode=DataMode.LIVE, raw={"venue_id": "polymarket"})
        ens.forecast_market(market, context={"deep_analysis": True})
        assert ens.llm_accounting()["forecasts_run"] == 1
        ens.reset_llm_accounting()
        assert ens.llm_accounting()["forecasts_run"] == 0

    def test_an_ask_that_was_never_counted_as_a_call_is_named(self):
        agent = _status_agent(router=_router(), shortlist=1, considered=10,
                              priced=1, deep_priced=1, asked=3, answered=0)
        status = agent._local_model_status()
        assert status["asks_without_calls"] == 3
        line = agent._local_model_line()
        assert "WIRING GAP" in line
        assert "never reached the provider" in line

    def test_no_wiring_gap_is_claimed_when_the_calls_match(self):
        router = _router()
        router.usage = {"calls": 3, "answered": 0, "failed": 3, "seconds": 5.0,
                        "models": {}, "last_model": "", "last_error": "boom"}
        agent = _status_agent(router=router, shortlist=1, considered=10,
                              priced=3, deep_priced=3, asked=3, answered=0,
                              problems=["boom"])
        assert agent._local_model_status()["asks_without_calls"] == 0
