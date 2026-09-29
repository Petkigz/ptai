"""
SENT MEANS ASKED, PRICED MEANS PRICED, AND A PLAN IS NEITHER.

The operator's 16:53 log, on a V55 build:

    Local model: qwen/qwen3-14b (LMStudioProvider at http://localhost:1234/v1)
    - auto: first loaded model is qwen/qwen3-14b | NOT USED THIS CYCLE: 1
    market(s) were priced in full without the model: this process has no model
    router wired to the forecast engine | model time went to 1 of 200 screened
    market(s); 199 priced on their measured book alone

Four things are wrong in one line, and each of them is a different counter:

  * "1 market(s) were priced in full without the model" - the one deep market
    was refused by the RESOLUTION gate before a forecast was ever built, so the
    model was not "without" anything; it was never reached;
  * "this process has no model router wired to the forecast engine" - false: a
    local model had just been detected and named two clauses earlier;
  * "model time went to 1 of 200 screened market(s)" - 1 was the deep shortlist,
    a PLAN. No model time went anywhere: the router recorded no calls;
  * "199 priced on their measured book alone" - the screen's 199 was the count
    it did NOT give model time; same cycle's scan said most of them were never
    priced at all ("not evaluated - their book is an estimate or absent").

These tests pin the three counters apart and make each sentence come from the
counter that can actually support it.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from src.ptai.agent.v3_loop import TradingAgentV3, _why_the_model_was_not_used


# ----------------------------------------------------------------------
# the harness: the two methods under test over a synthetic cycle state
# ----------------------------------------------------------------------

class _Router:
    """A router that reports exactly what the test says it did."""

    def __init__(self, usage: Optional[Dict[str, Any]] = None,
                 model: str = "stub-model"):
        self.usage = usage or {"calls": 0, "answered": 0, "failed": 0,
                               "seconds": 0.0, "models": {}, "last_model": "",
                               "last_error": ""}
        self.active_model = model
        self.active_model_reason = "auto: first loaded model is stub-model"
        self._last = ""

    class _Provider:
        base_url = "http://localhost:1234/v1"
        model = "stub-model"

    provider = _Provider()

    def describe(self) -> str:
        return f"{self.active_model} (StubProvider at {self._Provider.base_url})"

    def get_provider_name(self) -> str:
        return "StubProvider"

    def usage_report(self) -> Dict[str, Any]:
        return dict(self.usage)

    def chat(self, prompt: str, system: str = ""):
        return None

    def reset_usage(self) -> None:
        self.usage = {"calls": 0, "answered": 0, "failed": 0, "seconds": 0.0,
                      "models": {}, "last_model": "", "last_error": ""}


class _Ensemble:
    def __init__(self, accounting: Dict[str, Any]):
        self._accounting = accounting
        self.llm_router: Any = None

    def llm_accounting(self) -> Dict[str, Any]:
        return dict(self._accounting)


class _Agent:
    dry_run = True

    def __init__(self, router: Optional[_Router] = None, *, shortlist: int = 0,
                 considered: int = 0, screened_out: int = 0, priced: int = 0,
                 deep_priced: int = 0, asked: int = 0, answered: int = 0,
                 forecasts_run: int = 0, calls: int = 0, answered_calls: int = 0,
                 problems: List[str] = (), last_model: str = ""):
        self.llm_router = router
        self._screen = {"considered": considered,
                        "shortlist": [{}] * shortlist,
                        "screened_out": screened_out}
        self._cycle_context_calls = priced
        self._cycle_deep_context_calls = deep_priced
        self._last_cycle_model = last_model
        if router is not None:
            router.usage = {"calls": calls, "answered": answered_calls,
                            "failed": max(0, calls - answered_calls),
                            "seconds": 1.5, "models": {"stub-model": calls},
                            "last_model": "stub-model", "last_error": ""}
        self._accounting = {"asked": asked, "answered": answered,
                            "forecasts_run": forecasts_run,
                            "problems": list(problems)}
        self.ensemble_forecaster = _Ensemble(self._accounting)


def _status_agent(**kwargs) -> _Agent:
    agent = _Agent(**kwargs)
    agent._local_model_status = TradingAgentV3._local_model_status.__get__(agent)
    agent._local_model_line = TradingAgentV3._local_model_line.__get__(agent)
    return agent


# ----------------------------------------------------------------------
# the operator's exact case: a deep market the forecast never reached
# ----------------------------------------------------------------------

class TestTheOperatorsCase:
    def test_the_reason_is_the_gate_that_refused_it_not_a_missing_router(self):
        agent = _status_agent(router=_Router(), shortlist=1, considered=200,
                              screened_out=199, priced=1, deep_priced=1,
                              forecasts_run=0, asked=0, calls=0)
        reason = agent._local_model_status()["not_used_reason"]
        assert "reached the pricing stage and were refused before a forecast " \
               "was built" in reason
        assert "resolution risk" in reason
        assert "no model router wired" not in reason, (
            "the router was detected and named in the same line; that sentence "
            "was simply false")

    def test_the_line_does_not_call_a_plan_model_time(self):
        agent = _status_agent(router=_Router(), shortlist=1, considered=200,
                              screened_out=199, priced=1, deep_priced=1)
        agent._cycle_scan_counts = {"venues": 19, "evaluated": 1,
                                    "skipped_no_book": 68, "beyond_cap": 100,
                                    "discovered": 200}
        line = agent._local_model_line()
        assert "the screen chose 1 of 200 market(s) for deep analysis" in line
        assert "a plan" in line
        assert "model time went to" not in line
        assert "priced on their measured book alone" not in line
        assert "venue scan: priced 1 market(s) of the 200 read" in line
        assert "68 had no usable book" in line
        assert "100 beyond their venue's cap" in line

    def test_the_line_names_the_model_that_would_have_been_asked(self):
        agent = _status_agent(router=_Router(model="qwen/qwen3-14b"),
                              shortlist=1, considered=200, deep_priced=1,
                              priced=1)
        line = agent._local_model_line()
        assert line.startswith("Local model: qwen/qwen3-14b")
        assert "NOT USED THIS CYCLE" in line
        assert "no model router wired" not in line


# ----------------------------------------------------------------------
# an ask that was never a call
# ----------------------------------------------------------------------

class TestAnAskThatNeverBecameACall:
    def test_the_disagreement_is_named_not_smoothed_over(self):
        agent = _status_agent(router=_Router(), shortlist=8, considered=899,
                              priced=8, deep_priced=8, asked=8, answered=0,
                              calls=0, forecasts_run=8)
        status = agent._local_model_status()
        assert status["asks_without_calls"] == 8
        assert "the ask never reached the provider" in status["not_used_reason"]
        assert "WIRING GAP" in agent._local_model_line()

    def test_matching_counters_do_not_claim_a_wiring_gap(self):
        agent = _status_agent(router=_Router(), shortlist=8, considered=899,
                              priced=8, deep_priced=8, asked=8, answered=0,
                              calls=8, forecasts_run=8,
                              problems=["APITimeoutError: timed out"])
        status = agent._local_model_status()
        assert status["asks_without_calls"] == 0
        assert "APITimeoutError" in status["not_used_reason"]
        assert "WIRING GAP" not in agent._local_model_line()

    def test_the_reason_builder_is_reachable_without_an_agent(self):
        """The console and the CLI read this line through other objects."""
        reason = _why_the_model_was_not_used(
            {"available": False, "describe": "none", "calls": 0, "failed": 0,
             "last_error": "", "deep_shortlist": 0, "deep_priced": 0,
             "forecasts_run": 0, "considered": 5, "answered_by_model": 0},
            0, [])
        assert "no local model server answered" in reason


# ----------------------------------------------------------------------
# the counters are separate facts
# ----------------------------------------------------------------------

class TestThreeCountersThreeFacts:
    def test_the_status_reports_prices_forecasts_and_asks_separately(self):
        agent = _status_agent(router=_Router(), shortlist=3, considered=10,
                              priced=9, deep_priced=3, forecasts_run=2,
                              asked=1, answered=1, calls=1, answered_calls=1)
        status = agent._local_model_status()
        assert status["priced"] == 9
        assert status["deep_priced"] == 3
        assert status["forecasts_run"] == 2
        assert status["asked"] == 1
        assert status["deep_shortlist"] == 3
        assert status["used"] is True

    def test_used_is_decided_by_answers_not_by_a_plan(self):
        agent = _status_agent(router=_Router(), shortlist=8, considered=40,
                              priced=0, deep_priced=0)
        assert agent._local_model_status()["used"] is False

    def test_a_forecast_that_ran_without_any_model_ask_says_the_gap(self):
        agent = _status_agent(router=_Router(), shortlist=4, considered=40,
                              priced=4, deep_priced=4, forecasts_run=4,
                              asked=0, calls=0)
        reason = agent._local_model_status()["not_used_reason"]
        assert "no model router wired to the forecast engine" in reason


# ----------------------------------------------------------------------
# the counters are wired, not decorative
# ----------------------------------------------------------------------

class TestTheCountersAreWired:
    def test_one_ensemble_prices_the_cycle_and_counts_it(self):
        source = inspect.getsource(TradingAgentV3.__init__)
        assert "self.fair_value_engine.ensemble_forecaster = " \
               "self.ensemble_forecaster" in source, (
            "two ensemble objects meant the cycle's line read a counter that "
            "had never seen a market")

    def test_the_forecast_pipeline_counts_the_markets_it_runs(self):
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        source = inspect.getsource(EnsembleForecaster.forecast_market)
        assert "self.forecasts_run" in source
        source_reset = inspect.getsource(EnsembleForecaster.reset_llm_accounting)
        assert "forecasts_run" in source_reset

    def test_the_cycle_resets_and_reports_both(self):
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "ensemble_forecaster.reset_llm_accounting()" in source
        assert "self._cycle_scan_counts" in source
        assert "Venue scan priced" in source

    def test_the_scan_report_carries_the_pricing_counts(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3
        source = inspect.getsource(StrategyEngineV3.scan_venue)
        assert "evaluated=evaluated" in source
        assert "skipped_no_book=skipped_no_book" in source
        assert "beyond_cap" in source

    def test_the_screen_does_not_claim_to_have_priced_anything(self):
        source = inspect.getsource(TradingAgentV3._prescan_and_rank)
        assert "priced on the cheap context only" not in source
        assert "none of them has been priced yet" in source
        assert "they cannot be priced" in source
