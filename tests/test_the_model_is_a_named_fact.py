"""
The model id is a named fact, in every place it is decided or used.

The operator's question: "the lm studio model id is nowhere to be found in any
of these runs, whats happening, why is it not being used". The audit behind
this file found all four holes:

  * `LLMRouter._detect()` logged `model={self.model}` - the CONFIGURED value,
    which defaults to the placeholder "local-model", and it never resolved the
    auto choice at detection time (that happened lazily inside the first
    `chat()`, where nothing printed it);
  * `LMStudioProvider.chat()` logged nothing on success - no model, no time;
  * `LLMResponse.model` - the id the server says answered - was read NOWHERE in
    the codebase;
  * a market the model was never asked about was logged as
    `Using fallback heuristic for {market.id}` with no reason and no model id.

These tests pin the fix: the model is named when it is chosen, when it answers,
when it fails, per cycle with counts, on the result that a trade is recorded
from, and on the component row the console prints. A run with no model says so
and says why - it cannot look like a run with one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.ptai.llm import provider as provider_module
from src.ptai.llm.provider import (
    LMStudioProvider,
    LLMRouter,
    LLMResponse,
    choose_loaded_model,
    probe_local_model,
)

# The models the operator's machine has, in their order: the R1 first.
OPERATORS_MODELS = ["deepseek-r1-distill-qwen-32b",
                    "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp",
                    "qwen/qwen3-32b"]


def _captured():
    """A loguru sink the tests can read, and its remover."""
    from loguru import logger as loguru_logger
    lines: List[str] = []
    sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                level="DEBUG")
    return lines, (lambda: loguru_logger.remove(sink_id))


def _fake_lm_studio(monkeypatch, models: Optional[List[str]] = None,
                    *, connected: bool = True) -> None:
    """LM Studio answering with `models`, without any HTTP."""
    models = OPERATORS_MODELS if models is None else models
    monkeypatch.setattr(LMStudioProvider, "is_available", lambda self: connected)
    monkeypatch.setattr(LMStudioProvider, "list_models",
                        lambda self: list(models) if connected else [])


class _StubCompletions:
    def __init__(self, content: str, model: str):
        self._content, self._model = content, model

    def create(self, **_kwargs):
        return SimpleNamespace(
            model=self._model,
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._content))])


def _stub_openai(monkeypatch, content: str, model: str) -> None:
    """A stubbed `openai` module, so a call can be made with no server."""
    import sys
    import types
    module = types.ModuleType("openai")

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=_StubCompletions(content, model))

    module.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)


# ---------------------------------------------------------------------------
# 1. The model is named when it is CHOSEN - at detection, not in chat()
# ---------------------------------------------------------------------------

class TestTheModelIsNamedWhenItIsChosen:
    def test_the_startup_line_names_the_model_not_the_placeholder(self, monkeypatch):
        _fake_lm_studio(monkeypatch)
        lines, done = _captured()
        try:
            router = LLMRouter(preferred="lm_studio", model="local-model",
                               lm_studio_host="http://localhost:1234")
        finally:
            done()
        named = [l for l in lines if "LOCAL MODEL:" in l]
        assert named, "nothing named the model at startup"
        assert "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp" in named[0]
        assert "R1-style" in named[0], (
            "the line must say WHY that model - the R1 first in the list is "
            "the trap it exists to avoid")
        assert router.active_model == "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"
        # ...and the provider calls THAT model, so the log cannot name one
        # model while another is called.
        assert router.provider.model == router.active_model
        assert router.provider.model != "local-model"

    def test_a_pinned_model_is_named_as_pinned(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen/qwen3-32b", "deepseek-r1"])
        router = LLMRouter(preferred="lm_studio", model="qwen/qwen3-32b")
        assert router.active_model == "qwen/qwen3-32b"
        assert router.active_model_reason.startswith("pinned")

    def test_a_pinned_model_that_is_not_loaded_says_so(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen/qwen3-32b"])
        router = LLMRouter(preferred="lm_studio", model="prism-ml/bonsai-27b")
        assert router.active_model == "qwen/qwen3-32b"
        assert "NOT loaded" in router.active_model_reason
        assert "prism-ml/bonsai-27b" in router.describe()

    def test_a_connected_server_with_no_model_loaded_is_not_a_model(self, monkeypatch):
        _fake_lm_studio(monkeypatch, [])
        router = LLMRouter(preferred="lm_studio", model="local-model")
        assert router.active_model is None
        assert "no model is loaded" in router.active_model_reason

    def test_with_no_server_at_all_the_line_says_none_and_where(self, monkeypatch):
        _fake_lm_studio(monkeypatch, connected=False)
        lines, done = _captured()
        try:
            router = LLMRouter(preferred="lm_studio", model="local-model",
                               lm_studio_host="http://localhost:4999")
        finally:
            done()
        assert any("LOCAL MODEL: none" in l and "4999" in l for l in lines)
        assert router.active_model is None
        assert "no local model server answered" in router.describe()

    def test_the_probe_used_by_the_panel_makes_the_same_choice(self, monkeypatch):
        """
        One decision, two readers: the startup panel and the agent's router
        cannot name different models for the same machine.
        """
        _fake_lm_studio(monkeypatch)
        probed = probe_local_model("http://localhost:1234", "local-model")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        assert probed["model"] == router.active_model
        assert probed["reason"] == router.active_model_reason

    def test_the_probe_never_raises_on_a_dead_server(self, monkeypatch):
        _fake_lm_studio(monkeypatch, connected=False)
        probed = probe_local_model("http://localhost:4999", "local-model")
        assert probed["connected"] is False
        assert probed["model"] is None
        assert "not answering" in probed["reason"]


# ---------------------------------------------------------------------------
# 2. The model is named when it ANSWERS, and what it did is counted
# ---------------------------------------------------------------------------

class TestTheModelIsNamedWhenItAnswers:
    def test_the_answer_line_names_the_model_the_server_reported(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        _stub_openai(monkeypatch, json.dumps({"fair_value": 0.62}), "qwen2.5-14b-instruct")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        lines, done = _captured()
        try:
            response = router.chat("prompt", "system")
        finally:
            done()
        assert response is not None and response.model == "qwen2.5-14b-instruct"
        assert any("answered with 'qwen2.5-14b-instruct'" in l for l in lines)
        assert router.provider.last_call["model"] == "qwen2.5-14b-instruct"
        assert router.provider.last_call["ok"] is True
        usage = router.usage_report()
        assert usage["calls"] == 1 and usage["answered"] == 1 and usage["failed"] == 0
        assert usage["models"] == {"qwen2.5-14b-instruct": 1}
        # Elapsed time is recorded (a stubbed call is simply very fast).
        assert isinstance(usage["seconds"], float)
        assert isinstance(router.provider.last_call["seconds"], float)

    def test_a_proxy_that_answers_with_another_id_is_reported_as_it_answered(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["the-model-i-pinned"])
        _stub_openai(monkeypatch, json.dumps({"fair_value": 0.5}), "some-other-model")
        router = LLMRouter(preferred="lm_studio", model="the-model-i-pinned")
        lines, done = _captured()
        try:
            router.chat("prompt", "system")
        finally:
            done()
        assert any("answered with 'some-other-model'" in l for l in lines)
        assert any("requested 'the-model-i-pinned'" in l for l in lines)

    def test_a_failure_names_the_model_the_limit_and_the_error(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen1"])
        import sys
        import types
        module = types.ModuleType("openai")

        class _Boom:
            def create(self, **_kwargs):
                raise TimeoutError("Request timed out.")

        class _Client:
            def __init__(self, **_kwargs):
                self.chat = SimpleNamespace(completions=_Boom())

        module.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", module)
        router = LLMRouter(preferred="lm_studio", model="qwen1", timeout_seconds=90)
        lines, done = _captured()
        try:
            assert router.chat("prompt", "system") is None
        finally:
            done()
        failure = [l for l in lines if "LM Studio chat failed after" in l]
        assert failure and "qwen1" in failure[0] and "90s" in failure[0]
        assert "TimeoutError" in failure[0], (
            "the exception type is what tells a timeout from a refusal")
        usage = router.usage_report()
        assert usage["calls"] == 1 and usage["failed"] == 1
        assert "TimeoutError" in usage["last_error"]

    def test_a_dead_server_is_probed_on_a_leash_not_once_per_market(self, monkeypatch):
        """
        The retry exists (the operator may start LM Studio mid-run) but it must
        not print the same miss for every market in a scan.
        """
        _fake_lm_studio(monkeypatch, connected=False)
        router = LLMRouter(preferred="lm_studio", model="local-model")
        lines, done = _captured()
        try:
            for _ in range(25):
                router.chat("prompt", "system")
        finally:
            done()
        misses = [l for l in lines if "LLM Router detecting" in l]
        assert misses == [], (
            "the router re-detected inside the retry window: a dead server "
            "would print a line per market")
        assert router.usage_report()["failed"] == 25

    def test_the_usage_count_resets_per_cycle(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        _stub_openai(monkeypatch, json.dumps({"fair_value": 0.6}), "qwen2.5-14b-instruct")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        router.chat("prompt", "system")
        assert router.usage_report()["answered"] == 1
        router.reset_usage()
        assert router.usage_report() == {"calls": 0, "answered": 0, "failed": 0,
                                         "seconds": 0.0, "models": {},
                                         "last_model": "", "last_error": ""}


# ---------------------------------------------------------------------------
# 3. The market's own record says which model priced it
# ---------------------------------------------------------------------------

def _market():
    from src.ptai.markets.base import Market, MarketSource, Token
    return Market(id="m-1", source=MarketSource.POLYMARKET,
                  question="Will the bill pass?", outcomes=["YES", "NO"],
                  outcome_prices=[0.60, 0.40],
                  tokens=[Token(token_id="t1", outcome="YES", price=0.60)],
                  volume=100000.0, volume_24h=5000.0, liquidity=20000.0,
                  active=True, closed=False, raw={"venue": "polymarket"})


class TestTheMarketSRecordNamesTheModel:
    def test_a_model_answer_carries_the_model_id(self, monkeypatch):
        from src.ptai.agent.brain import Brain
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        _stub_openai(
            monkeypatch,
            json.dumps({"fair_value": 0.62, "confidence": 0.7, "side": "YES",
                        "basis": "research", "reasoning": "evidence"}),
            "qwen2.5-14b-instruct")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        brain = Brain(llm_router=router)
        result = brain.estimate_fair_value(_market())
        assert result.llm_model == "qwen2.5-14b-instruct"
        assert result.llm_provider == "LMStudioProvider"

    def test_the_per_market_line_names_the_model(self, monkeypatch):
        from src.ptai.agent.brain import Brain
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        _stub_openai(monkeypatch, json.dumps({"fair_value": 0.62, "confidence": 0.7}),
                     "qwen2.5-14b-instruct")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        lines, done = _captured()
        try:
            Brain(llm_router=router).estimate_fair_value(_market())
        finally:
            done()
        assert any("LLM lm_studio ['qwen2.5-14b-instruct']" in l for l in lines)
        assert any("Brain [LMStudioProvider/qwen2.5-14b-instruct]" in l for l in lines)

    def test_no_model_answer_is_a_heuristic_and_says_why(self, monkeypatch):
        from src.ptai.agent.brain import Brain
        _fake_lm_studio(monkeypatch, connected=False)
        from src.ptai.llm.provider import LLMRouter as _Router
        router = _Router(preferred="lm_studio", model="local-model",
                         lm_studio_host="http://localhost:4999")
        lines, done = _captured()
        try:
            result = Brain(llm_router=router).estimate_fair_value(_market())
        finally:
            done()
        assert result.llm_provider == "heuristic"
        assert result.llm_model == ""
        fallback = [l for l in lines if "Using fallback heuristic" in l]
        assert fallback, "the fallback was not logged at all"
        assert "4999" in fallback[0], (
            "the fallback line must say WHERE no model answered")
        assert "rule-based fallback" in fallback[0]

    def test_the_ensemble_component_row_names_the_model(self):
        """
        The console prints the component rows from the ensemble; "llm_reasoning"
        alone named the weight, never the model.
        """
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        from src.ptai.intelligence.forecaster import ModelForecast
        from src.ptai.markets.base import MarketSource
        forecaster = EnsembleForecaster()
        market = _market()
        forecast = forecaster.add_llm_forecast(market, {
            "fair_value": 0.62, "confidence": 0.7, "reasoning": "evidence",
            "llm_provider": "LMStudioProvider", "llm_model": "qwen2.5-14b-instruct"})
        assert forecast.model_id == "qwen2.5-14b-instruct"
        assert forecast.model_name == "llm_reasoning"
        result = forecaster.ensemble([forecast], market)
        row = [r for r in result.components if r["model"] == "llm_reasoning"][0]
        assert row["model_id"] == "qwen2.5-14b-instruct"

    def test_a_heuristic_component_is_still_named_as_one(self):
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        forecaster = EnsembleForecaster()
        market = _market()
        forecast = forecaster.add_llm_forecast(market, {
            "fair_value": 0.62, "confidence": 0.7, "reasoning": "rule of thumb",
            "llm_provider": "heuristic", "llm_model": ""})
        assert forecast.model_name != "llm_reasoning"
        assert "NOT AN LLM ANSWER" in forecast.reasoning


# ---------------------------------------------------------------------------
# 4. The cycle says which model it used, and why when it used none
# ---------------------------------------------------------------------------

def _agent_stub(router, screen: Optional[Dict[str, Any]] = None):
    from src.ptai.agent.v3_loop import TradingAgentV3

    class _Agent:
        llm_router = router
        _screen = screen if screen is not None else {
            "considered": 40, "shortlist": [{}] * 8, "screened_out": 32}
    _Agent._local_model_status = TradingAgentV3._local_model_status
    _Agent._local_model_line = TradingAgentV3._local_model_line
    return _Agent()


class TestTheCycleNamesTheModel:
    def test_a_working_cycle_names_the_model_and_counts_its_calls(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        _stub_openai(monkeypatch, json.dumps({"fair_value": 0.6}), "qwen2.5-14b-instruct")
        router = LLMRouter(preferred="lm_studio", model="local-model")
        router.chat("p", "s")
        router.chat("p", "s")
        line = _agent_stub(router)._local_model_line()
        assert line.startswith("Local model: qwen2.5-14b-instruct")
        assert "http://localhost:1234/v1" in line
        assert "2 of 2 call(s) answered" in line
        assert "qwen2.5-14b-instruct x2" in line
        assert "NOT USED" not in line
        assert "model time went to 8 of 40 screened market(s); 32 priced on their measured book alone" in line

    def test_a_cycle_with_no_shortlist_says_the_model_was_never_asked(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        router = LLMRouter(preferred="lm_studio", model="local-model")
        line = _agent_stub(router, {"considered": 0, "shortlist": [],
                                    "screened_out": 0})._local_model_line()
        assert "NOT USED THIS CYCLE" in line
        assert "no market reached the deep shortlist" in line

    def test_a_dead_server_cycle_says_not_used_and_where(self, monkeypatch):
        _fake_lm_studio(monkeypatch, connected=False)
        router = LLMRouter(preferred="lm_studio", model="local-model",
                           lm_studio_host="http://localhost:4999")
        status = _agent_stub(router)._local_model_status()
        assert status["used"] is False
        assert status["available"] is False
        assert "4999" in status["describe"]
        assert "no local model server answered" in status["not_used_reason"]
        assert "NOT USED THIS CYCLE" in _agent_stub(router)._local_model_line()

    def test_a_cycle_whose_every_call_failed_names_the_failure(self, monkeypatch):
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])
        router = LLMRouter(preferred="lm_studio", model="local-model")
        router.note_call(model="qwen2.5-14b-instruct", ok=False,
                         error="APITimeoutError: Request timed out.")
        status = _agent_stub(router)._local_model_status()
        assert status["calls"] == 1 and status["failed"] == 1
        assert "every model call failed" in status["not_used_reason"]
        assert "APITimeoutError" in status["not_used_reason"]

    def test_the_missing_router_is_not_a_crash(self):
        status = _agent_stub(None)._local_model_status()
        assert status["available"] is False
        assert "without a model router" in status["not_used_reason"]


# ---------------------------------------------------------------------------
# 5. The cycle is wired to say it: reset, line, report, page
# ---------------------------------------------------------------------------

class TestTheCycleIsWiredToSayIt:
    def test_the_cycle_resets_the_count_and_prints_the_line(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "self.llm_router.reset_usage()" in source
        assert "self._local_model_line()" in source

    def test_the_cycle_report_carries_the_model(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert '"local_model": self._local_model_status()' in source

    def test_a_cycle_that_found_nothing_still_reports_the_model(self):
        """
        The no_markets path returns early. It used to return before anything
        named the model, so a run of quiet cycles - which is what a machine with
        no live venue produces - had no model line anywhere.
        """
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        quiet = source[:source.index("no_markets_result = {")]
        assert "self._local_model_line()" in quiet, (
            "the quiet-cycle path does not print the model line")
        assert '"local_model": self._local_model_status()' in source

    def test_the_console_evidence_carries_the_model_too(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._store_forecast_evidence)
        assert '"local_model": self._local_model_status()' in source

    def test_the_page_trace_prints_the_model_beside_the_component(self):
        import inspect
        from src.ptai.ui import console as console_module
        source = inspect.getsource(console_module)
        assert "c.model_id" in source

    def test_the_page_shows_the_same_model_line_as_the_log(self):
        import inspect
        from src.ptai.ui import console as console_module
        source = inspect.getsource(console_module)
        assert "const lm = body.local_model || {};" in source
        assert "NOT USED</b>" in source
        assert "$('forecast').innerHTML = modelLine + screenLine + baseLine" in source

    def test_the_startup_panel_names_the_model_not_only_the_provider(self):
        import inspect
        from src.ptai import cli
        source = inspect.getsource(cli)
        assert "probe_local_model" in source, (
            "the CLI panel still prints only the provider preference and host")


# ---------------------------------------------------------------------------
# 6. A real cycle, through run_cycle, says it
# ---------------------------------------------------------------------------

class TestARealCycleSaysIt:
    def test_a_quiet_cycle_still_names_the_model_and_why_it_did_nothing(
            self, tmp_path, monkeypatch):
        """
        The operator's machine has no live venue yet, so most cycles discover
        nothing and take the early-return path in `run_cycle`. That path used to
        say nothing about the model at all, which is why a whole run of cycles
        could be searched for the model id without finding it.
        """
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "cycle.db"))
        monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path))
        _fake_lm_studio(monkeypatch, ["qwen2.5-14b-instruct"])

        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.llm.provider import LLMRouter

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        agent.llm_router = LLMRouter(preferred="lm_studio", model="local-model")

        import asyncio
        lines, done = _captured()
        try:
            result = asyncio.run(agent.run_cycle(target_per_venue=6, max_trades=0))
        finally:
            done()

        model_lines = [l for l in lines if l.startswith("Local model: ")]
        assert model_lines, (
            "the cycle never printed which model it had")
        assert "qwen2.5-14b-instruct" in model_lines[-1]
        assert "NOT USED THIS CYCLE" in model_lines[-1], (
            "a cycle that gave the model no work must say so, not look quiet")
        assert result["local_model"]["model"] == "qwen2.5-14b-instruct"
        assert result["local_model"]["used"] is False
        assert "model time went to" in model_lines[-1]


# ---------------------------------------------------------------------------
# 7. The bug these tests found: a CLI import killed the page's log
# ---------------------------------------------------------------------------

class TestTheLogRingSurvivesSomethingCallingLoggerRemove:
    """
    `src/ptai/cli.py` calls `logger.remove()` at import (a CLI process wants a
    clean slate). That takes every loguru handler with it - including the log
    ring the console page reads through - and the ring never came back, so the
    page showed an empty log with no explanation. This test file found it by
    importing `cli` for a source assertion and then watching an unrelated V50
    test fail; the fix is in `live_log`, not in the test.
    """

    def test_a_read_re_attaches_the_ring_and_keeps_its_lines(self):
        from loguru import logger as loguru_logger

        from src.ptai.ui import live_log

        live_log.install()
        loguru_logger.info("V53TEST before the wipe")
        loguru_logger.remove()                     # what `import ptai.cli` does
        loguru_logger.info("V53TEST while detached")

        body = live_log.tail(limit=200)            # the page asks for the log
        texts = " ".join(line["text"] for line in body["lines"])
        assert "before the wipe" in texts, (
            "re-attaching cleared the buffer, so the page lost the log it was "
            "already showing")
        assert body["next"] >= 0

        loguru_logger.info("V53TEST after the read")
        fresh = live_log.tail(after=body["next"], limit=200)
        assert any("after the read" in line["text"] for line in fresh["lines"]), (
            "the ring did not re-attach, so nothing new reaches the page")
