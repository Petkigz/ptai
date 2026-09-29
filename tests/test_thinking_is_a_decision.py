"""
WHY THINKING WAS OFF, AND WHO DECIDES NOW.

The operator's 2026-09-29 question, verbatim: "it seems thinking is disabled for
the model i want to know why . logs;"

The answer was in the code, in two places, and neither was a bug in the loaded
model:

  * `brain.py`'s system prompt said, unconditionally,
    "No chain-of-thought, no <think> tag, just direct JSON" - a speed decision
    made after a log where one market took ~9 minutes;
  * nothing in the request ever sent a thinking kwarg, so `enable_thinking`
    (the LM Studio / Qwen3 hybrid-thinking chat-template flag) was never
    asked for at all: whatever the model's own default was, applied.

Both are now controlled by ONE setting, `PTAI_LLM_THINKING` (or the console's
Brain panel), and these tests pin the four facts that make it honest:

  1. the request carries the kwarg when thinking is ON and does not when it is
     OFF (an absent kwarg is a fact, not an accident);
  2. the prompt follows the same setting, so prompt and request cannot disagree;
  3. the cycle's model line says which way it is set, and what turning it on
     does to a market's call, from the same string the console renders;
  4. the model line no longer borrows the V55 "no model router wired to the
     forecast engine" sentence for a market the RESOLUTION gate refused - that
     was the 16:53 lie: the gate refused before a forecast existed.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

from src.ptai.agent.brain import Brain
from src.ptai.agent.v3_loop import _thinking_note, _why_the_model_was_not_used
from src.ptai.config import LLMConfig
from src.ptai.llm.provider import LLMRouter, LMStudioProvider
from src.ptai.markets.base import Market, MarketSource


MODEL = "qwen/qwen3-14b"


# ---------------------------------------------------------------------------
# the harness: a stubbed OpenAI client that KEEPS what it was sent
# ---------------------------------------------------------------------------

SENT: List[Dict[str, Any]] = []


def _stub_openai(monkeypatch, model: str = MODEL) -> List[Dict[str, Any]]:
    """An `openai` module whose every request is recorded, verbatim."""
    SENT.clear()
    module = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            SENT.append(dict(kwargs))
            return SimpleNamespace(
                model=model,
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"fair_value": 0.61, "confidence": 0.7})))],
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    module.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)
    return SENT


def _market() -> Market:
    return Market(
        id="will-btc-close-higher",
        source=MarketSource.POLYMARKET,
        question="Will BTC close higher today?",
        outcome_prices=[0.55, 0.45],
        outcomes=["YES", "NO"],
    )


# ---------------------------------------------------------------------------
# 1. the request says it, or does not
# ---------------------------------------------------------------------------

class TestTheRequestCarriesTheDecision:
    def test_thinking_off_sends_no_enable_thinking_kwarg(self, monkeypatch):
        sent = _stub_openai(monkeypatch)
        provider = LMStudioProvider(model=MODEL, thinking=False)
        response = provider.chat("market question", "system")
        assert response is not None
        assert len(sent) == 1
        assert sent[0].get("extra_body") is None, (
            "thinking OFF must not ask the server for a reasoning pass, and must "
            "not send a half-built body either")
        assert "chat_template_kwargs" not in json.dumps(sent[0])

    def test_thinking_on_sends_the_chat_template_kwarg(self, monkeypatch):
        sent = _stub_openai(monkeypatch)
        provider = LMStudioProvider(model=MODEL, thinking=True)
        assert provider.chat("market question", "system") is not None
        assert sent[0]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": True}}, (
            "LM Studio/Qwen3 hybrid thinking is enabled per request through the "
            "chat template; without this kwarg the model decides by itself, "
            "which is what the operator's log showed")

    def test_the_router_gives_every_provider_the_same_answer(self, monkeypatch):
        monkeypatch.setattr(LMStudioProvider, "is_available", lambda self: True)
        monkeypatch.setattr(LMStudioProvider, "list_models", lambda self: [MODEL])
        router = LLMRouter(preferred="lm_studio", model=MODEL, thinking=True)
        assert router.thinking is True
        assert router.provider.thinking is True, (
            "the setting has to reach the object that builds the request")

    def test_a_router_built_without_the_setting_does_not_think(self, monkeypatch):
        monkeypatch.setattr(LMStudioProvider, "is_available", lambda self: True)
        monkeypatch.setattr(LMStudioProvider, "list_models", lambda self: [MODEL])
        router = LLMRouter(preferred="lm_studio", model=MODEL)
        assert router.thinking is False
        assert router.provider.thinking is False


# ---------------------------------------------------------------------------
# 2. the prompt follows the same setting
# ---------------------------------------------------------------------------

class TestThePromptFollowsTheSetting:
    def test_the_off_prompt_asks_for_no_chain_of_thought(self):
        brain = Brain(llm_config=LLMConfig(model=MODEL, thinking=False))
        system, user = brain._build_prompt(_market(), None)
        assert "No chain-of-thought" in system
        assert "NO <think> reasoning tags" in user, (
            "both turns say the same thing when thinking is off")

    def test_the_on_prompt_asks_for_reasoning_first(self):
        brain = Brain(llm_config=LLMConfig(model=MODEL, thinking=True))
        system, user = brain._build_prompt(_market(), None)
        assert "Reason through the market" in system
        assert "No chain-of-thought" not in system, (
            "the prompt and the request must not disagree about whether the "
            "model may think")
        assert "NO <think> reasoning tags" not in user, (
            "the user turn used to forbid the reasoning the system turn asked "
            "for, in the same request - the model was told both things")
        assert "Think it through" in user

    def test_both_turns_agree_with_the_switch(self):
        for thinking in (True, False):
            system, user = Brain(
                llm_config=LLMConfig(model=MODEL,
                                     thinking=thinking))._build_prompt(_market(),
                                                                       None)
            asked = "Reason through the market" in system
            allowed = "NO <think> reasoning tags" not in user
            assert asked is thinking and allowed is thinking, (
                f"thinking={thinking}: system asked={asked} user allowed="
                f"{allowed} - the two turns disagree")

    def test_the_brain_reads_the_setting_from_its_config(self):
        assert Brain(llm_config=LLMConfig(thinking=True)).llm_thinking() is True
        assert Brain(llm_config=LLMConfig(thinking=False)).llm_thinking() is False


# ---------------------------------------------------------------------------
# 3. the setting is visible, in one sentence, from one place
# ---------------------------------------------------------------------------

class TestTheSettingIsSaidOutLoud:
    def test_the_off_sentence_names_the_setting_and_where_to_change_it(self):
        note = _thinking_note(False)
        assert note.startswith("thinking: OFF")
        assert "PTAI_LLM_THINKING=0" in note
        assert "Setup > Brain" in note

    def test_the_on_sentence_says_what_it_costs(self):
        note = _thinking_note(True)
        assert note.startswith("thinking: ON")
        assert "reason" in note
        assert "Setup > Brain" in note, (
            "the way back has to be in the same sentence as the way in")
        assert "call limit" in note, (
            "the operator is choosing longer calls; the line has to say that a "
            "call over the limit gives no forecast for that market")

    def test_the_cycle_model_line_carries_the_sentence(self, monkeypatch):
        from src.ptai.agent.v3_loop import TradingAgentV3

        monkeypatch.setattr(LMStudioProvider, "is_available", lambda self: True)
        monkeypatch.setattr(LMStudioProvider, "list_models", lambda self: [MODEL])
        router = LLMRouter(preferred="lm_studio", model=MODEL, thinking=True)

        class _Ensemble:
            llm_router = router

            def llm_accounting(self):
                return {"asked": 0, "answered": 0, "problems": [],
                        "forecasts_run": 0}

        class _Agent:
            llm_router = router
            ensemble_forecaster = _Ensemble()
            _cycle_scan_counts: Dict[str, Any] = {}
            _cycle_context_calls = 0
            _cycle_deep_context_calls = 0
            _screen = {"considered": 3, "shortlist": [{}], "screened_out": 2}

        _Agent._local_model_status = TradingAgentV3._local_model_status
        _Agent._local_model_line = TradingAgentV3._local_model_line
        assert "thinking: ON" in _Agent()._local_model_line()

    def test_the_setting_defaults_to_the_configured_value(self, monkeypatch):
        from src.ptai.config import Settings
        assert Settings().llm_thinking is False, (
            "PTAI forbids chain-of-thought by default because a reasoning pass "
            "is what made one market take ~9 minutes in the operator's log")
        assert Settings(PTAI_LLM_THINKING="true").to_llm_config().thinking is True

    def test_the_console_switch_writes_the_setting(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "c.db"))
        console = importlib.import_module("src.ptai.ui.console")
        written: Dict[str, str] = {}
        import src.ptai.dashboard as dashboard
        monkeypatch.setattr(dashboard, "write_env_file",
                            lambda updates: written.update(updates))
        from fastapi.testclient import TestClient
        client = TestClient(console.app)

        ok = client.post("/api/console/brain", json={"thinking": True})
        assert ok.status_code == 200
        assert written == {"PTAI_LLM_THINKING": "1"}
        assert "next start" in ok.json()["note"]

        off = client.post("/api/console/brain", json={"thinking": False})
        assert off.status_code == 200
        assert written == {"PTAI_LLM_THINKING": "0"}

    def test_the_brain_panel_reports_the_setting(self, monkeypatch):
        from src.ptai.ui import console as console_module
        monkeypatch.setattr(console_module, "_BRAIN_CACHE",
                            {"at": 0.0, "value": None})
        import src.ptai.dashboard as dashboard
        monkeypatch.setattr(dashboard, "read_env_file",
                            lambda: {"PTAI_LLM_THINKING": "1",
                                     "LM_STUDIO_MODEL": MODEL})
        monkeypatch.setattr(dashboard, "check_lm_studio",
                            lambda *a, **k: {"connected": True, "models": [MODEL],
                                             "active_model": MODEL})
        status = console_module._brain_status(force=True)
        assert status["thinking"] is True


# ---------------------------------------------------------------------------
# 4. a refusal is named for the gate that made it
# ---------------------------------------------------------------------------

def _status(**over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "available": True, "describe": f"{MODEL} (LMStudioProvider)",
        "model": MODEL, "calls": 0, "failed": 0, "last_error": "",
        "answered_by_model": 0, "asked": 0, "deep_shortlist": 1,
        "deep_priced": 0, "forecasts_run": 0, "considered": 200,
        "ensemble_has_router": False, "deep_priced_markets": [],
    }
    base.update(over)
    return base


class TestARefusalIsNamedForItsGate:
    def test_a_market_the_gate_refused_before_a_forecast_is_not_a_wiring_gap(self):
        reason = _why_the_model_was_not_used(
            _status(deep_priced=1, forecasts_run=0), 0, [])
        assert "refused before a forecast was built" in reason
        assert "no model router wired" not in reason, (
            "the 16:53 log blamed the wiring for a market the resolution gate "
            "refused before the model was ever reachable; the router was wired "
            "and simply never reached")

    def test_a_shortlist_that_never_reached_pricing_does_not_claim_model_time(self):
        reason = _why_the_model_was_not_used(
            _status(deep_shortlist=8, deep_priced=0), 0, [])
        assert "none of them reached the pricing stage" in reason

    def test_an_engine_with_a_router_says_the_markets_were_not_deep(self):
        reason = _why_the_model_was_not_used(
            _status(deep_shortlist=3, deep_priced=3, forecasts_run=3,
                    ensemble_has_router=True), 0, [])
        assert "outside the deep shortlist" in reason
        assert "no model router wired" not in reason

    def test_an_engine_without_a_router_still_says_so(self):
        reason = _why_the_model_was_not_used(
            _status(deep_shortlist=3, deep_priced=3, forecasts_run=3,
                    ensemble_has_router=False), 0, [])
        assert "no model router wired to the forecast engine" in reason


# ---------------------------------------------------------------------------
# 5. a save the filesystem refused is not a lost record
# ---------------------------------------------------------------------------

class TestASaveTheFilesystemRefused:
    """`WARNING qualification:_save:427 - Qualification save failed: [WinError 5]
    Access is denied: 'data\\venue_qualification.json.tmp' ->
    'data\\venue_qualification.json'` - the operator's 17:27 cycle. Windows
    refuses `os.replace` while another process (the console, reading the file
    for its snapshot) has the destination open. The reader holds it for
    milliseconds, so a retry clears it - and if it never clears, the record is
    still written and the log says which trade-off was made."""

    def _engine(self, tmp_path):
        from src.ptai.venues.qualification import VenueQualificationEngine
        return VenueQualificationEngine(data_dir=str(tmp_path))

    def test_a_transient_lock_is_retried_and_the_record_still_lands(self, tmp_path,
                                                                   monkeypatch):
        import os as _os
        from src.ptai.venues import qualification as q

        engine = self._engine(tmp_path)
        real_replace = q.os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(13, "Access is denied")
            return real_replace(src, dst)

        monkeypatch.setattr(q.os, "replace", flaky)
        engine._save()
        assert calls["n"] == 3, "the save retries instead of giving up"
        assert (tmp_path / "venue_qualification.json").exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_a_lock_that_never_clears_still_writes_the_record_and_says_so(
            self, tmp_path, monkeypatch):
        from src.ptai.venues import qualification as q

        engine = self._engine(tmp_path)
        monkeypatch.setattr(q.os, "replace",
                            lambda src, dst: (_ for _ in ()).throw(
                                PermissionError(13, "Access is denied")))
        monkeypatch.setattr(q.time, "sleep", lambda _s: None)
        lines: List[str] = []
        from loguru import logger as _logger
        sink = _logger.add(lambda m: lines.append(m), level="WARNING")
        try:
            engine._save()
        finally:
            _logger.remove(sink)
        text = (tmp_path / "venue_qualification.json").read_text()
        assert text.strip().startswith("{"), (
            "the record must still be written in place, or the cycle's "
            "qualification evidence is lost")
        assert any("writing it in place" in line for line in lines), (
            "and the log has to say the trade-off, not swallow it")

    def test_the_temp_name_is_per_process(self, tmp_path):
        from src.ptai.venues import qualification as q
        engine = self._engine(tmp_path)
        seen: List[str] = []
        real_replace = q.os.replace
        import os as _os

        def watch(src, dst):
            seen.append(str(src))
            return real_replace(src, dst)

        q.os.replace = watch
        try:
            engine._save()
        finally:
            q.os.replace = real_replace
        assert seen and str(_os.getpid()) in seen[0], (
            "one fixed .tmp is shared by every writer; a console save and an "
            "agent save would move each other's half-written file into place")


# ---------------------------------------------------------------------------
# 6. the noise that buried the answer, and the docs that carry it
# ---------------------------------------------------------------------------

class TestTheNoiseAroundTheAnswer:
    def test_the_legacy_scan_warning_is_not_a_warning_any_more(self):
        import inspect
        from src.ptai.markets import scanner
        source = inspect.getsource(scanner.MarketScanner.scan)
        assert "NOT recommended" not in source, (
            "the adapter's own discovery call is the correct path; every cycle "
            "warned about it")
        assert "logger.debug" in source.split("use_registry=False")[1][:600]

    def test_the_console_says_why_the_agent_is_being_built(self):
        import inspect
        from src.ptai.ui import console
        source = inspect.getsource(console.api_run_cycle)
        before = source.split("_start_agent_in_process")[0]
        assert "The operator asked for a round" in before, (
            "PTAI V3 initialized after a Stop looked like the system restarting "
            "itself; the operator's action has to be logged first")

    def test_the_env_example_documents_the_switch(self):
        from pathlib import Path
        text = Path(".env.example").read_text(encoding="utf-8")
        assert "PTAI_LLM_THINKING" in text
        assert "enable_thinking" in text

    def test_the_readme_explains_why_it_was_off(self):
        from pathlib import Path
        readme = Path("README.md").read_text(encoding="utf-8")
        assert "## Thinking Is A Switch You Own" in readme
        assert "No chain-of-thought" in readme
        assert "enable_thinking" in readme
        assert "PTAI_LLM_THINKING" in readme


# ---------------------------------------------------------------------------
# 7. is OFF weaker, and does ON break the answer?
# ---------------------------------------------------------------------------

class TestTheAnswerSurvivesTheReasoning:
    """
    The operator's question, one step on: "does it have the same capability when
    thinking is off or are there errors".

    OFF changes nothing about the pipeline - same prompt shape, same schema, same
    parser, same timeout, same accounting - and every run in the suites and the
    live acceptance goes through it. ON is the one that can break an answer, in
    two ways this class pins shut: the budget (a reasoning pass is paid for out
    of `max_tokens`, so the answer must still fit) and the parser (a model that
    reasons first must still have its JSON read).
    """

    def test_the_budget_follows_the_switch(self, monkeypatch):
        off = Brain(llm_config=LLMConfig(model=MODEL, thinking=False))
        assert off.llm_router.max_tokens == LLMConfig(model=MODEL).max_tokens, (
            "thinking OFF must not change the call the agent already makes")
        on = Brain(llm_config=LLMConfig(model=MODEL, thinking=True))
        assert on.llm_router.max_tokens >= 2048, (
            "a reasoning pass plus the JSON has to fit in one call, or the "
            "model is cut off mid-thought and no answer parses")

    def test_a_reasoning_answer_still_parses(self):
        from src.ptai.llm.provider import extract_json_object
        answer = '{"fair_value": 0.61, "confidence": 0.7}'
        assert extract_json_object(answer)["fair_value"] == 0.61
        assert extract_json_object(
            f"<think>the market looks cheap because...</think>{answer}"
        )["fair_value"] == 0.61, "a closed thinking block must not hide the JSON"
        assert extract_json_object(
            f"<thinking>step one {{not json}}</thinking>\n{answer}"
        )["fair_value"] == 0.61
        assert extract_json_object(
            f"Let me think about this. The price is 0.55.\n{answer}\nHope that helps."
        )["fair_value"] == 0.61, "prose before and after the object is normal"
        assert extract_json_object(
            f"```json\n{answer}\n```"
        )["fair_value"] == 0.61, "a fenced block is not a reason to give up"
        assert extract_json_object(
            '{"reasoning": "a } brace in a string", "fair_value": 0.4}'
        )["fair_value"] == 0.4

    def test_a_truncated_thought_is_not_read_as_an_answer(self):
        from src.ptai.llm.provider import extract_json_object
        assert extract_json_object(
            "<think>the home side has won four of its last five, and the "
            "market prices them at 0.62, which looks") is None, (
            "no JSON means no answer - the caller must fall back and SAY so "
            "rather than parse half a thought")

    def test_the_brain_reads_a_wrapped_answer_instead_of_giving_up(self):
        from src.ptai.llm.provider import LLMResponse

        class _Router:
            active_model = MODEL

            def chat(self, prompt, system=""):
                return LLMResponse(
                    content=("<think>weighing the base rate</think>"
                             '{"fair_value": 0.58, "confidence": 0.66, '
                             '"side": "YES", "reasoning": "reasons first"}'),
                    model=MODEL, provider="lm_studio", parsed_json=None)

            def usage_report(self):
                return {"last_model": MODEL, "last_error": ""}

        brain = Brain(llm_router=_Router())
        parsed = brain._call_llm("system", "user")
        assert parsed and parsed["fair_value"] == 0.58, (
            "the V55 extractor only understood clean JSON or a closed </think> "
            "tag directly before it, so a wrapped answer was logged as "
            "'no usable JSON' and priced by the heuristic instead")

    def test_off_and_on_read_the_same_fields(self, monkeypatch):
        sent = _stub_openai(monkeypatch)
        off = LMStudioProvider(model=MODEL, thinking=False)
        on = LMStudioProvider(model=MODEL, thinking=True)
        assert off.chat("q", "s") is not None
        assert on.chat("q", "s") is not None
        assert sent[0]["messages"] == sent[1]["messages"], (
            "the prompt is the same request in both modes; only the reasoning "
            "instruction and the server-side kwarg differ")
        assert sent[0]["max_tokens"] == sent[1]["max_tokens"]
