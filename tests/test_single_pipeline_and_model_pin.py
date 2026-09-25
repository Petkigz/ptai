"""
One product, one pipeline, one model.

Three defects, all visible from the operator's own log:

1. The dashboard's "Run Cycle Now" launched the LEGACY TradingAgent inside
   the dashboard process - a second pipeline with its own scanner and its
   own fair-value path, producing a cycle that was NOT what the agent runs.

2. Model detection judged the whole DOWNLOADED list, so a fast session read
   as "R1 - 8 min per market" the moment deepseek-r1 was loaded in LM Studio
   (and the legacy loop then silently deep-analyzed 30 markets with whatever
   model happened to be first in the list).

3. The web-search sentiment fallback wrote `SentimentResult.sentiment_summary`
   - a field that does not exist - so it died with AttributeError on its
   first market ("Web search fallback failed: 'SentimentResult' object has
   no attribute 'sentiment_summary'").

Plus the PC runner: no venv. The agent's memory lives in the data folder, not in a
Python environment, and creating one on every start made the machine look
like it had amnesia.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Model detection judges the model in use, never the downloaded list
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, model):
        self.model = model


class FakeRouter:
    def __init__(self, model="local-model", detected=(), provider=None):
        self._model = model
        self._detected = list(detected)
        self._provider = provider

    def get_provider(self):
        return self._provider

    def get_detected_models(self):
        return self._detected


class FakeBrain:
    def __init__(self, router):
        self.llm_router = router


def _agent_with(brain):
    """A TradingAgent with only the attribute the helpers touch."""
    from src.ptai.agent.loop import TradingAgent
    agent = TradingAgent.__new__(TradingAgent)
    agent.brain = brain
    return agent


class TestActiveModel:
    def test_a_pinned_provider_model_is_the_active_model(self):
        # LM_STUDIO_MODEL set in .env -> the provider carries it
        brain = FakeBrain(FakeRouter(
            detected=["prism-ml/bonsai-27b", "deepseek-r1-distill-qwen-32b"],
            provider=FakeProvider("qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp")))
        assert _agent_with(brain)._active_llm_model() == \
            "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"

    def test_an_unpinned_router_falls_back_to_the_first_loaded_model(self):
        # the "local-model" placeholder -> chat() auto-detects models[0]
        brain = FakeBrain(FakeRouter(
            detected=["prism-ml/bonsai-27b", "qwen3.8-27b-uncensored"],
            provider=FakeProvider("local-model")))
        assert _agent_with(brain)._active_llm_model() == "prism-ml/bonsai-27b"

    def test_an_empty_router_returns_nothing_rather_than_guessing(self):
        brain = FakeBrain(FakeRouter(provider=None))
        assert _agent_with(brain)._active_llm_model() == ""

    def test_a_missing_router_does_not_raise(self):
        from src.ptai.agent.loop import TradingAgent
        agent = TradingAgent.__new__(TradingAgent)
        agent.brain = object()
        assert agent._active_llm_model() == ""


class TestIsR1Model:
    @pytest.mark.parametrize("model_id,expected", [
        ("deepseek-r1-distill-qwen-32b", True),
        ("deepseek-r1", True),
        ("deepseek-r1:32b", True),
        ("qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp", False),
        ("qwen/qwen3-32b", False),
        ("prism-ml/bonsai-27b", False),
        ("qwen2.5-coder-7b-instruct", False),
        ("text-embedding-nomic-embed-text-v1.5", False),
        ("", False),
    ])
    def test_segment_matching(self, model_id, expected):
        from src.ptai.agent.loop import TradingAgent
        assert TradingAgent._is_r1_model(model_id) is expected

    def test_the_operators_loaded_list_is_not_an_r1_server(self):
        """
        The exact scenario from the operator's PC: 16 models loaded,
        deepseek-r1 among the DOWNLOADED ones, qwen3.8-27b in use.
        """
        from src.ptai.agent.loop import TradingAgent
        in_use = "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"
        assert TradingAgent._is_r1_model(in_use) is False


# ---------------------------------------------------------------------------
# 2. The dashboard's "Run Cycle Now" runs the V3 pipeline
# ---------------------------------------------------------------------------

class TestRunOnceUsesTheRealPipeline:
    def test_run_once_constructs_trading_agent_v3(self, monkeypatch):
        import src.ptai.agent.v3_loop as v3_loop
        import src.ptai.dashboard as dashboard
        from fastapi.testclient import TestClient

        seen = {}

        class FakeV3:
            def __init__(self, country_code="UG"):
                seen["constructed"] = country_code

            async def run_cycle(self):
                seen["cycled"] = True
                return {"status": "ok"}

        monkeypatch.setattr(v3_loop, "TradingAgentV3", FakeV3)
        resp = TestClient(dashboard.app).post("/api/run-once")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "triggered"
        assert "V3" in body["message"]
        assert seen.get("constructed") == "UG"

    def test_run_once_does_not_import_the_legacy_agent(self):
        src = (REPO / "src" / "ptai" / "dashboard.py").read_text(encoding="utf-8")
        start = src.index("async def api_run_once")
        block = src[start:start + 1500]
        assert "from .agent.loop import TradingAgent" not in block, \
            "Run Cycle Now must run the V3 pipeline, not a second legacy one"


# ---------------------------------------------------------------------------
# 3. The web-search sentiment fallback writes a field that exists
# ---------------------------------------------------------------------------

class TestSentimentFallbackWritesARealField:
    def test_no_write_to_the_nonexistent_sentiment_summary(self):
        src = (REPO / "src" / "ptai" / "agent" / "loop.py").read_text(encoding="utf-8")
        assert ".sentiment_summary +=" not in src

    def test_the_fallback_uses_the_summary_field(self):
        src = (REPO / "src" / "ptai" / "agent" / "loop.py").read_text(encoding="utf-8")
        assert "summaries" not in src  # guard against a typo'd rename target
        assert ".summary = (" in src

    def test_sentiment_result_has_no_sentiment_summary_field(self):
        from src.ptai.sentiment import SentimentResult
        result = SentimentResult(score=0, bullish_pct=0.33, bearish_pct=0.33,
                                 neutral_pct=0.34, summary="x", key_phrases=[],
                                 tweet_count=0, confidence=0.2)
        assert not hasattr(result, "sentiment_summary")
        assert hasattr(result, "summary")


# ---------------------------------------------------------------------------
# 4. The LLM tab can pin the model (no .env editing for a normal user)
# ---------------------------------------------------------------------------

class TestModelPicker:
    def test_the_llm_tab_has_a_model_picker(self):
        src = (REPO / "src" / "ptai" / "dashboard.py").read_text(encoding="utf-8")
        assert 'id="llm-model-select"' in src
        assert "function pinLLMModel" in src

    def test_pinning_writes_lm_studio_model(self):
        src = (REPO / "src" / "ptai" / "dashboard.py").read_text(encoding="utf-8")
        start = src.index("async function pinLLMModel")
        block = src[start:start + 1200]
        assert "LM_STUDIO_MODEL" in block
        assert "/api/config" in block

    def test_lm_studio_model_is_an_allowed_config_key(self):
        src = (REPO / "src" / "ptai" / "dashboard.py").read_text(encoding="utf-8")
        start = src.index("allowed_keys = [")
        block = src[start:start + 700]
        assert "LM_STUDIO_MODEL" in block


# ---------------------------------------------------------------------------
# 5. The PC runner: no venv, and the cmd.exe traps stay fixed
# ---------------------------------------------------------------------------

class TestTheRunnerRunsOnTheMachine:
    @pytest.fixture()
    def bat(self):
        return (REPO / "run_ptai.bat").read_bytes()

    def test_no_venv_is_created(self, bat):
        text = bat.decode("utf-8", errors="replace")
        assert ".venv" not in text.lower()
        assert "venv" not in text.lower() or "no venv" in text.lower()

    def test_it_uses_the_system_python(self, bat):
        text = bat.decode("utf-8", errors="replace")
        assert "where py " in text or "where py>" in text
        assert "where python " in text

    def test_it_still_waits_for_the_dashboard_port(self, bat):
        text = bat.decode("utf-8", errors="replace")
        assert "socket" in text and "connect" in text
        assert ":port_up" in text

    def test_line_endings_are_crlf(self, bat):
        assert b"\r\n" in bat, "cmd.exe needs CRLF; LF-only made the window flash-close"
        assert re.search(rb"(?<!\r)\n", bat) is None

    def test_no_unquoted_parens_in_echo_text_inside_blocks(self, bat):
        """
        The V34 abort: a `)` inside echo text closed the `if (` block early
        and cmd.exe died with ". was unexpected at this time." before running
        anything.
        """
        lines = bat.decode("utf-8", errors="replace").split("\r\n")
        depth, offenders = 0, []
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("echo") and depth > 0 and re.search(r"[()]", stripped):
                offenders.append((lineno, stripped))
            if not low.startswith(("echo", "rem", "::")) and stripped:
                for ch in stripped:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
        assert offenders == []
