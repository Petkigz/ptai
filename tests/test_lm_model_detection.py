"""
The LLM status must judge the model the agent will ACTUALLY call.

The operator's real complaint: the agent is serving
qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp (fast), deepseek-r1
is merely DOWNLOADED in the same LM Studio, and the panel reported
"R1 model detected - 7-8 MIN per market SLOW" - a warning about a
model the agent never touches.

The rule must match llm/provider.py exactly:

  1. LM_STUDIO_MODEL from .env, when it names a loaded model,
  2. otherwise the FIRST loaded model - the provider auto-detects
     models[0] when the env value is the "local-model" placeholder.

Judging speed from the whole downloaded list is how a fast session
reads as R1 SLOW.
"""

from __future__ import annotations

import pytest

import src.ptai.dashboard as dashboard

# The operator's actual model list (reduced): the fast model that is
# actually serving comes first, the downloaded deepseek-r1 sits in it.
MODELS = [
    "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp",
    "qwen/qwen3-32b",
    "qwen/qwen3.8-27b",
    "deepseek-r1-distill-qwen-32b",
    "prism-ml/bonsai-27b",
    "qwen/qwen3-14b",
    "text-embedding-nomic-embed-text-v1.5",
    "qwen3-coder-30b-a3b-instruct",
]


class _FakeResponse:
    def __init__(self, model_ids):
        self.status_code = 200
        self._ids = model_ids

    def json(self):
        return {"data": [{"id": m} for m in self._ids]}


def _patch_models(monkeypatch, model_ids):
    monkeypatch.setattr(dashboard.requests, "get",
                        lambda *a, **k: _FakeResponse(model_ids))


class TestActiveModelSelection:
    def test_placeholder_env_falls_back_to_first_loaded(self):
        # "local-model" is a placeholder: the agent (llm/provider.py)
        # auto-detects the first loaded model, so the dashboard must
        # judge that one.
        assert dashboard._active_lm_model(MODELS, "local-model") == MODELS[0]

    def test_configured_loaded_model_wins(self):
        assert dashboard._active_lm_model(MODELS, "qwen/qwen3-32b") == "qwen/qwen3-32b"

    def test_configured_unknown_model_falls_back(self):
        # The agent would fail against it - the dashboard still judges
        # the first loaded model and flags model_not_loaded separately.
        assert dashboard._active_lm_model(MODELS, "some-model-not-loaded") == MODELS[0]

    def test_no_models(self):
        assert dashboard._active_lm_model([], "local-model") is None


class TestSlowReasoningDetection:
    @pytest.mark.parametrize("name,slow", [
        ("deepseek-r1-distill-qwen-32b", True),
        ("deepseek-r1", True),
        ("qwen3-r1", True),
        ("qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp", False),
        ("qwen/qwen3-32b", False),
        ("qwen/qwen3.8-27b", False),
        ("prism-ml/bonsai-27b", False),
        ("qwen2.5-3b-instruct", False),
        ("qwen3.5-9b", False),
    ])
    def test_detection(self, name, slow):
        assert dashboard._is_slow_reasoning_model(name) is slow

    def test_none_and_empty(self):
        assert dashboard._is_slow_reasoning_model(None) is False
        assert dashboard._is_slow_reasoning_model("") is False


class TestStatusDetection:
    def test_downloaded_r1_does_not_flag_a_fast_active_model(self, monkeypatch):
        # The operator's exact scenario: fast model serving first,
        # deepseek-r1 merely in the list.
        _patch_models(monkeypatch, MODELS)
        result = dashboard.check_lm_studio("http://localhost:1234",
                                           configured_model="local-model")
        assert result["connected"] is True
        assert result["active_model"] == \
            "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"
        assert result["is_r1"] is False
        assert result["model_not_loaded"] is False
        assert result["recommended"] == result["active_model"]

    def test_an_actually_configured_r1_is_still_flagged(self, monkeypatch):
        # The warning must survive for the model it exists for.
        _patch_models(monkeypatch, MODELS)
        result = dashboard.check_lm_studio(
            "http://localhost:1234",
            configured_model="deepseek-r1-distill-qwen-32b")
        assert result["active_model"] == "deepseek-r1-distill-qwen-32b"
        assert result["is_r1"] is True
        assert result["recommended"] == "qwen/qwen3-32b"

    def test_a_first_loaded_r1_is_avoided_when_a_fast_model_is_loaded(
            self, monkeypatch):
        """
        The operator's machine, 2026-09-25: LM Studio listed
        deepseek-r1-distill-qwen-32b first, `local-model` meant auto, and the
        auto pick took it - ~9 minutes for ONE market against a 10-minute cycle,
        so a cycle could not finish. When a fast model is loaded, "auto" must
        mean the fast one; the R1 is only ever a warning, never the pick.
        """
        _patch_models(monkeypatch, ["deepseek-r1", "qwen/qwen3-32b"])
        result = dashboard.check_lm_studio("http://localhost:1234",
                                           configured_model=None)
        assert result["active_model"] == "qwen/qwen3-32b"
        assert result["is_r1"] is False
        # ...and the screen can explain the pick rather than just assert it.
        assert result["first_loaded_model"] == "deepseek-r1"
        assert "R1-style" in result["model_reason"]

    def test_an_r1_that_is_the_only_choice_is_still_flagged(self, monkeypatch):
        """The warning must survive for the case it exists for."""
        _patch_models(monkeypatch, ["deepseek-r1"])
        result = dashboard.check_lm_studio("http://localhost:1234",
                                           configured_model=None)
        assert result["active_model"] == "deepseek-r1"
        assert result["is_r1"] is True

    def test_configured_model_not_loaded_is_reported(self, monkeypatch):
        _patch_models(monkeypatch, MODELS)
        result = dashboard.check_lm_studio("http://localhost:1234",
                                           configured_model="gpt-not-there")
        assert result["model_not_loaded"] is True
        assert result["active_model"] == MODELS[0]

    def test_speed_line_follows_the_active_model(self, monkeypatch):
        """
        The "SLOW - 8 min per market" string must not appear when the
        model in use is fast, even though R1 is downloaded - the exact
        panel the operator saw.
        """
        _patch_models(monkeypatch, MODELS)
        monkeypatch.setattr(dashboard, "read_env_file",
                            lambda: {"LM_STUDIO_HOST": "http://localhost:1234"})
        import asyncio
        result = asyncio.run(dashboard.api_llm_status())
        assert result["is_r1"] is False
        assert result["speed"].startswith("FAST")
        assert "R1" not in (result.get("warning") or "")
