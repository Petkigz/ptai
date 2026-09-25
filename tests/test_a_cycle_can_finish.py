"""
A cycle must be able to finish, and the operator must be able to see why.

The operator ran the console on their PC on 2026-09-25 and pasted the log. It
showed four separate reasons a 10-minute cycle could not complete, none of which
said "this is broken":

  1. `LM Studio auto-detected model: deepseek-r1-distill-qwen-32b` - the auto
     pick took an R1-style reasoning model, which needs ~9 minutes for ONE
     market. 21:11:05 -> 21:20:20 for a single forecast; the next market started
     524 seconds later, and the cycle could never fit its interval.
  2. Nothing bounded that call. The OpenAI client's own default is 600 s, so one
     market could hold the whole cycle with no upper limit in PTAI at all.
  3. `X scrape failed: object list can't be used in 'await' expression` on every
     market - the sentiment lane was dead before it scraped anything, and the
     failure was logged as if X were merely blocked.
  4. `[espn] fetch failed ... 403 Forbidden` on both leagues -> "cycle aborted:
     no fixtures from any feed" - the whole betting lane (goals, cards, corners,
     totals, matches) priced nothing, every cycle, because of a user agent.

Between two log lines five hundred seconds apart the console had nothing to
show, which is how a working system reads as a hung one. So this file also
asserts the screen can say what the cycle is doing while it does it.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.llm.provider import (
    LMStudioProvider,
    choose_loaded_model,
    is_slow_reasoning_model,
)


# ---------------------------------------------------------------------------
# 1. Which model "auto" means - the ~9-minutes-per-market trap
# ---------------------------------------------------------------------------

class TestAutoNeverPicksASlowModelWhenAFastOneIsLoaded:
    def test_the_operators_machine_picks_the_fast_model(self):
        # Their list, in their order: the R1 first.
        models = ["deepseek-r1-distill-qwen-32b",
                  "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp",
                  "qwen/qwen3-32b"]
        model, reason = choose_loaded_model(models, "local-model")
        assert model == "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp"
        assert "R1-style" in reason and "deepseek-r1" in reason

    def test_a_pinned_model_still_wins(self):
        models = ["deepseek-r1", "qwen/qwen3-32b"]
        model, reason = choose_loaded_model(models, "deepseek-r1")
        assert model == "deepseek-r1"
        assert reason.startswith("pinned")

    def test_a_pinned_model_that_is_not_loaded_falls_back_and_says_so(self):
        model, reason = choose_loaded_model(["qwen/qwen3-32b"], "not-loaded")
        assert model == "qwen/qwen3-32b"
        assert "NOT loaded" in reason

    def test_when_every_loaded_model_is_slow_it_takes_one_and_warns(self):
        model, reason = choose_loaded_model(["deepseek-r1"], None)
        assert model == "deepseek-r1"
        assert "EVERY loaded model is R1-style" in reason

    def test_nothing_loaded_is_not_a_model(self):
        model, reason = choose_loaded_model([], "local-model")
        assert model is None
        assert "no model is loaded" in reason

    def test_r1_is_a_name_segment_not_a_substring(self):
        for slow in ("deepseek-r1", "deepseek-r1:32b", "deepseek-r1-distill-qwen-32b",
                     "R1"):
            assert is_slow_reasoning_model(slow) is True, slow
        for fast in ("qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp",
                     "qwen/qwen3-32b", "prism-ml/bonsai-27b", "r1b-bert", None, ""):
            assert is_slow_reasoning_model(fast) is False, fast

    def test_the_provider_makes_the_same_choice_the_console_reports(self, monkeypatch):
        """
        One decision, two readers. The provider that makes the CALL and the
        screen that reports the model must not be able to disagree.
        """
        provider = LMStudioProvider(model="local-model")
        monkeypatch.setattr(provider, "list_models",
                            lambda: ["deepseek-r1", "qwen/qwen3-32b"])
        chosen, _reason = choose_loaded_model(provider.list_models(), provider.model)
        assert chosen == "qwen/qwen3-32b"

        import src.ptai.dashboard as dashboard
        assert dashboard._active_lm_model(provider.list_models(), provider.model) == chosen


# ---------------------------------------------------------------------------
# 2. One model call cannot hold the whole cycle
# ---------------------------------------------------------------------------

class TestTheModelCallIsBounded:
    def test_the_provider_defaults_to_a_bounded_wait(self):
        assert LMStudioProvider(model="m").timeout_seconds == 180.0

    def test_the_wait_is_configurable(self):
        assert LMStudioProvider(model="m", timeout_seconds=45).timeout_seconds == 45.0

    def test_the_router_forwards_the_bound_to_the_provider(self):
        from src.ptai.llm.provider import LLMRouter
        router = LLMRouter(preferred="lm_studio", model="m", timeout_seconds=90)
        assert router.timeout_seconds == 90.0
        assert router.provider is None or router.provider.timeout_seconds == 90.0

    def test_the_setting_exists_and_defaults_to_three_minutes(self):
        from src.ptai.config import get_settings
        settings = get_settings()
        assert settings.llm_timeout_seconds == 180.0

    def test_the_agent_loop_passes_it(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "src" / "ptai" / "agent" / "v3_loop.py"
        text = src.read_text(encoding="utf-8")
        block = text[text.index("self.llm_router = LLMRouter("):]
        assert "timeout_seconds" in block[:400], (
            "the trading loop builds its router without the bound, so the "
            "client's own 600s default applies again")


# ---------------------------------------------------------------------------
# 3. The two crashes in the log
# ---------------------------------------------------------------------------

class TestTheLoggedCrashesAreFixed:
    def test_the_sentiment_scrape_does_not_await_a_list(self):
        """
        `await` on the scraper's plain-list search raised on EVERY market, so
        the sentiment lane contributed nothing and the log blamed X blocking.
        """
        import asyncio

        from src.ptai.information.x_engine import XEngine

        class FakeScraper:
            circuit_open = False
            def search(self, question, limit=20):
                return []

        engine = XEngine(x_scraper=FakeScraper())
        signal = asyncio.run(engine.get_signal(_market()))
        assert signal is not None
        assert getattr(signal, "used", False) is False

    def test_the_sentiment_scrape_still_supports_a_coroutine(self):
        import asyncio

        from src.ptai.information.x_engine import XEngine

        class AsyncScraper:
            circuit_open = False
            async def search(self, question, limit=20):
                return []

        signal = asyncio.run(XEngine(x_scraper=AsyncScraper()).get_signal(_market()))
        assert signal is not None

    def test_the_sports_feeds_identify_as_a_browser(self):
        """
        403 Forbidden aborted the whole betting lane: "no fixtures from any
        feed" every cycle, so goals, cards, corners and match markets were
        never priced at all.
        """
        from src.ptai.betting.sports_data import BROWSER_USER_AGENT, BaseProvider
        assert "Mozilla" in BROWSER_USER_AGENT
        assert BaseProvider().user_agent == BROWSER_USER_AGENT

    def test_the_research_and_news_paths_use_the_same_identity(self):
        """A bot UA on a public feed is what gets a 403 - or a silent timeout."""
        import inspect

        from src.ptai.betting.sports_data import BROWSER_USER_AGENT
        from src.ptai.information import news_engine, web_researcher

        for module in (news_engine, web_researcher):
            text = inspect.getsource(module)
            assert "BROWSER_USER_AGENT" in text, module.__name__
            assert "ptai/1.0 (local research bot)" not in text, (
                f"{module.__name__} still sends the user agent that gets a 403")
        assert "Mozilla" in BROWSER_USER_AGENT


# ---------------------------------------------------------------------------
# 4. The screen can say what the cycle is doing, and why it is slow
# ---------------------------------------------------------------------------

class TestTheConsoleExplainsASlowCycle:
    def test_the_phase_counts_markets_as_it_prices_them(self, tmp_path):
        from src.ptai.operator_view import agent_state

        from tests.test_full_cycle_from_discovery_to_allocation import build_agent

        agent, adapter = build_agent(tmp_path)
        agent._cycle_markets_total = 24
        agent._cycle_market_index = 0
        agent._slowest_model_call_seconds = None
        import asyncio
        asyncio.run(agent.get_context(adapter._market()))
        asyncio.run(agent.get_context(adapter._market()))

        phase = json.loads(agent.storage.get_state("agent.phase"))
        assert phase["phase"] == "evaluating"
        assert phase["detail"].startswith("market 2 of 24")
        assert "previous market took" in phase["detail"]

        # ...and the console reads it without knowing anything about the loop.
        state = agent_state(agent.storage)
        assert state["phase"] == "evaluating"
        assert "market 2 of 24" in state["doing"]

    def test_the_completed_cycle_names_the_slowest_market(self, tmp_path,
                                                          monkeypatch):
        from tests.test_full_cycle_from_discovery_to_allocation import (
            _cycle, _force_qualified, _inject_opportunity, build_agent)

        agent, adapter = build_agent(tmp_path)
        _force_qualified(agent, monkeypatch)
        _inject_opportunity(agent, adapter._market(), monkeypatch)
        agent._slowest_model_call_seconds = 524.0
        _cycle(agent)
        phase = json.loads(agent.storage.get_state("agent.phase"))
        assert "slowest single market took 524s" in phase["detail"]

    def test_the_console_reports_the_model_reason_and_the_bound(self, tmp_path,
                                                                monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "c.db"))
        console = importlib.import_module("src.ptai.ui.console")
        monkeypatch.setattr(console, "_brain_status", lambda force=False: {
            "available": True, "connected": True,
            "models": ["deepseek-r1", "qwen/qwen3-32b"],
            "active_model": "qwen/qwen3-32b", "is_r1": False,
            "model_reason": "auto: picked qwen/qwen3-32b because it is a fast "
                            "model, and deepseek-r1 is R1-style",
            "timeout_seconds": 180.0, "env_model": "local-model", "pinned": False,
        })
        from fastapi.testclient import TestClient
        body = TestClient(console.app).get("/api/console/agent").json()
        assert body["brain"]["timeout_seconds"] == 180.0
        assert "fast model" in body["brain"]["model_reason"]

    def test_the_call_limit_is_settable_from_the_console(self, tmp_path,
                                                         monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "c2.db"))
        console = importlib.import_module("src.ptai.ui.console")
        written = {}
        import src.ptai.dashboard as dashboard
        monkeypatch.setattr(dashboard, "write_env_file",
                            lambda updates: written.update(updates))
        from fastapi.testclient import TestClient
        client = TestClient(console.app)

        ok = client.post("/api/console/brain", json={"timeout_seconds": 120})
        assert ok.status_code == 200
        assert written == {"LLM_TIMEOUT_SECONDS": "120"}
        assert "120" in ok.json()["note"]

        # Refused, not clamped: a limit that cannot work is not a setting.
        for bad in ({"timeout_seconds": 1}, {"timeout_seconds": 99999},
                    {"timeout_seconds": "soon"}):
            response = client.post("/api/console/brain", json=bad)
            assert response.status_code == 400, bad
        assert written == {"LLM_TIMEOUT_SECONDS": "120"}, (
            "a refused limit must not reach .env")


def _market():
    from src.ptai.markets.base import Market, MarketSource, Token
    return Market(id="m1", source=MarketSource.POLYMARKET,
                  question="Will X happen?", venue_id="polymarket",
                  tokens=[Token(token_id="t1", outcome="YES", price=0.5)],
                  liquidity=1000.0, volume=1000.0)
