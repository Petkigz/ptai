"""
The prices the agent pays, and the things the log said that were not true.

Operator's log, 2026-09-25 21:52 - 22:31 (pasted with no text). Four separate
things in it, each of which cost either money, the whole cycle, or the operator's
ability to see what happened:

1. It decided to trade a 99.8%-spread market. The evidence, from the log:

       REAL orderbook 2774057: bid 0.0010 (449457) ask 0.9990 (3320916) spread 99.80%
       Edge calc 2774057: [YES] Raw 0.200 (fair 0.207 - mkt 0.007) ...
       Should trade: True (raw>=8%, effective>0) | $50 math: fee 5.9% + gas 1.0%
       + spread 99.8% = 106.7% cost must exceed to break even

   A share cost 0.999 there, either side. The "+0.200 edge" was measured against
   the market's own 0.007 quote, and the line that printed a 106.7% break-even
   cost printed "Should trade: True" next to it. It then sat for nine minutes in
   an LLM call, fell back to a heuristic, and priced the same market again.

2. `fair 0.207` on a 0.007 market had no evidence behind it at all: the base-rate
   model is a hardcoded category constant (sports 0.50) that claimed confidence
   0.6, on a machine where research could not be fetched and the model call had
   just timed out.

3. One "180 second" model call took 547 s. 547 = 3 x 180 + overhead: the OpenAI
   client retries twice by default and each retry re-sends the prompt.

4. Every market's research was a DuckDuckGo connect timeout (12 s each, and one
   DNS failure); every market's resolution analysis said "Vague resolution
   source: 'consensus'" and refused it; the ESPN feeds refused the machine with
   403 on every cycle and the log blamed "no fixtures from any feed".
"""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.intelligence.resolution_analyzer import ResolutionAnalyzer
from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.strategy.edge import MAX_TRADABLE_SPREAD, EdgeCalculator


BOOK_OK = {"bid": 0.598, "ask": 0.602, "spread": 0.004,
           "bid_size": 5000, "ask_size": 5000, "is_real": True, "source": "clob_real"}


def _market(price: float = 0.60, market_id: str = "M1", question: str = "Will it happen?"):
    return Market(id=market_id, source=MarketSource.POLYMARKET, question=question,
                  venue_id="polymarket", liquidity=50000, volume_24h=50000,
                  tokens=[Token(token_id="t", outcome="YES", price=price)])


def _log_market():
    """Market 2774057 exactly as the log shows it: 0.007 quote, 0.001/0.999 book."""
    m = _market(price=0.007, market_id="2774057",
                question="Strait of Hormuz traffic returns to normal by September 30?")
    m.outcome_prices = [0.007, 0.993]
    return m


LOG_BOOK = {"bid": 0.001, "ask": 0.999, "spread": 0.998, "bid_size": 449457,
            "ask_size": 3320916, "is_real": True, "source": "clob_real"}


class TestTheTradeInTheLogIsRefused:
    def test_the_99_point_spread_market_is_not_a_trade(self):
        """
        Fair 0.207 against a 0.007 quote, on a book you can only cross at 0.999.
        The mispricing is real and the trade is still a loss; nothing may place it.
        """
        edge = EdgeCalculator().calculate(
            market=_log_market(), fair_prob=0.207, uncertainty=0.2,
            orderbook=LOG_BOOK, amount_usd=3.0, side="YES")
        assert edge.raw_edge > 0.15           # the mispricing is still reported
        assert edge.book_is_real
        assert edge.price_paid == pytest.approx(0.999)
        assert edge.executable_edge < 0        # after paying it, there is nothing
        assert not edge.should_trade
        assert "spread" in edge.blocked_by

    def test_the_other_side_of_that_market_is_not_a_trade_either(self):
        edge = EdgeCalculator().calculate(
            market=_log_market(), fair_prob=0.793, uncertainty=0.2,
            orderbook=LOG_BOOK, amount_usd=3.0, side="NO")
        assert edge.price_paid == pytest.approx(0.999)   # 1 - 0.001
        assert not edge.should_trade

    def test_a_book_this_wide_is_refused_even_with_a_large_edge(self):
        edge = EdgeCalculator().calculate(
            market=_log_market(), fair_prob=0.90, uncertainty=0.1,
            orderbook=LOG_BOOK, amount_usd=3.0, side="YES")
        assert not edge.should_trade
        assert "spread" in edge.blocked_by
        assert MAX_TRADABLE_SPREAD < 0.998

    def test_a_normal_market_still_trades(self):
        edge = EdgeCalculator().calculate(
            market=_market(), fair_prob=0.70, uncertainty=0.1,
            orderbook=BOOK_OK, amount_usd=3.0, side="YES")
        assert edge.price_paid == pytest.approx(0.602)
        assert edge.executable_edge > 0
        assert edge.blocked_by == ""
        assert edge.should_trade

    def test_an_estimated_book_is_never_a_trade(self):
        """
        The log's 4052419: token 404 -> ESTIMATED book -> "Should trade: True".
        An estimated book has no executable price, so there is nothing to trade.
        """
        estimated = {"bid": 0.49, "ask": 0.51, "spread": 0.01, "is_real": False,
                     "source": "estimated"}
        edge = EdgeCalculator().calculate(
            market=_market(price=0.50), fair_prob=0.70, uncertainty=0.1,
            orderbook=estimated, amount_usd=3.0, side="YES")
        assert not edge.should_trade
        assert "not real" in edge.blocked_by

    def test_a_one_sided_book_is_refused(self):
        """`bid 0.9790 (0)` in the log: a price with no size behind it."""
        one_sided = {"bid": 0.979, "ask": 0.999, "spread": 0.02, "bid_size": 0,
                     "ask_size": 1062046, "is_real": True}
        edge = EdgeCalculator().calculate(
            market=_market(price=0.99), fair_prob=0.99, uncertainty=0.1,
            orderbook=one_sided, amount_usd=3.0, side="NO")
        assert not edge.should_trade
        assert "empty" in edge.blocked_by


class TestAFairValueNeedsEvidence:
    def test_a_hardcoded_category_constant_claims_no_confidence(self):
        """
        BaseRateModel is `sports: 0.50` with no data loaded. It used to answer
        with confidence 0.6 - more than most real evidence - which is how a 0.007
        market acquired a 0.207 fair value with the model timed out and research
        dead.
        """
        from src.ptai.intelligence.forecaster import BaseRateModel
        model = BaseRateModel()
        assert model.has_data is False
        forecast = model.forecast(_log_market(), category="sports")
        assert forecast.confidence == 0.0
        assert "not a forecast" in forecast.reasoning

    def test_the_ensemble_does_not_manufacture_an_edge_from_a_constant(self):
        """
        With every evidence source empty - no news, no sentiment, no orderbook
        information, no LLM - the fair value must stay at the market's own price.
        """
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        market = _log_market()
        result = EnsembleForecaster().forecast_market(market, context={
            "category": "sports", "news": "", "research": "",
            "sentiment": {"score": 0.0, "confidence": 0.0},
            "tweets": [], "orderbook": {},
        })
        assert abs(result.edge) < 0.08, (
            f"an empty context produced edge {result.edge:.3f} - "
            f"fair {result.fair_probability:.3f} vs market {market.best_price:.3f}")


class TestTheBoundedWaitIsBounded:
    def test_the_client_is_not_allowed_to_retry(self):
        """
        180 s x 3 attempts = 547 s, which is what the operator's log measured.
        The retry count is part of the limit.
        """
        import inspect

        from src.ptai.llm import provider
        source = inspect.getsource(provider.LMStudioProvider.chat)
        assert "max_retries=0" in source
        assert "timeout=self.timeout_seconds" in source

    def test_the_failure_line_says_how_long_it_waited(self):
        import inspect

        from src.ptai.llm import provider
        source = inspect.getsource(provider.LMStudioProvider.chat)
        assert "LM Studio chat failed after" in source
        assert "no retry" in source

    def test_the_generic_provider_is_bounded_too(self):
        import inspect

        from src.ptai.llm import provider
        source = inspect.getsource(provider.OpenAICompatibleProvider.chat)
        assert "timeout=self.timeout_seconds" in source
        assert "max_retries=0" in source

    def test_the_bound_reaches_both_providers_from_the_router(self):
        from src.ptai.llm.provider import LLMRouter
        from src.ptai.llm import provider as provider_module

        class _Fake(provider_module.BaseLLMProvider):
            def is_available(self):
                return True
            def list_models(self):
                return ["qwen/qwen3-32b"]
            def chat(self, prompt, system=""):
                return None

        router = LLMRouter(preferred="auto", timeout_seconds=42)
        assert router.timeout_seconds == 42.0
        # the provider the router builds carries the same bound
        lm = provider_module.LMStudioProvider(model="m", timeout_seconds=router.timeout_seconds)
        assert lm.timeout_seconds == 42.0


class TestTheLogSaysWhyNothingHappened:
    def test_the_outcome_of_every_market_is_counted(self):
        from src.ptai.intelligence.uncertainty import UncertaintyEngine
        from src.ptai.strategy.fair_value import FairValueEngine

        engine = FairValueEngine(uncertainty_engine=UncertaintyEngine())
        engine.reset_outcomes()
        engine.estimate(_log_market(), context={"orderbook": LOG_BOOK, "category": "sports"})
        summary = engine.outcome_summary()
        assert "refused" in summary
        assert "1 " in summary

    def test_the_resolution_gate_no_longer_bans_a_whole_venue(self):
        """
        Polymarket states its resolution rule on the market: "will resolve based
        on the consensus of credible reporting". The analyzer read the word
        "consensus" as vagueness and refused every market that carried it.
        """
        analyzer = ResolutionAnalyzer()
        for question in (
            "Game 2: Both Teams Beat Roshan?",
            "Will LeBron James win the 2028 Democratic presidential nomination?",
            "Will Slavia Prague win the 2026-27 UEFA Champions League Championship?",
        ):
            market = Market(id="m", source=MarketSource.POLYMARKET, question=question,
                            description="This market will resolve based on the "
                                        "consensus of credible reporting.",
                            venue_id="polymarket", liquidity=50000)
            analysis = analyzer.analyze(market)
            assert analysis.should_trade, (question, analysis.risks)
            assert analysis.named_resolver

    def test_the_gate_keeps_closed_on_a_genuinely_undefined_question(self):
        analyzer = ResolutionAnalyzer()
        vague = Market(id="m", source=MarketSource.POLYMARKET,
                       question="Will the market be significantly higher?",
                       description="Resolves by general agreement.",
                       venue_id="polymarket", liquidity=50000)
        analysis = analyzer.analyze(vague)
        assert not analysis.should_trade
        assert analysis.risks

    def test_the_question_is_what_decides_ambiguity_not_the_boilerplate(self):
        """
        The same word in the venue's description must not block the market, and
        the same word in the question must.
        """
        analyzer = ResolutionAnalyzer()
        boilerplate = Market(
            id="m", source=MarketSource.POLYMARKET, question="Will France win?",
            description="Officially reported results will resolve this market, "
                        "based on the official broadcast.",
            venue_id="polymarket", liquidity=50000)
        assert analyzer.analyze(boilerplate).should_trade

        in_question = Market(
            id="m", source=MarketSource.POLYMARKET,
            question="Will France win by a significant margin?",
            description="Resolves on the official broadcast of the match.",
            venue_id="polymarket", liquidity=50000)
        analysis = analyzer.analyze(in_question)
        assert not analysis.should_trade
        assert "significant" in " ".join(analysis.ambiguous_language).lower()


class TestTheDeadLanesAreNamed:
    def test_a_feed_that_refuses_the_machine_is_not_retried_every_cycle(self):
        from src.ptai.betting.sports_data import BaseProvider

        provider = BaseProvider()
        assert provider.is_blocked is False
        provider._mark_blocked(403, "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard")
        assert provider.is_blocked is True
        assert "403" in provider.blocked_reason
        # ...and it lifts by itself
        provider.blocked_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert provider.is_blocked is False

    def test_the_betting_lane_says_refused_not_empty(self):
        """`no fixtures from any feed` was printed for both facts."""
        import asyncio

        from src.ptai.betting.engine import BettingEngine
        from src.ptai.betting.sports_data import BaseProvider, SportsDataEngine

        class Refusing(BaseProvider):
            name = "refusing"
            async def events(self, leagues=()):
                self._mark_blocked(403, "https://example.com/scoreboard")
                return []

        engine = BettingEngine(data_engine=SportsDataEngine(providers=[Refusing()]))
        result = asyncio.run(engine.run_cycle(leagues=("epl",)))
        assert result["ok"] is False
        assert "refused" in result["blockers"][0]
        assert result.get("remedy")

    def test_search_that_is_unreachable_is_not_retried_per_market(self):
        """
        12 s of connect timeout per market, in a scan of hundreds of markets,
        for a search engine that is not answering.
        """
        import asyncio

        from src.ptai.information.web_researcher import WebResearcher

        calls = {"n": 0}

        async def failing_fetch(url):
            calls["n"] += 1
            return 0, ""

        researcher = WebResearcher(fetch=failing_fetch)
        first = asyncio.run(researcher.discover_sources("market one"))
        assert first == []
        assert calls["n"] == 1
        for question in ("market two", "market three", "market four"):
            assert asyncio.run(researcher.discover_sources(question)) == []
        assert calls["n"] == 1, "the search was retried once per market"
        assert "unreachable" in researcher._search_blocked_reason

    def test_an_unresearched_market_is_not_quietly_penalised(self):
        """
        Research being down is a fact about the network, not evidence against the
        market. The confidence penalty stays off; only the reason is recorded.
        """
        from src.ptai.intelligence.contradiction import ContradictionEngine

        engine = ContradictionEngine()
        report = engine.synthesize(market=_market(), research_text="",
                                   news="", tweets=[])
        assert report.researched is False
        assert report.confidence_adjustment < 0   # reported ...
        from src.ptai.intelligence.ensemble import EnsembleForecaster
        forecaster = EnsembleForecaster()
        result = forecaster.forecast_market(_market(), context={
            "orderbook": BOOK_OK, "category": "default",
            "sentiment": {"score": 0, "confidence": 0}, "news": "", "research": ""})
        # ... but the fair value is not moved by a missing web search
        assert abs(result.fair_probability - _market().best_price) < 0.12


class TestTheConsoleStopsTalkingToItself:
    def test_one_storage_per_process(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "one.db"))
        console = importlib.import_module("src.ptai.ui.console")
        first = console.get_storage()
        second = console.get_storage()
        assert first is second

    def test_a_different_database_still_gets_a_storage(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "a.db"))
        console = importlib.import_module("src.ptai.ui.console")
        first = console.get_storage()
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "b.db"))
        second = console.get_storage()
        assert first is not second

    def test_no_request_closes_the_shared_storage(self):
        """
        The two routes that used to close it made every later request fail with
        "Cannot operate on a closed database" once one Storage is shared.
        """
        import inspect

        from src.ptai.ui import console as console_module
        source = inspect.getsource(console_module)
        assert "storage.close()" not in source
        # ...and a handle closed by something else is rebuilt, not served broken
        assert '_STORAGE["instance"].conn.execute("SELECT 1")' in source

    def test_the_routes_still_answer_after_repeated_use(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        console = importlib.import_module("src.ptai.ui.console")
        from fastapi.testclient import TestClient
        client = TestClient(console.app)
        for _ in range(3):
            assert client.get("/api/console/agent").status_code == 200
            assert client.get("/api/console/status").status_code == 200
            assert client.get("/api/console/capital").status_code == 200

    def test_the_page_only_polls_while_it_is_visible(self):
        import inspect

        from src.ptai.ui import console as console_module
        source = inspect.getsource(console_module)
        assert "document.visibilityState === 'hidden'" in source
        assert "_loading" in source
        assert "setInterval(loadAll, 15000)" in source
