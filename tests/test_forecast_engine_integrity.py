"""
The forecast engine: what each component actually contributed, and what it must
never claim.

The operator's 2026-09-27 log showed five problems at once, all in the prediction
stack rather than the risk layer:

  * seventeen LLM forecasts came back 0.65 / 0.72 / edge 0.15 on markets priced
    15%-60% - the numbers from the prompt's own example. Anchoring, not analysis.
  * web research was retrieved and then thrown away: the contradiction engine was
    handed the research TEXT, decided `researched=False` from the object it never
    received, and returned early - so "5/6 sources retrieved" was immediately
    followed by "NOT RESEARCHED (no sources)".
  * the console could show the LLM's 65% and the ensemble's 58% and nothing in
    between.
  * X was dead (circuit breaker open) and reported as a neutral-ish zero.
  * the base-rate model had no data at all, by its own admission.
  * and on every market the CLOB showed "bid 0.01 / ask 0.99, spread 98%", which
    the operator correctly refused to believe.

These tests pin the fixes for each, and the refusals that must survive them.
"""

from __future__ import annotations

import pytest

from ptai.agent.brain import AnswerRepetitionDetector, Brain
from ptai.information.web_researcher import FetchedSource, ResearchResult
from ptai.intelligence.base_rates import BaseRateBook
from ptai.intelligence.contradiction import ContradictionEngine
from ptai.intelligence.ensemble import EnsembleForecaster
from ptai.intelligence.forecaster import BaseRateModel, XModel
from ptai.markets.base import Market, MarketSource
from ptai.storage.db import Storage
from ptai.venues.polymarket_adapter import normalise_book


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def market(mid: str = "M1", price: float = 0.35,
           question: str = "Will the thing happen?") -> Market:
    return Market(id=mid, source=MarketSource.POLYMARKET, question=question,
                  outcome_prices=[price, 1 - price], raw={})


class FakeRouter:
    """An LLM that answers whatever it is told to, and records the prompt."""

    def __init__(self, fair: float = 0.65, confidence: float = 0.72,
                 captured: dict = None):
        self.fair = fair
        self.confidence = confidence
        self.captured = captured if captured is not None else {}

    def get_provider_name(self):
        return "fake-llm"

    def chat(self, prompt=None, system=None, **kw):
        from types import SimpleNamespace
        self.captured["user"] = prompt or ""
        self.captured["system"] = system or ""
        return SimpleNamespace(
            provider="fake-llm", content="",
            parsed_json={"fair_value": self.fair, "confidence": self.confidence,
                         "edge": self.fair - 0.35, "side": "YES",
                         "should_trade": True, "reasoning": "because",
                         "basis": "base_rate"})


class _Assess:
    overall_quality = 0.8


def _source(url: str, supports: str) -> FetchedSource:
    return FetchedSource(url=url, status=200, ok=True, text="Evidence body.",
                         assessment=_Assess(), supports=supports)


def _research(market_id: str = "M1", researched: bool = True) -> ResearchResult:
    return ResearchResult(
        market_id=market_id, question="Will the thing happen?",
        supporting_yes="yes side", supporting_no="no side", contradicting_both="",
        unknown="", resolution_risks="", final_summary="summary",
        sources=["https://a.example"], confidence=0.6, time_spent_seconds=4.0,
        researched=researched,
        source_details=([_source("https://a.example", "yes"),
                         _source("https://b.example", "no")] if researched else []),
        sources_retrieved=(2 if researched else 0),
    )


# --------------------------------------------------------------------------
# 1. the prompt must not carry an answer, and a repeated answer must not count
# --------------------------------------------------------------------------

class TestThePromptDoesNotHandOverTheAnswer:
    def test_no_numeric_example_reaches_the_model(self):
        captured = {}
        Brain(llm_router=FakeRouter(captured=captured)).estimate_fair_value(
            market=market(), sentiment=None)
        prompt = captured["user"]
        for leaked in ('"fair_value":0.65', '"edge":0.15', '"confidence":0.72',
                       "0.65,", "0.72,"):
            assert leaked not in prompt, (
                f"the prompt still contains the example answer {leaked!r}: a "
                f"model shown one worked example copies its numbers")
        # The response schema is what gets copied, so it must carry no numerals
        # at all - only the placeholders describing what to put there. (The rules
        # line after it keeps its bounds: "0.01-0.99" is an instruction about the
        # answer's range, not an answer anyone can copy.)
        schema = prompt.split("Respond ONLY JSON", 1)[-1].split("\n")[1]
        assert not any(ch.isdigit() for ch in schema), (
            f"the response schema still contains a number: {schema}")
        # The shape is still given, and the instruction says the numbers are not
        # answers.
        assert "PLACEHOLDERS" in prompt
        assert "fair_value" in schema

    def test_the_instruction_forbids_reusing_a_probability(self):
        captured = {}
        Brain(llm_router=FakeRouter(captured=captured)).estimate_fair_value(
            market=market(), sentiment=None)
        prompt = captured["user"].lower()
        assert "do not reuse a probability" in prompt
        assert "that is a correct answer, not a failure" in prompt


class TestARepeatedAnswerCarriesNoWeight:
    def _answer(self, detector, price, fair=0.65):
        return detector.observe(f"M{price}", price, fair)

    def test_a_value_repeated_across_different_markets_is_flagged(self):
        d = AnswerRepetitionDetector()
        results = [self._answer(d, p) for p in (0.15, 0.28, 0.42, 0.60)]
        assert results[:3] == [None, None, None], (
            "three answers is not yet a pattern")
        assert results[3], "the fourth identical answer across a 0.45-wide price range must flag"

    def test_agreeing_with_the_market_is_not_flagged(self):
        d = AnswerRepetitionDetector()
        assert not any(self._answer(d, p, fair=0.50)
                       for p in (0.50, 0.51, 0.49, 0.50)), (
            "a model that agrees with similar markets is not anchoring")

    def test_an_anchored_answer_loses_its_confidence_and_the_trade(self):
        router = FakeRouter(captured={})
        brain = Brain(llm_router=router)
        last = None
        for price in (0.15, 0.28, 0.42, 0.60):
            last = brain.estimate_fair_value(market=market(f"M{price}", price))
        assert last.raw["anchored"] is True
        assert last.confidence == 0.0
        assert last.should_trade is False
        assert "ANCHORED" in last.reasoning

    def test_the_ensemble_gives_an_anchored_component_no_weight(self):
        forecaster = EnsembleForecaster(llm_router=FakeRouter(captured={}))
        result = None
        for price in (0.15, 0.28, 0.42, 0.60):
            result = forecaster.forecast_market(
                market(f"M{price}", price), context={"category": "politics"})
        llm = [c for c in result.components if c["model"] == "llm_reasoning"][0]
        assert llm["anchored"] is True
        assert llm["weight_used"] == 0.0
        assert result.chain["llm_anchored"] is True
        assert "anchored" in llm["note"]
        # With the LLM out, the ensemble must not manufacture an opinion from
        # nothing: with no other evidence the fair value is the market's price.
        assert result.fair_probability == pytest.approx(result.market_price)

    def test_an_honest_answer_still_reaches_the_ensemble(self):
        forecaster = EnsembleForecaster(llm_router=FakeRouter(fair=0.40, captured={}))
        result = forecaster.forecast_market(market("M1", 0.35),
                                            context={"category": "politics"})
        llm = [c for c in result.components if c["model"] == "llm_reasoning"][0]
        assert llm["contributes"] is True
        assert llm["weight_used"] > 0
        assert result.chain["llm_raw"] == pytest.approx(0.40)


# --------------------------------------------------------------------------
# 2. every component's contribution is visible, in order
# --------------------------------------------------------------------------

class TestTheChainIsVisible:
    def test_the_explain_line_names_every_component_and_the_final_numbers(self):
        forecaster = EnsembleForecaster(llm_router=FakeRouter(fair=0.55, captured={}))
        result = forecaster.forecast_market(
            market("M1", 0.35),
            context={"category": "politics", "sentiment": {"score": 0.4},
                     "tweets": [{"text": "yes"}]})
        line = result.explain()
        for name in ("base_rate", "news", "x_sentiment",
                     "market_microstructure", "llm_reasoning"):
            assert name in line, f"{name} is missing from the chain: {line}"
        assert "-> calibrated" in line and "-> conservative" in line
        assert "market 0.350" in line
        # The four stages the operator asked to see together.
        for key in ("market_price", "llm_raw", "ensemble_raw", "calibrated",
                    "conservative"):
            assert key in result.chain

    def test_a_component_with_no_data_says_so_rather_than_showing_a_neutral(self):
        forecaster = EnsembleForecaster(llm_router=None)
        result = forecaster.forecast_market(market("M1", 0.35), context={})
        for row in result.components:
            if row["model"] in ("base_rate", "news"):
                assert row["contributes"] is False
                assert row["note"], "a zero-weight component must say why"

    def test_the_chain_travels_with_the_opportunity(self):
        """
        The console reads it from the opportunity, so a proposed trade can always
        show which component said what - not just the log file.
        """
        from ptai.strategy.strategy_engine import StrategyEngineV3
        import inspect
        src = inspect.getsource(StrategyEngineV3.evaluate_market_with_all_strategies)
        assert 'opp.raw["fair_value_chain"]' in src
        assert 'opp.raw["components"]' in src


# --------------------------------------------------------------------------
# 3. retrieved research must reach the contradiction engine
# --------------------------------------------------------------------------

class TestResearchIsConsumed:
    def test_a_researched_result_produces_evidence(self):
        engine = ContradictionEngine()
        report = engine.synthesize(market=market("M1"), research_text="summary",
                                   research_result=_research("M1"))
        assert report.researched is True
        assert report.supporting_yes and report.supporting_no, (
            "the retrieved sources were not turned into evidence - this is the "
            "stage that read 'NOT RESEARCHED (no sources)' over five sources")
        assert report.provenance == "web_fetch"

    def test_no_result_is_still_not_researched(self):
        report = ContradictionEngine().synthesize(market=market("M1"),
                                                  research_text="")
        assert report.researched is False
        assert report.research_blockers
        assert report.bull_case == "" and report.bear_case == ""

    def test_one_market_does_not_inherit_anothers_sources(self):
        """
        The engine is shared by every market in a cycle. It used to keep the last
        result and hand it to the next market, which was reported as researched on
        someone else's evidence.
        """
        engine = ContradictionEngine()
        engine.synthesize(market=market("M1"), research_result=_research("M1"))
        second = engine.synthesize(market=market("M2"), research_text="")
        assert second.researched is False
        assert second.supporting_yes == []

    def test_v3_hands_the_object_over(self):
        import inspect
        from ptai.agent import v3_loop
        src = inspect.getsource(v3_loop)
        assert 'context["research_result"] = research_result' in src, (
            "V3 must pass the ResearchResult, or the contradiction engine's own "
            "`researched` state stays False and it discards the evidence")
        assert 'context["research_result"] = None' in src


# --------------------------------------------------------------------------
# 4. X unavailable is said out loud
# --------------------------------------------------------------------------

class TestXIsExplicitWhenAbsent:
    def test_a_blocked_scraper_is_reported_as_unavailable(self):
        forecast = XModel().forecast(
            market(), sentiment_result={"score": 0}, tweets=[],
            status="blocked_circuit_breaker",
            unavailable_reason="X scraper is blocked (circuit breaker open)")
        assert forecast.confidence == 0.0
        assert "X UNAVAILABLE" in forecast.reasoning
        assert "blocked_circuit_breaker" in forecast.reasoning
        assert "no sentiment evidence" in forecast.reasoning.lower()

    def test_the_ensemble_passes_the_status_through(self):
        result = EnsembleForecaster(llm_router=None).forecast_market(
            market(), context={"x_status": "blocked_circuit_breaker",
                               "x_unavailable_reason": "circuit breaker open"})
        row = [c for c in result.components if c["model"] == "x_sentiment"][0]
        assert row["contributes"] is False
        assert "UNAVAILABLE" in row["reasoning"]

    def test_v3_records_a_reason_on_every_x_path(self):
        import inspect
        from ptai.agent import v3_loop
        src = inspect.getsource(v3_loop)
        for anchor in ('context["x_unavailable_reason"] = (\n                                "X scraper is blocked',
                       'context["x_status"] = "no_engine"',
                       'context["x_status"] = "failed"'):
            assert anchor in src, f"missing X availability record: {anchor[:40]}"


# --------------------------------------------------------------------------
# 5. base rates are counted frequencies, not constants
# --------------------------------------------------------------------------

def _closed(category: str, yes: int, n: int, price: float = 0.45):
    rows = []
    for i in range(n):
        rows.append({"category": category,
                     "outcomePrices": '["1", "0"]' if i < yes else '["0", "1"]',
                     "lastTradePrice": price})
    return rows


class TestBaseRatesAreRealFrequencies:
    def test_a_category_with_a_sample_becomes_a_prior(self):
        book = BaseRateBook(min_sample=30)
        book.build_from_markets(_closed("politics", yes=12, n=40))
        prior = book.prior_for("politics")
        assert prior["rate"] == pytest.approx(0.30)
        assert prior["n"] == 40
        assert prior["mean_last_price"] == pytest.approx(0.45)

    def test_a_thin_category_stays_unknown(self):
        book = BaseRateBook(min_sample=30)
        book.build_from_markets(_closed("sports", yes=4, n=5))
        assert book.prior_for("sports") is None, (
            "five markets is not a base rate; the model must keep saying so")
        assert book.status()["available"] is False

    def test_unreadable_settlements_are_skipped_not_counted_as_no(self):
        book = BaseRateBook(min_sample=30)
        book.build_from_markets([
            {"category": "politics", "outcomePrices": '["1", "0"]'},
            {"category": "politics", "outcomePrices": '["0.5", "0.5"]'},
            {"category": "politics"},
        ])
        stats = book.dataset["categories"]["politics"]
        assert stats["n"] == 1 and stats["yes"] == 1
        assert book.dataset["markets_skipped"] == 2

    def test_the_model_uses_the_prior_and_caps_its_weight(self):
        model = BaseRateModel()
        forecast = model.forecast(market(), "politics", prior={
            "rate": 0.38, "n": 412, "mean_last_price": 0.45,
            "source": "test", "built_at": "2026-09-27"})
        assert forecast.probability == pytest.approx(0.38)
        assert 0 < forecast.confidence <= 0.35
        assert "412" in forecast.reasoning
        assert "prior" in forecast.reasoning.lower()

    def test_no_prior_means_no_weight(self):
        forecast = BaseRateModel().forecast(market(), "politics", prior=None)
        assert forecast.confidence == 0.0
        assert "no historical data" in forecast.reasoning

    def test_the_dataset_survives_a_restart(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "b.db"))
        try:
            book = BaseRateBook(storage=storage)
            book.build_from_markets(_closed("politics", yes=12, n=40))
            assert book.save() is True
            reloaded = BaseRateBook(storage=storage)
            assert reloaded.prior_for("politics")["n"] == 40
            assert reloaded.status()["available"] is True
        finally:
            storage.close()

    def test_an_unusable_rate_never_reaches_the_model(self):
        """A rate with no count behind it is not a frequency."""
        forecast = BaseRateModel().forecast(market(), "politics",
                                            prior={"rate": 0.9, "n": 0})
        assert forecast.confidence == 0.0


# --------------------------------------------------------------------------
# 6. the orderbook: order, identity, and sanity
# --------------------------------------------------------------------------

class TestTheBookIsNormalisedBeforeItIsTrusted:
    # The operator's book, verbatim: worst-first bids, worst-first asks, on a
    # market that is quoted around the middle.
    OPERATOR_BOOK = {
        "asset_id": "TOKEN-YES",
        "market": "COND-1",
        "bids": [{"price": "0.01", "size": "100"}, {"price": "0.48", "size": "500"}],
        "asks": [{"price": "0.99", "size": "100"}, {"price": "0.52", "size": "300"}],
    }

    def test_the_best_quote_is_found_regardless_of_listing_order(self):
        book = normalise_book(self.OPERATOR_BOOK, expected_token_id="TOKEN-YES",
                              expected_condition_id="COND-1", reference_price=0.50)
        assert book["best_bid"] == pytest.approx(0.48)
        assert book["best_ask"] == pytest.approx(0.52)
        assert book["spread"] == pytest.approx(0.04), (
            "the 98% spread the operator saw was the API's list order, not the "
            "market: index 0 was the worst quote on both sides")
        assert book["validated"] is True
        assert "unsorted_bids" in book["validation"]

    def test_a_book_for_a_different_token_is_refused(self):
        book = normalise_book(self.OPERATOR_BOOK, expected_token_id="OTHER")
        assert book["validated"] is False
        assert "NOT the token requested" in book["validation"]["identity"]

    def test_a_stale_book_is_refused_with_the_reason(self):
        book = normalise_book(self.OPERATOR_BOOK, expected_token_id="TOKEN-YES",
                              reference_price=0.90)
        assert book["validated"] is False
        assert "stale" in book["validation"]["reference"]

    def test_a_crossed_book_is_refused(self):
        book = normalise_book({"bids": [{"price": "0.60", "size": "10"}],
                               "asks": [{"price": "0.55", "size": "10"}]})
        assert book["validated"] is False
        assert "stale or one-sided" in book["validation"]["shape"]

    def test_impossible_levels_are_dropped(self):
        book = normalise_book({"bids": [{"price": "0", "size": "5"},
                                        {"price": "1", "size": "5"},
                                        {"price": "0.4", "size": "0"}],
                               "asks": [{"price": "0.6", "size": "5"}]})
        assert book["best_bid"] is None and book["best_ask"] == pytest.approx(0.6)
        assert book["validated"] is False

    def test_the_executable_price_is_named_per_side(self):
        """
        Which side of the book you pay is a property of the side you trade, not
        of the market's own price. The old line picked the ask when the market
        was above 0.5 and the bid when it was below.
        """
        import inspect
        from ptai.venues.polymarket_adapter import PolymarketAdapter
        src = inspect.getsource(PolymarketAdapter.get_orderbook)
        assert "executable_price_yes" in src and "executable_price_no" in src
        assert "market.yes_price > 0.5 else best_bid" not in src


# --------------------------------------------------------------------------
# 7. no market gets 70 seconds of model time unless it earned it
# --------------------------------------------------------------------------

class TestTheTwoStageScan:
    """
    One LLM call per market, at 60-75 seconds each, cannot fit a 10-minute
    cycle. The cycle reads every book cheaply, ranks on what it can measure, and
    spends X, web research and the LLM on the shortlist only - with the rest
    priced on their measured book and SAID to be priced that way.
    """

    @pytest.fixture()
    def agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
        from ptai.agent.v3_loop import TradingAgentV3
        a = TradingAgentV3(country_code="UG", dry_run=True)
        yield a
        a.storage.close()

    def _priced(self, mid, spread, liquidity=20000, volume=5000):
        m = market(mid, 0.5)
        m.liquidity = liquidity
        m.volume_24h = volume
        book = {"validated": True, "spread": spread, "depth": 5000.0}
        return m, book

    def test_a_tight_book_scores_above_a_wide_one(self, agent):
        tight, tight_book = self._priced("TIGHT", 0.01)
        wide, wide_book = self._priced("WIDE", 0.09)
        s_tight, _ = agent._screen_score(tight, tight_book)
        s_wide, _ = agent._screen_score(wide, wide_book)
        assert s_tight > s_wide

    def test_an_unvalidated_book_never_reaches_the_shortlist(self, agent):
        score, why = agent._screen_score(market("BAD", 0.5), {
            "validated": False,
            "validation": {"identity": "the book is another token's"}})
        assert score < 0, "a book we cannot confirm must not earn model time"
        assert "another token" in why

    def test_a_spread_too_wide_to_trade_is_screened_out_with_the_reason(self, agent):
        score, why = agent._screen_score(
            market("WIDE", 0.5),
            {"validated": True, "spread": 0.98, "depth": 5000.0})
        assert score < 0
        assert "wider than" in why

    def test_without_a_screen_every_market_is_analysed_in_full(self, agent):
        """A direct caller outside the cycle keeps the old behaviour."""
        deep, why = agent._deep_analysis_for(market("M1", 0.5))
        assert deep is True
        assert "no screen" in why

    def test_only_the_shortlist_is_deep_analysed(self, agent):
        agent._deep_shortlist_active = True
        agent._screen = {"considered": 40, "screened_out": 38, "limit": 2,
                         "shortlist": [{"market_id": "KEEP", "why": "tight book"}]}
        agent._deep_market_ids = {"KEEP"}
        deep, why = agent._deep_analysis_for(market("KEEP", 0.5))
        assert deep is True and "deep shortlist" in why
        shallow, why2 = agent._deep_analysis_for(market("SKIP", 0.5))
        assert shallow is False
        assert "not in this cycle's deep shortlist" in why2, (
            "a screened-out market must carry the decision, or 'no LLM opinion' "
            "is indistinguishable from a model that failed")

    def test_the_screen_ranks_real_books_and_keeps_a_shortlist(self, agent):
        """Drive the real prescan with a stubbed adapter."""
        import asyncio

        class FakeAdapter:
            def __init__(self, books):
                self.books = books

            async def get_orderbook(self, m):
                return self.books.get(m.id, {"validated": False})

        markets = [market("A", 0.5), market("B", 0.5), market("C", 0.5)]
        for m in markets:
            m.volume_24h, m.liquidity = 20000, 20000
        books = {
            "A": {"validated": True, "spread": 0.01, "depth": 9000.0},
            "B": {"validated": True, "spread": 0.05, "depth": 3000.0},
            "C": {"validated": False, "validation": {"identity": "wrong token"}},
        }
        adapter = FakeAdapter(books)
        agent.venue_registry.get_adapter_for_market = lambda m: adapter
        agent.deep_analysis_limit = 2

        screen = asyncio.run(agent._prescan_and_rank(markets))
        assert screen["considered"] == 3
        assert screen["books_read"] == 3
        assert [row["market_id"] for row in screen["shortlist"]] == ["A", "B"]
        assert screen["screened_out"] == 1
        assert agent._deep_market_ids == {"A", "B"}
        assert agent._cycle_books["C"]["validated"] is False, (
            "every book read in the cheap pass must be reusable by the pricing "
            "pass rather than fetched twice")
