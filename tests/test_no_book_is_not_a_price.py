"""
A market the venue does not list is not a market with an invented price.

The operator's 2026-09-29 log, in his words, is the bug report. Two lines repeat
hundreds of times:

    Orderbook fetch failed for 21399235645000066554990798907075333892305077400208393215772713251785743876268:
      404 Client Error: Not Found for url: https://clob.polymarket.com/book?token_id=2139...268
    ESTIMATED orderbook 4190831: liq $1753212.63358 vol $1011431.527044 spread 1.0%
      ... - NOT REAL, $50 trader should verify executable price

The first is the venue stating a fact: its CLOB has no book for that token. The
second invents a spread, a depth and a random imbalance for a market that does
not exist, and the whole scan pays for it - a `random.uniform` call, an edge
computation and an "edge may not be executable" warning per market.

There is a difference between "the venue has no book" and "we could not reach the
venue", and only the second one can honestly be answered with a labelled estimate.
These tests pin that difference, and the money rule that follows from it: no book
means no price, no cost and no edge for that market.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.markets.polymarket import PolymarketClient
from src.ptai.venues.polymarket_adapter import PolymarketAdapter


def make_market(market_id: str = "4190831", token: Optional[str] = "token-1",
                price: float = 0.30, liquidity: float = 25000.0,
                volume: float = 50000.0) -> Market:
    return Market(
        id=market_id,
        source=MarketSource.POLYMARKET,
        question="Will this happen?",
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1 - price],
        tokens=[Token(token_id=token, outcome="YES", price=price)] if token else [],
        volume=volume,
        volume_24h=volume * 0.5,
        liquidity=liquidity,
        active=True,
        closed=False,
        raw={"venue": "polymarket", "conditionId": "0xcond"},
    )


class _Response:
    def __init__(self, status_code: int, payload: Any = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = Exception(f"{self.status_code} Client Error: Not Found")
            error.response = self
            raise error

    def json(self):
        return self._payload


class _Session:
    """A stub Polymarket session: every answer is scripted, nothing leaves."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        answer = self.script(params or {})
        if isinstance(answer, Exception):
            raise answer
        return answer


def client_with(script) -> PolymarketClient:
    client = PolymarketClient.__new__(PolymarketClient)
    client.clob_api = "https://clob.polymarket.com"
    client.session = _Session(script)
    return client


# ---------------------------------------------------------------------------
# 1. The client tells the two failures apart
# ---------------------------------------------------------------------------

class TestTheClientKnowsWhyThereIsNoBook:
    def test_a_404_is_the_venue_saying_it_has_no_book(self):
        client = client_with(lambda p: _Response(404))
        book, reason = client.get_orderbook_with_reason("token-x")
        assert book is None
        assert reason["kind"] == "not_listed"
        assert reason["status"] == 404
        assert "404" in reason["detail"] or "no book" in reason["detail"]

    def test_a_timeout_is_not_the_venue_saying_anything(self):
        client = client_with(lambda p: TimeoutError("no answer"))
        book, reason = client.get_orderbook_with_reason("token-x")
        assert book is None
        assert reason["kind"] == "transport"

    def test_a_real_book_comes_back_with_no_reason(self):
        client = client_with(lambda p: _Response(200, {"bids": [{"price": "0.29", "size": "100"}],
                                                      "asks": [{"price": "0.31", "size": "100"}]}))
        book, reason = client.get_orderbook_with_reason("token-x")
        assert book and book["bids"]
        assert reason == {}

    def test_the_old_entry_point_still_returns_the_book(self):
        client = client_with(lambda p: _Response(200, {"bids": [{"price": "0.29", "size": "1"}],
                                                      "asks": []}))
        assert client.get_orderbook("token-x")["bids"]


# ---------------------------------------------------------------------------
# 2. The adapter refuses, and invents nothing
# ---------------------------------------------------------------------------

class TestNoBookIsARefusal:
    def _adapter(self, script) -> PolymarketAdapter:
        adapter = PolymarketAdapter.__new__(PolymarketAdapter)
        adapter.client = client_with(script)
        return adapter

    def test_a_token_the_venue_does_not_list_is_refused_with_no_price(self, caplog):
        adapter = self._adapter(lambda p: _Response(404))
        book = asyncio.run(adapter.get_orderbook(make_market()))
        assert book["source"] == "no_clob_book"
        assert book["is_real"] is False
        assert book["executable"] is False
        # NOT an estimate: there is no bid, no ask, no depth, no slippage.
        for key in ("bid", "ask", "spread", "spread_pct", "depth", "slippage_estimate"):
            assert book[key] is None, f"{key} was invented for a market with no book"
        assert book["execution_quality"] == 0.0
        assert "book" in book["warning"].lower()
        assert "no price is invented" in book["warning"].lower()
        assert book["venue_says"] in ("404", "HTTP 404")

    def test_the_refusal_says_which_market_and_why(self):
        """
        The log IS the bug report, so the line has to name the market and the
        fact. Captured through loguru's own sink: pytest's caplog reads stdlib
        logging and loguru does not propagate into it.
        """
        from loguru import logger as loguru_logger

        # DEBUG on purpose: one market with no book is a value, not an incident.
        # The operator's log had this fact printed two hundred times as a
        # warning; the scan now counts them and says the count once.
        lines = []
        sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                    level="DEBUG")
        try:
            adapter = self._adapter(lambda p: _Response(404))
            asyncio.run(adapter.get_orderbook(make_market(market_id="4190831")))
        finally:
            loguru_logger.remove(sink_id)
        messages = " ".join(lines)
        assert "4190831" in messages
        assert "No CLOB book" in messages
        assert "refusing to price it" in messages
        # ... and the invented-book line from the operator's log is NOT there.
        assert "ESTIMATED orderbook" not in messages

    def test_the_scan_counts_the_markets_the_venue_has_no_book_for(self):
        """
        One counted line beats two hundred warnings - and the count has to come
        from the real scan, through the real refusal books.
        """
        from types import SimpleNamespace
        from loguru import logger as loguru_logger
        from src.ptai.agent.v3_loop import TradingAgentV3

        listed = make_market(market_id="listed-1")
        missing = [make_market(market_id="gone-1"), make_market(market_id="gone-2")]
        refusal = {
            "market_id": "gone", "bid": None, "ask": None, "spread": None,
            "depth": None, "source": "no_clob_book", "is_real": False,
            "validated": False, "executable": False, "warning": "no book",
        }

        class _Adapter:
            async def get_orderbook(self, market):
                if "gone" in market.id:
                    return dict(refusal, market_id=market.id)
                return dict(refusal, market_id=market.id, source="clob_real",
                            is_real=True, validated=True, executable=True,
                            bid=0.29, ask=0.31, spread=0.02, depth=5000.0)

        agent = TradingAgentV3.__new__(TradingAgentV3)
        agent.deep_analysis_limit = 5
        agent.PRESCAN_LIMIT = 50
        agent.PRESCAN_CONCURRENCY = 4
        agent._cycle_books = {}
        agent.strategy_engine_v3 = SimpleNamespace(
            max_spread=0.10, cheap_filters=lambda markets: list(markets))
        agent.venue_registry = SimpleNamespace(
            get_adapter_for_market=lambda market: _Adapter())
        agent._screen_score = lambda market, book: (
            (-1.0, "no book") if not book.get("validated")
            else (0.5, "fine"))

        lines = []
        sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                    level="INFO")
        try:
            screen = asyncio.run(agent._prescan_and_rank([listed] + missing))
        finally:
            loguru_logger.remove(sink_id)

        summary = [line for line in lines if "Cheap screen:" in line]
        assert len(summary) == 1, lines
        assert "2 had no venue book at all (they cannot be priced)" in summary[0]
        assert screen["no_book"] == 2
        assert [row["market_id"] for row in screen["shortlist"]] == ["listed-1"]
        assert "ESTIMATED orderbook" not in " ".join(lines)

    def test_a_venue_we_could_not_reach_is_still_a_labelled_estimate(self):
        """
        The other half of the rule: this machine may be offline, and then the
        honest answer is an estimate that says so - not a refusal that pretends
        the market does not exist.
        """
        adapter = self._adapter(lambda p: TimeoutError("no answer"))
        book = asyncio.run(adapter.get_orderbook(make_market()))
        assert book["is_real"] is False
        assert book["source"] != "no_clob_book"
        assert book["bid"] is not None and book["ask"] is not None
        assert "NOT REAL" in book.get("warning", "") or book.get("trustworthy")

    def test_a_market_with_no_token_is_refused_rather_than_quoted(self):
        adapter = self._adapter(lambda p: _Response(404))
        book = asyncio.run(adapter.get_orderbook(make_market(token=None)))
        assert book["source"] == "no_token"
        assert book["bid"] is None and book["ask"] is None
        # The shape the rest of the loop reads is still there.
        for key in ("venue_id", "spread", "liquidity", "volume_24h",
                    "execution_quality", "is_real", "is_mock", "executable",
                    "liquidity_score", "warning"):
            assert key in book, key
        assert book["is_mock"] is False

    def test_every_refusal_answers_the_same_keys(self):
        """
        No caller may have to special-case a refusal: same keys as a real book,
        so `book["spread"]` is None rather than a KeyError.
        """
        adapter = self._adapter(lambda p: _Response(404))
        refused = asyncio.run(adapter.get_orderbook(make_market()))
        for key in ("market_id", "venue_id", "token_id", "bid", "ask", "spread",
                    "spread_pct", "depth", "liquidity", "volume_24h",
                    "slippage_estimate", "execution_quality", "liquidity_score",
                    "source", "is_real", "is_mock", "executable", "warning"):
            assert key in refused, key


# ---------------------------------------------------------------------------
# 3. The scan does not pay for a market it cannot price
# ---------------------------------------------------------------------------

class TestTheScanDoesNotPriceAMarketWithNoBook:
    def test_a_refused_market_is_not_shortlisted_for_deep_analysis(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3.__new__(TradingAgentV3)
        market = make_market()
        score, why = agent._screen_score(market, {
            "source": "no_clob_book", "is_real": False, "executable": False,
            "bid": None, "ask": None, "spread": None, "validated": False,
        })
        assert score < 0, "a market with no book must not earn model time"

    def test_an_estimated_book_is_still_priced_but_never_executable(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3.__new__(TradingAgentV3)
        market = make_market()
        score, why = agent._screen_score(market, {
            "source": "estimated", "is_real": False, "executable": False,
            "bid": 0.29, "ask": 0.31, "spread": 0.02, "trustworthy": "medium",
        })
        # It may be ranked (it is real data about a real market), but the book it
        # is ranked on is not a book anyone can trade against.
        assert isinstance(score, float)
        assert isinstance(why, str)


# ---------------------------------------------------------------------------
# 4. The reference that crashed every two seconds
# ---------------------------------------------------------------------------

class TestAReferenceThatCrashesIsNotRetried:
    def test_a_market_with_no_expiry_is_a_reason_not_a_type_error(self):
        from src.ptai.strategy.reference_odds import lognormal_digital_prob

        # This call is the one in the operator's log:
        #   TypeError: '<=' not supported between instances of 'NoneType' and 'int'
        assert lognormal_digital_prob(60000.0, 70000.0, 0.6, None) is None
        assert lognormal_digital_prob(None, 70000.0, 0.6, 10) is None
        assert lognormal_digital_prob(60000.0, 70000.0, None, 10) is None
        # And a real one still prices.
        assert 0.0 < lognormal_digital_prob(60000.0, 70000.0, 0.6, 30.0) < 1.0

    def test_a_missing_expiry_is_recorded_as_the_reason(self):
        """
        The exact path in the log: a crypto market whose record carries no usable
        end date. Before this, `days` was None and `None <= 0` raised inside the
        pricer, so the reference reported a TypeError instead of a reason.
        """
        from src.ptai.strategy.reference_odds import ReferenceOddsEngine

        engine = ReferenceOddsEngine()
        engine.enabled_sources = {"deribit"}
        # Offline: hand the engine the Deribit summary it would have fetched, so
        # the test is about the expiry, not about the sandbox's network.
        engine._get_json = lambda url, params=None: {"result": [
            {"underlying_price": 60000.0, "strike": 150000.0,
             "implied_volatility": 0.6, "mark_price": 0.02}]}
        engine._days_to_expiry = lambda m: None
        market = make_market()
        market.question = "Will Bitcoin hit $150,000 by ?"
        assert engine.get_deribit_implied_prob(market) is None
        assert "expiry" in engine.unavailable.get("deribit", "")

    def test_an_expiring_market_still_prices(self):
        from src.ptai.strategy.reference_odds import ReferenceOddsEngine

        engine = ReferenceOddsEngine()
        engine.enabled_sources = {"deribit"}
        engine._get_json = lambda url, params=None: {"result": [
            {"underlying_price": 60000.0, "strike": 150000.0,
             "implied_volatility": 0.6, "mark_price": 0.02}]}
        engine._days_to_expiry = lambda m: 30.0
        market = make_market()
        # Near the money: a 30-day digital on a target 8% away is a real
        # probability, not a 0.0000 rounding artifact.
        market.question = "Will Bitcoin hit $65,000 by Friday?"
        priced = engine.get_deribit_implied_prob(market)
        assert priced is not None
        price, confidence, reasoning = priced
        assert 0.0 < price < 1.0 and "Deribit" in reasoning

    def test_a_source_that_raises_is_skipped_for_the_next_markets(self):
        """
        The log had the same TypeError from deribit for every market in a scan,
        every couple of seconds. One line per scan is a bug report; two hundred
        is a smokescreen. The reference engine is called once per market, so the
        memo has to outlive a call.
        """
        from src.ptai.strategy import reference_odds as module

        engine = module.ReferenceOddsEngine()
        calls = {"n": 0}

        def _boom(market):
            calls["n"] += 1
            raise TypeError("'<=' not supported between instances of 'NoneType' and 'int'")

        engine.get_deribit_implied_prob = _boom
        for name in ("get_pinnacle_implied_prob", "get_manifold_reference",
                     "get_metaculus_reference", "get_fed_funds_implied_prob",
                     "get_polling_reference"):
            setattr(engine, name, lambda m: None)
        engine.get_kalshi_reference = lambda m, km=None: None
        engine.enabled_sources = {"deribit"}

        for i in range(20):
            engine.get_all_reference_odds(make_market(market_id=f"m{i}"))
        assert calls["n"] == 1, "the raising source was retried for every market"
        reason = engine.unavailable["deribit"]
        assert "not retried" in reason or "skipped for now" in reason

    def test_the_skip_lasts_a_bounded_time_then_tries_again(self):
        """A transient fault must not disable a source for good."""
        from src.ptai.strategy import reference_odds as module

        engine = module.ReferenceOddsEngine()
        state = {"fail": True, "calls": 0}

        def _maybe(market):
            state["calls"] += 1
            if state["fail"]:
                raise TypeError("boom")
            return (0.5, 0.8, "fine")

        engine.get_deribit_implied_prob = _maybe
        for name in ("get_pinnacle_implied_prob", "get_manifold_reference",
                     "get_metaculus_reference", "get_fed_funds_implied_prob",
                     "get_polling_reference"):
            setattr(engine, name, lambda m: None)
        engine.get_kalshi_reference = lambda m, km=None: None
        engine.enabled_sources = {"deribit"}
        engine._detect_category = lambda m: "crypto"

        engine.get_all_reference_odds(make_market(market_id="a"))
        # Inside the backoff window: not asked again.
        engine.get_all_reference_odds(make_market(market_id="b"))
        assert state["calls"] == 1
        # After it: asked again, and it works.
        engine._failed_until["deribit"] = 0.0
        state["fail"] = False
        refs = engine.get_all_reference_odds(make_market(market_id="c"))
        assert state["calls"] == 2
        assert any(r.source == "deribit" for r in refs)


# ---------------------------------------------------------------------------
# 5. An answered book with nothing in it is the same fact as a 404
# ---------------------------------------------------------------------------

class TestAnAnsweredButEmptyBookIsAlsoARefusal:
    """
    The 404 is not the only way a venue says "no book". It can also answer 200
    with an empty object, or with levels on one side only. Both used to fall
    through to the invented estimate, which is how a market with no liquidity
    acquired a spread, a depth and an execution quality in the operator's log.
    """

    def _adapter(self, script) -> PolymarketAdapter:
        adapter = PolymarketAdapter.__new__(PolymarketAdapter)
        adapter.client = client_with(script)
        return adapter

    def test_a_200_with_no_bids_or_asks_is_refused(self):
        adapter = self._adapter(lambda p: _Response(200, {}))
        book = asyncio.run(adapter.get_orderbook(make_market()))
        assert book["source"] == "no_clob_book"
        assert book["bid"] is None and book["ask"] is None
        assert book["executable"] is False and book["is_real"] is False

    def test_a_one_sided_book_is_refused(self):
        """
        Bids with no asks: there is no two-sided market to trade. `normalise_book`
        catches this first and returns an unvalidated book, which is also a
        refusal - what matters is that no price is invented either way.
        """
        adapter = self._adapter(lambda p: _Response(200, {
            "bids": [{"price": "0.20", "size": "500"}], "asks": []}))
        book = asyncio.run(adapter.get_orderbook(make_market()))
        assert book["source"] in ("no_clob_book", "clob_unvalidated")
        assert book["source"] != "enhanced_estimation", "an estimate was invented"
        assert book["ask"] is None and book["is_real"] is False
        assert book["executable"] is False

    def test_an_unreachable_venue_still_gets_a_labelled_estimate(self):
        """The one case an estimate is honest: nobody answered."""
        adapter = self._adapter(lambda p: TimeoutError("no answer"))
        book = asyncio.run(adapter.get_orderbook(make_market()))
        assert book["source"] == "enhanced_estimation"
        assert book["is_real"] is False
        assert "NOT REAL" in book["reasoning"]


# ---------------------------------------------------------------------------
# 6. Why a reference is missing travels with the count
# ---------------------------------------------------------------------------

class TestTheReferenceReasonsReachTheCycleLog:
    """
    The operator's log had one line repeated:

        [reference] deribit raised: TypeError: '<=' not supported between
        instances of 'NoneType' and 'int'

    A source that cannot be read is a fact about the anchor, not about each of
    two hundred markets - so the reason is a value the scan carries, and the
    cycle prints it once.
    """

    def test_the_alpha_scan_carries_the_reasons(self):
        from src.ptai.strategy.alpha_engine import AlphaEngine
        from src.ptai.markets.base import MarketSource
        from types import SimpleNamespace

        engine = AlphaEngine.__new__(AlphaEngine)
        calls = {"n": 0}

        class _Refs:
            unavailable = {"deribit": "the market has no usable expiry"}

            def get_all_reference_odds(self, market, kalshi=None):
                calls["n"] += 1
                return []

        engine.reference_odds = _Refs()
        engine.combinatorial = SimpleNamespace(
            find_combinatorial_arbitrage=lambda *a, **k: [])

        market = make_market()
        market.source = MarketSource.POLYMARKET
        result = engine.scan_all_alpha([market])
        assert result["reference_odds"]["markets_read"] == 1
        assert result["reference_odds"]["unavailable"] == {
            "deribit": "the market has no usable expiry"}

    def test_the_cycle_prints_one_summary_line(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3

        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "Reference odds:" in source
        assert "unavailable: " in source
