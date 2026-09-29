"""
A pair of similar questions is not an arbitrage.

The operator's 2026-09-29 log, eleven times, in the same cycle:

    ARBITRAGE FOUND: Same event confidence 0.82: 'Will Sarah Knafo win the 2027
      French presidential election?' vs 'Who will win the next French
      presidential election?' | Price 0.009 vs 0.590 spread 0.581 | Cost 0.420
      profit 0.581 (138.4%) adjusted 112.0% | Trade: True

and then the same finding, advertised as the cycle's best opportunity (edge
1.323, netEV $1218.82 on a $1.00 stake), reached execution and died:

    ERROR ... Adapter for venue_id marketsource.polymarket not found
    ERROR ... ABORT: opportunity venue_id MarketSource.POLYMARKET+MarketSource.
      PREDICTIT_arb does not match adapter polymarket - exact routing required
    ERROR ... Arb lane could not build the legs of ...: 'ArbitrageOpportunity'
      object has no attribute 'fee_adjusted_profit'

These tests pin the three rules that follow: an arb needs the same market and
executable prices on both legs; a two-leg pair is never sent down the
single-venue path; and no number that cannot be paid is reported as an edge.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.strategy.arbitrage import ArbitrageEngine

END = datetime(2027, 4, 11, tzinfo=timezone.utc)


def make_market(market_id: str, question: str, source: MarketSource,
                price: float = 0.30, end=END) -> Market:
    return Market(
        id=market_id,
        source=source,
        question=question,
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1 - price],
        tokens=[Token(token_id=f"t-{market_id}", outcome="YES", price=price)],
        volume=100000.0,
        volume_24h=25000.0,
        liquidity=25000.0,
        active=True,
        closed=False,
        end_date=end,
        raw={"venue": source.value if hasattr(source, "value") else str(source)},
    )


def book(bid: float, ask: float, *, real: bool = True) -> Dict[str, Any]:
    return {"bid": bid, "ask": ask, "is_real": real, "validated": real,
            "source": "clob_real" if real else "enhanced_estimation"}


def pair(kalshi_book=None, poly_book=None, *, q1=None, q2=None,
         fee: Optional[float] = 0.0):
    """The two legs of the operator's log, with books supplied per leg."""
    q1 = q1 or "Will Sarah Knafo win the 2027 French presidential election?"
    q2 = q2 or "Who will win the next French presidential election?"
    a = make_market("a-1", q1, MarketSource.POLYMARKET, price=0.009)
    b = make_market("b-1", q2, MarketSource.PREDICTIT, price=0.590)
    books = {"a-1": poly_book, "b-1": kalshi_book}
    engine = ArbitrageEngine(min_spread=0.03, min_confidence_same_event=0.6)
    result = engine.find_arbitrage(
        [a, b],
        book_lookup=lambda mid: books.get(mid),
        fee_rate_lookup=(lambda _market: fee),
    )
    return engine, result[0]


# ---------------------------------------------------------------------------
# 1. The pair in the log is not an arbitrage, and is not called one
# ---------------------------------------------------------------------------

class TestTheLogPairIsNotAnArbitrage:
    def test_similar_questions_are_not_the_same_market(self):
        _engine, opp = pair(book(0.55, 0.60), book(0.30, 0.35))
        assert opp.should_trade is False
        assert opp.identity_verified is False
        assert "not the same market" in opp.blocked_reason
        # The mid-price fantasy is kept as research, never as the trade's number.
        assert opp.estimated_profit_pct == 0.0
        assert opp.indicative_profit_pct > 1.0
        assert "indicative" in opp.reasoning

    def test_the_log_line_the_operator_reported_is_not_emitted(self):
        from loguru import logger as loguru_logger

        lines: List[str] = []
        sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                    level="DEBUG")
        try:
            _engine, _opp = pair(book(0.55, 0.60), book(0.30, 0.35))
        finally:
            loguru_logger.remove(sink_id)
        assert not [line for line in lines if "ARBITRAGE FOUND" in line]
        assert [line for line in lines if "Arbitrage scan:" in line]


# ---------------------------------------------------------------------------
# 2. Executable prices, not quoted prices
# ---------------------------------------------------------------------------

class TestAnArbNeedsExecutablePrices:
    def test_no_book_on_a_leg_is_a_refusal_even_with_identical_questions(self):
        _engine, opp = pair(None, None, q1="Will X win the 2027 election?",
                            q2="Will X win the 2027 election?")
        assert opp.identity_verified is True
        assert opp.should_trade is False
        assert "no validated executable book" in opp.blocked_reason
        assert opp.estimated_profit_pct == 0.0

    def test_a_cost_above_the_payout_is_refused_with_the_cost_named(self):
        # Same market, real books: buy YES at 0.55, buy NO at 1 - 0.45 = 0.55.
        _engine, opp = pair(book(0.45, 0.50), book(0.55, 0.60),
                            q1="Will X win the 2027 election?",
                            q2="Will X win the 2027 election?")
        assert opp.should_trade is False
        assert opp.executable_price_a is not None and opp.executable_price_b is not None
        assert opp.cost >= 1.0
        assert "leaves nothing" in opp.blocked_reason

    def test_a_two_sided_real_book_pair_is_tradeable_and_logged(self):
        from loguru import logger as loguru_logger

        lines: List[str] = []
        sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                    level="INFO")
        try:
            _engine, opp = pair(book(0.62, 0.66), book(0.40, 0.44),
                                q1="Will X win the 2027 election?",
                                q2="Will X win the 2027 election?")
        finally:
            loguru_logger.remove(sink_id)
        assert opp.should_trade is True
        assert opp.identity_verified and opp.executable
        # Poly's ask 0.44 (the cheap leg) + PredictIt's NO at 1 - 0.62 = 0.38.
        assert opp.cost == pytest.approx(0.82, abs=1e-9)
        assert opp.estimated_profit_pct == pytest.approx(0.219512, abs=1e-5)
        assert "SAME MARKET, EXECUTABLE" in opp.reasoning
        assert [line for line in lines if "ARBITRAGE FOUND" in line]

    def test_an_unreadable_venue_fee_is_a_refusal(self):
        a = make_market("a-1", "Will X win the 2027 election?",
                        MarketSource.POLYMARKET, price=0.40)
        b = make_market("b-1", "Will X win the 2027 election?",
                        MarketSource.PREDICTIT, price=0.60)
        engine = ArbitrageEngine(min_spread=0.03, min_confidence_same_event=0.6)
        opps = engine.find_arbitrage(
            [a, b], book_lookup=lambda mid: book(0.62, 0.66) if mid == "b-1" else book(0.40, 0.44),
            fee_rate_lookup=lambda _market: None)
        assert opps[0].should_trade is False
        assert "fee could not be read" in opps[0].blocked_reason


# ---------------------------------------------------------------------------
# 3. A pair never goes down the single-venue path
# ---------------------------------------------------------------------------

class TestAPairIsNotADirectionalOrder:
    def test_the_pair_id_is_routable_and_not_an_enum_repr(self):
        from src.ptai.strategy.arbitrage import ArbitrageEngine as Engine

        _engine, opp = pair(book(0.62, 0.66), book(0.40, 0.44),
                            q1="Will X win the 2027 election?",
                            q2="Will X win the 2027 election?")
        opps = Engine().to_venue_opportunities([opp])
        assert len(opps) == 1
        venue_opp = opps[0]
        assert venue_opp.venue_id == "polymarket+predictit"
        assert "MarketSource" not in venue_opp.venue_id
        # The directional path trades single markets; a pair is research here
        # and is executed only by the arbitrage lane, which builds both legs.
        assert venue_opp.should_trade is False
        assert venue_opp.raw.get("research_only") is True
        assert venue_opp.raw.get("pair") is True
        assert len(venue_opp.raw.get("legs") or []) == 2

    def test_a_refused_pair_produces_no_opportunity_at_all(self):
        from src.ptai.strategy.arbitrage import ArbitrageEngine as Engine

        _engine, opp = pair(book(0.55, 0.60), book(0.30, 0.35))
        assert Engine().to_venue_opportunities([opp]) == []

    def test_the_executor_can_build_the_legs_without_raising(self):
        """The log's AttributeError: 'fee_adjusted_profit' did not exist."""
        from src.ptai.execution.multi_venue_executor import MultiVenueExecutor

        _engine, opp = pair(book(0.62, 0.66), book(0.40, 0.44),
                            q1="Will X win the 2027 election?",
                            q2="Will X win the 2027 election?")
        executor = MultiVenueExecutor.__new__(MultiVenueExecutor)
        opp_a, opp_b = executor.arb_leg_opportunities(opp)
        assert opp_a.venue_id == "polymarket" and opp_b.venue_id == "predictit"
        assert opp_a.effective_edge == opp.fee_adjusted_profit
        assert opp_a.side != opp_b.side


# ---------------------------------------------------------------------------
# 4. A market with no real book is counted, not priced
# ---------------------------------------------------------------------------

class TestNoRealBookIsCountedNotPriced:
    def test_the_venue_scan_skips_markets_without_a_real_book(self):
        from loguru import logger as loguru_logger
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        engine = StrategyEngineV3.__new__(StrategyEngineV3)
        engine.cheap_filters = lambda markets: list(markets)
        engine.liquidity_filter = lambda markets: list(markets)
        called: List[str] = []

        def _evaluate(market, context=None):
            called.append(market.id)
            return []

        engine.evaluate_market_with_all_strategies = _evaluate

        markets = [make_market(f"c-{i}", f"Will crypto {i} close higher?",
                               MarketSource.POLYMARKET) for i in range(3)]
        lines: List[str] = []
        sink_id = loguru_logger.add(lambda m: lines.append(m.record["message"]),
                                    level="INFO")
        try:
            report, opps = asyncio.run(engine.scan_venue(
                "crypto_binance", markets, context_provider=None,
                book_lookup=lambda mid: book(0.40, 0.44, real=False)))
        finally:
            loguru_logger.remove(sink_id)
        assert called == []
        assert opps == []
        assert report.total_discovered == 3
        assert [line for line in lines if "not evaluated - their book is an estimate" in line]

    def test_a_market_with_a_real_book_is_evaluated(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        engine = StrategyEngineV3.__new__(StrategyEngineV3)
        engine.cheap_filters = lambda markets: list(markets)
        engine.liquidity_filter = lambda markets: list(markets)
        called: List[str] = []
        engine.evaluate_market_with_all_strategies = (
            lambda market, context=None: (called.append(market.id), [])[1])
        asyncio.run(engine.scan_venue(
            "polymarket", [make_market("p-1", "Will X win?", MarketSource.POLYMARKET)],
            book_lookup=lambda mid: book(0.40, 0.44)))
        assert called == ["p-1"]

    def test_the_best_opportunity_line_never_names_a_research_only_pair(self):
        import inspect
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        source = inspect.getsource(StrategyEngineV3.scan_all_venues)
        assert "_is_research_only" in source
        assert "next(" in source


# ---------------------------------------------------------------------------
# 5. An EV larger than the position can pay is refused
# ---------------------------------------------------------------------------

class TestAnImpossibleEVIsRefused:
    def _opportunity(self, price: float, fair: float, edge: float):
        from src.ptai.venues.adapter import VenueOpportunity, VenueType

        market = make_market("m-1", "Will X win?", MarketSource.POLYMARKET, price=price)
        opp = VenueOpportunity(
            market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION,
            side="YES", market_price=price, estimated_fair=fair,
            raw_edge=edge, effective_edge=edge, confidence=0.82, uncertainty=0.18,
            fees_pct=0.02, spread_pct=0.0, slippage_pct=0.0,
        )
        return opp

    def test_the_pair_that_produced_1218_dollars_of_ev_is_refused(self):
        """
        'fair' 0.610 is the OTHER venue's price and the pair was never
        executable; pricing it as a long shot at 0.0005 is where $1218.82 came
        from, and it was printed, ranked first, and pushed to execution.
        """
        from src.ptai.strategy.expected_ev import ExpectedNetEVEngine

        engine = ExpectedNetEVEngine()
        opp = self._opportunity(0.0005, 0.610, 1.323)
        opp.raw = {"research_only": True, "pair": True}
        result = engine.calculate(opportunity=opp, amount_usd=1.0, orderbook={})
        assert "research-only" in result.reasoning
        assert result.net_ev_usd <= 0
        assert result.should_trade is False

    def test_a_normal_opportunity_is_still_priced(self):
        from src.ptai.strategy.expected_ev import ExpectedNetEVEngine
        from src.ptai.venues.adapter import VenueOpportunity, VenueType

        engine = ExpectedNetEVEngine()
        market = make_market("m-2", "Will X win?", MarketSource.POLYMARKET, price=0.30)
        opp = VenueOpportunity(
            market=market, venue_id="polymarket", venue_type=VenueType.PREDICTION,
            side="YES", market_price=0.30, estimated_fair=0.55,
            raw_edge=0.25, effective_edge=0.25, confidence=0.82, uncertainty=0.18,
            fees_pct=0.02, spread_pct=0.0, slippage_pct=0.0)
        result = engine.calculate(opportunity=opp, amount_usd=1.0, orderbook={})
        assert "research-only" not in result.reasoning
        assert result.gross_ev_usd > 0

# ---------------------------------------------------------------------------
# 6. The agent hands the pair the books it read and the venue's own fee
# ---------------------------------------------------------------------------

class TestTheAgentFeedsThePairRealBooksAndRealFees:
    """
    The engine can only refuse or price a pair off what the cycle gives it.

    The wiring is the part that has failed before in this codebase (an engine
    written, tested, and connected to nothing), so it is pinned here: the scan
    is called with the cycle's books and with a fee reader that answers from
    the venue's own market info - and answers None, refusing the pair, when the
    venue cannot be read at all.
    """

    @staticmethod
    def _agent_stub(mechanics: Any = None, declared: Any = None,
                    adapter: bool = True):
        from types import SimpleNamespace
        from src.ptai.agent.v3_loop import TradingAgentV3

        class _Adapter:
            venue_id = "polymarket"
            capabilities = SimpleNamespace(fee_taker_pct=declared)

            def get_mechanics(self, *_args, **_kwargs):
                if mechanics is None:
                    raise RuntimeError("no clob market info")
                return mechanics

        class _Registry:
            def get_adapter_for_market(self, _market):
                return _Adapter() if adapter else None

        stub = SimpleNamespace(venue_registry=_Registry(), _arb_facts_cache={})
        stub._arb_venue_facts = lambda m: TradingAgentV3._arb_venue_facts(stub, m)
        return TradingAgentV3, stub

    def test_the_agents_fee_reader_is_the_venues_own_taker_fee(self):
        from types import SimpleNamespace
        agent_cls, stub = self._agent_stub(
            mechanics=SimpleNamespace(is_real=True, taker_fee_rate=0.0123,
                                      tick_size="0.01", neg_risk=True))
        market = make_market("m-1", "Will X win?", MarketSource.POLYMARKET)
        fee = agent_cls._arb_fee_rate(stub, market)
        assert fee["rate"] == pytest.approx(0.0123)
        assert "clob market info" in fee["source"]
        assert agent_cls._arb_neg_risk(stub, market) is True

    def test_an_unreadable_venue_fee_is_none_not_a_guess(self):
        agent_cls, stub = self._agent_stub(mechanics=None, declared=None)
        market = make_market("m-1", "Will X win?", MarketSource.POLYMARKET)
        assert agent_cls._arb_fee_rate(stub, market) is None

    def test_a_declared_capability_fee_says_where_it_came_from(self):
        agent_cls, stub = self._agent_stub(mechanics=None, declared=0.02)
        market = make_market("m-1", "Will X win?", MarketSource.POLYMARKET)
        fee = agent_cls._arb_fee_rate(stub, market)
        assert fee["rate"] == pytest.approx(0.02)
        assert "declared capability" in fee["source"]

    def test_the_cycle_scan_is_given_the_books_and_the_fee_reader(self):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3

        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "book_lookup=lambda mid: (getattr(self, \"_cycle_books\", {}) or {}).get(mid)" in source
        assert "fee_rate_lookup=self._arb_fee_rate" in source

    def test_a_priced_pair_reaches_the_arbitrage_lane_not_the_single_path(self):
        """
        End to end through the scan: the cycle's books + the venue's own fee
        make the pair tradeable, it lands in `arbitrage_opportunities` (which
        the arb lane reads), and the single-venue selection never sees it.
        """
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        engine = StrategyEngineV3.__new__(StrategyEngineV3)
        engine.arbitrage_engine = ArbitrageEngine(min_spread=0.03,
                                                 min_confidence_same_event=0.6)
        a = make_market("a-1", "Will X win the 2027 election?", MarketSource.POLYMARKET, price=0.60)
        b = make_market("b-1", "Will X win the 2027 election?", MarketSource.KALSHI, price=0.72)
        books = {"a-1": book(0.58, 0.62), "b-1": book(0.70, 0.75)}
        opps = engine.arbitrage_engine.find_arbitrage(
            [a, b], book_lookup=books.get,
            fee_rate_lookup=lambda _m: {"rate": 0.0, "source": "test"})
        assert len(opps) == 1 and opps[0].should_trade is True
        converted = engine.arbitrage_engine.to_venue_opportunities(opps)
        assert len(converted) == 1
        assert converted[0].venue_id == "polymarket+kalshi"
        # `not should_trade` is what keeps it out of final_selected - the list
        # the single-venue sizing loop and the exploration lane both walk.
        assert converted[0].should_trade is False
        assert converted[0].raw.get("executed_by", "").startswith("arbitrage lane")
