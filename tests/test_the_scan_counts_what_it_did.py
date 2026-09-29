"""
THE SCAN COUNTS WHAT IT DID, NOT WHAT THE SCREEN PLANNED.

The 16:53 cycle printed two numbers that could not both be true:

    Cheap screen: 200 market(s) read ..., 199 priced on the cheap context only
    Venue scan: ... 99 not evaluated - their book is an estimate or absent

and then a model line whose tail said "199 priced on their measured book
alone". The screen's 199 was a PLAN about model time; the scan priced the
markets it could and refused the rest on purpose. These tests pin the split:
the report carries what the scan actually did with each market, the screen's
numbers are labelled as a plan, and a market the scan will refuse cannot be
chosen for deep analysis in the first place.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.strategy.strategy_engine import StrategyEngineV3


def _market(market_id: str, **overrides) -> Market:
    fields = dict(id=market_id, source=MarketSource.POLYMARKET,
                  question=f"Will thing {market_id} happen?",
                  outcome_prices=[0.40, 0.60], volume_24h=100_000,
                  liquidity=20_000, venue_id="polymarket",
                  data_mode=DataMode.LIVE,
                  raw={"venue_id": "polymarket", "token_id": f"t{market_id}"})
    fields.update(overrides)
    return Market(**fields)


def _real_book(market_id: str):
    return {"market_id": market_id, "venue_id": "polymarket",
            "bids": [{"price": 0.40, "size": 500}],
            "asks": [{"price": 0.42, "size": 500}],
            "bid": 0.40, "ask": 0.42, "spread": 0.02, "depth": 20_000.0,
            "is_real": True, "is_mock": False, "validated": True,
            "executable": True, "executable_price": 0.42}


class _Ctx:
    async def get_context(self, market):
        return {"deep_analysis": True}


def _scan(engine, markets, book_lookup):
    async def _run():
        return await engine.scan_venue("polymarket", markets,
                                       context_provider=_Ctx(),
                                       book_lookup=book_lookup)
    return asyncio.run(_run())


class TestTheScanReportIsAHonestRecord:
    def test_markets_that_reached_the_pricing_stage_are_counted(self):
        engine = StrategyEngineV3()
        markets = [_market(f"m{i}") for i in range(3)]
        report, _opps = _scan(engine, markets, lambda mid: _real_book(mid))
        assert report.evaluated == 3
        assert report.skipped_no_book == 0
        assert report.beyond_cap == 0

    def test_a_market_with_no_usable_book_is_counted_as_refused(self):
        engine = StrategyEngineV3()
        markets = [_market("a"), _market("b")]
        report, _opps = _scan(
            engine, markets,
            lambda mid: {"market_id": mid, "is_real": False,
                         "validated": False, "source": "clob_unvalidated"})
        assert report.evaluated == 0
        assert report.skipped_no_book == 2

    def test_markets_beyond_the_per_venue_cap_are_counted_not_ignored(self):
        engine = StrategyEngineV3()
        engine.evaluate_limit = 5
        markets = [_market(f"m{i}", volume_24h=90_000 - i) for i in range(12)]
        report, _opps = _scan(engine, markets, lambda mid: _real_book(mid))
        assert report.evaluated == 5
        assert report.beyond_cap == 7

    def test_the_counts_account_for_every_market_the_scan_read(self):
        engine = StrategyEngineV3()
        engine.evaluate_limit = 4
        markets = [_market(f"m{i}", volume_24h=90_000 - i) for i in range(10)]
        report, _opps = _scan(
            engine, markets,
            lambda mid: (_real_book(mid) if mid >= "m4"
                         else {"is_real": False, "validated": False}))
        assert (report.evaluated + report.skipped_no_book + report.beyond_cap
                == 10), "every market read is either priced, refused for its book, or past the cap"


class TestTheScreenIsAPlan:
    def test_the_screen_line_does_not_say_priced(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._prescan_and_rank)
        assert "priced on the cheap context only" not in source
        assert "none of them has been priced yet" in source

    def test_the_reasons_do_not_claim_the_scan_priced_them(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3._prescan_and_rank)
        assert "was still read, priced" not in source
        assert "decided by the venue scan" in source.lower()

    def test_the_cycle_reports_the_venues_own_counts(self):
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "Venue scan priced" in source
        assert "had no usable book" in source
        assert "were beyond their venue's " in source


class TestTheScreenCannotPlanForAMarketTheScanWillDrop:
    """A shortlist of markets the pricing stage will refuse is model time
    that can never happen - the 16:53 cycle chose 8 and priced none of them.
    The screen therefore applies the same floors the scan does."""

    def _score(self, market):
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.strategy.strategy_engine import StrategyEngineV3

        class _Stub:
            strategy_engine_v3 = StrategyEngineV3()
            dry_run = True
        stub = _Stub()
        method = TradingAgentV3._screen_score.__get__(stub)
        return method(market, _real_book(market.id))[0]

    def test_a_market_with_too_little_volume_is_not_shortlisted(self):
        engine = StrategyEngineV3()
        floor = getattr(engine, "min_volume_24h", 0) or 0
        market = _market("thin", volume_24h=max(0.0, floor - 1.0))
        assert self._score(market) < 0, (
            "the scan would refuse this market on its volume floor, so the "
            "screen must not spend model time planning for it")

    def test_a_market_with_no_liquidity_is_not_shortlisted(self):
        engine = StrategyEngineV3()
        floor = getattr(engine, "min_liquidity", 0) or 0
        market = _market("dry", liquidity=max(0.0, floor - 1.0))
        assert self._score(market) < 0

    def test_a_healthy_market_is_still_shortlistable(self):
        market = _market("good", volume_24h=100_000, liquidity=20_000)
        assert self._score(market) > 0
