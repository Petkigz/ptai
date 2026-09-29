"""
ONE FACT, ONE LINE - the 18:14 run's repetition, and the sentence it was missing.

Two things in that log:

  * the model line said `markets asked: 3` while the screen's tail said
    `5 of them reached the pricing stage`, and nothing connected the two: two
    deep markets were refused by the resolution gate before a forecast was ever
    built. The model WAS used, so no NOT USED clause appeared, and the two
    markets simply vanished from the sentence that counts them;
  * one fact printed three or four times: snscrape's failure carried its whole
    ~1.5 KB GraphQL URL on every line, ESPN's HTTP 403 printed once per league
    inside a window that was already open, and the base-rate model's "no
    historical data" warning printed once per MARKET because the flag lived on
    an object built per forecast.

Every test here is that: one fact, said once, in words that add up.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.ptai.intelligence.forecaster import BaseRateModel


class _Lines:
    def __init__(self):
        from loguru import logger
        self.lines: List[tuple] = []
        self._sink = logger.add(
            lambda m: self.lines.append((m.record["level"].name,
                                         m.record["message"])),
            level="DEBUG", format="{message}")

    def close(self):
        from loguru import logger
        logger.remove(self._sink)

    def said(self, needle: str) -> List[tuple]:
        return [(lvl, msg) for lvl, msg in self.lines if needle in msg]


# ---------------------------------------------------------------------------
# 1. the deep markets the model was never asked about
# ---------------------------------------------------------------------------

def _status(**over: Any) -> Dict[str, Any]:
    status = {
        "available": True, "describe": "qwen/qwen3-14b (LMStudioProvider)",
        "model": "qwen/qwen3-14b", "calls": 3, "answered": 3, "failed": 0,
        "seconds": 57.1, "models": {"qwen/qwen3-14b": 3}, "last_error": "",
        "used": True, "not_used_reason": "", "asked": 3, "answered_by_model": 3,
        "considered": 200, "deep_shortlist": 5, "screened_out_no_model_time": 195,
        "priced": 5, "deep_priced": 5, "forecasts_run": 3,
        "thinking": False, "thinking_note": "thinking: OFF",
    }
    status.update(over)
    return status


def _line(status: Dict[str, Any]) -> str:
    from src.ptai.agent.v3_loop import TradingAgentV3

    class _Agent:
        llm_router = None
        _cycle_scan_counts: Dict[str, Any] = {}

    _Agent._local_model_status = lambda self: status
    _Agent._local_model_line = TradingAgentV3._local_model_line
    return _Agent()._local_model_line()


class TestTheDeepMarketsThatWereNotAsked:
    def test_the_line_accounts_for_the_gap_between_priced_and_asked(self):
        line = _line(_status(deep_priced=5, asked=3))
        assert "markets asked: 3" in line
        assert ("2 of the 5 deep market(s) were refused before a forecast was "
                "built") in line, (
            "5 reached the pricing stage and 3 were asked; the other 2 were "
            "refused before any forecast and the line has to say so")
        assert "the model was not asked about them" in line

    def test_no_clause_when_every_deep_market_was_asked(self):
        line = _line(_status(deep_priced=3, asked=3))
        assert "were refused before a forecast" not in line

    def test_a_not_used_cycle_keeps_its_own_sentence(self):
        line = _line(_status(used=False, asked=0, answered_by_model=0,
                             not_used_reason="2 market(s) reached the pricing "
                                             "stage and were refused before a "
                                             "forecast was built, so the model "
                                             "was never asked",
                             deep_priced=2, forecasts_run=0))
        assert "NOT USED THIS CYCLE" in line
        assert "refused before a forecast was built" in line
        assert "of the 2 deep market(s)" not in line, (
            "the NOT USED sentence already covers it; saying it twice is the "
            "habit this round is removing")


# ---------------------------------------------------------------------------
# 2. snscrape's failure is a sentence, not a URL
# ---------------------------------------------------------------------------

def _stub_snscrape(monkeypatch, error: BaseException) -> None:
    """`import snscrape.modules.twitter` has to succeed; the search has to fail."""
    import sys
    import types

    for name in ("snscrape", "snscrape.modules", "snscrape.modules.twitter"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class _Scraper:
        def __init__(self, query):
            self.query = query

        def get_items(self):
            def _iter():
                raise error
                yield  # pragma: no cover - keeps this a generator
            return _iter()

    sys.modules["snscrape.modules.twitter"].TwitterSearchScraper = _Scraper


class TestSnscrapeFailuresAreReadable:
    def _scraper(self):
        from src.ptai.sentiment.x_scraper import XScraper
        return XScraper(method="snscrape")

    def test_the_url_is_not_in_the_warning(self, monkeypatch):
        scraper = self._scraper()
        huge = ("4 requests to https://twitter.com/i/api/graphql/"
                "7jT5GT59P8IFjgxwqnEdQw/SearchTimeline?variables=%7B%22rawQuery"
                "%22%3A%22Valorant%3A%20Karmine%20Corp%20vs%20NRG%20%28BO3%29"
                "%22%7D&features=%7B%22rweb_lists_timeline_redesign_enabled%22"
                "%3Afalse%7D failed, giving up.")
        monkeypatch.setattr(scraper, "_clean_query", lambda q: q)
        _stub_snscrape(monkeypatch, RuntimeError(huge))
        lines = _Lines()
        try:
            scraper.search_snscrape("Valorant: Karmine Corp vs NRG (BO3)")
        finally:
            lines.close()
        warnings = [msg for lvl, msg in lines.said("snscrape failed")
                    if lvl == "WARNING"]
        assert len(warnings) == 1
        assert "twitter.com" not in warnings[0], (
            "the GraphQL URL is 1.5 KB per line and says nothing the query name "
            "does not")
        assert len(warnings[0]) < 200
        assert "(failure 1/3)" in warnings[0], (
            "the count is the fact the breaker turns on; it has to survive")
        assert scraper.consecutive_failures == 1, (
            "the failure must be counted once, not once per log line")

    def test_the_breaker_still_opens_after_three(self, monkeypatch):
        scraper = self._scraper()
        scraper.consecutive_failures = 2
        _stub_snscrape(monkeypatch, RuntimeError("blocked 404"))
        monkeypatch.setattr(scraper, "_clean_query", lambda q: q)
        lines = _Lines()
        try:
            scraper.search_snscrape("anything")
        finally:
            lines.close()
        assert scraper.circuit_open is True
        assert lines.said("X BLOCKING DETECTED")


# ---------------------------------------------------------------------------
# 3. a feed that is already blocked is not news
# ---------------------------------------------------------------------------

class TestABlockedFeedSaysItOnce:
    def test_the_second_request_in_the_window_is_not_a_warning(self):
        from src.ptai.betting.sports_data import EspnProvider
        provider = EspnProvider()
        lines = _Lines()
        try:
            provider._mark_blocked(403, "https://site.api.espn.com/apis/site/v2/"
                                        "sports/soccer/eng.1/scoreboard")
            provider._mark_blocked(403, "https://site.api.espn.com/apis/site/v2/"
                                        "sports/soccer/esp.1/scoreboard")
        finally:
            lines.close()
        warnings = [msg for lvl, msg in lines.said("HTTP 403")
                    if lvl == "WARNING"]
        assert len(warnings) == 1, (
            "two leagues of one blocked feed is one fact; the operator's log "
            "printed it twice in the same second")
        assert provider.is_blocked is True
        assert "espn" in provider.blocked_reason


# ---------------------------------------------------------------------------
# 4. the base-rate warning belongs to the process, not to the model object
# ---------------------------------------------------------------------------

class TestTheBaseRateWarningIsOncePerProcess:
    def test_a_new_model_object_does_not_print_it_again(self):
        import src.ptai.intelligence.forecaster as fc
        fc._NO_DATA_WARNED = False
        lines = _Lines()
        try:
            for _ in range(3):
                model = BaseRateModel()
                model.forecast(_market(), category="politics")
        finally:
            lines.close()
        said = lines.said("Base-rate model has no historical data loaded")
        assert len(said) == 1, (
            "the model is built per forecast, so a per-instance flag printed "
            "this once per deep market - three times in the 18:14 run, and "
            "eight with a full shortlist")


def _market():
    from src.ptai.markets.base import DataMode, Market, MarketSource
    return Market(id="m", source=MarketSource.POLYMARKET,
                  question="Will it rain?",
                  outcome_prices=[0.5, 0.5], volume_24h=5000, liquidity=5000,
                  venue_id="polymarket", data_mode=DataMode.LIVE)


# ---------------------------------------------------------------------------
# 5. the two "tradeable" counts are named
# ---------------------------------------------------------------------------

class TestTwoBarsAreNamed:
    def test_the_venue_count_says_which_bar_it_cleared(self):
        import inspect
        from src.ptai.strategy import strategy_engine
        source = inspect.getsource(
            strategy_engine.StrategyEngineV3.scan_all_venues)
        assert "tradeable at the venue's own bar" in source
        assert ("a venue's own count above is before these gates") in source, (
            "'polymarket: tradeable 1' beside 'After fees/liquidity/uncertainty/"
            "risk: 0 actually tradeable' read as a contradiction; they are two "
            "different bars and the line now says which")


# ---------------------------------------------------------------------------
# 6. the venue scan's numbers add up
# ---------------------------------------------------------------------------

class TestTheVenueScanNumbersAddUp:
    def _line(self, **counts):
        from src.ptai.agent.v3_loop import TradingAgentV3
        status = _status(deep_priced=3, asked=3)

        class _Agent:
            llm_router = None
            _cycle_scan_counts = dict(counts)

        _Agent._local_model_status = lambda self: status
        _Agent._local_model_line = TradingAgentV3._local_model_line
        return _Agent()._local_model_line()

    def test_the_operator_1814_case_accounts_for_all_900(self):
        line = self._line(venues=19, evaluated=5, skipped_no_book=95,
                          beyond_cap=100, discovered=900)
        assert "priced 5 market(s) of the 900 read" in line
        assert "95 had no usable book" in line
        assert "100 beyond their venue's cap" in line
        assert "700 in venues or below the floors this cycle does not price" \
               in line, (
            "5 + 95 + 100 = 200 of 900; the other 700 have to be named or the "
            "sentence does not add up in front of the operator")

    def test_a_fully_accounted_cycle_says_zero_left_over(self):
        line = self._line(venues=2, evaluated=10, skipped_no_book=5,
                          beyond_cap=0, discovered=15)
        assert "0 in venues or below the floors this cycle does not price" in line
