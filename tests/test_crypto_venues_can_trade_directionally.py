"""
The directional lane: the honest way a price venue is paper-traded.

Two crypto venues read live markets every cycle and were never used for
anything, because the probability lane is right to refuse them: a crypto
exchange quotes a PRICE, and a forecast probability has nothing to be an edge
against. What was missing was not a client or a login - it was an instrument
that matches what an exchange sells.

This suite pins the lane that does that, and every honest boundary around it:

  * the market record carries the venue's quote and no invented probability
    (`prob_up = 0.5 + change x 2` used to be written into `outcome_prices`);
  * entry and exit are the venue's own bid/ask, and the stop and target are the
    venue's own measured volatility;
  * a stop and a target in the same candle settles at the STOP - the conservative
    reading, applied uniformly;
  * no model answer, no measured volatility, no two-sided quote or no EV after
    both fees means NO trade, and the reason is recorded;
  * the money is a separate paper purse in its own table, so a directional trade
    can never enter the resolved-probability record that unlocks live capital.

No network: the venue calls are driven through recorded payload shapes.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from src.ptai.execution.directional import (
    DIRECTIONAL_MAX_OPEN,
    DIRECTIONAL_PURSE_SEED_PCT,
    DirectionalForecaster,
    DirectionalPaperLane,
    DirectionalQuote,
    directional_prompt,
    measured_sigma,
    plan_directional_trade,
    pnl_for,
    settle_position,
)
from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.storage.db import Storage


# ---------------------------------------------------------------------------
# fixtures and fakes
# ---------------------------------------------------------------------------

def _response(payload: Any, status_code: int = 200):
    class _R:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    return _R()


class _Session:
    """A requests.Session stand-in that answers from a route table and records."""

    def __init__(self, routes: Dict[str, Any]):
        self.routes = routes
        self.headers: Dict[str, str] = {}
        self.calls: List[str] = []

    def get(self, url, params=None, timeout=None, **kwargs):
        self.calls.append(url)
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return _response(payload)
        return _response({}, status_code=404)


def _binance_tickers() -> List[Dict[str, Any]]:
    return [
        {"symbol": "BTCUSDT", "lastPrice": "68120.10", "quoteVolume": "900000000",
         "priceChangePercent": "1.20"},
        {"symbol": "ETHUSDT", "lastPrice": "3450.55", "quoteVolume": "400000000",
         "priceChangePercent": "-0.80"},
    ]


def _klines(count: int = 48, start: float = 100.0, step: float = 0.002) -> List[List[Any]]:
    """Binance kline rows: [open_time, open, high, low, close, volume, ...]."""
    rows = []
    price = start
    for i in range(count):
        price = price * (1.0 + (step if i % 2 == 0 else -step * 0.6))
        rows.append([1700000000000 + i * 3_600_000, price * 0.999, price * 1.004,
                     price * 0.996, price, 12.5, 0])
    return rows


def _candles(count: int = 48, start: float = 100.0, step: float = 0.002) -> List[Dict[str, Any]]:
    return [{"open_time": int(r[0]), "open": float(r[1]), "high": float(r[2]),
             "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
            for r in _klines(count, start, step)]


def _crypto_market(symbol: str = "BTCUSDT", venue_id: str = "crypto_binance") -> Market:
    raw = {"venue": venue_id, "symbol": symbol, "last_price": 68120.10,
           "change_pct": 1.2, "probability_market": False,
           "quote_scale": "currency", "market_kind": "directional"}
    return Market(
        id=f"CRYPTO-{symbol}", source=MarketSource.PREDICTIT,
        question=f"{symbol}: price higher or lower in 24h?",
        outcomes=[symbol], outcome_prices=[68120.10],
        volume=900_000_000.0, volume_24h=900_000_000.0, liquidity=90_000_000.0,
        active=True, closed=False, slug=symbol.lower(),
        market_type="directional", raw=raw, venue_id=venue_id,
        venue_type="financial", data_mode=DataMode.LIVE,
        data_source="binance_api", is_mock=False,
    )


class _Reply:
    def __init__(self, parsed=None, content="", model="test-model"):
        self.parsed_json = parsed
        self.content = content
        self.model = model


class _Router:
    def __init__(self, parsed=None, content="", available=True, model="test-model"):
        self.parsed = parsed
        self.content = content
        self.available = available
        self.model = model
        self.prompts: List[str] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, prompt="", system=""):
        self.prompts.append(prompt)
        if self.available is False:
            return None
        return _Reply(parsed=self.parsed, content=self.content, model=self.model)


@pytest.fixture()
def storage(tmp_path):
    store = Storage(str(tmp_path / "directional.db"))
    yield store
    store.close()


# ---------------------------------------------------------------------------
# 1. a market record is a quote, not a forecast
# ---------------------------------------------------------------------------

class TestAMarketRecordIsAQuoteNotAForecast:
    def test_the_invented_probability_is_gone_from_the_code(self):
        import inspect
        from src.ptai.venues.crypto_adapter import CryptoAdapter
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        for source in (inspect.getsource(CryptoAdapter.discover_markets),
                       inspect.getsource(WhiteBITAdapter.discover_markets)):
            assert "prob_up" not in source, (
                "a probability the venue never quoted must not be written into a "
                "market record")
            assert '"probability_market": False' in source
            assert '"quote_scale": "currency"' in source
            assert '"market_kind": "directional"' in source

    def test_the_record_carries_the_quote_and_the_symbol_alone(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"ticker/24hr": _binance_tickers()})
        markets = asyncio.run(adapter.discover_markets(target_count=2))
        assert [m.raw["symbol"] for m in markets] == ["BTCUSDT", "ETHUSDT"]
        btc = markets[0]
        assert btc.outcomes == ["BTCUSDT"], "no YES/NO shares exist here"
        assert btc.outcome_prices == [68120.10], (
            "the price of the UP token is the quote, not a probability")
        assert btc.market_type == "directional"
        assert btc.is_probability_market is False
        assert btc.raw["last_price"] == 68120.10
        assert btc.raw["change_pct"] == 1.2

    def test_the_whitebit_record_is_the_same_kind(self, monkeypatch):
        import requests as requests_module
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        monkeypatch.setattr(requests_module, "get", lambda url, **kwargs: _response(
            {"BTC_USDT": {"last_price": "68000", "quote_volume": "5000",
                          "change": "1.5"}}))
        markets = asyncio.run(WhiteBITAdapter().discover_markets(target_count=1))
        assert markets and markets[0].outcomes == ["BTC_USDT"]
        assert markets[0].is_probability_market is False
        assert markets[0].market_type == "directional"

    def test_both_adapters_declare_which_lane_they_run_in(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        for adapter in (CryptoAdapter(), WhiteBITAdapter()):
            caps = adapter.capabilities
            assert caps.quotes_prices_not_probabilities is True
            assert caps.supports_trading is False, (
                "declaring the lane must not hand a price venue a real order path")


# ---------------------------------------------------------------------------
# 2. the quote is the venue's
# ---------------------------------------------------------------------------

class TestTheQuoteIsTheVenues:
    def test_a_two_sided_quote_is_real_and_labelled(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"bookTicker": {"bidPrice": "68100.00",
                                                   "askPrice": "68110.00"}})
        quote = adapter.directional_quote(_crypto_market())
        assert quote.is_real is True
        assert quote.bid == 68100.0 and quote.ask == 68110.0
        assert quote.mid == 68105.0
        assert quote.spread_pct == pytest.approx(10.0 / 68105.0, abs=1e-9)
        assert quote.valid is True
        assert quote.source == "binance_book_ticker"
        assert quote.symbol == "BTCUSDT"

    def test_a_venue_that_does_not_answer_has_no_quote(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"bookTicker": {"ignored": True}})
        adapter.session.get = lambda url, params=None, timeout=None, **k: _response({}, 503)
        quote = adapter.directional_quote("BTCUSDT")
        assert quote.valid is False
        assert "503" in quote.reason
        assert quote.bid == 0.0 and quote.ask == 0.0

    def test_a_one_sided_or_crossed_quote_is_refused(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"bookTicker": {"bidPrice": "0", "askPrice": "0"}})
        assert adapter.directional_quote("BTCUSDT").valid is False
        adapter.session = _Session({"bookTicker": {"bidPrice": "100", "askPrice": "99"}})
        quote = adapter.directional_quote("BTCUSDT")
        assert quote.valid is False
        assert "bid" in quote.reason

    def test_a_transport_error_is_a_reason_not_a_quote(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"bookTicker": ConnectionError("no route")})
        quote = adapter.directional_quote("BTCUSDT")
        assert quote.valid is False
        assert "ConnectionError" in quote.reason

    def test_whitebit_reads_its_own_book(self, monkeypatch):
        import requests as requests_module
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        monkeypatch.setattr(requests_module, "get", lambda url, **kwargs: _response(
            {"bids": [["100.0", "5"]], "asks": [["100.2", "4"]]}))
        quote = WhiteBITAdapter().directional_quote("BTC_USDT")
        assert quote.is_real is True
        assert quote.bid == 100.0 and quote.ask == 100.2
        assert quote.source == "whitebit_public_orderbook"

    def test_whitebit_refuses_a_one_sided_book(self, monkeypatch):
        import requests as requests_module
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        monkeypatch.setattr(requests_module, "get", lambda url, **kwargs: _response(
            {"bids": [], "asks": [["100.2", "4"]]}))
        quote = WhiteBITAdapter().directional_quote("BTC_USDT")
        assert quote.valid is False
        assert "one-sided" in quote.reason


# ---------------------------------------------------------------------------
# 3. volatility is measured, or there is none
# ---------------------------------------------------------------------------

class TestVolatilityIsMeasured:
    def test_a_measured_sigma_comes_from_the_venues_candles(self):
        sigma = measured_sigma(_candles(48))
        assert sigma is not None
        assert 0.001 < sigma < 0.2

    def test_too_few_candles_measure_nothing(self):
        assert measured_sigma(_candles(5)) is None

    def test_a_junk_row_measures_nothing_rather_than_guessing(self):
        rows = _candles(48)
        rows[7]["close"] = None
        assert measured_sigma(rows) is None

    def test_flat_candles_are_not_a_zero_volatility_trade(self):
        flat = [{"close": 100.0, "high": 100.0, "low": 100.0} for _ in range(48)]
        assert measured_sigma(flat) is None, (
            "a measured volatility of zero on a real market means the candles are "
            "not what we think they are")

    def test_sigma_scales_with_the_square_root_of_the_horizon(self):
        one = measured_sigma(_candles(48), horizon_hours=1)
        four = measured_sigma(_candles(48), horizon_hours=4)
        assert four == pytest.approx(one * 2.0, rel=1e-6)

    def test_the_binance_parser_reads_klines(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"klines": _klines(20)})
        rows = adapter.recent_candles("BTCUSDT", hours=20)
        assert len(rows) == 20
        assert set(rows[0]) >= {"open_time", "open", "high", "low", "close"}
        assert rows[0]["high"] >= rows[0]["low"]

    def test_an_unanswered_kline_call_returns_no_candles(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter

        adapter = CryptoAdapter()
        adapter.session = _Session({"klines": {}})
        adapter.session.get = lambda url, params=None, timeout=None, **k: _response({}, 418)
        assert adapter.recent_candles("BTCUSDT") == []
        assert "418" in adapter.last_error

    def test_the_whitebit_parser_reads_its_own_row_order(self, monkeypatch):
        import requests as requests_module
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter

        # WhiteBIT rows: [time, open, close, high, low, volume]
        monkeypatch.setattr(requests_module, "get", lambda url, **kwargs: _response(
            {"success": True,
             "result": [[1700000000000, 100.0, 101.0, 102.0, 99.0, 5.0]]}))
        rows = WhiteBITAdapter().recent_candles("BTC_USDT")
        assert rows[0]["open"] == 100.0 and rows[0]["close"] == 101.0
        assert rows[0]["high"] == 102.0 and rows[0]["low"] == 99.0


# ---------------------------------------------------------------------------
# 4. the plan: real prices, measured move, fees on both sides
# ---------------------------------------------------------------------------

class TestThePlanUsesRealPrices:
    def _quote(self, bid: float = 100.0, ask: float = 100.2) -> DirectionalQuote:
        return DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                bid=bid, ask=ask, last=bid, is_real=True)

    def test_a_long_enters_at_the_ask_and_brackets_itself(self):
        plan = plan_directional_trade(self._quote(), sigma=0.05, prob_up=0.65,
                                      purse_usd=100.0, fee_pct=0.001)
        assert plan.should_trade is True, plan.refusals
        assert plan.side == "LONG"
        assert plan.entry == 100.2, "a buyer pays the ask"
        assert plan.stop == pytest.approx(100.2 * 0.95, rel=1e-9)
        assert plan.target == pytest.approx(100.2 * 1.05, rel=1e-9)
        assert plan.win_per_usd == pytest.approx(0.05 - 0.002, rel=1e-6)
        assert plan.loss_per_usd == pytest.approx(0.05 + 0.002, rel=1e-6)
        assert plan.ev_per_usd == pytest.approx(0.65 * 0.048 - 0.35 * 0.052, rel=1e-6)
        assert 0 < plan.size_usd <= 6.0

    def test_a_short_sells_at_the_bid_and_mirrors_the_levels(self):
        plan = plan_directional_trade(self._quote(), sigma=0.05, prob_up=0.30,
                                      purse_usd=100.0, fee_pct=0.001)
        assert plan.side == "SHORT"
        assert plan.entry == 100.0, "a seller receives the bid"
        assert plan.stop == pytest.approx(105.0, rel=1e-9)
        assert plan.target == pytest.approx(95.0, rel=1e-9)
        assert plan.should_trade is True, plan.refusals

    def test_a_coin_flip_is_refused_after_the_fees(self):
        plan = plan_directional_trade(self._quote(), sigma=0.05, prob_up=0.50,
                                      purse_usd=100.0, fee_pct=0.001)
        assert plan.should_trade is False
        assert any("expected value" in r for r in plan.refusals)

    def test_an_unmeasured_volatility_is_refused(self):
        for sigma in (None, 0.0):
            plan = plan_directional_trade(self._quote(), sigma=sigma, prob_up=0.9,
                                          purse_usd=100.0, fee_pct=0.001)
            assert plan.should_trade is False
            assert any("volatility" in r for r in plan.refusals)

    def test_a_model_with_no_answer_is_refused(self):
        for prob in (None, 1.4, -0.1):
            plan = plan_directional_trade(self._quote(), sigma=0.03, prob_up=prob,
                                          purse_usd=100.0, fee_pct=0.001)
            assert plan.should_trade is False
            assert any("directional probability" in r for r in plan.refusals)

    def test_a_wide_spread_is_a_venue_problem_not_an_edge(self):
        plan = plan_directional_trade(self._quote(bid=100.0, ask=101.5), sigma=0.05,
                                      prob_up=0.9, purse_usd=100.0, fee_pct=0.001)
        assert plan.should_trade is False
        assert any("spread" in r for r in plan.refusals)

    def test_a_fee_that_eats_the_move_is_refused(self):
        plan = plan_directional_trade(self._quote(), sigma=0.0005, prob_up=0.9,
                                      purse_usd=100.0, fee_pct=0.01)
        assert plan.should_trade is False
        assert any("round-trip fee" in r or "expected value" in r
                   for r in plan.refusals)

    def test_the_size_is_half_kelly_capped_at_six_percent(self):
        plan = plan_directional_trade(self._quote(), sigma=0.05, prob_up=0.9,
                                      purse_usd=1000.0, fee_pct=0.001)
        assert plan.size_usd <= 1000.0 * 0.06 + 1e-9
        assert plan.size_usd > 0

    def test_an_empty_purse_opens_nothing(self):
        plan = plan_directional_trade(self._quote(), sigma=0.05, prob_up=0.9,
                                      purse_usd=0.0, fee_pct=0.001)
        assert plan.should_trade is False
        assert plan.refusals

    def test_a_stale_quote_is_refused_before_any_arithmetic(self):
        quote = DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                 bid=0.0, ask=0.0, is_real=False,
                                 reason="no quote from the venue")
        plan = plan_directional_trade(quote, sigma=0.02, prob_up=0.9,
                                      purse_usd=100.0, fee_pct=0.001)
        assert plan.should_trade is False
        assert "no quote from the venue" in plan.refusals


# ---------------------------------------------------------------------------
# 5. settlement walks the venue's candles
# ---------------------------------------------------------------------------

def _position(side: str = "LONG", entry: float = 100.0, size: float = 10.0,
              stop: float = 98.0, target: float = 102.0,
              opened_hours_ago: float = 1.0, horizon: float = 24.0) -> Dict[str, Any]:
    return {"id": 1, "side": side, "entry_price": entry,
            "position_size_usd": size, "stop_price": stop, "target_price": target,
            "opened_at": (datetime.now(timezone.utc)
                          - timedelta(hours=opened_hours_ago)).isoformat(),
            "horizon_hours": horizon, "venue_id": "crypto_binance",
            "symbol": "BTCUSDT", "status": "open"}


class TestSettlementWalksTheCandles:
    def test_the_stop_closes_the_position_at_the_stop(self):
        candles = [{"high": 100.5, "low": 97.5, "close": 98.0}]
        verdict = settle_position(_position(), candles, fee_pct=0.001)
        assert verdict["exit_reason"] == "stop"
        assert verdict["exit_price"] == 98.0
        money = pnl_for("LONG", 100.0, 98.0, 10.0, 0.001)
        assert money["pnl_usd"] == pytest.approx(-0.2 - 0.02, abs=1e-6)

    def test_the_target_closes_it_at_the_target(self):
        candles = [{"high": 100.5, "low": 99.5, "close": 100.0},
                   {"high": 102.5, "low": 100.4, "close": 102.0}]
        verdict = settle_position(_position(), candles, fee_pct=0.001)
        assert verdict["exit_reason"] == "target"
        assert verdict["exit_price"] == 102.0
        assert pnl_for("LONG", 100.0, 102.0, 10.0, 0.001)["pnl_usd"] == pytest.approx(
            0.2 - 0.02, abs=1e-6)

    def test_a_candle_that_contains_both_settles_at_the_stop(self):
        candles = [{"high": 103.0, "low": 97.0, "close": 101.0}]
        verdict = settle_position(_position(), candles, fee_pct=0.001)
        assert verdict["exit_reason"] == "stop", (
            "when one candle can have hit either level, the conservative reading "
            "is the one that applies - uniformly, not per trade")

    def test_nothing_hit_before_the_horizon_stays_open(self):
        candles = [{"high": 100.5, "low": 99.5, "close": 100.0}]
        assert settle_position(_position(opened_hours_ago=1.0), candles,
                               fee_pct=0.001) is None

    def test_the_horizon_settles_at_the_venues_last_price(self):
        candles = [{"high": 100.5, "low": 99.5, "close": 100.0}]
        verdict = settle_position(_position(opened_hours_ago=25.0), candles,
                                  last_price=100.6, fee_pct=0.001)
        assert verdict["exit_reason"] == "horizon"
        assert verdict["exit_price"] == 100.6

    def test_the_horizon_falls_back_to_the_last_candle_close(self):
        candles = [{"high": 100.5, "low": 99.5, "close": 100.4}]
        verdict = settle_position(_position(opened_hours_ago=25.0), candles,
                                  last_price=None, fee_pct=0.001)
        assert verdict["exit_reason"] == "horizon"
        assert verdict["exit_price"] == 100.4

    def test_a_short_stops_up_and_targets_down(self):
        position = _position(side="SHORT", stop=102.0, target=98.0)
        assert settle_position(position, [{"high": 102.5, "low": 101.0,
                                           "close": 102.0}],
                               fee_pct=0.001)["exit_reason"] == "stop"
        assert settle_position(position, [{"high": 101.0, "low": 97.5,
                                           "close": 98.0}],
                               fee_pct=0.001)["exit_reason"] == "target"

    def test_a_short_profits_when_the_price_falls(self):
        money = pnl_for("SHORT", 100.0, 98.0, 10.0, 0.001)
        assert money["pnl_usd"] == pytest.approx(10.0 * (100.0 / 98.0 - 1.0) - 0.02,
                                                 abs=1e-6)
        losing = pnl_for("SHORT", 100.0, 103.0, 10.0, 0.001)
        assert losing["pnl_usd"] < 0

    def test_fees_are_charged_on_both_sides(self):
        money = pnl_for("LONG", 100.0, 100.0, 50.0, 0.002)
        assert money["fees_usd"] == pytest.approx(0.2, abs=1e-6)
        assert money["pnl_usd"] == pytest.approx(-0.2, abs=1e-6)

    def test_a_position_with_no_readable_candles_settles_nothing(self):
        assert settle_position(_position(opened_hours_ago=99.0), [],
                               last_price=None, fee_pct=0.001) is None

    def test_a_position_with_a_broken_row_settles_nothing(self):
        position = _position()
        position["entry_price"] = None
        assert settle_position(position, [{"high": 103.0, "low": 99.0,
                                           "close": 100.0}],
                               fee_pct=0.001) is None


# ---------------------------------------------------------------------------
# 6. the model's one question
# ---------------------------------------------------------------------------

class TestTheForecasterAsksOneQuestion:
    def _quote(self) -> DirectionalQuote:
        return DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                bid=100.0, ask=100.2, last=100.1, change_pct=1.2,
                                volume_24h=900_000_000.0, is_real=True)

    def test_no_model_means_no_answer_and_a_reason(self):
        forecaster = DirectionalForecaster(_Router(available=False))
        assert forecaster.forecast(self._quote(), 0.02) is None
        assert forecaster.last_problem
        assert forecaster.calls == 0

    def test_a_missing_router_is_reported_not_guessed(self):
        forecaster = DirectionalForecaster(None)
        assert forecaster.forecast(self._quote(), 0.02) is None
        assert "no model router" in forecaster.last_problem

    def test_a_parsed_probability_is_returned_with_its_model(self):
        router = _Router(parsed={"prob_up": 0.68, "confidence": 0.7,
                                 "basis": "trend", "reasoning": "up trend"})
        forecaster = DirectionalForecaster(router)
        prob = forecaster.forecast(self._quote(), 0.02)
        assert prob == 0.68
        assert forecaster.last_model == "test-model"
        assert forecaster.calls == 1
        assert "MEASURED VOLATILITY" in router.prompts[0]
        assert "is HIGHER than" in router.prompts[0]

    def test_prose_wrapped_json_is_still_read(self):
        router = _Router(content='Sure. {"prob_up": 0.31, "confidence": 0.5}')
        forecaster = DirectionalForecaster(router)
        assert forecaster.forecast(self._quote(), 0.02) == 0.31

    def test_an_out_of_range_probability_is_refused(self):
        forecaster = DirectionalForecaster(_Router(parsed={"prob_up": 1.5}))
        assert forecaster.forecast(self._quote(), 0.02) is None
        assert "0.01-0.99" in forecaster.last_problem

    def test_an_answer_with_no_json_is_refused_with_the_model_named(self):
        forecaster = DirectionalForecaster(_Router(content="I cannot say."))
        assert forecaster.forecast(self._quote(), 0.02) is None
        assert "test-model" in forecaster.last_problem

    def test_the_prompt_states_the_measured_numbers(self):
        prompt = directional_prompt(self._quote(), 0.031, horizon_hours=24)
        assert "3.10%" in prompt
        assert "BTCUSDT" in prompt
        assert "+1.20%" in prompt
        assert "0.01-0.99" in prompt

    def test_the_prompt_says_this_is_not_a_probability_market(self):
        from src.ptai.execution.directional import DIRECTIONAL_SYSTEM_PROMPT
        assert "not a probability market" in DIRECTIONAL_SYSTEM_PROMPT
        assert "no Yes share" in DIRECTIONAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 7. the purse and the record are the lane's own
# ---------------------------------------------------------------------------

class TestThePurseAndTheRecord:
    def test_the_purse_is_seeded_once_from_the_paper_bankroll(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        assert lane.purse() == pytest.approx(200.0 * DIRECTIONAL_PURSE_SEED_PCT)
        # and it does not grow a second time
        storage.set_paper_bankroll(1000.0)
        assert lane.purse() == pytest.approx(200.0 * DIRECTIONAL_PURSE_SEED_PCT)

    def _plan(self, prob: float = 0.7, purse: float = 100.0):
        quote = DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                 bid=100.0, ask=100.2, last=100.1, is_real=True)
        return plan_directional_trade(quote, sigma=0.05, prob_up=prob,
                                      purse_usd=purse, fee_pct=0.001)

    def test_opening_draws_the_purse_and_writes_its_own_table(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        before = lane.purse()
        plan = self._plan(purse=before)
        result = lane.open(plan, market_question="BTCUSDT higher?")
        assert result["opened"] is True
        assert lane.purse() == pytest.approx(before - plan.size_usd, abs=0.011)
        open_rows = storage.get_open_directional_positions()
        assert len(open_rows) == 1
        assert open_rows[0]["symbol"] == "BTCUSDT"
        assert open_rows[0]["side"] == plan.side
        assert open_rows[0]["stop_price"] < open_rows[0]["entry_price"] < \
            open_rows[0]["target_price"]

    def test_a_refused_plan_opens_nothing(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        refused = plan_directional_trade(
            DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                             bid=0.0, ask=0.0, is_real=False, reason="no quote"),
            sigma=0.05, prob_up=0.9, purse_usd=100.0, fee_pct=0.001)
        result = lane.open(refused)
        assert result["opened"] is False
        assert storage.get_open_directional_positions() == []

    def test_the_same_symbol_is_not_opened_twice(self, storage):
        storage.set_paper_bankroll(1000.0)
        lane = DirectionalPaperLane(storage)
        assert lane.open(self._plan())["opened"] is True
        second = lane.open(self._plan())
        assert second["opened"] is False
        assert "already has an open" in second["reason"]

    def test_the_lane_holds_at_most_two_positions(self, storage):
        storage.set_paper_bankroll(1000.0)
        lane = DirectionalPaperLane(storage)
        for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
            quote = DirectionalQuote(venue_id="crypto_binance", symbol=symbol,
                                     bid=100.0, ask=100.2, last=100.1, is_real=True)
            plan = plan_directional_trade(quote, sigma=0.05, prob_up=0.9,
                                          purse_usd=lane.purse(), fee_pct=0.001)
            lane.open(plan)
        assert len(storage.get_open_directional_positions()) == DIRECTIONAL_MAX_OPEN

    def test_closing_returns_the_stake_and_the_pnl(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        plan = self._plan()
        lane.open(plan)
        purse_after_open = lane.purse()
        position = storage.get_open_directional_positions()[0]
        storage.close_directional_position(int(position["id"]), exit_price=102.1,
                                           exit_reason="target", pnl_usd=5.0,
                                           fees_usd=0.2)
        lane._set_purse(purse_after_open + plan.size_usd + 5.0)
        assert lane.purse() == pytest.approx(purse_after_open + plan.size_usd + 5.0)

    def test_a_position_cannot_be_closed_twice(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        lane.open(self._plan())
        position = storage.get_open_directional_positions()[0]
        assert storage.close_directional_position(int(position["id"]), 101.0,
                                                 "target", 1.0, 0.1) is True
        assert storage.close_directional_position(int(position["id"]), 101.0,
                                                 "target", 1.0, 0.1) is False

    def test_settle_open_walks_candles_and_returns_the_money(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        plan = self._plan()
        lane.open(plan)
        purse_after_open = lane.purse()

        def candles_lookup(venue_id, symbol):
            mid = (plan.stop + plan.target) / 2.0
            if plan.side == "LONG":
                return [{"high": mid, "low": plan.stop - 0.01, "close": mid}]
            return [{"high": plan.stop + 0.01, "low": mid, "close": mid}]

        report = lane.settle_open(lambda v, s: None, candles_lookup)
        assert report["checked"] == 1 and report["closed"] == 1
        assert report["pnl_usd"] < 0, "the stop was hit"
        assert lane.purse() == pytest.approx(purse_after_open + plan.size_usd
                                             + report["pnl_usd"], abs=0.02)
        assert storage.get_open_directional_positions() == []

    def test_unreadable_candles_leave_the_position_open(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        lane.open(self._plan())
        report = lane.settle_open(lambda v, s: None, lambda v, s: [])
        assert report["closed"] == 0 and report["unreadable"] == 1
        assert len(storage.get_open_directional_positions()) == 1

    def test_the_record_states_what_it_is_and_is_not(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        plan = self._plan()
        lane.open(plan)
        position = storage.get_open_directional_positions()[0]
        storage.close_directional_position(int(position["id"]), 102.1, "target",
                                           pnl_usd=1.5, fees_usd=0.1)
        record = lane.record()
        assert record["available"] is True
        assert record["closed_count"] == 1
        assert record["wins"] == 1
        assert record["win_rate"] == 1.0
        assert "do NOT count toward the resolved" in record["note"]
        assert record["purse_usd"] is not None


# ---------------------------------------------------------------------------
# 8. the separation that makes this lane honest
# ---------------------------------------------------------------------------

class TestDirectionalMoneyNeverTouchesTheProbabilityBooks:
    def test_a_directional_round_trip_leaves_the_probability_ledger_empty(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        quote = DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                 bid=100.0, ask=100.2, last=100.1, is_real=True)
        plan = plan_directional_trade(quote, sigma=0.05, prob_up=0.8,
                                      purse_usd=lane.purse(), fee_pct=0.001)
        lane.open(plan)
        position = storage.get_open_directional_positions()[0]
        storage.close_directional_position(int(position["id"]), 101.0, "target",
                                           pnl_usd=2.0, fees_usd=0.1)

        # The probability ledgers, all three of them, see nothing.
        assert storage.get_open_positions() == []
        assert storage.get_performance_summary()["total_trades"] == 0
        calibration = storage.conn.execute("SELECT COUNT(*) AS n FROM calibration").fetchone()
        assert calibration["n"] == 0
        outcomes = storage.conn.execute(
            "SELECT COUNT(*) AS n FROM trade_outcomes").fetchone()
        assert outcomes["n"] == 0, (
            "a spot position is not a probability forecast and must never be "
            "scored as one")

    def test_the_paper_bankroll_is_not_moved_by_directional_pnl(self, storage):
        storage.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(storage)
        lane.purse()  # seeds the directional purse from the paper one
        quote = DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                 bid=100.0, ask=100.2, last=100.1, is_real=True)
        plan = plan_directional_trade(quote, sigma=0.05, prob_up=0.8,
                                      purse_usd=lane.purse(), fee_pct=0.001)
        lane.open(plan)
        position = storage.get_open_directional_positions()[0]
        storage.close_directional_position(int(position["id"]), 101.0, "target",
                                           pnl_usd=2.0, fees_usd=0.1)
        lane._set_purse(lane.purse() + plan.size_usd + 2.0)
        assert storage.get_paper_bankroll() == pytest.approx(200.0), (
            "the probability purse and the directional purse are separate books")


# ---------------------------------------------------------------------------
# 9. the agent runs the lane, and says which way it ran
# ---------------------------------------------------------------------------

class _FakeDirectionalAdapter:
    """A price venue: a quote, candles, and one directional market."""

    venue_id = "crypto_binance"
    venue_type = None

    def __init__(self, quote: bool = True, candles: bool = True):
        from src.ptai.venues.adapter import AdapterCapability, VenueType
        self.venue_type = VenueType.FINANCIAL
        self.last_error = ""
        self._quote = quote
        self._candles = candles
        self.capabilities = AdapterCapability(
            supports_market_discovery=True, supports_orderbook=True,
            supports_trading=False, requires_credentials=False,
            quotes_prices_not_probabilities=True, fee_taker_pct=0.001)

    async def discover_markets(self, target_count: int = 500, filters=None):
        return [_crypto_market()]

    def directional_quote(self, symbol):
        if not self._quote:
            return DirectionalQuote(venue_id=self.venue_id, symbol="BTCUSDT",
                                    bid=0.0, ask=0.0, is_real=False,
                                    reason="the venue did not answer")
        return DirectionalQuote(venue_id=self.venue_id, symbol="BTCUSDT",
                                bid=100.0, ask=100.2, last=100.1,
                                change_pct=1.2, volume_24h=1e9, is_real=True,
                                source="fake_venue")

    def recent_candles(self, symbol, hours=48, interval="1h"):
        # A volatility the lane can trade on: ~10% over the horizon, like a real
        # crypto pair on a moving day.
        return _candles(48, step=0.02) if self._candles else []


def _agent(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
    from src.ptai.agent.v3_loop import TradingAgentV3
    agent = TradingAgentV3(country_code="UG", dry_run=True)
    # The registry is where the lane looks for price venues; the real one is
    # replaced so nothing here depends on a network.
    agent.venue_registry.adapters = {}
    agent.venue_registry.register(_FakeDirectionalAdapter())
    return agent


class TestTheAgentRunsTheLaneAndSaysWhichWay:
    def test_the_switch_off_is_stated_not_implied(self, monkeypatch, tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            monkeypatch.setattr(agent.settings, "directional_paper_enabled", False,
                                raising=False)
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["enabled"] is False
            assert "OFF" in summary["note"]
            assert agent.storage.get_open_directional_positions() == []
        finally:
            agent.storage.close()

    def test_no_price_venue_means_nothing_to_run(self, monkeypatch, tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            agent.venue_registry.adapters = {}
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["enabled"] is False
            assert "nothing to run" in summary["note"]
        finally:
            agent.storage.close()

    def test_with_no_model_no_position_is_opened_and_it_says_so(self, monkeypatch,
                                                                tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            class _Down:
                def is_available(self):
                    return False

            agent.llm_router = _Down()
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["opened"] == 0
            assert "no model answered" in summary["note"]
            assert agent.storage.get_open_directional_positions() == []
        finally:
            agent.storage.close()

    def test_a_model_that_answers_opens_a_position_in_its_own_table(self, monkeypatch,
                                                                    tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            agent.llm_router = _Router(parsed={"prob_up": 0.72, "confidence": 0.7})
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["enabled"] is True
            assert summary["candidates"] == 1
            assert summary["considered"] == 1
            assert summary["opened"] == 1, summary
            positions = agent.storage.get_open_directional_positions()
            assert len(positions) == 1
            assert positions[0]["venue_id"] == "crypto_binance"
            assert positions[0]["model_prob"] == 0.72
            assert "test-model" in summary["model"]
            assert "do not count" in summary["note"]
            # and the probability ledgers are untouched
            assert agent.storage.get_open_positions() == []
        finally:
            agent.storage.close()

    def test_a_venue_with_no_candles_is_refused_for_the_right_reason(self, monkeypatch,
                                                                     tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            agent.venue_registry.adapters = {}
            agent.venue_registry.register(_FakeDirectionalAdapter(candles=False))
            agent.llm_router = _Router(parsed={"prob_up": 0.9})
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["opened"] == 0
            assert any("candles" in r for r in summary["refused"])
        finally:
            agent.storage.close()

    def test_a_venue_with_no_quote_is_refused_for_the_right_reason(self, monkeypatch,
                                                                   tmp_path):
        agent = _agent(monkeypatch, tmp_path)
        try:
            agent.venue_registry.adapters = {}
            agent.venue_registry.register(_FakeDirectionalAdapter(quote=False))
            agent.llm_router = _Router(parsed={"prob_up": 0.9})
            summary = asyncio.run(agent._run_directional_lane())
            assert summary["opened"] == 0
            assert any("did not answer" in r for r in summary["refused"])
        finally:
            agent.storage.close()

    def test_the_cycle_report_carries_the_lane(self, monkeypatch, tmp_path):
        import inspect
        from src.ptai.agent.v3_loop import TradingAgentV3
        source = inspect.getsource(TradingAgentV3.run_cycle)
        assert "await self._run_directional_lane()" in source
        assert '"directional": directional' in source


# ---------------------------------------------------------------------------
# 10. the panel: the record is on the page, with the warning attached
# ---------------------------------------------------------------------------

class TestTheConsoleShowsTheDirectionalRecord:
    def test_the_panel_answers_with_the_record_and_the_warning(self, monkeypatch,
                                                               tmp_path):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        import importlib
        from fastapi.testclient import TestClient

        module = importlib.import_module("src.ptai.ui.console")
        store = module.get_storage()
        store.set_paper_bankroll(200.0)
        lane = DirectionalPaperLane(store)
        quote = DirectionalQuote(venue_id="crypto_binance", symbol="BTCUSDT",
                                 bid=100.0, ask=100.2, last=100.1, is_real=True)
        plan = plan_directional_trade(quote, sigma=0.05, prob_up=0.8,
                                      purse_usd=lane.purse(), fee_pct=0.001)
        lane.open(plan, market_question="BTCUSDT higher?")
        position = store.get_open_directional_positions()[0]
        store.close_directional_position(int(position["id"]), exit_price=102.1,
                                        exit_reason="target", pnl_usd=1.5,
                                        fees_usd=0.1)

        client = TestClient(module.app)
        body = client.get("/api/console/directional").json()
        assert body["available"] is True
        assert body["closed_count"] >= 1
        assert body["wins"] >= 1
        assert "do NOT count toward the resolved" in body["note"]
        assert body["open"] == []
        assert all("exit_reason" in row for row in body["closed"])

    def test_the_panel_is_on_the_page_and_polled(self):
        import inspect
        from src.ptai.ui import console as console_module
        source = inspect.getsource(console_module)
        assert 'id="directional"' in source
        assert "loadDirectional()" in source
        assert "does <b>not</b> count" in source or "do not count" in source
