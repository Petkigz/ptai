"""
A CRYPTO PRICE IS NOT A PROBABILITY.

The operator's 2026-09-29 log priced the crypto lane through the prediction
stack:

    Best opportunity: predictit YES edge 0.093 score 0.089 strategy momentum |
    Will 0GUSDT close higher in 24h? (Crypto momentum)

    whitebit-BCH_TRY ... spread 6300.0% ... $50 math: fee 2.0% + gas 0.0% +
    spread 6300.0% = 6302.0% cost must exceed to break even

    whitebit-ADA_PERP ... Executable: pay 0.748 -> -0.953 REFUSED: no
    executable edge: fair 0.100 vs the 0.748 a share actually costs

Three separate lies produced those lines:

  * a venue label: the market is a Binance crypto pair, and its venue was read
    from `market.source`, a three-value prediction enum - so it printed
    "predictit" for a coin;
  * a probability: 'ADA_PERP at 0.748' is $0.748, not a 74.8% chance, and the
    order book behind it is a currency book, so a spread against a 0-1
    probability is meaningless (hence 6300%);
  * a forecast: the ensemble still produced 'fair 0.9' for a price level, which
    a reader would take as a probability of 90%.

These tests pin: financial markets say they are financial, the pricing lane
refuses them with that reason instead of inventing a probability, the book
refuses to present a currency quote as a probability book, and the venue on an
opportunity is the market's own venue id.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

import pytest

from src.ptai.markets.base import DataMode, Market, MarketSource


def _crypto_market(**overrides) -> Market:
    raw = {"venue": "crypto_binance", "symbol": "0GUSDT", "last_price": 3.1,
           "data_source": "binance_api", "probability_market": False,
           "quote_scale": "currency"}
    raw.update(overrides.pop("raw", {}))
    fields = dict(id="CRYPTO-0GUSDT", source=MarketSource.PREDICTIT,
                  question="Will 0GUSDT close higher in 24h? (Crypto momentum)",
                  outcome_prices=[0.52, 0.48], volume_24h=50_000,
                  liquidity=5_000, venue_id="crypto_binance",
                  venue_type="financial", data_mode=DataMode.LIVE,
                  raw=raw)
    fields.update(overrides)
    return Market(**fields)


def _prediction_market(**overrides) -> Market:
    fields = dict(id="poly-1", source=MarketSource.POLYMARKET,
                  question="Will the bill pass?", outcome_prices=[0.42, 0.58],
                  volume_24h=50_000, liquidity=5_000, venue_id="polymarket",
                  data_mode=DataMode.LIVE, raw={"venue_id": "polymarket"})
    fields.update(overrides)
    return Market(**fields)


# ----------------------------------------------------------------------
# the flag
# ----------------------------------------------------------------------

class TestAFinancialMarketSaysSo:
    def test_a_prediction_market_is_a_probability_market_by_default(self):
        assert _prediction_market().is_probability_market is True

    def test_a_crypto_market_is_not(self):
        assert _crypto_market().is_probability_market is False

    def test_the_binance_adapter_marks_its_markets(self):
        from src.ptai.venues.crypto_adapter import CryptoAdapter
        source = inspect.getsource(CryptoAdapter.discover_markets)
        assert '"probability_market": False' in source
        assert '"quote_scale": "currency"' in source

    def test_the_whitebit_adapter_marks_its_markets(self):
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter
        source = inspect.getsource(WhiteBITAdapter.discover_markets)
        assert '"probability_market": False' in source

    def test_a_market_with_no_raw_is_still_a_probability_market(self):
        market = _prediction_market()
        market.raw = None
        assert market.is_probability_market is True


# ----------------------------------------------------------------------
# the book
# ----------------------------------------------------------------------

class TestACurrencyBookIsNotAProbabilityBook:
    @pytest.mark.asyncio
    async def test_the_exchange_refuses_before_it_even_fetches(self):
        from src.ptai.venues.whitebit_adapter import WhiteBITAdapter
        book = await WhiteBITAdapter().get_orderbook(
            _crypto_market(id="whitebit-ADA_PERP",
                           venue_id="whitebit",
                           raw={"venue": "whitebit", "symbol": "ADA_PERP"}))
        assert book["source"] == "not_a_probability_book"
        assert book["executable"] is False
        assert book["validated"] is False
        assert book["scale"] == "currency"
        assert "no Yes share" in book["warning"]
        # nothing that could be mistaken for a priceable book
        assert "spread" not in book
        assert book["is_real"] is True, "the exchange's book IS real - it is the wrong kind"

    @pytest.mark.asyncio
    async def test_the_refusal_is_in_the_code_before_the_http_call(self):
        from src.ptai.venues import whitebit_adapter
        source = inspect.getsource(whitebit_adapter.WhiteBITAdapter.get_orderbook)
        guard = source.index("is_probability_market")
        fetch = source.index("requests.get")
        assert guard < fetch, ("the refusal must not depend on the network "
                               "answering, or it cannot be relied on")


# ----------------------------------------------------------------------
# the pricing lane
# ----------------------------------------------------------------------

class TestThePricingLaneRefusesThem:
    def test_the_screen_will_not_spend_model_time_on_them(self):
        from src.ptai.agent.v3_loop import TradingAgentV3

        class _Engine:
            min_volume_24h = 500
            min_liquidity = 100
            max_spread = 0.10

        class _Self:
            strategy_engine_v3 = _Engine()

        score, why = TradingAgentV3._screen_score(
            _Self(), _crypto_market(),
            {"validated": True, "spread": 0.02, "depth": 5000})
        assert score == -1.0
        assert "quotes a price, not a probability" in why

    @pytest.mark.asyncio
    async def test_the_venue_scan_counts_them_once_with_the_reason(self):
        from src.ptai.strategy.strategy_engine import StrategyEngineV3
        engine = StrategyEngineV3()
        markets = [_crypto_market(id=f"CRYPTO-{i}USDT") for i in range(4)]
        report, opps = await engine.scan_venue("crypto_binance", markets,
                                               context_provider=None)
        assert opps == []
        assert report.total_discovered == 4
        assert report.candidates == 0

    def test_the_reason_is_a_sentence_not_a_zero(self):
        from src.ptai.strategy import strategy_engine
        source = inspect.getsource(strategy_engine.StrategyEngineV3.scan_venue)
        assert "quotes PRICES, not probabilities" in source
        assert "no Yes share" in source


# ----------------------------------------------------------------------
# the venue on an opportunity is the market's venue
# ----------------------------------------------------------------------

class TestTheVenueIsTheVenues:
    def test_momentum_reports_the_markets_own_venue_id(self):
        from src.ptai.strategy.momentum import MomentumEngine, MomentumSignal
        market = _crypto_market()
        signal = MomentumSignal(market_id=market.id, strategy="momentum",
                                price_velocity=0.05, volume_trend=1.0,
                                estimated_edge=0.09, confidence=0.8,
                                reasoning="momentum", should_trade=True)
        opps = MomentumEngine().to_venue_opportunities(market, [signal])
        assert opps, "the strategy still produces its own signal"
        assert opps[0].venue_id == "crypto_binance"
        assert opps[0].venue_id != "predictit"

    def test_the_venue_lookup_is_ordered_by_what_routes(self):
        from src.ptai.strategy.momentum import MomentumEngine
        source = inspect.getsource(MomentumEngine.to_venue_opportunities)
        # market.venue_id first; the MarketSource enum is only the last resort
        assert source.index("getattr(market, 'venue_id'") < \
            source.index("getattr(market, 'source'")
        assert "venue_id=str(_venue_id).lower()" in source
