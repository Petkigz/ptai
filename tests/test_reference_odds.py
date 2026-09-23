"""
Reference odds: every anchor must be independent of the market's own price.

These tests exist because four of the five sources used to fabricate a
number. The Pinnacle one was the worst: it returned

    best_price + 0.05        # "5% edge mock"

as a "sharp sportsbook" reference at 0.80 confidence. That is the market
disagreeing with itself by a chosen offset, and alpha_engine read it to apply
a 20% score boost. A market was being boosted because it differed from a
number generated from itself.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.markets.base import Market, MarketSource
from src.ptai.strategy.reference_odds import (
    PROV_CROSS_VENUE, PROV_LIVE_API, ReferenceOdds, ReferenceOddsEngine,
    UnavailableReference, lognormal_digital_prob, _text_similarity,
)


def make_market(question, price=0.5, mid="m1", end_date=None):
    return Market(id=mid, source=MarketSource.POLYMARKET, question=question,
                  outcomes=["Yes", "No"], outcome_prices=[price, round(1 - price, 4)],
                  end_date=end_date)


# ═══════════════════════════════════════════════════════════════════════════
# The core regression: nothing may be fabricated
# ═══════════════════════════════════════════════════════════════════════════

def test_no_references_are_fabricated_when_no_source_is_reachable():
    """
    With no API key and no network, there are no anchors. Zero references is
    the correct answer - a number would be invented.
    """
    engine = ReferenceOddsEngine()
    for question, price in [
        ("Will the Lakers beat the Celtics in the NBA finals?", 0.62),
        ("Will the Fed raise rates at the next FOMC meeting?", 0.40),
        ("Will Trump win the 2028 election?", 0.52),
        ("Will BTC be above $100k by December?", 0.62),
    ]:
        refs = engine.get_all_reference_odds(make_market(question, price))
        assert refs == [], f"'{question}' produced {len(refs)} unreachable references"


def test_pinnacle_never_derives_a_reference_from_the_market_price():
    """
    The exact bug: `best_price + 0.05`. Without a key there must be no
    Pinnacle reference at all, and the recorded reason must say why.
    """
    engine = ReferenceOddsEngine()
    market = make_market("Will the Lakers win the NBA championship?", 0.62)
    assert engine.get_pinnacle_implied_prob(market) is None
    assert "pinnacle" in engine.unavailable
    assert "invent" in engine.unavailable["pinnacle"].lower()


def test_pinnacle_reference_is_independent_of_the_market_price_at_every_price():
    """
    A regression guard on the shape of the bug. The old code moved with the
    market, so the 'edge' was constant at 5% whatever the price. With no key
    the result must be None at every price, not a constant offset.
    """
    engine = ReferenceOddsEngine()
    results = {p: engine.get_pinnacle_implied_prob(
        make_market("Will the Lakers win the NBA finals?", p)) for p in (0.1, 0.3, 0.5, 0.7, 0.9)}
    assert set(results.values()) == {None}


def test_fed_funds_is_always_none_and_says_why():
    """No public CME API means no Fed anchor. It used to return 0.35/0.60/0.55."""
    engine = ReferenceOddsEngine()
    for question in ["Will the Fed raise rates at the next FOMC?",
                     "Will there be a rate cut in June?",
                     "Will the Fed hold rates unchanged?"]:
        market = make_market(question, 0.5)
        assert engine.get_fed_funds_implied_prob(market) is None
        assert "CME" in engine.unavailable["fed_funds"]


def test_polling_is_always_none_and_says_why():
    """It used to return a hardcoded 0.52 for Trump and 0.48 for Biden."""
    engine = ReferenceOddsEngine()
    for question in ["Will Trump win the 2028 election?",
                     "Will a Democrat win the presidency?",
                     "Will Republicans keep the Senate?"]:
        market = make_market(question, 0.5)
        assert engine.get_polling_reference(market) is None
        assert "polling" in engine.unavailable["polling"].lower()


def test_deribit_without_a_reachable_api_returns_none_not_an_assumed_price():
    """The old version assumed BTC was at $90,000 and decayed from there."""
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    market = make_market("Will BTC be above $100k by December?", 0.62)
    assert engine.get_deribit_implied_prob(market) is None
    assert "unreachable" in engine.unavailable["deribit"].lower()


def test_deribit_ignores_non_crypto_markets():
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    assert engine.get_deribit_implied_prob(make_market("Will Trump win?", 0.5)) is None
    assert "deribit" not in engine.unavailable      # not attempted, not failed


def test_deribit_needs_a_dollar_target_in_the_question():
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    assert engine.get_deribit_implied_prob(make_market("Will BTC rally?", 0.5)) is None
    assert "no dollar target" in engine.unavailable["deribit"]


def test_price_target_parsing_handles_suffixes_and_commas():
    parse = ReferenceOddsEngine._extract_price_target
    assert parse("Will BTC be above $100k?") == 100_000
    assert parse("Will ETH reach $5000?") == 5_000
    assert parse("Will BTC hit $150,000?") == 150_000
    assert parse("Will BTC reach $2M?") == 2_000_000
    assert parse("Will BTC go up?") is None


def test_ensemble_returns_none_rather_than_the_market_price():
    """
    An ensemble equal to the market price would report zero edge while
    implying several independent sources had confirmed it.
    """
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    market = make_market("Will BTC be above $100k by December?", 0.62)
    assert engine.get_ensemble_reference(market) is None


def test_unavailable_sources_are_reported_not_hidden():
    """
    Silently dropping the sources that cannot be reached would let a caller
    believe it had more anchors than it did.
    """
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    market = make_market("Will the Fed cut rates at the next FOMC?", 0.5)
    engine.get_all_reference_odds(market)
    missing = engine.get_unavailable_references(market)
    names = {u.source for u in missing}
    assert "fed_funds" in names
    assert all(isinstance(u, UnavailableReference) and u.reason for u in missing)


def test_every_returned_reference_is_marked_non_synthetic():
    """Provenance must distinguish a fetched price from a cross-venue one."""
    engine = ReferenceOddsEngine(http_get=lambda url, params: None)
    kalshi = make_market("Will BTC be above $100k by December?", 0.55, mid="k1")
    market = make_market("Will BTC be above $100k by December?", 0.62)
    refs = engine.get_all_reference_odds(market, kalshi_markets=[kalshi])
    for r in refs:
        assert r.is_synthetic is False
        assert r.provenance in (PROV_LIVE_API, PROV_CROSS_VENUE)


# ═══════════════════════════════════════════════════════════════════════════
# Real references do work when data is supplied
# ═══════════════════════════════════════════════════════════════════════════

def test_kalshi_cross_venue_reference_is_a_real_price():
    """The one source that was already honest, still works."""
    engine = ReferenceOddsEngine()
    market = make_market("Will BTC be above $100k by December 31?", 0.62)
    kalshi = make_market("Will BTC be above $100k by December 31?", 0.55, mid="k1")
    result = engine.get_kalshi_reference(market, kalshi_markets=[kalshi])
    assert result is not None
    prob, conf, reasoning = result
    assert prob == 0.55                 # the actual Kalshi price, not an offset
    assert "Kalshi" in reasoning


def test_kalshi_without_markets_is_unavailable_not_zero():
    engine = ReferenceOddsEngine()
    assert engine.get_kalshi_reference(make_market("Will BTC rise?", 0.5)) is None
    assert "no Kalshi markets" in engine.unavailable["kalshi"]


def test_kalshi_rejects_a_weak_match():
    """A false match produces a reference for the wrong event."""
    engine = ReferenceOddsEngine()
    market = make_market("Will BTC be above $100k by December?", 0.62)
    unrelated = make_market("Will the Super Bowl go to overtime?", 0.30, mid="k2")
    assert engine.get_kalshi_reference(market, kalshi_markets=[unrelated]) is None


def test_manifold_reference_uses_the_returned_probability():
    payload = [{"question": "Will BTC be above $100k by December?", "probability": 0.41}]
    engine = ReferenceOddsEngine(http_get=lambda url, params: payload)
    result = engine.get_manifold_reference(
        make_market("Will BTC be above $100k by December?", 0.62))
    assert result is not None
    prob, conf, reasoning = result
    assert prob == 0.41
    assert conf <= 0.45, "play-money market must stay a weak anchor"
    assert "play-money" in reasoning


def test_manifold_rejects_a_dissimilar_match():
    payload = [{"question": "Will it rain in Paris tomorrow?", "probability": 0.30}]
    engine = ReferenceOddsEngine(http_get=lambda url, params: payload)
    assert engine.get_manifold_reference(
        make_market("Will BTC be above $100k by December?", 0.62)) is None


def test_metaculus_reference_uses_the_returned_forecast():
    payload = {"results": [{
        "title": "Will BTC be above $100k by December?",
        "prediction_timeseries": [{"c": 0.38}, {"c": 0.44}],
    }]}
    engine = ReferenceOddsEngine(http_get=lambda url, params: payload)
    result = engine.get_metaculus_reference(
        make_market("Will BTC be above $100k by December?", 0.62))
    assert result is not None
    assert result[0] == 0.44            # latest point in the series
    assert result[1] <= 0.50


def test_deribit_prices_a_target_from_the_returned_option_surface():
    """
    With a real-shaped payload the reference must come from the mark price and
    the nearest-strike IV, not from an assumed spot.
    """
    payload = {"result": [
        {"strike": 90000, "implied_volatility": 0.50, "mark_price": 0.15,
         "underlying_price": 90000},
        {"strike": 100000, "implied_volatility": 0.58, "mark_price": 0.09,
         "underlying_price": 90000},
        {"strike": 120000, "implied_volatility": 0.72, "mark_price": 0.03,
         "underlying_price": 90000},
    ]}
    engine = ReferenceOddsEngine(http_get=lambda url, params: payload)
    market = make_market("Will BTC be above $100k by December?", 0.62,
                         end_date=datetime.now(timezone.utc) + timedelta(days=30))
    result = engine.get_deribit_implied_prob(market)
    assert result is not None
    prob, conf, reasoning = result
    # nearest strike to 100k is 100k at 58% IV
    assert "58.0%" in reasoning
    assert 0.0 < prob < 0.5, f"90k -> 100k in 30d at 58% IV should be unlikely, got {prob}"
    assert conf > 0


def test_deribit_confidence_falls_with_an_absurd_iv():
    """Confidence comes from the data returned, not from a constant."""
    sane = {"result": [{"strike": 100000, "implied_volatility": 0.55,
                         "mark_price": 0.1, "underlying_price": 90000}]}
    wild = {"result": [{"strike": 100000, "implied_volatility": 9.0,
                         "mark_price": 0.1, "underlying_price": 90000}]}
    market = make_market("Will BTC be above $100k by December?", 0.62,
                         end_date=datetime.now(timezone.utc) + timedelta(days=30))
    good = ReferenceOddsEngine(http_get=lambda u, p: sane).get_deribit_implied_prob(market)
    bad = ReferenceOddsEngine(http_get=lambda u, p: wild).get_deribit_implied_prob(market)
    assert good is not None and bad is not None
    assert bad[1] < good[1]


def test_deribit_without_an_underlying_price_refuses():
    """mark_price is a fraction of spot, so it cannot recover spot alone."""
    payload = {"result": [{"strike": 100000, "implied_volatility": 0.55, "mark_price": 0.1}]}
    engine = ReferenceOddsEngine(http_get=lambda url, params: payload)
    assert engine.get_deribit_implied_prob(
        make_market("Will BTC be above $100k by December?", 0.62)) is None
    assert "mark price" in engine.unavailable["deribit"].lower()


def test_deribit_reference_wrapper_exposes_reference_price():
    payload = {"result": [{"strike": 100000, "implied_volatility": 0.55,
                            "mark_price": 0.1, "underlying_price": 90000}]}
    engine = ReferenceOddsEngine(http_get=lambda u, p: payload)
    ref = engine.get_deribit_reference(
        make_market("Will BTC be above $100k by December?", 0.62,
                    end_date=datetime.now(timezone.utc) + timedelta(days=30)))
    assert ref is not None and ref.source == "deribit"
    assert 0 < ref.reference_price < 1


def test_out_of_range_reference_price_is_dropped():
    """A source returning a nonsense price must not enter the ensemble."""
    engine = ReferenceOddsEngine()
    engine.get_manifold_reference = lambda m: (1.4, 0.9, "bogus")
    refs = engine.get_all_reference_odds(make_market("Will BTC be above $100k?", 0.62))
    assert refs == []
    assert "out-of-range" in engine.unavailable["manifold"]


def test_a_source_raising_does_not_kill_the_others():
    engine = ReferenceOddsEngine()

    def boom(market):
        raise ValueError("upstream exploded")

    engine.get_manifold_reference = boom
    kalshi = make_market("Will BTC be above $100k by December 31?", 0.55, mid="k1")
    market = make_market("Will BTC be above $100k by December 31?", 0.62)
    refs = engine.get_all_reference_odds(market, kalshi_markets=[kalshi])
    assert [r.source for r in refs] == ["kalshi"]
    assert "upstream exploded" in engine.unavailable["manifold"]


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble and trade decisions
# ═══════════════════════════════════════════════════════════════════════════

def test_ensemble_weights_by_confidence_and_uses_only_real_sources():
    engine = ReferenceOddsEngine()
    kalshi = make_market("Will BTC be above $100k by December 31?", 0.55, mid="k1")
    market = make_market("Will BTC be above $100k by December 31?", 0.62)
    result = engine.get_ensemble_reference(market, kalshi_markets=[kalshi])
    assert result is not None
    price, conf, reasoning, refs = result
    assert price == pytest.approx(0.55, abs=1e-6)     # one source, so it is that source
    assert "Ensemble of 1 real source" in reasoning
    assert "no anchor from" in reasoning              # the gaps are stated
    assert all(not r.is_synthetic for r in refs)


def test_should_trade_respects_the_per_source_threshold():
    engine = ReferenceOddsEngine()
    market = make_market("Will BTC be above $100k by December 31?", 0.62)
    close = make_market("Will BTC be above $100k by December 31?", 0.60, mid="k1")
    refs = engine.get_all_reference_odds(market, kalshi_markets=[close])
    assert refs and refs[0].should_trade is False      # 2% edge is below the 5% rule

    far = make_market("Will BTC be above $100k by December 31?", 0.30, mid="k2")
    refs = engine.get_all_reference_odds(market, kalshi_markets=[far])
    assert refs and refs[0].should_trade is True


def test_edge_sign_is_reference_minus_market():
    engine = ReferenceOddsEngine()
    market = make_market("Will BTC be above $100k by December 31?", 0.62)
    kalshi = make_market("Will BTC be above $100k by December 31?", 0.70, mid="k1")
    ref = engine.get_all_reference_odds(market, kalshi_markets=[kalshi])[0]
    assert ref.edge == pytest.approx(0.08, abs=1e-6)   # market is underpriced
    assert ref.abs_edge == pytest.approx(0.08, abs=1e-6)


def test_alpha_engine_gets_no_boost_from_a_fabricated_reference():
    """
    alpha_engine applies a 20% score boost when a reference confirms an edge.
    The old Pinnacle fabrication cleared that threshold on every sports market
    by construction. With no key there must be no reference and no boost.
    """
    from src.ptai.strategy.alpha_engine import AlphaEngine

    alpha = AlphaEngine()
    market = make_market("Will the Lakers win the NBA finals?", 0.62)
    refs = alpha.reference_odds.get_all_reference_odds(market)
    assert refs == [], "a fabricated reference would leak into the alpha score"

    # signature is (base_score, market, context)
    baseline = alpha.calculate_alpha_adjusted_score(1.0, market)
    # a fabricated 5% "edge" used to clear the 0.05 threshold and apply a 20%
    # boost here; with no reference the score must be unboosted
    assert baseline < 1.2, f"score {baseline} suggests a reference boost was applied"


def test_disabled_sources_are_not_attempted():
    calls = []
    engine = ReferenceOddsEngine(enabled_sources=[],
                                 http_get=lambda u, p: calls.append(u))
    assert engine.get_all_reference_odds(make_market("Will BTC be above $100k?", 0.62)) == []
    assert calls == [], "a disabled source must not even be fetched"


# ═══════════════════════════════════════════════════════════════════════════
# Math
# ═══════════════════════════════════════════════════════════════════════════

def test_lognormal_digital_prob_is_monotonic_in_every_input():
    base = lognormal_digital_prob(90000, 100000, 0.55, 30)
    assert lognormal_digital_prob(100000, 100000, 0.55, 30) > base   # closer spot
    assert lognormal_digital_prob(90000, 90000, 0.55, 30) > base    # lower target
    assert lognormal_digital_prob(90000, 100000, 0.55, 365) > base  # more time
    assert lognormal_digital_prob(90000, 150000, 0.55, 30) < base   # further target


def test_lognormal_digital_prob_at_the_money_is_about_half():
    """With zero drift, an at-the-money digital is just under 0.5 (the -sigma^2/2 term)."""
    p = lognormal_digital_prob(100000, 100000, 0.55, 30)
    assert 0.40 < p < 0.50


def test_lognormal_digital_prob_rejects_bad_inputs():
    assert lognormal_digital_prob(0, 100000, 0.55, 30) is None
    assert lognormal_digital_prob(90000, 0, 0.55, 30) is None
    assert lognormal_digital_prob(90000, 100000, -0.5, 30) is None
    assert lognormal_digital_prob(90000, 100000, 0.55, 0) is None


def test_text_similarity_is_conservative():
    assert _text_similarity("Will BTC be above $100k by December?",
                            "Will BTC be above $100k by December?") == pytest.approx(1.0)
    assert _text_similarity("Will BTC be above $100k?",
                            "Will the Super Bowl go to overtime?") < 0.2
    assert _text_similarity("", "anything") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def test_report_states_which_sources_are_real_and_which_are_not():
    """
    The old report advertised seven sources at 65-85% confidence, four of
    which were hardcoded numbers.
    """
    report = ReferenceOddsEngine().get_report()
    assert report["sources"]["fed_funds"]["status"] == "unavailable"
    assert report["sources"]["polling"]["status"] == "unavailable"
    assert report["sources"]["pinnacle"]["configured"] is False
    assert len(report["removed_fabrications"]) == 4


def test_report_reflects_configuration():
    report = ReferenceOddsEngine(odds_api_key="secret").get_report()
    assert report["sources"]["pinnacle"]["configured"] is True


def test_fetch_counts_track_what_was_actually_obtained():
    engine = ReferenceOddsEngine()
    kalshi = make_market("Will BTC be above $100k by December 31?", 0.55, mid="k1")
    engine.get_all_reference_odds(
        make_market("Will BTC be above $100k by December 31?", 0.62), kalshi_markets=[kalshi])
    assert engine.fetch_counts["kalshi"] == 1
    assert engine.fetch_counts["pinnacle"] == 0
    assert engine.last_run_at is not None
