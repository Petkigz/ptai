"""
Liquidity rewards and market making: quoted from the real book, not assumed.

The spread was invented in two places -

    spread = 0.02  # assume 2% spread

- even though OrderbookAnalyzer already computes a real spread from the bid and
ask. The one number that decides whether market making is profitable was a
constant, and a flattering one.

The reward side asserted 0.1%/day (36.5% APR) with no source, and the gate
meant to filter bad opportunities tested `reward_apr > 0.1`, which against that
constant could never fail. `mid_price = market.best_price` quoted around one
side of the book as though it were the midpoint, and `spread/2` was presented
as income rather than the gross upper bound it is.
"""
import pytest

from src.ptai.markets.base import Market, MarketSource
from src.ptai.strategy.liquidity_rewards import (
    BookQuote, LiquidityRewardsEngine, MarketMakingQuote, RewardConfig,
    _first_price, _top_size_usd,
)

BOOK = {"bids": [{"price": 0.48, "size": 400}], "asks": [{"price": 0.52, "size": 350}]}
TIGHT = {"bids": [{"price": 0.495, "size": 400}], "asks": [{"price": 0.505, "size": 350}]}
WIDE = {"bids": [{"price": 0.30, "size": 400}], "asks": [{"price": 0.70, "size": 350}]}
CROSSED = {"bids": [{"price": 0.55, "size": 400}], "asks": [{"price": 0.45, "size": 350}]}


def make_market(price=0.5, liquidity=20000.0, volume=50000.0, book=None, mid="m1"):
    m = Market(id=mid, source=MarketSource.POLYMARKET, question="Will X happen?",
               outcomes=["Yes", "No"], outcome_prices=[price, round(1 - price, 4)],
               liquidity=liquidity, volume_24h=volume)
    if book is not None:
        m.raw["orderbook"] = book
    return m


@pytest.fixture
def engine():
    return LiquidityRewardsEngine(bankroll=50.0)


@pytest.fixture
def rewarded_engine():
    return LiquidityRewardsEngine(
        bankroll=50.0,
        rewards=RewardConfig(reward_rate_per_day=0.001, maker_fee=0.0,
                             min_size_to_qualify=5.0, source="measured"))


# ═══════════════════════════════════════════════════════════════════════════
# Reading the book
# ═══════════════════════════════════════════════════════════════════════════

def test_a_real_book_yields_a_real_midpoint_and_spread(engine):
    b = engine.read_book(make_market(book=BOOK))
    assert b.has_book
    assert b.mid == pytest.approx(0.50)
    assert b.spread == pytest.approx(0.04)
    assert b.bid == 0.48 and b.ask == 0.52
    assert b.source == "orderbook"


def test_a_missing_book_is_reported_not_assumed(engine):
    """The old code substituted 0.02 and carried on quoting."""
    b = engine.read_book(make_market())
    assert not b.has_book
    assert b.mid is None and b.spread is None
    assert b.warnings


def test_a_spread_figure_without_sides_is_not_a_book(engine):
    """
    `{"spread": 0.02}` with no bid/ask is exactly the assumed constant this
    rewrite removes; it cannot produce a midpoint.
    """
    b = engine.read_book(make_market(book={"spread": 0.02}))
    assert not b.has_book
    assert b.mid is None
    assert b.source == "spread_only"


def test_a_crossed_book_is_refused(engine):
    b = engine.read_book(make_market(book=CROSSED))
    assert not b.has_book
    assert any("crossed" in w for w in b.warnings)


def test_top_of_book_size_is_captured(engine):
    b = engine.read_book(make_market(book=BOOK))
    assert b.bid_size_usd > 0 and b.ask_size_usd > 0


@pytest.mark.parametrize("levels,expected", [
    ([{"price": 0.5, "size": 10}], 0.5),
    ([[0.5, 10]], 0.5),
    ([0.5, 10], 0.5),
    ({"price": 0.5}, 0.5),
    (0.5, 0.5),
    ([], None),
    (None, None),
    ([{}], None),
])
def test_first_price_handles_the_shapes_adapters_emit(levels, expected):
    assert _first_price(levels) == expected


def test_top_size_converts_shares_to_notional():
    # 400 shares at 0.48 -> $192 notional
    assert _top_size_usd([{"price": 0.48, "size": 400}], 0.48) == pytest.approx(192.0)
    assert _top_size_usd([[0.48, 400]], 0.48) == pytest.approx(192.0)
    assert _top_size_usd(None, 0.48) == 0.0
    assert _top_size_usd([{"price": 0.48}], 0.48) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# The assumed spread is gone
# ═══════════════════════════════════════════════════════════════════════════

def test_the_spread_comes_from_the_book(engine):
    est = engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.spread == pytest.approx(0.04)
    assert est.spread_source == "orderbook"


def test_different_books_give_different_spreads(engine):
    """A constant could not do this."""
    a = engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    b = engine.estimate_rewards(make_market(book=TIGHT), amount_usd=5.0)
    assert a.spread != b.spread
    assert a.spread > b.spread


def test_no_book_means_no_spread_capture(engine):
    est = engine.estimate_rewards(make_market(), amount_usd=5.0)
    assert est.spread is None
    assert est.spread_capture_per_trade == 0.0
    assert est.should_provide_liquidity is False


def test_spread_capture_is_flagged_as_an_upper_bound(engine):
    """
    spread/2 assumes both sides fill and ignores adverse selection. Presenting
    it as income was the bug.
    """
    est = engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.gross_capture_is_upper_bound is True
    assert "upper bound" in est.reasoning


def test_maker_fee_reduces_the_capture():
    free = LiquidityRewardsEngine(bankroll=50.0, rewards=RewardConfig(maker_fee=0.0))
    costly = LiquidityRewardsEngine(bankroll=50.0, rewards=RewardConfig(maker_fee=0.01))
    a = free.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    b = costly.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert b.spread_capture_per_trade < a.spread_capture_per_trade
    assert a.spread_capture_per_trade - b.spread_capture_per_trade == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Rewards are not invented
# ═══════════════════════════════════════════════════════════════════════════

def test_no_configured_rate_means_no_apr(engine):
    """The old default asserted 36.5% APR from nothing."""
    assert engine.rewards.reward_available is False
    est = engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.reward_apr == 0.0
    assert est.reward_per_day_usd == 0.0
    assert est.reward_available is False
    assert any("no liquidity reward rate configured" in b for b in est.blockers)


def test_a_configured_rate_produces_a_real_apr(rewarded_engine):
    est = rewarded_engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.reward_available is True
    assert est.reward_apr == pytest.approx(0.365)
    assert est.reward_per_day_usd == pytest.approx(0.005)


def test_reward_config_has_no_default_rate():
    cfg = RewardConfig()
    assert cfg.reward_rate_per_day is None
    assert cfg.reward_available is False
    assert cfg.reward_apr == 0.0


def test_an_inactive_reward_program_is_not_available():
    cfg = RewardConfig(reward_rate_per_day=0.01, active=False)
    assert cfg.reward_available is False


def test_a_zero_rate_is_not_a_reward():
    assert RewardConfig(reward_rate_per_day=0.0).reward_available is False


def test_the_apr_gate_is_not_decorative(rewarded_engine):
    """
    The old `reward_apr > 0.1` test could never fail against the hardcoded
    0.365. A low configured rate must now actually fail it.
    """
    low = LiquidityRewardsEngine(bankroll=50.0, rewards=RewardConfig(
        reward_rate_per_day=0.0001, min_size_to_qualify=5.0))   # 3.65% APR
    est = low.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.reward_apr < 0.1
    # spread capture alone can still justify it, so check the reasoning is honest
    assert "3.7% APR" in est.reasoning or "3.6% APR" in est.reasoning


def test_no_return_at_all_means_do_not_provide():
    """No reward rate and a spread too wide to capture: nothing to earn."""
    e = LiquidityRewardsEngine(bankroll=50.0, max_quotable_spread=0.01)
    est = e.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.should_provide_liquidity is False
    assert any("no positive expected return" in b or "outside the quotable range" in b
               for b in est.blockers)


def test_a_profitable_setup_is_approved(rewarded_engine):
    est = rewarded_engine.estimate_rewards(make_market(book=BOOK), amount_usd=5.0)
    assert est.should_provide_liquidity is True
    assert est.blockers == []


# ═══════════════════════════════════════════════════════════════════════════
# Quoting
# ═══════════════════════════════════════════════════════════════════════════

def test_no_book_means_no_quote(engine):
    """
    A two-sided quote needs a two-sided book. Quoting around best_price with an
    assumed spread is how a maker gets picked off.
    """
    q = engine.create_quotes(make_market(), inventory=0, amount_usd=5.0)
    assert q.should_quote is False
    assert q.bid_price == 0.0 and q.ask_price == 0.0
    assert q.mid_source == "none"
    assert q.blockers


def test_quotes_straddle_the_real_midpoint(rewarded_engine):
    q = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=0, amount_usd=5.0)
    assert q.should_quote is True
    assert q.mid_price == pytest.approx(0.50)
    assert q.bid_price < q.mid_price < q.ask_price
    assert q.mid_source == "orderbook"


def test_the_midpoint_is_not_best_price(rewarded_engine):
    """
    best_price is one side. With this book it is 0.5 by construction, so use an
    asymmetric book where the two differ.
    """
    book = {"bids": [{"price": 0.30, "size": 400}], "asks": [{"price": 0.36, "size": 350}]}
    market = make_market(price=0.30, book=book)
    q = rewarded_engine.create_quotes(market, inventory=0, amount_usd=5.0)
    assert market.best_price == 0.30
    assert q.mid_price == pytest.approx(0.33)
    assert q.mid_price != market.best_price


def test_a_wide_book_is_not_quoted(rewarded_engine):
    q = rewarded_engine.create_quotes(make_market(book=WIDE), inventory=0, amount_usd=5.0)
    assert q.should_quote is False
    assert any("exceeds" in b for b in q.blockers)


def test_quotes_fit_inside_the_observed_spread(rewarded_engine):
    """Quoting wider than the book earns nothing."""
    q = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=0, amount_usd=5.0)
    assert q.spread <= 0.04


def test_a_thin_book_is_not_quoted(rewarded_engine):
    thin = {"bids": [{"price": 0.48, "size": 2}], "asks": [{"price": 0.52, "size": 2}]}
    q = rewarded_engine.create_quotes(make_market(book=thin), inventory=0, amount_usd=5.0)
    assert q.should_quote is False
    assert any("thin" in b for b in q.blockers)


def test_an_illiquid_market_is_not_quoted(rewarded_engine):
    q = rewarded_engine.create_quotes(
        make_market(liquidity=100.0, book=BOOK), inventory=0, amount_usd=5.0)
    assert q.should_quote is False
    assert any("liquidity" in b for b in q.blockers)


# ═══════════════════════════════════════════════════════════════════════════
# Inventory
# ═══════════════════════════════════════════════════════════════════════════

def test_a_long_position_quotes_down_to_sell(rewarded_engine):
    flat = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=0, amount_usd=5.0)
    long_ = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=2.5, amount_usd=5.0)
    assert long_.ask_price < flat.ask_price
    assert long_.bid_price < flat.bid_price
    assert long_.ask_size_usd > long_.bid_size_usd


def test_a_short_position_quotes_up_to_buy_back(rewarded_engine):
    flat = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=0, amount_usd=5.0)
    short = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=-2.5, amount_usd=5.0)
    assert short.bid_price > flat.bid_price
    assert short.bid_size_usd > short.ask_size_usd


def test_at_the_inventory_limit_quoting_stops(rewarded_engine):
    """$50 bankroll, 10% cap = $5. At the cap there is no room to add risk."""
    q = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=5.0, amount_usd=5.0)
    assert q.should_quote is False
    assert any("inventory" in b for b in q.blockers)


def test_sizes_never_collapse_to_zero(rewarded_engine):
    """A zero-sized quote silently stops trading without saying so."""
    q = rewarded_engine.create_quotes(make_market(book=BOOK), inventory=5.0, amount_usd=5.0)
    assert q.bid_size_usd > 0 and q.ask_size_usd > 0


def test_inventory_limit_tracks_the_bankroll():
    assert LiquidityRewardsEngine(bankroll=50.0).inventory_limit == pytest.approx(5.0)
    assert LiquidityRewardsEngine(bankroll=1000.0).inventory_limit == pytest.approx(100.0)
    assert LiquidityRewardsEngine(bankroll=100.0,
                                  inventory_fraction=0.05).inventory_limit == pytest.approx(5.0)


def test_a_qualified_amount_above_the_limit_is_blocked(rewarded_engine):
    est = rewarded_engine.estimate_rewards(make_market(book=BOOK), amount_usd=50.0)
    assert est.should_provide_liquidity is False
    assert any("exceeds" in b for b in est.blockers)


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def test_report_has_no_mock_key(engine):
    """The old report advertised the fabricated rate under a key named 'mock'."""
    assert "mock" not in engine.get_report()


def test_report_lists_what_was_removed(engine):
    report = engine.get_report()
    assert len(report["removed_assumptions"]) == 5
    assert any("0.02" in a for a in report["removed_assumptions"])
    assert any("36.5" in a for a in report["removed_assumptions"])


def test_report_states_whether_rewards_are_configured(engine, rewarded_engine):
    assert engine.get_report()["rewards"]["configured"] is False
    assert rewarded_engine.get_report()["rewards"]["configured"] is True
    assert rewarded_engine.get_report()["rewards"]["source"] == "measured"


def test_dataclass_defaults_are_inert():
    assert BookQuote().has_book is False
    q = MarketMakingQuote(market_id="m", bid_price=0, ask_price=0, bid_size_usd=0,
                          ask_size_usd=0, spread=0, mid_price=0, inventory=0,
                          inventory_limit=0, reward_estimate=0, should_quote=False,
                          reasoning="")
    assert q.should_quote is False and q.mid_source == "none"
