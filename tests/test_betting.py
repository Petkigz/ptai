"""
Tests for the betting / sports exchange subsystem (V11).

These are money-maths tests, not smoke tests. Every assertion below is a
property that must hold for real money to be safe: de-vigged books sum to
one, lay liability is computed from odds not stake, a green-up equalises
profit across both outcomes, a surebet's stake allocation returns the same
amount whichever outcome wins, and the engine refuses to act when it has
no real data.
"""
import asyncio
import math

import pytest

from src.ptai.betting.odds_math import (
    american_to_decimal,
    analyse_book,
    betfair_tick_size,
    closing_line_value,
    commission_adjusted_decimal,
    decimal_to_american,
    decimal_to_fractional,
    devig,
    devig_power,
    devig_proportional,
    devig_shin,
    dutch,
    expected_value,
    find_arbitrage,
    ArbitrageLeg,
    green_up_stake,
    implied_probability,
    is_sharp_process,
    kelly_fraction,
    ladder_walk,
    lay_liability,
    lay_stake_for_liability,
    next_ladder_step,
    parse_odds,
    snap_to_ladder,
)
from src.ptai.betting.exchange import (
    BetSide,
    ExchangeAccount,
    ExchangeBook,
    ExchangeMarketMaker,
    OrderStatus,
    PriceLevel,
    size_back,
    size_lay,
)
from src.ptai.betting.models import (
    elo_forecast,
    poisson_soccer,
    scoreline_matrix,
    shrink_to_base_rate,
    SportsModelEngine,
)
from src.ptai.betting.arb import (
    BookmakerRiskManager,
    book_vs_exchange_arb,
    dutch_outcome_group,
    hedge_to_green,
    normalise_team,
)
from src.ptai.betting.sports_data import (
    BookOdds,
    EspnProvider,
    SportsDataEngine,
    SportsEvent,
    sample_book,
    sample_events,
)
from src.ptai.betting.engine import BettingEngine
from src.ptai.markets.base import DataMode


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

def test_odds_conversions():
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200
    assert decimal_to_fractional(2.5) == "3/2"
    assert parse_odds("+150") == 2.5
    assert parse_odds("-200") == 1.5
    assert parse_odds("3/2") == 2.5
    assert parse_odds(2.5) == 2.5
    # a bare number >= 100 is American, nobody quotes 150.0 decimal
    assert parse_odds(150) == 2.5


def test_implied_probability_and_ev():
    assert implied_probability(2.0) == 0.5
    assert expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Vig
# ---------------------------------------------------------------------------

def test_book_metrics_hold_is_not_margin():
    """Margin 6% on a 1.06 book means hold is 5.66%, not 6%."""
    m = analyse_book([1.91, 1.91])
    assert m.overround == pytest.approx(1.0471, abs=1e-3)
    assert m.margin_pct == pytest.approx(4.71, abs=0.02)
    assert m.theoretical_hold_pct == pytest.approx(4.5, abs=0.05)
    assert not m.is_arbitrage

    fair = analyse_book([2.0, 2.0])
    assert fair.overround == pytest.approx(1.0)
    assert fair.is_fair_book

    arb = analyse_book([2.2, 2.2])
    assert arb.is_arbitrage
    assert arb.arb_edge_pct == pytest.approx(9.09, abs=0.02)


@pytest.mark.parametrize("prices", [
    [1.91, 1.91],
    [1.30, 5.50, 9.00],
    [2.10, 3.60, 3.40],
    [1.05, 21.0],
])
@pytest.mark.parametrize("method", ["proportional", "power", "additive"])
def test_devig_sums_to_one(prices, method):
    fair = devig(prices, method)
    assert sum(fair) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < p < 1.0 for p in fair)


def test_power_devig_shaves_favourites_harder_than_longshots():
    """
    The whole reason to prefer the power method: proportional removal
    leaves longshots overpriced, power does not.
    """
    prices = [1.15, 5.50]   # overround 1.0514, so the methods must differ
    prop = devig_proportional(prices)
    power = devig_power(prices)
    # longshot fair prob should be LOWER under power than under proportional
    assert power[1] < prop[1]
    assert power[0] > prop[0]


def test_shin_devig_returns_valid_probs_and_z():
    prices = [1.91, 1.91]
    fair, z = devig_shin(prices)
    assert sum(fair) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= z < 1.0
    assert all(0.0 < p < 1.0 for p in fair)


def test_devig_on_arb_book_still_sums_to_one():
    """An overround below 1 must not crash the solver."""
    fair = devig_power([2.2, 2.2])
    assert sum(fair) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Commission and ladder
# ---------------------------------------------------------------------------

def test_commission_applies_to_winnings_not_stake():
    # 2.00 at 5% pays 0.95 net per unit -> effective 1.95
    assert commission_adjusted_decimal(2.0, 0.05, "back") == pytest.approx(1.95)
    # a lay's effective odds rise because you keep less of the profit
    assert commission_adjusted_decimal(2.0, 0.05, "lay") == pytest.approx(2.0526, abs=1e-3)
    assert commission_adjusted_decimal(2.0, 0.0, "back") == 2.0


def test_betfair_ladder_is_non_uniform():
    assert betfair_tick_size(1.5) == 0.01
    assert betfair_tick_size(2.5) == 0.02
    assert betfair_tick_size(3.5) == 0.05
    assert betfair_tick_size(7.0) == 0.20
    assert betfair_tick_size(25.0) == 1.00
    # snapping must land on a legal tick
    assert snap_to_ladder(2.513, "down") == 2.50
    assert snap_to_ladder(2.513, "up") == 2.52
    assert next_ladder_step(1.99, "up") == 2.00
    assert ladder_walk(2.0, 2) == 2.04


# ---------------------------------------------------------------------------
# Back / lay liability
# ---------------------------------------------------------------------------

def test_lay_liability_exceeds_stake_above_even_money():
    assert lay_liability(10.0, 2.0) == 10.0
    assert lay_liability(10.0, 3.5) == 25.0
    assert lay_stake_for_liability(25.0, 3.5) == pytest.approx(10.0)


def test_green_up_equalises_profit_across_both_outcomes():
    stake, back_odds, lay_odds = 100.0, 3.0, 2.0
    lay = green_up_stake(stake, back_odds, lay_odds, commission_pct=0.0,
                         target_profit_pct=0.0)
    assert lay is not None
    profit_if_wins = stake * (back_odds - 1.0) - lay
    profit_if_loses = lay * (lay_odds - 1.0) - stake
    assert profit_if_wins == pytest.approx(profit_if_loses, abs=1e-6)
    assert profit_if_wins > 0


def test_green_up_unavailable_when_price_has_not_moved():
    """Laying at the same price you backed at must not manufacture a profit."""
    assert green_up_stake(100.0, 2.0, 2.0, 0.0, 0.0) is None


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_matches_textbook_and_respects_cap():
    # f* = (p*b - q)/b with b=1, p=0.6 -> 0.2
    assert kelly_fraction(0.6, 2.0, 1.0, cap=1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 2.0, 0.25, cap=1.0) == pytest.approx(0.05)
    assert kelly_fraction(0.99, 2.0, 1.0, cap=0.06) == 0.06


def test_size_back_uses_commission_adjusted_odds():
    """A bet that looks +EV before commission can be -EV after it."""
    # price 2.00, model 0.51 -> raw edge +2%, after 5% commission eff odds 1.95 -> -0.55%
    s = size_back(1000.0, 0.51, 2.0, kelly_frac=1.0, cap=1.0, commission_pct=0.05, min_bet=0.0)
    assert s["stake"] == 0.0
    assert s["edge_pct"] < 0
    s2 = size_back(1000.0, 0.60, 2.0, kelly_frac=1.0, cap=1.0, commission_pct=0.0, min_bet=0.0)
    assert s2["stake"] > 0


def test_size_lay_reports_liability_not_stake():
    # Laying at 3.0 with 5% commission breaks even at a true win prob of
    # (1-c)/(X-c) = 0.95/2.95 = 0.322, so a 0.25 model has real edge.
    r = size_lay(1000.0, 0.25, 3.0, kelly_frac=1.0, cap=0.06, commission_pct=0.05, min_bet=0.0)
    assert r["stake"] > 0, r
    # liability, not stake, is what the account reserves: L*(X-1) = 2*stake
    assert r["liability"] == pytest.approx(r["stake"] * 2.0, abs=1.0)
    assert r["edge_pct"] > 0


def test_size_lay_refuses_when_the_lay_price_has_no_edge():
    # 0.35 > 0.322 breakeven, so laying here loses money
    r = size_lay(1000.0, 0.35, 3.0, kelly_frac=1.0, cap=0.06, commission_pct=0.05, min_bet=0.0)
    assert r["stake"] == 0.0


# ---------------------------------------------------------------------------
# Dutching and arbitrage
# ---------------------------------------------------------------------------

def test_dutch_gives_identical_return_whichever_outcome_wins():
    odds = {"A": 3.0, "B": 3.0, "C": 3.0}
    r = dutch(odds, 90.0)
    returns = {k: r.stakes[k] * odds[k] for k in odds}
    assert len(set(round(v, 6) for v in returns.values())) == 1
    assert r.total_staked == pytest.approx(90.0, abs=0.1)
    # 3 outcomes at 3.0 is an exactly fair book: overround 1.0, zero profit
    assert r.overround == pytest.approx(1.0, abs=1e-9)
    assert not r.is_arbitrage
    assert r.guaranteed_profit == pytest.approx(0.0, abs=0.05)


def test_dutch_arb_positive_when_overround_below_one():
    odds = {"A": 2.2, "B": 2.2}
    r = dutch(odds, 100.0)
    assert r.is_arbitrage
    assert r.guaranteed_profit > 0
    assert r.overround < 1.0


def test_dutch_guaranteed_loss_when_overround_above_one():
    r = dutch({"A": 1.91, "B": 1.91}, 100.0)
    assert not r.is_arbitrage
    assert r.guaranteed_profit < 0


def test_find_arbitrage_sizes_equal_return_and_reports_limiting_leg():
    legs = [
        ArbitrageLeg(outcome="A", venue="book1", decimal_odds=2.20, max_stake=50.0),
        ArbitrageLeg(outcome="B", venue="book2", decimal_odds=2.20),
    ]
    arb = find_arbitrage("evt", legs, total_bankroll=1000.0, max_total_exposure_pct=0.5)
    assert arb is not None
    assert arb.executable
    assert arb.edge_pct > 0
    # equal return whichever wins
    ret_a = arb.stakes["A@book1"] * 2.20
    ret_b = arb.stakes["B@book2"] * 2.20
    assert ret_a == pytest.approx(ret_b, abs=0.05)
    assert arb.guaranteed_profit == pytest.approx(ret_a - arb.total_staked, abs=0.05)
    assert arb.limiting_leg and "book1" in arb.limiting_leg


def test_find_arbitrage_rejects_synthetic_prices():
    legs = [
        ArbitrageLeg(outcome="A", venue="book1", decimal_odds=2.20, is_synthetic=True),
        ArbitrageLeg(outcome="B", venue="book2", decimal_odds=2.20),
    ]
    arb = find_arbitrage("evt", legs, 1000.0)
    assert not arb.executable
    assert any("synthetic" in b for b in arb.blockers)


def test_find_arbitrage_rejects_when_no_arb():
    legs = [
        ArbitrageLeg(outcome="A", venue="b1", decimal_odds=1.91),
        ArbitrageLeg(outcome="B", venue="b2", decimal_odds=1.91),
    ]
    arb = find_arbitrage("evt", legs, 1000.0)
    assert not arb.executable


# ---------------------------------------------------------------------------
# Exchange account mechanics
# ---------------------------------------------------------------------------

def make_book(back=None, lay=None):
    return ExchangeBook(
        market_id="m1",
        back_levels=[PriceLevel(p, s) for p, s in (back or [])],
        lay_levels=[PriceLevel(p, s) for p, s in (lay or [])],
        is_real=True,
    )


def test_book_walk_consumes_depth_and_returns_average_price():
    book = make_book(back=[(1.90, 10), (1.92, 20), (1.94, 50)])
    filled, avg = book.walk(BetSide.BACK, 25)
    assert filled == 25
    # 10 @ 1.90 + 15 @ 1.92
    assert avg == pytest.approx((10 * 1.90 + 15 * 1.92) / 25)


def test_partial_fill_is_the_normal_case():
    book = make_book(back=[(1.90, 10)], lay=[])
    acct = ExchangeAccount("ex", balance=500.0, min_bet=2.0)
    bet = acct.submit("b1", "m1", "Home", BetSide.BACK, 1.90, 50.0, book=book)
    assert bet.status == OrderStatus.PARTIALLY_MATCHED
    assert bet.matched_size == 10.0
    assert bet.unmatched_size == 40.0


def test_order_without_book_stays_pending_not_fake_filled():
    """No liquidity supplied means no fill - the engine must not invent one."""
    acct = ExchangeAccount("ex", balance=500.0, min_bet=2.0)
    bet = acct.submit("b1", "m1", "Home", BetSide.BACK, 1.90, 50.0, book=None)
    assert bet.status == OrderStatus.PENDING
    assert bet.matched_size == 0.0


def test_lay_rejected_when_liability_exceeds_balance():
    book = make_book(back=[(4.0, 100)])
    acct = ExchangeAccount("ex", balance=50.0, min_bet=2.0)
    bet = acct.submit("l1", "m1", "Home", BetSide.LAY, 4.0, 30.0, book=book)
    # liability = 30*(4-1) = 90 > 50 available
    assert bet.status == OrderStatus.REJECTED
    assert "liability" in bet.void_reason


def test_settle_pays_the_winner_and_charges_the_loser():
    book = make_book(back=[(2.0, 100)], lay=[(2.02, 100)])
    acct = ExchangeAccount("ex", balance=100.0, commission_pct=0.0, min_bet=2.0)
    acct.submit("b1", "m1", "Home", BetSide.BACK, 2.0, 50.0, book=book)
    pnl = acct.settle("m1", "Home")
    assert pnl == pytest.approx(50.0)          # 50 staked at 2.0 -> +50 profit
    assert acct.balance == pytest.approx(150.0)

    acct2 = ExchangeAccount("ex", balance=100.0, commission_pct=0.0, min_bet=2.0)
    acct2.submit("b2", "m1", "Home", BetSide.BACK, 2.0, 50.0, book=book)
    pnl2 = acct2.settle("m1", "Away")
    assert pnl2 == pytest.approx(-50.0)
    assert acct2.balance == pytest.approx(50.0)


def test_voided_market_returns_stake():
    book = make_book(back=[(2.0, 100)])
    acct = ExchangeAccount("ex", balance=100.0, min_bet=2.0)
    acct.submit("b1", "m1", "Home", BetSide.BACK, 2.0, 50.0, book=book)
    pnl = acct.settle("m1", None)
    assert pnl == 0.0
    assert acct.balance == pytest.approx(100.0)


def test_position_worst_case_and_hedged_flag():
    book = make_book(back=[(3.0, 100)], lay=[(2.0, 100)])
    acct = ExchangeAccount("ex", balance=1000.0, commission_pct=0.0, min_bet=2.0)
    acct.submit("b1", "m1", "Home", BetSide.BACK, 3.0, 50.0, book=book)
    acct.green_up("m1", "Home", 2.0, book=book, bet_id="gu1", target_profit_pct=0.0)
    pos = acct.positions["m1:Home"]
    assert pos.is_hedged()
    assert pos.worst_case() >= -0.01


# ---------------------------------------------------------------------------
# Market making
# ---------------------------------------------------------------------------

def test_market_maker_refuses_to_quote_without_edge():
    mm = ExchangeMarketMaker(commission_pct=0.05, min_edge_pct=2.0, quote_size=10.0)
    book = make_book(back=[(1.98, 100)], lay=[(2.02, 100)])
    q = mm.quote("Home", fair_prob=0.50, book=book)
    assert q.skipped
    assert "no edge" in q.skip_reason


def test_market_maker_quotes_when_model_disagrees_with_market():
    mm = ExchangeMarketMaker(commission_pct=0.05, min_edge_pct=1.0, quote_size=10.0)
    book = make_book(back=[(1.98, 100)], lay=[(2.02, 100)])
    q = mm.quote("Home", fair_prob=0.60, book=book)
    assert not q.skipped
    assert q.back_size > 0


def test_market_maker_will_not_quote_one_sided_book():
    mm = ExchangeMarketMaker()
    book = make_book(back=[(1.98, 100)], lay=[])
    q = mm.quote("Home", 0.60, book)
    assert q.skipped
    assert "one-sided" in q.skip_reason


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_poisson_scoreline_matrix_is_a_distribution():
    grid = scoreline_matrix(1.6, 1.1)
    total = sum(sum(r) for r in grid)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_poiscon_home_stronger_than_away_favours_home():
    fc = poisson_soccer(home_attack=1.4, home_defence=0.85,
                        away_attack=0.8, away_defence=1.25)
    assert fc.p_home > fc.p_away
    assert fc.p_home + fc.p_draw + fc.p_away == pytest.approx(1.0, abs=1e-6)
    assert fc.p_over_2_5 + fc.p_under_2_5 == pytest.approx(1.0, abs=1e-6)


def test_poisson_expected_goals_match_lambdas():
    fc = poisson_soccer(1.3, 1.0, 1.0, 1.0)
    grid = scoreline_matrix(fc.home_lambda, fc.away_lambda)
    eh = sum(x * sum(row) for x, row in enumerate(grid))
    assert eh == pytest.approx(fc.home_lambda, abs=0.05)


def test_elo_favourite_and_draw_band():
    close = elo_forecast(1560, 1500, home_advantage=0, draw_width=25,
                         games_played_home=40, games_played_away=40)
    assert close.p_home > close.p_away
    assert 0.0 < close.p_draw < 0.35
    assert close.p_home + close.p_draw + close.p_away == pytest.approx(1.0, abs=1e-6)

    # equal ratings -> maximum draw probability, exactly 50/50 with no HA
    even = elo_forecast(1500, 1500, home_advantage=0, draw_width=25)
    assert even.p_home == pytest.approx(even.p_away, abs=1e-6)
    assert even.p_draw > close.p_draw

    # a 260-point gap plus home advantage leaves no realistic draw
    blowout = elo_forecast(1700, 1500, home_advantage=60, draw_width=25,
                           games_played_home=40, games_played_away=40)
    assert blowout.p_home > blowout.p_away
    assert blowout.p_draw < close.p_draw
    assert blowout.p_home + blowout.p_draw + blowout.p_away == pytest.approx(1.0, abs=1e-6)


def test_elo_reliability_grows_with_games():
    few = elo_forecast(1700, 1500, games_played_home=3, games_played_away=3)
    many = elo_forecast(1700, 1500, games_played_home=60, games_played_away=60)
    assert many.reliability > few.reliability


def test_shrinkage_pulls_small_samples_to_the_prior():
    extreme = {"home": 0.99, "away": 0.01}
    shrunk = shrink_to_base_rate(extreme, n_games=1, half_life=10.0)
    assert shrunk["home"] < 0.99
    full = shrink_to_base_rate(extreme, n_games=1000, half_life=10.0)
    assert full["home"] == pytest.approx(0.99, abs=0.02)


def test_model_weight_is_low_without_games():
    eng = SportsModelEngine()
    assert eng._model_weight(0) <= 0.10
    assert eng._model_weight(100) == eng.max_model_weight


def test_blend_with_market_overrides_with_sharp_price():
    eng = SportsModelEngine()
    ev = sample_events(1)[0]
    consensus = SportsDataEngine(providers=[]).consensus(ev, "h2h")
    # build a consensus by hand
    books = sample_book(ev.home_team, ev.away_team)
    from src.ptai.betting.sports_data import ConsensusOdds
    c = ConsensusOdds(
        event=ev, market="h2h", books=books,
        fair_probs={ev.home_team: 0.52, ev.away_team: 0.48},
        best_price={ev.home_team: 1.91, ev.away_team: 1.95},
        best_book={ev.home_team: "sample-book-a", ev.away_team: "sample-book-b"},
        sharp_probs={ev.home_team: 0.60, ev.away_team: 0.40},
        mean_overround=1.05, n_books=len(books),
    )
    fc = eng.forecast_elo(ev, {"home": {"rating": 1600, "games": 40},
                               "away": {"rating": 1500, "games": 40}})
    blended = eng.blend_with_market(fc, c, {ev.home_team: "home", ev.away_team: "away"})
    # sharp says 0.60 home; blend must move toward it, not away
    assert blended.probs["home"] > fc.probs["home"] or abs(
        blended.probs["home"] - 0.60) < abs(fc.probs["home"] - 0.60)


# ---------------------------------------------------------------------------
# Team matching
# ---------------------------------------------------------------------------

def test_team_normalisation_handles_aliases():
    assert normalise_team("Manchester United FC") == "man utd"
    assert normalise_team("Man Utd") == "man utd"
    assert normalise_team("Paris Saint-Germain") == "psg"
    assert normalise_team("Tottenham Hotspur") == "spurs"
    assert normalise_team("Golden State Warriors") == "warriors"


def test_event_matching_across_feeds():
    from src.ptai.betting.arb import event_similarity
    from datetime import datetime, timezone, timedelta
    t = datetime.now(timezone.utc) + timedelta(hours=3)
    a = SportsEvent("1", "soccer", "epl", "Manchester City", "Arsenal", t)
    b = SportsEvent("2", "soccer", "epl", "Man City", "Arsenal FC", t)
    assert event_similarity(a, b) >= 0.85
    c = SportsEvent("3", "soccer", "epl", "Liverpool", "Chelsea", t)
    assert event_similarity(a, c) < 0.85


# ---------------------------------------------------------------------------
# Bookmaker risk
# ---------------------------------------------------------------------------

def test_risk_manager_blocks_limited_and_over_exposed_books():
    rm = BookmakerRiskManager(daily_book_exposure_pct=0.10)
    rm.profile("book1").limited = True
    ok, why = rm.can_bet("book1", 10.0, 1000.0)
    assert not ok and "limited" in why

    ok, why = rm.can_bet("book2", 200.0, 1000.0)
    assert not ok and "daily exposure" in why

    ok, _ = rm.can_bet("book2", 50.0, 1000.0)
    assert ok


def test_risk_manager_caps_arb_ratio():
    rm = BookmakerRiskManager(max_arb_ratio=0.35)
    for i in range(6):
        rm.record_bet("book3", 5.0, is_arb=True)
    for i in range(4):
        rm.record_bet("book3", 5.0, is_arb=False)
    ok, why = rm.can_bet("book3", 5.0, 100000.0, is_arb=True)
    assert not ok and "arb ratio" in why


def test_risk_manager_avoids_round_stakes():
    rm = BookmakerRiskManager()
    assert rm.round_stake(100.0) != 100.0
    assert rm.round_stake(7.0) == 7.0


# ---------------------------------------------------------------------------
# Book vs exchange arbitrage
# ---------------------------------------------------------------------------

def test_book_vs_exchange_arb_is_profitable_on_both_outcomes():
    ev = sample_events(1)[0]
    from src.ptai.betting.sports_data import ConsensusOdds
    soft = BookOdds(book="softbook", market="h2h",
                    outcomes={ev.home_team: 2.10, ev.away_team: 1.85})
    ex = BookOdds(book="betfair-exchange", market="h2h",
                  outcomes={ev.home_team: 2.00, ev.away_team: 1.80},
                  is_exchange=True, commission_pct=0.05)
    c = ConsensusOdds(event=ev, market="h2h", books=[soft, ex],
                      fair_probs={ev.home_team: 0.5, ev.away_team: 0.5},
                      best_price={ev.home_team: 2.10, ev.away_team: 1.85},
                      best_book={ev.home_team: "softbook", ev.away_team: "softbook"},
                      sharp_probs={}, mean_overround=1.03, n_books=2)

    results = book_vs_exchange_arb(ev, c, ex, bankroll=1000.0)
    home_arb = next(r for r in results if r.outcome == ev.home_team)
    assert home_arb.executable, home_arb.blockers

    # verify the arithmetic independently, with exchange semantics:
    # the layer wins the stake (less commission) if the selection loses,
    # and pays L*(X-1) if it wins.
    S, L = home_arb.back_stake, home_arb.lay_stake
    B, X, com = 2.10, 2.00, 0.05
    assert L == pytest.approx(S * B / (X - com), abs=0.05)
    profit_win = S * (B - 1.0) - L * (X - 1.0)
    profit_lose = L * (1.0 - com) - S
    assert profit_win == pytest.approx(profit_lose, abs=0.05)
    assert profit_win == pytest.approx(home_arb.guaranteed_profit, abs=0.05)
    assert profit_win > 0


def test_book_vs_exchange_no_arb_when_commission_eats_it():
    ev = sample_events(1)[0]
    from src.ptai.betting.sports_data import ConsensusOdds
    soft = BookOdds(book="softbook", market="h2h",
                    outcomes={ev.home_team: 1.90, ev.away_team: 1.90})
    ex = BookOdds(book="betfair-exchange", market="h2h",
                  outcomes={ev.home_team: 1.90, ev.away_team: 1.90},
                  is_exchange=True, commission_pct=0.05)
    c = ConsensusOdds(event=ev, market="h2h", books=[soft, ex],
                      fair_probs={ev.home_team: 0.5, ev.away_team: 0.5},
                      best_price={ev.home_team: 1.90, ev.away_team: 1.90},
                      best_book={ev.home_team: "softbook", ev.away_team: "softbook"},
                      sharp_probs={}, mean_overround=1.05, n_books=2)
    results = book_vs_exchange_arb(ev, c, ex, bankroll=1000.0)
    assert all(not r.executable for r in results)


# ---------------------------------------------------------------------------
# Hedge / dutch helpers
# ---------------------------------------------------------------------------

def test_hedge_to_green_equalises():
    r = hedge_to_green(100.0, 3.0, 2.0, commission_pct=0.0)
    assert r["executable"]
    assert r["profit_if_wins"] == pytest.approx(r["profit_if_loses"], abs=0.02)


def test_dutch_outcome_group_reports_reason_when_not_an_arb():
    r = dutch_outcome_group({"A": {"b1": 1.91, "b2": 1.90}, "B": {"b1": 1.91, "b2": 1.92}}, 100.0)
    assert not r["executable"]
    assert "loss" in r["reason"]


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------

def test_closing_line_value():
    assert closing_line_value(2.20, 2.00) > 0     # got a better price than the close
    assert closing_line_value(1.90, 2.00) < 0     # got a worse price
    assert closing_line_value(2.00, 2.00) == pytest.approx(0.0)


def test_sharpness_requires_a_sample():
    sharp, clv, n = is_sharp_process([0.02] * 5, min_samples=20)
    assert not sharp and n == 5
    sharp, clv, n = is_sharp_process([0.02] * 30, min_samples=20)
    assert sharp and n == 30 and clv > 0


# ---------------------------------------------------------------------------
# ESPN parser (real payload shape, served locally)
# ---------------------------------------------------------------------------

ESPN_SCOREBOARD_PAYLOAD = {
    "events": [{
        "id": "401584788",
        "date": "2026-10-01T23:30Z",
        "competitions": [{
            "id": "401584788",
            "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "venue": {"fullName": "Crypto.com Arena"},
            "odds": [{
                "provider": {"name": "ESPN BET"},
                "details": "LAL -5.5",
                "overUnder": 224.5,
                "moneyLine": {"homeOdds": -260, "awayOdds": 215},
                "overOdds": -110, "underOdds": -110,
            }],
            "competitors": [
                {"homeAway": "home", "score": "", "team": {"displayName": "Los Angeles Lakers"}},
                {"homeAway": "away", "score": "", "team": {"displayName": "Golden State Warriors"}},
            ],
        }],
    }]
}


def test_espn_parser_extracts_real_fields():
    provider = EspnProvider()
    ev = provider._parse_event(ESPN_SCOREBOARD_PAYLOAD["events"][0], "nba")
    assert ev is not None
    assert ev.home_team == "Los Angeles Lakers"
    assert ev.away_team == "Golden State Warriors"
    assert ev.commence_time.year == 2026
    assert not ev.is_live

    books = provider.odds_for_event(ev)
    h2h = [b for b in books if b.market == "h2h"]
    assert h2h, "expected an h2h book from the moneyline"
    outcomes = h2h[0].outcomes
    assert outcomes["Los Angeles Lakers"] == pytest.approx(american_to_decimal(-260))
    assert outcomes["Golden State Warriors"] == pytest.approx(american_to_decimal(215))

    spreads = [b for b in books if b.market == "spreads"]
    assert spreads and spreads[0].spread == pytest.approx(-5.5)

    totals = [b for b in books if b.market == "totals"]
    assert totals and totals[0].point == 224.5

    # the book must be recognised as carrying vig
    assert h2h[0].overround > 1.0


def test_espn_parser_handles_live_status():
    payload = {"id": "1", "date": "2026-10-01T23:30Z", "competitions": [{
        "status": {"type": {"name": "STATUS_IN_PROGRESS"}},
        "competitors": [
            {"homeAway": "home", "score": "55", "team": {"displayName": "A"}},
            {"homeAway": "away", "score": "51", "team": {"displayName": "B"}},
        ]}]}
    ev = EspnProvider()._parse_event(payload, "nba")
    assert ev.is_live
    assert ev.home_score == 55.0


# ---------------------------------------------------------------------------
# Engine: no-data refusal and end-to-end on a local feed
# ---------------------------------------------------------------------------

class _StubProvider:
    """Serves a fixed fixture+odds set so the engine path is exercised offline."""
    name = "stub"
    is_synthetic = False
    last_error = ""
    last_fetch_at = None
    requests_made = 0

    def __init__(self, events, odds_map):
        self._events = events
        self._odds = odds_map

    async def events(self, leagues=()):
        return list(self._events)

    def odds_for_event(self, event):
        return list(self._odds.get(event.event_id, []))

    def health(self):
        return {"provider": self.name, "reachable": True, "last_error": "",
                "last_fetch_at": None, "requests_made": 0, "is_synthetic": False}


class _DeadProvider:
    name = "dead"
    is_synthetic = False
    last_error = "ConnectError: no route to host"
    last_fetch_at = None
    requests_made = 0

    async def events(self, leagues=()):
        return []

    def health(self):
        return {"provider": self.name, "reachable": False,
                "last_error": self.last_error, "last_fetch_at": None,
                "requests_made": 0, "is_synthetic": False}


def test_engine_refuses_to_act_without_any_real_data():
    engine = BettingEngine(bankroll=100.0,
                           data_engine=SportsDataEngine(providers=[_DeadProvider()]))
    result = asyncio.run(engine.run_cycle(data_mode=DataMode.LIVE_SHADOW))
    assert result["ok"] is False
    assert result["opportunities"] == 0
    assert "no fixtures" in result["blockers"][0]
    assert "Invented" in result["message"]


def test_engine_end_to_end_produces_risk_checked_opportunities():
    ev = sample_events(1)[0]
    ev.event_id = "evt-1"
    books = [
        BookOdds(book="softbook", market="h2h",
                 outcomes={ev.home_team: 2.10, ev.away_team: 1.95}),
        BookOdds(book="pinnacle", market="h2h",
                 outcomes={ev.home_team: 2.05, ev.away_team: 1.90}),
    ]
    engine = BettingEngine(
        bankroll=500.0, min_edge_pct=1.0,
        data_engine=SportsDataEngine(providers=[_StubProvider([ev], {ev.event_id: books})]))
    ratings = {ev.key: {"home": {"rating": 1650, "games": 40},
                        "away": {"rating": 1480, "games": 40}}}

    result = asyncio.run(engine.run_cycle(
        data_mode=DataMode.LIVE_SHADOW, ratings=ratings, account_health_ok=False))

    assert result["ok"] is True
    assert result["events"] == 1
    assert result["with_odds"] == 1
    assert result["modelled"] == 1
    assert result["opportunities"] >= 1

    # every opportunity must carry mode/source and honest edge numbers
    for o in engine.opportunities:
        assert o.data_mode == DataMode.LIVE_SHADOW.value
        assert o.model_prob > 0
        assert o.price > 1.0
        # an opportunity not marked executable must say why
        if not o.executable:
            assert o.blockers


def test_engine_blocks_live_capital_without_verified_account_health():
    ev = sample_events(1)[0]
    books = [BookOdds(book="softbook", market="h2h",
                      outcomes={ev.home_team: 2.40, ev.away_team: 1.70})]
    engine = BettingEngine(
        bankroll=500.0, min_edge_pct=1.0,
        data_engine=SportsDataEngine(providers=[_StubProvider([ev], {ev.event_id: books})]))
    ratings = {ev.key: {"home": {"rating": 1750, "games": 40},
                        "away": {"rating": 1420, "games": 40}}}

    result = asyncio.run(engine.run_cycle(
        data_mode=DataMode.LIVE, ratings=ratings, account_health_ok=False))
    for o in engine.opportunities:
        assert not o.executable
        assert any("account health" in b for b in o.blockers), o.blockers


def test_engine_blocks_synthetic_data_mode():
    ev = sample_events(1)[0]
    books = [BookOdds(book="softbook", market="h2h",
                      outcomes={ev.home_team: 2.40, ev.away_team: 1.70})]
    engine = BettingEngine(
        bankroll=500.0, min_edge_pct=1.0,
        data_engine=SportsDataEngine(providers=[_StubProvider([ev], {ev.event_id: books})]))
    ratings = {ev.key: {"home": {"rating": 1750, "games": 40},
                        "away": {"rating": 1420, "games": 40}}}
    result = asyncio.run(engine.run_cycle(
        data_mode=DataMode.MOCK, ratings=ratings, account_health_ok=True))
    for o in engine.opportunities:
        assert not o.executable
        assert any("synthetic" in b for b in o.blockers)


def test_engine_report_exposes_gates():
    engine = BettingEngine(bankroll=100.0)
    rep = engine.get_report()
    assert rep["layer"] == "betting / sports exchange"
    assert len(rep["gates"]) >= 6
    assert rep["no_fabrication"]
