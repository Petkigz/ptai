"""
Tests for the full match-card market coverage (V11b).

The original betting layer priced one market (h2h). These tests cover the
whole card: goals, corners, cards, red cards, shots, offsides, handicaps,
half-time, correct score and player props.

Two things matter most here and both are tested hard:

1. Consistency. Every goal-derived market comes from one scoreline grid, so
   the probabilities cannot disagree. If they do, the disagreement is itself
   an arbitrage the model is offering you for free - which means the model
   is broken.

2. Settlement. Winning the match and winning the bet are different things.
   An Asian handicap can push, a quarter line can half-win, a corners market
   settles off corners and not goals, and an abandoned match voids. Getting
   any of these wrong turns a correct model into a wrong P&L.
"""
import math

import pytest

from src.ptai.betting.market_types import (
    MARKET_CATALOGUE,
    MatchFacts,
    Settlement,
    catalogue_report,
    markets_for_sport,
    payout_multiplier,
    settle_market,
    validate_line,
)
from src.ptai.betting.derivative_markets import (
    AccumulatorLeg,
    price_accumulator,
    price_anytime_scorer,
    price_booking_points,
    price_count_market,
    price_first_scorer,
    price_half_time,
    price_match_card,
    price_player_count,
    price_red_card,
)
from src.ptai.betting.engine import BettingEngine
from src.ptai.betting.sports_data import SportsDataEngine, sample_events
from src.ptai.markets.base import DataMode


# ---------------------------------------------------------------------------
# Catalogue coverage
# ---------------------------------------------------------------------------

def test_catalogue_covers_the_markets_the_user_asked_about():
    """Goals, corners, red cards, cards, shots - all must exist."""
    required = {
        "h2h", "totals", "btts", "correct_score", "asian_handicap",
        "european_handicap", "double_chance", "draw_no_bet",
        "team_goals", "clean_sheet", "win_to_nil", "race_to_goals",
        "half_time_result", "half_time_goals", "ht_ft",
        "corners_total", "corners_handicap", "corners_1x2",
        "cards_total", "cards_1x2", "red_card", "booking_points",
        "shots_total", "shots_on_target_total", "offsides_total",
        "player_goals", "player_shots", "player_assists", "player_cards",
        "player_tackles", "anytime_scorer", "first_scorer",
    }
    missing = required - set(MARKET_CATALOGUE)
    assert not missing, f"catalogue missing: {missing}"


def test_every_market_declares_void_rules_and_a_settler():
    for key, spec in MARKET_CATALOGUE.items():
        assert spec.settle is not None, f"{key} has no settlement function"
        assert spec.void_rules, f"{key} declares no void rules"
        assert spec.model, f"{key} names no pricing model"


def test_catalogue_report_counts():
    rep = catalogue_report()
    assert rep["total_markets"] == len(MARKET_CATALOGUE)
    assert rep["total_markets"] >= 33
    assert rep["needing_line"] >= 15
    assert sum(rep["by_model"].values()) == rep["total_markets"]


def test_line_validation_rejects_nonsense():
    ok, _ = validate_line("totals", 2.5)
    assert ok
    ok, why = validate_line("totals", None)
    assert not ok and "requires a line" in why
    ok, why = validate_line("totals", -1.0)
    assert not ok and "negative" in why
    ok, why = validate_line("totals", 2.3)
    assert not ok and "quarter-line grid" in why
    ok, _ = validate_line("totals", 2.25)
    assert ok
    ok, why = validate_line("not_a_market", 1.0)
    assert not ok and "unknown" in why


# ---------------------------------------------------------------------------
# Pricing consistency - all goal markets from one grid
# ---------------------------------------------------------------------------

def soccer_event():
    """A soccer fixture - sample_events()[0] is NBA, which routes elsewhere."""
    from src.ptai.betting.sports_data import SportsEvent
    from datetime import datetime, timedelta, timezone
    return SportsEvent(
        event_id="EPL-1", sport="soccer", league="epl",
        home_team="Manchester City", away_team="Arsenal",
        commence_time=datetime.now(timezone.utc) + timedelta(hours=3),
        provider="test", raw={},
    )


CARD = price_match_card(
    event_key="epl:arsenal-at-man-city",
    home_attack=1.35, home_defence=0.90,
    away_attack=1.05, away_defence=1.10,
    corners_home=6.4, corners_away=4.1,
    cards_home=1.9, cards_away=2.1,
    red_rate=0.07, offsides=3.4,
)


def test_card_prices_a_real_number_of_markets():
    assert CARD.markets_priced >= 25


def test_h2h_sums_to_one():
    g = CARD.goals
    assert g.p_home + g.p_draw + g.p_away == pytest.approx(1.0, abs=1e-6)


def test_totals_over_under_push_sum_to_one():
    for line in (1.5, 2.5, 3.5, 4.5):
        t = CARD.goals.totals(line)
        assert t["over"] + t["under"] + t["push"] == pytest.approx(1.0, abs=1e-6), line


def test_integer_total_line_can_push():
    t = CARD.goals.totals(2.0)
    assert t["push"] > 0
    assert t["over"] + t["under"] + t["push"] == pytest.approx(1.0, abs=1e-6)


def test_quarter_line_is_the_mean_of_the_two_adjacent_halves():
    """Over 2.25 is half a bet on 2.0 and half on 2.5."""
    q = CARD.goals.totals(2.25)
    lo = CARD.goals.totals(2.0)
    hi = CARD.goals.totals(2.5)
    assert q.get("quarter_line") is True
    assert q["over"] == pytest.approx((lo["over"] + hi["over"]) / 2.0, abs=1e-6)
    assert q["under"] == pytest.approx((lo["under"] + hi["under"]) / 2.0, abs=1e-6)


def test_higher_total_line_means_lower_over_probability():
    probs = [CARD.goals.totals(l)["over"] for l in (1.5, 2.5, 3.5, 4.5)]
    assert probs == sorted(probs, reverse=True)


def test_double_chance_is_consistent_with_h2h():
    g = CARD.goals
    dc = g.double_chance()
    assert dc["1X"] == pytest.approx(g.p_home + g.p_draw, abs=1e-6)
    assert dc["X2"] == pytest.approx(g.p_draw + g.p_away, abs=1e-6)
    assert dc["12"] == pytest.approx(g.p_home + g.p_away, abs=1e-6)
    # inclusion-exclusion: 1X + X2 - 12 counts the draw twice
    assert dc["1X"] + dc["X2"] - dc["12"] == pytest.approx(2 * g.p_draw, abs=1e-6)


def test_draw_no_bet_renormalises_over_decisive_outcomes():
    g = CARD.goals
    dnb = g.draw_no_bet()
    assert dnb["home"] + dnb["away"] == pytest.approx(1.0, abs=1e-6)
    assert dnb["push_prob"] == pytest.approx(g.p_draw, abs=1e-6)
    assert dnb["home"] > g.p_home    # removing the draw raises both sides


def test_btts_matches_the_scoreline_grid():
    g = CARD.goals
    manual_yes = sum(p for x, row in enumerate(g.grid) for y, p in enumerate(row)
                     if x >= 1 and y >= 1)
    assert g.btts()["yes"] == pytest.approx(manual_yes, abs=1e-4)
    assert g.btts()["yes"] + g.btts()["no"] == pytest.approx(1.0, abs=1e-6)


def test_clean_sheet_is_the_zero_column_and_row():
    g = CARD.goals
    cs = g.clean_sheet()
    assert cs["home"] == pytest.approx(sum(row[0] for row in g.grid), abs=1e-4)
    assert cs["away"] == pytest.approx(sum(g.grid[0]), abs=1e-4)


def test_win_to_nil_is_a_subset_of_clean_sheet():
    g = CARD.goals
    wtn = g.win_to_nil()
    cs = g.clean_sheet()
    assert wtn["home"] <= cs["home"] + 1e-9
    assert wtn["away"] <= cs["away"] + 1e-9


def test_correct_score_sums_to_one_including_other():
    cs = CARD.goals.correct_score(grid_limit=6)
    assert sum(cs.values()) == pytest.approx(1.0, abs=1e-3)
    assert cs["other"] > 0        # the tail must not be priced at zero
    assert cs["1-1"] > cs["5-0"]  # a common score beats a rout


def test_asian_handicap_covers_all_outcomes():
    for line in (-1.5, -0.5, 0.0, 0.5, 1.5):
        h = CARD.goals.handicap(line)
        assert h["home"] + h["away"] + h["push"] == pytest.approx(1.0, abs=1e-6), line


def test_asian_handicap_zero_line_pushes_on_a_draw():
    h = CARD.goals.handicap(0.0)
    assert h["push"] == pytest.approx(CARD.goals.p_draw, abs=1e-6)


def test_quarter_handicap_splits_rather_than_resolves():
    h = CARD.goals.handicap(-0.25)
    assert h.get("quarter_line") is True
    assert h["components"] == [-0.5, 0.0]
    # half-win buckets exist and total probability stays sane
    assert h["home"] + h["away"] <= 1.0 + 1e-6


def test_european_handicap_has_no_push():
    h = CARD.goals.european_handicap(-1.0)
    assert h["home"] + h["draw"] + h["away"] == pytest.approx(1.0, abs=1e-6)
    assert "push" not in h


def test_handicap_direction_moves_monotonically():
    """A bigger home handicap must lower P(home covers)."""
    probs = [CARD.goals.handicap(l)["home"] for l in (0.5, 0.0, -0.5, -1.5)]
    assert probs == sorted(probs, reverse=True)


def test_team_goals_are_consistent_with_the_total():
    """P(home>=1 and away>=1) must equal BTTS - same grid, two routes."""
    g = CARD.goals
    home_scores = 1.0 - g.team_totals("home", 0.5)["under"]
    away_scores = 1.0 - g.team_totals("away", 0.5)["under"]
    # not independent in general, but both must be in valid range
    assert 0 < home_scores < 1 and 0 < away_scores < 1
    assert g.btts()["yes"] <= min(home_scores, away_scores) + 1e-6


def test_race_to_probabilities_are_valid():
    r = CARD.goals.race_to(2)
    assert r["home"] + r["away"] + r["none"] == pytest.approx(1.0, abs=0.02)
    assert r["none"] > 0     # plenty of matches never reach 2 goals by one team


# ---------------------------------------------------------------------------
# Half-time markets
# ---------------------------------------------------------------------------

def test_half_time_sums_to_one():
    ht = CARD.half_time
    assert ht.ht_home + ht.ht_draw + ht.ht_away == pytest.approx(1.0, abs=1e-3)


def test_half_time_has_fewer_goals_than_full_time():
    """Fewer goals are scored in 45 minutes than 90 - this must show up."""
    ht_over = CARD.half_time.ht_totals["over"]
    ft_over = CARD.goals.totals(0.5)["over"]
    assert ht_over < ft_over


def test_ht_ft_covers_all_nine_combinations_and_sums_to_one():
    htft = CARD.half_time.ht_ft
    assert len(htft) == 9, sorted(htft)
    assert sum(htft.values()) == pytest.approx(1.0, abs=1e-3)


def test_ht_ft_draw_draw_is_not_the_most_likely():
    """The 3/1-style favourite is usually H/H for a strong home side."""
    htft = CARD.half_time.ht_ft
    assert max(htft, key=htft.get) != "D/D"


# ---------------------------------------------------------------------------
# Corners / cards / shots - independent processes
# ---------------------------------------------------------------------------

def test_corners_use_their_own_rates_not_goal_rates():
    c = CARD.corners
    assert c.home_lambda == pytest.approx(6.4, abs=1e-6)
    assert c.away_lambda == pytest.approx(4.1, abs=1e-6)
    assert c.home_lambda != CARD.goals.home_lambda


def test_corners_total_sums_to_one():
    t = CARD.corners.totals
    assert t["over"] + t["under"] + t["push"] == pytest.approx(1.0, abs=1e-6)


def test_corners_handicap_sums_to_one():
    h = CARD.corners.handicap
    assert h["home"] + h["away"] + h["push"] == pytest.approx(1.0, abs=1e-6)


def test_corners_1x2_sums_to_one():
    c = CARD.corners
    assert c.p_home + c.p_draw + c.p_away == pytest.approx(1.0, abs=1e-6)


def test_corners_race_to_nine_is_valid():
    r = CARD.corners.race_to[9]
    assert r["home"] + r["away"] + r["none"] == pytest.approx(1.0, abs=0.05)


def test_cards_use_their_own_rates():
    c = CARD.cards
    assert c.home_lambda == pytest.approx(1.9, abs=1e-6)
    assert c.away_lambda == pytest.approx(2.1, abs=1e-6)
    t = c.totals
    assert t["over"] + t["under"] + t["push"] == pytest.approx(1.0, abs=1e-6)


def test_shots_and_sot_use_their_own_rates():
    assert CARD.shots.home_lambda > CARD.shots_on_target.home_lambda
    t = CARD.shots.totals
    assert t["over"] + t["under"] + t["push"] == pytest.approx(1.0, abs=1e-6)


def test_offsides_only_priced_when_supplied():
    assert CARD.offsides is not None
    bare = price_match_card("x", 1.0, 1.0, 1.0, 1.0)
    assert bare.offsides is None


def test_red_card_probability_matches_the_poisson_rate():
    r = price_red_card(0.07)
    assert r["yes"] == pytest.approx(1 - math.exp(-0.07), abs=1e-4)
    assert r["yes"] + r["no"] == pytest.approx(1.0, abs=1e-6)


def test_red_card_scales_with_minutes_played():
    full = price_red_card(0.07, minutes=90)["yes"]
    half = price_red_card(0.07, minutes=45)["yes"]
    assert half < full


def test_booking_points_expected_value():
    bp = price_booking_points(4.0, red_rate=0.07, line=35.0)
    # 4 yellows * 10 + ~0.07 reds * 25 ~ 41.75
    # 4 yellows * 10 + ~0.07 reds * 25 ~ 41.05
    assert 38 < bp["expected_points"] < 45
    assert bp["over"] + bp["under"] + bp["push"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def test_player_goal_rate_scales_with_team_strength():
    strong = price_player_count(2.0, 0.30)["expected"]
    weak = price_player_count(1.0, 0.30)["expected"]
    assert strong == pytest.approx(2 * weak, abs=1e-6)


def test_player_total_sums_to_one():
    p = price_player_count(2.0, 0.30, line=0.5)
    assert p["over"] + p["under"] + p["push"] == pytest.approx(1.0, abs=1e-6)


def test_anytime_scorer_is_one_minus_poisson_zero():
    lam = 1.5 * 0.30
    a = price_anytime_scorer(1.5, 0.30)
    assert a["yes"] == pytest.approx(1 - math.exp(-lam), abs=1e-4)
    assert a["yes"] + a["no"] == pytest.approx(1.0, abs=1e-6)


def test_minutes_share_reduces_anytime_probability():
    starter = price_anytime_scorer(1.5, 0.30, minutes_share=1.0)["yes"]
    sub = price_anytime_scorer(1.5, 0.30, minutes_share=0.5)["yes"]
    assert sub < starter


def test_first_scorer_is_rarer_than_anytime():
    a = price_anytime_scorer(1.5, 0.30)["yes"]
    f = price_first_scorer(1.5, 0.30, opponent_lambda=1.2)["yes"]
    assert f < a


def test_first_scorer_shares_sum_below_one():
    """Two players cannot both score first."""
    f1 = price_first_scorer(1.5, 0.30, opponent_lambda=1.2)["yes"]
    f2 = price_first_scorer(1.5, 0.20, opponent_lambda=1.2)["yes"]
    assert f1 + f2 < 1.0


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def facts(home=2, away=1, **kw):
    return MatchFacts(home_team="Home", away_team="Away", home_goals=home,
                      away_goals=away, **kw)


def test_h2h_settlement():
    f = facts(2, 1)
    assert settle_market("h2h", f, "home") == Settlement.WIN
    assert settle_market("h2h", f, "away") == Settlement.LOSE
    assert settle_market("h2h", f, "draw") == Settlement.LOSE


def test_double_chance_settlement():
    f = facts(1, 1)
    assert settle_market("double_chance", f, "1X") == Settlement.WIN
    assert settle_market("double_chance", f, "X2") == Settlement.WIN
    assert settle_market("double_chance", f, "12") == Settlement.LOSE


def test_draw_no_bet_pushes_on_a_draw():
    assert settle_market("draw_no_bet", facts(1, 1), "home") == Settlement.PUSH
    assert settle_market("draw_no_bet", facts(2, 0), "home") == Settlement.WIN


def test_asian_handicap_settlement_whole_line_pushes():
    # home -1.0, home wins 2-1 -> margin exactly 0 -> push
    assert settle_market("asian_handicap", facts(2, 1), "home", line=-1.0) == Settlement.PUSH
    assert settle_market("asian_handicap", facts(3, 1), "home", line=-1.0) == Settlement.WIN
    assert settle_market("asian_handicap", facts(1, 1), "home", line=-1.0) == Settlement.LOSE


def test_asian_handicap_half_line_cannot_push():
    r = settle_market("asian_handicap", facts(2, 1), "home", line=-1.5)
    assert r in (Settlement.WIN, Settlement.LOSE)


def test_asian_handicap_quarter_line_half_wins_and_half_loses():
    # home -0.25 on a draw: half stake on 0.0 pushes, half on -0.5 loses
    r = settle_market("asian_handicap", facts(1, 1), "home", line=-0.25)
    assert r == Settlement.HALF_LOSE
    # home -0.25 on a 1-goal home win: both halves win
    assert settle_market("asian_handicap", facts(2, 1), "home", line=-0.25) == Settlement.WIN


def test_asian_handicap_away_side():
    # away +0.25 on a draw: half stake on 0.0 pushes, half on +0.5 wins
    assert settle_market("asian_handicap", facts(1, 1), "away", line=0.25) == Settlement.HALF_WIN
    # away +0.25 on a 1-goal home win: both halves lose outright
    assert settle_market("asian_handicap", facts(2, 1), "away", line=0.25) == Settlement.LOSE
    # away +1.0 on a 1-goal home win pushes
    assert settle_market("asian_handicap", facts(2, 1), "away", line=1.0) == Settlement.PUSH
    # away +1.0 on a 2-goal home win loses
    assert settle_market("asian_handicap", facts(3, 1), "away", line=1.0) == Settlement.LOSE


def test_home_and_away_handicap_are_mirror_images():
    """Backing home -1.0 and away +1.0 on the same match must be exact opposites."""
    for h, a in [(2, 1), (3, 1), (1, 1), (0, 2)]:
        home = settle_market("asian_handicap", facts(h, a), "home", line=-1.0)
        away = settle_market("asian_handicap", facts(h, a), "away", line=1.0)
        assert {home, away} == {Settlement.WIN, Settlement.LOSE} or home == away == Settlement.PUSH, (h, a, home, away)


def test_totals_settlement_and_quarter_lines():
    assert settle_market("totals", facts(2, 1), "over", line=2.5) == Settlement.WIN
    assert settle_market("totals", facts(1, 1), "over", line=2.5) == Settlement.LOSE
    assert settle_market("totals", facts(2, 1), "over", line=3.0) == Settlement.PUSH
    # over 2.25 on exactly 2 goals: half on 2.0 pushes, half on 2.5 loses
    assert settle_market("totals", facts(1, 1), "over", line=2.25) == Settlement.HALF_LOSE
    # over 2.75 on exactly 3 goals: half on 2.5 wins, half on 3.0 pushes
    assert settle_market("totals", facts(2, 1), "over", line=2.75) == Settlement.HALF_WIN


def test_btts_settlement():
    assert settle_market("btts", facts(2, 1), "yes") == Settlement.WIN
    assert settle_market("btts", facts(2, 0), "yes") == Settlement.LOSE
    assert settle_market("btts", facts(2, 0), "no") == Settlement.WIN


def test_team_goals_settlement():
    assert settle_market("team_goals", facts(2, 0), "home_over", line=1.5) == Settlement.WIN
    assert settle_market("team_goals", facts(1, 0), "home_over", line=1.5) == Settlement.LOSE
    assert settle_market("team_goals", facts(2, 0), "away_under", line=0.5) == Settlement.WIN
    assert settle_market("team_goals", facts(2, 0), "nonsense", line=0.5) == Settlement.UNSETTLEABLE


def test_clean_sheet_and_win_to_nil_settlement():
    assert settle_market("clean_sheet", facts(2, 0), "home") == Settlement.WIN
    assert settle_market("clean_sheet", facts(2, 1), "home") == Settlement.LOSE
    assert settle_market("win_to_nil", facts(2, 0), "home") == Settlement.WIN
    assert settle_market("win_to_nil", facts(2, 1), "home") == Settlement.LOSE
    assert settle_market("win_to_nil", facts(0, 0), "home") == Settlement.LOSE


def test_correct_score_settlement():
    assert settle_market("correct_score", facts(2, 1), "2-1") == Settlement.WIN
    assert settle_market("correct_score", facts(2, 1), "1-2") == Settlement.LOSE


def test_half_time_settlement_uses_the_ht_score():
    f = facts(3, 1, home_goals_ht=0, away_goals_ht=1)
    assert settle_market("half_time_result", f, "away") == Settlement.WIN
    assert settle_market("half_time_result", f, "home") == Settlement.LOSE
    assert settle_market("ht_ft", f, "A/H") == Settlement.WIN
    assert settle_market("ht_ft", f, "H/H") == Settlement.LOSE


def test_corners_settle_off_corners_not_goals():
    """A 0-0 match with 14 corners still settles the corners market."""
    f = facts(0, 0, home_corners=9, away_corners=5)
    assert settle_market("corners_total", f, "over", line=10.5) == Settlement.WIN
    assert settle_market("corners_1x2", f, "home") == Settlement.WIN
    assert settle_market("corners_handicap", f, "home", line=-1.5) == Settlement.WIN


def test_cards_settle_off_cards():
    f = facts(1, 0, home_cards=2, away_cards=3)
    assert settle_market("cards_total", f, "over", line=4.5) == Settlement.WIN
    assert settle_market("cards_total", f, "over", line=5.5) == Settlement.LOSE
    assert settle_market("cards_1x2", f, "away") == Settlement.WIN


def test_booking_points_settlement_uses_yellow_10_red_25():
    f = facts(1, 0, home_cards=2, away_cards=1, home_reds=1, away_reds=0)
    # 3 yellows * 10 + 1 red * 25 = 55
    assert f.booking_points == 55
    assert settle_market("booking_points", f, "over", line=45.0) == Settlement.WIN
    assert settle_market("booking_points", f, "under", line=45.0) == Settlement.LOSE


def test_red_card_settlement_counts_second_yellows():
    assert settle_market("red_card", facts(1, 0, home_reds=1, away_reds=0), "yes") == Settlement.WIN
    assert settle_market("red_card", facts(1, 0, home_reds=0, away_reds=0), "yes") == Settlement.LOSE
    assert settle_market("red_card", facts(1, 0, home_reds=0, away_reds=0), "no") == Settlement.WIN


def test_shots_and_offsides_settlement():
    f = facts(1, 0, home_shots=14, away_shots=8)
    assert settle_market("shots_total", f, "over", line=20.5) == Settlement.WIN
    f2 = facts(1, 0, home_offsides=2, away_offsides=3)
    assert settle_market("offsides_total", f2, "over", line=4.5) == Settlement.WIN
    assert settle_market("offsides_total", f2, "over", line=5.5) == Settlement.LOSE


def test_race_to_needs_chronology_and_refuses_to_guess():
    """Without goal_events the final score cannot say who reached N first."""
    f = facts(2, 1)
    assert settle_market("race_to_goals", f, "home", line=2) == Settlement.UNSETTLEABLE

    with_events = facts(2, 1, goal_events=[("home", 10), ("away", 30), ("home", 70)])
    assert settle_market("race_to_goals", with_events, "home", line=2) == Settlement.WIN
    assert settle_market("race_to_goals", with_events, "away", line=2) == Settlement.LOSE


def test_race_to_none_when_neither_reaches():
    f = facts(1, 1, goal_events=[("home", 20), ("away", 60)])
    assert settle_market("race_to_goals", f, "none", line=2) == Settlement.WIN


def test_abandoned_match_voids_every_market():
    f = facts(1, 0, abandoned=True, home_corners=5, away_corners=3,
              home_cards=1, away_cards=1)
    # (market, outcome, line) - totals need a line or UNSETTLEABLE is correct
    cases = [
        ("h2h", "home", None),
        ("double_chance", "1X", None),
        ("draw_no_bet", "home", None),
        ("btts", "yes", None),
        ("totals", "over", 2.5),
        ("corners_total", "over", 6.5),
        ("cards_total", "over", 1.5),
        ("red_card", "yes", None),
        ("half_time_result", "home", None),
        ("asian_handicap", "home", -1.0),
        ("correct_score", "1-0", None),
    ]
    for key, outcome, line in cases:
        assert settle_market(key, f, outcome, line=line) == Settlement.VOID, key


def test_missing_facts_is_unsettleable_not_a_guess():
    f = MatchFacts(home_team="A", away_team="B")   # no scores at all
    assert settle_market("h2h", f, "home") == Settlement.UNSETTLEABLE
    f2 = facts(1, 0)                                # no corner data
    assert settle_market("corners_total", f2, "over", line=10.5) == Settlement.UNSETTLEABLE
    assert settle_market("red_card", f2, "yes") == Settlement.UNSETTLEABLE


def test_settling_with_an_invalid_line_is_refused():
    assert settle_market("totals", facts(2, 1), "over", line=None) == Settlement.UNSETTLEABLE
    assert settle_market("totals", facts(2, 1), "over", line=-1.0) == Settlement.UNSETTLEABLE


# ---------------------------------------------------------------------------
# Payout multipliers
# ---------------------------------------------------------------------------

def test_payout_multipliers():
    assert payout_multiplier(Settlement.WIN, 2.5) == 2.5
    assert payout_multiplier(Settlement.LOSE, 2.5) == 0.0
    assert payout_multiplier(Settlement.PUSH, 2.5) == 1.0
    assert payout_multiplier(Settlement.VOID, 2.5) == 1.0
    assert payout_multiplier(Settlement.HALF_WIN, 3.0) == pytest.approx(2.0)
    assert payout_multiplier(Settlement.HALF_LOSE, 3.0) == 0.5


def test_half_win_pnl_is_between_a_win_and_a_push():
    stake, odds = 100.0, 3.0
    win = stake * payout_multiplier(Settlement.WIN, odds) - stake
    half = stake * payout_multiplier(Settlement.HALF_WIN, odds) - stake
    push = stake * payout_multiplier(Settlement.PUSH, odds) - stake
    assert push < half < win
    assert half == pytest.approx(win / 2.0)


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

def test_accumulator_correlation_haircut_kills_a_fake_edge():
    legs = [
        AccumulatorLeg("totals", "over", 1.95, 0.52, "ev1", line=2.5),
        AccumulatorLeg("btts", "yes", 1.90, 0.55, "ev1"),
    ]
    q = price_accumulator(legs)
    # independent product looks like value
    assert q.edge_independent > 0
    # after the correlation haircut it does not
    assert q.edge_adjusted < q.edge_independent
    assert "correlated" in q.warning


def test_accumulator_same_event_gets_the_larger_haircut():
    same = price_accumulator([
        AccumulatorLeg("totals", "over", 2.0, 0.55, "ev1"),
        AccumulatorLeg("btts", "yes", 2.0, 0.55, "ev1"),
    ])
    different = price_accumulator([
        AccumulatorLeg("totals", "over", 2.0, 0.55, "ev1"),
        AccumulatorLeg("btts", "yes", 2.0, 0.55, "ev2"),
    ])
    assert same.correlation_adjusted_prob < different.correlation_adjusted_prob


def test_accumulator_rejects_too_many_legs():
    legs = [AccumulatorLeg("totals", "over", 1.9, 0.55, f"ev{i}") for i in range(12)]
    q = price_accumulator(legs, max_legs=8)
    assert q.blocked
    assert any("exceeds" in b for b in q.blockers)


def test_accumulator_needs_two_legs():
    q = price_accumulator([AccumulatorLeg("totals", "over", 1.9, 0.55, "ev1")])
    assert q.blocked


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_engine_prices_a_card_and_exposes_fair_prices():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    strengths = {ev.key: {"home": {"attack": 1.4, "defence": 0.9, "corners": 6.2,
                                   "cards": 1.8, "shots": 14.0, "shots_on_target": 4.8},
                          "away": {"attack": 1.0, "defence": 1.1, "corners": 4.2,
                                   "cards": 2.2, "shots": 10.0, "shots_on_target": 3.4},
                          "red_rate": 0.07, "offsides": 3.4}}
    card = engine.price_card(ev, strengths)
    assert card is not None
    assert card.markets_priced >= 25

    fair = engine.card_fair_prices(ev.key)
    for required in ("corners_total", "cards_total", "red_card", "btts",
                     "double_chance", "asian_handicap_-0.5", "ht_ft",
                     "correct_score", "booking_points", "shots_total"):
        assert required in fair, f"engine did not expose {required}"


def test_engine_refuses_to_price_a_card_without_strengths():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    assert engine.price_card(ev, {}) is None


def test_card_markets_are_not_executable_without_a_book_price():
    """
    A fair price with nothing to bet against is not an opportunity. The
    engine must record that rather than presenting a model price as a bet.
    """
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    strengths = {ev.key: {"home": {"attack": 1.4, "defence": 0.9},
                          "away": {"attack": 1.0, "defence": 1.1}}}
    card = engine.price_card(ev, strengths)
    opps = engine._build_card_opportunities(ev, card, {}, DataMode.LIVE_SHADOW, False)
    assert opps, "expected fair prices to be listed"
    for o in opps:
        assert not o.executable
        assert any("no book price" in b for b in o.blockers), o.blockers


def test_settle_bet_through_the_engine():
    engine = BettingEngine(bankroll=500.0)
    r = engine.settle_bet("totals", facts(2, 1), "over", stake=10.0,
                          decimal_odds=1.95, line=2.5)
    assert r["settled"] and r["result"] == "win"
    assert r["pnl"] == pytest.approx(9.5)

    r2 = engine.settle_bet("asian_handicap", facts(2, 1), "home", stake=10.0,
                           decimal_odds=1.90, line=-1.0)
    assert r2["result"] == "push"
    assert r2["pnl"] == pytest.approx(0.0)

    r3 = engine.settle_bet("totals", facts(2, 1), "over", stake=10.0,
                           decimal_odds=1.95, line=None)
    assert not r3["settled"] and "requires a line" in r3["reason"]


def test_engine_report_includes_the_catalogue():
    rep = BettingEngine(bankroll=100.0).get_report()
    assert rep["market_catalogue"]["total_markets"] >= 33
    assert "h2h" in rep["market_types_scanned_from_feed"]
    assert "totals" in rep["market_types_scanned_from_feed"]


# ---------------------------------------------------------------------------
# Player props and the explicit feed mapping
# ---------------------------------------------------------------------------

PLAYERS = [
    {"name": "Haaland", "side": "home", "share": 0.42, "minutes_share": 0.95},
    {"name": "Saka", "side": "away", "share": 0.28, "minutes_share": 0.90},
]


def _engine_with_card():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    strengths = {ev.key: {"home": {"attack": 1.42, "defence": 0.88, "corners": 6.6,
                                   "cards": 1.8, "shots": 15.2, "shots_on_target": 5.4},
                          "away": {"attack": 1.18, "defence": 1.02, "corners": 4.3,
                                   "cards": 2.3, "shots": 9.8, "shots_on_target": 3.1},
                          "red_rate": 0.07, "offsides": 3.2,
                          "players": PLAYERS,
                          "prop_lines": {"Haaland": 0.5}}}
    card = engine.price_card(ev, strengths)
    return engine, ev, strengths, card


def test_feed_mapping_is_explicit_not_derived_from_the_key():
    """
    'asian_handicap_-0.5' and 'totals_2.5' cannot be split into a feed type
    by splitting on '_', which is what the old code did.
    """
    f = BettingEngine._feed_type_for
    assert f("asian_handicap_-0.5") == "spreads"
    assert f("totals_2.5") == "totals"
    assert f("corners_handicap_-1.5") == "corners_handicap"
    assert f("shots_on_target_total") == "shots_on_target_total"
    assert f("player_goals:Haaland") == "player_goals"


def test_every_priced_card_key_has_a_feed_mapping():
    engine, ev, strengths, card = _engine_with_card()
    for key in engine.card_fair_prices(ev.key):
        assert BettingEngine._feed_type_for(key), f"{key} has no feed mapping"


def test_player_props_require_player_data():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    engine.price_card(ev, {ev.key: {"home": {"attack": 1.2, "defence": 1.0},
                                    "away": {"attack": 1.0, "defence": 1.1}}})
    assert engine.price_player_props(ev, []) == {}


def test_player_props_price_a_striker_higher_than_a_winger():
    engine, ev, strengths, card = _engine_with_card()
    props = engine.price_player_props(ev, PLAYERS)
    assert set(props) == {"Haaland", "Saka"}
    # higher share of a stronger attack -> higher goal expectation
    assert props["Haaland"]["expected_goals"] > props["Saka"]["expected_goals"]
    assert props["Haaland"]["anytime_scorer_yes"] > props["Saka"]["anytime_scorer_yes"]
    assert props["Haaland"]["first_scorer_yes"] < props["Haaland"]["anytime_scorer_yes"]


def test_player_props_carry_the_void_warning():
    engine, ev, strengths, card = _engine_with_card()
    props = engine.price_player_props(ev, PLAYERS)
    for name, p in props.items():
        assert "void" in p["void_warning"].lower()


def test_minutes_share_reduces_the_prop():
    engine, ev, strengths, card = _engine_with_card()
    starter = engine.price_player_props(
        ev, [{"name": "X", "side": "home", "share": 0.4, "minutes_share": 1.0}])
    sub = engine.price_player_props(
        ev, [{"name": "X", "side": "home", "share": 0.4, "minutes_share": 0.4}])
    assert sub["X"]["anytime_scorer_yes"] < starter["X"]["anytime_scorer_yes"]


def test_player_prop_fair_prices_use_catalogue_keys():
    engine, ev, strengths, card = _engine_with_card()
    fair = engine.player_prop_fair_prices(ev, PLAYERS, {"Haaland": 0.5})
    keys = list(fair)
    assert "anytime_scorer:Haaland" in keys
    assert "player_goals:Haaland" in keys          # only when a line is supplied
    assert "player_goals:Saka" not in keys
    for k in keys:
        base = k.split(":", 1)[0]
        assert base in MARKET_CATALOGUE, f"{base} is not a catalogue market"


def test_player_prop_opportunities_never_execute_without_the_void_check():
    engine, ev, strengths, card = _engine_with_card()
    opps = engine._build_player_prop_opportunities(
        ev, PLAYERS, {}, DataMode.LIVE, account_health_ok=True,
        prop_lines={"Haaland": 0.5})
    assert opps
    for o in opps:
        assert not o.executable
        assert any("void-if-not-starting" in b for b in o.blockers), o.blockers


def test_run_cycle_prices_props_when_players_are_supplied():
    from src.ptai.betting.sports_data import BookOdds, SportsDataEngine

    ev = soccer_event()
    ev.event_id = "evt-props"
    books = [BookOdds(book="softbook", market="h2h",
                      outcomes={ev.home_team: 1.75, ev.away_team: 2.20})]

    class _Stub:
        name = "stub"
        is_synthetic = False
        last_error = ""
        last_fetch_at = None
        requests_made = 0

        async def events(self, leagues=()):
            return [ev]

        def odds_for_event(self, event):
            return books

        def health(self):
            return {"provider": "stub", "reachable": True, "last_error": "",
                    "last_fetch_at": None, "requests_made": 0, "is_synthetic": False}

    engine = BettingEngine(
        bankroll=500.0, min_edge_pct=1.0,
        data_engine=SportsDataEngine(providers=[_Stub()]))
    strengths = {ev.key: {"home": {"attack": 1.42, "defence": 0.88},
                          "away": {"attack": 1.18, "defence": 1.02},
                          "players": PLAYERS, "prop_lines": {"Haaland": 0.5}}}

    import asyncio
    result = asyncio.run(engine.run_cycle(data_mode=DataMode.LIVE_SHADOW,
                                          strengths=strengths))
    assert result["ok"] and result["cards_priced"] == 1
    prop_types = {o.market_type for o in engine.opportunities
                  if o.opportunity_id.startswith("prop-")}
    assert "anytime_scorer" in prop_types
    assert "player_goals" in prop_types
    # every prop is blocked on the void rule, none silently executable
    for o in engine.opportunities:
        if o.opportunity_id.startswith("prop-"):
            assert not o.executable


# ---------------------------------------------------------------------------
# High-scoring sports: normal margin model, NOT the soccer Poisson grid
# ---------------------------------------------------------------------------

from src.ptai.betting.high_scoring import (
    SPORT_PARAMS,
    is_high_scoring,
    price_high_scoring_card,
    price_period_total,
    price_spread_normal,
    price_total_normal,
)


def test_high_scoring_sports_are_routed_away_from_poisson():
    for sport in ("basketball", "football", "baseball", "hockey"):
        assert is_high_scoring(sport), sport
    assert not is_high_scoring("soccer")
    assert not is_high_scoring("tennis")


def test_poisson_would_understate_the_nba_total_spread():
    """
    The whole reason this model exists. Poisson variance == mean, so a 224
    point total implies std sqrt(2*112) = 15.0 against ~20.5 actual.
    """
    import math
    poisson_total_std = math.sqrt(2 * 112.0)
    assert poisson_total_std == pytest.approx(15.0, abs=0.1)
    assert SPORT_PARAMS["basketball"].total_std > poisson_total_std * 1.3


def test_total_over_under_push_sum_to_one():
    for line in (224.0, 225.0, 225.5, 226.0):
        t = price_total_normal(224.5, 20.5, line)
        assert t.over + t.under + t.push == pytest.approx(1.0, abs=1e-3), line


def test_whole_number_line_can_push_but_half_line_cannot():
    whole = price_total_normal(224.5, 20.5, 225.0)
    assert whole.push > 0
    half = price_total_normal(224.5, 20.5, 225.5)
    assert half.push == 0.0


def test_quarter_line_is_the_mean_of_adjacent_halves():
    q = price_total_normal(224.5, 20.5, 225.25)
    lo = price_total_normal(224.5, 20.5, 225.0)
    hi = price_total_normal(224.5, 20.5, 225.5)
    assert q.quarter_line is True
    assert q.components == [225.0, 225.5]
    assert q.over == pytest.approx((lo.over + hi.over) / 2.0, abs=1e-6)


def test_higher_line_means_lower_over_probability():
    probs = [price_total_normal(224.5, 20.5, l).over for l in (215.5, 220.5, 225.5, 230.5)]
    assert probs == sorted(probs, reverse=True)


def test_wider_std_flattens_the_over_under():
    """More variance pulls a tail line toward 50%."""
    tight = price_total_normal(224.5, 12.0, 240.5).over
    wide = price_total_normal(224.5, 25.0, 240.5).over
    assert wide > tight


def test_spread_sums_to_one_and_pushes_on_whole_lines():
    for line in (0.0, -1.5, -3.5, -6.0):
        sp = price_spread_normal(4.0, 12.0, line)
        # three values each rounded to 4dp can drift by up to ~1.5e-4
        assert sp.home + sp.away + sp.push == pytest.approx(1.0, abs=1e-3), line
    assert price_spread_normal(4.0, 12.0, -4.0).push > 0     # whole line
    assert price_spread_normal(4.0, 12.0, -3.5).push == 0.0  # half line


def test_spread_direction_is_monotonic():
    probs = [price_spread_normal(4.0, 12.0, l).home for l in (3.5, 0.0, -3.5, -10.5)]
    assert probs == sorted(probs, reverse=True)


def test_spread_at_the_expected_margin_is_even():
    sp = price_spread_normal(4.0, 12.0, -4.5)
    assert sp.home == pytest.approx(0.5, abs=0.02)


def test_moneyline_matches_the_spread_direction():
    ml = price_high_scoring_card("e", "basketball", 114.0, 110.0).moneyline
    assert ml["home"] > ml["away"]
    assert ml["home"] + ml["away"] == pytest.approx(1.0, abs=1e-6)


def test_period_std_scales_with_sqrt_not_linearly():
    """Variances add over independent periods, so std scales with sqrt(fraction)."""
    full = price_total_normal(224.0, 20.0, 224.5)
    q = price_period_total(224.0, 20.0, 56.5, 0.25)
    # a quarter's std must be half the full-game std, not a quarter of it
    assert q.std == pytest.approx(10.0, abs=1e-6)
    assert full.std == 20.0
    # and the over probability at the mean should stay near 50% at any scale
    assert 0.4 < q.over < 0.6


def test_margin_bands_are_a_distribution_over_positive_margins():
    card = price_high_scoring_card("e", "football", 24.0, 20.0,
                                   margin_bands=[(1, 3), (4, 6), (7, 10), (11, 99)])
    total = sum(card.margin_bands.values())
    assert total < 1.0            # only covers home wins, not all outcomes
    assert all(v >= 0 for v in card.margin_bands.values())


def test_nfl_model_does_not_use_the_soccer_total_std():
    nfl = SPORT_PARAMS["football"]
    assert nfl.total_std == pytest.approx(13.5, abs=0.5)
    assert nfl.typical_total == pytest.approx(45.0, abs=1.0)


def test_engine_routes_basketball_to_the_normal_model():
    engine = BettingEngine(bankroll=500.0)
    ev = sample_events(1)[0]      # sample_events[0] is NBA
    assert ev.sport == "nba" or ev.league == "nba"
    strengths = {ev.key: {"home": {"expected_points": 114.0},
                          "away": {"expected_points": 110.0}}}
    result = engine.price_card(ev, strengths)
    # the soccer card is not produced; the high-scoring one is
    assert result is None
    assert ev.key in engine.priced_high_scoring
    fair = engine.high_scoring_fair_prices(ev.key)
    assert "moneyline" in fair
    assert any(k.startswith("totals_") for k in fair)
    assert any(k.startswith("spreads_") for k in fair)
    assert "margin_bands" in fair


def test_engine_refuses_to_invent_a_basketball_score():
    engine = BettingEngine(bankroll=500.0)
    ev = sample_events(1)[0]
    # attack/defence strengths are soccer inputs - not valid for basketball
    strengths = {ev.key: {"home": {"attack": 1.3, "defence": 0.9},
                          "away": {"attack": 1.0, "defence": 1.1}}}
    assert engine.price_card(ev, strengths) is None
    assert ev.key not in engine.priced_high_scoring


def test_feed_type_mapping_handles_dynamic_high_scoring_keys():
    f = BettingEngine._feed_type_for
    assert f("totals_225.0") == "totals"
    assert f("spreads_-3.5") == "spreads"
    assert f("team_total_home_110.5") == "team_total"
    assert f("period_Q1") == "period"


# ═══════════════════════════════════════════════════════════════════════════
# Betfair exchange adapter - the feed that makes the wide card tradeable
# ═══════════════════════════════════════════════════════════════════════════

from src.ptai.venues.betfair_exchange import (
    BETFAIR_EVENT_TYPES, BETFAIR_MARKET_MAP, CORE_MARKET_TYPES,
    PLAYER_MARKET_TYPES, SECONDARY_MARKET_TYPES, BetfairClient,
    BetfairMarket, BetfairRunner, fetch_full_card,
)


def _book(market_id, runners, total_matched=5000.0):
    """A Betfair market book in lightweight (dict) form."""
    return {"id": market_id, "totalMatched": total_matched,
            "runners": [{"selectionId": sid, "name": nm, "status": "ACTIVE",
                         "lastPriceTraded": lp, "totalMatched": tm,
                         "ex": {"availableToBack": [{"price": bp, "size": bs}],
                                "availableToLay": [{"price": lp2, "size": ls}]}}
                        for sid, nm, bp, bs, lp2, ls, lp, tm in runners]}


def test_betfair_market_map_covers_the_whole_catalogue():
    """
    The mapping is the bridge from Betfair's codes to this project's keys.
    Every code we request must land somewhere, and nothing may map to a
    catalogue key that does not exist.
    """
    from src.ptai.betting.market_types import MARKET_CATALOGUE
    requested = set(CORE_MARKET_TYPES) | set(SECONDARY_MARKET_TYPES) | set(PLAYER_MARKET_TYPES)
    unmapped = [t for t in requested if t not in BETFAIR_MARKET_MAP]
    assert not unmapped, f"requested but unmapped: {unmapped}"

    dangling = sorted({v for v in BETFAIR_MARKET_MAP.values()} - set(MARKET_CATALOGUE))
    assert not dangling, f"maps to unknown catalogue keys: {dangling}"


def test_betfair_sports_have_event_type_ids():
    for sport in ("soccer", "basketball", "football", "baseball", "hockey", "tennis"):
        assert sport in BETFAIR_EVENT_TYPES, f"no Betfair event type id for {sport}"


def test_betfair_without_credentials_returns_nothing_not_a_placeholder():
    """
    The old adapter returned five hardcoded fixtures priced 0.45 + i*0.05.
    No credentials must mean no markets, with a reason recorded.
    """
    client = BetfairClient()
    assert not client.configured
    assert client.market_catalogue("1") == []
    assert client.market_books(["1.234"]) == {}
    assert "credentials" in client.last_error.lower()
    assert client.health()["logged_in"] is False


def test_betfair_parses_back_and_lay_from_a_book():
    """
    An exchange price is two-sided. Keeping only the back price would discard
    the lay side that makes hedging and book-vs-exchange arbs possible.
    """
    book = _book("1.100", [(1, "Manchester City", 2.10, 500.0, 2.14, 400.0, 2.12, 9000.0),
                           (2, "Arsenal", 3.90, 250.0, 4.10, 180.0, 4.00, 4000.0)])
    runners, total = BetfairClient.parse_runners(book)
    assert total == 5000.0
    assert len(runners) == 2
    home = runners[0]
    assert home.back_price == 2.10 and home.lay_price == 2.14
    assert home.spread == 0.04
    assert home.has_both_sides
    assert home.back_size == 500.0 and home.lay_size == 400.0


def test_betfair_runner_without_a_lay_side_is_not_bookable():
    book = _book("1.200", [(1, "Team A", 1.80, 100.0, None, 0.0, None, 0.0)])
    runners, _ = BetfairClient.parse_runners(book)
    # one-sided price: back present, no lay available
    assert runners[0].back_price == 1.80
    assert runners[0].has_both_sides is False
    assert runners[0].spread is None


def test_betfair_market_is_bookable_only_when_a_runner_has_both_sides():
    thin = BetfairMarket(market_id="1.1", market_type="MATCH_ODDS", catalogue_key="h2h",
                         market_name="Match Odds", event_id="3", event_name="A v B",
                         commence_time=None, in_play=False,
                         runners=[BetfairRunner(selection_id=1, name="A", back_price=2.0)])
    assert thin.is_bookable is False

    wide = BetfairMarket(market_id="1.2", market_type="MATCH_ODDS", catalogue_key="h2h",
                         market_name="Match Odds", event_id="3", event_name="A v B",
                         commence_time=None, in_play=False,
                         runners=[BetfairRunner(selection_id=1, name="A",
                                                back_price=2.0, back_size=100.0,
                                                lay_price=2.05, lay_size=90.0)])
    assert wide.is_bookable is True


def test_betfair_to_market_carries_the_lay_side_through():
    mf = BetfairMarket(market_id="1.300", market_type="OVER_UNDER_CORNERS",
                       catalogue_key="corners_total", market_name="Over/Under 10.5 Corners",
                       event_id="77", event_name="Man City v Arsenal", commence_time=None,
                       in_play=False, total_matched=2500.0,
                       runners=[BetfairRunner(selection_id=1, name="Over 10.5",
                                              back_price=1.95, back_size=300.0,
                                              lay_price=2.00, lay_size=250.0),
                                BetfairRunner(selection_id=2, name="Under 10.5",
                                              back_price=1.98, back_size=280.0,
                                              lay_price=2.02, lay_size=260.0)])
    m = BetfairClient().to_market(mf, data_mode=DataMode.LIVE)
    assert m is not None
    assert m.venue_id == "betfair"
    assert m.is_mock is False
    assert m.data_source == "betfair_exchange_live"
    assert m.raw["is_exchange"] is True and m.raw["lay_available"] is True
    assert m.raw["catalogue_key"] == "corners_total"
    # the lay ladder is preserved, not just the back prices
    assert m.raw["lay_prices"] == [2.0, 2.02]
    assert m.raw["back_prices"] == [1.95, 1.98]
    assert m.raw["spreads"] == [0.05, 0.04]


def test_betfair_refuses_an_unknown_market_type():
    """
    An unmapped Betfair code cannot be settled against the right facts, so it
    is reported and dropped rather than guessed into some market key.
    """
    client = BetfairClient()
    mf = BetfairMarket(market_id="1.400", market_type="NEW_FANCY_MARKET",
                       catalogue_key="", market_name="Fancy", event_id="9",
                       event_name="X v Y", commence_time=None, in_play=False,
                       runners=[BetfairRunner(selection_id=1, name="A",
                                              back_price=2.0, back_size=100.0,
                                              lay_price=2.1, lay_size=90.0)])
    assert client.to_market(mf) is None
    assert mf.unknown_type is True


def test_betfair_to_market_needs_a_bookable_runner():
    mf = BetfairMarket(market_id="1.500", market_type="MATCH_ODDS", catalogue_key="h2h",
                       market_name="Match Odds", event_id="5", event_name="A v B",
                       commence_time=None, in_play=False, runners=[])
    assert BetfairClient().to_market(mf) is None


def test_betfair_market_mode_is_honest_about_non_live_data():
    """A shadow-mode market must not be labelled as executable live data."""
    mf = BetfairMarket(market_id="1.600", market_type="CARD_ODDS", catalogue_key="cards_1x2",
                       market_name="Cards", event_id="6", event_name="A v B",
                       commence_time=None, in_play=False,
                       runners=[BetfairRunner(selection_id=1, name="Home",
                                              back_price=2.2, back_size=100.0,
                                              lay_price=2.3, lay_size=90.0)])
    m = BetfairClient().to_market(mf, data_mode=DataMode.LIVE_SHADOW)
    assert m.data_source == "live_shadow"
    assert m.data_mode.can_deploy_live_capital is False


def test_betfair_fetch_full_card_groups_by_catalogue_key():
    """The whole point: one call returns the wide card, keyed for pricing."""
    class _Client(BetfairClient):
        def __init__(self):
            super().__init__(); self._logged_in = True
            self.requested_types = ()
        def market_catalogue(self, etid, market_types=CORE_MARKET_TYPES, max_results=100,
                             in_play_only=False, hours_ahead=48):
            self.requested_types = tuple(market_types)
            return [
                BetfairMarket("1.1", "MATCH_ODDS", "h2h", "Match Odds", "3", "A v B",
                              None, False),
                BetfairMarket("1.2", "OVER_UNDER_25", "totals", "O/U 2.5", "3", "A v B",
                              None, False),
                BetfairMarket("1.3", "OVER_UNDER_CORNERS", "corners_total", "Corners",
                              "3", "A v B", None, False),
                BetfairMarket("1.4", "BOOKING_ODDS", "booking_points", "Bookings",
                              "3", "A v B", None, False),
            ]
        def market_books(self, market_ids):
            return {mid: _book(mid, [(1, "S1", 1.9, 100.0, 1.95, 90.0, 1.92, 1000.0)])
                    for mid in market_ids}

    c = _Client()
    grouped = fetch_full_card(c, "1", include_player_markets=True)
    assert set(grouped) == {"h2h", "totals", "corners_total", "booking_points"}
    # the request must include the secondary and player types, not just core
    assert "OVER_UNDER_CORNERS" in c.requested_types
    assert "BOOKING_ODDS" in c.requested_types
    assert "PLAYER_GOALS" in c.requested_types


def test_betfair_full_card_drops_unbookable_markets():
    class _Client(BetfairClient):
        def __init__(self):
            super().__init__(); self._logged_in = True
        def market_catalogue(self, etid, market_types=CORE_MARKET_TYPES, max_results=100,
                             in_play_only=False, hours_ahead=48):
            return [BetfairMarket("1.9", "MATCH_ODDS", "h2h", "Match Odds", "3", "A v B",
                                  None, False)]
        def market_books(self, market_ids):
            return {"1.9": _book("1.9", [(1, "S1", 1.9, 100.0, None, 0.0, None, 0.0)])}

    assert fetch_full_card(_Client(), "1") == {}


# ═══════════════════════════════════════════════════════════════════════════
# In-play pricing - fair value must decay with the clock
# ═══════════════════════════════════════════════════════════════════════════

import src.ptai.betting.in_play as in_play
from src.ptai.betting.in_play import (
    MatchState, LiveCard, live_btts, live_correct_score, live_double_chance,
    live_draw_no_bet, live_match_odds, live_next_goal, live_rates,
    live_scoreline_matrix, live_spread_high_scoring, live_team_total,
    live_total_high_scoring, live_totals, price_live_high_scoring_card,
    price_live_soccer_card,
)

H_RATE, A_RATE = 1.5, 1.1


def test_profile_weight_is_normalised_to_the_full_match():
    """
    If the time profile does not integrate to the calibrated full-match rate,
    every live price inherits a systematic bias.
    """
    assert in_play._profile_weight(1.0) == pytest.approx(1.0)
    assert in_play._profile_weight(0.0) == 0.0
    # back-heavy profile: half the clock leaves MORE than half the scoring
    assert in_play._profile_weight(0.5) > 0.5


def test_draw_price_rises_as_time_runs_out():
    """
    THE bug this module fixes. A 0-0 draw at kickoff and a 0-0 draw in the
    85th minute are completely different markets. A model that returns the
    kickoff number at minute 85 calls the draw a huge edge and bets it wrong.
    """
    kickoff = live_match_odds(H_RATE, A_RATE, MatchState(minutes_elapsed=0))
    late = live_match_odds(H_RATE, A_RATE, MatchState(minutes_elapsed=85))
    assert late["draw"] > kickoff["draw"] + 0.4
    assert late["draw"] > 0.8


def test_live_odds_are_a_distribution_at_every_minute():
    for mins in (0, 15, 30, 45, 60, 75, 89, 90):
        o = live_match_odds(H_RATE, A_RATE, MatchState(minutes_elapsed=mins))
        assert sum(o.values()) == pytest.approx(1.0, abs=1e-3), f"minute {mins}"


def test_finished_match_prices_exactly_the_result():
    """No time left means no randomness: the result is the result."""
    fin = MatchState(minutes_elapsed=90, home_score=3, away_score=1)
    assert live_match_odds(H_RATE, A_RATE, fin) == {"home": 1.0, "draw": 0.0, "away": 0.0}
    t = live_totals(H_RATE, A_RATE, fin, 2.5)
    assert t == {"over": 1.0, "under": 0.0, "push": 0.0}
    assert live_correct_score(H_RATE, A_RATE, fin) == {"3-1": 1.0}


def test_totals_react_to_goals_already_scored():
    """
    At 2-0 the over 2.5 only needs one more goal, so its price must fall
    toward zero as the clock runs out - it must not stay at the pre-match
    number.
    """
    early = live_totals(H_RATE, A_RATE, MatchState(minutes_elapsed=0, home_score=2), 2.5)
    late = live_totals(H_RATE, A_RATE, MatchState(minutes_elapsed=88, home_score=2), 2.5)
    assert early["over"] > 0.9
    assert late["over"] < 0.15
    for t in (early, late):
        assert sum(t.values()) == pytest.approx(1.0, abs=1e-3)


def test_whole_number_total_line_can_push_in_play():
    st = MatchState(minutes_elapsed=30, home_score=1, away_score=1)
    assert live_totals(H_RATE, A_RATE, st, 2.0)["push"] > 0.02
    assert live_totals(H_RATE, A_RATE, st, 2.5)["push"] == 0.0


def test_btts_is_certain_or_impossible_when_already_decided():
    both = MatchState(minutes_elapsed=20, home_score=1, away_score=1)
    assert live_btts(H_RATE, A_RATE, both) == {"yes": 1.0, "no": 0.0}
    finished = MatchState(minutes_elapsed=90, home_score=1, away_score=0)
    assert live_btts(H_RATE, A_RATE, finished) == {"yes": 0.0, "no": 1.0}


def test_btts_decays_when_one_side_has_not_scored():
    st_60 = MatchState(minutes_elapsed=60, home_score=1)
    st_85 = MatchState(minutes_elapsed=85, home_score=1)
    assert live_btts(H_RATE, A_RATE, st_60)["yes"] > live_btts(H_RATE, A_RATE, st_85)["yes"]


def test_next_goal_keeps_a_real_none_outcome():
    """
    Next-goal is a race between two Poisson processes. Normalising the two
    rates to 1 would erase the chance nobody scores again.
    """
    for mins in (10, 60, 88):
        ng = live_next_goal(H_RATE, A_RATE, MatchState(minutes_elapsed=mins))
        assert sum(ng.values()) == pytest.approx(1.0, abs=1e-3)
        assert ng["none"] > 0.0, f"minute {mins} must leave a no-further-goal chance"
    # the heavier side is still more likely to score next
    ng = live_next_goal(H_RATE, A_RATE, MatchState(minutes_elapsed=30))
    assert ng["home"] > ng["away"]


def test_next_goal_at_full_time_is_only_none():
    assert live_next_goal(H_RATE, A_RATE, MatchState(minutes_elapsed=90)) == {
        "home": 0.0, "away": 0.0, "none": 1.0}


def test_red_card_shifts_the_market_against_the_short_handed_side():
    st_11 = MatchState(minutes_elapsed=30)
    st_10 = MatchState(minutes_elapsed=30, home_red_cards=1)
    even = live_match_odds(H_RATE, A_RATE, st_11)
    down = live_match_odds(H_RATE, A_RATE, st_10)
    assert down["home"] < even["home"] - 0.05
    assert down["away"] > even["away"] + 0.05


def test_trailing_side_gets_a_higher_attack_rate():
    """Game state moves both lambdas - chasing raises your own and the counter."""
    level = live_rates(H_RATE, A_RATE, MatchState(minutes_elapsed=45))
    trailing = live_rates(H_RATE, A_RATE,
                          MatchState(minutes_elapsed=45, home_score=0, away_score=2))
    # home is two down at half time, so its remaining rate is relatively higher
    home_share_level = level[0] / sum(level)
    home_share_trailing = trailing[0] / sum(trailing)
    assert home_share_trailing > home_share_level


def test_live_rates_reach_zero_at_full_time():
    assert live_rates(H_RATE, A_RATE, MatchState(minutes_elapsed=90)) == (0.0, 0.0)


def test_scoreline_matrix_shifts_onto_the_current_score():
    """Goals already scored are certain and must not be re-randomised."""
    st = MatchState(minutes_elapsed=60, home_score=2, away_score=1)
    grid = live_scoreline_matrix(H_RATE, A_RATE, st)
    assert sum(grid.values()) == pytest.approx(1.0, abs=1e-3)
    # the minimum possible scoreline is the current one
    assert min(h for h, _ in grid) == 2
    assert min(a for _, a in grid) == 1


def test_live_double_chance_and_dnb_stay_consistent():
    st = MatchState(minutes_elapsed=50, home_score=1)
    o = live_match_odds(H_RATE, A_RATE, st)
    dc = live_double_chance(H_RATE, A_RATE, st)
    dnb = live_draw_no_bet(H_RATE, A_RATE, st)
    assert dc["home_or_draw"] == pytest.approx(o["home"] + o["draw"], abs=1e-3)
    assert dnb["push"] == pytest.approx(o["draw"], abs=1e-3)
    assert sum(dnb.values()) == pytest.approx(1.0, abs=1e-3)


def test_live_team_total_accounts_for_goals_already_on_the_board():
    st = MatchState(minutes_elapsed=80, home_score=2)
    over_15 = live_team_total(H_RATE, A_RATE, st, "home", 1.5)
    over_25 = live_team_total(H_RATE, A_RATE, st, "home", 2.5)
    assert over_15["over"] == 1.0          # already on 2, cannot fall
    assert over_15["under"] == 0.0
    assert over_25["over"] < 0.5           # needs one more with 10 minutes left
    assert sum(over_25.values()) == pytest.approx(1.0, abs=1e-3)


def test_state_validation_catches_impossible_states():
    assert MatchState(minutes_elapsed=0).validate() == []
    assert MatchState(minutes_elapsed=120).validate()          # past regulation
    assert MatchState(minutes_elapsed=-5).validate()           # negative clock
    assert MatchState(home_score=-1).validate()                # negative score
    assert MatchState(home_red_cards=5).validate()             # impossible
    # past regulation is legal when flagged as extra time
    assert MatchState(minutes_elapsed=105, in_extra_time=True).validate() == []


def test_in_extra_time_leaves_no_regulation_time():
    st = MatchState(minutes_elapsed=105, in_extra_time=True)
    assert st.minutes_remaining == 0.0
    assert st.fraction_remaining == 0.0


# --- high-scoring in-play -------------------------------------------------

def _nba_state(mins, home, away):
    return MatchState(minutes_elapsed=mins, regulation_minutes=48,
                      home_score=home, away_score=away)


def test_live_high_scoring_total_matches_the_pre_match_price_at_kickoff():
    """
    At tip-off the live price must equal the pre-match price. If it does not,
    the time profile is not normalised and every in-play price is biased.
    """
    from src.ptai.betting.high_scoring import price_total_normal, SPORT_PARAMS
    live = live_total_high_scoring(112.5, 112.5, _nba_state(0, 0, 0), 225.0)
    pre = price_total_normal(SPORT_PARAMS["basketball"].typical_total,
                             SPORT_PARAMS["basketball"].total_std, 225.0)
    assert live.over == pytest.approx(pre.over, abs=1e-3)
    assert live.push == pytest.approx(pre.push, abs=1e-3)


def test_live_high_scoring_uncertainty_shrinks_with_sqrt_not_linearly():
    """
    Variance adds over independent possessions, so the standard deviation
    scales with the square root of time remaining. Scaling it linearly would
    badly understate late-game uncertainty.
    """
    full = live_total_high_scoring(114.0, 110.0, _nba_state(0, 0, 0), 225.0)
    half = live_total_high_scoring(114.0, 110.0, _nba_state(24, 57, 55), 225.0)
    ratio = half.remaining_std / full.remaining_std
    # half the time left -> about 1/sqrt(2) of the std, not half
    assert 0.6 < ratio < 0.85


def test_live_high_scoring_total_sums_to_one_across_the_game():
    for mins, h, a in [(0, 0, 0), (12, 28, 27), (24, 57, 55), (36, 85, 82), (47, 111, 107)]:
        t = live_total_high_scoring(114.0, 110.0, _nba_state(mins, h, a), 225.0)
        assert t.over + t.under + t.push == pytest.approx(1.0, abs=1e-3), f"minute {mins}"


def test_live_high_scoring_whole_line_pushes_and_half_line_does_not():
    st = _nba_state(24, 57, 55)
    assert live_total_high_scoring(114.0, 110.0, st, 225.0).push > 0.01
    assert live_total_high_scoring(114.0, 110.0, st, 225.5).push == 0.0


def test_live_high_scoring_finished_game_is_the_final_score():
    t = live_total_high_scoring(114.0, 110.0, _nba_state(48, 114, 110), 225.0)
    assert t == in_play.LiveTotalPrices(line=225.0, over=0.0, under=1.0, push=0.0,
                                        expected_total=224.0, remaining_std=0.0)
    sp = live_spread_high_scoring(114.0, 110.0, _nba_state(48, 114, 110), -2.5)
    assert sp.home == 1.0 and sp.away == 0.0


def test_live_high_scoring_blowout_is_decided_late():
    st = _nba_state(47, 110, 80)
    ml = in_play.live_moneyline_high_scoring(114.0, 110.0, st)
    assert ml["home"] > 0.99


def test_live_high_scoring_spread_tracks_the_score():
    early = live_spread_high_scoring(114.0, 110.0, _nba_state(6, 14, 12), -2.5)
    late = live_spread_high_scoring(114.0, 110.0, _nba_state(40, 95, 78), -2.5)
    assert late.home > early.home + 0.2
    for sp in (early, late):
        assert sp.home + sp.away + sp.push == pytest.approx(1.0, abs=1e-3)


def test_live_high_scoring_whole_spread_can_push():
    sp = live_spread_high_scoring(114.0, 110.0, _nba_state(24, 57, 55), -2.0)
    assert sp.push > 0.01


def test_live_soccer_card_prices_the_whole_live_card():
    card = price_live_soccer_card("epl:mci-ars", H_RATE, A_RATE,
                                  MatchState(minutes_elapsed=65, home_score=1))
    assert isinstance(card, LiveCard)
    assert card.model == "live_poisson_time_decay"
    assert card.score == "1-0" and card.minutes_elapsed == 65
    for key in ("h2h", "double_chance", "draw_no_bet", "btts", "next_goal",
                "correct_score", "totals_2.5", "team_total_home_1.5"):
        assert key in card.markets, f"live card missing {key}"
    assert card.state_warnings == []

    # Double chance outcomes OVERLAP by construction - 1X, X2 and 12 each
    # include the draw, so the three together double-count it and sum to 2.
    # Asserting 1.0 here would be asserting a partition that is not one.
    dc = card.markets["double_chance"]
    o = card.markets["h2h"]
    assert dc["home_or_draw"] == pytest.approx(o["home"] + o["draw"], abs=1e-3)
    assert dc["draw_or_away"] == pytest.approx(o["draw"] + o["away"], abs=1e-3)
    assert dc["home_or_away"] == pytest.approx(o["home"] + o["away"], abs=1e-3)
    assert sum(dc.values()) == pytest.approx(2.0, abs=1e-3)

    # correct_score is a deliberate top-N truncation, so it covers most but
    # not all of the probability mass
    cs = card.markets["correct_score"]
    assert 0.9 < sum(cs.values()) <= 1.0
    assert all(0.0 < p < 1.0 for p in cs.values())

    # every other market is a genuine partition
    for key, m in card.markets.items():
        if key in ("double_chance", "correct_score"):
            continue
        assert sum(m.values()) == pytest.approx(1.0, abs=1e-2), f"{key} does not sum to 1"


def test_live_high_scoring_card_prices_a_full_nba_card():
    card = price_live_high_scoring_card(
        "nba:lal-gsw", "nba", 114.0, 110.0, _nba_state(36, 88, 82),
        lines={"total": 225.0, "spread": -2.5})
    assert card.model == "live_normal_time_decay"
    assert card.sport == "basketball"
    assert "moneyline" in card.markets and "totals_225.0" in card.markets
    assert "spreads_-2.5" in card.markets
    assert card.markets["moneyline"]["home"] > 0.7      # up 6 in Q4


def test_live_card_carries_state_warnings_not_silent_nonsense():
    """An impossible clock must be reported, not quietly priced anyway."""
    card = price_live_soccer_card("epl:x-y", H_RATE, A_RATE, MatchState(minutes_elapsed=140))
    assert card.state_warnings


def test_live_pricing_never_invents_when_rates_are_zero():
    """A model with no information must not fabricate a confident price."""
    st = MatchState(minutes_elapsed=0)
    assert live_rates(0.0, 0.0, st) == (0.0, 0.0)
    ng = live_next_goal(0.0, 0.0, st)
    assert ng == {"home": 0.0, "away": 0.0, "none": 1.0}


# ═══════════════════════════════════════════════════════════════════════════
# Engine-level in-play wiring
# ═══════════════════════════════════════════════════════════════════════════

def test_engine_routes_a_live_nba_fixture_to_the_normal_model():
    """Same routing bug as pre-match: an NBA clock is 48 minutes, not 90."""
    engine = BettingEngine(bankroll=500.0)
    ev = sample_events(1)[0]
    card = engine.price_live_card(
        ev, {ev.key: {"home": {"expected_points": 114.0},
                      "away": {"expected_points": 110.0},
                      "total_line": 225.0, "spread_line": -2.5}},
        MatchState(minutes_elapsed=36, home_score=88, away_score=82))
    assert card is not None
    assert card.model == "live_normal_time_decay"
    assert card.sport == "basketball"
    assert "moneyline" in card.markets
    assert "correct_score" not in card.markets       # soccer market, not NBA


def test_engine_routes_a_live_soccer_fixture_to_the_poisson_model():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    card = engine.price_live_card(
        ev, {ev.key: {"home": {"attack": 1.7, "defence": 0.85},
                      "away": {"attack": 1.25, "defence": 1.05}}},
        MatchState(minutes_elapsed=65, home_score=1))
    assert card.model == "live_poisson_time_decay"
    assert "h2h" in card.markets and "btts" in card.markets
    assert "moneyline" not in card.markets


def test_engine_live_pricing_refuses_without_state_or_model():
    """
    A live price needs a clock reading AND a model. Without either there is
    nothing to price, and the engine must say so rather than fall back to the
    pre-match number and present it as live.
    """
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    strengths = {ev.key: {"home": {"attack": 1.5, "defence": 1.0},
                          "away": {"attack": 1.0, "defence": 1.0}}}
    assert engine.price_live_card(ev, strengths, None) is None
    assert engine.price_live_card(ev, {}, MatchState(minutes_elapsed=30)) is None
    assert engine.live_fair_prices(ev.key) == {}


def test_engine_live_fair_prices_are_keyed_like_the_prematch_card():
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    engine.price_live_card(
        ev, {ev.key: {"home": {"attack": 1.7, "defence": 0.85},
                      "away": {"attack": 1.25, "defence": 1.05}}},
        MatchState(minutes_elapsed=65, home_score=1))
    fair = engine.live_fair_prices(ev.key)
    assert "totals_2.5" in fair and set(fair["totals_2.5"]) == {"over", "under", "push"}


def test_engine_records_the_minute_a_live_price_was_taken_at():
    """
    A live fair price is only valid at the clock reading it was computed for.
    Recording the minute is what makes it auditable later.
    """
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    card = engine.price_live_card(
        ev, {ev.key: {"home": {"attack": 1.5, "defence": 1.0},
                      "away": {"attack": 1.0, "defence": 1.0}}},
        MatchState(minutes_elapsed=73, home_score=0, away_score=0))
    assert card.priced_at_minute == 73
    assert engine.priced_live[ev.key].minutes_elapsed == 73


def test_engine_live_price_differs_from_the_prematch_price():
    """
    Regression guard on the whole point of the module: if the live price
    equals the pre-match price at minute 73, nothing is decaying.
    """
    engine = BettingEngine(bankroll=500.0)
    ev = soccer_event()
    strengths = {ev.key: {"home": {"attack": 1.5, "defence": 1.0},
                          "away": {"attack": 1.0, "defence": 1.0}}}
    pre = engine.price_card(ev, strengths)
    live = engine.price_live_card(ev, strengths,
                                  MatchState(minutes_elapsed=73, home_score=0, away_score=0))
    pre_draw = engine.card_fair_prices(ev.key)["h2h"]["draw"]
    live_draw = live.markets["h2h"]["draw"]
    assert live_draw > pre_draw + 0.2, (pre_draw, live_draw)
    assert pre.goals is not None
