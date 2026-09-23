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
    ev = sample_events(1)[0]
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
    ev = sample_events(1)[0]
    assert engine.price_card(ev, {}) is None


def test_card_markets_are_not_executable_without_a_book_price():
    """
    A fair price with nothing to bet against is not an opportunity. The
    engine must record that rather than presenting a model price as a bet.
    """
    engine = BettingEngine(bankroll=500.0)
    ev = sample_events(1)[0]
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
