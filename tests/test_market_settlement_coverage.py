"""
Every catalogued market must be settleable.

The standing acceptance criterion: "every market type a real book offers must
be catalogued, priced from a consistent model, and settleable - including
push/half-win/void and player-prop start rules."

Player props failed all of it, in two independent ways:

  1. `_s_player_prop` was a stub that returned UNSETTLEABLE unconditionally.
     Its docstring referred to a `player_counts` mechanism in the engine that
     does not exist anywhere in the codebase - the same shape of gap as the
     account-health portfolio check reading a capability flag: a comment
     describing work that was never done.

  2. The pricing layer keys props as "player_goals:Harry Kane", because the
     same market exists once per player. `get_spec` looked the whole string up
     in MARKET_CATALOGUE, found nothing, and returned UNSETTLEABLE before any
     settlement logic ran. So even a correct `_s_player_prop` would have been
     unreachable, and the seven prop markets could never close - a prop bet
     would sit open forever and its outcome would never reach calibration.

These tests execute settlement for all 33 catalogued markets and assert that
each one produces a verdict when its facts are supplied, and refuses when they
are not.
"""

from __future__ import annotations

import pytest

from src.ptai.betting.market_types import (
    MARKET_CATALOGUE,
    MatchFacts,
    Settlement,
    get_spec,
    payout_multiplier,
    settle_market,
    split_market_key,
)

SOCCER_CARD = {
    "h2h": (2, 1, None),
    "double_chance": (2, 1, None),
    "draw_no_bet": (2, 1, None),
    "btts": (2, 1, None),
    "correct_score": (2, 1, None),
    "clean_sheet": (2, 1, None),
    "win_to_nil": (2, 1, None),
    "totals": (2, 1, 2.5),
    "spreads": (2, 1, -0.5),
    "asian_handicap": (2, 1, -0.25),
    "european_handicap": (2, 1, -1.0),
    "team_goals": (2, 1, 1.5),
    "half_time_result": (2, 1, None),
    "half_time_goals": (2, 1, 1.5),
    "ht_ft": (2, 1, None),
    "race_to_goals": (2, 1, 1),
    "corners_total": (2, 1, 8.5),
    "corners_1x2": (2, 1, None),
    "corners_handicap": (2, 1, -1.5),
    "cards_total": (2, 1, 3.5),
    "cards_1x2": (2, 1, None),
    "booking_points": (2, 1, 20.0),
    "red_card": (2, 1, None),
    "shots_total": (2, 1, 20.5),
    "shots_on_target_total": (2, 1, 8.5),
    "offsides_total": (2, 1, 3.5),
}


def _facts(**overrides) -> MatchFacts:
    base = dict(
        home_team="H", away_team="A",
        home_goals=2, away_goals=1,
        home_goals_ht=1, away_goals_ht=0,
        home_corners=6, away_corners=4,
        home_corners_ht=3, away_corners_ht=2,
        home_cards=2, away_cards=3,
        home_reds=0, away_reds=1,
        home_shots=14, away_shots=9,
        home_shots_on_target=6, away_shots_on_target=3,
        home_offsides=2, away_offsides=1,
        scorers=["Harry Kane", "Bukayo Saka"],
        home_scorers=["Harry Kane"],
        away_scorers=["Bukayo Saka"],
        goal_minutes=[12, 40],
        goal_events=[("home", 12), ("away", 40)],
        player_stats={
            "Harry Kane": {"goals": 1, "assists": 0, "cards": 1, "shots": 3, "tackles": 2},
            "Bukayo Saka": {"goals": 0, "assists": 1, "cards": 0, "shots": 2, "tackles": 4},
        },
        players_started={"Harry Kane": True, "Bukayo Saka": True},
        first_scorer_name="Harry Kane",
    )
    base.update(overrides)
    return MatchFacts(**base)


class TestEveryCataloguedMarketSettles:
    """No catalogued market may be permanently unsettleable."""

    def test_catalogue_size_is_what_we_think(self):
        assert len(MARKET_CATALOGUE) == 33

    @pytest.mark.parametrize("key", sorted(MARKET_CATALOGUE))
    def test_market_returns_a_settlement(self, key):
        """
        Every market must produce a Settlement for SOME valid input. A market
        that can only ever answer UNSETTLEABLE is not a market - it is a bet
        that can never be closed.
        """
        facts = _facts()
        assert settle_market(key, facts, "home") in set(Settlement), (
            f"{key} did not return a Settlement"
        )

    def test_goal_derived_markets_settle(self):
        facts = _facts()
        for key, (outcome, line) in {
            "h2h": ("home", None),
            "btts": ("yes", None),
            "double_chance": ("home", None),
            "draw_no_bet": ("home", None),
            "team_goals": ("home_over", 1.5),
            "totals": ("over", 2.5),
            "ht_ft": ("home-home", None),
            "half_time_result": ("home", None),
            "race_to_goals": ("home", 1),
            "correct_score": ("2-1", None),
        }.items():
            result = settle_market(key, facts, outcome, line)
            assert result != Settlement.UNSETTLEABLE, (
                f"{key} ({outcome}, line={line}) could not settle with full facts"
            )

    def test_per_stat_markets_settle(self):
        facts = _facts()
        for key, (outcome, line) in {
            "corners_total": ("over", 8.5),
            "cards_total": ("over", 3.5),
            "shots_total": ("over", 20.5),
            "shots_on_target_total": ("over", 8.5),
            "offsides_total": ("over", 3.5),
            "booking_points": ("over", 20.0),
            "corners_1x2": ("home", None),
            "cards_1x2": ("home", None),
            "red_card": ("yes", None),
        }.items():
            result = settle_market(key, facts, outcome, line)
            assert result != Settlement.UNSETTLEABLE, (
                f"{key} ({outcome}, line={line}) could not settle with full facts"
            )

    def test_handicaps_settle(self):
        facts = _facts()
        for key, line in (("spreads", -0.5), ("asian_handicap", -0.25),
                          ("european_handicap", -1.0), ("corners_handicap", -1.5)):
            result = settle_market(key, facts, "home", line)
            assert result != Settlement.UNSETTLEABLE, f"{key} line={line}"


class TestPlayerPropsSettle:
    """The seven prop markets, which previously could not settle at all."""

    PROPS = ["anytime_scorer", "first_scorer", "player_goals", "player_assists",
             "player_cards", "player_shots", "player_tackles"]

    @pytest.mark.parametrize("prop", PROPS)
    def test_prop_key_resolves_with_a_subject(self, prop):
        """
        The pricing layer emits "player_goals:Harry Kane". get_spec used to
        return None for that, so settlement never ran.
        """
        assert get_spec(f"{prop}:Harry Kane") is not None
        assert get_spec(f"{prop}:Harry Kane") is get_spec(prop)

    @pytest.mark.parametrize("prop", PROPS)
    def test_prop_can_reach_a_verdict(self, prop):
        facts = _facts()
        outcome = "yes" if prop in ("anytime_scorer", "first_scorer") else "over"
        line = None if prop in ("anytime_scorer", "first_scorer") else 0.5
        result = settle_market(f"{prop}:Harry Kane", facts, outcome, line)
        assert result != Settlement.UNSETTLEABLE, (
            f"{prop} could not settle despite full player facts"
        )

    def test_split_market_key(self):
        assert split_market_key("player_goals:Harry Kane") == ("player_goals", "Harry Kane")
        assert split_market_key("h2h") == ("h2h", None)
        assert split_market_key("player_goals:") == ("player_goals", None)
        assert split_market_key("") == ("", None)

    def test_goals_prop_over_and_under(self):
        facts = _facts()
        assert settle_market("player_goals:Harry Kane", facts, "over", 0.5) == Settlement.WIN
        assert settle_market("player_goals:Harry Kane", facts, "under", 0.5) == Settlement.LOSE

    def test_goals_prop_push_on_an_integer_line(self):
        facts = _facts()
        assert settle_market("player_goals:Harry Kane", facts, "over", 1.0) == Settlement.PUSH

    def test_goals_prop_half_win_on_a_quarter_line(self):
        facts = _facts()
        assert settle_market("player_goals:Harry Kane", facts, "over", 0.75) == Settlement.HALF_WIN
        assert settle_market("player_goals:Harry Kane", facts, "under", 1.25) == Settlement.HALF_WIN

    def test_goals_prop_half_lose_on_a_quarter_line(self):
        facts = _facts()
        assert settle_market("player_goals:Harry Kane", facts, "under", 0.75) == Settlement.HALF_LOSE

    def test_each_prop_uses_its_own_stat(self):
        """
        A prop must not fall back to goals. Kane has 1 goal, 0 assists, 1 card,
        3 shots, 2 tackles.
        """
        facts = _facts()
        assert settle_market("player_assists:Harry Kane", facts, "over", 0.5) == Settlement.LOSE
        assert settle_market("player_assists:Harry Kane", facts, "under", 0.5) == Settlement.WIN
        assert settle_market("player_shots:Harry Kane", facts, "over", 2.5) == Settlement.WIN
        assert settle_market("player_tackles:Harry Kane", facts, "over", 2.5) == Settlement.LOSE
        assert settle_market("player_cards:Harry Kane", facts, "over", 0.5) == Settlement.WIN

    def test_anytime_scorer_yes_and_no(self):
        facts = _facts()
        assert settle_market("anytime_scorer:Harry Kane", facts, "yes") == Settlement.WIN
        assert settle_market("anytime_scorer:Harry Kane", facts, "no") == Settlement.LOSE
        assert settle_market("anytime_scorer:Bukayo Saka", facts, "no") == Settlement.WIN
        assert settle_market("anytime_scorer:Bukayo Saka", facts, "yes") == Settlement.LOSE

    def test_first_scorer_uses_the_first_goal_not_the_tally(self):
        """
        Both players scored, but only one scored first. A goal tally cannot
        determine this, which is why the market needs the chronology.
        """
        facts = _facts()
        assert settle_market("first_scorer:Harry Kane", facts, "yes") == Settlement.WIN
        assert settle_market("first_scorer:Bukayo Saka", facts, "yes") == Settlement.LOSE

    def test_first_scorer_without_the_first_scorer_refuses(self):
        facts = _facts(first_scorer_name=None)
        assert settle_market("first_scorer:Harry Kane", facts, "yes") == Settlement.UNSETTLEABLE


class TestPlayerPropStartRule:
    """
    "player-prop start rules" are named in the acceptance criteria. The rule
    differs between books, so it is applied only when the start status is
    known - guessing would settle bets under a rule nobody confirmed.
    """

    def test_did_not_start_voids(self):
        facts = _facts(
            player_stats={"Sub Guy": {"goals": 0, "assists": 0, "cards": 0,
                                      "shots": 0, "tackles": 0}},
            players_started={"Sub Guy": False})
        assert settle_market("player_goals:Sub Guy", facts, "over", 0.5) == Settlement.VOID

    def test_unknown_start_status_is_unsettleable(self):
        facts = _facts(
            player_stats={"Sub Guy": {"goals": 0}}, players_started={})
        assert settle_market("player_goals:Sub Guy", facts, "over", 0.5) == Settlement.UNSETTLEABLE, (
            "without a confirmed start rule the prop must not be settled"
        )

    def test_started_but_no_stat_line_is_unsettleable(self):
        facts = _facts(player_stats={}, players_started={"Harry Kane": True})
        assert settle_market("player_goals:Harry Kane", facts, "over", 0.5) == Settlement.UNSETTLEABLE

    def test_unnamed_player_is_unsettleable(self):
        facts = _facts()
        assert settle_market("player_goals", facts, "over", 0.5) == Settlement.UNSETTLEABLE

    def test_abandoned_match_voids(self):
        facts = _facts(abandoned=True)
        assert settle_market("player_goals:Harry Kane", facts, "over", 0.5) == Settlement.VOID


class TestMissingFactsNeverGuess:
    """
    The counterpart: a market that lacks its facts must report UNSETTLEABLE.
    Recording a guessed outcome writes a permanent, wrong calibration point.
    """

    def test_goals_only_facts_settle_only_goal_markets(self):
        sparse = MatchFacts(home_team="H", away_team="A",
                            home_goals=2, away_goals=1)
        # Markets whose outcome the goals alone determine. Everything else
        # needs corners, cards, shots, offsides, half-time or player facts.
        derivable = {
            "h2h", "btts", "double_chance", "draw_no_bet", "team_goals",
            "totals", "clean_sheet", "win_to_nil", "european_handicap",
            "spreads", "correct_score", "asian_handicap", "half_time_goals",
        }
        guessed = []
        for key in MARKET_CATALOGUE:
            if key in derivable:
                continue
            line = 2.5 if MARKET_CATALOGUE[key].needs_line else None
            if settle_market(key, sparse, "home", line) != Settlement.UNSETTLEABLE:
                guessed.append(key)
        assert not guessed, (
            f"these markets settled without the facts they need: {guessed}"
        )

    @pytest.mark.parametrize("key,outcome,line", [
        ("corner", "home", None),
        ("corners_total", "home", None),        # needs a line
        ("totals", None, 2.5),                  # needs an outcome
        ("player_goals:Harry Kane", "sideways", 0.5),
        ("unknown_market", "home", None),
    ])
    def test_bad_input_is_unsettleable(self, key, outcome, line):
        facts = _facts()
        assert settle_market(key, facts, outcome, line) == Settlement.UNSETTLEABLE

    def test_unknown_market_is_unsettleable(self):
        assert settle_market("no_such_market", _facts(), "home") == Settlement.UNSETTLEABLE


class TestPayoutMultipliers:
    """
    The payouts that make push/half-win/void materially different from win/lose.
    """

    @pytest.mark.parametrize("settlement,odds,expected", [
        (Settlement.WIN, 2.5, 2.5),
        (Settlement.LOSE, 2.5, 0.0),
        (Settlement.PUSH, 2.5, 1.0),
        (Settlement.VOID, 2.5, 1.0),
        (Settlement.HALF_WIN, 2.5, 1.75),    # 1 + (2.5-1)/2
        (Settlement.HALF_LOSE, 2.5, 0.5),
        (Settlement.WIN, 1.9, 1.9),
        (Settlement.HALF_WIN, 1.9, 1.45),
    ])
    def test_known_multipliers(self, settlement, odds, expected):
        assert payout_multiplier(settlement, odds) == pytest.approx(expected)

    def test_a_push_returns_the_stake(self):
        assert payout_multiplier(Settlement.PUSH, 5.0) == 1.0

    def test_a_void_returns_the_stake(self):
        assert payout_multiplier(Settlement.VOID, 5.0) == 1.0

    def test_void_and_push_differ_from_win(self):
        for odds in (1.5, 2.0, 3.0):
            assert payout_multiplier(Settlement.PUSH, odds) != payout_multiplier(Settlement.WIN, odds)
            assert payout_multiplier(Settlement.VOID, odds) != payout_multiplier(Settlement.WIN, odds)
            assert payout_multiplier(Settlement.HALF_WIN, odds) != payout_multiplier(Settlement.WIN, odds)
            assert payout_multiplier(Settlement.HALF_LOSE, odds) != payout_multiplier(Settlement.LOSE, odds)
