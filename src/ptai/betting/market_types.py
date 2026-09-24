"""
Market Type Catalogue - what can actually be bet, how it settles, when it voids.

The original betting layer handled one market (h2h). A real football book
offers twenty-plus per match, and each has different settlement mechanics:
an Asian handicap can PUSH and refund your stake, a quarter-line handicap
splits your stake in two and half-refunds, corners markets void if the match
is abandoned, and a red-card market settles on a second yellow.

Getting settlement wrong is not a modelling error, it is a money error: you
can be right about the match and still lose the bet, or win a bet the system
records as a loss. So settlement is explicit and per-market here, and every
market declares its own void conditions.

Line conventions
----------------
Goals/corners/cards totals are quoted as a decimal line. Asian handicap
lines can be whole (0.0, -1.0), half (-0.5, -1.5) or quarter (-0.25, -0.75).
A quarter line is two bets: half the stake on each adjacent line. That is
modelled explicitly rather than approximated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Match facts - the single record a settled match is reduced to
# ---------------------------------------------------------------------------

@dataclass
class MatchFacts:
    """
    Everything needed to settle any market on one match.

    Optional fields are None when the feed does not carry them; a market
    that needs a missing fact must report UNSETTLEABLE rather than guessing.
    """
    home_team: str
    away_team: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_corners_ht: Optional[int] = None
    away_corners_ht: Optional[int] = None
    home_cards: Optional[int] = None          # yellows
    away_cards: Optional[int] = None
    home_reds: Optional[int] = None
    away_reds: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_offsides: Optional[int] = None
    away_offsides: Optional[int] = None
    scorers: List[str] = field(default_factory=list)          # chronological
    home_scorers: List[str] = field(default_factory=list)
    away_scorers: List[str] = field(default_factory=list)
    goal_minutes: List[int] = field(default_factory=list)
    # (side, minute) in chronological order. Required to settle "race to N
    # goals" and "first goalscorer" - the final score alone cannot tell you
    # who reached a milestone first.
    goal_events: List[Tuple[str, int]] = field(default_factory=list)
    # Per-player actual counts, keyed by player name: {"goals": 1, "assists": 0,
    # "cards": 1, "shots": 3, "tackles": 2}. Player props cannot settle without
    # these, and the team's final score cannot supply one player's share.
    player_stats: Optional[Dict[str, Dict[str, int]]] = None
    # Did each player start? Required before applying a void-if-not-starting
    # rule, because books differ: some void the prop, some settle it anyway.
    # Unknown means no rule can be applied, so the prop stays unsettleable
    # rather than being resolved under a rule nobody confirmed.
    players_started: Optional[Dict[str, bool]] = None
    # Who scored the first goal, for first-scorer markets.
    first_scorer_name: Optional[str] = None
    minutes_played: int = 90
    abandoned: bool = False
    postponed: bool = False
    extra_time_applies: bool = False      # cup tie; some markets count ET, some don't

    @property
    def total_goals(self) -> Optional[int]:
        if self.home_goals is None or self.away_goals is None:
            return None
        return self.home_goals + self.away_goals

    @property
    def total_corners(self) -> Optional[int]:
        if self.home_corners is None or self.away_corners is None:
            return None
        return self.home_corners + self.away_corners

    @property
    def booking_points(self) -> Optional[int]:
        """Standard booking points: yellow = 10, red = 25."""
        if self.home_cards is None or self.away_cards is None:
            return None
        yellows = self.home_cards + self.away_cards
        reds = (self.home_reds or 0) + (self.away_reds or 0)
        return yellows * 10 + reds * 25

    @property
    def result(self) -> Optional[str]:
        if self.home_goals is None or self.away_goals is None:
            return None
        if self.home_goals > self.away_goals:
            return "home"
        if self.home_goals < self.away_goals:
            return "away"
        return "draw"

    @property
    def ht_result(self) -> Optional[str]:
        if self.home_goals_ht is None or self.away_goals_ht is None:
            return None
        if self.home_goals_ht > self.away_goals_ht:
            return "home"
        if self.home_goals_ht < self.away_goals_ht:
            return "away"
        return "draw"


class Settlement(str, Enum):
    WIN = "win"
    LOSE = "lose"
    PUSH = "push"          # stake refunded - line landed exactly
    HALF_WIN = "half_win"  # quarter-line: half the stake wins
    HALF_LOSE = "half_lose"
    VOID = "void"          # market did not happen - full refund
    UNSETTLEABLE = "unsettleable"   # missing facts, cannot decide


# ---------------------------------------------------------------------------
# Market catalogue
# ---------------------------------------------------------------------------

@dataclass
class MarketSpec:
    """Declarative definition of one bettable market."""
    key: str
    name: str
    sports: Tuple[str, ...]
    outcome_kind: str              # binary | three_way | handicap | total | list
    needs_line: bool = False
    line_is_quarterable: bool = False
    push_possible: bool = False
    settle: Optional[Callable[[MatchFacts, str, Optional[float]], Settlement]] = None
    void_rules: Tuple[str, ...] = ()
    model: str = ""                # which pricing routine covers it
    notes: str = ""


def _quarter_components(line: float) -> List[float]:
    """
    Split a quarter line into the two adjacent half lines it actually is.

    -0.25 -> [-0.5, 0.0];  0.25 -> [0.0, 0.5];  0.75 -> [0.5, 1.0]
    """
    lower = math.floor(line * 2) / 2.0
    return [lower, lower + 0.5]


def _combine_components(results: Sequence[Settlement]) -> Settlement:
    """
    Combine the two halves of a quarter-line bet.

    A quarter line is two half-stakes, so the outcome is the average of the
    two component results: win+push = half-win, lose+push = half-lose.
    Treating the quarter line as a single number pays the wrong amount.
    """
    kinds = {r for r in results}
    if kinds == {Settlement.WIN}:
        return Settlement.WIN
    if kinds == {Settlement.LOSE}:
        return Settlement.LOSE
    if kinds == {Settlement.PUSH}:
        return Settlement.PUSH
    if kinds == {Settlement.WIN, Settlement.PUSH}:
        return Settlement.HALF_WIN
    if kinds == {Settlement.LOSE, Settlement.PUSH}:
        return Settlement.HALF_LOSE
    if kinds == {Settlement.WIN, Settlement.LOSE}:
        return Settlement.PUSH
    return Settlement.UNSETTLEABLE


def _is_quarter(line: float) -> bool:
    return (abs(line * 4 - round(line * 4)) < 1e-9
            and abs(line * 2 - round(line * 2)) > 1e-9)


def _s_result(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """1X2 settlement."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    r = f.result
    if r is None:
        return Settlement.UNSETTLEABLE
    return Settlement.WIN if r == outcome else Settlement.LOSE


def _s_double_chance(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """1X / X2 / 12 - win if either of the two named results happens."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    r = f.result
    if r is None:
        return Settlement.UNSETTLEABLE
    covers = {"1X": ("home", "draw"), "X2": ("draw", "away"), "12": ("home", "away")}
    return Settlement.WIN if r in covers.get(outcome.upper(), ()) else Settlement.LOSE


def _s_draw_no_bet(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """Moneyline: draw refunds the stake."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    r = f.result
    if r is None:
        return Settlement.UNSETTLEABLE
    if r == "draw":
        return Settlement.PUSH
    return Settlement.WIN if r == outcome else Settlement.LOSE


def _s_handicap(f: MatchFacts, outcome: str, line: Optional[float]) -> Settlement:
    """
    Handicap settlement, whole/half/quarter lines.

    `line` is the home team's handicap (negative = home gives goals).
    outcome 'home' or 'away'. A quarter line (-0.25, -0.75) is two bets and
    can half-win or half-lose; that is returned explicitly because the
    payout differs from a full win.
    """
    if line is None:
        return Settlement.UNSETTLEABLE
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE

    diff = f.home_goals - f.away_goals

    def _one(ln: float) -> Settlement:
        # Margin from the bettor's perspective, positive = covers.
        # The handicap is applied to the team being backed: home -1.0 adds
        # -1.0 to the home score, away +0.25 adds +0.25 to the away score.
        # Writing the away case as -(diff + ln) applies the line on the wrong
        # side of the subtraction and reverses quarter-line results.
        margin = (diff + ln) if outcome == "home" else (ln - diff)
        if margin > 1e-9:
            return Settlement.WIN
        if abs(margin) <= 1e-9:
            return Settlement.PUSH
        return Settlement.LOSE

    if _is_quarter(line):
        return _combine_components([_one(c) for c in _quarter_components(line)])
    return _one(line)


def _s_total(f: MatchFacts, outcome: str, line: Optional[float],
             stat: str = "goals") -> Settlement:
    """
    Over/under on a stat. Supports quarter lines (over 2.25, under 2.75).

    stat selects which fact to read, so the same routine prices goals,
    corners, cards, shots and offsides.
    """
    if line is None:
        return Settlement.UNSETTLEABLE
    if f.abandoned:
        return Settlement.VOID

    total = {
        "goals": f.total_goals,
        "corners": f.total_corners,
        "cards": None if (f.home_cards is None or f.away_cards is None)
                 else f.home_cards + f.away_cards,
        "booking_points": f.booking_points,
        "shots": None if (f.home_shots is None or f.away_shots is None)
                 else f.home_shots + f.away_shots,
        "shots_on_target": None if (f.home_shots_on_target is None or f.away_shots_on_target is None)
                 else f.home_shots_on_target + f.away_shots_on_target,
        "offsides": None if (f.home_offsides is None or f.away_offsides is None)
                 else f.home_offsides + f.away_offsides,
    }.get(stat)

    if total is None:
        return Settlement.UNSETTLEABLE

    # Validate the side before settling. `outcome.lower().startswith("over")`
    # treated anything that was not "over" as "under", so an unrecognised
    # outcome string silently settled as LOSE - paying 0 instead of returning
    # the stake, and writing a wrong outcome into calibration. A None outcome
    # raised AttributeError straight out of the settlement entry point.
    _want = str(outcome or "").strip().lower()
    if _want not in ("over", "under"):
        return Settlement.UNSETTLEABLE
    is_over = _want == "over"
    # `diff` is always "how far past the line, in the bettor's favour", so
    # positive wins for either side. Inverting this silently pays out the
    # wrong side of every total on the card.
    def _one(ln: float) -> Settlement:
        diff = (total - ln) if is_over else (ln - total)
        if diff > 1e-9:
            return Settlement.WIN
        if abs(diff) <= 1e-9:
            return Settlement.PUSH
        return Settlement.LOSE

    if _is_quarter(line):
        return _combine_components([_one(c) for c in _quarter_components(line)])
    return _one(line)


def _settle_over_under_by_diff(diff: float, line: float) -> Settlement:
    """Kept for the team-goals path, which passes a raw count."""
    if diff > 1e-9:
        return Settlement.WIN
    if abs(diff) <= 1e-9:
        return Settlement.PUSH
    return Settlement.LOSE


def _s_most_stat(f: MatchFacts, outcome: str, stat: str, line=None) -> Settlement:
    """
    'Most corners' / 'most cards'. Settles off the stat itself.

    This must NOT reuse the result settler: a team can win 1-0 and still
    have fewer corners, and settling that off the scoreline pays the wrong
    side of the bet.
    """
    if f.abandoned:
        return Settlement.VOID
    pairs = {
        "corners": (f.home_corners, f.away_corners),
        "cards": (f.home_cards, f.away_cards),
        "shots": (f.home_shots, f.away_shots),
    }
    home, away = pairs.get(stat, (None, None))
    if home is None or away is None:
        return Settlement.UNSETTLEABLE
    if outcome == "home" or outcome == "1":
        return Settlement.WIN if home > away else Settlement.LOSE
    if outcome == "away" or outcome == "2":
        return Settlement.WIN if away > home else Settlement.LOSE
    return Settlement.WIN if home == away else Settlement.LOSE


def _s_stat_handicap(f: MatchFacts, outcome: str, stat: str, line: Optional[float]) -> Settlement:
    """
    Handicap on a non-goal stat (corners, cards, shots).

    Same quarter-line decomposition as the goal handicap, but it must read
    the stat itself - settling a corners handicap off the scoreline pays the
    wrong side whenever the dominant team does not win.
    """
    if line is None:
        return Settlement.UNSETTLEABLE
    if f.abandoned:
        return Settlement.VOID
    pairs = {
        "corners": (f.home_corners, f.away_corners),
        "cards": (f.home_cards, f.away_cards),
        "shots": (f.home_shots, f.away_shots),
    }
    home, away = pairs.get(stat, (None, None))
    if home is None or away is None:
        return Settlement.UNSETTLEABLE
    diff = home - away

    def _one(ln: float) -> Settlement:
        margin = (diff + ln) if outcome == "home" else (ln - diff)
        if margin > 1e-9:
            return Settlement.WIN
        if abs(margin) <= 1e-9:
            return Settlement.PUSH
        return Settlement.LOSE

    if _is_quarter(line):
        return _combine_components([_one(c) for c in _quarter_components(line)])
    return _one(line)


def _s_corners_handicap(f: MatchFacts, outcome: str, line=None) -> Settlement:
    return _s_stat_handicap(f, outcome, "corners", line)


def _s_most_corners(f: MatchFacts, outcome: str, line=None) -> Settlement:
    return _s_most_stat(f, outcome, "corners", line)


def _s_most_cards(f: MatchFacts, outcome: str, line=None) -> Settlement:
    return _s_most_stat(f, outcome, "cards", line)


def _s_btts(f: MatchFacts, outcome: str, line=None) -> Settlement:
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE
    both = f.home_goals >= 1 and f.away_goals >= 1
    want = outcome.lower() in ("yes", "true", "over")
    return Settlement.WIN if both == want else Settlement.LOSE


def _s_clean_sheet(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """outcome is the team name side: 'home' or 'away'."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE
    kept = f.away_goals == 0 if outcome == "home" else f.home_goals == 0
    return Settlement.WIN if kept else Settlement.LOSE


def _s_win_to_nil(f: MatchFacts, outcome: str, line=None) -> Settlement:
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE
    if outcome == "home":
        hit = f.home_goals > f.away_goals and f.away_goals == 0
    else:
        hit = f.away_goals > f.home_goals and f.home_goals == 0
    return Settlement.WIN if hit else Settlement.LOSE


def _s_correct_score(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """outcome is '2-1', or 'other' for anything outside the listed grid."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE
    actual = f"{f.home_goals}-{f.away_goals}"
    if outcome.lower() == "other":
        return Settlement.LOSE  # caller supplies the explicit grid; handled by engine
    return Settlement.WIN if actual == outcome else Settlement.LOSE


def _s_ht_ft(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """outcome like 'home/draw' or 'H/D'."""
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.ht_result is None or f.result is None:
        return Settlement.UNSETTLEABLE
    norm = {"H": "home", "D": "draw", "A": "away"}
    parts = [p.strip() for p in outcome.replace("/", "-").split("-")]
    if len(parts) != 2:
        return Settlement.UNSETTLEABLE
    ht = norm.get(parts[0].upper(), parts[0].lower())
    ft = norm.get(parts[1].upper(), parts[1].lower())
    return Settlement.WIN if (f.ht_result == ht and f.result == ft) else Settlement.LOSE


def _s_ht_result(f: MatchFacts, outcome: str, line=None) -> Settlement:
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.ht_result is None:
        return Settlement.UNSETTLEABLE
    return Settlement.WIN if f.ht_result == outcome else Settlement.LOSE


def _s_red_card(f: MatchFacts, outcome: str, line=None) -> Settlement:
    """
    Any red card in the match. A second yellow counts as a red - feeds
    should already have folded that in, but the void rule matters: red-card
    markets are usually void if the match is abandoned.
    """
    if f.abandoned:
        return Settlement.VOID
    if f.home_reds is None or f.away_reds is None:
        return Settlement.UNSETTLEABLE
    any_red = (f.home_reds + f.away_reds) > 0
    want = outcome.lower() in ("yes", "true", "over")
    return Settlement.WIN if any_red == want else Settlement.LOSE


def _settle_over_under(total: int, line: float, is_over: bool) -> Settlement:
    """Shared over/under settlement against one count, quarter-line aware."""
    diff = (total - line) if is_over else (line - total)
    is_quarter = abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9
    if diff > 0:
        return Settlement.WIN
    if diff == 0:
        return Settlement.HALF_WIN if is_quarter else Settlement.PUSH
    if is_quarter and diff > -0.5:
        return Settlement.HALF_LOSE
    return Settlement.LOSE


def _s_team_total(f: MatchFacts, outcome: str, line: Optional[float]) -> Settlement:
    """
    Team goals over/under. outcome format: 'home_over', 'away_under'.
    """
    if line is None:
        return Settlement.UNSETTLEABLE
    if f.abandoned:
        return Settlement.VOID
    parts = outcome.lower().split("_")
    if len(parts) != 2:
        return Settlement.UNSETTLEABLE
    side, direction = parts
    if side not in ("home", "away") or direction not in ("over", "under"):
        return Settlement.UNSETTLEABLE
    goals = f.home_goals if side == "home" else f.away_goals
    if goals is None:
        return Settlement.UNSETTLEABLE
    return _settle_over_under(goals, line, direction == "over")


def _settle_over_under(total: int, line: float, is_over: bool) -> Settlement:
    def _one(ln: float) -> Settlement:
        diff = (total - ln) if is_over else (ln - total)
        if diff > 1e-9:
            return Settlement.WIN
        if abs(diff) <= 1e-9:
            return Settlement.PUSH
        return Settlement.LOSE

    if _is_quarter(line):
        return _combine_components([_one(c) for c in _quarter_components(line)])
    return _one(line)


def _s_race_to(f: MatchFacts, outcome: str, line: Optional[float]) -> Settlement:
    """
    First team to reach N goals. outcome 'home' | 'away' | 'none'.

    Needs goal_events (side, minute) - the final score cannot tell you who
    reached the milestone first, so without chronology this refuses to
    settle rather than guessing from the aggregate.
    """
    if line is None or line < 1:
        return Settlement.UNSETTLEABLE
    if f.abandoned:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE

    n = int(line)
    counts = {"home": 0, "away": 0}
    winner = "none"
    for side, _minute in f.goal_events:
        if side in counts:
            counts[side] += 1
            if counts[side] >= n:
                winner = side
                break

    if winner == "none" and f.goal_events:
        # nobody reached N - but only trustworthy if we saw every goal
        if len(f.goal_events) != f.total_goals:
            return Settlement.UNSETTLEABLE
    elif winner == "none" and not f.goal_events:
        if f.home_goals >= n or f.away_goals >= n:
            return Settlement.UNSETTLEABLE   # someone reached it, we just lack chronology

    return Settlement.WIN if winner == outcome.lower() else Settlement.LOSE


# Which per-player count each prop market settles on.
PLAYER_PROP_STAT: Dict[str, str] = {
    "player_goals": "goals",
    "player_assists": "assists",
    "player_cards": "cards",
    "player_shots": "shots",
    "player_tackles": "tackles",
    "anytime_scorer": "goals",
}


def split_market_key(market_key: str) -> Tuple[str, Optional[str]]:
    """
    Separate a market key from an optional subject.

    Player props are keyed "player_goals:Harry Kane" by the pricing layer,
    because the same market exists once per player. `get_spec` looked the whole
    string up in MARKET_CATALOGUE, found nothing, and returned UNSETTLEABLE
    before any settlement logic ran - so every player prop was unsettleable no
    matter what facts were supplied.
    """
    if not market_key:
        return "", None
    base, sep, subject = market_key.partition(":")
    if not sep:
        return market_key, None
    subject = subject.strip()
    return base.strip(), (subject or None)


def _s_european_handicap(f: MatchFacts, outcome: str, line: Optional[float]) -> Settlement:
    """
    European handicap: the line IS applied, and the handicap draw is a real
    third outcome rather than a refund.

    This market was wired to `_s_result`, which ignores the line completely, so
    a home +2 bet on a 1-1 draw was settled as a plain draw - LOSE - when the
    handicap makes it a home win. `validate_line` also rejected negative lines
    for it, even though "home -1" is the standard way the market is quoted.
    """
    if line is None:
        return Settlement.UNSETTLEABLE
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if f.home_goals is None or f.away_goals is None:
        return Settlement.UNSETTLEABLE

    # line is the HOME handicap: -1.0 means home gives a goal.
    margin = (f.home_goals - f.away_goals) + float(line)
    if margin > 1e-9:
        adjusted = "home"
    elif margin < -1e-9:
        adjusted = "away"
    else:
        adjusted = "draw"

    norm = {"H": "home", "D": "draw", "A": "away"}
    want = norm.get(str(outcome).strip().upper(), str(outcome).strip().lower())
    if want not in ("home", "draw", "away"):
        return Settlement.UNSETTLEABLE
    return Settlement.WIN if adjusted == want else Settlement.LOSE


def _s_player_prop(f: MatchFacts, outcome: str, line: Optional[float],
                   player: Optional[str] = None,
                   market_key: Optional[str] = None) -> Settlement:
    """
    Settle a per-player market.

    Three things must be true before a verdict is possible, and each returns
    UNSETTLEABLE rather than a guess:

      1. the player must be named - the key carries them
      2. their start status must be known, because the void-if-not-starting
         rule differs between books
      3. their actual count must be supplied; the team total does not give one
         player's share

    First-scorer is handled here too: it needs the first goal's scorer, which a
    goal tally cannot reconstruct.
    """
    if f.abandoned or f.postponed:
        return Settlement.VOID
    if not player:
        return Settlement.UNSETTLEABLE

    started = (f.players_started or {}).get(player)
    if started is None:
        # The void-if-not-starting rule differs between books. Applying one
        # nobody confirmed would settle bets under a rule that may not exist.
        return Settlement.UNSETTLEABLE
    if started is False:
        return Settlement.VOID

    want = str(outcome or "").strip().lower()

    if market_key == "first_scorer":
        if want not in ("yes", "no") or f.first_scorer_name is None:
            return Settlement.UNSETTLEABLE
        first = str(f.first_scorer_name).strip().lower() == player.strip().lower()
        return Settlement.WIN if first == (want == "yes") else Settlement.LOSE

    stat = PLAYER_PROP_STAT.get(market_key or "")
    if stat is None:
        return Settlement.UNSETTLEABLE

    counts = (f.player_stats or {}).get(player)
    if not isinstance(counts, dict):
        return Settlement.UNSETTLEABLE
    count = counts.get(stat)
    if count is None:
        return Settlement.UNSETTLEABLE

    if want in ("over", "under"):
        if line is None:
            return Settlement.UNSETTLEABLE
        return _settle_over_under(int(count), float(line), want == "over")
    if want in ("yes", "no"):
        # anytime_scorer: "yes" means at least one.
        hit = int(count) > 0
        return Settlement.WIN if hit == (want == "yes") else Settlement.LOSE
    return Settlement.UNSETTLEABLE


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

SOCCER = ("soccer",)
ALL = ("soccer", "basketball", "football", "baseball", "hockey", "tennis")

MARKET_CATALOGUE: Dict[str, MarketSpec] = {
    # --- result markets -------------------------------------------------
    "h2h": MarketSpec(
        key="h2h", name="Match Result (1X2)", sports=ALL, outcome_kind="three_way",
        settle=_s_result, model="dixon_coles_poisson",
        void_rules=("Void if the match is abandoned or postponed",),
    ),
    "double_chance": MarketSpec(
        key="double_chance", name="Double Chance", sports=SOCCER, outcome_kind="binary",
        settle=_s_double_chance, model="derived_from_scoreline",
        void_rules=("Void if abandoned",),
    ),
    "draw_no_bet": MarketSpec(
        key="draw_no_bet", name="Draw No Bet", sports=ALL, outcome_kind="binary",
        push_possible=True, settle=_s_draw_no_bet, model="derived_from_scoreline",
        void_rules=("Void if abandoned", "Draw refunds the stake"),
    ),
    "asian_handicap": MarketSpec(
        key="asian_handicap", name="Asian Handicap", sports=SOCCER, outcome_kind="handicap",
        needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_handicap, model="derived_from_scoreline",
        void_rules=("Void if abandoned", "Whole/half line pushes refund stake",
                    "Quarter line half-wins or half-loses"),
        notes="A quarter line is two half-stakes on adjacent lines, priced separately",
    ),
    "european_handicap": MarketSpec(
        key="european_handicap", name="European Handicap (3-way)", sports=SOCCER,
        outcome_kind="three_way", needs_line=True, settle=_s_european_handicap,
        model="derived_from_scoreline", void_rules=("Void if abandoned",),
        notes="No push - the draw on the handicap is a third outcome. The line "
              "IS applied; settling this off the plain result pays the wrong side.",
    ),
    "spreads": MarketSpec(
        key="spreads", name="Point Spread", sports=("basketball", "football", "hockey"),
        outcome_kind="handicap", needs_line=True, push_possible=True,
        settle=_s_handicap, model="elo_margin", void_rules=("Void if abandoned",),
    ),

    # --- goals ----------------------------------------------------------
    "totals": MarketSpec(
        key="totals", name="Total Goals (Over/Under)", sports=ALL, outcome_kind="total",
        needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_total, model="derived_from_scoreline",
        void_rules=("Void if abandoned", "Quarter lines half-settle"),
    ),
    "btts": MarketSpec(
        key="btts", name="Both Teams To Score", sports=SOCCER, outcome_kind="binary",
        settle=_s_btts, model="derived_from_scoreline", void_rules=("Void if abandoned",),
    ),
    "team_goals": MarketSpec(
        key="team_goals", name="Team Goals Over/Under", sports=SOCCER, outcome_kind="total",
        needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_team_total, model="derived_from_scoreline",
        void_rules=("Void if abandoned",),
    ),
    "correct_score": MarketSpec(
        key="correct_score", name="Correct Score", sports=SOCCER, outcome_kind="list",
        settle=_s_correct_score, model="derived_from_scoreline",
        void_rules=("Void if abandoned",),
        notes="Usually offered as a fixed grid plus 'Other'",
    ),
    "clean_sheet": MarketSpec(
        key="clean_sheet", name="Clean Sheet", sports=SOCCER, outcome_kind="binary",
        settle=_s_clean_sheet, model="derived_from_scoreline", void_rules=("Void if abandoned",),
    ),
    "win_to_nil": MarketSpec(
        key="win_to_nil", name="Win To Nil", sports=SOCCER, outcome_kind="binary",
        settle=_s_win_to_nil, model="derived_from_scoreline", void_rules=("Void if abandoned",),
    ),
    "race_to_goals": MarketSpec(
        key="race_to_goals", name="Race To N Goals", sports=SOCCER, outcome_kind="three_way",
        needs_line=True, settle=_s_race_to, model="derived_from_scoreline",
        void_rules=("Void if abandoned", "Needs goal chronology to settle 'none'"),
    ),

    # --- half markets ---------------------------------------------------
    "half_time_result": MarketSpec(
        key="half_time_result", name="Half-Time Result", sports=SOCCER,
        outcome_kind="three_way", settle=_s_ht_result, model="derived_from_scoreline",
        void_rules=("Void if abandoned",),
    ),
    "half_time_goals": MarketSpec(
        key="half_time_goals", name="Half-Time Goals Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_total, model="derived_from_scoreline", void_rules=("Void if abandoned",),
    ),
    "ht_ft": MarketSpec(
        key="ht_ft", name="Half-Time / Full-Time", sports=SOCCER, outcome_kind="list",
        settle=_s_ht_ft, model="derived_from_scoreline", void_rules=("Void if abandoned",),
        notes="Nine combinations; the draws make these longshots",
    ),

    # --- corners --------------------------------------------------------
    "corners_total": MarketSpec(
        key="corners_total", name="Total Corners Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_total, model="corners_poisson", void_rules=("Void if abandoned",),
    ),
    "corners_handicap": MarketSpec(
        key="corners_handicap", name="Corners Handicap", sports=SOCCER, outcome_kind="handicap",
        needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_corners_handicap, model="corners_poisson", void_rules=("Void if abandoned",),
    ),
    "corners_1x2": MarketSpec(
        key="corners_1x2", name="Corners 1X2 (most corners)", sports=SOCCER,
        outcome_kind="three_way", settle=_s_most_corners, model="corners_poisson",
        void_rules=("Void if abandoned", "Draw = equal corners"),
    ),

    # --- cards ----------------------------------------------------------
    "cards_total": MarketSpec(
        key="cards_total", name="Total Cards Over/Under", sports=SOCCER, outcome_kind="total",
        needs_line=True, line_is_quarterable=True, push_possible=True,
        settle=_s_total, model="cards_poisson", void_rules=("Void if abandoned",),
    ),
    "booking_points": MarketSpec(
        key="booking_points", name="Booking Points Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_total, model="cards_poisson",
        void_rules=("Void if abandoned", "Yellow=10, Red=25 by default - check book rules"),
        notes="Booking point values differ by book; verify before trading",
    ),
    "red_card": MarketSpec(
        key="red_card", name="Any Red Card", sports=SOCCER, outcome_kind="binary",
        settle=_s_red_card, model="cards_poisson",
        void_rules=("Void if abandoned", "A second yellow counts as a red"),
    ),
    "cards_1x2": MarketSpec(
        key="cards_1x2", name="Most Cards", sports=SOCCER, outcome_kind="three_way",
        settle=_s_most_cards, model="cards_poisson",
        void_rules=("Void if abandoned", "Draw = equal cards"),
    ),

    # --- shots / misc ---------------------------------------------------
    "shots_total": MarketSpec(
        key="shots_total", name="Total Shots Over/Under", sports=SOCCER, outcome_kind="total",
        needs_line=True, push_possible=True, settle=_s_total, model="shots_poisson",
        void_rules=("Void if abandoned",),
    ),
    "shots_on_target_total": MarketSpec(
        key="shots_on_target_total", name="Shots On Target Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_total, model="shots_poisson", void_rules=("Void if abandoned",),
    ),
    "offsides_total": MarketSpec(
        key="offsides_total", name="Total Offsides Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_total, model="offsides_poisson", void_rules=("Void if abandoned",),
    ),

    # --- player props ---------------------------------------------------
    "player_goals": MarketSpec(
        key="player_goals", name="Player Goals Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Usually void if the player does not start - CHECK BOOK RULES",
                    "Some books void if player plays <N minutes"),
        notes="Start/no-start rules vary widely between books; this is the single "
              "biggest settlement risk in props",
    ),
    "player_shots": MarketSpec(
        key="player_shots", name="Player Shots Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",),
    ),
    "player_assists": MarketSpec(
        key="player_assists", name="Player Assists Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",),
    ),
    "player_cards": MarketSpec(
        key="player_cards", name="Player To Be Carded", sports=SOCCER,
        outcome_kind="binary", settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",),
    ),
    "player_tackles": MarketSpec(
        key="player_tackles", name="Player Tackles Over/Under", sports=SOCCER,
        outcome_kind="total", needs_line=True, push_possible=True,
        settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",),
    ),
    "anytime_scorer": MarketSpec(
        key="anytime_scorer", name="Anytime Goalscorer", sports=SOCCER,
        outcome_kind="binary", settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",),
    ),
    "first_scorer": MarketSpec(
        key="first_scorer", name="First Goalscorer", sports=SOCCER,
        outcome_kind="binary", settle=_s_player_prop, model="player_share_poisson",
        void_rules=("Void if player does not start",
                    "Usually stands if player comes on before the first goal"),
    ),
}


# Which stat each total-style market reads
MARKET_STAT: Dict[str, str] = {
    "totals": "goals",
    "half_time_goals": "goals",
    "corners_total": "corners",
    "cards_total": "cards",
    "booking_points": "booking_points",
    "shots_total": "shots",
    "shots_on_target_total": "shots_on_target",
    "offsides_total": "offsides",
}


def get_spec(market_key: str) -> Optional[MarketSpec]:
    """
    Look up a market, tolerating a ":subject" suffix.

    The pricing layer emits per-player keys like "player_goals:Harry Kane";
    looking those up verbatim found nothing, so every prop was unsettleable
    before its settlement code was even reached.
    """
    base, _subject = split_market_key(market_key)
    return MARKET_CATALOGUE.get(base) or MARKET_CATALOGUE.get(market_key)


def markets_for_sport(sport: str) -> List[MarketSpec]:
    return [m for m in MARKET_CATALOGUE.values() if sport in m.sports]


def markets_needing_model(model: str) -> List[MarketSpec]:
    return [m for m in MARKET_CATALOGUE.values() if m.model == model]


def validate_line(market_key: str, line: Optional[float]) -> Tuple[bool, str]:
    """
    Reject nonsense lines before they reach the pricing layer.

    A negative goals line or a 0.75-card line is a data error, and pricing it
    produces a confident answer about a market that cannot exist.
    """
    spec = get_spec(market_key)
    if spec is None:
        return False, f"unknown market {market_key}"
    if spec.needs_line and line is None:
        return False, f"{market_key} requires a line"
    if not spec.needs_line:
        return True, ""
    _handicap_markets = {"spreads", "asian_handicap", "european_handicap",
                         "corners_handicap"}
    if (line is not None and line < 0
            and spec.outcome_kind != "handicap"
            and market_key not in _handicap_markets):
        # Handicaps are negative by convention (home -1.0 gives a goal);
        # every other market counts things, and a negative count is an error.
        return False, f"{market_key} line {line} cannot be negative"
    if spec.line_is_quarterable and line is not None:
        q = line * 4
        if abs(q - round(q)) > 1e-9:
            return False, f"{market_key} line {line} is not on a quarter-line grid"
    return True, ""


def settle_market(market_key: str, facts: MatchFacts, outcome: str,
                  line: Optional[float] = None) -> Settlement:
    """
    Settle one bet. The single entry point the engine, paper trading and the
    backtester all use, so settlement cannot drift between them.
    """
    base, subject = split_market_key(market_key)
    spec = get_spec(market_key)
    if spec is None or spec.settle is None:
        return Settlement.UNSETTLEABLE

    ok, why = validate_line(market_key, line)
    if not ok:
        return Settlement.UNSETTLEABLE

    try:
        # Per-player markets dispatch first so the subject (the player) reaches
        # the settler. Their callback signature carries no subject, so routing
        # them here avoids changing the protocol every other market uses.
        if spec.settle is _s_player_prop:
            return _s_player_prop(facts, outcome, line, player=subject, market_key=base)

        # Stat totals all share one settler and differ only in which fact they
        # read. This dispatch must come BEFORE the generic call: it was
        # accidentally placed after a `return`, which silently made it dead code
        # and made every stat market read goals instead of its own stat, so
        # booking points of 55 were compared against a goals line.
        if base in ("totals", "half_time_goals", "corners_total", "cards_total",
                    "booking_points", "shots_total", "shots_on_target_total",
                    "offsides_total"):
            return _s_total(facts, outcome, line, stat=MARKET_STAT[base])

        return spec.settle(facts, outcome, line)
    except Exception as e:
        # A settler must always produce a verdict. UNSETTLEABLE returns the
        # stake, whereas letting an exception escape mid-settlement leaves the
        # caller with no answer and a position that cannot be closed. Loud, not
        # silent: a settler that raises is a bug to fix.
        logger.error(
            f"Settlement for {market_key} raised {type(e).__name__}: {e} - "
            f"reporting UNSETTLEABLE so the stake is not confiscated")
        return Settlement.UNSETTLEABLE


def payout_multiplier(settlement: Settlement, decimal_odds: float) -> float:
    """
    Convert a settlement into a stake multiplier.

    win       -> odds
    half_win  -> 1 + (odds-1)/2
    push      -> 1
    half_lose -> 0.5
    lose      -> 0
    void      -> 1
    """
    return {
        Settlement.WIN: decimal_odds,
        Settlement.HALF_WIN: 1.0 + (decimal_odds - 1.0) / 2.0,
        Settlement.PUSH: 1.0,
        Settlement.HALF_LOSE: 0.5,
        Settlement.LOSE: 0.0,
        Settlement.VOID: 1.0,
        Settlement.UNSETTLEABLE: 1.0,
    }[settlement]


def catalogue_report() -> Dict[str, int]:
    by_model: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    for m in MARKET_CATALOGUE.values():
        by_model[m.model] = by_model.get(m.model, 0) + 1
        by_kind[m.outcome_kind] = by_kind.get(m.outcome_kind, 0) + 1
    return {
        "total_markets": len(MARKET_CATALOGUE),
        "needing_line": sum(1 for m in MARKET_CATALOGUE.values() if m.needs_line),
        "push_possible": sum(1 for m in MARKET_CATALOGUE.values() if m.push_possible),
        "by_model": by_model,
        "by_outcome_kind": by_kind,
    }
