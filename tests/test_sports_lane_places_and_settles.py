"""
The sports lane has the same lifecycle as the trading lane.

It could price a full match card and had nowhere to put the bets: `run_cycle`
returned "3 executable" and dropped them. No position, no settlement, no P&L, no
learning, no panel - a calculator next to a trading system, which is what "the
sports side looks lacking compared to the trading side" means.

These tests pin the lifecycle: quoted -> placed -> recorded -> settled -> learned,
and the four things it must never do:

  * invent a result (a bet settles only on a FINISHED fixture with a score);
  * refund what it could not read (an unmatchable outcome stays OPEN);
  * accept a bet it can never settle (corners/cards/btts are refused with the
    reason on them, quarter-line Asian handicaps included);
  * settle against an assumption (an Elo rating is only given to a team with real
    results behind it).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from ptai.betting.positions import SportsBook, classify_market
from ptai.betting.ratings import MIN_MATCHES, RatingsBook
from ptai.storage.db import Storage


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@dataclass
class FakeEvent:
    # Distinct fixtures need distinct keys; a real feed supplies that, and the
    # tests need to be able to say "these are three different matches".
    key_override: Optional[str] = None
    event_id: str = "1"
    sport: str = "soccer"
    league: str = "epl"
    home_team: str = "Arsenal"
    away_team: str = "Chelsea FC"
    commence_time: datetime = datetime(2026, 9, 20, tzinfo=timezone.utc)
    is_live: bool = False
    status: str = "STATUS_SCHEDULED"
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    provider: str = "test-feed"
    raw: Dict[str, Any] = None

    @property
    def key(self) -> str:
        if self.key_override:
            return self.key_override
        return f"{self.league}:{self.away_team.strip().lower().replace(' ', '-')}-at-{self.home_team.strip().lower().replace(' ', '-')}"


def finished(hg=2, ag=1, home="Arsenal", away="Chelsea FC", league="epl",
             key_override: Optional[str] = None) -> FakeEvent:
    return FakeEvent(league=league, home_team=home, away_team=away,
                     status="STATUS_FINAL", home_score=hg, away_score=ag,
                     raw={}, key_override=key_override)


def opportunity(**over) -> Dict[str, Any]:
    base = {
        "opportunity_id": "bet-1", "event_key": "epl:chelsea-fc-at-arsenal",
        "sport": "soccer", "league": "epl", "market_type": "h2h",
        "outcome": "Arsenal", "side": "back", "price": 2.10, "book": "Pinnacle",
        "stake": 3.0, "liability": 3.0, "model_prob": 0.55, "fair_prob": 0.55,
        "edge": 0.155, "edge_pct": 15.5, "conservative_prob": 0.5,
        "conservative_edge": 0.05, "net_ev_usd": 0.46, "kelly_stake": 3.0,
        "uncertainty": 0.05, "confidence": 0.6, "data_mode": "live_shadow",
        "data_source": "sports_live_feed", "commission_pct": 0.0,
        "is_arb": False, "executable": True, "blockers": [], "warnings": [],
        "reasoning": "value", "created_at": datetime.now(timezone.utc).isoformat(),
        "line": None,
    }
    base.update(over)
    return base


def cycle(top: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"ok": True, "events": 4, "opportunities": len(top), "top": top,
            "executable": sum(1 for o in top if o.get("executable"))}


class StubTracker:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.resolutions: List[Dict[str, Any]] = []

    def record_trade(self, **kw):
        self.trades.append(kw)
        return True

    def record_resolution(self, trade_id, actual_outcome, pnl):
        self.resolutions.append({"trade_id": trade_id,
                                 "actual_outcome": actual_outcome, "pnl": pnl})
        return True


@pytest.fixture()
def book(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    storage.set_bankroll(50.0)
    tracker = StubTracker()
    yield SportsBook(storage=storage, tracker=tracker, min_stake=1.0)
    storage.close()


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------

def test_a_bet_is_recorded_and_learned_from(book):
    result = book.place(cycle([opportunity()]))
    assert result["placed"] == 1
    bet = result["bets"][0]
    assert bet["stake_usd"] == 3.0
    assert bet["execution_mode"] == "paper"
    assert bet["liability_usd"] == 3.0

    open_bets = book.open_bets()
    assert len(open_bets) == 1
    assert open_bets[0]["outcome"] == "Arsenal"

    # The stake reached the learning record, so a settled bet reaches
    # qualification and the money guard like every other trade.
    assert book.tracker.trades and book.tracker.trades[0]["trade_id"] == bet["bet_id"]
    assert book.tracker.trades[0]["strategy"] == "sports:h2h"


def test_a_lay_bet_records_its_liability(book):
    result = book.place(cycle([opportunity(side="lay", price=1.8)]))
    bet = result["bets"][0]
    assert bet["side"] == "lay"
    assert bet["liability_usd"] == pytest.approx(3.0 * 0.8)


def test_a_market_that_cannot_be_settled_is_refused_with_the_reason(book):
    result = book.place(cycle([opportunity(market_type="cards_total",
                                           outcome="Over")]))
    assert result["placed"] == 0
    assert result["refused"] == 1
    reason = result["refusals"][0]["reasons"][0]
    assert "cannot be settled" in reason
    assert "card statistics" in reason


def test_a_quarter_line_handicap_is_refused_rather_than_guessed(book):
    kind, line, refusal = classify_market("asian_handicap_-0.25")
    assert kind == "spreads" and line == -0.25
    assert "quarter-line" in refusal
    result = book.place(cycle([opportunity(market_type="asian_handicap_-0.25",
                                           outcome="home", line=-0.25)]))
    assert result["placed"] == 0


def test_a_totals_bet_keeps_the_line_it_was_placed_on(book):
    result = book.place(cycle([opportunity(market_type="totals_2.5",
                                           outcome="Over")]))
    assert result["placed"] == 1, result["refusals"]
    assert result["bets"][0]["line"] == 2.5


def test_only_one_bet_per_fixture_identity(book):
    first = book.place(cycle([opportunity()]))
    assert first["placed"] == 1
    again = book.place(cycle([opportunity()]))
    assert again["placed"] == 0, "the same bet was placed twice"


def test_the_limit_bounds_a_cycle(book):
    book.max_bets_per_cycle = 5
    top = [opportunity(outcome="Arsenal", event_key="epl:a"),
           opportunity(outcome="Chelsea", event_key="epl:b")]
    assert book.place(cycle(top), limit=1)["placed"] == 1


def test_a_zero_stake_is_refused_not_placed(book):
    result = book.place(cycle([opportunity(stake=0.0)]))
    assert result["placed"] == 0
    assert any("stake" in r for r in result["refusals"][0]["reasons"])


def test_a_missing_price_is_refused(book):
    result = book.place(cycle([opportunity(price=0.0)]))
    assert result["placed"] == 0
    assert any("price" in r for r in result["refusals"][0]["reasons"])


def test_the_money_guard_is_consulted_like_any_other_order(book):
    class StoppingGuard:
        def check(self, amount, lane, bankroll, venue_id=""):
            class D:
                refused = True
                reason = "daily loss limit reached"
                approved_usd = 0.0
            return D()

    result = book.place(cycle([opportunity()]), guard=StoppingGuard())
    assert result["placed"] == 0
    assert "money guard" in result["refusals"][0]["reasons"][0]


def test_a_guard_that_cannot_judge_refuses_the_bet(book):
    class BrokenGuard:
        def check(self, *a, **k):
            raise RuntimeError("boom")

    result = book.place(cycle([opportunity()]), guard=BrokenGuard())
    assert result["placed"] == 0
    assert "money guard failed" in result["refusals"][0]["reasons"][0]


# --------------------------------------------------------------------------
# settlement
# --------------------------------------------------------------------------

def test_a_finished_fixture_settles_the_bet_and_pays_it(book):
    book.place(cycle([opportunity(price=2.10)]))
    import asyncio

    report = asyncio.run(book.settle(events=[finished(2, 1)]))
    assert report["settled"] == 1
    row = report["bets"][0]
    assert row["status"] == "won"
    assert row["pnl"] == pytest.approx(3.0 * 1.10, abs=0.01)
    assert book.open_bets() == []
    assert book.tracker.resolutions[0]["actual_outcome"] == 1.0


def test_a_lost_bet_costs_exactly_the_stake(book):
    book.place(cycle([opportunity(price=2.10)]))
    import asyncio

    report = asyncio.run(book.settle(events=[finished(0, 3)]))
    assert report["bets"][0]["status"] == "lost"
    assert report["bets"][0]["pnl"] == -3.0
    assert book.tracker.resolutions[0]["actual_outcome"] == 0.0


def test_an_unfinished_fixture_does_not_settle_anything(book):
    import asyncio

    book.place(cycle([opportunity()]))
    pending = FakeEvent(status="STATUS_IN_PROGRESS", is_live=True, home_score=1,
                        away_score=0, raw={})
    report = asyncio.run(book.settle(events=[pending]))
    assert report["settled"] == 0
    assert report["awaiting_result"] == 1
    assert len(book.open_bets()) == 1, "a live fixture settled a bet"


def test_a_finished_fixture_with_no_score_stays_open(book):
    import asyncio

    book.place(cycle([opportunity()]))
    no_score = FakeEvent(status="STATUS_FINAL", raw={})
    report = asyncio.run(book.settle(events=[no_score]))
    assert report["settled"] == 0
    assert report["unreadable"][0]["reason"].startswith("the fixture is FINISHED")
    assert len(book.open_bets()) == 1


def test_an_outcome_the_label_cannot_be_matched_is_not_refunded(book):
    import asyncio

    # A book label nobody can read is NOT a refund: refunding it would invent
    # money the venue never returned.
    book.place(cycle([opportunity(outcome="Real Madrid")]))
    report = asyncio.run(book.settle(events=[finished(2, 1)]))
    assert report["settled"] == 0
    assert report["open"] == 1
    assert "could not be matched" in report["unreadable"][0]["reason"]
    assert len(book.open_bets()) == 1


def test_a_totals_push_returns_the_stake_and_teaches_nothing(book):
    import asyncio

    book.place(cycle([opportunity(market_type="totals_3.0", outcome="Over")]))
    report = asyncio.run(book.settle(events=[finished(2, 1)]))
    assert report["bets"][0]["status"] == "void"
    assert book.tracker.resolutions == [], (
        "a refunded bet was recorded as a won or lost outcome")


def test_settlement_needs_no_feed_to_report_that_it_has_none(book):
    import asyncio

    book.data = None
    book.place(cycle([opportunity()]))
    report = asyncio.run(book.settle())
    assert report["settled"] == 0
    assert "no results feed" in report["reason"]


# --------------------------------------------------------------------------
# ratings
# --------------------------------------------------------------------------

def test_a_team_is_not_rated_until_it_has_results(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    try:
        ratings = RatingsBook(storage=storage)
        one = finished(hg=2, ag=1)
        ratings.update([one])
        assert ratings.rated_table("epl") == {}, (
            "a team was given a rating before it had any real results")
        assert ratings.ratings_for("epl")["Arsenal"]["matches"] == 1
    finally:
        storage.close()


def test_three_results_produce_a_rating_that_moves_with_the_score(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    try:
        ratings = RatingsBook(storage=storage)
        events = []
        for i, opponent in enumerate(("Everton", "Leeds", "Wolves")):
            events.append(finished(hg=3, ag=0, away=opponent,
                                   key_override=f"epl:match-{i}"))
        ratings.update(events)
        table = ratings.rated_table("epl")
        # Three wins earn Arsenal a rating; each opponent has played once and is
        # still UNRATED, which is the whole point of the minimum.
        assert "Arsenal" in table
        assert table["Arsenal"]["rating"] > 1500
        assert "Everton" not in table
        assert ratings.ratings_for("epl")["Everton"]["matches"] == 1
    finally:
        storage.close()


def test_a_result_is_never_counted_twice(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    try:
        ratings = RatingsBook(storage=storage)
        one = finished(2, 1)
        ratings.update([one])
        again = ratings.update([one])
        assert again["updated"] == 0 and again["already_counted"] == 1
        assert ratings.ratings_for("epl")["Arsenal"]["matches"] == 1
    finally:
        storage.close()


def test_engine_inputs_skip_fixtures_with_an_unrated_side(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    try:
        ratings = RatingsBook(storage=storage)
        # Arsenal and Everton have both played three real matches, so BOTH sides
        # of the upcoming fixture can be rated.
        for i in range(MIN_MATCHES):
            ratings.update([finished(hg=2, ag=0, away="Everton",
                                     key_override=f"epl:a-{i}")])
        upcoming = FakeEvent(home_team="Arsenal", away_team="Everton", raw={})
        strengths, table = ratings.engine_inputs([upcoming])
        assert upcoming.key in table
        assert table[upcoming.key]["home"]["rating"] > 1500
        assert strengths[upcoming.key]["home"]["games"] >= MIN_MATCHES
        # A fixture with a team nobody has seen is left out entirely: handing the
        # model a default 1500 would be a forecast with nothing behind it.
        stranger = FakeEvent(home_team="Arsenal", away_team="Nobody FC", raw={})
        strengths2, table2 = ratings.engine_inputs([stranger])
        assert stranger.key not in table2
    finally:
        storage.close()


def test_the_snapshot_says_how_a_rating_is_earned(tmp_path):
    storage = Storage(db_path=str(tmp_path / "ptai.db"))
    try:
        snap = RatingsBook(storage=storage).snapshot()
        assert snap["min_matches"] == MIN_MATCHES
        assert "Elo from finished fixtures" in snap["how"]
    finally:
        storage.close()


# --------------------------------------------------------------------------
# the agent's own lane: one fetch feeds ratings, pricing, placement and purse
# --------------------------------------------------------------------------
#
# The unit tests above prove each piece. These drive the price of the PRODUCT:
# TradingAgentV3's own lane method, with only the feed stubbed, so "bets are
# placed and settled through the same ledger as the trading side" is checked
# against the code the agent actually runs - not a re-implementation of it.

class _StubSportsData:
    """A feed. Supplies fixtures; refuses to invent them."""

    def __init__(self, events):
        self._events = list(events)
        self.asked = []

    async def fetch_events(self, leagues=None, **kw):
        self.asked.append(tuple(leagues or ()))
        return list(self._events)


def test_the_agents_lane_places_a_bet_and_debits_the_paper_purse(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    from ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    try:
        upcoming = FakeEvent(key_override="epl:upcoming", raw={})
        results = [finished(2, 1, key_override=f"epl:m-{i}") for i in range(3)]
        agent.betting_engine.data = _StubSportsData(results + [upcoming])

        async def fake_cycle(**kw):
            # The engine is given the SAME fixtures the lane fetched - the
            # single-fetch rule - and returns one executable opportunity on the
            # fixture whose two sides are rated.
            assert upcoming in kw["events"]
            assert kw["account_health_ok"] is False, \
                "no account-health proof exists, so live must stay refused"
            return cycle([opportunity(event_key=upcoming.key, outcome="Arsenal")])

        agent.betting_engine.run_cycle = fake_cycle
        result = asyncio.run(agent._run_sports_lane("paper"))

        assert result["bets_placed"] == 1
        assert result["opportunities"] == 1
        assert result["ratings"]["rated_teams"] >= 2, \
            "both sides of the fixture must carry a rating for the model to price it"
        open_bets = agent.sports_book.open_bets()
        assert len(open_bets) == 1
        bet = open_bets[0]
        assert bet["execution_mode"] == "paper"
        assert bet["status"] == "open"
        # The stake left the paper purse at placement...
        assert float(agent.storage.get_paper_bankroll()) == pytest.approx(47.0)
        # ...and the trade went into the same learning record as any other trade:
        # the agent's REAL tracker, read back by venue, is what performance and
        # allocation are computed from.
        assert agent.sports_book.tracker is not None
        recorded = [o for o in agent.sports_book.tracker.outcomes
                    if getattr(o, "trade_id", "") == bet["bet_id"]]
        assert recorded, ("a placed bet must enter TradeOutcomeTracker, or the "
                          "sports lane learns nothing")
    finally:
        agent.storage.close()


def test_the_agents_lane_settles_on_a_final_score_and_returns_the_purse(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    from ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    try:
        upcoming = FakeEvent(key_override="epl:upcoming", raw={})
        results = [finished(2, 1, key_override=f"epl:m-{i}") for i in range(3)]
        agent.betting_engine.data = _StubSportsData(results + [upcoming])

        async def fake_cycle(**kw):
            return cycle([opportunity(event_key=upcoming.key, outcome="Arsenal",
                                      price=2.10, stake=3.0)])

        agent.betting_engine.run_cycle = fake_cycle
        asyncio.run(agent._run_sports_lane("paper"))
        assert float(agent.storage.get_paper_bankroll()) == pytest.approx(47.0)

        # Next cycle: the same fixture comes back FINISHED, 2-1 to Arsenal, which
        # is what was backed.
        agent._sports_fixtures = [finished(2, 1, key_override="epl:upcoming")]
        settlement = asyncio.run(agent.sports_book.settle(
            agent._sports_events_for_settlement()))
        assert settlement.get("settled") == 1
        agent._credit_sports_settlements(settlement)

        # Stake back ($3) plus the profit ($3 x (2.10-1) = $3.30) = $53.30.
        assert float(agent.storage.get_paper_bankroll()) == pytest.approx(53.30)
        assert agent.sports_book.open_bets() == []
        settled = (settlement.get("bets") or [])[0]
        assert settled["status"] == "won"
        assert float(settled["pnl"]) == pytest.approx(3.30)
        by_venue = agent.sports_book.tracker.get_venue_performance()
        assert by_venue, "a settled bet must teach the learner, like any trade"
    finally:
        agent.storage.close()


def test_a_quiet_feed_still_gets_a_reason_back(tmp_path, monkeypatch):
    """The operator's button must be able to explain a quiet cycle."""
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    from ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    try:
        agent.betting_engine.data = _StubSportsData([])

        async def fake_cycle(**kw):
            return {"ok": False, "events": 0, "opportunities": 0, "executable": 0,
                    "blockers": ["no fixtures from any feed: ['espn: 0 events " \
                                 "(ConnectError)']"]}

        agent.betting_engine.run_cycle = fake_cycle
        result = asyncio.run(agent._run_sports_lane("paper"))
        assert result["bets_placed"] == 0
        assert result["blockers"], "a quiet cycle must say WHY it was quiet"
        assert result["how_a_bet_works"], \
            "the block must carry what a bet would have to clear"
        # A feed that returned nothing must never be mistaken for a bet.
        assert agent.sports_book.open_bets() == []
        assert float(agent.storage.get_paper_bankroll()) == pytest.approx(50.0)
    finally:
        agent.storage.close()
