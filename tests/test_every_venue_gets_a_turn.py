"""
Every venue gets one turn before any venue takes a second.

The operator's ask was to give the venues enough code to reach Polymarket, and
the venues that came back alive - Manifold, PredictIt, Simmer - share one
property that decides whether their work ever produces a trade: they publish NO
volume and NO depth. The deep-analysis budget is small (8 markets by default) and
the screen's score is built out of exactly those two figures, so a venue that
publishes them scores near zero however good its own book is, and would never be
asked. That is the same defect as "nineteen venues and seventeen say
unavailable", one level down: registered, readable, and never asked.

The fix has two halves, and both are pinned here:

  * `_screen_score` skips the volume and liquidity FLOORS for a venue that
    declares it publishes neither (`volume_is_published`) - otherwise the market
    is refused before the book that could price it is ever read - and scores it
    zero on those terms, so it ranks honestly below a venue that does publish
    them.
  * `_prescan_and_rank` reserves ONE slot per venue with a priceable market
    before any venue is given a second, then fills the rest strictly by score.

The reservation is not a bypass: a reserved row still has to be a market with a
validated two-sided book, inside the spread limit, with an active market - and a
venue with no priceable market gets no turn at all. The tests below drive the
real prescan with stubbed adapters and assert exactly that, including the case
where the budget is too small for the reservation to help (one slot, two
venues), because that is the honest limit of the rule.
"""
from __future__ import annotations

import asyncio

import pytest

from src.ptai.markets.base import Market, MarketSource


# ---------------------------------------------------------------------------
# fixtures: markets and books, no network
# ---------------------------------------------------------------------------

def _market(market_id: str, venue_id: str, *, volume=20000.0, liquidity=20000.0,
            publishes: bool = True, end_date=None) -> Market:
    """
    A priceable market in the shape its venue really publishes.

    `publishes=False` is the PredictIt/Manifold/Simmer shape: no volume, no
    depth, and `volume_basis: not_published` in raw - which is what tells the
    screen the floors do not apply rather than that nobody trades here.
    """
    raw = {"venue": venue_id}
    if not publishes:
        raw["volume_basis"] = "not_published"
        raw["liquidity_basis"] = "not_published"
    return Market(id=market_id, source=MarketSource.POLYMARKET,
                  question=f"Will {market_id} happen?", venue_id=venue_id,
                  volume_24h=volume if publishes else 0.0,
                  liquidity=liquidity if publishes else 0.0,
                  end_date=end_date, raw=raw)


def _book(spread: float, *, validated: bool = True, depth=None) -> dict:
    """A book in the shape the venue adapters publish."""
    return {
        "validated": validated,
        "is_real": validated,
        "spread": spread,
        "depth": depth,
        "source": "test_book",
        "validation": {"identity": "the market this record names"},
    }


class _FakeAdapter:
    def __init__(self, books):
        self.books = books

    async def get_orderbook(self, market):
        return self.books.get(market.id, {"validated": False,
                                          "validation": {"identity": "no book"}})


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "turns.db"))
    from ptai.agent.v3_loop import TradingAgentV3

    a = TradingAgentV3(country_code="UG", dry_run=True)
    yield a
    a.storage.close()


def _screen(agent, markets, books, limit):
    agent.venue_registry.get_adapter_for_market = lambda m: _FakeAdapter(books)
    agent.deep_analysis_limit = limit
    return asyncio.run(agent._prescan_and_rank(markets))


def _shortlist_ids(screen):
    return [row["market_id"] for row in screen["shortlist"]]


# ---------------------------------------------------------------------------
# 1. the score alone would starve every venue that publishes no volume
# ---------------------------------------------------------------------------

class TestTheScoreAloneWouldStarveThem:
    def test_a_venue_that_publishes_no_volume_is_still_priceable(self, agent):
        """The floors are skipped for it; the book and spread rules are not."""
        rich = _market("RICH-1", "polymarket")
        bare = _market("BARE-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich, bare], books, limit=2)
        assert screen["refused_at_screen"] == 0, (
            "a venue that publishes no volume must not be refused before its "
            "book is read")
        assert set(_shortlist_ids(screen)) == {"RICH-1", "BARE-1"}

    def test_the_bare_market_scores_below_the_rich_one(self, agent):
        """
        It ranks honestly last - which is WHY the reservation exists.

        If this ever stops being true (the bare venue out-scoring a venue with
        20k of volume and depth), the reservation would be solving a problem
        that no longer exists and this file should be revisited.
        """
        rich = _market("RICH-1", "polymarket")
        bare = _market("BARE-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich, bare], books, limit=2)
        scored = {row["market_id"]: row["score"] for row in screen["shortlist"]}
        assert scored["BARE-1"] < scored["RICH-1"]
        assert scored["BARE-1"] > 0, "still priceable, not refused"

    def test_with_one_slot_the_bare_venue_gets_no_turn(self, agent):
        """
        The honest limit of the rule: a budget of one is one market.

        Pinned so nobody claims the reservation guarantees a turn regardless of
        the budget - the deep budget is the operator's setting, and one market
        per cycle is what one means.
        """
        rich = _market("RICH-1", "polymarket")
        bare = _market("BARE-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich, bare], books, limit=1)
        assert _shortlist_ids(screen) == ["RICH-1"]
        assert screen["venues_shortlisted"] == ["polymarket"]

    def test_with_two_slots_both_venues_get_a_turn(self, agent):
        rich = _market("RICH-1", "polymarket")
        bare = _market("BARE-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich, bare], books, limit=2)
        assert screen["one_turn_per_venue"] is True
        assert screen["venues_shortlisted"] == ["polymarket", "predictit"]
        # Score order in the list itself: the best market is still first.
        assert _shortlist_ids(screen) == ["RICH-1", "BARE-1"]

    def test_the_other_venue_turn_comes_before_the_favourite_takes_two(self,
                                                                      agent):
        """
        A venue with the best markets still takes most of the budget - but its
        SECOND market waits until every other venue has had one.
        """
        rich_a = _market("RICH-A", "polymarket")
        rich_b = _market("RICH-B", "polymarket", volume=12000.0, liquidity=12000.0)
        bare = _market("BARE-1", "manifold", publishes=False)
        books = {"RICH-A": _book(0.01, depth=9000.0),
                 "RICH-B": _book(0.02, depth=6000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich_a, rich_b, bare], books, limit=2)
        assert set(_shortlist_ids(screen)) == {"RICH-A", "BARE-1"}
        assert screen["venues_shortlisted"] == ["manifold", "polymarket"]
        # One more slot, and the rich venue's second market takes it.
        screen = _screen(agent, [rich_a, rich_b, bare], books, limit=3)
        assert set(_shortlist_ids(screen)) == {"RICH-A", "RICH-B", "BARE-1"}
        assert _shortlist_ids(screen)[0] == "RICH-A", (
            "the shortlist is still ordered by score, so the reserve does not "
            "promote a market above a better one")


# ---------------------------------------------------------------------------
# 2. the reserved turn is not a bypass
# ---------------------------------------------------------------------------

class TestTheReservedTurnIsNotABypass:
    def test_a_wide_book_is_refused_and_gets_no_turn(self, agent):
        rich = _market("RICH-1", "polymarket")
        wide = _market("WIDE-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "WIDE-1": _book(0.20)}
        screen = _screen(agent, [rich, wide], books, limit=2)
        assert _shortlist_ids(screen) == ["RICH-1"]
        assert screen["venues_shortlisted"] == ["polymarket"]
        assert screen["refused_at_screen"] == 1

    def test_no_validated_book_is_no_turn(self, agent):
        rich = _market("RICH-1", "polymarket")
        unbooked = _market("NOBOOK-1", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "NOBOOK-1": _book(0.02, validated=False)}
        screen = _screen(agent, [rich, unbooked], books, limit=3)
        assert _shortlist_ids(screen) == ["RICH-1"]
        assert "predictit" not in screen["venues_shortlisted"]

    def test_a_venue_gets_one_reserved_turn_not_one_per_market(self, agent):
        rich = _market("RICH-1", "polymarket")
        bare_a = _market("BARE-A", "predictit", publishes=False)
        bare_b = _market("BARE-B", "predictit", publishes=False)
        bare_c = _market("BARE-C", "predictit", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "BARE-A": _book(0.02), "BARE-B": _book(0.03),
                 "BARE-C": _book(0.04)}
        screen = _screen(agent, [rich, bare_a, bare_b, bare_c], books, limit=3)
        assert len(screen["venues_shortlisted"]) == len(
            set(screen["venues_shortlisted"])), "a venue is listed once"
        assert screen["venues_shortlisted"] == ["polymarket", "predictit"]
        assert _shortlist_ids(screen) == ["RICH-1", "BARE-A", "BARE-B"], (
            "one reserved market from the second venue, then the rest by score")

    def test_the_fill_after_the_reserve_is_strictly_by_score(self, agent):
        rich = _market("RICH-1", "polymarket")
        other = _market("OTHER-1", "kalshi", volume=6000.0, liquidity=6000.0)
        bare = _market("BARE-1", "manifold", publishes=False)
        books = {"RICH-1": _book(0.01, depth=9000.0),
                 "OTHER-1": _book(0.03, depth=3000.0),
                 "BARE-1": _book(0.02)}
        screen = _screen(agent, [rich, other, bare], books, limit=2)
        assert set(_shortlist_ids(screen)) == {"RICH-1", "OTHER-1"}, (
            "with the reserve taken, the remaining slot goes to the better "
            "market, not to the venue that was read last")
