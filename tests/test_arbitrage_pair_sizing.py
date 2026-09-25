"""
Arbitrage: the pair is sized on what happened, not on what was requested.

Every bug in this area has the same shape - a number describing the PLAN being
used where a number describing the RESULT was needed:

  * leg B sized on `amount_per_leg` while leg A filled a fraction of it, so the
    "balanced pair" held a one-sided position,
  * the hedge sized on `amount_per_leg` rather than the unmatched gap, so closing
    an exposure bought MORE of the opposite side than the exposure it closed,
  * leg B admitted at a price cap computed from a spread measured before leg A
    was sent, so a moved book turned a riskless pair into a certain loss,
  * the docstring calling the whole thing "atomic" when no cross-venue primitive
    exists, which is what stopped anyone looking for the exposure window.

The tests below drive a real `MultiVenueExecutor` against a stub venue that
reports exactly what the test tells it to.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.ptai.execution.multi_venue_executor import (
    ExecutionResult,
    MultiVenueExecutor,
)
from src.ptai.markets.base import Market, MarketSource, Token
from src.ptai.venues.registry import VenueRegistry


def _market(market_id: str, price: float) -> Market:
    return Market(
        id=market_id, source=MarketSource.POLYMARKET,
        question=f"Will {market_id} happen?",
        outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
        tokens=[Token(token_id=f"{market_id}-Y", outcome="YES", price=price)],
        volume=200_000.0, volume_24h=100_000.0, liquidity=50_000.0,
        active=True, closed=False, slug=market_id,
        raw={"venue": "stub_a"},
    )


@dataclass
class _Arb:
    """The shape `execute_arbitrage_pair` reads off an arbitrage opportunity."""
    venue_a: str
    venue_b: str
    market_a: Market
    market_b: Market
    price_a: float
    price_b: float
    spread: float
    fee_adjusted_profit: float
    confidence_same_event: float
    should_trade: bool = True


class _StubVenue:
    """
    A venue that reports exactly the fill it is told to.

    `fills` is consumed per order, so a test can say "leg A fills $1.20 of the
    $3.00 requested, then leg B fills nothing" - the partial-fill case that the
    old sizing turned into a one-sided position.
    """

    def __init__(self, venue_id: str, fills: List[Dict[str, Any]],
                 book: Optional[Dict[str, Any]] = None,
                 books: Optional[List[Dict[str, Any]]] = None):
        self.venue_id = venue_id
        self.fills = list(fills)
        # `books` is consumed per read, so a test can move the book between the
        # pre-trade verification and the recheck that happens after leg A fills.
        # The last book repeats. Each book carries an explicit `spread` because
        # `read_spread` reports a book without one as unmeasured, and the
        # executor correctly refuses to trade on an unmeasured spread.
        self.books = list(books) if books else [book or dict(BOOK_A)]
        self.orders: List[Dict[str, Any]] = []
        self.dry_run = False
        self.can_place_real_orders = True

    # The executor asks the adapter what the order costs before sending it; a
    # stub without these is not a venue, it is a failure mode.
    def calculate_fees(self, market, amount_usd):
        return 0.0

    def estimate_slippage(self, market, amount_usd):
        return 0.0

    async def get_orderbook(self, market: Market) -> Dict[str, Any]:
        if len(self.books) > 1:
            return dict(self.books.pop(0))
        return dict(self.books[0])

    async def place_order(self, opportunity, max_spend_usd, max_price):
        self.orders.append({"market_id": opportunity.market.id,
                            "side": opportunity.side,
                            "max_spend_usd": max_spend_usd,
                            "max_price": max_price})
        if not self.fills:
            return {"status": "rejected", "reason": "no fill scripted"}
        payload = dict(self.fills.pop(0))
        fill_at = payload.pop("fill_all_at", None)
        if payload.pop("fill_all", False) or fill_at is not None:
            # "The venue filled the whole order at the limit it was given."
            # Scripting that in dollars would hardcode a number the sizing
            # decides - and the sizing is what these tests are about.
            payload.setdefault("status", "matched")
            payload.setdefault("orderID", "o-auto")
            payload["filled_usd"] = max_spend_usd
            # `fill_all_at` says what the book actually pays: sized on the CAP
            # it would buy more shares than the other leg holds the moment the
            # fill is better than the limit.
            payload["filled_price"] = fill_at or max_price
        return payload


class _Registry:
    def __init__(self, adapters):
        self.adapters = adapters
        self.get_adapter_for_venue_id = adapters.get


def _executor(venues: Dict[str, _StubVenue], bankroll: float = 50.0):
    registry = _Registry(venues)
    ex = MultiVenueExecutor(registry=registry, bankroll=bankroll)
    ex.rate_limits = {}
    # Rate limiting is one request per venue per second, a real constraint that
    # would make this test sleep. The behaviour under test is the sizing.
    ex.check_rate_limit = lambda venue_id: True
    return ex


def _arb(spread: float = 0.05) -> _Arb:
    return _Arb(
        venue_a="stub_a", venue_b="stub_b",
        market_a=_market("A-1", 0.44), market_b=_market("B-1", 0.50),
        price_a=0.44, price_b=0.50, spread=spread,
        fee_adjusted_profit=spread, confidence_same_event=0.9,
    )


# Venue A quotes YES at 0.44 (ask) / 0.43 (bid). Venue B bids 0.50 for YES,
# which is what makes the NO side cost 1 - 0.50 = 0.50 there. The pair costs
# 0.44 + 0.50 = 0.94 for a $1 payoff: a real 6c of spread, and the book that
# makes it has to be quoted consistently, because the recheck recomputes what
# leg B would actually cost rather than trusting the discovered spread.
BOOK_A = {"bids": [{"price": "0.43", "size": "5000"}],
          "asks": [{"price": "0.44", "size": "5000"}],
          "spread": 0.01, "is_real": True, "source": "stub_book"}
BOOK_B = {"bids": [{"price": "0.50", "size": "5000"}],
          "asks": [{"price": "0.52", "size": "5000"}],
          "spread": 0.02, "is_real": True, "source": "stub_book"}


def _book(bid: float, ask: float) -> Dict[str, Any]:
    """A real, measured book: `spread` present and `is_real` True."""
    return {
        "bids": [{"price": f"{bid:.2f}", "size": "5000"}],
        "asks": [{"price": f"{ask:.2f}", "size": "5000"}],
        "spread": round(ask - bid, 4),
        "is_real": True,
        "source": "stub_book",
    }


def _fills(**kwargs):
    """A venue response meaning "filled this much at this price"."""
    payload = {"status": "matched", "orderID": "o-1", "size": 5.0,
               "price": 0.45, "filled_usd": 2.25, "filled_price": 0.45}
    payload.update(kwargs)
    return payload


class TestLegBSizedOnLegAsActualFill:
    def test_a_partial_leg_a_does_not_become_a_full_leg_b(self):
        """
        The bug: B was sent for the requested $3.00 while A bought $1.20, so a
        "riskless pair" held $1.80 of one-sided position.

        The corrected rule is stronger than "send what A filled in dollars": B
        is sent for the SHARES A filled. Here A filled 2.6667 shares for $1.20,
        and B is asked for 2.6667 shares - which at its 0.52 cap costs $1.3867,
        NOT the $1.20 A spent. Equal dollars would have left 0.36 of a share
        unhedged while the pair was reported matched.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [_fills(filled_usd=1.20, size=2.67)]),
            "stub_b": _StubVenue("stub_b", [{"fill_all_at": 0.50}],
                                 book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        b_orders = venues["stub_b"].orders
        assert len(b_orders) == 1
        a_shares = results[0].filled_shares
        assert a_shares == pytest.approx(2.6667, abs=1e-3)
        # The order is sized so that TWO DOLLARS AT THE QUOTED PRICE buy the
        # shares A holds: 2.6667 x 0.50 = $1.3333, against a 0.52 limit. Sizing
        # it on the LIMIT would buy 2.56 shares - fewer than A holds - and the
        # pair would be one-sided while reading as matched.
        assert b_orders[0]["max_spend_usd"] == pytest.approx(a_shares * 0.50,
                                                            abs=1e-4), (
            f"leg B was sent for ${b_orders[0]['max_spend_usd']:.4f} for "
            f"{a_shares:.4f} shares of A at a 0.50 quote")
        assert b_orders[0]["max_spend_usd"] != pytest.approx(1.20), (
            "B was sent for the DOLLARS A spent, which is a different share "
            "count on the other side")
        assert results[1].filled_shares == pytest.approx(a_shares, abs=1e-4), (
            "the two legs did not end up holding the same number of shares")
        assert len(results) == 2
        assert "PAIR MATCHED" in results[0].reasoning

    def test_a_leg_a_that_never_filled_does_not_send_leg_b(self):
        """
        A resting order that bought nothing is not a position. Sending B anyway
        would place a naked bet the arb had nothing to do with.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                {"status": "resting", "orderID": "o-1", "size": 0.0,
                 "price": 0.45, "filled_usd": 0.0, "resting_usd": 3.0}]),
            "stub_b": _StubVenue("stub_b", [_fills()], book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        assert venues["stub_b"].orders == [], (
            "a second leg was sent although the first bought nothing")
        assert len(results) == 1
        assert "committed no capital" in results[0].reasoning


class TestTheHedgeClosesTheRealGap:
    def test_the_hedge_is_the_unmatched_remainder_not_the_request(self):
        """
        A fills everything it was asked for, B fills part of it. The hedge must
        close the SHARES left over, not the request and not the dollars.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                _fills(fill_all=True),                      # leg A
                _fills(fill_all=True),                      # the hedge
            ]),
            # Leg B takes 2 shares where A holds ~6.1: the remainder is big
            # enough to hedge under the venue's $1 minimum order.
            "stub_b": _StubVenue("stub_b", [
                _fills(filled_usd=1.00, filled_price=0.50),
            ], book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        a_shares = results[0].filled_shares
        b_shares = 1.00 / 0.50        # what the venue reported for leg B
        orders = venues["stub_a"].orders
        assert len(orders) == 2, "the unmatched remainder was not hedged"
        hedge_shares = orders[1]["max_spend_usd"] / orders[1]["max_price"]
        assert hedge_shares == pytest.approx(a_shares - b_shares, abs=1e-3), (
            f"the hedge was sent for {hedge_shares:.4f} shares: it must close "
            f"the {a_shares - b_shares:.4f} unmatched shares of A, not the "
            f"{a_shares:.4f} that were requested")
        assert orders[1]["side"] == "NO", "the hedge must buy the other side"
        assert "HEDGE" in results[0].reasoning

    def test_a_fully_matched_pair_is_not_hedged(self):
        """The control: no gap, no hedge, no extra order."""
        venues = {
            "stub_a": _StubVenue("stub_a", [_fills(fill_all=True)]),
            "stub_b": _StubVenue("stub_b", [{"fill_all_at": 0.50}],
                                 book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        assert len(venues["stub_a"].orders) == 1, "a matched pair was hedged"
        assert len(results) == 2
        assert "PAIR MATCHED" in results[0].reasoning

    def test_a_failed_hedge_says_the_exposure_remains(self):
        """
        The label matters. "HEDGE attempted" reads as though the exposure was
        closed; a hedge that bought nothing must say so.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                _fills(fill_all=True),
                {"status": "rejected", "reason": "no liquidity"},
            ]),
            "stub_b": _StubVenue("stub_b", [{"status": "rejected",
                                             "reason": "closed"}],
                                 book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))
        assert "NAKED EXPOSURE REMAINS" in results[0].reasoning, (
            f"a failed hedge was reported as something better: "
            f"{results[0].reasoning}")


class TestThePriceIsRecheckedBeforeLegB:
    def test_a_moved_book_stops_leg_b_and_hedges_instead(self):
        """
        The spread got PTAI here; by the time A filled it was gone. Completing
        the pair at the old cap would buy the second leg at a price where the
        two legs cost more than the $1 they pay out.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                _fills(fill_all=True),
                _fills(fill_all=True),   # the hedge
            ]),
            # B now asks 0.62 against A's 0.44: 1.06 for a $1 payoff.
            # The book PTAI verified, then the book it would actually face.
            "stub_b": _StubVenue("stub_b", [_fills()],
                                 books=[dict(BOOK_B), _book(0.38, 0.40)]),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        assert venues["stub_b"].orders == [], (
            "leg B was sent into a book where the arb no longer exists")
        assert "RECHECK" in results[1].reasoning
        a_orders = venues["stub_a"].orders
        assert len(a_orders) == 2 and a_orders[1]["side"] == "NO", (
            "A's exposure was left open instead of hedged")

    def test_a_healthier_book_lets_the_pair_complete(self):
        """The other direction, so the recheck is not simply refusing everything."""
        venues = {
            "stub_a": _StubVenue("stub_a", [_fills(fill_all=True)]),
            "stub_b": _StubVenue("stub_b", [{"fill_all_at": 0.50}],
                                 books=[dict(BOOK_B)]),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))
        assert len(results) == 2
        # NO on B costs 1 - 0.50 = 0.50 on the book that is actually there, so
        # the cap is 0.52 - not the 0.53 the discovered spread would have allowed.
        assert venues["stub_b"].orders[0]["max_price"] == pytest.approx(0.52), (
            "the cap must come from the book as it is now, not from the spread "
            "measured before leg A was sent")


class TestTheHedgeBuysSharesNotDollars:
    def test_the_hedge_is_sized_on_the_shares_that_are_open(self):
        """
        The failure the dollar-sized hedge could not survive.

        Leg A bought its shares at the ask; by the time the hedge is taken, the
        market has moved and the opposite side costs more than 1 - that ask.
        Hedging the naked DOLLARS buys fewer shares than the position holds, so
        part of A stays directional while the log line reports the exposure
        closed. The order has to be sized in shares, and the price it is capped
        at has to come from the book it will actually face.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                _fills(fill_all=True),          # leg A, at its cap
                _fills(fill_all=True),          # the hedge
            ],
                # The book the pair was verified against, then the book that is
                # there when the hedge is taken. A bid of 0.30 makes the NO side
                # cost 0.70 - above the 0.58 the discovered price would allow.
                books=[_book(0.43, 0.44), _book(0.30, 0.31)]),
            "stub_b": _StubVenue("stub_b", [
                {"status": "rejected", "reason": "venue closed"},
            ], book=dict(BOOK_B)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        orders = venues["stub_a"].orders
        assert len(orders) == 2 and orders[1]["side"] == "NO", (
            "A was left with a one-sided position instead of being hedged")
        shares = results[0].filled_shares
        assert orders[1]["max_price"] > 0.58 + 1e-9, (
            f"the hedge was capped at {orders[1]['max_price']:.3f}, a price the "
            f"market has left behind: an insurance order that cannot fill "
            f"leaves the exposure it existed to close")
        assert orders[1]["max_spend_usd"] / orders[1]["max_price"] == pytest.approx(
            shares, abs=1e-4), (
            "the hedge must be sized to buy the shares A is holding")
        assert orders[1]["max_spend_usd"] > results[0].filled_usd, (
            "hedging a position whose opposite side has become dearer costs "
            "more dollars than the position did - sending the naked dollars "
            "would have bought fewer shares than A holds")

    def test_a_pair_that_cannot_be_completed_is_hedged_not_half_filled(self):
        """
        The safety net. If the share-matched second leg no longer fits inside
        the reserved pair budget, sending what fits would leave an arbitrary
        one-sided position - neither a pair nor a hedge. Nothing is sent, and A
        is closed instead.

        A venue that fills ABOVE the cap it was given is what makes the budget
        run out here; the executor has to survive a fill it did not expect.
        """
        venues = {
            "stub_a": _StubVenue("stub_a", [
                _fills(filled_usd=3.20, filled_price=0.40),   # 8 shares, over the cap
                _fills(fill_all=True),                        # the hedge
            ]),
            "stub_b": _StubVenue("stub_b", [_fills(fill_all=True)],
                                 # B now asks 0.55 for the NO side: 0.40 + 0.55 is
                                 # still under $1, so the pair is still worth
                                 # having - it just costs more than is left.
                                 book=_book(0.45, 0.46)),
        }
        ex = _executor(venues)
        results = asyncio.run(ex.execute_arbitrage_pair(_arb(), amount_per_leg=3.0))

        assert venues["stub_b"].orders == [], (
            "a partial second leg was sent: that is a directional position "
            "wearing the word 'pair'")
        orders = venues["stub_a"].orders
        assert len(orders) == 2 and orders[1]["side"] == "NO"
        assert orders[1]["max_spend_usd"] / orders[1]["max_price"] == pytest.approx(
            results[0].filled_shares, abs=1e-4)


class TestFilledSharesIsMeasured:
    """
    `filled_shares` is the unit the rest of this file reasons in, so it is worth
    two direct tests rather than only being exercised through the pair.
    """

    def test_the_venues_own_size_wins(self):
        from src.ptai.execution.multi_venue_executor import ExecutionResult
        r = ExecutionResult(venue_id="v", market_id="m", status="filled",
                            amount_usd=3.0, price=0.60, fees_usd=0.0,
                            gas_usd=0.0, latency_ms=1.0, reasoning="",
                            filled_usd=3.0, filled_price=0.60, size_matched=5.0)
        assert r.filled_shares == pytest.approx(5.0)

    def test_shares_are_derived_when_the_venue_only_reported_dollars(self):
        from src.ptai.execution.multi_venue_executor import ExecutionResult
        r = ExecutionResult(venue_id="v", market_id="m", status="filled",
                            amount_usd=3.0, price=0.60, fees_usd=0.0,
                            gas_usd=0.0, latency_ms=1.0, reasoning="",
                            filled_usd=2.25, filled_price=0.45)
        assert r.filled_shares == pytest.approx(5.0), (
            "$2.25 at 0.45 is 5 shares; a hedge needs the count, not the "
            "dollars")
        empty = ExecutionResult(venue_id="v", market_id="m", status="rejected",
                                amount_usd=3.0, price=0.0, fees_usd=0.0,
                                gas_usd=0.0, latency_ms=1.0, reasoning="")
        assert empty.filled_shares == 0.0, (
            "nothing filled is zero shares, not an unknown")


class TestTheExecutorDoesNotClaimAtomicity:
    def test_the_report_says_what_this_actually_is(self):
        """
        The claim was the bug: calling a sequential pair "atomic" is what stopped
        anyone looking for the exposure window between the legs.
        """
        ex = _executor({})
        report = ex.get_report()
        text = report["atomicity"].lower()
        assert "no cross-venue atomic primitive" in text
        assert "sequential" in text
        assert "hedge" in text
        # And the helper's contract is documented where it runs.
        doc = MultiVenueExecutor.execute_arbitrage_pair.__doc__.lower()
        assert "not an atomic transaction" in doc
        assert "unmatched" in doc
        assert "share" in doc, (
            "the pair is sized in shares and the docstring has to say so, or "
            "the next reader will size it in dollars again")
