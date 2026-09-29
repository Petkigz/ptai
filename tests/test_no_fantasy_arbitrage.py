"""
A basket is an arbitrage only when the evidence says it is one.

The operator's 2026-09-29 log printed this, from 200 real Polymarket markets:

    COMBINATORIAL ARB FOUND: MECE group elon-musk-of-tweets-...: 10 markets sum
      YES 0.003 | Type buy_all_yes cost $0.003 payout $1.000 profit $0.997
      (33233.3%) adj 33223.3% after fees | Exhaustive True Exclusive True |
      Should trade True

    COMBINATORIAL ARB FOUND: MECE group bitcoin-above-on-september-29-2026: 10
      markets sum YES 4.498 | Type sell_all_yes_buy_all_no cost $5.502 payout
      $9.000 profit $3.498 (63.6%) adj 53.6% after fees | ... Should trade True

Neither is an arbitrage and neither could be traded:

  * The ten Musk-tweet brackets are a field the 200-market scan caught only part
    of, so of course their prices sum to 0.003. A basket missing 99.7% of its
    outcomes is not "cheap", it is incomplete.
  * The ten Bitcoin strikes are not mutually exclusive at all - several win
    together - so "exactly one YES wins, therefore all but one NO wins" is false
    and the $3.50 payout was invented.
  * `is_exhaustive` was true for both because a group held three or more markets,
    and `is_exclusive` was `True  # Assume exclusive if same event_slug`.
  * The cost was read off `best_price`, which is not a price anyone can trade at,
    and the fee was the constant `2% x legs x 0.5`.

These tests pin the replacement: the venue's own mark and complete outcome list,
the executable price per leg, and the venue's own fee - and a refusal that names
which one was missing when any of them is.
"""

from __future__ import annotations

import pytest

from src.ptai.markets.base import Market, MarketSource
from src.ptai.strategy.alpha_engine import AlphaEngine
from src.ptai.strategy.combinatorial import CombinatorialArbitrageEngine

VENUE_FEE = {"rate": 0.0, "source": "polymarket clob market info"}


def market(mid: str, slug: str, price: float, *, neg_risk: bool | None = False,
           declared_outcomes: int | None = None, question: str | None = None,
           liquidity: float = 50_000.0) -> Market:
    """
    One market, with the venue payload the engine reads its evidence from.

    `neg_risk=None` means the payload carries no mark at all, which is the case
    the venue's own CLOB answer exists for.
    """
    event: dict = {"slug": slug}
    market_payload: dict = {}
    if neg_risk is not None:
        event["negRisk"] = neg_risk
        market_payload["negRisk"] = neg_risk
    if declared_outcomes is not None:
        event["markets"] = [{"id": f"{slug}-{i}"} for i in range(declared_outcomes)]
    return Market(
        id=mid, source=MarketSource.POLYMARKET,
        question=question or f"Will candidate {mid} win?",
        outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
        tokens=[], volume=100_000.0, volume_24h=50_000.0, liquidity=liquidity,
        active=True, closed=False, slug=mid, event_slug=slug,
        raw={"venue": "polymarket", "event": event, "market": market_payload},
    )


def book(best_ask: float, best_bid: float, *, real: bool = True) -> dict:
    return {
        "is_real": real, "validated": real,
        "best_ask": best_ask, "best_bid": best_bid,
        "executable_price_yes": best_ask,
        "executable_price_no": round(1.0 - best_bid, 4),
        "ask_size": 5_000.0, "bid_size": 5_000.0,
        "spread": round(best_ask - best_bid, 4),
        "source": "clob_real" if real else "estimated_orderbook",
    }


def fee(rate: float = 0.0):
    return lambda m: {"rate": rate, "source": "test venue rate"}


# ---------------------------------------------------------------------------
# the two baskets from the operator's log
# ---------------------------------------------------------------------------

class TestTheFantasyBasketsAreRefused:
    def test_a_field_the_scan_only_partly_captured_is_not_an_arbitrage(self):
        """
        The Musk-tweet shape: 10 brackets summing 0.003, in a field of 200.
        """
        markets = [market(f"E{i}", "elon-musk-of-tweets", 0.0003, neg_risk=False,
                          declared_outcomes=200, question=f"Will Elon Musk post {i} tweets?")
                   for i in range(10)]

        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(markets)

        assert opps
        opp = opps[0]
        assert opp.sum_yes == pytest.approx(0.003, abs=0.01)
        assert opp.verified is False
        assert opp.should_trade is False
        assert opp.is_exclusive is False and opp.is_exhaustive is False
        # The 33,233% is not reported at all - not as a profit, not as a
        # percentage, not as a payout.
        assert opp.estimated_profit_pct == 0.0
        assert opp.net_profit_usd is None
        assert "33233" not in opp.reasoning
        assert opp.blocked_reason

    def test_ten_bitcoin_strikes_are_not_mutually_exclusive(self):
        """
        The claim "exactly one wins" is false here, so the n-1 NO payout is
        invented. The refusal must be about exclusivity, not about the sum.
        """
        markets = [market(f"B{i}", "bitcoin-above-on-september-29-2026", 0.45,
                          neg_risk=False, declared_outcomes=10,
                          question=f"Will the price of Bitcoin be above ${90 + i}k?")
                   for i in range(10)]

        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(markets)

        opp = opps[0]
        assert opp.sum_yes == pytest.approx(4.5, abs=0.01)
        assert opp.verified is False and opp.should_trade is False
        assert opp.blocked_reason and "mutually exclusive" in opp.blocked_reason
        assert opp.net_profit_usd is None

    def test_a_group_of_three_is_not_treated_as_complete_by_its_size(self):
        """
        `is_exhaustive = ... or len(group_markets) >= 3` was the old rule. Size
        is not evidence of anything.
        """
        markets = [market(f"X{i}", "some-field", 0.01, neg_risk=False,
                          declared_outcomes=500) for i in range(4)]

        opp = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(markets)[0]
        assert opp.is_exhaustive is False


# ---------------------------------------------------------------------------
# each check, one at a time
# ---------------------------------------------------------------------------

class TestWhatMustBeTrue:
    def _basket(self, prices=(0.30, 0.30, 0.20), **kw):
        defaults = dict(neg_risk=True, declared_outcomes=len(prices))
        defaults.update(kw)
        return [market(f"M{i}", "election-winner", p, **defaults)
                for i, p in enumerate(prices)]

    def _books(self, asks=(0.31, 0.31, 0.32), bids=(0.29, 0.29, 0.28)):
        return {f"M{i}": book(a, b) for i, (a, b) in enumerate(zip(asks, bids))}

    def test_the_venue_must_mark_the_basket_exclusive(self, ):
        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(
            self._basket(neg_risk=False), book_lookup=self._books().get,
            fee_rate_lookup=fee())
        assert opps[0].verified is False
        assert "does not mark" in opps[0].blocked_reason

    def test_every_outcome_of_the_event_must_be_present(self):
        """
        A leg missing makes the sum meaningless, and the message must carry both
        numbers so the operator can see the scan, not the market, was incomplete.
        """
        markets = self._basket()
        markets[0].raw["event"]["markets"] = [{"id": f"x-{i}"} for i in range(34)]

        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(
            markets, book_lookup=self._books().get, fee_rate_lookup=fee())

        assert opps[0].verified is False
        assert "3 of the event's 34" in opps[0].blocked_reason
        assert opps[0].outcomes_declared == 34
        assert opps[0].outcomes_seen == 3

    def test_a_sum_outside_the_band_is_a_broken_basket_not_free_money(self):
        markets = self._basket(prices=(0.0001, 0.0001, 0.0001))
        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(
            markets, book_lookup=self._books(asks=(0.0002, 0.0002, 0.0002),
                                             bids=(0.0001, 0.0001, 0.0001)).get,
            fee_rate_lookup=fee())
        assert opps[0].verified is False
        assert "outside the" in opps[0].blocked_reason
        assert "0.75" in opps[0].blocked_reason

    def test_a_basket_with_no_real_book_for_a_leg_is_refused_and_names_the_leg(self):
        books = self._books()
        books["M1"] = book(0.31, 0.29, real=False)

        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(
            self._basket(), book_lookup=books.get, fee_rate_lookup=fee())

        assert opps[0].verified is False
        assert "M1" in opps[0].blocked_reason
        assert "no validated real book" in opps[0].blocked_reason

    def test_without_any_book_lookup_a_basket_cannot_be_called_tradeable(self):
        """
        The old dashboard endpoints call the engine with no book source at all.
        That must produce "not verified", never a profit percentage.
        """
        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(self._basket())
        assert opps[0].verified is False
        assert opps[0].should_trade is False
        assert opps[0].estimated_profit_pct == 0.0

    def test_an_unreadable_fee_rate_verifies_the_structure_but_claims_no_net(self):
        opps = CombinatorialArbitrageEngine().find_combinatorial_arbitrage(
            self._basket(), book_lookup=self._books().get,
            fee_rate_lookup=lambda m: None)
        opp = opps[0]
        assert opp.verified is True
        assert opp.should_trade is False
        assert opp.net_profit_usd is None
        assert "not claimed" in opp.blocked_reason


# ---------------------------------------------------------------------------
# a real basket, priced the way it would actually trade
# ---------------------------------------------------------------------------

class TestAVerifiedBasket:
    def _basket(self, prices=(0.30, 0.30, 0.20)):
        return [market(f"M{i}", "election-winner", p, neg_risk=True,
                       declared_outcomes=3) for i, p in enumerate(prices)]

    def test_it_is_priced_at_the_asks_and_reports_a_net(self):
        books = {"M0": book(0.31, 0.29), "M1": book(0.31, 0.29), "M2": book(0.32, 0.28)}

        opp = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=books.get, fee_rate_lookup=fee())[0]

        assert opp.verified is True and opp.should_trade is True
        assert opp.cost_basis == "executable_asks"
        assert opp.cost == pytest.approx(0.94, abs=0.001)
        assert opp.net_profit_usd == pytest.approx(0.06, abs=0.001)
        assert opp.payout == pytest.approx(1.0)
        # It is still not an order. The engine has no order path at all.
        assert opp.wired_to_execution is False
        assert "places no orders" in opp.reasoning

    def test_the_market_price_can_show_a_gap_the_asks_do_not_have(self):
        """
        The whole point of pricing on the executable side: the venue's mid says
        0.94 (a 6.4% "profit") while the asks cost 1.02 - a loss.
        """
        books = {"M0": book(0.35, 0.25), "M1": book(0.35, 0.25), "M2": book(0.32, 0.28)}

        opp = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=books.get, fee_rate_lookup=fee())[0]

        assert opp.sum_yes == pytest.approx(0.80, abs=0.01)   # what the mid says
        assert opp.cost == pytest.approx(1.02, abs=0.001)     # what it costs
        assert opp.verified is True
        assert opp.should_trade is False
        assert opp.net_profit_usd == pytest.approx(-0.02, abs=0.001)
        assert "no profit at executable prices" in opp.blocked_reason

    def test_a_venue_fee_is_charged_per_leg_on_what_each_leg_pays(self):
        books = {"M0": book(0.30, 0.28), "M1": book(0.30, 0.28), "M2": book(0.30, 0.28)}

        opp = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=books.get, fee_rate_lookup=fee(0.02))[0]

        # cost 0.90, fee 2% of 0.90 = 0.018, payout 1.00 => net 0.082
        assert opp.fee_rate == pytest.approx(0.02)
        assert opp.net_profit_usd == pytest.approx(0.082, abs=0.001)
        assert opp.fee_source == "test venue rate"

    def test_the_over_one_direction_buys_every_no(self):
        markets = [market(f"M{i}", "election-winner", p, neg_risk=True,
                          declared_outcomes=3)
                   for i, p in enumerate((0.45, 0.40, 0.30))]
        books = {"M0": book(0.44, 0.42), "M1": book(0.39, 0.37), "M2": book(0.29, 0.27)}

        opp = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            markets, book_lookup=books.get, fee_rate_lookup=fee())[0]

        assert opp.arbitrage_type == "sell_all_yes_buy_all_no"
        assert opp.payout == pytest.approx(2.0)
        assert opp.cost == pytest.approx(1.94, abs=0.001)
        assert opp.should_trade is True


# ---------------------------------------------------------------------------
# ranking and reporting cannot be led by a refusal
# ---------------------------------------------------------------------------

class TestWhatTheReportLeadsWith:
    def test_scan_reports_verified_and_tradeable_separately(self):
        fantasy = [market(f"E{i}", "elon-musk-of-tweets", 0.0003,
                          declared_outcomes=200) for i in range(10)]
        real = [market(f"R{i}", "election-winner", p, neg_risk=True,
                       declared_outcomes=3)
                for i, p in enumerate((0.30, 0.30, 0.20))]
        books = {"R0": book(0.31, 0.29), "R1": book(0.31, 0.29), "R2": book(0.32, 0.28)}

        results = AlphaEngine(bankroll=50.0).scan_all_alpha(
            fantasy + real, book_lookup=books.get, fee_rate_lookup=fee())

        comb = results["combinatorial"]
        assert comb["total"] == 2
        assert comb["verified"] == 1
        assert comb["tradeable"] == 1
        assert comb["refused"] == 1
        assert comb["refused_reasons"]
        # The headline profit is the VERIFIED one: $0.06 net, 6.38%.
        assert comb["top_profit"] == pytest.approx(0.0638, abs=0.001)
        assert comb["top_profit_usd"] == pytest.approx(0.06, abs=0.001)
        assert comb["research_only"] is True and comb["places_orders"] is False

    def test_the_fantasy_never_outranks_the_real_basket(self):
        fantasy = [market(f"E{i}", "elon-musk-of-tweets", 0.0003,
                          declared_outcomes=200) for i in range(10)]
        real = [market(f"R{i}", "election-winner", p, neg_risk=True,
                       declared_outcomes=3)
                for i, p in enumerate((0.30, 0.30, 0.20))]
        books = {"R0": book(0.31, 0.29), "R1": book(0.31, 0.29), "R2": book(0.32, 0.28)}

        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            fantasy + real, book_lookup=books.get, fee_rate_lookup=fee())

        assert opps[0].verified is True, (
            "a 33,233% refusal must not sort above a real basket")
        assert opps[-1].verified is False

class TestTheVenueAnswerCountsToo:
    """
    `negRisk` may be absent from a payload while the venue still knows the
    answer - the CLOB's own `getClobMarketInfo.neg_risk`, which is what the
    order path relies on. That answer is accepted as evidence, and it is
    recorded as the source. It cannot be invented: an unreadable answer leaves
    the group unverified.
    """

    def _basket(self, *, neg_risk=None):
        return [market(f"M{i}", "election-winner", p, neg_risk=neg_risk,
                       declared_outcomes=3)
                for i, p in enumerate((0.30, 0.30, 0.20))]

    def _books(self):
        return {"M0": book(0.31, 0.29), "M1": book(0.31, 0.29), "M2": book(0.32, 0.28)}

    def test_the_clob_answer_verifies_a_basket_the_payload_did_not_mark(self):
        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=self._books().get,
            fee_rate_lookup=fee(), neg_risk_lookup=lambda m: {"neg_risk": True})

        opp = opps[0]
        assert opp.verified is True and opp.should_trade is True
        assert opp.verification["exclusive_source"] == "the venue's own clob market info"

    def test_a_venue_answer_of_false_refuses(self):
        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=self._books().get,
            fee_rate_lookup=fee(), neg_risk_lookup=lambda m: False)
        assert opps[0].verified is False

    def test_two_venue_sources_that_disagree_are_not_evidence(self):
        """
        Payload says no, CLOB says yes. One of them is stale, and a stale answer
        must not decide a trade.
        """
        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(neg_risk=False), book_lookup=self._books().get,
            fee_rate_lookup=fee(), neg_risk_lookup=lambda m: {"neg_risk": True})
        assert opps[0].verified is False
        assert "disagree" in opps[0].blocked_reason
        assert opps[0].verification["exclusive_conflicts"]

    def test_an_unreadable_answer_leaves_it_unverified_rather_than_assumed(self):
        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=self._books().get,
            fee_rate_lookup=fee(), neg_risk_lookup=lambda m: None)
        assert opps[0].verified is False
        assert "assumption and not a fact" in opps[0].blocked_reason

    def test_a_lookup_that_raises_is_not_a_yes(self):
        def boom(m):
            raise RuntimeError("the venue timed out")

        opps = CombinatorialArbitrageEngine(min_profit_pct=0.01).find_combinatorial_arbitrage(
            self._basket(), book_lookup=self._books().get,
            fee_rate_lookup=fee(), neg_risk_lookup=boom)
        assert opps[0].verified is False
