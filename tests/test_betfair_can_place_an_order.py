"""
Betfair: the third venue that can place an order.

The adapter could read real exchange prices and refuse to execute, and it said
why: placing a real order needs the side, the price, the size, persistence and a
liability check. This suite pins the pieces that now exist, and - more
importantly - the things that must still be refused.

Everything here runs against a fake client that records the instructions it is
given, so the assertions are about what PTAI WOULD send to the exchange, not
about what a mock returned.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.venues.adapter import VenueOpportunity, VenueType
from src.ptai.venues.betfair_exchange import (
    BETFAIR_COMMISSION_RATE,
    BETFAIR_MIN_PRICE,
    BETFAIR_PRICE_LADDER,
    BetfairExchangeAdapter,
    snap_odds,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _market(**raw_overrides) -> Market:
    raw = {
        "venue": "betfair",
        "market_id": "1.234567",
        "market_type": "MATCH_ODDS",
        "catalogue_key": "h2h",
        "selection_ids": [111, 222],
        "runner_names": ["Arsenal", "Chelsea"],
        "back_prices": [2.5, 2.6],
        "lay_prices": [2.52, 2.62],
        "in_play": False,
    }
    raw.update(raw_overrides)
    return Market(
        id="betfair-1.234567",
        source=MarketSource.POLYMARKET,
        question="Arsenal v Chelsea - Match Odds",
        outcomes=["Arsenal", "Chelsea"],
        outcome_prices=[0.4, 0.3846],
        volume=50_000.0,
        liquidity=1200.0,
        active=True,
        closed=False,
        slug="1.234567",
        market_type="binary",
        raw=raw,
        data_mode=DataMode.LIVE,
        data_source="betfair_exchange_live",
        is_mock=False,
        venue_id="betfair",
    )


def _opportunity(side: str = "YES", market: Optional[Market] = None) -> VenueOpportunity:
    m = market or _market()
    return VenueOpportunity(
        market=m, venue_id="betfair", venue_type=VenueType.OTHER, side=side,
        market_price=0.4, estimated_fair=0.5, raw_edge=0.1, effective_edge=0.08,
    )


def _runner(selection_id: int, name: str, back: List[List[float]],
            lay: List[List[float]], status: str = "ACTIVE") -> Dict[str, Any]:
    return {
        "selectionId": selection_id,
        "name": name,
        "status": status,
        "lastPriceTraded": back[0][0] if back else None,
        "ex": {
            "availableToBack": [{"price": p, "size": s} for p, s in back],
            "availableToLay": [{"price": p, "size": s} for p, s in lay],
        },
    }


class _Funds:
    currency = "GBP"
    available_to_bet_balance = 42.0


class _Account:
    def get_account_funds(self):
        return _Funds()


class _Inner:
    """Stands in for the betfairlightweight APIClient's `.account`."""
    account = _Account()


class _FakeClient:
    """
    A client that records what it is asked to do.

    `configured` is a plain bool because that is the only thing the adapter
    reads from the real one before logging in.
    """

    def __init__(self, book: Optional[Dict[str, Any]] = None,
                 place_report: Optional[Dict[str, Any]] = None,
                 cancel_report: Optional[Dict[str, Any]] = None,
                 login_ok: bool = True,
                 current_orders: Optional[Dict[str, Any]] = None):
        self.configured = True
        self.book = book
        self.place_report = place_report
        self.cancel_report = cancel_report
        self.login_ok = login_ok
        self.current_orders_report = current_orders
        self.instructions: List[List[Dict[str, Any]]] = []
        self.cancels: List[List[Dict[str, Any]]] = []
        self.calls: List[str] = []
        self.last_error = ""
        self._client = _Inner()
        self._logged_in = login_ok

    def login(self) -> bool:
        self.calls.append("login")
        return self.login_ok

    def market_books(self, market_ids) -> Dict[str, Any]:
        self.calls.append("market_books")
        return {str(market_ids[0]): self.book} if self.book else {}

    def place_orders(self, market_id, instructions, customer_ref=""):
        self.calls.append("place_orders")
        self.instructions.append(instructions)
        return self.place_report

    def cancel_orders(self, market_id, instructions, customer_ref=""):
        self.calls.append("cancel_orders")
        self.cancels.append(instructions)
        return self.cancel_report

    def current_orders(self, market_ids=None):
        self.calls.append("current_orders")
        return self.current_orders_report


def _success_report(bet_id: str = "0abc", matched: float = 0.0,
                    price: Optional[float] = None,
                    instr_status: str = "SUCCESS",
                    order_status: str = "EXECUTABLE",
                    error: str = "") -> Dict[str, Any]:
    return {
        "status": "SUCCESS",
        "instructionReports": [{
            "status": instr_status,
            "betId": bet_id,
            "sizeMatched": matched,
            "averagePriceMatched": price,
            "orderStatus": order_status,
            "errorCode": error,
        }],
    }


def _adapter(client: _FakeClient, dry_run: bool = False) -> BetfairExchangeAdapter:
    adapter = BetfairExchangeAdapter(username="u", password="p", app_key="k",
                                     client=client)
    adapter.dry_run = dry_run
    return adapter


# ---------------------------------------------------------------------------
# 1. the exchange's own rules
# ---------------------------------------------------------------------------

class TestTheExchangeRulesAreWrittenDown:
    def test_the_price_ladder_is_the_exchanges_not_one_cent(self):
        """Betfair's increments widen with the odds; an off-ladder price is rejected."""
        assert snap_odds(2.0, "down") == 2.0
        assert snap_odds(2.03, "down") == 2.02
        assert snap_odds(2.03, "up") == 2.04
        # 3.0-4.0 steps by 0.05, so 3.24 floors to 3.20 and ceils to 3.25.
        assert snap_odds(3.24, "down") == 3.20
        assert snap_odds(3.24, "up") == 3.25
        # 6.0-10.0 steps by 0.2
        assert snap_odds(7.1, "up") == 7.2
        # Below the minimum and above the maximum are not orderable at all.
        assert snap_odds(1.0) is None
        assert snap_odds(2000.0) is None
        assert snap_odds(None) is None
        assert BETFAIR_PRICE_LADDER[0][0] == BETFAIR_MIN_PRICE

    def test_the_odds_floor_for_a_probability_cap_never_pays_more(self):
        """
        The cap is a probability, so the odds floor is 1/cap rounded UP. A
        backbet at odds >= the floor pays a price <= the cap: rounding down would
        authorise a bet above the limit.
        """
        cap = 0.865
        floor = snap_odds(1.0 / cap, "up")
        assert floor == 1.16
        assert 1.0 / floor <= cap

    def test_the_fee_is_a_curve_not_a_flat_two_percent(self):
        """
        Commission is charged on net winnings and only on a winning bet. As a
        per-stake number the market's own probability of the charge gives
        c x (1-p), and the worse leg is charged because the side is unknown at
        this point.
        """
        adapter = BetfairExchangeAdapter()
        assert adapter.fee_rate_for_market(_market()) == pytest.approx(
            BETFAIR_COMMISSION_RATE * 0.6, abs=1e-6)
        rich = Market(id="x", source=MarketSource.POLYMARKET, question="q",
                      outcomes=["A", "B"], outcome_prices=[0.9, 0.1], raw={})
        assert adapter.fee_rate_for_market(rich) == pytest.approx(
            BETFAIR_COMMISSION_RATE * 0.9, abs=1e-6)
        # At even money the commission rate is the per-stake cost: 5% x 0.5.
        even = Market(id="y", source=MarketSource.POLYMARKET, question="q",
                      outcomes=["A", "B"], outcome_prices=[0.5, 0.5], raw={})
        assert adapter.fee_rate_for_market(even) == pytest.approx(0.025, abs=1e-6)
        # A market with no readable price reports no rate rather than a guess.
        assert adapter.fee_rate_for_market(
            Market(id="z", source=MarketSource.POLYMARKET, question="q",
                   outcomes=[], outcome_prices=[], raw={})) is None

    def test_the_flat_field_is_the_ceiling_not_the_rate(self):
        adapter = BetfairExchangeAdapter()
        assert adapter.capabilities.fee_taker_pct == BETFAIR_COMMISSION_RATE
        assert adapter.fee_rate_for_market(_market()) < BETFAIR_COMMISSION_RATE

    def test_a_venue_with_no_login_cannot_trade_and_does_not_claim_it(self):
        bare = BetfairExchangeAdapter()
        assert bare.capabilities.supports_trading is False
        assert bare.capabilities.real_order_path is True
        assert bare.capabilities.supports_order_probe is True
        armed = BetfairExchangeAdapter(username="u", password="p", app_key="k")
        assert armed.capabilities.supports_trading is True
        # ...but arming is still TWO conditions: live mode AND the login.
        assert armed.dry_run is True
        assert armed.can_place_real_orders is False


# ---------------------------------------------------------------------------
# 2. a side is a selection
# ---------------------------------------------------------------------------

class TestASideIsASelection:
    def test_yes_is_the_first_published_outcome(self):
        adapter = _adapter(_FakeClient())
        selection, why = adapter._runner_for_side(_market(), "YES")
        assert selection == 111 and why == ""

    def test_no_is_the_other_runner_of_a_two_outcome_market(self):
        adapter = _adapter(_FakeClient())
        selection, why = adapter._runner_for_side(_market(), "NO")
        assert selection == 222 and why == ""

    def test_no_on_a_three_outcome_market_is_refused_not_synthesised(self):
        """
        On a 1X2 market "not the home win" is two selections. A lay would risk
        odds x size, which is not the quantity the executor sizes, so this path
        refuses and says which market and why.
        """
        adapter = _adapter(_FakeClient())
        three_way = _market(selection_ids=[1, 2, 3],
                            runner_names=["Arsenal", "Draw", "Chelsea"])
        selection, why = adapter._runner_for_side(three_way, "NO")
        assert selection is None
        assert "not one selection" in why and "lay" in why

    def test_a_market_without_selection_ids_is_refused(self):
        adapter = _adapter(_FakeClient())
        selection, why = adapter._runner_for_side(_market(selection_ids=[]), "YES")
        assert selection is None and "selection ids" in why


# ---------------------------------------------------------------------------
# 3. the order that is sent
# ---------------------------------------------------------------------------

class TestTheOrderThatIsSent:
    def test_a_crossing_price_takes_the_offer(self):
        """
        The market offers 2.50 and the cap allows 1/0.4 = 2.50, so the order
        takes the offered price and the stake is capped by what is on offer.
        """
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]]),
            _runner(222, "Chelsea", [[2.6, 40.0]], [[2.62, 30.0]]),
        ]}
        client = _FakeClient(book=book, place_report=_success_report(matched=12.0, price=2.5))
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=30.0,
                                              max_price=0.4))
        sent = client.instructions[0][0]
        assert sent["side"] == "BACK"
        assert sent["selectionId"] == 111
        assert sent["orderType"] == "LIMIT"
        assert sent["limitOrder"]["price"] == 2.5
        # 40.0 offered at 2.50 is 100 contracts = $100 of stake, so the $30 cap
        # binds and the whole order crosses.
        assert sent["limitOrder"]["size"] == 30.0
        # LAPSE: the exchange cancels it at the off.
        assert sent["limitOrder"]["persistenceType"] == "LAPSE"
        assert sent["customerOrderRef"].startswith("ptai")
        assert res["status"] == "partial", "12 of 30 matched is a partial fill"
        assert res["success"] is True
        assert res["filled_usd"] == pytest.approx(12.0)
        assert res["resting"] is True
        assert res["remaining_size"] == pytest.approx(18.0)
        assert res["payoff_if_wins"] == pytest.approx(18.0), "12 matched at 2.5 pays 12 x 1.5"

    def test_a_thin_market_produces_a_smaller_bet_not_a_rejected_one(self):
        """$3 is all that is offered at 2.50, so the stake becomes $3."""
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 1.2]], [[2.52, 30.0]]),
            _runner(222, "Chelsea", [[2.6, 40.0]], [[2.62, 30.0]]),
        ]}
        client = _FakeClient(book=book, place_report=_success_report(matched=3.0, price=2.5))
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=25.0,
                                              max_price=0.4))
        assert client.instructions[0][0]["limitOrder"]["size"] == 3.0
        assert res["status"] == "filled"
        assert res["filled"] is True

    def test_a_market_offering_worse_odds_rests_at_our_own_limit(self):
        """
        A BACK at 2.20 is paying a probability of 0.4545 for something the trade
        was authorised at 0.40 - worse than the cap. The order is not taken at
        that price; it rests AT the cap's own odds (2.50) and is simply not
        matched while the market offers less than we are willing to pay.
        """
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.2, 100.0]], [[2.24, 30.0]]),
            _runner(222, "Chelsea", [[2.3, 40.0]], [[2.34, 30.0]]),
        ]}
        client = _FakeClient(book=book, place_report=_success_report())
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=20.0,
                                              max_price=0.4))
        sent = client.instructions[0][0]
        assert sent["limitOrder"]["price"] == 2.5, "never worse than the authorised cap"
        assert sent["limitOrder"]["size"] == 20.0
        assert res["status"] == "submitted"
        assert res["filled"] is False
        assert res["resting"] is True
        assert res["filled_usd"] == 0.0

    def test_the_no_side_bets_the_other_runner(self):
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]]),
            _runner(222, "Chelsea", [[2.6, 40.0]], [[2.62, 30.0]]),
        ]}
        client = _FakeClient(book=book, place_report=_success_report())
        adapter = _adapter(client)
        asyncio.run(adapter.place_order(_opportunity("NO"), max_spend_usd=10.0,
                                        max_price=0.42))
        sent = client.instructions[0][0]
        assert sent["selectionId"] == 222
        assert sent["limitOrder"]["price"] == 2.6

    def test_dry_run_places_nothing(self):
        client = _FakeClient(book={}, place_report=_success_report())
        adapter = _adapter(client, dry_run=True)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.5))
        assert res["status"] == "dry_run"
        assert client.instructions == []
        assert "place_orders" not in client.calls

    def test_without_credentials_nothing_is_signed(self):
        adapter = BetfairExchangeAdapter()
        adapter.dry_run = False
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.5))
        assert res["success"] is False
        assert "credentials not configured" in res["reason"]

    def test_an_in_play_market_is_refused(self):
        client = _FakeClient(book={"status": "OPEN", "runners": []})
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(
            _opportunity(market=_market(in_play=True)), max_spend_usd=10.0,
            max_price=0.5))
        assert res["status"] == "rejected"
        assert "in play or closed" in res["reason"]
        assert client.instructions == []

    def test_a_market_the_exchange_has_closed_is_refused(self):
        book = {"status": "CLOSED", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]])]}
        client = _FakeClient(book=book)
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.4))
        assert res["status"] == "rejected"
        assert "not accepting orders" in res["reason"]

    def test_a_selection_with_no_back_price_is_refused(self):
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [], [[2.52, 30.0]]),
            _runner(222, "Chelsea", [[2.6, 40.0]], [[2.62, 30.0]]),
        ]}
        client = _FakeClient(book=book)
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.5))
        assert res["status"] == "rejected"
        assert "no back price" in res["reason"]
        assert client.instructions == []

    def test_an_exchange_rejection_is_reported_verbatim(self):
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]])]}
        client = _FakeClient(book=book, place_report={
            "status": "FAILURE",
            "instructionReports": [{"status": "FAILURE",
                                    "errorCode": "INSUFFICIENT_FUNDS"}],
        })
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.4))
        assert res["status"] == "rejected"
        assert res["success"] is False
        assert res["error_code"] == "INSUFFICIENT_FUNDS"
        assert "Refusal, not a fill" in res["message"]

    def test_an_unanswered_order_is_not_a_fill(self):
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]])]}
        client = _FakeClient(book=book, place_report=None)
        client.last_error = "connection reset"
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.4))
        assert res["status"] == "error"
        assert res["success"] is False
        assert "Treat it as unplaced" in res["message"]

    def test_a_non_usd_account_says_the_ledger_unit_is_wrong(self):
        book = {"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [[2.5, 40.0]], [[2.52, 30.0]])]}
        client = _FakeClient(book=book, place_report=_success_report(matched=5.0, price=2.5))
        adapter = _adapter(client)
        res = asyncio.run(adapter.place_order(_opportunity(), max_spend_usd=10.0,
                                              max_price=0.4))
        assert res["currency"] == "GBP"
        assert "GBP" in res["message"] and "dollars" in res["message"]


# ---------------------------------------------------------------------------
# 4. the probe: prove it by doing it
# ---------------------------------------------------------------------------

class TestTheProbe:
    def test_dry_run_refuses_without_placing_anything(self):
        client = _FakeClient()
        adapter = _adapter(client, dry_run=True)
        assert asyncio.run(adapter.probe_order_permission(_opportunity())) is False
        assert "dry_run" in adapter.last_order_probe["reason"]
        assert client.instructions == []

    def test_place_then_cancel_must_both_succeed(self):
        client = _FakeClient(
            place_report=_success_report(bet_id="1.999", matched=0.0),
            cancel_report={"status": "SUCCESS",
                           "instructionReports": [{"status": "SUCCESS",
                                                   "betId": "1.999"}]},
        )
        adapter = _adapter(client)
        assert asyncio.run(adapter.probe_order_permission(_opportunity())) is True
        sent = client.instructions[0][0]
        # The shortest price on the ladder: it queues rather than matching.
        assert sent["limitOrder"]["price"] == BETFAIR_MIN_PRICE
        assert sent["limitOrder"]["size"] == 1.0
        assert client.cancels[0][0]["betId"] == "1.999"
        assert adapter.last_order_probe["cancelled"] is True

    def test_an_unconfirmed_cancel_is_not_a_pass(self):
        client = _FakeClient(
            place_report=_success_report(bet_id="1.999"),
            cancel_report={"status": "FAILURE",
                           "instructionReports": [{"status": "FAILURE"}]},
        )
        adapter = _adapter(client)
        assert asyncio.run(adapter.probe_order_permission(_opportunity())) is False
        assert "NOT confirmed" in adapter.last_order_probe["reason"]
        assert "1.999" in adapter.last_order_probe["reason"]

    def test_a_refused_place_never_reaches_the_cancel(self):
        client = _FakeClient(place_report={"status": "FAILURE",
                                           "instructionReports": [
                                               {"status": "FAILURE",
                                                "errorCode": "INSUFFICIENT_FUNDS"}]})
        adapter = _adapter(client)
        assert asyncio.run(adapter.probe_order_permission(_opportunity())) is False
        assert client.cancels == []
        assert "INSUFFICIENT_FUNDS" in adapter.last_order_probe["reason"]

    def test_no_market_means_no_probe(self):
        client = _FakeClient()
        adapter = _adapter(client)
        assert asyncio.run(adapter.probe_order_permission(None)) is False
        assert "no tradeable market" in adapter.last_order_probe["reason"]


# ---------------------------------------------------------------------------
# 5. the book, in the project's price space
# ---------------------------------------------------------------------------

class TestReadingTheBook:
    def _book(self) -> Dict[str, Any]:
        return {"status": "OPEN", "inplay": False, "totalMatched": 12345.0,
                "runners": [
                    _runner(111, "Arsenal", [[2.5, 40.0], [2.6, 100.0]],
                            [[2.52, 30.0], [2.6, 50.0]]),
                    _runner(222, "Chelsea", [[2.6, 20.0]], [[2.7, 25.0]]),
                ]}

    def test_odds_become_probabilities_and_stakes_become_contracts(self):
        client = _FakeClient(book=self._book())
        adapter = _adapter(client)
        book = asyncio.run(adapter.get_orderbook(_market()))
        assert book["is_real"] is True
        # Backing at 2.50 is buying the outcome at 0.40 per $1 contract, and the
        # GBP 40.00 on offer is 100 contracts.
        assert book["asks"][0] == {"price": 0.4, "size": 100.0}
        # Laying at 2.52 is selling it at 0.3968..., 75.6 contracts.
        assert book["bids"][0]["price"] == pytest.approx(0.396825, abs=1e-5)
        assert book["bids"][0]["size"] == pytest.approx(75.6, abs=0.05)
        assert book["spread"] == pytest.approx(0.396825 - 0.4, abs=1e-5)
        assert book["executable"] is True
        # The exchange's own terms are kept alongside, undeformed.
        assert book["back"][0]["price"] == 2.5
        assert book["runners"][0]["selection_id"] == 111
        assert book["runners"][1]["selection_id"] == 222
        assert book["assumed_fields"] == []

    def test_an_unconfigured_venue_publishes_no_spread(self):
        adapter = BetfairExchangeAdapter()
        book = asyncio.run(adapter.get_orderbook(_market()))
        assert book["is_real"] is False
        assert book["bids"] == [] and book["asks"] == []
        assert book["spread"] is None
        assert "credentials not configured" in book["reason"]

    def test_a_missing_market_book_is_not_an_empty_spread(self):
        adapter = _adapter(_FakeClient(book=None))
        book = asyncio.run(adapter.get_orderbook(_market()))
        assert book["is_real"] is False
        assert book["asks"] == [] and book["spread"] is None

    def test_the_book_never_invents_a_level(self):
        """A book with no prices publishes nothing, not a placeholder."""
        client = _FakeClient(book={"status": "OPEN", "inplay": False,
                                   "runners": [_runner(111, "Arsenal", [], [])]})
        adapter = _adapter(client)
        book = asyncio.run(adapter.get_orderbook(_market()))
        assert book["asks"] == [] and book["bids"] == []
        assert book["executable"] is False
        assert book["is_real"] is True, "the exchange DID answer; it is simply empty"


# ---------------------------------------------------------------------------
# 6. what the exchange says won
# ---------------------------------------------------------------------------

class TestSettlement:
    def test_the_first_published_outcome_winning_is_one(self):
        client = _FakeClient(book={"status": "CLOSED", "runners": [
            _runner(111, "Arsenal", [], [], status="WINNER"),
            _runner(222, "Chelsea", [], [], status="LOSER")]})
        adapter = _adapter(client)
        verdict = asyncio.run(adapter.get_settlement("betfair-1.234567"))
        assert verdict["settled"] is True
        assert verdict["outcome"] == 1.0
        assert verdict["winner_selection_id"] == 111
        assert verdict["is_real"] is True

    def test_the_second_outcome_winning_is_zero(self):
        client = _FakeClient(book={"status": "CLOSED", "runners": [
            _runner(111, "Arsenal", [], [], status="LOSER"),
            _runner(222, "Chelsea", [], [], status="WINNER")]})
        adapter = _adapter(client)
        verdict = asyncio.run(adapter.get_settlement("betfair-1.234567"))
        assert verdict["settled"] is True and verdict["outcome"] == 0.0

    def test_a_running_market_settles_nothing(self):
        client = _FakeClient(book={"status": "OPEN", "runners": [
            _runner(111, "Arsenal", [], [], status="ACTIVE")]})
        adapter = _adapter(client)
        verdict = asyncio.run(adapter.get_settlement("betfair-1.234567"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is True
        assert "not closed" in verdict["reason"]

    def test_a_void_market_resolves_nothing(self):
        """No winner means no outcome: a guess here is a permanent wrong fact."""
        client = _FakeClient(book={"status": "CLOSED", "runners": [
            _runner(111, "Arsenal", [], [], status="REMOVED"),
            _runner(222, "Chelsea", [], [], status="REMOVED")]})
        adapter = _adapter(client)
        verdict = asyncio.run(adapter.get_settlement("betfair-1.234567"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is True
        assert "no single winner" in verdict["reason"]

    def test_unconfigured_cannot_report_settlement(self):
        adapter = BetfairExchangeAdapter()
        verdict = asyncio.run(adapter.get_settlement("1.234567"))
        assert verdict["is_real"] is False
        assert verdict["settled"] is False

    def test_an_empty_market_id_is_refused(self):
        adapter = _adapter(_FakeClient())
        verdict = asyncio.run(adapter.get_settlement(""))
        assert verdict["is_real"] is False
        assert "empty" in verdict["reason"]


# ---------------------------------------------------------------------------
# 7. the account read
# ---------------------------------------------------------------------------

class TestTheAccountRead:
    def test_balance_and_open_orders_are_read_together(self):
        client = _FakeClient(current_orders={"currentOrders": [{
            "betId": "1.5", "marketId": "1.234567", "selectionId": 111,
            "side": "BACK", "price": 2.5, "sizeRemaining": 12.0,
            "sizeMatched": 3.0, "status": "EXECUTABLE",
            "placedDate": "2026-10-01T10:00:00Z"}]})
        adapter = _adapter(client)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is True
        assert portfolio["balance"] == 42.0
        assert portfolio["currency"] == "GBP"
        assert portfolio["source"] == "betfair_account_api"
        assert portfolio["open_orders"] == 1
        assert portfolio["orders"][0]["bet_id"] == "1.5"
        # An exchange holds BETS, and the adapter says so instead of inventing a
        # position list it cannot value.
        assert portfolio["positions"] == []
        assert "matched BETS" in portfolio["positions_note"]


# ---------------------------------------------------------------------------
# 8. the outcome order the settlement depends on
# ---------------------------------------------------------------------------

class TestThePublishedOrderIsTheExchanges:
    def test_a_market_whose_first_runner_is_unpriced_is_not_published(self):
        """
        If the exchange's first runner has no two-sided price, publishing the
        second runner as outcome 0 would make a later settlement read the FIRST
        runner's win as a win for the first PUBLISHED outcome.
        """
        from src.ptai.venues.betfair_exchange import BetfairClient, BetfairMarket, BetfairRunner

        client = BetfairClient()
        mf = BetfairMarket(
            market_id="1.1", market_type="MATCH_ODDS", catalogue_key="h2h",
            market_name="A v B", event_id="e1", event_name="A v B",
            commence_time=None, in_play=False,
            runners=[
                BetfairRunner(selection_id=1, name="A", lay_price=1.02, lay_size=5.0),
                BetfairRunner(selection_id=2, name="B", back_price=15.0, back_size=2.0,
                              lay_price=16.0, lay_size=2.0),
            ])
        assert client.to_market(mf) is None
        assert client.skipped_markets.get("first runner unpriced") == 1
        assert client.health()["skipped_markets"]["first runner unpriced"] == 1

    def test_a_market_the_exchange_ordered_normally_is_published_in_order(self):
        from src.ptai.venues.betfair_exchange import BetfairClient, BetfairMarket, BetfairRunner

        client = BetfairClient()
        mf = BetfairMarket(
            market_id="1.2", market_type="MATCH_ODDS", catalogue_key="h2h",
            market_name="A v B", event_id="e2", event_name="A v B",
            commence_time=None, in_play=False,
            runners=[
                BetfairRunner(selection_id=1, name="A", back_price=2.5, back_size=10.0,
                              lay_price=2.52, lay_size=10.0),
                BetfairRunner(selection_id=2, name="B", back_price=2.6, back_size=10.0,
                              lay_price=2.62, lay_size=10.0),
            ])
        market = client.to_market(mf)
        assert market is not None
        assert market.raw["selection_ids"] == [1, 2]
        assert market.raw["runner_names"] == ["A", "B"]
        assert market.outcomes == ["A", "B"]
