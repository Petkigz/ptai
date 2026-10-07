"""
PredictIt: the venue's own quotes, and the venue's own resolutions.

The operator asked for the venues to be given enough code to reach Polymarket,
and PredictIt was the last one at distance two - it could read markets and could
not say how any of them ended, so since V68 the executor refused to open a
position there at all (a position that can never be closed holds a slot forever
and never counts).

Two things were wrong beyond the missing settlement read, and both are pinned
here:

  1. the book was a MOCK. `bid = price - 0.02, ask = price + 0.02`, a fabricated
     depth, and `source: "predictit_mock"` in the payload - while the venue
     publishes a real bid and a real ask per contract, and publishes NO size.
     The book is now those two prices, one share a side, and it says so.
  2. the record INVENTED what the venue does not publish: `liquidity=1000` on
     every market, no end date at all, one contract per market (so a field of
     outcomes became a fake two-way market on the first name), and a missing
     price quietly became 0.50. Each of those decided something downstream, so
     each is now either read from the venue or absent and declared absent.

The settlement rule is the interesting one: the venue has no results endpoint,
but a closed contract's own final prices are its outcome. The rule is strict -
every final price the venue published must agree on a decided side - and
anything ambiguous is refused with the numbers that were read.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.venues.predictit_adapter import (
    DECIDED_PRICE_NO,
    DECIDED_PRICE_YES,
    PREDICTIT_ID_PREFIX,
    PUBLISHED_QUANTITY_SHARES,
    PredictItAdapter,
)


# ---------------------------------------------------------------------------
# fixtures: recorded response shapes, no network
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    """Answers from {url fragment: payload} and records every call."""

    def __init__(self, routes: dict = None):
        self.routes = routes or {}
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None, **kwargs):
        self.calls.append(url)
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return _Response(payload)
        return _Response({"error": "not found"}, status_code=404)


def _contract(**overrides) -> dict:
    contract = {
        "id": 33624,
        "name": "Republican",
        "shortName": "Republican",
        "status": "Open",
        "lastTradePrice": 0.67,
        "bestBuyYesCost": 0.68,
        "bestSellYesCost": 0.66,
        "bestBuyNoCost": 0.34,
        "bestSellNoCost": 0.32,
        "lastClosePrice": 0.66,
        "dateEnd": "2026-11-03T00:00:00",
        "displayOrder": 0,
    }
    contract.update(overrides)
    return contract


def _raw_market(**overrides) -> dict:
    raw = {
        "id": 8155,
        "name": "Which party will control the Senate after the 2026 election?",
        "shortName": "Senate control 2026?",
        "url": "https://www.predictit.org/markets/detail/8155",
        "timeStamp": "2026-10-06T09:00:00Z",
        "status": "Open",
        "contracts": [_contract()],
    }
    raw.update(overrides)
    return raw


def _adapter(routes: dict = None) -> PredictItAdapter:
    adapter = PredictItAdapter()
    adapter.session = _Session(routes or {"/marketdata/all/": {"markets": [_raw_market()]}})
    return adapter


def _market_from_adapter(adapter) -> Market:
    markets = asyncio.run(adapter.discover_markets(target_count=10))
    return markets[0]


def _quoted(adapter: PredictItAdapter, contract_id: int = 33624, **quote):
    """A discovered market whose cached quote is the given one.

    Discovery only returns contracts the venue published BOTH sides for, so a
    one-sided or crossed quote is applied to the cache - which is exactly how it
    would reach `get_orderbook` for a market held from an earlier cycle.
    """
    market = _market_from_adapter(adapter)
    cached = adapter._quotes[str(contract_id)]
    cached.update(quote)
    return market


# ---------------------------------------------------------------------------
# 1. what the venue publishes, and what it does not
# ---------------------------------------------------------------------------

class TestTheRecordSaysWhatTheVenuePublishes:
    def test_one_market_per_contract_and_the_contract_is_named(self):
        """
        A market with three contracts is three instruments. The old adapter took
        `contracts[0]` and called the whole market a two-way bet on the first
        name - "Republican" priced as though it were "which party controls the
        Senate".
        """
        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market(contracts=[
            _contract(id=1, name="Republican", bestBuyYesCost=0.51,
                      bestSellYesCost=0.49),
            _contract(id=2, name="Democratic", bestBuyYesCost=0.50,
                      bestSellYesCost=0.48),
            _contract(id=3, name="Independent", bestBuyYesCost=0.03,
                      bestSellYesCost=0.01),
        ])]}})
        markets = asyncio.run(adapter.discover_markets(target_count=10))
        assert len(markets) == 3
        assert [m.id for m in markets] == [
            f"{PREDICTIT_ID_PREFIX}-8155-1", f"{PREDICTIT_ID_PREFIX}-8155-2",
            f"{PREDICTIT_ID_PREFIX}-8155-3"]
        assert all("Republican" in markets[0].question
                   or "Republican" in markets[0].description for _ in [0])
        assert "Democratic" in markets[1].question
        assert "Independent" in markets[2].question
        # The record price is the MID of the venue's own quote, not the last
        # trade - a print from yesterday is not today's market.
        assert [m.outcome_prices[0] for m in markets] == [0.50, 0.49, 0.02]

    def test_a_single_contract_market_keeps_the_market_question(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        assert market.question == ("Which party will control the Senate after the "
                                  "2026 election?")
        assert market.id == "predictit-8155-33624"
        assert market.venue_id == "predictit"
        assert market.source == MarketSource.PREDICTIT

    def test_no_volume_no_liquidity_and_the_record_says_why(self):
        """
        `liquidity=1000` on every market was a number the venue never published,
        and it decided which markets got model time. Both figures are 0.0 now and
        the record carries the reason, which is what the scan reads.
        """
        market = _market_from_adapter(_adapter())
        assert market.volume_24h == 0.0 and market.liquidity == 0.0
        assert market.raw["volume_basis"] == "not_published"
        assert market.volume_is_published is False
        assert market.raw["depth_basis"] == "not_published"
        # ...and every other venue keeps the floors.
        other = Market(id="pm1", source=MarketSource.POLYMARKET, question="q")
        assert other.volume_is_published is True

    def test_the_end_date_is_read_and_na_is_no_date(self):
        """
        The adapter never set `end_date`, so the lane's preference for markets
        that settle soon could not apply to this venue at all.
        """
        market = _market_from_adapter(_adapter())
        assert market.end_date is not None
        assert (market.end_date.year, market.end_date.month) == (2026, 11)
        assert market.end_date.tzinfo is not None

        adapter = _adapter({"/marketdata/all/": {"markets": [
            _raw_market(contracts=[_contract(dateEnd="N/A")])]}})
        assert _market_from_adapter(adapter).end_date is None

    def test_closed_contracts_are_not_discovered(self):
        adapter = _adapter({"/marketdata/all/": {"markets": [
            _raw_market(contracts=[_contract(status="Closed")])]}})
        assert asyncio.run(adapter.discover_markets()) == []
        assert "no open, priced contracts" in adapter.last_error

    def test_a_contract_with_no_published_price_is_not_a_half_market(self):
        """
        A missing price used to become 0.50 - a coin flip, priced as a market.
        The contract is skipped instead.
        """
        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market(contracts=[
            _contract(id=9, lastTradePrice=None, bestBuyYesCost=None,
                      bestSellYesCost=None)])]}})
        assert asyncio.run(adapter.discover_markets()) == []
        assert "with no publishable price" not in adapter.last_error  # wording
        assert "no open, priced contracts" in adapter.last_error

    def test_cents_are_read_as_cents_and_junk_is_refused(self):
        assert PredictItAdapter._price(0.67) == 0.67
        assert PredictItAdapter._price(67) == 0.67
        for junk in (None, "N/A", -1, 0, 1, 0, 150, True, "x"):
            assert PredictItAdapter._price(junk) is None

    def test_the_http_failure_is_stated_rather_than_papered_over(self):
        adapter = _adapter({"/marketdata/all/": RuntimeError("connection reset")})
        assert asyncio.run(adapter.discover_markets()) == []
        assert "connection reset" in adapter.last_error


# ---------------------------------------------------------------------------
# 2. the book is the venue's quote
# ---------------------------------------------------------------------------

class TestTheBookIsTheVenuesOwnQuote:
    def test_the_book_is_the_published_bid_and_ask(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["is_real"] is True and book["validated"] is True
        assert book["ask"] == 0.68 and book["bid"] == 0.66
        assert round(book["spread"], 6) == 0.02
        assert book["source"] == "predictit_published_quote"
        assert book["depth"] is None, (
            "the venue publishes no size, so depth is absent rather than invented")
        assert "ONE share" in book["size_basis"]
        assert book["executable_price_yes"] == 0.68
        assert book["executable_price_no"] == 0.34
        # The old mock spread is gone. The only remaining mention of that word
        # is the flag saying this is NOT one - which is the truth and must stay.
        payload = json.dumps(book).lower()
        assert '"is_mock": false' in payload or "'is_mock': false" in payload
        assert json.dumps(book).lower().replace('"is_mock": false', "") \
            .replace("'is_mock': false", "").count("mock") == 0

    def test_the_ladder_is_one_share_a_side(self):
        """
        `bestBuyYesCost` is documented as the cost to buy a SINGLE Yes share.
        That is the whole ladder: one share, at the venue's price.
        """
        adapter = _adapter()
        book = asyncio.run(adapter.get_orderbook(_market_from_adapter(adapter)))
        assert [level["size"] for level in book["asks"]] == [PUBLISHED_QUANTITY_SHARES]
        assert [level["size"] for level in book["bids"]] == [PUBLISHED_QUANTITY_SHARES]

    def test_a_no_buy_takes_the_no_ladder_not_the_yes_one(self):
        """
        The bug this pins: `PaperBroker` reads the `bids` list for a non-YES
        side, so a NO buy handed the YES-space book filled at the YES BID
        (0.66) - a price nobody would have paid for a 0.34 share, booked as
        the cost of the position.
        """
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        result = asyncio.run(adapter.place_order(_Opp(market, "NO"), 1.0, 0.34))
        assert result["filled_price"] == 0.34
        assert result["paper_fill"]["best_price"] == 0.34

    def test_a_one_sided_quote_is_not_a_price(self):
        adapter = _adapter()
        market = _quoted(adapter, yes_ask=0.60, yes_bid=None, no_ask=None,
                         no_bid=None)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["is_real"] is False and book["validated"] is False
        assert book["source"] == "predictit_no_quote"
        assert "one side" in book["warning"]

    def test_a_crossed_quote_is_refused(self):
        adapter = _adapter()
        market = _quoted(adapter, yes_ask=0.40, yes_bid=0.60)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["validated"] is False
        assert "crossed" in book["warning"]

    def test_the_no_side_is_named_where_it_comes_from_the_equivalence(self):
        adapter = _adapter()
        market = _quoted(adapter, no_ask=None, no_bid=None)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["executable_price_no"] == round(1.0 - 0.66, 6)
        assert "binary equivalence" in book["executable_price_no_source"]

    def test_a_wide_quote_is_not_executable(self):
        adapter = _adapter()
        market = _quoted(adapter, yes_ask=0.80, yes_bid=0.20)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["executable"] is False
        assert "wider than" in book["warning"]

    def test_a_quote_not_in_the_cycle_payload_is_asked_for_once(self):
        """
        A market the scan holds from an earlier cycle is priced with one request
        per contract, not by falling back to a fabricated spread.
        """
        adapter = _adapter({
            "/marketdata/all/": {"markets": [_raw_market()]},
            "/markets/8155": _raw_market(),
        })
        market = _market_from_adapter(adapter)
        adapter._quotes = {}  # nothing cached: the venue must be asked
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["ask"] == 0.68
        assert any("markets/8155" in url for url in adapter.session.calls)
        # ...and the second read is served from the cache, not asked again.
        calls = len(adapter.session.calls)
        asyncio.run(adapter.get_orderbook(market))
        assert len(adapter.session.calls) == calls

    def test_no_contract_id_is_a_refusal(self):
        market = Market(id="predictit-8155", source=MarketSource.PREDICTIT,
                        question="q", venue_id="predictit")
        book = asyncio.run(_adapter().get_orderbook(market))
        assert book["is_real"] is False
        assert "nothing to ask the venue for" in book["warning"]


# ---------------------------------------------------------------------------
# 3. the fill: one share at the venue's price, and it says which
# ---------------------------------------------------------------------------

class _Opp:
    def __init__(self, market, side="YES"):
        self.market = market
        self.side = side


class TestAPaperFillAtTheVenuesOwnPrice:
    def test_the_fill_is_the_quoted_price_for_one_share(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        result = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        assert result["status"] == "paper" and result["simulated"] is True
        assert result["filled_price"] == 0.68
        assert result["filled_usd"] == pytest.approx(0.68, abs=1e-6), (
            "one share at the venue's own quoted price")
        assert result["fees_usd"] == 0.0, (
            "the venue charges nothing to open; its profit fee is charged when a "
            "position is closed early or profits withdrawn")
        assert "size_basis" in result and "ONE share" in result["size_basis"]

    def test_a_no_buy_pays_the_published_no_price(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        result = asyncio.run(adapter.place_order(_Opp(market, "NO"), 1.0, 0.34))
        assert result["filled_price"] == 0.34
        assert result["side"] == "NO"

    def test_a_limit_under_the_ask_does_not_fill(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        result = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.50))
        assert result["status"] == "refused"
        assert "above the limit" in result["reason"]

    def test_a_one_sided_book_produces_a_refusal_not_a_fill(self):
        adapter = _adapter()
        market = _quoted(adapter, yes_bid=None, no_ask=None, no_bid=None)
        result = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        assert result["status"] == "refused" and result["success"] is False
        assert "Refusal, not a fill" in result["message"]

    def test_live_mode_gets_the_refusal_that_names_the_real_reason(self):
        adapter = _adapter()
        market = _market_from_adapter(adapter)
        adapter.dry_run = False
        assert adapter.can_place_real_orders is False
        result = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        assert result["status"] == "refused"
        assert result["venue_id"] == "predictit"
        assert "no order path" in result["message"]
        assert "filled_usd" not in result, (
            "a refusal must not carry a fill: nothing was opened")

    def test_the_declared_fee_is_zero_because_the_venue_charges_nothing_to_open(self):
        adapter = _adapter()
        assert adapter.capabilities.fee_taker_pct == 0.0
        assert adapter.capabilities.real_order_path is False
        assert adapter.capabilities.supports_trading is False

    def test_mechanics_come_from_the_venues_documented_rules_and_say_so(self):
        mech = _adapter().get_mechanics()
        assert mech.tick_size == "0.01"
        assert mech.min_order_size == 1.0
        assert mech.taker_fee_rate == 0.0
        assert mech.is_real is False, (
            "a documented rule is not a per-market read, and the object says so")
        assert "documented_rules" in mech.source
        assert mech.warnings, ("a rule that was not read per-market says so")


# ---------------------------------------------------------------------------
# 4. the settlement read: the venue's own record
# ---------------------------------------------------------------------------

class TestTheVenueResolvesItsOwnContracts:
    def _settle(self, position: dict, market_payload: dict):
        adapter = _adapter({"/markets/8155": market_payload})
        return asyncio.run(adapter.get_settlement(
            f"{PREDICTIT_ID_PREFIX}-8155-{position}"))

    def test_a_closed_contract_at_a_dollar_resolves_yes(self):
        verdict = self._settle(33624, _raw_market(
            status="Closed",
            contracts=[_contract(status="Closed", lastTradePrice=1.0,
                                 lastClosePrice=1.0)]))
        assert verdict["settled"] is True and verdict["outcome"] == 1.0
        assert verdict["is_real"] is True
        assert verdict["source"] == "predictit_closed_contract"
        assert "won" in verdict["reason"]

    def test_a_closed_contract_at_zero_resolves_no(self):
        verdict = self._settle(33624, _raw_market(
            status="Closed",
            contracts=[_contract(status="Closed", lastTradePrice=0.0,
                                 lastClosePrice=0.0)]))
        assert verdict["settled"] is True and verdict["outcome"] == 0.0
        assert "lost" in verdict["reason"]

    def test_an_open_market_is_not_settled(self):
        verdict = self._settle(33624, _raw_market())
        assert verdict["settled"] is False
        assert verdict["source"] == "predictit_open"
        assert verdict["is_real"] is True, "the venue answered; nothing is owed"

    def test_a_closed_contract_with_an_undecided_price_is_refused(self):
        """
        The one thing a settlement read must never do. A contract the venue
        stopped at 0.62 could have gone either way, and calling it would be a
        fabricated outcome in the calibration record forever.
        """
        verdict = self._settle(33624, _raw_market(
            status="Closed",
            contracts=[_contract(status="Closed", lastTradePrice=0.62,
                                 lastClosePrice=0.60)]))
        assert verdict["settled"] is False and verdict["outcome"] is None
        assert verdict["source"] == "predictit_undecided"
        assert "0.62" in verdict["reason"] and "0.60" in verdict["reason"]
        assert "invented" in verdict["reason"]

    def test_the_two_published_prices_must_agree(self):
        """
        `lastClosePrice` and `lastTradePrice` are two different fields. If one
        says decided and the other does not, the venue's record is not
        unambiguous, and this adapter does not pick a winner from it.
        """
        verdict = self._settle(33624, _raw_market(
            status="Closed",
            contracts=[_contract(status="Closed", lastTradePrice=1.0,
                                 lastClosePrice=0.55)]))
        assert verdict["settled"] is False
        assert verdict["source"] == "predictit_undecided"

    def test_the_decided_thresholds_are_edge_of_the_range_not_a_guess(self):
        assert DECIDED_PRICE_YES >= 0.99 and DECIDED_PRICE_NO <= 0.01

    def test_a_contract_the_venue_no_longer_lists_says_so(self):
        verdict = self._settle(99999, _raw_market(
            contracts=[_contract(id=33624)]))
        assert verdict["settled"] is False
        assert verdict["source"] == "predictit_contract_missing"
        assert "does not list contract" in verdict["reason"]

    def test_a_multi_contract_market_with_no_contract_id_is_refused(self):
        adapter = _adapter({"/markets/8155": _raw_market(contracts=[
            _contract(id=1), _contract(id=2)])})
        verdict = asyncio.run(adapter.get_settlement("predictit-8155"))
        assert verdict["settled"] is False
        assert verdict["source"] == "predictit_contract_missing"

    def test_every_refusal_avoids_the_word_unsupported(self):
        """
        `source: "unsupported"` is what a venue with NO settlement read returns,
        and V68 records it in storage as a venue whose positions can never
        close. A refusal that could still resolve must not be filed as that.
        """
        adapter = _adapter()
        for verdict in (
                asyncio.run(adapter.get_settlement("")),
                asyncio.run(adapter.get_settlement("predictit-8155-1")),
        ):
            assert verdict["source"] != "unsupported"

    def test_a_transport_failure_is_not_a_resolution(self):
        adapter = _adapter({"/markets/8155": RuntimeError("reset by peer")})
        verdict = asyncio.run(adapter.get_settlement("predictit-8155-33624"))
        assert verdict["settled"] is False and verdict["is_real"] is False
        assert "reset by peer" in verdict["reason"]


# ---------------------------------------------------------------------------
# 5. the venue is not a funded account
# ---------------------------------------------------------------------------

class TestNoMoneyCanReachItAndItSaysSo:
    def test_no_account_read_is_claimed(self):
        adapter = _adapter()
        assert adapter.capabilities.supports_portfolio is False
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert "no submission path" in portfolio["reason"]

    def test_only_the_us_is_eligible_to_trade_there(self):
        adapter = _adapter()
        assert adapter.check_eligibility("US").value == "eligible"
        for country in ("UG", "GB", "DE"):
            assert adapter.check_eligibility(country).value == (
                "requires_verification")


# ---------------------------------------------------------------------------
# 6. what the settlement read is FOR: the trade closes and the record moves
# ---------------------------------------------------------------------------

class TestTheVenueNowFeedsTheRecord:
    """
    The whole reason the settlement read was built. Before it, V68's gate
    refused every PredictIt opportunity at execution time - a position there
    could never be closed and could never count - so the venue read markets
    forever and contributed nothing to the record the operator is waiting on.
    This walks the real adapter, the real storage and the real settlement
    engine: a paper fill at the venue's quote, then the venue's own resolution.
    """

    def _trade(self, tmp_path, market, result, side="YES"):
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=str(tmp_path / "predictit.db"))
        storage.log_trade({
            "market_id": market.id, "market_question": market.question,
            "side": side, "market_price": result["filled_price"],
            "fair_value": 0.5, "edge": 0.05, "kelly_fraction": 0.02,
            "position_size_usd": result["filled_usd"], "confidence": 0.7,
            "status": "paper", "venue_id": "predictit",
            "execution_mode": "paper",
            "token_price_at_entry": result["filled_price"],
            "fees_usd": result["fees_usd"],
        })
        row = storage.conn.execute(
            "SELECT id FROM trades WHERE market_id = ? ORDER BY id DESC LIMIT 1",
            (market.id,)).fetchone()
        return storage, int(row["id"])

    def test_the_gate_accepts_predictit_now(self):
        from src.ptai.venues.adapter import can_report_settlement

        assert can_report_settlement(_adapter()) is True, (
            "the check the executor gate reads is about the CLASS override, and "
            "the real adapter has one")

    def test_a_paper_fill_then_the_venues_own_resolution(self, tmp_path):
        from src.ptai.execution.settlement import SettlementEngine
        from src.ptai.venues.qualification import paper_record_progress

        closed = _raw_market(status="Closed", contracts=[
            _contract(status="Closed", lastTradePrice=1.0, lastClosePrice=1.0)])
        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market()]},
                            "/markets/8155": closed})
        market = _market_from_adapter(adapter)
        fill = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        assert fill["status"] == "paper" and fill["filled_price"] == 0.68

        storage, trade_id = self._trade(tmp_path, market, fill)
        try:
            before = paper_record_progress(storage, "predictit")
            assert (before["resolved"], before["open_trades"]) == (0, 1)

            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())

            assert report.paper_settled == 1, (
                "the venue answered with its own closed contract")
            assert report.stuck == 0, (
                "a settlement read that works is never filed as a venue that "
                "cannot report one")
            row = storage.conn.execute(
                "SELECT resolved, pnl, execution_mode FROM trades WHERE id = ?",
                (trade_id,)).fetchone()
            assert row["resolved"] == 1 and row["execution_mode"] == "paper"
            # One share bought at 0.68, settled at $1.00 (the venue's own rule):
            # $0.32, and nothing charged to open the position.
            assert round(float(row["pnl"]), 6) == pytest.approx(0.32, abs=1e-6)

            after = paper_record_progress(storage, "predictit")
            assert (after["resolved"], after["open_trades"]) == (1, 0)
            assert after["net_pnl_usd"] == pytest.approx(0.32, abs=1e-6)
        finally:
            storage.close()

    def test_an_open_market_leaves_the_trade_open_and_the_venue_usable(self,
                                                                      tmp_path):
        """
        The distinction the V68 gate turns on: "not settled yet" is not "cannot
        report a settlement". A venue filed as unsupported is refused for the
        rest of the run, so an open market must not put PredictIt in that set.
        """
        from src.ptai.execution.settlement import SettlementEngine

        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market()]},
                            "/markets/8155": _raw_market()})
        market = _market_from_adapter(adapter)
        fill = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        storage, trade_id = self._trade(tmp_path, market, fill)
        try:
            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 0
            assert report.stuck == 0
            assert engine._unsupported_venues == set(), (
                "no venue may be written off because a market is still open")
            row = storage.conn.execute(
                "SELECT resolved FROM trades WHERE id = ?", (trade_id,)).fetchone()
            assert row["resolved"] == 0
        finally:
            storage.close()


class TestAStaleNoteIsRevised:
    """
    `settlement_unsupported_venues` is written to storage and read by the panel,
    which names the open positions that can never close. It was never revised,
    so a venue whose adapter later learned to settle kept being described by a
    note from before it could answer - on this operator's own database, which
    already carries the word `predictit` from V68-era runs.
    """

    def test_a_venue_that_answers_is_no_longer_filed_as_one_that_cannot(
            self, tmp_path):
        from src.ptai.execution.settlement import SettlementEngine
        from src.ptai.storage.db import Storage
        from src.ptai.venues.qualification import paper_slot_report

        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market()]},
                            "/markets/8155": _raw_market(
                                status="Closed",
                                contracts=[_contract(status="Closed",
                                                     lastTradePrice=1.0,
                                                     lastClosePrice=1.0)])})
        market = _market_from_adapter(adapter)
        fill = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        storage, _ = self._trade(tmp_path, market, fill)
        try:
            # The note as an older run left it.
            storage.set_state("settlement_unsupported_venues", '["predictit"]')
            assert paper_slot_report(storage)["stuck_venues"] == [
                {"venue_id": "predictit", "open": 1}]

            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 1

            assert storage.get_state("settlement_unsupported_venues") in ("[]", None)
            # ...and with the position closed the panel has nothing to name.
            slots = paper_slot_report(storage)
            assert slots["stuck_open"] == 0 and slots["open"] == 0
        finally:
            storage.close()

    def test_an_open_market_also_revises_the_note(self, tmp_path):
        from src.ptai.execution.settlement import SettlementEngine
        from src.ptai.storage.db import Storage

        adapter = _adapter({"/marketdata/all/": {"markets": [_raw_market()]},
                            "/markets/8155": _raw_market()})
        market = _market_from_adapter(adapter)
        fill = asyncio.run(adapter.place_order(_Opp(market, "YES"), 1.0, 0.68))
        storage, _ = self._trade(tmp_path, market, fill)
        try:
            storage.set_state("settlement_unsupported_venues", '["predictit"]')
            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            asyncio.run(engine.settle_pending())
            assert storage.get_state("settlement_unsupported_venues") == "[]", (
                "'not settled yet' is not 'cannot report a settlement'")
        finally:
            storage.close()

    @staticmethod
    def _trade(tmp_path, market, result, side="YES"):
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=str(tmp_path / "stale.db"))
        storage.log_trade({
            "market_id": market.id, "market_question": market.question,
            "side": side, "market_price": result["filled_price"],
            "fair_value": 0.5, "edge": 0.05, "kelly_fraction": 0.02,
            "position_size_usd": result["filled_usd"], "confidence": 0.7,
            "status": "paper", "venue_id": "predictit",
            "execution_mode": "paper",
            "token_price_at_entry": result["filled_price"],
            "fees_usd": result["fees_usd"],
        })
        row = storage.conn.execute(
            "SELECT id FROM trades WHERE market_id = ? ORDER BY id DESC LIMIT 1",
            (market.id,)).fetchone()
        return storage, int(row["id"])


class _Registry:
    """The registry shape the settlement engine reads (`adapters` by id)."""

    def __init__(self, adapter):
        self.adapters = {adapter.venue_id: adapter}
