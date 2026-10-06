"""
Manifold: the venue that cannot hold money and can still tell the truth.

Manifold's currency is Mana and Mana cannot be cashed out, so this venue will
never be a place capital sits - the product must never suggest otherwise. What it
can do is price real markets against its own published mechanism and resolve
them for real, which is exactly what a paper trade needs to become evidence.

This suite pins three things:

  1. the fill curve is computed from the venue's own CPMM pool (a derived curve,
     labelled as one) rather than the invented +/-1.5% spread the adapter used to
     return - and the published worked example is pinned to four decimals;
  2. a market whose own numbers disagree with each other publishes NO depth,
     because a curve from contradictory inputs is a plausible-looking guess;
  3. Mana is not money: the order path stays closed, the venue stays unfundable,
     and the account read answers only when there is a key to answer with.

No network is touched: the settlement tests use a recorded response shape and the
curve tests use the venue's own documented numbers.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src.ptai.markets.base import DataMode, Market, MarketSource
from src.ptai.venues import credentials as credential_store
from src.ptai.venues.manifold_adapter import (
    AMM_LADDER_STEPS,
    CPMM_WEIGHT_TOLERANCE,
    ManifoldAdapter,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _raw_market(**overrides) -> dict:
    raw = {
        "id": "i95HLfK9N6hu5H7orfNj",
        "question": "Will it rain in Kampala tomorrow?",
        "outcomeType": "BINARY",
        "mechanism": "cpmm-1",
        "probability": 0.5,
        "p": 0.5,
        "pool": {"YES": 100.0, "NO": 100.0},
        "isResolved": False,
        "volume": 1234.0,
        "volume24Hours": 12.0,
        "totalLiquidity": 200.0,
    }
    raw.update(overrides)
    return raw


def _market(raw: dict = None, price: float = 0.5) -> Market:
    raw = raw if raw is not None else _raw_market()
    return Market(
        id=str(raw.get("id")),
        source=MarketSource.PREDICTIT,  # MarketSource has no MANIFOLD member
        question=raw.get("question", ""),
        outcomes=["YES", "NO"],
        outcome_prices=[price, 1.0 - price],
        tokens=[],
        volume=float(raw.get("volume") or 0.0),
        volume_24h=float(raw.get("volume24Hours") or 0.0),
        liquidity=float(raw.get("totalLiquidity") or 0.0),
        active=not raw.get("isResolved", False),
        closed=bool(raw.get("isResolved", False)),
        slug="manifold-market",
        market_type="binary",
        raw={**raw, "venue": "manifold"},
        venue_id="manifold",
        venue_type="prediction",
        data_mode=DataMode.LIVE,
        data_source="manifold_api",
        is_mock=False,
    )


def _adapter(api_key: str = None) -> ManifoldAdapter:
    return ManifoldAdapter(api_key=api_key)


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    """A session that answers from a dict of {url: payload} and records calls."""

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


# ---------------------------------------------------------------------------
# 1. the venue's rules, written down
# ---------------------------------------------------------------------------

class TestTheVenueRulesAreWrittenDown:
    def test_the_curve_is_modelled_for_cpmm_binary_markets_only(self):
        adapter = _adapter()
        book = asyncio.run(adapter.get_orderbook(_market(_raw_market(mechanism="dpm-2"))))
        assert book["is_real"] is False
        assert "dpm-2" in book["reason"]
        assert "cpmm-1" in book["reason"]

    def test_mana_is_not_money_and_no_order_path_is_claimed(self):
        adapter = _adapter()
        assert adapter.capabilities.supports_trading is False
        assert adapter.can_place_real_orders is False
        adapter.dry_run = False
        assert adapter.can_place_real_orders is False
        refusal = adapter.real_order_refusal(20.0, 0.4, "YES", "m1")
        assert refusal["status"] == "refused"
        assert "play money" in json.dumps(refusal).lower()
        adapter.dry_run = True
        paper = adapter.real_order_refusal(20.0, 0.4, "YES", "m1")
        assert "Mana" in paper["reason"] or "play money" in paper["reason"]

    def test_the_fee_is_declared_as_zero_not_left_unset(self):
        """`None` means "assume 2%" downstream; Manifold's protocol fee is zero."""
        caps = _adapter().capabilities
        assert caps.fee_taker_pct == 0.0
        assert caps.fee_maker_pct == 0.0

    def test_the_market_feed_needs_no_login_but_the_account_read_does(self):
        caps = _adapter().capabilities
        assert caps.requires_credentials is False
        assert caps.supports_market_discovery is True
        assert caps.supports_portfolio is False
        assert _adapter(api_key="key123").capabilities.supports_portfolio is True


# ---------------------------------------------------------------------------
# 2. the login the product offers
# ---------------------------------------------------------------------------

class TestTheLoginIsOffered:
    def test_there_is_a_manifold_login_form(self, tmp_path):
        tool = credential_store.TOOLS["manifold"]
        assert tool.venue_id == "manifold"
        assert tool.kind == "venue"
        assert [f.name for f in tool.fields] == ["api_key"]
        assert tool.fields[0].required is False
        assert credential_store.TOOL_FOR_VENUE["manifold"] == "manifold"

    def test_saving_a_key_reaches_the_running_adapter(self, tmp_path):
        saved = credential_store.save("manifold", {"api_key": "mana-key-abc"},
                                     str(tmp_path))
        assert saved["ok"] is True
        adapter = _adapter()
        applied = credential_store._apply_manifold(adapter, str(tmp_path))
        assert "api_key" in applied
        assert adapter.api_key == "mana-key-abc"
        assert adapter.session.headers["Authorization"] == "Key mana-key-abc"
        assert adapter.capabilities.supports_portfolio is True

    def test_saving_the_same_key_twice_is_a_no_op(self, tmp_path):
        credential_store.save("manifold", {"api_key": "mana-key-abc"}, str(tmp_path))
        adapter = _adapter(api_key="mana-key-abc")
        assert credential_store._apply_manifold(adapter, str(tmp_path)) == []

    def test_no_key_saved_changes_nothing(self, tmp_path):
        adapter = _adapter()
        assert credential_store._apply_manifold(adapter, str(tmp_path)) == []
        assert adapter.capabilities.supports_portfolio is False

    def test_the_agent_registers_the_adapter_with_the_saved_key(self, monkeypatch):
        from src.ptai.agent.v3_loop import TradingAgentV3

        monkeypatch.setattr(credential_store, "resolve",
                            lambda name, data_dir="./data": {"api_key": "wired-key"}
                            if name == "manifold" else {})
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        registered = agent.venue_registry.get_adapter_for_venue_id("manifold")
        assert registered is not None
        assert registered.api_key == "wired-key"
        assert registered.capabilities.supports_portfolio is True


# ---------------------------------------------------------------------------
# 3. the curve IS the venue's mechanism
# ---------------------------------------------------------------------------

class TestTheCurveIsTheVenuesOwn:
    def test_the_published_worked_example_is_reproduced(self):
        """
        Manifold's own documented example: a 100/100 pool at p=0.5, a 10-mana
        YES bet buys 19.09 shares and moves the market to 54.75%.

        https://docs.manifold.markets/ - "How do bets work on CPMM markets?"
        """
        result = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 10.0)
        assert result["shares"] == pytest.approx(19.09, abs=0.005)
        assert result["price_before"] == pytest.approx(0.5, abs=1e-9)
        assert result["price_after"] == pytest.approx(0.5475, abs=1e-4)
        assert result["avg_price"] == pytest.approx(10.0 / 19.09, abs=1e-3)

    def test_no_is_symmetric_for_a_balanced_pool(self):
        yes = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 10.0)
        no = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "NO", 10.0)
        assert no["shares"] == pytest.approx(yes["shares"], abs=1e-9)
        assert no["price_after"] == pytest.approx(1.0 - yes["price_after"], abs=1e-9)

    def test_the_implied_weight_round_trips(self):
        weight = ManifoldAdapter.cpmm_weight_from_pool(300.0, 100.0, 0.75)
        assert weight is not None
        # 300 YES vs 100 NO at p=0.75 resolves to a consistent price.
        shares = ManifoldAdapter.cpmm_shares(300.0, 100.0, weight, "YES", 1.0)
        assert shares["price_before"] == pytest.approx(0.75, abs=1e-9)

    def test_the_weight_is_not_assumed_to_be_one_half(self):
        """A p=0.25 market is a different curve, and the pool says which."""
        wide = ManifoldAdapter.cpmm_weight_from_pool(100.0, 100.0, 0.25)
        assert wide == pytest.approx(0.25, abs=1e-9)

    def test_a_bigger_bet_pays_a_worse_average_price(self):
        small = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 1.0)
        large = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 50.0)
        assert small["avg_price"] < large["avg_price"]
        assert large["price_after"] > small["price_after"]

    def test_the_price_impact_is_the_whole_cost(self):
        """
        Manifold charges no bet fee, so a bettor's cost IS the movement along the
        curve. The average price must sit at or above the touch for a buy.
        """
        result = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "NO", 25.0)
        assert result["avg_price"] >= result["price_before"]

    def test_inputs_that_cannot_be_a_market_return_nothing(self):
        assert ManifoldAdapter.cpmm_shares(0.0, 100.0, 0.5, "YES", 1.0) is None
        assert ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 0.0) is None
        assert ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.0, "YES", 1.0) is None
        assert ManifoldAdapter.cpmm_shares(100.0, 100.0, 1.0, "YES", 1.0) is None
        assert ManifoldAdapter.cpmm_weight_from_pool(100.0, 100.0, 1.0) is None
        assert ManifoldAdapter.cpmm_weight_from_pool("x", 100.0, 0.5) is None


# ---------------------------------------------------------------------------
# 4. the book is the pool, or nothing
# ---------------------------------------------------------------------------

class TestTheBookIsThePoolOrNothing:
    def test_the_book_is_labelled_as_a_curve_not_an_orderbook(self):
        book = asyncio.run(_adapter().get_orderbook(_market()))
        assert book["is_real"] is True
        assert book["book_kind"] == "amm_curve"
        assert book["source"] == "manifold_cpmm_pool"
        assert "orderbook" in book["note"].lower()
        assert book["assumed_fields"] == []

    def test_the_touch_is_the_markets_own_probability(self):
        book = asyncio.run(_adapter().get_orderbook(_market()))
        # Touch = an infinitesimal trade; the first LEVEL is what a 1-mana trade
        # pays on average, so it must be at or above the touch.
        assert book["ask"] == pytest.approx(0.5, abs=1e-9)
        assert book["bid"] == pytest.approx(0.5, abs=1e-9)
        assert book["asks"][0]["price"] > book["ask"]
        assert book["bids"][0]["price"] < book["bid"]
        assert book["spread_kind"] == "amm_price_impact_of_a_1_mana_round_trip"
        assert book["spread"] == pytest.approx(
            (book["asks"][0]["price"] - book["ask"]) + (book["bid"] - book["bids"][0]["price"]),
            abs=1e-6)

    def test_both_sides_are_published_in_yes_token_space(self):
        """
        Everywhere else in PTAI a book is expressed in YES-token prices (a NO fill
        costs 1 - YES bid). Manifold quotes the NO token, so the NO ladder is
        flipped into that space, and buying NO is then priced like any other NO.
        """
        book = asyncio.run(_adapter().get_orderbook(_market()))
        no_first = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "NO", 1.0)
        assert book["bids"][0]["price"] == pytest.approx(1.0 - no_first["avg_price"],
                                                        abs=1e-6)

    def test_the_ladder_walks_the_curve_and_adds_up(self):
        book = asyncio.run(_adapter().get_orderbook(_market()))
        asks = book["asks"]
        assert len(asks) == len(AMM_LADDER_STEPS)
        prices = [level["price"] for level in asks]
        assert prices == sorted(prices), "a buy ladder must not get cheaper"
        assert prices[-1] > prices[0], "size must move the price"
        cumulative = [level["cumulative_cost"] for level in asks]
        assert cumulative == sorted(cumulative)
        assert book["depth"] == pytest.approx(sum(l["size"] for l in asks + book["bids"]),
                                              abs=1e-4)
        assert book["depth_usd"] == pytest.approx(sum(l["cost"] for l in asks + book["bids"]),
                                                  abs=1e-2)
        assert book["executable"] is True
        assert book["pool"] == {"YES": 100.0, "NO": 100.0}

    def test_a_market_whose_pool_contradicts_its_weight_is_refused(self):
        raw = _raw_market(p=0.9)  # pool implies 0.5
        book = asyncio.run(_adapter().get_orderbook(_market(raw)))
        assert book["is_real"] is False
        assert book["source"] == "manifold_pool_inconsistent"
        assert book["asks"] == [] and book["bids"] == []
        assert book["spread"] is None

    def test_a_market_whose_pool_contradicts_its_probability_is_refused(self):
        raw = _raw_market()
        market = _market(raw, price=0.9)  # pool still says 0.5
        book = asyncio.run(_adapter().get_orderbook(market))
        assert book["is_real"] is False
        assert book["asks"] == []

    def test_a_market_with_no_pool_is_refused(self):
        raw = _raw_market()
        raw.pop("pool")
        book = asyncio.run(_adapter().get_orderbook(_market(raw)))
        assert book["is_real"] is False
        assert "pool" in book["reason"]

    def test_an_empty_pool_is_refused(self):
        book = asyncio.run(_adapter().get_orderbook(_market(_raw_market(pool={"YES": 0, "NO": 0}))))
        assert book["is_real"] is False
        assert book["asks"] == []

    def test_a_pool_without_share_counts_is_refused(self):
        book = asyncio.run(_adapter().get_orderbook(_market(_raw_market(pool={}))))
        assert book["is_real"] is False
        assert "YES/NO" in book["reason"] or "usable" in book["reason"]

    def test_an_unpriceable_probability_is_refused(self):
        market = _market(_raw_market(), price=0.0)
        book = asyncio.run(_adapter().get_orderbook(market))
        assert book["is_real"] is False

    @pytest.mark.parametrize("raw_overrides", [
        {"mechanism": "cpmm-multi-1"},
        {"pool": None},
        {"p": 0.2},
    ])
    def test_every_refusal_is_dry(self, raw_overrides):
        book = asyncio.run(_adapter().get_orderbook(_market(_raw_market(**raw_overrides))))
        assert book["is_real"] is False
        assert book["is_mock"] is False
        assert book["bids"] == [] and book["asks"] == []
        assert book["spread"] is None and book["spread_pct"] is None
        assert book["executable"] is False
        assert book["reason"]

    def test_the_tolerance_is_a_guard_not_a_blanket(self):
        raw = _raw_market(p=0.5 + CPMM_WEIGHT_TOLERANCE / 2)
        book = asyncio.run(_adapter().get_orderbook(_market(raw)))
        assert book["is_real"] is True


# ---------------------------------------------------------------------------
# 5. the paper lane fills against that curve
# ---------------------------------------------------------------------------

class TestThePaperLaneFillsAgainstTheCurve:
    def test_the_ladder_is_read_by_the_paper_parsers(self):
        from src.ptai.execution.paper_broker import PaperBroker

        book = asyncio.run(_adapter().get_orderbook(_market()))
        levels = PaperBroker.levels_from_raw(book, "YES")
        assert levels, "a real Manifold curve must survive the paper lane's parser"
        assert levels[0].price == pytest.approx(book["asks"][0]["price"], abs=1e-9)
        assert all(0.0 < level.price < 1.0 for level in levels)
        assert all(level.size > 0 for level in levels)

    def test_a_one_dollar_paper_trade_pays_the_curves_average(self):
        from src.ptai.execution.paper_broker import PaperBroker

        book = asyncio.run(_adapter().get_orderbook(_market()))
        levels = PaperBroker.levels_from_raw(book, "YES")
        expected = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "YES", 1.0)
        assert levels[0].price == pytest.approx(expected["avg_price"], abs=1e-5)
        assert levels[0].size == pytest.approx(expected["shares"], abs=1e-4)

    def test_a_no_fill_is_priced_in_the_yes_book(self):
        from src.ptai.execution.paper_broker import PaperBroker

        book = asyncio.run(_adapter().get_orderbook(_market()))
        bids = PaperBroker.levels_from_raw(book, "NO")
        expected = ManifoldAdapter.cpmm_shares(100.0, 100.0, 0.5, "NO", 1.0)
        assert bids
        # levels_from_raw sorts bids descending; the best NO price is the highest
        # YES-space bid, i.e. the cheapest NO.
        assert max(level.price for level in bids) == pytest.approx(
            1.0 - expected["avg_price"], abs=1e-5)

    def test_an_expensive_no_is_not_free(self):
        """A NO bought near 0.9 must cost near 0.9, not be reported as a bargain."""
        raw = _raw_market(probability=0.9, p=0.9)
        market = _market(raw, price=0.9)
        book = asyncio.run(_adapter().get_orderbook(market))
        assert book["is_real"] is True
        from src.ptai.execution.paper_broker import PaperBroker

        bids = PaperBroker.levels_from_raw(book, "NO")
        assert max(level.price for level in bids) > 0.7

    def test_the_book_carries_no_fabricated_default_source(self):
        book = asyncio.run(_adapter().get_orderbook(_market()))
        assert book["source"] not in ("assumed_default", "unknown", "unavailable")
        assert book["is_mock"] is False


# ---------------------------------------------------------------------------
# 6. settlement is the venue's record
# ---------------------------------------------------------------------------

class TestSettlementIsTheVenuesRecord:
    def _adapter_with(self, payload, status_code: int = 200) -> ManifoldAdapter:
        adapter = _adapter()
        adapter.session = _Session({"market/": payload} if status_code == 200 else {})
        if status_code != 200:
            adapter.session = _Session({})
            adapter.session.routes["market/"] = {"ignored": True}
            adapter.session.get = lambda url, timeout=None, **k: _Response(payload, status_code)
        return adapter

    def test_yes_is_one(self):
        adapter = self._adapter_with(_raw_market(isResolved=True, resolution="YES",
                                                 resolutionTime=1_700_000_000_000))
        out = asyncio.run(adapter.get_settlement("manifold-i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is True and out["outcome"] == 1.0
        assert out["is_real"] is True and out["source"] == "manifold_api_real"
        assert out["resolution_time"] == 1_700_000_000_000

    def test_no_is_zero(self):
        adapter = self._adapter_with(_raw_market(isResolved=True, resolution="NO"))
        out = asyncio.run(adapter.get_settlement("i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is True and out["outcome"] == 0.0

    def test_a_percentage_payout_is_not_a_zero_or_a_one(self):
        adapter = self._adapter_with(_raw_market(isResolved=True, resolution="MKT",
                                                 resolutionProbability=0.42))
        out = asyncio.run(adapter.get_settlement("i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is False and out["outcome"] is None
        assert "MKT" in out["reason"]
        assert out["resolution_probability"] == 0.42

    def test_a_cancelled_market_settles_nothing(self):
        adapter = self._adapter_with(_raw_market(isResolved=True, resolution="CANCEL"))
        out = asyncio.run(adapter.get_settlement("i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is False and out["outcome"] is None

    def test_an_unresolved_market_settles_nothing(self):
        adapter = self._adapter_with(_raw_market())
        out = asyncio.run(adapter.get_settlement("i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is False and out["is_real"] is True
        assert "not resolved" in out["reason"]

    def test_a_non_binary_market_is_refused_with_its_type(self):
        adapter = self._adapter_with(_raw_market(outcomeType="FREE_RESPONSE"))
        out = asyncio.run(adapter.get_settlement("i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is False
        assert "FREE_RESPONSE" in out["reason"]

    def test_a_missing_market_settles_nothing(self):
        adapter = _adapter()
        adapter.session = _Session({})
        out = asyncio.run(adapter.get_settlement("does-not-exist"))
        assert out["settled"] is False and out["outcome"] is None
        assert "404" in out["reason"]

    def test_a_network_error_settles_nothing(self):
        adapter = _adapter()
        adapter.session = _Session({"market/": ConnectionError("no route to host")})
        out = asyncio.run(adapter.get_settlement("x"))
        assert out["settled"] is False and out["outcome"] is None
        assert "ConnectionError" in out["reason"]

    def test_an_unreadable_body_settles_nothing(self):
        adapter = _adapter()
        adapter.session = _Session({"market/": ValueError("bad json")})
        out = asyncio.run(adapter.get_settlement("x"))
        assert out["settled"] is False
        assert "ValueError" in out["reason"]

    def test_an_empty_id_is_refused(self):
        out = asyncio.run(_adapter().get_settlement(""))
        assert out["settled"] is False and out["is_real"] is False

    def test_the_prefix_is_stripped_before_the_lookup(self):
        adapter = _adapter()
        session = _Session({"market/i95HLfK9N6hu5H7orfNj": _raw_market(
            isResolved=True, resolution="YES")})
        adapter.session = session
        out = asyncio.run(adapter.get_settlement("manifold-i95HLfK9N6hu5H7orfNj"))
        assert out["settled"] is True
        assert session.calls == ["https://api.manifold.markets/v0/market/i95HLfK9N6hu5H7orfNj"]


# ---------------------------------------------------------------------------
# 7. the account read: the venue's answer, or none
# ---------------------------------------------------------------------------

class TestTheAccountRead:
    def test_without_a_key_there_is_no_fabricated_balance(self):
        out = asyncio.run(_adapter().get_portfolio())
        assert out["available"] is False
        assert out["balance"] is None
        assert out["positions"] == [] and out["orders"] == []
        assert "API key" in out["reason"] or "key" in out["reason"]

    def test_with_a_key_the_balance_is_the_venues(self):
        adapter = _adapter(api_key="mana-key")
        adapter.session = _Session({"/me": {"username": "operator", "balance": 1234.5}})
        out = asyncio.run(adapter.get_portfolio())
        assert out["available"] is True
        assert out["balance"] == 1234.5
        assert out["currency"] == "MANA"
        assert out["username"] == "operator"
        assert out["source"] == "manifold_api_real"
        assert out["paper"] is True

    def test_a_rejected_key_is_not_a_balance(self):
        adapter = _adapter(api_key="bad")
        adapter.session = _Session({"/me": {"error": "unauthorized"}})
        adapter.session.get = lambda url, timeout=None, **k: _Response({}, 401)
        out = asyncio.run(adapter.get_portfolio())
        assert out["available"] is False and out["balance"] is None
        assert "401" in out["reason"]

    def test_a_transport_error_is_reported_not_hidden(self):
        adapter = _adapter(api_key="mana-key")
        adapter.session = _Session({"/me": ConnectionError("timed out")})
        out = asyncio.run(adapter.get_portfolio())
        assert out["available"] is False and out["balance"] is None
        assert "ConnectionError" in out["reason"]

    def test_mana_is_labelled_as_mana_and_positions_explained(self):
        adapter = _adapter(api_key="mana-key")
        adapter.session = _Session({"/me": {"balance": 10}})
        out = asyncio.run(adapter.get_portfolio())
        assert out["currency"] == "MANA"
        assert "withdraw" in out["positions_note"]


# ---------------------------------------------------------------------------
# 8. the inventory tells the truth about it
# ---------------------------------------------------------------------------

class TestTheInventoryTellsTheTruth:
    def _inventory(self, tmp_path):
        from src.ptai.venues.inventory import build_inventory
        from src.ptai.venues.registry import VenueRegistry

        registry = VenueRegistry()
        registry.register(ManifoldAdapter())
        return build_inventory(registry, country_code="UG", data_dir=str(tmp_path))

    def test_manifold_is_paper_only_and_never_fundable(self, tmp_path):
        row = self._inventory(tmp_path)["venues"]["manifold"]
        assert row["use"] == "paper_only"
        assert row["real_order_path"] is False
        assert row["reach"]["layers"]["places_real_orders"] is False
        assert row["reach"]["layers"]["reads_markets"] is True
        assert row["reach"]["layers"]["reads_account"] is False
        assert row["fundable_from_here"] is False
        assert "play-money" in row["reach"]["next_step"] or "play money" in row["reach"]["next_step"]

    def test_the_missing_piece_named_is_the_login_not_missing_code(self, tmp_path):
        row = self._inventory(tmp_path)["venues"]["manifold"]
        assert row["login"]["tool"] == "manifold"
        assert "Manifold" in row["reach"]["next_step"]
        assert "login" in row["reach"]["next_step"]

    def test_the_venue_runs_today_as_a_paper_venue(self, tmp_path):
        """The row is honest about the venue, not about a mocked book."""
        row = self._inventory(tmp_path)["venues"]["manifold"]
        assert row["reads_live_markets_now"] is True
        assert row["paper_tradable"] is True
        assert row["reach"]["runs_today"] is True
        assert row["reach"]["distance"] == 2
