"""
Simmer: the first venue in this build that will TAKE an order.

Every other venue PTAI reads either needs money it does not have (an exchange
account, KYC, a funded wallet) or cannot report how a market ended, so since V68
the executor refused to open a position there. Simmer's synthetic venue - its
own $SIM currency, its own AMM, its own resolutions - needs one free API key and
nothing else, and the venue itself tracks the position and publishes the outcome.

So the two things pinned here are the two that make that worth anything:

  1. THE ORDER REALLY GOES OUT. `place_order` on the $SIM venue submits through
     the SDK and reports what the VENUE said filled - not a local simulation that
     guesses at the same numbers. The tests drive a fake SDK, so what is asserted
     is the call (market, side, amount, venue) and the reporting of its answer.
  2. IT IS NOT REAL MONEY, AND IT SAYS SO. Simmer's real venues (Polymarket,
     Kalshi) need a signed wallet; this adapter never signs for a third party,
     so `supports_trading` is False, `real_order_path` is False, live mode gets a
     refusal, and every result names the $SIM currency.

The tests that pin why the venue was listed as unbuilt before this: no client was
written, so it returned no markets and no outcome - which is what every other
"unavailable" row still means.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime

import pytest

from src.ptai.markets.base import Market
from src.ptai.venues import credentials as credential_store
from src.ptai.venues.simmer_adapter import (
    PUBLISHED_QUANTITY_SHARES,
    SIMMER_PAPER_VENUE,
    SimmerAdapter,
)


# ---------------------------------------------------------------------------
# a fake SDK, of the shape simmer-sdk 0.25.10 documents
# ---------------------------------------------------------------------------

class _MarketRow:
    def __init__(self, **fields):
        self.id = fields.pop("id")
        self.question = fields.pop("question", f"Will {self.id} happen?")
        self.status = fields.pop("status", "active")
        self.current_probability = fields.pop("current_probability", 0.5)
        self.outcome = fields.pop("outcome", None)
        self.resolves_at = fields.pop("resolves_at", "2026-11-03T00:00:00Z")
        self.import_source = fields.pop("import_source", "polymarket")
        self.is_sdk_only = fields.pop("is_sdk_only", False)
        self.sim_tradeable = fields.pop("sim_tradeable", True)
        self.best_bid = fields.pop("best_bid", None)
        self.best_ask = fields.pop("best_ask", None)
        self.best_bid_size = fields.pop("best_bid_size", None)
        self.best_ask_size = fields.pop("best_ask_size", None)
        self.resolution_criteria = fields.pop("resolution_criteria", None)
        for key, value in fields.items():
            setattr(self, key, value)


class _TradeResult:
    def __init__(self, **fields):
        self.success = fields.pop("success", True)
        self.trade_id = fields.pop("trade_id", "trade-1")
        self.shares_bought = fields.pop("shares_bought", 19.09)
        self.new_price = fields.pop("new_price", 0.5475)
        self.cost = fields.pop("cost", 10.0)
        self.balance = fields.pop("balance", 990.0)
        self.fill_status = fields.pop("fill_status", "filled")
        self.error = fields.pop("error", None)
        self.warnings = fields.pop("warnings", [])
        for key, value in fields.items():
            setattr(self, key, value)


class _Position:
    def __init__(self, **fields):
        self.market_id = fields.pop("market_id", "m1")
        self.shares_yes = fields.pop("shares_yes", 19.09)
        self.shares_no = fields.pop("shares_no", 0.0)
        self.current_value = fields.pop("current_value", 10.45)
        self.pnl = fields.pop("pnl", 0.45)
        self.status = fields.pop("status", "open")
        self.sim_balance = fields.pop("sim_balance", 990.0)
        self.question = fields.pop("question", "Will m1 happen?")


class _Client:
    """Records every call, answers from what the constructor was given."""

    def __init__(self, markets=None, market_by_id=None, positions=None,
                 trade_result=None, trade_error=None):
        self.markets = markets if markets is not None else []
        self._by_id = market_by_id
        self.positions = positions if positions is not None else []
        self.trade_result = trade_result
        self.trade_error = trade_error
        self.calls = []
        self.trades = []

    def get_markets(self, **kwargs):
        self.calls.append(("get_markets", kwargs))
        return list(self.markets)

    def get_market_by_id(self, market_id):
        self.calls.append(("get_market_by_id", market_id))
        if isinstance(self._by_id, Exception):
            raise self._by_id
        return self._by_id

    def get_positions(self, **kwargs):
        self.calls.append(("get_positions", kwargs))
        return list(self.positions)

    def trade(self, **kwargs):
        self.trades.append(kwargs)
        if self.trade_error is not None:
            raise self.trade_error
        return self.trade_result


class _Sdk:
    """A stand-in for the `simmer_sdk` module."""

    def __init__(self, client):
        self.client = client
        self.constructed = []

    def SimmerClient(self, **kwargs):  # noqa: N802 - the SDK's own name
        self.constructed.append(kwargs)
        return self.client


class _FakeSdkModule:
    def __init__(self, client):
        self.client = client


def _install_sdk(monkeypatch, client):
    """Install a fake `simmer_sdk` and return the record of construction."""
    constructed = []

    class _Module:
        def SimmerClient(self, **kwargs):  # noqa: N802
            constructed.append(kwargs)
            return client

    monkeypatch.setitem(sys.modules, "simmer_sdk", _Module())
    return constructed


def _adapter(monkeypatch, client=None, api_key="sim-key", dry_run=True):
    adapter = SimmerAdapter(api_key=api_key)
    adapter.dry_run = dry_run
    if client is not None:
        _install_sdk(monkeypatch, client)
    return adapter


def _row(market_id="m1", **overrides):
    return _MarketRow(id=market_id, **overrides)


class _Registry:
    """What `build_inventory` needs: `.adapters` by venue id, and a country."""

    country_code = "UG"

    def __init__(self, adapters):
        self.adapters = dict(adapters)


class _Opportunity:
    def __init__(self, market, side="YES"):
        self.market = market
        self.side = side
        self.venue_id = "simmer"


def _discover(adapter, count=10):
    return asyncio.run(adapter.discover_markets(target_count=count))


def _market(adapter, client, market_id="m1", **overrides):
    client.markets = [_row(market_id, **overrides)]
    return _discover(adapter)[0]


# ---------------------------------------------------------------------------
# 1. the venue has a client now, and it says which one
# ---------------------------------------------------------------------------

class TestTheVenueHasAClientNow:
    def test_the_flags_say_what_the_code_does(self):
        caps = SimmerAdapter().capabilities
        assert caps.implementation_status == "live"
        assert caps.supports_market_discovery is True
        assert caps.supports_orderbook is True
        # NO REAL-MONEY PATH: Simmer's live venues need a signed wallet.
        assert caps.supports_trading is False
        assert caps.real_order_path is False
        assert caps.requires_credentials is True
        assert SimmerAdapter().can_place_real_orders is False
        assert SimmerAdapter(api_key="k").can_place_real_orders is False
        # Even in live mode: the adapter never signs for a third party.
        live = SimmerAdapter(api_key="k")
        live.dry_run = False
        assert live.can_place_real_orders is False

    def test_the_account_read_needs_the_key_and_says_so(self):
        assert SimmerAdapter().capabilities.supports_portfolio is False
        assert SimmerAdapter(api_key="k").capabilities.supports_portfolio is True

    def test_with_no_key_nothing_is_read_and_the_reason_names_the_free_key(self,
                                                                        monkeypatch):
        adapter = SimmerAdapter()
        monkeypatch.setitem(sys.modules, "simmer_sdk", None)
        assert _discover(adapter) == []
        assert "simmer.markets/dashboard" in adapter.last_error
        assert "moves no money" in adapter.last_error
        book = asyncio.run(adapter.get_orderbook(
            Market(id="simmer-m1", source="polymarket", question="q",
                   venue_id="simmer")))
        assert book["available"] is False
        assert book["is_real"] is False

    def test_with_no_sdk_installed_the_reason_names_the_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "simmer_sdk", None)
        adapter = SimmerAdapter(api_key="k")
        assert adapter.sdk_available is False
        assert _discover(adapter) == []
        assert "pip install simmer-sdk" in adapter.last_error

    def test_the_client_is_built_on_the_virtual_venue_and_never_a_real_one(
            self, monkeypatch):
        client = _Client(markets=[_row()])
        constructed = _install_sdk(monkeypatch, client)
        adapter = SimmerAdapter(api_key="k")
        _discover(adapter)
        assert constructed == [{"api_key": "k", "venue": SIMMER_PAPER_VENUE,
                                "live": True, "_ignore_env_wallets": True}]
        # No wallet can be handed to this client, even if the operator's
        # environment carries one: PTAI signs for nobody.
        assert constructed[0]["_ignore_env_wallets"] is True
        assert SIMMER_PAPER_VENUE == "sim"
        # And the client is reused, not rebuilt per call.
        _discover(adapter)
        assert len(constructed) == 1


# ---------------------------------------------------------------------------
# 2. the login the product offers
# ---------------------------------------------------------------------------

class TestTheLoginIsOffered:
    def test_there_is_a_simmer_login_form(self):
        tool = credential_store.TOOLS["simmer"]
        assert tool.venue_id == "simmer"
        assert tool.kind == "venue"
        assert [f.name for f in tool.fields] == ["api_key"]
        assert tool.fields[0].env == "SIMMER_API_KEY"
        assert credential_store.TOOL_FOR_VENUE["simmer"] == "simmer"
        # The unlock statement must not imply real money can reach this venue.
        assert "No card, no wallet, no deposit" in tool.unlocks
        assert "not used here" in tool.then

    def test_saving_a_key_reaches_the_running_adapter(self, tmp_path):
        saved = credential_store.save("simmer", {"api_key": "sim-key-abc"},
                                      str(tmp_path))
        assert saved["ok"] is True
        adapter = SimmerAdapter()
        applied = credential_store._apply_simmer(adapter, str(tmp_path))
        assert "api_key" in applied
        assert adapter.api_key == "sim-key-abc"
        assert adapter.capabilities.supports_portfolio is True
        assert adapter.capabilities.requires_credentials is True

    def test_saving_the_same_key_twice_is_a_no_op(self, tmp_path):
        credential_store.save("simmer", {"api_key": "sim-key-abc"}, str(tmp_path))
        adapter = SimmerAdapter(api_key="sim-key-abc")
        assert credential_store._apply_simmer(adapter, str(tmp_path)) == []

    def test_a_cached_client_built_without_the_key_is_dropped(self, tmp_path,
                                                              monkeypatch):
        client = _Client(markets=[_row()])
        adapter = SimmerAdapter()
        _install_sdk(monkeypatch, client)
        assert _discover(adapter) == []          # no key: nothing was built
        adapter._client_obj = object()           # pretend a client was cached
        credential_store.save("simmer", {"api_key": "sim-key-abc"}, str(tmp_path))
        applied = credential_store._apply_simmer(adapter, str(tmp_path))
        assert "client_rebuilt" in applied
        assert adapter._client_obj is None

    def test_the_agent_registers_the_adapter_with_the_saved_key(self, monkeypatch):
        from src.ptai.agent.v3_loop import TradingAgentV3

        monkeypatch.setattr(credential_store, "resolve",
                            lambda name, data_dir="./data": {"api_key": "wired-key"}
                            if name == "simmer" else {})
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        registered = agent.venue_registry.get_adapter_for_venue_id("simmer")
        assert registered is not None
        assert registered.api_key == "wired-key"
        assert registered is not None
        assert registered.capabilities.implementation_status == "live"

    def test_the_refresh_wiring_knows_about_simmer(self):
        import inspect

        source = inspect.getsource(credential_store.refresh_adapters)
        assert '"simmer"' in source


# ---------------------------------------------------------------------------
# 3. discovery: the venue's markets, and only the tradeable ones
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_one_market_per_venue_market_with_the_venues_probability(self,
                                                                    monkeypatch):
        client = _Client(markets=[
            _row("m1", current_probability=0.62),
            _row("m2", current_probability=0.31),
        ])
        adapter = _adapter(monkeypatch, client)
        markets = _discover(adapter)
        assert [m.id for m in markets] == ["simmer-m1", "simmer-m2"]
        assert markets[0].outcome_prices[0] == pytest.approx(0.62)
        assert markets[0].outcome_prices[1] == pytest.approx(0.38)
        assert markets[0].venue_id == "simmer"
        assert markets[0].end_date == datetime.fromisoformat(
            "2026-11-03T00:00:00+00:00")
        assert markets[0].raw["volume_basis"] == "not_published"
        # The scan's volume floor is only applied where a venue publishes the
        # figure; a fabricated 0.0 would exclude every market here.
        assert markets[0].volume_is_published is False

    def test_the_tradeable_filter_is_the_venues_own(self, monkeypatch):
        client = _Client(markets=[_row("m1"), _row("m2", sim_tradeable=False)])
        adapter = _adapter(monkeypatch, client)
        assert [m.id for m in _discover(adapter)] == ["simmer-m1"]

    def test_closed_markets_are_not_candidates(self, monkeypatch):
        client = _Client(markets=[_row("m1"), _row("m2", status="resolved"),
                                  _row("m3", status="closed")])
        adapter = _adapter(monkeypatch, client)
        assert [m.id for m in _discover(adapter)] == ["simmer-m1"]

    def test_a_market_with_no_probability_is_not_a_fifty_fifty(self, monkeypatch):
        client = _Client(markets=[_row("m1"), _row("m2", current_probability=None),
                                  _row("m3", current_probability=0.0),
                                  _row("m4", current_probability=1.0)])
        adapter = _adapter(monkeypatch, client)
        assert [m.id for m in _discover(adapter)] == ["simmer-m1"]

    def test_the_venue_is_asked_for_tradeable_markets_only(self, monkeypatch):
        client = _Client(markets=[_row("m1")])
        adapter = _adapter(monkeypatch, client)
        _discover(adapter)
        call, kwargs = client.calls[0]
        assert call == "get_markets"
        assert kwargs["venue"] == SIMMER_PAPER_VENUE
        assert kwargs["status"] == "active"

    def test_a_venue_read_that_raises_is_a_reason_not_a_crash(self, monkeypatch):
        class _Broken(_Client):
            def get_markets(self, **kwargs):
                raise RuntimeError("connection reset")

        adapter = _adapter(monkeypatch, _Broken())
        assert _discover(adapter) == []
        assert "connection reset" in adapter.last_error


# ---------------------------------------------------------------------------
# 4. the price a $SIM order fills at
# ---------------------------------------------------------------------------

class TestTheBook:
    def test_the_synthetic_venue_quotes_one_price_and_the_book_says_why(
            self, monkeypatch):
        client = _Client(markets=[_row("m1", current_probability=0.42)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.42)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["bid"] == pytest.approx(0.42)
        assert book["ask"] == pytest.approx(0.42)
        assert book["spread"] == 0.0
        assert book["spread_basis"] == "synthetic_venue_amm_price"
        assert "one share a side" in book["size_basis"]
        assert book["executable_price_yes"] == pytest.approx(0.42)
        assert book["executable_price_no"] == pytest.approx(0.58)
        assert book["is_real"] is True
        assert book["is_mock"] is False
        assert book["validated"] is True
        # No depth is invented for a venue that publishes none.
        assert book["depth"] is None
        assert book["asks"][0]["size"] == PUBLISHED_QUANTITY_SHARES

    def test_a_published_top_of_book_is_used_and_is_not_one_price(
            self, monkeypatch):
        client = _Client(markets=[_row("m1", current_probability=0.5)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.5,
                         best_bid=0.47, best_ask=0.51, best_bid_size=120.0,
                         best_ask_size=80.0)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["bid"] == pytest.approx(0.47)
        assert book["ask"] == pytest.approx(0.51)
        assert book["spread"] == pytest.approx(0.04)
        assert book["spread_basis"] == "venue_published_top_of_book"
        assert book["depth"] == pytest.approx(80.0 * 0.49, rel=1e-3)
        assert book["depth_basis"].startswith("the smaller published size")

    def test_a_crossed_quote_is_a_refusal_not_a_price(self, monkeypatch):
        client = _Client(markets=[_row("m1")])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.5,
                         best_bid=0.55, best_ask=0.51)
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is False
        assert book["is_real"] is False
        assert "crossed" in book["reason"]
        assert "bid" not in book

    def test_a_market_with_no_price_at_all_has_no_book(self):
        adapter = SimmerAdapter(api_key="k")
        market = Market(id="simmer-m1", source="polymarket", question="q",
                        venue_id="simmer")
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is False
        assert book["source"] == "simmer_no_quote"
        assert "no price to fill at" in book["reason"]


# ---------------------------------------------------------------------------
# 5. the order goes to the venue, and the venue's answer is what is recorded
# ---------------------------------------------------------------------------

class TestTheOrderGoesToTheVenue:
    def test_a_paper_order_is_submitted_and_the_venues_fill_is_recorded(
            self, monkeypatch):
        client = _Client(
            markets=[_row("m1", current_probability=0.42)],
            trade_result=_TradeResult(shares_bought=23.8095, new_price=0.42,
                                      cost=10.0, balance=990.0))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.42)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        # The venue was called - THIS is the difference from a local simulation.
        assert len(client.trades) == 1
        sent = client.trades[0]
        assert sent["market_id"] == "m1"
        assert sent["side"] == "yes"
        assert sent["amount"] == pytest.approx(10.0)
        assert sent["venue"] == SIMMER_PAPER_VENUE
        # And the numbers recorded are the venue's own.
        assert result["status"] == "paper"
        assert result["is_real"] is False
        assert result["simulated"] is True
        assert result["size"] == pytest.approx(23.8095)
        assert result["filled_price"] == pytest.approx(0.42)
        assert result["filled_usd"] == pytest.approx(10.0)
        assert result["currency"] == "SIM (virtual)"
        assert result["venue_trade_id"] == "trade-1"
        assert "No real money was involved" in result["reason"]

    def test_the_recorder_accepts_that_result_as_a_paper_fill(self, monkeypatch):
        """The executor's own reader must see the venue's numbers, not zeros."""
        from src.ptai.execution.multi_venue_executor import MultiVenueExecutor

        executor = MultiVenueExecutor.__new__(MultiVenueExecutor)
        assert hasattr(executor, "_read_fill")
        raw = {
            "status": "paper",
            "filled_price": 0.42,
            "size": 23.8095,
            "size_matched": 23.8095,
            "filled_usd": 10.0,
        }
        parsed = executor._read_fill(raw, 10.0, 0.45)
        assert parsed["status"] == "dry_run"
        assert parsed["filled_usd"] == pytest.approx(10.0)
        assert parsed["price"] == pytest.approx(0.42)

    def test_a_side_the_venue_does_not_speak_is_refused(self, monkeypatch):
        client = _Client(markets=[_row("m1")], trade_result=_TradeResult())
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1")
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "MAYBE"), max_spend_usd=10.0, max_price=0.5))
        assert result["status"] == "refused"
        assert client.trades == []

    def test_a_price_above_the_limit_is_not_sent(self, monkeypatch):
        client = _Client(markets=[_row("m1", current_probability=0.60)],
                         trade_result=_TradeResult())
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.60)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.55))
        assert result["status"] == "refused"
        assert "above the limit" in result["reason"]
        assert client.trades == []

    def test_a_venue_refusal_opens_nothing(self, monkeypatch):
        client = _Client(
            markets=[_row("m1")],
            trade_result=_TradeResult(success=False, error="insufficient $SIM"))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1")
        result = asyncio.run(adapter.place_order(
            _Opportunity(market), max_spend_usd=10.0, max_price=0.9))
        assert result["status"] == "refused"
        assert "insufficient $SIM" in result["reason"]
        assert "No Simmer position was opened" in result["message"]

    def test_success_without_a_size_is_not_a_fill(self, monkeypatch):
        client = _Client(markets=[_row("m1")],
                         trade_result=_TradeResult(shares_bought=0.0,
                                                  new_price=None))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1")
        result = asyncio.run(adapter.place_order(
            _Opportunity(market), max_spend_usd=10.0, max_price=0.9))
        assert result["status"] == "refused"
        assert "cannot account" in result["reason"] or \
            "not a fill this agent can account for" in result["reason"]

    def test_a_failing_venue_call_is_a_refusal_not_an_error_out(self, monkeypatch):
        client = _Client(markets=[_row("m1")],
                         trade_error=TimeoutError("read timed out"))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1")
        result = asyncio.run(adapter.place_order(
            _Opportunity(market), max_spend_usd=10.0, max_price=0.9))
        assert result["status"] == "refused"
        assert "TimeoutError" in result["reason"]

    def test_live_mode_refuses_and_never_calls_the_venue(self, monkeypatch):
        client = _Client(markets=[_row("m1")], trade_result=_TradeResult())
        adapter = _adapter(monkeypatch, client, dry_run=False)
        market = _market(adapter, client, "m1")
        result = asyncio.run(adapter.place_order(
            _Opportunity(market), max_spend_usd=10.0, max_price=0.9))
        assert result["status"] == "refused"
        assert "will not place REAL-money orders" in result["reason"]
        assert client.trades == [], "no order may leave this adapter in live mode"


# ---------------------------------------------------------------------------
# 6. the outcome: the venue's own field, or a refusal
# ---------------------------------------------------------------------------

class TestTheOutcome:
    def test_a_yes_resolution(self, monkeypatch):
        client = _Client(market_by_id=_row("m1", status="resolved",
                                           outcome=True))
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is True
        assert settlement["outcome"] == 1.0
        assert settlement["is_real"] is True
        assert settlement["source"] == "simmer_market_outcome"

    def test_a_no_resolution(self, monkeypatch):
        client = _Client(market_by_id=_row("m1", status="resolved",
                                           outcome=False))
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is True
        assert settlement["outcome"] == 0.0

    def test_an_open_market_is_not_settled(self, monkeypatch):
        client = _Client(market_by_id=_row("m1", status="active"))
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is False
        assert settlement["outcome"] is None
        assert settlement["source"] == "simmer_open"
        assert settlement["is_real"] is True

    def test_a_resolved_market_with_no_outcome_is_refused(self, monkeypatch):
        """The one thing this read must never do is invent an outcome."""
        client = _Client(market_by_id=_row("m1", status="resolved", outcome=None))
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is False
        assert settlement["outcome"] is None
        assert settlement["source"] == "simmer_outcome_missing"
        assert "does not infer one" in settlement["reason"]

    def test_a_market_the_venue_does_not_have_is_refused(self, monkeypatch):
        client = _Client(market_by_id=None)
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is False
        assert settlement["source"] == "simmer_market_missing"

    def test_a_venue_error_is_reported_not_swallowed(self, monkeypatch):
        client = _Client(market_by_id=RuntimeError("502 bad gateway"))
        adapter = _adapter(monkeypatch, client)
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is False
        assert settlement["is_real"] is False
        assert settlement["source"] == "simmer_unreadable"
        assert "502 bad gateway" in settlement["reason"]

    def test_with_no_key_the_read_says_so(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "simmer_sdk", None)
        adapter = SimmerAdapter()
        settlement = asyncio.run(adapter.get_settlement("simmer-m1"))
        assert settlement["settled"] is False
        assert settlement["is_real"] is False
        assert settlement["source"] == "simmer_unreachable"

    def test_an_empty_id_is_refused(self):
        adapter = SimmerAdapter(api_key="k")
        settlement = asyncio.run(adapter.get_settlement(""))
        assert settlement["settled"] is False
        assert settlement["source"] == "simmer_no_market_id"

    def test_the_adapter_counts_as_one_that_can_close_its_positions(self):
        """V68's gate asks this; a wrong answer here re-opens a dead venue."""
        from src.ptai.venues.adapter import can_report_settlement

        assert can_report_settlement(SimmerAdapter()) is True


# ---------------------------------------------------------------------------
# 7. the account read: $SIM, named as such
# ---------------------------------------------------------------------------

class TestTheAccount:
    def test_with_no_key_it_is_unavailable_with_a_reason(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "simmer_sdk", None)
        adapter = SimmerAdapter()
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert portfolio["balance"] is None
        assert portfolio["reason"]

    def test_positions_and_balance_come_from_the_venue(self, monkeypatch):
        client = _Client(positions=[_Position(market_id="m1", sim_balance=990.0),
                                    _Position(market_id="m2", sim_balance=990.0)])
        adapter = _adapter(monkeypatch, client)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is True
        assert portfolio["balance"] == pytest.approx(990.0)
        assert portfolio["currency"] == "SIM (virtual)"
        assert portfolio["paper"] is True
        assert portfolio["position_count"] == 2
        assert portfolio["positions"][0]["market_id"] == "m1"

    def test_a_broken_account_read_is_a_reason_not_a_crash(self, monkeypatch):
        class _Broken(_Client):
            def get_positions(self, **kwargs):
                raise RuntimeError("no")

        adapter = _adapter(monkeypatch, _Broken())
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert "could not read positions" in portfolio["reason"]


# ---------------------------------------------------------------------------
# 8. end to end: a fill, a recorded position, and the record counted
# ---------------------------------------------------------------------------

class TestAWholePaperRoundThroughTheVenue:
    def test_a_venue_fill_becomes_a_counted_paper_trade(self, monkeypatch, tmp_path):
        """
        The whole loop, through the REAL adapter, storage and settlement engine.

        Only the SDK is fake. The order goes to (the fake) Simmer, the fill that
        comes back is what gets written to the ledger, and the settlement read is
        the adapter's own - so this is the path the 100-resolved-trades gate is
        actually fed by, not a simulation of it.
        """
        from src.ptai.execution.settlement import SettlementEngine
        from src.ptai.storage.db import Storage
        from src.ptai.venues.qualification import paper_record_progress

        class _Registry:
            def __init__(self, adapter):
                self.adapters = {adapter.venue_id: adapter}

            def get_adapter_for_venue_id(self, venue_id):
                return self.adapters.get(venue_id)

        client = _Client(
            markets=[_row("m1", current_probability=0.42)],
            trade_result=_TradeResult(shares_bought=23.8095, new_price=0.42,
                                      cost=10.0, balance=990.0))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, "m1", current_probability=0.42)

        fill = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert fill["status"] == "paper"

        storage = Storage(db_path=str(tmp_path / "simmer.db"))
        try:
            storage.log_trade({
                "market_id": market.id, "market_question": market.question,
                "side": "YES", "market_price": fill["filled_price"],
                "fair_value": 0.5, "edge": 0.08, "kelly_fraction": 0.02,
                "position_size_usd": fill["filled_usd"], "confidence": 0.7,
                "status": "paper", "venue_id": "simmer",
                "execution_mode": "paper",
                "token_price_at_entry": fill["filled_price"], "fees_usd": 0.0,
            })
            before = paper_record_progress(storage, "simmer")
            assert before["open_trades"] == 1
            assert before["resolved"] == 0

            # The venue resolves it: outcome True.
            client._by_id = _row("m1", status="resolved", outcome=True)
            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 1
            assert report.stuck == 0, "a venue that CAN answer is never stuck"

            after = paper_record_progress(storage, "simmer")
            assert after["resolved"] == 1
            assert after["open_trades"] == 0
            # 23.8095 shares redeemed at $1.00 for $10.00: 13.81 profit.
            assert after["net_pnl_usd"] == pytest.approx(13.81, abs=0.05)
        finally:
            storage.close()

    def test_an_open_market_leaves_the_position_open_and_not_stuck(
            self, monkeypatch, tmp_path):
        from src.ptai.execution.settlement import SettlementEngine
        from src.ptai.storage.db import Storage
        from src.ptai.venues.qualification import paper_record_progress

        class _Registry:
            def __init__(self, adapter):
                self.adapters = {adapter.venue_id: adapter}

            def get_adapter_for_venue_id(self, venue_id):
                return self.adapters.get(venue_id)

        client = _Client(market_by_id=_row("m1", status="active"))
        adapter = _adapter(monkeypatch, client)
        storage = Storage(db_path=str(tmp_path / "open.db"))
        try:
            storage.log_trade({
                "market_id": "simmer-m1", "market_question": "Will m1 happen?",
                "side": "YES", "market_price": 0.42, "fair_value": 0.5,
                "edge": 0.08, "kelly_fraction": 0.02,
                "position_size_usd": 10.0, "confidence": 0.7,
                "status": "paper", "venue_id": "simmer",
                "execution_mode": "paper", "token_price_at_entry": 0.42,
                "fees_usd": 0.0,
            })
            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 0
            assert report.stuck == 0, (
                "\"not settled yet\" must never be filed as a venue that "
                "cannot settle")
            assert paper_record_progress(storage, "simmer")["open_trades"] == 1
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# 9. $SIM is not dollars, and no panel may say it is
# ---------------------------------------------------------------------------

class TestVirtualMoneyIsNotCapital:
    def test_a_sim_balance_is_reported_under_its_own_currency(self, monkeypatch):
        client = _Client(positions=[_Position(sim_balance=990.0)])
        adapter = _adapter(monkeypatch, client)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["virtual"] is True
        assert portfolio["paper"] is True
        assert "cannot be withdrawn" in portfolio["currency_note"]

    def test_the_capital_ledger_does_not_count_play_money_as_capital(self, tmp_path):
        """
        The one place a $SIM figure could turn into a claim about real money.

        A venue that reports a balance and no budget used to produce an account
        saying "the venue reports $990.00" - and for Simmer or Manifold that
        sentence would be about play money. The number is kept, under the currency
        that issued it, and the account is not funded by it.
        """
        from src.ptai.execution.capital import CapitalLedger
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=str(tmp_path / "capital.db"))
        try:
            plan = CapitalLedger(storage=storage).build(
                mode="paper", budgets={},
                venue_labels={"simmer": "Simmer"},
                balances={"simmer": {"available": True, "balance": 990.0,
                                     "currency": "SIM (virtual)",
                                     "virtual": True,
                                     "source": "simmer_sdk"}})
            account = next(a for a in plan.accounts if a.venue_id == "simmer")
            assert account.funded is False
            assert account.balance_is_real is False
            assert account.reported_balance_usd == 0.0
            assert account.virtual_balance == pytest.approx(990.0)
            assert account.virtual_currency == "SIM"
            assert account.can_deploy_live is False
            assert not any("reports $990" in w for w in account.warnings)
            assert any("play money" in w for w in account.warnings)
            assert plan.total_available_usd == 0.0
        finally:
            storage.close()

    def test_a_mana_balance_is_treated_the_same_way(self, tmp_path):
        from src.ptai.execution.capital import CapitalLedger
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=str(tmp_path / "mana.db"))
        try:
            plan = CapitalLedger(storage=storage).build(
                mode="paper", budgets={},
                balances={"manifold": {"available": True, "balance": 500.0,
                                       "currency": "MANA", "paper": True,
                                       "virtual": True}})
            account = next(a for a in plan.accounts if a.venue_id == "manifold")
            assert account.reported_balance_usd == 0.0
            assert account.virtual_balance == pytest.approx(500.0)
            assert any("play money" in w for w in account.warnings)
        finally:
            storage.close()

    def test_a_real_usd_balance_is_untouched(self, tmp_path):
        """The guard must not zero a venue that holds actual money."""
        from src.ptai.execution.capital import CapitalLedger, is_virtual_balance
        from src.ptai.storage.db import Storage

        assert is_virtual_balance({"available": True, "balance": 25.0,
                                   "currency": "USD"}) is False
        assert is_virtual_balance({"available": True, "balance": 25.0}) is False
        storage = Storage(db_path=str(tmp_path / "usd.db"))
        try:
            plan = CapitalLedger(storage=storage).build(
                mode="paper", budgets={},
                balances={"polymarket": {"available": True, "balance": 25.0,
                                         "currency": "USDC",
                                         "source": "polymarket_clob"}})
            account = next(a for a in plan.accounts
                           if a.venue_id == "polymarket")
            assert account.reported_balance_usd == pytest.approx(25.0)
            assert account.virtual_balance == 0.0
            assert any("no budget has been authorised" in w
                       for w in account.warnings)
        finally:
            storage.close()

    def test_the_budget_endpoint_refuses_play_money(self, monkeypatch, tmp_path):
        """
        Belt to the braces: even if a funding route were added for a play-money
        venue, the endpoint that authorises capital must refuse it.
        """
        import src.ptai.ui.console as console

        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        monkeypatch.setitem(console.FUNDING_ROUTES, "simmer",
                            {"label": "Simmer", "currency": "USD",
                             "minimum_deposit_usd": 0, "available_from": "anywhere"})

        # The venue answers with its $SIM balance, read from the venue: available,
        # real, and not money.
        async def _virtual_balances(agent=None, force=False):
            return {"simmer": {"available": True, "balance": 990.0,
                               "currency": "SIM (virtual)", "virtual": True,
                               "source": "simmer_sdk"}}

        monkeypatch.setattr(console, "_venue_balances", _virtual_balances)
        from fastapi.testclient import TestClient

        response = TestClient(console.app).post("/api/console/budget",
                                                json={"venue": "simmer",
                                                      "amount": 10})
        assert response.status_code == 409
        assert "cannot be deposited, withdrawn or spent" in response.json()["error"]


# ---------------------------------------------------------------------------
# 10. the venue is no longer listed as having no client
# ---------------------------------------------------------------------------

class TestTheVenueIsNoLongerUnbuilt:
    def test_the_inventory_row_is_not_no_client_and_says_what_it_needs(self):
        from src.ptai.venues.inventory import build_inventory

        inventory = build_inventory(_Registry({"simmer": SimmerAdapter()}))
        row = inventory["venues"]["simmer"]
        assert row["use"] != "no_client"
        # No key saved in this vault: the row says the login is what it needs.
        assert row["use"] == "needs_login"
        assert row["can_report_settlement"] is True
        assert row["fundable"] is False

    def test_with_a_saved_key_the_row_runs_today(self, monkeypatch, tmp_path):
        from src.ptai.venues.inventory import build_inventory

        saved = credential_store.save("simmer", {"api_key": "k"}, str(tmp_path))
        assert saved["ok"] is True
        inventory = build_inventory(_Registry({"simmer": SimmerAdapter()}),
                                    data_dir=str(tmp_path))
        row = inventory["venues"]["simmer"]
        assert row["use"] == "paper_only"
        assert row["can_run_today"] is True
        assert row["paper_tradable"] is True
        assert row["can_place_real_orders"] is False
        assert row["fundable_from_here"] is False
