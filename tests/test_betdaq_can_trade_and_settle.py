"""
Betdaq can take an order - the venue's own play market - and play money is
never counted as capital.

Betdaq is a betting exchange whose API ships as a Python wrapper on PyPI
(`betdaq`, SOAP against api.betdaq.com/v2.0). The venue runs PLAY markets
alongside its real-money ones, and this adapter is pinned to them: discovery
asks for play markets and publishes only rows the venue marks `is_play_market`,
so every stake it places is play money and no real order can leave. The tests
below drive the REAL adapter against a fake `betdaq` package of the shape the
wrapper documents - the parsed row shapes its own `parse_*` functions return -
and the one end-to-end test runs the real storage and settlement engine on the
other side of a venue fill, which is the path the resolved-trades record is
actually fed by.
"""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime

import pytest

from src.ptai.markets.base import Market
from src.ptai.venues import credentials as credential_store
from src.ptai.venues.betdaq_adapter import (
    BETDAQ_ID_PREFIX,
    MIN_STAKE,
    BetdaqAdapter,
)


# ---------------------------------------------------------------------------
# a fake SDK, of the shape the `betdaq` wrapper (0.0.7) documents
# ---------------------------------------------------------------------------

class _Enum:
    """An enum stand-in: the wrapper keeps some statuses as enums, not names."""

    def __init__(self, name):
        self.name = name


def _market_row(market_id=555, *, market_name="Match Odds",
                market_type="MatchOdds", is_play_market=True,
                market_status="ACTIVE", number_of_winners=1,
                event_name="Team A v Team B", sport_name="Soccer",
                withdrawal_sequence_number=3, runners=None,
                market_start_time="2026-10-20 15:00:00.000000"):
    return {
        "runners": runners if runners is not None else [
            {"runner_id": 11, "runner_name": "Team A", "runner_status": "Active",
             "reset_count": 0, "deduction_factor": None,
             "runner_display_order": 1},
            {"runner_id": 12, "runner_name": "Team B", "runner_status": "Active",
             "reset_count": 0, "deduction_factor": None,
             "runner_display_order": 2},
        ],
        "market_id": market_id,
        "market_name": market_name,
        "market_type": market_type,
        "is_play_market": is_play_market,
        "market_status": market_status,
        "number_of_winners": number_of_winners,
        "market_start_time": market_start_time,
        "withdrawal_sequence_number": withdrawal_sequence_number,
        "market_display_order": 1,
        "enabled_for_multiples": False,
        "in_play_available": True,
        "race_grade": None,
        "managed_in_running": False,
        "in_play": False,
        "in_play_delay": 0,
        "event_id": 9001,
        "place_payout": None,
        "event_name": event_name,
        "tournament_id": 77,
        "tournament_name": "League",
        "competition_id": 88,
        "competition_name": "League",
        "sport_id": 100004,
        "sport_name": sport_name,
    }


def _price_row(market_id=555, *, market_status=None, runners=None,
               total_matched=1500.0, withdrawal_sequence_number=3):
    return {
        "market_id": market_id,
        "market_name": "Match Odds",
        "market_type": "MatchOdds",
        "market_start_time": "2026-10-20 15:00:00.000000",
        "runners": runners if runners is not None else [
            {"runner_book": {"batb": [[2.5, 100.0]], "batl": [[2.6, 80.0]]},
             "runner_id": 11, "runner_name": "Team A",
             "runner_status": "Active", "runner_reset_count": 0,
             "deduction_factor": None, "runner_back_matched_size": 900.0,
             "runner_lay_matched_size": 600.0,
             "runner_last_matched_time": "2026-10-09 10:00:00.000000",
             "runner_last_matched_price": 2.5},
            {"runner_book": {"batb": [[1.6, 50.0]], "batl": [[1.65, 40.0]]},
             "runner_id": 12, "runner_name": "Team B",
             "runner_status": "Active", "runner_reset_count": 0,
             "deduction_factor": None, "runner_back_matched_size": 300.0,
             "runner_lay_matched_size": 200.0,
             "runner_last_matched_time": "2026-10-09 10:00:00.000000",
             "runner_last_matched_price": 1.6},
        ],
        "is_play_market": True,
        # The wrapper keeps THIS one as an enum, not a name.
        "status": _Enum(market_status or "ACTIVE"),
        "number_of_winners": 1.0,
        "withdrawal_sequence_number": withdrawal_sequence_number,
        "market_display_order": 1,
        "enabled_for_multiples": False,
        "in_play_available": True,
        "race_grade": None,
        "managed_in_running": False,
        "in_play": False,
        "in_running_delay": 0,
        "event_id": 9001,
        "market_total_matched": total_matched,
        "market_back_matched": 1200.0,
        "market_lay_matched": 800.0,
        "home_team_score": None,
        "away_team_score": None,
        "score_type": None,
    }


def _change_row(market_id=555, runner_id=11, runner_name="Team A",
                result="Winner", sequence_number=41,
                runner_status="Settled"):
    return {
        "runner_id": runner_id,
        "runner_name": runner_name,
        "runner_display_order": 1,
        "runner_hidden": False,
        # The wrapper keeps THIS one as an enum, not a name.
        "runner_status": _Enum(runner_status),
        "reset_count": 0,
        "market_id": market_id,
        "withdrawal_factor": None,
        "sequence_number": sequence_number,
        "cancel_orders_time": None,
        "settlement_info": [{"settled_time": "2026-10-21 17:00:00.000000",
                             "void_percentage": None, "result": result,
                             "left_side_percentage": None,
                             "right_side_percentage": None}],
    }


def _receipt(*, order_id=9001, matched_size=10.0, matched_price=2.5,
             status="Matched", return_code=0, runner_id=11):
    return {
        "order_id": order_id,
        "side": "back",
        "size_remaining": 0.0,
        "matched_price": matched_price,
        "matched_size": matched_size,
        "matched_lay_size": matched_size,
        "sent_time": "2026-10-09 10:05:00.000000",
        # The wrapper keeps THIS one as an enum, not a name.
        "status": _Enum(status),
        "runner_sequence_number": 41,
        "runner_id": runner_id,
        "customer_reference": None,
        "return_code": return_code,
    }


def _order_row(*, order_id=9001, market_id=555, runner_id=11, status="Matched",
               side="back", matched_size=10.0, average_price=2.5):
    return {
        "order_id": order_id,
        "commission_information": {},
        "runner_id": runner_id,
        "market_id": market_id,
        "sequence_number": 41,
        "status": status,
        "side": side,
        "sent_time": "2026-10-09 10:05:00.000000",
        "price": 2.5,
        "remaining_size": 0.0,
        "average_price": average_price,
        "matched_price": 2.5,
        "matched_size": matched_size,
        "matched_lay_size": matched_size,
    }


class _MarketData:
    def __init__(self, client):
        self._client = client

    def get_sports(self):
        self._client.calls.append(("get_sports", {}))
        return list(self._client.sports)

    def get_sport_markets(self, sport_ids, include_selections=True,
                          WantPlayMarkets=None):
        self._client.calls.append(("get_sport_markets",
                                   {"sport_ids": list(sport_ids),
                                    "include_selections": include_selections,
                                    "WantPlayMarkets": WantPlayMarkets}))
        return [dict(row) for row in self._client.market_rows]

    def get_prices(self, market_ids, **kwargs):
        self._client.calls.append(("get_prices", {"market_ids": list(market_ids)}))
        wanted = {int(m) for m in market_ids}
        return [dict(row) for row in self._client.price_rows
                if int(row["market_id"]) in wanted]

    def get_markets(self, market_ids):
        self._client.calls.append(("get_markets", {"market_ids": list(market_ids)}))
        wanted = {int(m) for m in market_ids}
        if isinstance(self._client.markets_by_id, Exception):
            raise self._client.markets_by_id
        return [dict(row) for row in self._client.market_info_rows
                if int(row["market_id"]) in wanted]

    def get_selection_changes(self, since):
        self._client.calls.append(("get_selection_changes", {"since": since}))
        return [dict(row) for row in self._client.change_rows]

    def get_selection_sequence_number(self):
        self._client.calls.append(("get_selection_sequence_number", {}))
        return self._client.sequence_number

    def get_odds_ladder(self):
        self._client.calls.append(("get_odds_ladder", {}))
        return list(self._client.odds_ladder)


class _Betting:
    def __init__(self, client):
        self._client = client

    def place_orders(self, order_list, receipt=True):
        self._client.placed.append(list(order_list))
        self._client.calls.append(("place_orders",
                                   {"order_list": list(order_list),
                                    "receipt": receipt}))
        if isinstance(self._client.receipt, Exception):
            raise self._client.receipt
        return [dict(row) for row in self._client.receipts]

    def get_orders(self, **kwargs):
        self._client.calls.append(("get_orders", kwargs))
        if isinstance(self._client.orders, Exception):
            raise self._client.orders
        return [dict(row) for row in self._client.order_rows]


class _Account:
    def __init__(self, client):
        self._client = client

    def get_account_balances(self):
        self._client.calls.append(("get_account_balances", {}))
        return dict(self._client.balance_row)


class _Client:
    """Records every call, answers from what the constructor was given."""

    def __init__(self, market_rows=None, price_rows=None, market_info_rows=None,
                 change_rows=None, receipts=None, receipt=None, orders=None,
                 order_rows=None, sports=None, odds_ladder=None,
                 markets_by_id=None, sequence_number=41, balance_row=None):
        self.market_rows = market_rows if market_rows is not None else [
            _market_row()]
        self.price_rows = price_rows if price_rows is not None else [
            _price_row()]
        self.market_info_rows = (market_info_rows
                                 if market_info_rows is not None
                                 else list(self.market_rows))
        self.change_rows = change_rows if change_rows is not None else []
        self.receipts = receipts if receipts is not None else [_receipt()]
        self.receipt = receipt
        self.orders = orders
        self.order_rows = order_rows if order_rows is not None else []
        self.sports = sports if sports is not None else [
            {"display_order": 1, "sport_id": 100004,
             "sport_name": "Soccer"}]
        self.odds_ladder = odds_ladder if odds_ladder is not None else [
            {"price": 1.01, "value": "1.01"}, {"price": 1.02, "value": "1.02"},
            {"price": 1.03, "value": "1.03"}, {"price": 2.5, "value": "2.50"},
            {"price": 2.52, "value": "2.52"}]
        self.markets_by_id = markets_by_id
        self.sequence_number = sequence_number
        self.balance_row = balance_row or {"currency": "GBP",
                                            "available_funds": 25.0,
                                            "balance": 25.0, "credit": 0.0,
                                            "exposure": 0.0}
        self.marketdata = _MarketData(self)
        self.betting = _Betting(self)
        self.account = _Account(self)
        self.calls = []
        self.placed = []


def _install_sdk(monkeypatch, client):
    """Install a fake `betdaq` package and return the construction record."""
    constructed = []
    package = types.ModuleType("betdaq")
    apiclient = types.ModuleType("betdaq.apiclient")
    enums = types.ModuleType("betdaq.enums")
    filters = types.ModuleType("betdaq.filters")

    class _Polarity:
        back = 1
        lay = 2

    enums.Polarity = _Polarity

    def create_order(**kwargs):
        return dict(kwargs)

    filters.create_order = create_order

    def APIClient(username, password):  # noqa: N802 - the SDK's own name
        constructed.append({"username": username, "password": password})
        return client

    apiclient.APIClient = APIClient
    package.apiclient = apiclient
    package.enums = enums
    package.filters = filters
    monkeypatch.setitem(sys.modules, "betdaq", package)
    monkeypatch.setitem(sys.modules, "betdaq.apiclient", apiclient)
    monkeypatch.setitem(sys.modules, "betdaq.enums", enums)
    monkeypatch.setitem(sys.modules, "betdaq.filters", filters)
    return constructed


def _adapter(monkeypatch, client=None, username="bettor", password="pw",
             dry_run=True):
    adapter = BetdaqAdapter(username=username, password=password)
    adapter.dry_run = dry_run
    if client is not None:
        _install_sdk(monkeypatch, client)
    return adapter


class _Registry:
    """What `build_inventory` needs: `.adapters` by venue id, and a country."""

    country_code = "UG"

    def __init__(self, adapters):
        self.adapters = dict(adapters)


class _Opportunity:
    def __init__(self, market, side="YES"):
        self.market = market
        self.side = side
        self.venue_id = "betdaq"


def _discover(adapter, count=10):
    return asyncio.run(adapter.discover_markets(target_count=count))


_UNSET = object()


def _market(adapter, client, market_id=555, price_row=_UNSET, **overrides):
    client.market_rows = [_market_row(market_id, **overrides)]
    # price_row is a parameter because several book tests need a CUSTOM
    # price row (a wide book, a crossed book, a suspended market, no prices);
    # the default is the plain two-runner book.
    client.price_rows = ([_price_row(market_id)] if price_row is _UNSET
                         else price_row)
    # market_info_rows is deliberately NOT touched: the default mirrors
    # market_rows, and a test that needs a settled market-info row (the
    # settlement tests) constructs its client with one.
    return _discover(adapter)[0]


# ---------------------------------------------------------------------------
# 1. the venue has a client now, and it says which one
# ---------------------------------------------------------------------------

class TestTheVenueHasAClientNow:
    def test_the_flags_say_what_the_code_does(self):
        caps = BetdaqAdapter().capabilities
        assert caps.implementation_status == "live"
        assert caps.supports_market_discovery is True
        assert caps.supports_orderbook is True
        # NO REAL-MONEY PATH: the adapter is pinned to the venue's play markets.
        assert caps.supports_trading is False
        assert caps.real_order_path is False
        assert caps.requires_credentials is True
        assert BetdaqAdapter().can_place_real_orders is False
        assert BetdaqAdapter(username="u", password="p").can_place_real_orders \
            is False
        # Even in live mode: no real-money market is ever read.
        live = BetdaqAdapter(username="u", password="p")
        live.dry_run = False
        assert live.can_place_real_orders is False

    def test_the_account_read_needs_the_login_and_says_so(self):
        assert BetdaqAdapter().capabilities.supports_portfolio is False
        assert BetdaqAdapter(username="u").capabilities.supports_portfolio is False
        assert BetdaqAdapter(username="u", password="p").capabilities \
            .supports_portfolio is True

    def test_with_no_login_nothing_is_read_and_the_reason_says_so(self,
                                                                 monkeypatch):
        adapter = BetdaqAdapter()
        monkeypatch.setitem(sys.modules, "betdaq", None)
        assert _discover(adapter) == []
        assert "no Betdaq login saved" in adapter.last_error
        book = asyncio.run(adapter.get_orderbook(
            Market(id="betdaq-555", source="polymarket", question="q",
                   venue_id="betdaq")))
        assert book["available"] is False
        assert book["is_real"] is False

    def test_with_no_sdk_installed_the_reason_names_the_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "betdaq", None)
        adapter = BetdaqAdapter(username="u", password="p")
        assert adapter.sdk_available is False
        assert _discover(adapter) == []
        assert "pip install betdaq" in adapter.last_error

    def test_the_client_is_built_with_the_login_and_reused(self, monkeypatch):
        client = _Client()
        constructed = _install_sdk(monkeypatch, client)
        adapter = BetdaqAdapter(username="bettor", password="pw")
        _discover(adapter)
        assert constructed == [{"username": "bettor", "password": "pw"}]
        # And the client is reused, not rebuilt per call.
        _discover(adapter)
        assert len(constructed) == 1

    def test_a_client_that_cannot_be_built_is_a_reason_not_a_crash(
            self, monkeypatch):
        # The SDK imports fine; the CLIENT construction fails (the wrapper
        # builds its SOAP clients from the venue's WSDL at construction).
        package = types.ModuleType("betdaq")
        apiclient = types.ModuleType("betdaq.apiclient")
        enums = types.ModuleType("betdaq.enums")
        filters = types.ModuleType("betdaq.filters")

        class _Polarity:
            back = 1
            lay = 2

        enums.Polarity = _Polarity
        filters.create_order = lambda **kwargs: dict(kwargs)

        def _broken(username, password):
            raise RuntimeError("wsdl unreachable")

        apiclient.APIClient = _broken
        package.apiclient = apiclient
        package.enums = enums
        package.filters = filters
        monkeypatch.setitem(sys.modules, "betdaq", package)
        monkeypatch.setitem(sys.modules, "betdaq.apiclient", apiclient)
        monkeypatch.setitem(sys.modules, "betdaq.enums", enums)
        monkeypatch.setitem(sys.modules, "betdaq.filters", filters)
        adapter = BetdaqAdapter(username="u", password="p")
        assert _discover(adapter) == []
        assert "could not be built" in adapter.last_error
        assert "wsdl unreachable" in adapter.last_error


# ---------------------------------------------------------------------------
# 2. the login the product offers
# ---------------------------------------------------------------------------

class TestTheLoginIsOffered:
    def test_there_is_a_betdaq_login_form(self):
        tool = credential_store.TOOLS["betdaq"]
        assert tool.venue_id == "betdaq"
        assert tool.kind == "venue"
        assert [f.name for f in tool.fields] == ["username", "password"]
        assert tool.fields[0].env == "BETDAQ_USERNAME"
        assert tool.fields[1].env == "BETDAQ_PASSWORD"
        assert credential_store.TOOL_FOR_VENUE["betdaq"] == "betdaq"
        # The unlock statement must not imply real money can reach this venue.
        assert "play money needs no card, no wallet and no deposit" in tool.unlocks
        assert "not used here" in tool.then

    def test_saving_a_login_reaches_the_running_adapter(self, tmp_path):
        saved = credential_store.save(
            "betdaq", {"username": "bettor", "password": "pw-abc"},
            str(tmp_path))
        assert saved["ok"] is True
        adapter = BetdaqAdapter()
        applied = credential_store._apply_betdaq(adapter, str(tmp_path))
        assert "username" in applied
        assert "password" in applied
        assert adapter.username == "bettor"
        assert adapter.password == "pw-abc"
        assert adapter.capabilities.supports_portfolio is True

    def test_saving_the_same_login_twice_is_a_no_op(self, tmp_path):
        credential_store.save("betdaq", {"username": "bettor", "password": "pw"},
                              str(tmp_path))
        adapter = BetdaqAdapter(username="bettor", password="pw")
        assert credential_store._apply_betdaq(adapter, str(tmp_path)) == []

    def test_a_cached_client_built_without_the_login_is_dropped(self, tmp_path,
                                                                monkeypatch):
        client = _Client()
        adapter = BetdaqAdapter()
        _install_sdk(monkeypatch, client)
        assert _discover(adapter) == []          # no login: nothing was built
        adapter._client_obj = object()           # pretend a client was cached
        credential_store.save("betdaq", {"username": "bettor", "password": "pw"},
                              str(tmp_path))
        applied = credential_store._apply_betdaq(adapter, str(tmp_path))
        assert "client_rebuilt" in applied
        assert adapter._client_obj is None

    def test_the_agent_registers_the_adapter_with_the_saved_login(
            self, monkeypatch):
        from src.ptai.agent.v3_loop import TradingAgentV3

        def _resolve(name, data_dir="./data"):
            if name == "betdaq":
                return {"username": "wired-user", "password": "wired-pw"}
            return {}

        monkeypatch.setattr(credential_store, "resolve", _resolve)
        agent = TradingAgentV3(country_code="UG", dry_run=True)
        registered = agent.venue_registry.get_adapter_for_venue_id("betdaq")
        assert registered is not None
        assert registered.username == "wired-user"
        assert registered.password == "wired-pw"
        assert registered.capabilities.implementation_status == "live"

    def test_the_refresh_wiring_knows_about_betdaq(self):
        import inspect

        source = inspect.getsource(credential_store.refresh_adapters)
        assert '"betdaq"' in source


# ---------------------------------------------------------------------------
# 3. discovery: the venue's play markets, and only the tradeable ones
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_one_market_per_venue_market_with_the_venues_probability(
            self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555), _market_row(556, market_name="Race 1")],
            price_rows=[_price_row(555), _price_row(556)])
        adapter = _adapter(monkeypatch, client)
        markets = _discover(adapter)
        assert [m.id for m in markets] == ["betdaq-555", "betdaq-556"]
        # Decimal odds become the probability 1/odds, per runner.
        assert markets[0].outcome_prices[0] == pytest.approx(1 / 2.5)
        assert markets[0].outcome_prices[1] == pytest.approx(1 / 1.6)
        assert markets[0].outcomes == ["Team A", "Team B"]
        assert markets[0].venue_id == "betdaq"
        assert markets[0].raw["is_play_market"] is True
        assert markets[0].raw["selection_ids"] == [11, 12]
        assert markets[0].end_date == datetime(2026, 10, 20, 15, 0,
                                               tzinfo=__import__("datetime")
                                               .timezone.utc)
        # The venue publishes matched amounts in PLAY money, which is not the
        # unit the scan's floors are written in: the basis is recorded, and the
        # volume floors are skipped rather than fed a different unit.
        assert markets[0].volume_is_published is False
        assert markets[0].raw["volume_basis"].startswith("play_money")

    def test_the_venue_is_asked_for_play_markets_only(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        _discover(adapter)
        calls = {name: kwargs for name, kwargs in client.calls}
        assert calls["get_sport_markets"]["WantPlayMarkets"] is True
        assert calls["get_sport_markets"]["include_selections"] is True
        assert calls["get_sport_markets"]["sport_ids"] == [100004]

    def test_real_money_markets_are_not_candidates(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555),
                         _market_row(556, is_play_market=False)],
            price_rows=[_price_row(555), _price_row(556)])
        adapter = _adapter(monkeypatch, client)
        markets = _discover(adapter)
        assert [m.id for m in markets] == ["betdaq-555"]
        assert adapter.skipped_markets.get("not a play market (this adapter "
                                           "reads play markets only)") == 1

    def test_non_active_markets_are_not_candidates(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555),
                         _market_row(556, market_status="SUSPENDED"),
                         _market_row(557, market_status="SETTLED")],
            price_rows=[_price_row(555), _price_row(556), _price_row(557)])
        adapter = _adapter(monkeypatch, client)
        assert [m.id for m in _discover(adapter)] == ["betdaq-555"]

    def test_place_markets_are_not_one_outcome(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555, number_of_winners=2)],
            price_rows=[_price_row(555)])
        adapter = _adapter(monkeypatch, client)
        assert _discover(adapter) == []
        assert adapter.skipped_markets.get("not a single-winner market") == 1

    def test_an_unnamed_market_type_is_refused(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555, market_type="Unspecified")],
            price_rows=[_price_row(555)])
        adapter = _adapter(monkeypatch, client)
        assert _discover(adapter) == []
        assert adapter.skipped_markets.get("market type not named by the "
                                           "venue") == 1

    def test_a_market_with_one_runner_is_not_a_market(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555, runners=[
                {"runner_id": 11, "runner_name": "Team A",
                 "runner_status": "Active", "reset_count": 0}])],
            price_rows=[_price_row(555)])
        adapter = _adapter(monkeypatch, client)
        assert _discover(adapter) == []
        assert adapter.skipped_markets.get("fewer than two runners") == 1

    def test_a_market_whose_first_runner_is_unpriced_is_not_published(
            self, monkeypatch):
        # The published outcome order must match the venue's settlement order;
        # a first runner with no two-sided price would break that mapping.
        client = _Client(
            market_rows=[_market_row(555)],
            price_rows=[_price_row(555, runners=[
                {"runner_book": {"batb": [], "batl": []}, "runner_id": 11,
                 "runner_name": "Team A", "runner_status": "Active",
                 "runner_reset_count": 0},
                {"runner_book": {"batb": [[1.6, 50.0]],
                                 "batl": [[1.65, 40.0]]},
                 "runner_id": 12, "runner_name": "Team B",
                 "runner_status": "Active", "runner_reset_count": 0},
            ])])
        adapter = _adapter(monkeypatch, client)
        assert _discover(adapter) == []
        assert adapter.skipped_markets.get("first runner unpriced") == 1

    def test_a_market_with_no_two_sided_price_is_not_published(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555)],
            price_rows=[_price_row(555, runners=[
                {"runner_book": {"batb": [[2.5, 100.0]], "batl": []},
                 "runner_id": 11, "runner_name": "Team A",
                 "runner_status": "Active", "runner_reset_count": 0},
                {"runner_book": {"batb": [[1.6, 50.0]], "batl": []},
                 "runner_id": 12, "runner_name": "Team B",
                 "runner_status": "Active", "runner_reset_count": 0},
            ])])
        adapter = _adapter(monkeypatch, client)
        assert _discover(adapter) == []
        assert adapter.skipped_markets.get("no two-sided price") == 1

    def test_a_venue_read_that_raises_is_a_reason_not_a_crash(self, monkeypatch):
        class _Broken(_Client):
            def __init__(self):
                super().__init__()
                self.marketdata = self.marketdata  # keep

        client = _Client()
        client.sports = RuntimeError("connection reset")
        adapter = _adapter(monkeypatch, client)
        # get_sports raising is exercised through a broken endpoint object.
        class _BrokenMarketData(_MarketData):
            def get_sports(self):
                raise RuntimeError("connection reset")

        client.marketdata = _BrokenMarketData(client)
        assert _discover(adapter) == []
        assert "connection reset" in adapter.last_error


# ---------------------------------------------------------------------------
# 4. the book: the venue's own ladders, in the project's price space
# ---------------------------------------------------------------------------

class TestTheBook:
    def test_the_venue_ladders_become_a_probability_book(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        book = asyncio.run(adapter.get_orderbook(market))
        # The back side is what a backer pays: the asks. The lay side is what a
        # layer sells at: the bids. Stakes become contracts (stake x odds).
        assert book["ask"] == pytest.approx(1 / 2.5)
        assert book["bid"] == pytest.approx(1 / 2.6)
        assert book["asks"][0]["size"] == pytest.approx(100.0 * 2.5)
        assert book["bids"][0]["size"] == pytest.approx(80.0 * 2.6)
        # ask minus bid, positive in a normal market.
        assert book["spread"] == pytest.approx(1 / 2.5 - 1 / 2.6, abs=1e-6)
        assert book["executable"] is True
        assert book["validated"] is True
        assert book["is_real"] is True
        assert book["is_mock"] is False
        assert book["currency"] == "play money (virtual)"
        assert book["source"] == "betdaq_play_book"
        assert len(book["runners"]) == 2

    def test_a_wide_book_is_not_executable_but_is_still_a_book(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, price_row=[_price_row(555, runners=[
            {"runner_book": {"batb": [[1.05, 100.0]], "batl": [[10.0, 80.0]]},
             "runner_id": 11, "runner_name": "Team A",
             "runner_status": "Active", "runner_reset_count": 0},
            {"runner_book": {"batb": [[1.6, 50.0]], "batl": [[1.65, 40.0]]},
             "runner_id": 12, "runner_name": "Team B",
             "runner_status": "Active", "runner_reset_count": 0},
        ])])
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is True
        assert book["executable"] is False
        assert "wider than the" in book["warning"]

    def test_a_crossed_quote_is_a_refusal_not_a_price(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client, price_row=[_price_row(555, runners=[
            # batl below batb: the lay side paying more than the back side
            # costs - a crossed market in probability space.
            {"runner_book": {"batb": [[2.5, 100.0]], "batl": [[2.4, 80.0]]},
             "runner_id": 11, "runner_name": "Team A",
             "runner_status": "Active", "runner_reset_count": 0},
            {"runner_book": {"batb": [[1.6, 50.0]], "batl": [[1.65, 40.0]]},
             "runner_id": 12, "runner_name": "Team B",
             "runner_status": "Active", "runner_reset_count": 0},
        ])])
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is False
        assert book["is_real"] is False
        assert "crossed" in book["reason"]

    def test_a_market_the_venue_is_not_running_is_not_orderable(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client,
                         price_row=[_price_row(555, market_status="SUSPENDED")])
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is False
        assert "SUSPENDED" in book["reason"]

    def test_a_market_with_no_prices_has_no_book(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)   # published from the default book
        client.price_rows = []              # the venue stops answering
        book = asyncio.run(adapter.get_orderbook(market))
        assert book["available"] is False
        assert book["is_real"] is False
        assert "no prices" in book["reason"]

    def test_no_login_means_no_book_and_the_reason_says_so(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "betdaq", None)
        adapter = BetdaqAdapter(username="u", password="p")
        book = asyncio.run(adapter.get_orderbook(
            Market(id="betdaq-555", source="polymarket", question="q",
                   venue_id="betdaq")))
        assert book["available"] is False
        assert "pip install betdaq" in book["reason"]


# ---------------------------------------------------------------------------
# 5. the order goes to the venue, and the venue's answer is what is recorded
# ---------------------------------------------------------------------------

class TestTheOrderGoesToTheVenue:
    def test_a_paper_order_is_submitted_and_the_venues_match_is_recorded(
            self, monkeypatch):
        client = _Client(receipts=[_receipt(matched_size=10.0,
                                           matched_price=2.5)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        # The venue was called - THIS is the difference from a local simulation.
        assert len(client.placed) == 1
        sent = client.placed[0][0]
        assert sent["SelectionId"] == 11
        assert sent["Stake"] == pytest.approx(10.0)
        assert sent["Price"] == pytest.approx(2.5)
        assert sent["Polarity"] == 1          # Polarity.back
        assert sent["ExpectedSelectionResetCount"] == 0
        assert sent["ExpectedWithdrawalSequenceNumber"] == 3
        # What the venue matched, in the terms the ledger records: the stake at
        # risk is the cost, and the contracts are stake x odds.
        assert result["status"] == "paper"
        assert result["success"] is True
        assert result["is_real"] is False
        assert result["filled_price"] == pytest.approx(0.4)
        assert result["size"] == pytest.approx(25.0)
        assert result["filled_usd"] == pytest.approx(10.0)
        assert result["currency"] == "play money (virtual)"
        assert result["venue_trade_id"] == 9001
        assert "No real money was involved" in result["reason"]

    def test_live_mode_is_refused_and_says_why(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client, dry_run=False)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert result["status"] == "refused"
        assert result["success"] is False
        assert "will not place REAL-money orders" in result["reason"]
        assert client.placed == []

    def test_no_on_a_two_outcome_market_bets_the_other_runner(self, monkeypatch):
        client = _Client(receipts=[_receipt(matched_size=10.0, matched_price=1.6,
                                           runner_id=12)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "NO"), max_spend_usd=10.0, max_price=0.7))
        assert client.placed[0][0]["SelectionId"] == 12
        assert result["status"] == "paper"
        assert result["filled_price"] == pytest.approx(1 / 1.6)

    def test_no_on_a_three_outcome_market_is_refused(self, monkeypatch):
        client = _Client(
            market_rows=[_market_row(555, runners=[
                {"runner_id": 11, "runner_name": "Team A",
                 "runner_status": "Active", "reset_count": 0},
                {"runner_id": 12, "runner_name": "Draw",
                 "runner_status": "Active", "reset_count": 0},
                {"runner_id": 13, "runner_name": "Team B",
                 "runner_status": "Active", "reset_count": 0}])],
            price_rows=[_price_row(555, runners=[
                {"runner_book": {"batb": [[2.5, 100.0]], "batl": [[2.6, 80.0]]},
                 "runner_id": 11, "runner_name": "Team A",
                 "runner_status": "Active", "runner_reset_count": 0},
                {"runner_book": {"batb": [[3.4, 30.0]], "batl": [[3.5, 25.0]]},
                 "runner_id": 12, "runner_name": "Draw",
                 "runner_status": "Active", "runner_reset_count": 0},
                {"runner_book": {"batb": [[3.4, 30.0]], "batl": [[3.5, 25.0]]},
                 "runner_id": 13, "runner_name": "Team B",
                 "runner_status": "Active", "runner_reset_count": 0}])])
        adapter = _adapter(monkeypatch, client)
        market = _discover(adapter)[0]
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "NO"), max_spend_usd=10.0, max_price=0.45))
        assert result["status"] == "rejected"
        assert "not one runner" in result["reason"]
        assert client.placed == []

    def test_a_price_above_the_cap_rests_at_the_limit_and_is_honest_about_it(
            self, monkeypatch):
        # The venue offers 2.50 (probability 0.40); the cap is 0.30, so the
        # order rests at the limit odds (1/0.30 rounded up) and the venue
        # matches nothing - reported as submitted, not as a fill.
        client = _Client(receipts=[_receipt(matched_size=0.0)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.30))
        sent = client.placed[0][0]
        assert sent["Price"] == pytest.approx(3.34)
        assert result["status"] == "submitted"
        assert result["success"] is False
        assert "matched nothing" in result["reason"]

    def test_a_stake_below_the_venue_minimum_is_refused(self, monkeypatch):
        client = _Client()
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=MIN_STAKE / 2,
            max_price=0.45))
        assert result["status"] == "rejected"
        assert "below the venue minimum" in result["reason"]
        assert client.placed == []

    def test_a_venue_refusal_is_a_refusal_not_a_fill(self, monkeypatch):
        client = _Client(receipts=[_receipt(return_code=7)])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert result["status"] == "rejected"
        assert "return code 7" in result["reason"]

    def test_a_venue_error_is_an_error_not_a_fill(self, monkeypatch):
        client = _Client(receipt=RuntimeError("soap fault"))
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert result["status"] == "error"
        assert "soap fault" in result["reason"]

    def test_no_login_is_a_refusal(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "betdaq", None)
        adapter = BetdaqAdapter(username="u", password="p")
        market = Market(id="betdaq-555", source="polymarket", question="q",
                        venue_id="betdaq", raw={"selection_ids": [11, 12]})
        result = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert result["status"] == "refused"
        assert "pip install betdaq" in result["reason"]


# ---------------------------------------------------------------------------
# 6. the outcome: the venue's own result string, mapped onto the outcome order
# ---------------------------------------------------------------------------

class TestTheOutcome:
    def _settled_client(self, **overrides):
        rows = overrides.pop("change_rows", None)
        info = _market_row(555, market_status="SETTLED")
        return _Client(
            market_info_rows=[info],
            change_rows=rows if rows is not None else [
                _change_row(555, 11, "Team A", "Winner", 41),
                _change_row(555, 12, "Team B", "Loser", 42),
            ])

    def test_an_open_market_is_not_settled_and_says_so(self, monkeypatch):
        client = _Client(market_info_rows=[_market_row(555,
                                                      market_status="ACTIVE")])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is True
        assert verdict["source"] == "betdaq_open"
        assert "ACTIVE" in verdict["reason"]

    def test_a_closed_market_with_no_result_yet_is_not_settled(self, monkeypatch):
        client = _Client(market_info_rows=[_market_row(555,
                                                      market_status="CLOSED")])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["source"] == "betdaq_open"

    def test_a_settled_market_resolves_to_the_first_outcome_when_it_won(
            self, monkeypatch):
        client = self._settled_client()
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is True
        assert verdict["outcome"] == 1.0
        assert verdict["is_real"] is True
        assert verdict["source"] == "betdaq_selection_result"
        assert verdict["winner_runner_id"] == 11
        assert verdict["winner_name"] == "Team A"
        assert verdict["runner_results"] == {"11": "Winner", "12": "Loser"}

    def test_a_settled_market_resolves_to_no_when_another_runner_won(
            self, monkeypatch):
        client = self._settled_client(change_rows=[
            _change_row(555, 11, "Team A", "Loser", 41),
            _change_row(555, 12, "Team B", "Winner", 42),
        ])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is True
        assert verdict["outcome"] == 0.0
        assert verdict["winner_runner_id"] == 12

    def test_a_settled_market_with_no_winner_is_refused_not_guessed(
            self, monkeypatch):
        client = self._settled_client(change_rows=[
            _change_row(555, 11, "Team A", "Loser", 41),
            _change_row(555, 12, "Team B", "Loser", 42),
        ])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is True
        assert verdict["source"] == "betdaq_settled_no_winner"
        assert "does not infer" in verdict["reason"]

    def test_an_unrecognised_result_string_is_not_the_winner(self, monkeypatch):
        client = self._settled_client(change_rows=[
            _change_row(555, 11, "Team A", "Dead heat", 41),
            _change_row(555, 12, "Team B", "Void", 42),
        ])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["source"] == "betdaq_settled_no_winner"

    def test_a_voided_market_resolves_nothing(self, monkeypatch):
        client = _Client(
            market_info_rows=[_market_row(555, market_status="VOIDED")])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["outcome"] is None
        assert verdict["is_real"] is True
        assert verdict["source"] == "betdaq_voided"
        assert "resolves nothing" in verdict["reason"]

    def test_a_missing_market_is_reported_not_invented(self, monkeypatch):
        client = _Client(market_info_rows=[])
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-999"))
        assert verdict["settled"] is False
        assert verdict["source"] == "betdaq_market_missing"

    def test_an_unreadable_market_is_a_reason_not_a_crash(self, monkeypatch):
        client = _Client(markets_by_id=RuntimeError("soap fault"))
        adapter = _adapter(monkeypatch, client)
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is False
        assert verdict["source"] == "betdaq_unreadable"
        assert "soap fault" in verdict["reason"]

    def test_the_first_read_polls_the_full_history_once_then_increments(
            self, monkeypatch):
        client = self._settled_client()
        adapter = _adapter(monkeypatch, client)
        asyncio.run(adapter.get_settlement("betdaq-555"))
        first = [c for c in client.calls if c[0] == "get_selection_changes"]
        assert first[0][1]["since"] == 0
        assert adapter._selection_sequence == 42
        # A second, different market polls from the advanced cursor. Its own
        # first runner is 21, so the winner IS its first published outcome.
        client.market_info_rows = [
            _market_row(555, market_status="SETTLED"),
            _market_row(556, market_status="SETTLED", market_name="Race 1",
                        runners=[
                            {"runner_id": 21, "runner_name": "Horse A",
                             "runner_status": "Active", "reset_count": 0,
                             "deduction_factor": None,
                             "runner_display_order": 1},
                            {"runner_id": 22, "runner_name": "Horse B",
                             "runner_status": "Active", "reset_count": 0,
                             "deduction_factor": None,
                             "runner_display_order": 2},
                        ]),
        ]
        client.change_rows = [
            _change_row(556, 21, "Horse A", "Winner", 55),
            _change_row(556, 22, "Horse B", "Loser", 56),
        ]
        verdict = asyncio.run(adapter.get_settlement("betdaq-556"))
        second = [c for c in client.calls if c[0] == "get_selection_changes"]
        assert second[-1][1]["since"] == 42
        assert verdict["settled"] is True
        assert verdict["outcome"] == 1.0

    def test_a_settled_verdict_is_cached_not_re_polled(self, monkeypatch):
        client = self._settled_client()
        adapter = _adapter(monkeypatch, client)
        asyncio.run(adapter.get_settlement("betdaq-555"))
        polls = len([c for c in client.calls
                     if c[0] == "get_selection_changes"])
        verdict = asyncio.run(adapter.get_settlement("betdaq-555"))
        assert verdict["settled"] is True
        assert len([c for c in client.calls
                    if c[0] == "get_selection_changes"]) == polls


# ---------------------------------------------------------------------------
# 7. the account: play-money positions, and no real balance reported as ours
# ---------------------------------------------------------------------------

class TestTheAccount:
    def test_with_no_login_it_is_unavailable_with_a_reason(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "betdaq", None)
        adapter = BetdaqAdapter()
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert portfolio["balance"] is None
        assert portfolio["reason"]

    def test_positions_are_play_money_and_the_real_balance_is_not_reported(
            self, monkeypatch):
        client = _Client(
            order_rows=[
                _order_row(order_id=9001, market_id=555, runner_id=11),
                _order_row(order_id=9002, market_id=555, runner_id=12,
                           status="Settled"),
                _order_row(order_id=9003, market_id=777, runner_id=31),
            ])
        adapter = _adapter(monkeypatch, client)
        _discover(adapter)          # populates the play-market id set (555)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is True
        # The venue publishes one REAL-money balance; this adapter does not
        # report it as this venue's, because it trades only play markets.
        assert portfolio["balance"] is None
        assert "real money" in portfolio["balance_note"]
        assert portfolio["virtual"] is True
        assert portfolio["paper"] is True
        assert portfolio["currency"] == "play money (virtual)"
        # Only the matched, unsettled order on a PLAY market is a position.
        assert portfolio["position_count"] == 1
        assert portfolio["positions"][0]["market_id"] == 555
        assert portfolio["positions"][0]["virtual"] is True
        assert portfolio["orders_skipped_not_play"] == 1
        assert "cannot be withdrawn" in portfolio["currency_note"]

    def test_a_broken_account_read_is_a_reason_not_a_crash(self, monkeypatch):
        client = _Client(orders=RuntimeError("no"))
        adapter = _adapter(monkeypatch, client)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["available"] is False
        assert "could not read orders" in portfolio["reason"]


# ---------------------------------------------------------------------------
# 8. end to end: a fill, a recorded position, and the record counted
# ---------------------------------------------------------------------------

class TestAWholePaperRoundThroughTheVenue:
    def test_a_venue_fill_becomes_a_counted_paper_trade(self, monkeypatch,
                                                        tmp_path):
        """
        The whole loop, through the REAL adapter, storage and settlement engine.

        Only the SDK is fake. The order goes to (the fake) Betdaq, the match
        that comes back is what gets written to the ledger, and the settlement
        read is the adapter's own - so this is the path the
        resolved-trades record is actually fed by, not a simulation of it.
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
            receipts=[_receipt(matched_size=10.0, matched_price=2.5)],
            market_info_rows=[_market_row(555, market_status="SETTLED")],
            change_rows=[
                _change_row(555, 11, "Team A", "Winner", 41),
                _change_row(555, 12, "Team B", "Loser", 42),
            ])
        adapter = _adapter(monkeypatch, client)
        market = _market(adapter, client)

        fill = asyncio.run(adapter.place_order(
            _Opportunity(market, "YES"), max_spend_usd=10.0, max_price=0.45))
        assert fill["status"] == "paper"
        assert fill["filled_price"] == pytest.approx(0.4)

        storage = Storage(db_path=str(tmp_path / "betdaq.db"))
        try:
            storage.log_trade({
                "market_id": market.id, "market_question": market.question,
                "side": "YES", "market_price": fill["filled_price"],
                "fair_value": 0.5, "edge": 0.08, "kelly_fraction": 0.02,
                "position_size_usd": fill["filled_usd"], "confidence": 0.7,
                "status": "paper", "venue_id": "betdaq",
                "execution_mode": "paper",
                "token_price_at_entry": fill["filled_price"], "fees_usd": 0.0,
            })
            before = paper_record_progress(storage, "betdaq")
            assert before["open_trades"] == 1
            assert before["resolved"] == 0

            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 1
            assert report.stuck == 0, "a venue that CAN answer is never stuck"

            after = paper_record_progress(storage, "betdaq")
            assert after["resolved"] == 1
            assert after["open_trades"] == 0
            # 25.0 contracts (10.00 of play money at odds 2.50) redeemed at
            # 1.00 for a 10.00 stake: 15.00 profit.
            assert after["net_pnl_usd"] == pytest.approx(15.0, abs=0.05)
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

        client = _Client(
            market_info_rows=[_market_row(555, market_status="ACTIVE")])
        adapter = _adapter(monkeypatch, client)
        storage = Storage(db_path=str(tmp_path / "open.db"))
        try:
            storage.log_trade({
                "market_id": "betdaq-555", "market_question": "Team A v Team B - Match Odds",
                "side": "YES", "market_price": 0.4, "fair_value": 0.5,
                "edge": 0.08, "kelly_fraction": 0.02,
                "position_size_usd": 10.0, "confidence": 0.7,
                "status": "paper", "venue_id": "betdaq",
                "execution_mode": "paper", "token_price_at_entry": 0.4,
                "fees_usd": 0.0,
            })
            engine = SettlementEngine(storage=storage,
                                      venue_registry=_Registry(adapter))
            report = asyncio.run(engine.settle_pending())
            assert report.paper_settled == 0
            assert report.stuck == 0, (
                "\"not settled yet\" must never be filed as a venue that "
                "cannot settle")
            assert paper_record_progress(storage, "betdaq")["open_trades"] == 1
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# 9. play money is not dollars, and no panel may say it is
# ---------------------------------------------------------------------------

class TestVirtualMoneyIsNotCapital:
    def test_a_play_money_balance_is_reported_under_its_own_currency(
            self, monkeypatch):
        from src.ptai.execution.capital import is_virtual_balance

        client = _Client(order_rows=[_order_row()])
        adapter = _adapter(monkeypatch, client)
        _discover(adapter)
        portfolio = asyncio.run(adapter.get_portfolio())
        assert portfolio["virtual"] is True
        assert portfolio["paper"] is True
        assert is_virtual_balance(portfolio) is True
        assert "cannot be withdrawn" in portfolio["currency_note"]

    def test_the_capital_ledger_does_not_count_play_money_as_capital(
            self, tmp_path):
        """
        The one place a play-money figure could turn into a claim about money.

        A venue that reports a balance and no budget used to produce an account
        saying "the venue reports $5,000.00" - and for Betdaq's play money that
        sentence would be about money that cannot be withdrawn or spent. The
        figure is kept, under the currency that issued it, and the account is
        not funded by it.
        """
        from src.ptai.execution.capital import CapitalLedger
        from src.ptai.storage.db import Storage

        storage = Storage(db_path=str(tmp_path / "capital.db"))
        try:
            plan = CapitalLedger(storage=storage).build(
                mode="paper", budgets={},
                venue_labels={"betdaq": "Betdaq"},
                balances={"betdaq": {"available": True, "balance": 5000.0,
                                     "currency": "play money (virtual)",
                                     "virtual": True,
                                     "source": "betdaq_sdk"}})
            account = next(a for a in plan.accounts if a.venue_id == "betdaq")
            assert account.funded is False
            assert account.balance_is_real is False
            assert account.reported_balance_usd == 0.0
            assert account.virtual_balance == pytest.approx(5000.0)
            assert account.can_deploy_live is False
            assert not any("reports $5,000" in w for w in account.warnings)
            assert any("play money" in w for w in account.warnings)
            assert plan.total_available_usd == 0.0
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
        finally:
            storage.close()

    def test_the_budget_endpoint_refuses_play_money(self, monkeypatch, tmp_path):
        """
        Belt to the braces: even if a funding route were added for a play-money
        venue, the endpoint that authorises capital must refuse it.
        """
        import src.ptai.ui.console as console

        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        monkeypatch.setitem(console.FUNDING_ROUTES, "betdaq",
                            {"label": "Betdaq", "currency": "GBP",
                             "minimum_deposit_usd": 10,
                             "available_from": "anywhere"})

        async def _virtual_balances(agent=None, force=False):
            return {"betdaq": {"available": True, "balance": 5000.0,
                               "currency": "play money (virtual)",
                               "virtual": True, "source": "betdaq_sdk"}}

        monkeypatch.setattr(console, "_venue_balances", _virtual_balances)
        from fastapi.testclient import TestClient

        response = TestClient(console.app).post("/api/console/budget",
                                                json={"venue": "betdaq",
                                                      "amount": 10})
        assert response.status_code == 409
        assert "cannot be deposited, withdrawn or spent" in response.json()["error"]


# ---------------------------------------------------------------------------
# 10. the venue is no longer listed as having no client
# ---------------------------------------------------------------------------

class TestTheVenueIsNoLongerUnbuilt:
    def test_the_inventory_row_is_not_no_client_and_says_what_it_needs(self):
        from src.ptai.venues.inventory import build_inventory

        inventory = build_inventory(_Registry({"betdaq": BetdaqAdapter()}))
        row = inventory["venues"]["betdaq"]
        assert row["use"] != "no_client"
        # No login saved in this vault: the row says the login is what it needs.
        assert row["use"] == "needs_login"
        assert row["can_report_settlement"] is True
        assert row["fundable"] is False

    def test_with_a_saved_login_the_row_runs_today(self, monkeypatch, tmp_path):
        from src.ptai.venues.inventory import build_inventory

        saved = credential_store.save(
            "betdaq", {"username": "bettor", "password": "pw"}, str(tmp_path))
        assert saved["ok"] is True
        inventory = build_inventory(_Registry({"betdaq": BetdaqAdapter()}),
                                    data_dir=str(tmp_path))
        row = inventory["venues"]["betdaq"]
        assert row["use"] == "paper_only"
        assert row["can_run_today"] is True
        assert row["paper_tradable"] is True
        assert row["can_place_real_orders"] is False
        assert row["fundable_from_here"] is False
