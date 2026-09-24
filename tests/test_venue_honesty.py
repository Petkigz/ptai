"""
Venue adapter honesty: which venues actually work, and which only say they do.

Thirteen adapters advertised `supports_market_discovery=True` and returned
hardcoded rows. One of them, PolymarketAdapter, took whatever the scanner gave
it - including the scanner's own mock fallback - and stamped
`data_mode=LIVE`, `is_mock=False`, `"safety": "LIVE_DATA - executable"` on top.
The execution guard checks exactly those two fields to refuse mock data, so the
relabelling defeated it.

These tests pin the contract: an adapter with no client returns nothing and
declares itself unimplemented, and an adapter that reports live data has
actually verified it is live.
"""
import asyncio

import pytest

from src.ptai.markets.base import Market, MarketSource, Token, DataMode
from src.ptai.venues.adapter import (MarketAdapter, STATUS_LIVE, STATUS_SCANNER,
                                     STATUS_UNIMPLEMENTED, UnimplementedVenueAdapter,
                                     VenueType)

# Modules that must declare themselves unimplemented, and the class name.
# (betfair_adapter.BetfairAdapter is the legacy stub, superseded by the real
# exchange adapter in betfair_exchange.py.)
UNIMPLEMENTED = [
    ("apify_adapter", "ApifyAdapter"),
    ("afx_adapter", "AFXAdapter"),
    ("cymetica_adapter", "CymeticaAdapter"),
    ("grvt_adapter", "GRVTAdapter"),
    ("pionex_adapter", "PionexAdapter"),
    ("simmer_adapter", "SimmerAdapter"),
    ("veynor_adapter", "VeynorAdapter"),
    ("openpx_adapter", "OpenPXAdapter"),
    ("ccxt_adapter", "CCXTUnifiedAdapter"),
    ("stock_adapter", "StockAdapter"),
    ("betfair_adapter", "BetfairAdapter"),
    ("betfair_adapter", "BetdaqAdapter"),
    ("betfair_adapter", "BetConnectAdapter"),
]


def _instantiate(module_name, class_name):
    mod = __import__(f"src.ptai.venues.{module_name}", fromlist=[class_name])
    return getattr(mod, class_name)()


def make_market(**over):
    kw = dict(id="m1", source=MarketSource.POLYMARKET, question="Will X happen?",
              outcomes=["YES", "NO"], outcome_prices=[0.6, 0.4],
              tokens=[Token(token_id="t1", outcome="YES", price=0.6)])
    kw.update(over)
    return Market(**kw)


# ---------------------------------------------------------------------------
# The declaration itself
# ---------------------------------------------------------------------------

class TestImplementationStatus:
    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_declares_unimplemented_or_scanner(self, module, cls):
        a = _instantiate(module, cls)
        assert a.capabilities.implementation_status in (STATUS_UNIMPLEMENTED, STATUS_SCANNER)

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_does_not_claim_discovery(self, module, cls):
        a = _instantiate(module, cls)
        assert a.capabilities.supports_market_discovery is False, \
            f"{a.venue_id} claims discovery with no client"

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_is_not_implemented_per_the_capability(self, module, cls):
        a = _instantiate(module, cls)
        # scanner is a legitimate read-only status; unimplemented is not
        if a.capabilities.implementation_status == STATUS_UNIMPLEMENTED:
            assert a.capabilities.is_implemented is False

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_note_explains_what_is_missing(self, module, cls):
        a = _instantiate(module, cls)
        assert a.capabilities.implementation_note, f"{a.venue_id} has no explanation"

    def test_live_status_is_the_default_for_a_real_adapter(self):
        """A plain MarketAdapter is assumed real; stubs must opt out."""
        class Real(MarketAdapter):
            def __init__(self):
                super().__init__(venue_id="real", venue_type=VenueType.PREDICTION)
            def check_eligibility(self, country_code="UG"):
                from src.ptai.venues.adapter import EligibilityStatus
                return EligibilityStatus.ELIGIBLE
            async def discover_markets(self, target_count=500, filters=None):
                return []
            async def get_orderbook(self, market):
                return {}
            async def get_portfolio(self):
                return {}
            async def place_order(self, opportunity, max_spend_usd, max_price):
                return {}

        assert Real().capabilities.implementation_status == STATUS_LIVE
        assert Real().capabilities.is_implemented is True


# ---------------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------------

class TestUnimplementedBehaviour:
    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_returns_no_markets(self, module, cls):
        a = _instantiate(module, cls)
        assert asyncio.run(a.discover_markets(target_count=50)) == []

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_eligibility_is_unknown_not_eligible(self, module, cls):
        """
        Claiming ELIGIBLE asserts the venue can be traded on. An adapter with
        no client cannot be traded on by definition.
        """
        a = _instantiate(module, cls)
        assert a.check_eligibility("UG").value == "unknown"

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_orderbook_reports_unavailable_not_empty(self, module, cls):
        """An absent orderbook must not look like an empty one."""
        a = _instantiate(module, cls)
        ob = asyncio.run(a.get_orderbook(make_market()))
        assert ob.get("available") is False
        assert ob.get("reason")

    @pytest.mark.parametrize("module,cls", UNIMPLEMENTED)
    def test_place_order_refuses(self, module, cls):
        a = _instantiate(module, cls)
        res = asyncio.run(a.place_order(None, max_spend_usd=10.0, max_price=0.5))
        assert res["success"] is False
        assert res["status"] == "unimplemented"
        assert res["error"]

    def test_portfolio_does_not_report_a_zero_balance_as_real(self):
        """A fabricated 0 balance is indistinguishable from an empty account."""
        res = asyncio.run(_instantiate("apify_adapter", "ApifyAdapter").get_portfolio())
        assert res["available"] is False
        assert res["balance"] is None


# ---------------------------------------------------------------------------
# The Polymarket relabelling bug
# ---------------------------------------------------------------------------

class TestNoProvenanceLaundering:
    def test_fabricated_market_is_dropped_not_relabelled(self):
        """
        The scanner generates mock markets on failure with is_mock=True. The
        adapter used to overwrite that with data_mode=LIVE and is_mock=False,
        defeating the execution guard.
        """
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        fake = make_market(id="MOCK-mock_0", question="Mock question",
                           data_mode=DataMode.MOCK, is_mock=True)
        assert PolymarketAdapter._looks_fabricated(fake) is True

    def test_detects_fabrication_by_each_marker(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        assert PolymarketAdapter._looks_fabricated(make_market(is_mock=True)) is True
        assert PolymarketAdapter._looks_fabricated(
            make_market(data_mode=DataMode.MOCK)) is True
        assert PolymarketAdapter._looks_fabricated(
            make_market(id="POLY-MOCK-123")) is True
        assert PolymarketAdapter._looks_fabricated(
            make_market(raw={"mock": True})) is True
        assert PolymarketAdapter._looks_fabricated(
            make_market(venue_id="mock")) is True

    def test_a_real_market_is_not_flagged(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        real = make_market(id="0xabc123", data_mode=DataMode.LIVE, is_mock=False)
        assert PolymarketAdapter._looks_fabricated(real) is False

    @pytest.mark.asyncio
    async def test_adapter_asks_the_scanner_for_no_mock(self, monkeypatch):
        """allow_mock=False means the scanner cannot inject fabricated rows."""
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        captured = {}

        def fake_scan(target_count=None, order_by=None, allow_mock=True, use_registry=True):
            captured["allow_mock"] = allow_mock
            return []

        adapter = PolymarketAdapter()
        monkeypatch.setattr(adapter, "_scanner", type("S", (), {"scan": staticmethod(fake_scan)})())
        await adapter.discover_markets(target_count=5)
        assert captured["allow_mock"] is False

    @pytest.mark.asyncio
    async def test_adapter_drops_fabricated_markets_from_the_scanner(self, monkeypatch):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        fake = make_market(id="MOCK-mock_0", is_mock=True, data_mode=DataMode.MOCK)
        real = make_market(id="0xreal", liquidity=50000.0, volume_24h=90000.0,
                           data_mode=DataMode.LIVE, is_mock=False)

        def fake_scan(target_count=None, order_by=None, allow_mock=True, use_registry=True):
            return [fake, real]

        adapter = PolymarketAdapter()
        monkeypatch.setattr(adapter, "_scanner", type("S", (), {"scan": staticmethod(fake_scan)})())
        got = await adapter.discover_markets(target_count=5)
        assert [m.id for m in got] == ["0xreal"]


# ---------------------------------------------------------------------------
# The real Betfair exchange adapter
# ---------------------------------------------------------------------------

class TestBetfairExchangeAdapter:
    def test_is_live_not_a_stub(self):
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        a = BetfairExchangeAdapter()
        assert a.venue_id == "betfair"
        assert a.capabilities.implementation_status == STATUS_LIVE

    def test_legacy_stub_does_not_shadow_it(self):
        """
        The stub registered itself under venue_id "betfair", so the registry was
        handed the stub while the working adapter sat unused. That is how the
        feed for goals, corners, cards and props went unconnected.
        """
        from src.ptai.venues.betfair_adapter import BetfairAdapter
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        assert BetfairAdapter().venue_id != BetfairExchangeAdapter().venue_id

    def test_unconfigured_returns_nothing_and_says_why(self):
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        a = BetfairExchangeAdapter()
        assert a.configured is False
        assert asyncio.run(a.discover_markets()) == []
        assert "credentials not configured" in a.last_error.lower()

    def test_unconfigured_is_not_eligible(self):
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        a = BetfairExchangeAdapter()
        assert a.check_eligibility("UG").value == "requires_verification"

    def test_restricted_jurisdiction_is_refused(self):
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        a = BetfairExchangeAdapter(username="u", password="p", app_key="k")
        assert a.check_eligibility("US").value == "restricted"

    def test_order_placement_is_declared_unimplemented(self):
        """
        Market data is real; execution is not. A half-built order path would
        place bets the risk layer never sized.
        """
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        a = BetfairExchangeAdapter()
        assert a.capabilities.supports_trading is False
        assert a.capabilities.supports_orderbook is True
        res = asyncio.run(a.place_order(None, max_spend_usd=10.0, max_price=0.5))
        assert res["success"] is False
        assert res["status"] == "unimplemented"

    def test_unknown_sport_is_reported_not_guessed(self):
        from src.ptai.venues.betfair_exchange import (BETFAIR_EVENT_TYPE_IDS,
                                                     BetfairExchangeAdapter)
        assert "football" in BETFAIR_EVENT_TYPE_IDS
        assert "basketball" in BETFAIR_EVENT_TYPE_IDS
        assert "table_tennis" not in BETFAIR_EVENT_TYPE_IDS


# ---------------------------------------------------------------------------
# Registry report
# ---------------------------------------------------------------------------

class TestCapabilityReport:
    def _registry(self):
        from src.ptai.venues.registry import VenueRegistry
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        from src.ptai.venues.betfair_exchange import BetfairExchangeAdapter
        from src.ptai.venues.apify_adapter import ApifyAdapter
        reg = VenueRegistry("UG")
        reg.register(PolymarketAdapter())
        reg.register(BetfairExchangeAdapter())
        reg.register(ApifyAdapter())
        return reg.capability_report()

    def test_separates_live_from_declared(self):
        rep = self._registry()
        assert "polymarket" in rep["live"]
        assert "betfair" in rep["live"]
        assert "apify" not in rep["live"]

    def test_counts_by_status(self):
        rep = self._registry()
        assert rep["by_status"].get(STATUS_LIVE, 0) == 2
        assert rep["by_status"].get(STATUS_SCANNER, 0) == 1

    def test_nothing_claims_to_be_tradeable(self):
        """No adapter here places orders, so the list must be empty."""
        assert self._registry()["tradeable"] == []

    def test_every_venue_reports_its_supports_flags(self):
        for v in self._registry()["venues"]:
            assert set(v["supports"]) == {"market_discovery", "orderbook",
                                          "trading", "portfolio"}

    def test_unimplemented_venues_have_a_note(self):
        rep = self._registry()
        for v in rep["venues"]:
            if v["implementation_status"] == STATUS_UNIMPLEMENTED:
                assert v["note"]

    def test_report_explains_how_to_read_it(self):
        assert "implemented" in self._registry()["how_to_read"].lower()

    def test_registry_excludes_stubs_from_eligibility(self):
        from src.ptai.venues.registry import VenueRegistry
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter
        from src.ptai.venues.apify_adapter import ApifyAdapter
        reg = VenueRegistry("UG")
        reg.register(PolymarketAdapter())
        reg.register(ApifyAdapter())
        reg.registered = None  # guard against accidental reliance on ordering
        eligible = [a.venue_id for a in reg.get_eligible_adapters()]
        assert "apify" not in eligible


# ---------------------------------------------------------------------------
# No fabrication left anywhere in the package
# ---------------------------------------------------------------------------

class TestNoFabricationRemains:
    def test_no_fabricated_id_markers_in_venues(self):
        """
        Scans code, not prose.

        The docstrings deliberately describe the fabrication that was removed,
        so the markers legitimately appear in documentation. A string literal
        that would build a market id is the thing that must not appear.
        """
        import ast
        import pathlib

        venues = pathlib.Path(__file__).resolve().parents[1] / "src" / "ptai" / "venues"
        markers = ("-MOCK-", "mock_qs", "mock_arbs", "mock_questions")
        offenders = []

        for path in venues.glob("*.py"):
            tree = ast.parse(path.read_text())

            # Every docstring node, so documentation can be excluded.
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node, clean=False)
                    if doc is not None:
                        docstrings.add(doc)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if node.value in docstrings:
                    continue
                if any(m in node.value for m in markers):
                    offenders.append(f"{path.name}:{node.value[:50]!r}")

        assert offenders == [], f"fabricated market markers remain: {offenders}"


    def test_manifold_returns_nothing_when_offline(self):
        """Its mock fallback fabricated up to 150 questions on any failure."""
        from src.ptai.venues.manifold_adapter import ManifoldAdapter
        a = ManifoldAdapter()
        got = asyncio.run(a.discover_markets(target_count=5))
        assert all(not m.is_mock for m in got)
        assert all("MOCK" not in str(m.id).upper() for m in got)

    def test_predictit_has_no_mock_path(self):
        """It fabricated 2024 election questions, in 2026."""
        import inspect
        from src.ptai.venues import predictit_adapter
        src = inspect.getsource(predictit_adapter)
        assert "_mock_markets" not in src
        assert "using mock" not in src.lower()

    def test_kalshi_client_has_no_mock_path(self):
        import inspect
        from src.ptai.markets import kalshi
        src = inspect.getsource(kalshi)
        assert "_mock_markets" not in src
        assert "KALSHI-MOCK" not in src
