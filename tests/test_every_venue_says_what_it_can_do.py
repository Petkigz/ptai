"""
Every venue says what PTAI can do with it - and where a login actually goes.

The operator's report:

    "i have alot of venues but all of them wescept two are saying unavailable
     even in papaer mode which doesnt make sense unless tyhey require login but
     it seems i cant even connect my credentials to most of them"

Three separate faults sit inside that sentence, and each has its own test below:

1. NINETEEN VENUES, SEVENTEEN BADGED "UNAVAILABLE". The word was being used for
   "cannot hold real money from here" - but six of those venues were being
   scanned and paper-traded every cycle. Unavailable now means what it says: no
   client written, so there is nothing to use and nothing a login could unlock.

2. A SAVED LOGIN LOOKED LIKE IT HAD DONE NOTHING. Betfair's classification asked
   whether its adapter REQUIRES credentials, not whether they were still missing,
   so saving the login changed nothing on the panel. Whether a login is still
   needed is now a fact about the vault.

3. NO WAY TO TELL WHICH VENUES TAKE A LOGIN AT ALL. Seven forms sat beside
   nineteen venues, and "no form" read as "broken". The coverage block says, per
   venue, which of the four things is true: login saved, login needed, no login
   needed (public data), no client written - and flags the one state that is
   genuinely a defect, a venue that requires a login with no form to enter it.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from src.ptai.storage.db import Storage
from src.ptai.strategy.venue_selection import ROLE_PAPER, ROLE_UNAVAILABLE
from src.ptai.venues import credentials as credential_store
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueType,
)
from src.ptai.venues.inventory import build_inventory, record_inventory


class _Venue(MarketAdapter):
    """A venue whose row is entirely decided by its declared capability."""

    def __init__(self, venue_id: str, **caps):
        super().__init__(venue_id=venue_id, venue_type=VenueType.PREDICTION)
        self.capabilities = AdapterCapability(**caps)
        self.label = venue_id.title()

    def check_eligibility(self, country_code="UG"):
        return EligibilityStatus.ELIGIBLE

    async def discover_markets(self, target_count=100, **kwargs):
        return []

    async def get_orderbook(self, market):
        return {}

    async def get_portfolio(self):
        return {"venue_id": self.venue_id, "available": False,
                "reason": "test double"}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        return {"status": "dry_run", "reason": "test double"}


class _Registry:
    def __init__(self, adapters):
        self.adapters = adapters


@pytest.fixture()
def data_dir(tmp_path):
    return str(tmp_path)


def _open_venue(venue_id: str = "manifold") -> _Venue:
    """Reads public data, no funding route, no real order path."""
    return _Venue(venue_id, implementation_status="live",
                  supports_market_discovery=True)


def _closed_venue(venue_id: str = "betfair") -> _Venue:
    """Its public feed is closed: no login, no markets."""
    return _Venue(venue_id, implementation_status="live",
                  supports_market_discovery=True, requires_credentials=True)


def _unbuilt_venue(venue_id: str = "simmer") -> _Venue:
    return _Venue(venue_id, implementation_status="unimplemented",
                  implementation_note="no Simmer SDK client")


# ---------------------------------------------------------------------------
# 1. a saved login is not still "needed"
# ---------------------------------------------------------------------------

class TestASavedLoginCountsAsSaved:
    def test_a_closed_venue_needs_a_login_until_one_is_saved(self, data_dir):
        inv = build_inventory(_Registry({"betfair": _closed_venue()}), data_dir)
        row = inv["venues"]["betfair"]
        assert row["use"] == "needs_login"
        assert row["can_run_today"] is False
        assert row["logged_in"] is False
        assert row["login"]["tool"] == "betfair"
        assert inv["counts"]["need_credentials"] == 1

    def test_saving_the_login_makes_the_venue_run(self, data_dir):
        saved = credential_store.save(
            "betfair",
            {"username": "op@example.com", "password": "hunter2x",
             "app_key": "APPKEY1234567890"},
            data_dir)
        assert saved["ok"] is True, saved.get("error")

        inv = build_inventory(_Registry({"betfair": _closed_venue()}), data_dir)
        row = inv["venues"]["betfair"]
        assert row["logged_in"] is True
        # ...and it is no longer described as needing one.
        assert row["use"] == "paper_only"
        assert row["needs_credentials"] is False
        assert row["can_run_today"] is True
        assert row["reads_live_markets_now"] is True
        assert "saved login" in row["what_it_needs"]
        assert inv["counts"]["need_credentials"] == 0
        assert inv["counts"]["logins_configured"] == 1

    def test_a_caller_with_no_vault_claims_nothing_about_the_login(self):
        """
        The classification without a data directory names the login and claims
        nothing about it - it must not report "saved" for a vault it never read.
        """
        inv = build_inventory(_Registry({"betfair": _closed_venue()}))
        row = inv["venues"]["betfair"]
        assert row["login"]["tool"] == "betfair"
        assert row["logged_in"] is False
        assert row["use"] == "needs_login"


# ---------------------------------------------------------------------------
# 2. the coverage block: where a login goes, and where it cannot
# ---------------------------------------------------------------------------

class TestTheLoginCoverageBlock:
    def _inventory(self, data_dir):
        return build_inventory(_Registry({
            "manifold": _open_venue(),          # public data, no login
            "betfair": _closed_venue(),         # closed without a login
            "simmer": _unbuilt_venue(),         # no client at all
            "polymarket": _Venue(               # reads now, login for real orders
                "polymarket", implementation_status="live",
                supports_market_discovery=True, real_order_path=True),
        }), data_dir)["venues"]

    def test_every_venue_appears_exactly_once(self, data_dir):
        cov = credential_store.coverage(self._inventory(data_dir), data_dir)
        assert cov["counts"]["registered"] == 4
        assert len(cov["venues"]) == 4
        counted = sum(v for k, v in cov["counts"].items() if k != "registered")
        assert counted == 4, "every venue must land in exactly one bucket"

    def test_the_four_states_are_told_apart(self, data_dir):
        cov = credential_store.coverage(self._inventory(data_dir), data_dir)
        state = {r["venue_id"]: r["state"] for r in cov["venues"]}
        assert state["manifold"] == "no_login_needed"
        assert state["betfair"] == "login_required"
        assert state["simmer"] == "no_client"
        assert state["polymarket"] == "login_available"
        # The venue with no client says a login would not be used, rather than
        # leaving the operator to wonder whether they typed it in wrong.
        simmer = next(r for r in cov["venues"] if r["venue_id"] == "simmer")
        assert "no client" in simmer["why"]
        betfair = next(r for r in cov["venues"] if r["venue_id"] == "betfair")
        assert "save the" in betfair["why"]

    def test_a_venue_that_needs_a_login_with_no_form_is_a_defect(self, data_dir,
                                                                 monkeypatch):
        monkeypatch.setattr(credential_store, "TOOL_FOR_VENUE", {})
        cov = credential_store.coverage(self._inventory(data_dir), data_dir)
        assert cov["counts"]["missing_form"] == 1
        row = next(r for r in cov["venues"] if r["state"] == "missing_form")
        assert row["venue_id"] == "betfair"
        assert "defect" in row["why"]


# ---------------------------------------------------------------------------
# 3. the console: no venue is called unusable while it is being worked
# ---------------------------------------------------------------------------

@pytest.fixture()
def console_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    module = importlib.import_module("src.ptai.ui.console")
    return module


@pytest.fixture()
def console_client(console_app):
    return TestClient(console_app.app)


def _record(console_app):
    """The nineteen-venue registry the agent actually builds, recorded."""
    from src.ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    try:
        return record_inventory(agent.storage, agent.venue_registry)
    finally:
        agent.storage.close()


class TestNoVenueIsCalledUnusableWhileItIsBeingWorked:
    def test_only_venues_with_no_client_read_as_unavailable(self, console_app,
                                                            console_client):
        inventory = _record(console_app)
        body = console_client.get("/api/console/venue").json()
        rows = {a["venue_id"]: a for a in body["assessments"]}
        assert len(rows) == 19, "every registered venue must be on the page"

        unbuilt = {vid for vid, row in inventory["venues"].items()
                   if row["use"] == "no_client"}
        for venue_id, a in rows.items():
            if a["role"] == ROLE_UNAVAILABLE:
                assert venue_id in unbuilt, (
                    f"{venue_id} is badged unavailable but its client exists")
            else:
                assert venue_id not in unbuilt

    def test_a_paper_traded_venue_says_so_and_says_why_it_cannot_hold_money(
            self, console_app, console_client):
        _record(console_app)
        body = console_client.get("/api/console/venue").json()
        rows = {a["venue_id"]: a for a in body["assessments"]}
        # Manifold and crypto_binance have no funding route in this build and are
        # read every cycle; they were two of the seventeen "unavailable". Neither
        # has an order path, so both are still paper-only.
        for venue_id in ("manifold", "crypto_binance"):
            a = rows[venue_id]
            assert a["role"] == ROLE_PAPER
            assert a["blockers"], "the money truth must be stated, not implied"
            assert a["use"] == "paper_only"
            assert a["can_run_today"] is True

        # Kalshi gained a REAL order path (the RSA-signed V2 order endpoint), so
        # "paper_only" is no longer true of it - and the row still has to state
        # the money truth, because an order path is not a funding route. From
        # Uganda the venue is restricted and its rail is a US bank account.
        kalshi = rows["kalshi"]
        assert kalshi["use"] == "real_money"
        assert kalshi["blockers"], "an order path must not read as 'funded'"

    def test_a_venue_with_an_order_path_still_says_money_cannot_reach_it(
            self, console_app, console_client):
        """
        The new state has to be visible, not implied: Kalshi has all three code
        layers and still cannot hold this operator's money from Uganda.
        """
        _record(console_app)
        body = console_client.get("/api/console/venue").json()
        kalshi = body["inventory"]["venues"]["kalshi"]
        assert kalshi["use"] == "real_money"
        assert kalshi["real_order_path"] is True
        assert kalshi["fundable_from_here"] is False, (
            "an order path must not read as a fundable venue")
        assert kalshi["eligibility"]["status"] == "restricted"
        # And the row says what is left, rather than "nothing".
        assert kalshi["reach"]["distance"] == 0
        assert "cannot hold real money from here" in kalshi["reach"]["next_step"]

    def test_an_order_path_and_a_closed_feed_are_two_different_facts(
            self, console_app, console_client):
        """
        Betfair gained a real submission path while its market feed stays closed.
        Both facts belong on the row: "it can place a real order" must not read as
        "it is running now", and a venue that returns nothing until a login is
        saved must not be described as paper-traded every cycle.
        """
        _record(console_app)
        body = console_client.get("/api/console/venue").json()
        betfair = body["inventory"]["venues"]["betfair"]
        assert betfair["use"] == "real_money"
        assert betfair["real_order_path"] is True
        assert betfair["reads_live_markets_now"] is False
        assert betfair["needs_credentials"] is True
        assert betfair["can_run_today"] is False
        assert betfair["fundable_from_here"] is False
        assert "no markets until the login" in betfair["why"]
        assert "cannot hold real money from here" in betfair["reach"]["next_step"]

    def test_the_page_says_real_money_is_the_only_thing_that_is_missing(
            self, console_app, console_client):
        _record(console_app)
        body = console_client.get("/api/console/venue").json()
        text = " ".join(body["selection"]["reasons"])
        assert "Cannot be used at all yet" in text
        assert "paper-traded where the venue can hold a position" in text

    def test_the_logins_panel_says_which_venues_take_a_login(self, console_app,
                                                             console_client):
        _record(console_app)
        body = console_client.get("/api/console/logins").json()
        cov = body["coverage"]
        assert cov["counts"]["registered"] == 19
        assert cov["counts"]["missing_form"] == 0, (
            "a venue that requires a login with no form is a defect")
        betfair = next(r for r in cov["venues"] if r["venue_id"] == "betfair")
        assert betfair["state"] in ("login_required", "login_saved")
        assert betfair["tool"] == "betfair"


    def test_a_play_money_venue_can_price_and_settle_without_holding_money(
            self, console_app, console_client):
        """
        Manifold's depth is derived from its own AMM pool and its resolutions come
        back YES/NO, so a paper trade there is evidence rather than an estimate.
        Two things must still hold on the row: it must not read as a venue that
        can hold the operator's money or submit an order, and its optional Mana
        key must not read as a missing login - nor be hidden, because the account
        read uses it.
        """
        _record(console_app)
        body = console_client.get("/api/console/venue").json()
        manifold = body["inventory"]["venues"]["manifold"]
        assert manifold["use"] == "paper_only"
        assert manifold["real_order_path"] is False
        assert manifold["paper_tradable"] is True
        assert manifold["reads_live_markets_now"] is True
        assert manifold["can_run_today"] is True
        assert manifold["fundable_from_here"] is False
        assert manifold["reach"]["layers"]["reads_markets"] is True
        assert manifold["reach"]["layers"]["places_real_orders"] is False
        assert manifold["reason_unfundable"].startswith("play-money")
        assert "optional" in manifold["what_it_needs"]
        assert "account read" in manifold["what_it_needs"]

        cov = console_client.get("/api/console/logins").json()["coverage"]
        row = next(r for r in cov["venues"] if r["venue_id"] == "manifold")
        assert row["state"] == "no_login_needed", (
            "an optional key on a public venue is not a missing login")
        assert row["tool"] == "manifold"
        assert cov["counts"]["missing_form"] == 0
