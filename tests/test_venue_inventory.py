"""
Every venue PTAI knows about, and what it can honestly be used for.

The operator's question was "I've seen trading venues but none of the other
venues in the UI, so I don't know if I can run them too". Two failures were
behind it:

  1. the venue panel listed only venues that could hold money or already had a
     trade history, so seventeen registered adapters were invisible - and an
     invisible adapter is indistinguishable from a broken one; and
  2. the capability flags described CREDENTIALS rather than CODE. Kalshi, Binance,
     WhiteBIT and Manifold each declared `supports_trading=bool(api_key)` while
     their `place_order` returns a dry-run stub or "not implemented" - so with an
     API key in settings they would have reported `can_place_real_orders = True`
     for a venue that cannot place an order at all.

Both are about the same thing: a venue claim has to match the code. These tests
pin the claim for every adapter that ships, so a new adapter cannot be added with
a hopeful flag, and pin the surfaces that show it.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src.ptai.storage.db import Storage
from src.ptai.venues.inventory import (
    USE_NEEDS_LOGIN,
    USE_NO_CLIENT,
    USE_PAPER_ONLY,
    USE_REAL_MONEY,
    USE_SCANNER,
    VENUE_INVENTORY_KEY,
    build_inventory,
    inventory_line,
    load_inventory,
    record_inventory,
)

REPO_VENUES = 19  # the registry the agent builds at startup

# Venues whose public feed needs no account: PTAI reads them today, at no cost,
# and paper-trades them. If one of these stops being readable the UI must stop
# saying it is - which is what this list is for.
PUBLIC_READS = {"polymarket", "kalshi", "manifold", "predictit",
                "crypto_binance", "whitebit"}


@pytest.fixture(scope="module")
def registry():
    from src.ptai.agent.v3_loop import TradingAgentV3

    agent = TradingAgentV3(country_code="UG", dry_run=True)
    return agent.venue_registry


@pytest.fixture(scope="module")
def inventory(registry):
    return build_inventory(registry)


# ---------------------------------------------------------------------------
# 1. the claim matches the code
# ---------------------------------------------------------------------------

class TestACapabilityIsAStatementAboutTheCode:
    def test_the_whole_registry_is_described(self, inventory):
        assert len(inventory["venues"]) == REPO_VENUES
        assert inventory["counts"]["registered"] == REPO_VENUES

    def test_no_venue_claims_a_submission_path_it_does_not_have(self, registry):
        """
        `real_order_path` is the honest version of "can hold real money".

        Exactly one adapter submits to a venue - Polymarket - and it is the only
        one allowed to say so. An adapter that returns a dry-run stub forever may
        read live markets and be paper-traded, and is not a trading venue.
        """
        claiming = {vid for vid, adapter in registry.adapters.items()
                    if adapter.capabilities.real_order_path}
        assert claiming == {"polymarket"}, (
            "only an adapter whose place_order really submits may claim an order "
            f"path; these claimed it: {sorted(claiming)}")

    def test_a_credential_cannot_arm_a_venue_with_no_order_path(self):
        """
        The bug this file was written for: an API key must not manufacture a
        capability. Kalshi's place_order answers "Live trading not implemented
        for Kalshi yet" however many credentials it is given.
        """
        from src.ptai.venues.kalshi_adapter import KalshiAdapter

        adapter = KalshiAdapter(api_key="a-key-that-changes-nothing")
        adapter.dry_run = False  # even fully armed
        assert not adapter.capabilities.supports_trading, (
            "an API key does not create an order path")
        assert not adapter.can_place_real_orders, (
            "a venue that cannot submit must never report as able to place a "
            "real order")

    @pytest.mark.parametrize("module_name,class_name", [
        ("src.ptai.venues.crypto_adapter", "CryptoAdapter"),
        ("src.ptai.venues.whitebit_adapter", "WhiteBITAdapter"),
        ("src.ptai.venues.manifold_adapter", "ManifoldAdapter"),
    ])
    def test_credential_holding_stub_adapters_are_not_trading_venues(
            self, module_name, class_name):
        module = importlib.import_module(module_name)
        adapter = getattr(module, class_name)()
        adapter.dry_run = False
        assert not adapter.can_place_real_orders

    def test_polymarket_is_armed_only_by_credentials_and_arming(self):
        from src.ptai.venues.polymarket_adapter import PolymarketAdapter

        bare = PolymarketAdapter()
        bare.dry_run = False
        assert not bare.can_place_real_orders, (
            "no credentials must mean no real orders")

        credentialed = PolymarketAdapter(private_key="0x" + "ab" * 32,
                                         funder="0x" + "cd" * 20)
        assert not credentialed.can_place_real_orders, (
            "credentials alone must not arm an adapter - dry_run is the gate")
        credentialed.dry_run = False
        assert credentialed.can_place_real_orders
        assert credentialed.capabilities.real_order_path


# ---------------------------------------------------------------------------
# 2. what the operator can actually run
# ---------------------------------------------------------------------------

class TestTheAnswersAnOperatorNeeds:
    def test_public_venues_read_today_with_no_account(self, inventory):
        rows = inventory["venues"]
        readable = {vid for vid, row in rows.items()
                    if row["reads_live_markets_now"]}
        assert readable == PUBLIC_READS, (
            "these venues read live public data; the panel must not hide them "
            "or claim they need credentials first")

    def test_a_venue_that_needs_a_login_says_so(self, inventory):
        """Betfair's feed is closed: no credentials means no markets at all."""
        row = inventory["venues"]["betfair"]
        assert row["use"] == USE_NEEDS_LOGIN
        assert not row["reads_live_markets_now"]
        assert not row["can_run_today"]
        assert "credentials" in row["what_it_needs"]

    def test_no_client_venues_are_named_as_such(self, inventory):
        """
        Eleven adapters have no client. They must not be presented as available
        ("run it and see") and must not be dropped from the list either - an
        operator is entitled to know a venue exists and is unbuilt.
        """
        rows = inventory["venues"]
        unbuilt = {vid: row for vid, row in rows.items()
                   if row["use"] == USE_NO_CLIENT}
        assert len(unbuilt) == inventory["counts"]["no_client"]
        assert len(unbuilt) >= 10
        for vid, row in unbuilt.items():
            assert not row["can_run_today"], f"{vid} cannot run and says it can"
            assert not row["reads_live_markets_now"]
            assert row["what_it_needs"], f"{vid} does not say what is missing"

    def test_the_scanner_is_not_dressed_as_a_venue(self, inventory):
        row = inventory["venues"]["apify"]
        assert row["use"] == USE_SCANNER
        assert not row["paper_tradable"], (
            "a read-only aggregator cannot be filled; it finds opportunities "
            "someone else priced")
        assert not row["can_place_real_orders"]

    def test_every_row_answers_the_question(self, inventory):
        for vid, row in inventory["venues"].items():
            assert row["use"] in {USE_REAL_MONEY, USE_PAPER_ONLY,
                                  USE_NEEDS_LOGIN, USE_NO_CLIENT, USE_SCANNER}
            assert row["why"], f"{vid} has no explanation"
            assert row["what_it_needs"], f"{vid} does not say what it needs"
            assert "fundable" in row
            if not row["fundable"]:
                assert row["reason_unfundable"], (
                    f"{vid} is not fundable and does not say why")

    def test_the_summary_line_counts_the_same_things(self, inventory):
        line = inventory_line(inventory)
        assert "19 venues registered" in line
        assert "6 readable now" in line
        assert "can ever: Polymarket" in line, (
            "the operator has to be able to see which venue real money can ever "
            "go into")


# ---------------------------------------------------------------------------
# 3. the record the console reads (two processes)
# ---------------------------------------------------------------------------

class TestTheConsoleCanSeeTheListWithoutTheAgentInItsProcess:
    """
    The runner starts the agent and the console as SEPARATE processes, so the
    console has no registry to read. It reads the record the agent wrote - and
    that record is written by the agent, never by a check or a screen.
    """

    def test_the_recorded_list_survives_a_fresh_process(self, tmp_path, registry):
        db = str(tmp_path / "ptai.db")
        writer = Storage(db_path=db)
        try:
            payload = record_inventory(writer, registry)
        finally:
            writer.close()

        reader = Storage(db_path=db)
        try:
            loaded = load_inventory(reader)
        finally:
            reader.close()

        assert loaded["available"]
        assert len(loaded["venues"]) == REPO_VENUES
        assert loaded["counts"] == payload["counts"]
        assert loaded["venues"]["polymarket"]["use"] == USE_REAL_MONEY

    def test_no_record_says_so_instead_of_showing_an_empty_list(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "fresh.db"))
        try:
            loaded = load_inventory(storage)
        finally:
            storage.close()

        assert loaded["available"] is False
        assert "start it once" in loaded["reason"], (
            "an empty list and an unknown list are different answers")
        assert loaded["venues"] == {}

    def test_an_unreadable_record_is_reported_not_swallowed(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "broken.db"))
        try:
            storage.set_state(VENUE_INVENTORY_KEY, "{not json")
            loaded = load_inventory(storage)
        finally:
            storage.close()
        assert loaded["available"] is False
        assert "unreadable" in loaded["reason"]

    def test_the_agent_writes_the_list_on_its_first_cycle(self):
        """
        Source pin: the write happens in the loop that HAS the registry, at the
        start of a cycle so the panel is populated before the work finishes.
        """
        from pathlib import Path

        source = Path("src/ptai/agent/v3_loop.py").read_text()
        assert "record_inventory(self.storage, self.venue_registry)" in source
        assert "inventory_line(_inventory)" in source


# ---------------------------------------------------------------------------
# 4. the surfaces
# ---------------------------------------------------------------------------

class TestThePanelShowsTheWholeRegistry:
    @pytest.fixture()
    def console_app(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
        module = importlib.import_module("src.ptai.ui.console")
        return module

    @pytest.fixture()
    def console_client(self, console_app):
        return TestClient(console_app.app)

    def test_the_venue_payload_carries_every_venue(self, console_app,
                                                   console_client, tmp_path,
                                                   registry):
        storage = Storage(db_path=str(tmp_path / "console.db"))
        try:
            record_inventory(storage, registry)
        finally:
            storage.close()

        body = console_client.get("/api/console/venue").json()
        assert len(body["inventory"]["venues"]) == REPO_VENUES
        # ...and the ranking table lists them too, instead of the two venues
        # that happened to be fundable.
        ranked = {a["venue_id"] for a in body["assessments"]}
        assert ranked == set(body["inventory"]["venues"])
        assert "binance (crypto)" not in ranked  # ids, not labels

    def test_without_a_record_the_payload_says_why_it_is_short(self,
                                                               console_client):
        body = console_client.get("/api/console/venue").json()
        inv = body["inventory"]
        assert inv["available"] is False
        assert inv["reason"]
        # The fallback still ranks what it can - and does not pretend to be the
        # whole list.
        assert body["assessments"], "the fundable venues are still shown"

    def test_the_page_renders_the_list_and_its_counts(self, console_app):
        html = console_app.CONSOLE_HTML
        assert 'id="venueAll"' in html, "no panel to render the venues into"
        assert "Every venue PTAI knows about" in html
        script = html.split("<script>")[1]
        assert "body.inventory" in script
        assert "readable right now with no account" in script
        assert "no client built" in script

    def test_the_page_does_not_carry_its_own_venue_list(self, console_app):
        """
        The classifier lives in one place. A hard-coded venue list in the page -
        or a second copy of the capability rules in the route - is how the screen
        starts disagreeing with the gate.
        """
        html = console_app.CONSOLE_HTML
        for venue_id in ("veynor", "openpx", "grvt", "pionex"):
            assert f"'{venue_id}'" not in html, (
                f"{venue_id} is classified in the page instead of the "
                "inventory module")
        # ...and it applies no capability rules of its own: every flag it reads
        # comes off the payload row (`v.use`, `v.real_order_path`), so the page
        # cannot classify a venue differently to the module that owns the rules.
        for rule in ("supports_market_discovery", "capabilities",
                     "requires_credentials", "implementation_status"):
            assert rule not in html, (
                f"the page is applying the capability rule {rule} itself")
        assert "v.use" in html and "v.real_order_path" in html


class TestTheDoctorCountsTheSameThings:
    def test_the_venues_check_reports_four_numbers(self):
        from src.ptai.doctor import _venues_check

        check = _venues_check()
        assert "19 registered" in check.detail
        assert "readable now" in check.detail
        assert "no client written" in check.detail
        assert "with an order path at all" in check.detail, (
            "'19 registered' answered none of the questions an operator has")


class TestTheCliPrintsTheSameLine:
    def test_the_operator_lines_name_the_venue_list(self, tmp_path, registry):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        storage = Storage(db_path=str(tmp_path / "cli.db"))
        try:
            record_inventory(storage, registry)
            lines = describe_snapshot(operator_snapshot(storage))
        finally:
            storage.close()

        assert lines[0].startswith("Live venue:"), (
            "the CLI's first line is a contract; the venue list comes after it")
        venue_lines = [line for line in lines if line.startswith("Venues: ")]
        assert venue_lines, "the terminal cannot answer 'what else is there'"
        assert "19 venues registered" in venue_lines[0]

    def test_a_fresh_database_does_not_invent_a_venue_list(self, tmp_path):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        storage = Storage(db_path=str(tmp_path / "fresh.db"))
        try:
            lines = describe_snapshot(operator_snapshot(storage))
        finally:
            storage.close()

        assert not [line for line in lines if line.startswith("Venues: ")], (
            "with no record there is nothing to say, and saying it anyway would "
            "be worse than silence")


# ---------------------------------------------------------------------------
# 5. the snapshot carries it (both front ends read one payload)
# ---------------------------------------------------------------------------

class TestTheSnapshotCarriesTheInventory:
    def test_the_operator_payload_has_the_venue_list(self, tmp_path, registry):
        from src.ptai.operator_view import operator_snapshot

        storage = Storage(db_path=str(tmp_path / "snap.db"))
        try:
            record_inventory(storage, registry)
            snapshot = operator_snapshot(storage)
        finally:
            storage.close()

        inv = snapshot["venues"]["inventory"]
        assert inv["available"]
        assert len(inv["venues"]) == REPO_VENUES
        # One source: the panel and the CLI read this, not their own copies.
        assert json.dumps(snapshot["venues"]["inventory"]["counts"])


# ---------------------------------------------------------------------------
# 6. the panel actually renders (the JS runs, not just the payload)
# ---------------------------------------------------------------------------

class TestThePanelRendersTheRows:
    """
    A payload the page cannot render is the same to an operator as no payload.

    The console's script is executed here in node against a stubbed DOM, with
    the real venue payload from the route, and the markup it produces is
    inspected. That catches the failure a payload test cannot: a field renamed
    in Python and still read by the page.
    """

    def test_loadVenue_draws_every_venue(self, tmp_path, registry):
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed")

        from fastapi.testclient import TestClient
        from src.ptai.ui import console as console_module

        # The payload, from the route, with a real recorded registry.
        storage = Storage(db_path=str(tmp_path / "render.db"))
        try:
            record_inventory(storage, registry)
            import os

            previous = os.environ.get("PTAI_DB")
            os.environ["PTAI_DB"] = str(tmp_path / "render.db")
            try:
                client = TestClient(console_module.app)
                payload = client.get("/api/console/venue").json()
            finally:
                if previous is None:
                    os.environ.pop("PTAI_DB", None)
                else:
                    os.environ["PTAI_DB"] = previous
        finally:
            storage.close()

        script = console_module.CONSOLE_HTML.split("<script>")[1].split("</script>")[0]
        # Drop the bootstrap tail: this harness drives one panel, not the page.
        script = script.split("loadAll();")[0]

        harness = f"""
// A DOM thin enough to run the console's own script, and nothing more. Every
// helper ($, esc, money, api) is the page's own: overriding them would test a
// different page than the one that ships.
const payload = {json.dumps(payload)};
const els = {{}};
function makeEl(id) {{
  const node = {{
    id, innerHTML: '', textContent: '', value: '', className: '', dataset: {{}},
    style: {{}},
    classList: {{toggle() {{}}, add() {{}}, remove() {{}}}},
    querySelector: () => makeEl(id + ':child'),
    getBoundingClientRect: () => ({{top: 0}}),
    scrollIntoView() {{}},
    addEventListener() {{}},
  }};
  return node;
}}
function el(id) {{ els[id] = els[id] || makeEl(id); return els[id]; }}
global.window = {{addEventListener() {{}}}};
global.document = {{
  getElementById: el,
  querySelectorAll: () => [],
  addEventListener() {{}},
  visibilityState: 'visible',
}};
{script}
// The route is the only thing replaced: the panel under test gets the real
// payload that /api/console/venue returns.
api = async () => ({{ok: true, status: 200, body: payload}});
loadVenue().then(() => {{
  const html = el('venueAll').innerHTML;
  const rows = (html.match(/<tr>/g) || []).length;
  console.log(JSON.stringify({{rows: rows, html: html}}));
}}).catch(err => {{ console.error('RENDER FAILED: ' + err.message); process.exit(2); }});
"""
        harness_path = tmp_path / "harness.js"
        harness_path.write_text(harness)
        result = subprocess.run([node, str(harness_path)], capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, result.stderr[-2000:]
        rendered = json.loads(result.stdout.strip().splitlines()[-1])

        # 19 venues + the header row.
        assert rendered["rows"] >= REPO_VENUES, (
            f"the panel drew {rendered['rows']} rows, not one per venue")
        html = rendered["html"]
        for label in ("Betfair Exchange", "Binance (crypto)", "Apify scanner",
                      "Polymarket"):
            assert label in html, f"{label} is missing from the panel"
        assert "no client built" in html and "paper + live data" in html
        assert "6" in html and "readable right now with no account" in html
