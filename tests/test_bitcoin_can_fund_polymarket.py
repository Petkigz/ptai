"""
Bitcoin as a funding rail.

The operator's question, verbatim: "funding the venues is impossible in ugx of
course but what if i use bitcoin".

The honest answer is that it is not impossible, and Bitcoin is one of the ways
in. Polymarket runs a bridge that accepts deposits from several chains and
converts whatever arrives to pUSD on Polygon; Bitcoin is a supported source
chain with a $9 minimum, and that matters precisely for an operator whose local
money moves through mobile money and P2P rather than a USD card or bank wire.

What the code has to get right, and what these tests pin:

  * the rails are recorded individually, because they differ in minimum, cost
    and FAILURE MODE - "send crypto" is not an instruction;
  * the USDC rail keeps the warning that pays for itself: Polygon only, because
    a wrong network is how people lose a deposit;
  * the Bitcoin rail sends plain BTC on the Bitcoin network, and says where the
    address comes from (the operator's own Polymarket Deposit screen, never the
    agent);
  * local money is a recorded step per country, so "UGX cannot do this" is
    answered with the rails that can rather than with silence;
  * nothing here guesses a country or a currency it has no entry for.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from src.ptai.execution.capital import (
    FUNDING_ROUTES,
    LOCAL_MONEY_ENTRY,
    POLYMARKET_DEPOSIT_OPTIONS,
    deposit_options,
    local_money_entry,
    plan_for_budget,
)
from src.ptai.venues.registry import VenueRegistry

RAILS = {option["id"]: option for option in POLYMARKET_DEPOSIT_OPTIONS}


@pytest.fixture()
def console_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    return importlib.import_module("src.ptai.ui.console")


@pytest.fixture()
def console_client(console_app):
    return TestClient(console_app.app)


class TestTheRailExistsAndSaysWhatItCosts:
    def test_polymarket_takes_bitcoin_through_its_bridge(self):
        rail = RAILS["bitcoin_bridge"]
        # The documented minimum for the Bitcoin chain (bridge supported-assets,
        # 2026-09). If this number changes, the copy must change with it.
        assert rail["minimum_usd"] == 9.0
        assert rail["asset"] == "BTC"
        assert rail["network"] == "Bitcoin"
        assert "bridge" in rail["label"].lower()

    def test_the_minimum_is_explained_rather_than_stated(self):
        rail = RAILS["bitcoin_bridge"]
        assert "$9" in rail["cost"], "the cost must explain the minimum"
        # A minimum that is explained is a minimum an operator can budget for,
        # and the risk a Bitcoin deposit carries is the landing price, not a
        # chain to choose.
        assert "$9" in rail["watch_out"]
        assert "not processed" in rail["watch_out"]
        assert "price moves" in rail["watch_out"]

    def test_every_rail_states_minimum_cost_watchout_and_steps(self):
        assert len(POLYMARKET_DEPOSIT_OPTIONS) >= 2
        ids = [option["id"] for option in POLYMARKET_DEPOSIT_OPTIONS]
        assert len(ids) == len(set(ids)), "rail ids must be unique"
        for option in POLYMARKET_DEPOSIT_OPTIONS:
            assert option["minimum_usd"] > 0, option["id"]
            for field in ("label", "asset", "network", "cost", "best_for",
                          "watch_out"):
                assert str(option.get(field) or "").strip(), \
                    f"{option['id']} must record {field}"
            assert option["steps"], option["id"]

    def test_bitcoin_costs_more_to_bridge_than_usdc_on_polygon(self):
        # Follows from the docs: higher bridging cost is why the BTC minimum is
        # $9 rather than the $2 USDC on Polygon needs.
        assert RAILS["usdc_polygon"]["minimum_usd"] == 2.0
        assert RAILS["bitcoin_bridge"]["minimum_usd"] > 2.0

    def test_the_routes_the_console_reads_carry_the_rails(self):
        assert FUNDING_ROUTES["polymarket"]["deposit_options"] == \
            POLYMARKET_DEPOSIT_OPTIONS


class TestBitcoinCannotBeSentToTheWrongNetwork:
    def test_the_bitcoin_rail_never_asks_for_a_network_choice(self):
        rail = RAILS["bitcoin_bridge"]
        joined = " ".join(rail["steps"])
        assert "Bitcoin network" in joined
        # The USDC rail's whole risk is a chain choice; Bitcoin must not
        # reintroduce it by naming another chain in the instructions.
        assert "Polygon" not in joined

    def test_plain_btc_and_only_btc(self):
        rail = RAILS["bitcoin_bridge"]
        joined = " ".join(rail["steps"]).lower()
        assert "plain btc" in joined
        assert "no other coin" in joined

    def test_the_address_comes_from_the_operators_own_deposit_screen(self):
        joined = " ".join(RAILS["bitcoin_bridge"]["steps"]).lower()
        assert "deposit" in joined and "polymarket" in joined
        assert "unique to your polymarket wallet" in joined
        # The agent never holds the deposit address, and the copy must not
        # suggest it needs one.
        assert "the agent never sees it" in RAILS["usdc_polygon"]["steps"][1]

    def test_the_usdc_rail_still_warns_about_the_wrong_chain(self):
        rail = RAILS["usdc_polygon"]
        assert "Polygon and nothing else" in rail["watch_out"]
        assert "recovery.polymarket.com" in rail["watch_out"]


class TestLocalMoneyIsAStepNotAnExcuse:
    def test_ugx_is_recorded_as_a_route_and_not_a_dead_end(self):
        entry = local_money_entry("UG")
        assert entry is not None
        assert entry["currency"] == "UGX"
        joined = " ".join(entry["how"])
        assert "mobile money" in joined
        # Both practical rails, named, with escrow called out.
        assert "Yellow Card" in joined
        assert "P2P" in joined
        assert "escrow" in joined.lower()

    def test_it_says_what_is_not_available_rather_than_only_what_is(self):
        entry = local_money_entry("ug")  # case must not matter
        joined = " ".join(entry["how"])
        assert "not available" in joined
        assert "do not convert crypto" in joined

    def test_p2p_is_described_as_a_counterparty_not_a_bank(self):
        entry = local_money_entry("UG")
        assert "counterparty" in entry["watch_out"]
        assert "never release" in entry["watch_out"]

    def test_a_country_with_no_entry_gets_silence_not_a_guess(self):
        assert local_money_entry("XX") is None
        assert local_money_entry(None) is None
        assert local_money_entry("") is None

    def test_a_venue_with_one_route_reports_no_rails(self):
        # Empty means "one route recorded", and the single deposit_steps path
        # still carries it - it must not read as "no way in".
        assert deposit_options("kalshi") == []
        assert deposit_options("not_a_venue") == []
        assert FUNDING_ROUTES["kalshi"]["deposit_steps"]


class TestTheConsoleShowsIt:
    def test_the_payload_carries_the_rails_and_the_local_step(self, console_client):
        body = console_client.get("/api/console/funding?total=50").json()
        route = body["routes"]["polymarket"]
        ids = [option["id"] for option in route["deposit_options"]]
        assert "bitcoin_bridge" in ids
        assert body["local_money"]["currency"] == "UGX"
        assert body["country"] == "UG"

    def test_the_country_is_the_one_the_venue_table_uses(self, console_client):
        body = console_client.get("/api/console/funding?total=50").json()
        # VenueRegistry carries the country the eligibility verdicts are made
        # for; the funding panel must not invent a second answer.
        assert body["country"] == VenueRegistry().country_code
        assert body["country"] in LOCAL_MONEY_ENTRY

    def test_the_page_renders_the_rails_and_the_local_step(self, console_client):
        page = console_client.get("/").text
        assert "Ways to get money in." in page
        assert "railsHtml" in page
        assert "body.local_money" in page

    def test_the_rails_do_not_change_how_the_budget_is_split(self):
        # The rails answer "how does money get in"; the budget answers "how much
        # may the agent use". Adding rails must not touch the second answer.
        plan = plan_for_budget(50.0, mode="live")
        assert plan["allocations"] == {"polymarket": 50.0}
        assert plan["primary_venue"] == "polymarket"
        assert not any("bitcoin" in str(w).lower() for w in plan["warnings"])
