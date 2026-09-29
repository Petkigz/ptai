"""
The road to Polymarket: how far each venue is, and how many may hold money.

Two questions from the operator, and what the code has to be able to answer:

1. "lets start giving the venues enough code to reach polymarket. i dont know
   whats better all at once or one by ne."

   Reaching Polymarket is THREE layers, and they have to be built in order:
   read live markets -> read the account -> place a real order. The inventory
   scores every venue on that ladder and names the next piece of work, so the
   queue is a fact about the code rather than a judgement call. It is ordered by
   distance AND by whether money could ever reach the venue from where the
   operator is - a venue whose deposit route he cannot use is a paper exercise
   whatever is written for it.

2. "the one venue at a time was me thinking it would require more resources to
   run several venues at once but now that i can disable thinking i think its ok
   to run them."

   Correct about resources, and it was never the real constraint: PTAI already
   scans nineteen venues every cycle for the same model calls. What one venue at
   a time actually protected was CAPITAL - every live venue is a separate funded
   account. So the number is a setting (`PTAI_MAX_LIVE_VENUES`, default 1), and
   the protection that matters is kept: free cash is spent DOWN the list of live
   venues, so the same dollar is never promised to two accounts.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src.ptai.storage.db import Storage
from src.ptai.strategy.venue_selection import (
    MAX_LIVE_VENUES_CEILING,
    ROLE_LIVE,
    VenueSelector,
    max_live_venues_default,
)
from src.ptai.venues.adapter import (
    AdapterCapability,
    EligibilityStatus,
    MarketAdapter,
    VenueType,
)
from src.ptai.venues.inventory import build_inventory, inventory_line


class _Venue(MarketAdapter):
    """A venue whose reach is decided entirely by its declarations."""

    def __init__(self, venue_id: str, eligibility=EligibilityStatus.ELIGIBLE,
                 **caps):
        super().__init__(venue_id=venue_id, venue_type=VenueType.PREDICTION)
        self.capabilities = AdapterCapability(**caps)
        self.label = venue_id.title()
        self._eligibility = eligibility

    def check_eligibility(self, country_code="UG"):
        return self._eligibility

    async def discover_markets(self, target_count=100, **kwargs):
        return []

    async def get_orderbook(self, market):
        return {}

    async def get_portfolio(self):
        return {"venue_id": self.venue_id, "available": False}

    async def place_order(self, opportunity, max_spend_usd, max_price):
        return {"status": "dry_run"}


class _Registry:
    country_code = "UG"

    def __init__(self, adapters):
        self.adapters = adapters


# ---------------------------------------------------------------------------
# 1. the distance is a fact about the code, layer by layer
# ---------------------------------------------------------------------------

class TestTheDistanceToPolymarket:
    def _rows(self, **venues):
        inv = build_inventory(_Registry(venues), country_code="UG")
        return inv["venues"], inv

    def test_the_three_layers_are_counted_in_order(self):
        rows, _inv = self._rows(
            # no client at all
            unbuilt=_Venue("unbuilt", implementation_status="unimplemented",
                           implementation_note="no SDK client"),
            # reads markets, cannot read the account, cannot order
            reader=_Venue("reader", implementation_status="live",
                          supports_market_discovery=True),
            # reads and can read the account, no order path
            account=_Venue("account", implementation_status="live",
                           supports_market_discovery=True,
                           supports_portfolio=True),
            # all three
            complete=_Venue("complete", implementation_status="live",
                            supports_market_discovery=True,
                            supports_portfolio=True, real_order_path=True),
        )
        assert rows["unbuilt"]["reach"]["distance"] == 3
        assert rows["unbuilt"]["reach"]["missing"] == [
            "reads_markets", "reads_account", "places_real_orders"]
        assert rows["reader"]["reach"]["distance"] == 2
        assert rows["account"]["reach"]["distance"] == 1
        assert rows["account"]["reach"]["missing"] == ["places_real_orders"]
        assert rows["complete"]["reach"]["distance"] == 0
        assert rows["complete"]["reach"]["next_step"].startswith("nothing")

    def test_the_next_step_names_the_piece_of_work(self):
        rows, _inv = self._rows(
            unbuilt=_Venue("unbuilt", implementation_status="unimplemented",
                           implementation_note="no SDK client"),
            account=_Venue("account", implementation_status="live",
                           supports_market_discovery=True,
                           supports_portfolio=True),
        )
        assert "no SDK client" in rows["unbuilt"]["reach"]["next_step"]
        assert "order path" in rows["account"]["reach"]["next_step"]

    def test_the_queue_puts_the_closest_and_fundable_first(self):
        _rows, inv = self._rows(
            unbuilt=_Venue("unbuilt", implementation_status="unimplemented",
                           implementation_note="no SDK client"),
            account=_Venue("account", implementation_status="live",
                           supports_market_discovery=True,
                           supports_portfolio=True),
        )
        queue = [r["venue_id"] for r in inv["reach"]["next_steps"]]
        # The venue one layer away comes before the one that needs a whole client.
        assert queue.index("account") < queue.index("unbuilt")
        # ...and the summary says how many can place a real order today.
        assert inv["reach"]["real_orders"] == []
        assert "0 can place a real order" in inventory_line(inv)

    def test_a_venue_that_needs_a_login_says_so_after_the_code(self):
        rows, _inv = self._rows(
            closed=_Venue("closed", implementation_status="live",
                          supports_market_discovery=True,
                          supports_portfolio=True,
                          requires_credentials=True))
        # The code work comes first, then the login: a login unlocks nothing
        # until there is something to read and submit with.
        assert "order path" in rows["closed"]["reach"]["next_step"]
        assert "login" in rows["closed"]["reach"]["next_step"]
        assert "the this" not in rows["closed"]["reach"]["next_step"]


class TestMoneyHasToBeAbleToReachIt:
    def test_a_restricted_country_is_not_described_as_fundable(self):
        inv = build_inventory(_Registry({
            "kalshi": _Venue("kalshi", eligibility=EligibilityStatus.RESTRICTED,
                             implementation_status="live",
                             supports_market_discovery=True,
                             supports_portfolio=True),
        }), country_code="UG")
        row = inv["venues"]["kalshi"]
        assert row["eligibility"]["status"] == "restricted"
        assert row["fundable_from_here"] is False
        assert "cannot hold real money from here" in row["reach"]["next_step"]
        assert inv["counts"]["can_hold_real_money_from_here"] == 0

    def test_an_unverified_country_says_unverified_rather_than_yes(self):
        # `polymarket` has a funding route recorded in this build, so the only
        # unknown left is whether the operator's country is accepted.
        inv = build_inventory(_Registry({
            "polymarket": _Venue("polymarket",
                                 eligibility=EligibilityStatus.REQUIRES_VERIFICATION,
                                 implementation_status="live",
                                 supports_market_discovery=True),
        }), country_code="UG")
        row = inv["venues"]["polymarket"]
        assert row["fundable"] is True
        assert row["fundable_from_here"] is None
        assert "unverified" in row["reach"]["next_step"]

    def test_one_venue_that_can_really_trade_is_counted(self):
        inv = build_inventory(_Registry({
            "polymarket": _Venue("polymarket",
                                 implementation_status="live",
                                 supports_market_discovery=True,
                                 supports_portfolio=True,
                                 real_order_path=True),
        }), country_code="UG")
        # BOTH halves are required for the count: an order path AND a funding
        # route from here. Polymarket has both in this build.
        assert inv["venues"]["polymarket"]["reach"]["layers"][
            "places_real_orders"] is True
        assert inv["counts"]["can_hold_real_money_from_here"] == 1
        assert "1 can place a real order (Polymarket)" in inventory_line(inv)


# ---------------------------------------------------------------------------
# 2. more than one venue may hold capital, and free cash is not double-counted
# ---------------------------------------------------------------------------

class TestMoreThanOneVenueMayHoldCapital:
    def test_the_default_is_still_one(self, monkeypatch):
        monkeypatch.delenv("PTAI_MAX_LIVE_VENUES", raising=False)
        assert max_live_venues_default() == 1

    def test_the_setting_raises_it_and_the_ceiling_clamps_it(self, monkeypatch):
        monkeypatch.setenv("PTAI_MAX_LIVE_VENUES", "3")
        assert max_live_venues_default() == 3
        monkeypatch.setenv("PTAI_MAX_LIVE_VENUES", "99")
        assert max_live_venues_default() == MAX_LIVE_VENUES_CEILING
        # Garbage is not a licence to go unlimited; it is the default.
        monkeypatch.setenv("PTAI_MAX_LIVE_VENUES", "all of them")
        assert max_live_venues_default() == 1

    def test_the_remembered_list_survives_a_restart(self, tmp_path):
        storage = Storage(db_path=str(tmp_path / "venue.db"))
        try:
            selector = VenueSelector(storage=storage)
            selector.remember_live_venues(["polymarket", "kalshi"])
            assert selector.remembered_live_venues() == ["polymarket", "kalshi"]
            # The single key still answers, with the first venue: nothing that
            # reads the old key breaks.
            assert selector.remembered_live_venue() == "polymarket"
            # ...and an install that only has the single key still answers.
            storage.set_state(selector.LIVE_VENUES_KEY, "")
            assert selector.remembered_live_venues() == ["polymarket"]
            # Clearing the list clears the single key too.
            selector.remember_live_venues([])
            assert selector.remembered_live_venues() == []
            assert selector.remembered_live_venue() is None
            assert json.loads(storage.get_state(selector.LIVE_VENUES_KEY)) == []
        finally:
            storage.close()

    def test_free_cash_is_spent_down_the_list_not_handed_to_every_venue(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent.db"))
        monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        try:
            selector = VenueSelector(storage=agent.storage)
            selector.remember_live_venues(["venue_a", "venue_b"])
            monkeypatch.setattr(
                "src.ptai.execution.capital.authorised_budget",
                lambda storage, venue_id: 100.0)
            first, caps = agent._resolve_live_capital(
                portfolio={"venue_confirmed_balances":
                           {"venue_a": 100.0, "venue_b": 100.0}},
                free_cash=30.0)
            assert first == "venue_a"
            total = sum(c["cap_usd"] for c in caps.values())
            assert total <= 30.0, (
                "the same $30 must never be promised to two venues")
            assert caps["venue_a"]["cap_usd"] == 30.0
            assert caps["venue_b"]["cap_usd"] == 0.0
            assert "free $0.00" in caps["venue_b"]["detail"]
        finally:
            agent.storage.close()

    def test_a_second_live_venue_is_allowed_real_money(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PTAI_DB", str(tmp_path / "agent2.db"))
        monkeypatch.setenv("LM_STUDIO_HOST", "http://127.0.0.1:9")
        from src.ptai.agent.v3_loop import TradingAgentV3

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        try:
            agent._cycle_live_venues = {"venue_a", "venue_b"}
            agent._cycle_live_capital = {
                "venue_a": {"cap_usd": 10.0, "detail": "min(venue $50, authorised $10, free $10)"},
                "venue_b": {"cap_usd": 5.0, "detail": "min(venue $20, authorised $5, free $5)"},
            }

            class _Opp:
                venue_id = "venue_b"
                _is_exploration = False

            agent._adapter_is_paper = lambda opp: False
            lane, why = agent._money_lane(_Opp(), 5.0)
            assert lane == "live"
            assert "$5" in why or "5.00" in why
            assert agent._will_simulate(_Opp()) is False

            _Opp.venue_id = "venue_c"
            lane, why = agent._money_lane(_Opp(), 5.0)
            assert lane == "paper"
            assert "does not hold live capital" in why
            assert agent._will_simulate(_Opp()) is True
        finally:
            agent.storage.close()


# ---------------------------------------------------------------------------
# 3. the page shows the queue
# ---------------------------------------------------------------------------

@pytest.fixture()
def console_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console.db"))
    return importlib.import_module("src.ptai.ui.console")


@pytest.fixture()
def console_client(console_app):
    return TestClient(console_app.app)


class TestThePageShowsTheRoad:
    def test_the_queue_is_on_the_page(self, console_app, console_client):
        from src.ptai.agent.v3_loop import TradingAgentV3
        from src.ptai.venues.inventory import record_inventory

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        try:
            record_inventory(agent.storage, agent.venue_registry)
        finally:
            agent.storage.close()
        body = console_client.get("/api/console/venue").json()
        reach = body["reach"]
        assert reach["real_orders"] == ["polymarket"], (
            "exactly one venue can place a real order in this build")
        queue = [r["venue_id"] for r in reach["next_steps"]]
        assert queue and queue[0] in {r["venue_id"] for r in
                                      body["assessments"]}
        # Every venue with work left says what that work is.
        assert all(r["next_step"] for r in reach["next_steps"] if r["distance"])

    def test_the_capital_row_reads_the_setting(self, console_app, console_client,
                                               monkeypatch):
        monkeypatch.setenv("PTAI_MAX_LIVE_VENUES", "3")
        body = console_client.get("/api/console/venue").json()
        assert body["max_live_venues"] == 3
        assert body["one_live_venue_cap"] is False
        assert body["max_live_venues_setting"] == "PTAI_MAX_LIVE_VENUES"
