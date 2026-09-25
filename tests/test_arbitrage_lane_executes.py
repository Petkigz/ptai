"""
The arbitrage lane EXECUTES.

`execute_arbitrage_pair` was written, tested and unreachable: the loop found
arbitrage, reported it in the cycle report, and never traded any. The strategy
with the cleanest economics in the system - a matched pair pays $1 whoever wins -
was the one wired to nothing, so its state machine, its price recheck and its
hedge were bounding an exposure window that never opened.

These tests run a real cycle with a pair injected into the scan result and assert
on what the agent DID: orders reaching both venues, positions in the ledger,
outcome rows for the learning chain, and a cycle report that says which gate
refused a pair that was refused. Only the discovery of the pair and the venue
adapters are stubbed; everything between them is production code.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from src.ptai.markets.base import Market, MarketSource

VENUE_A = "polymarket"
VENUE_B = "kalshi"

_SOURCE_BY_VENUE = {"kalshi": MarketSource.KALSHI,
                    "predictit": MarketSource.PREDICTIT}


def _market(market_id: str, price: float, venue: str) -> Market:
    """A market whose SOURCE agrees with the venue it is on.

    The executor aborts an order whose adapter venue and market source disagree,
    which is a safety property - so a fixture that ignored it would be testing
    the abort path instead of the lane.
    """
    return Market(
        id=market_id,
        source=_SOURCE_BY_VENUE.get(venue, MarketSource.POLYMARKET),
        question=f"Will {market_id} happen?",
        outcomes=["YES", "NO"], outcome_prices=[price, 1 - price],
        tokens=[], volume=200_000.0, volume_24h=100_000.0, liquidity=50_000.0,
        active=True, closed=False, slug=market_id,
        raw={"venue": venue, "venue_type": "prediction"},
    )


@dataclass
class _Arb:
    """The shape the executor reads off an arbitrage opportunity."""
    venue_a: str
    venue_b: str
    market_a: Market
    market_b: Market
    price_a: float
    price_b: float
    spread: float
    fee_adjusted_profit: float
    confidence_same_event: float
    should_trade: bool = True
    estimated_profit_pct: float = 6.0


def _pair(venue_a=VENUE_A, venue_b=VENUE_B, should_trade=True) -> _Arb:
    """
    YES at A for 0.44, NO at B for 0.50.

    One share of each costs 0.94 and pays $1, so the pair is worth 6% before
    costs - the shape the arb state machine exists to execute.
    """
    return _Arb(
        venue_a=venue_a, venue_b=venue_b,
        market_a=_market("PAIR-A", 0.44, venue_a),
        market_b=_market("PAIR-B", 0.50, venue_b),
        price_a=0.44, price_b=0.50, spread=0.05, fee_adjusted_profit=0.05,
        confidence_same_event=0.9, should_trade=should_trade,
    )


def _inject_pair(agent, monkeypatch, arb) -> None:
    """Feed the chain one arbitrage pair, exactly as the scan would."""
    from src.ptai.strategy.strategy_engine import MultiVenueScanResult

    result = MultiVenueScanResult(
        total_scanned=0, total_after_cheap=0, total_after_liquidity=0,
        total_candidates=0, total_tradeable=0,
        venue_reports=[], all_opportunities=[], final_selected=[],
        best_opportunity=None, reasoning="injected arbitrage pair",
        arbitrage_opportunities=[arb],
    )

    async def scan_all_venues(self, markets_by_venue, context_provider=None,
                              max_final_trades=3):
        return result

    monkeypatch.setattr(type(agent.strategy_engine_v3), "scan_all_venues",
                        scan_all_venues, raising=True)


def _paper_venue(venue_id: str, yes_price: float, ask_size: str = "5000"):
    """
    A dry-run venue that SIMULATES its fill, the way the real adapters do.

    Not a stub that reports success and not a refusal that reports nothing: the
    PaperBroker walks a real ladder at the limit the executor set, so the fill is
    the size the book would have given and the position recorded from it is a
    paper position. That is the established paper idiom in this suite, and it
    keeps the arb lane's paper evidence as good as the single path's.

    TWO books, because a venue has two and they are not the same object. The one
    `get_orderbook` returns is the YES token's book - the real adapter reads
    `market.yes_token_id` - and a NO order is the OTHER token, which buys at
    `1 - the YES bid`. A stub that quoted the NO side's price from a YES-style
    book would fill orders at a price the venue would never show, which is
    exactly the sort of fixture that makes a wrong executor look right.
    """
    from src.ptai.execution.paper_broker import PaperBroker
    from src.ptai.markets.mechanics import MarketMechanics

    import tests.test_full_cycle_from_discovery_to_allocation as base

    stub = base.StubVenue(dry_run=True, venue_id=venue_id)
    stub.order_log: List[Dict[str, Any]] = []
    # A two-cent spread around the YES price, the same convention the sizing
    # tests use: NO costs 1 - the YES bid, and a YES buy pays the ask.
    yes_bid = yes_price
    yes_ask = yes_price + 0.02
    yes_book = {"bids": [{"price": f"{yes_bid:.2f}", "size": "5000"}],
                "asks": [{"price": f"{yes_ask:.2f}", "size": ask_size}],
                "spread": round(yes_ask - yes_bid, 4), "is_real": True,
                "source": "stub_book"}

    async def get_orderbook(market):
        return dict(yes_book)

    async def place_order(opportunity, max_spend_usd, max_price):
        stub.order_log.append({
            "market_id": opportunity.market.id,
            "side": opportunity.side,
            "max_spend_usd": max_spend_usd,
            "max_price": max_price})
        stub.orders_placed += 1
        side = str(opportunity.side).upper()
        # The token actually being bought: YES at its ask, NO at 1 - YES bid.
        token_ask = yes_ask if side == "YES" else round(1.0 - yes_bid, 6)
        side_book = {"bids": [{"price": f"{token_ask - 0.01:.2f}", "size": "5000"}],
                     "asks": [{"price": f"{token_ask:.2f}", "size": ask_size}]}
        mechanics = MarketMechanics(tick_size="0.01", min_order_size=1.0,
                                    source="clob_market_info", is_real=True)
        fill = PaperBroker(taker_fee_rate=0.0).simulate(
            side_book, "BUY", float(max_spend_usd), limit_price=float(max_price),
            mechanics=mechanics, book_source="orderbook")
        result = {
            "status": "paper", "is_real": False, "simulated": True,
            "venue_id": venue_id, "market_id": opportunity.market.id,
            "simulated_filled_usd": fill.filled_usd,
            "filled_usd": fill.filled_usd,
            "filled_price": fill.avg_price,
            "price": fill.avg_price or float(max_price),
            "paper_fill": fill.to_dict(),
        }
        # The adapter's paper contract (V29): when the book cannot supply the
        # order, report the remainder the same way a live partial does, so the
        # lane can see the unfilled shares and cancel them after the hedge.
        if fill.unfilled_usd > 1e-9:
            signed = fill.filled_shares + (
                fill.unfilled_usd / float(max_price)
                if fill.unfilled_usd > 0 and max_price else 0.0)
            result["size_matched"] = fill.filled_shares
            result["original_size"] = signed
            result["resting_usd"] = fill.unfilled_usd
            result["resting_limit_price"] = float(max_price)
        return result

    stub.get_orderbook = get_orderbook
    stub.place_order = place_order
    return stub


def _live_venue(venue_id: str):
    """
    A venue that CAN place real orders, and fills the request at its limit.

    Used for the pair that must NOT be sent. With the operator's capital on one
    venue, a cross-venue pair has a leg that would move real money on a venue
    that is not authorised - so the lane has to refuse it rather than fund it.
    """
    import tests.test_full_cycle_from_discovery_to_allocation as base

    stub = base.StubVenue(dry_run=False, venue_id=venue_id)
    stub.order_log: List[Dict[str, Any]] = []
    # `can_place_real_orders` is a property derived from dry_run and the
    # credentials the base stub sets, so it is not assigned here: an armed
    # adapter is what the live-capital boundary has to see, and faking it would
    # be testing the fake.
    assert stub.can_place_real_orders, (
        "the fixture must be a venue that CAN move real money, or the "
        "one-venue rule it is testing never engages")

    async def place_order(opportunity, max_spend_usd, max_price):
        stub.order_log.append({
            "market_id": opportunity.market.id,
            "side": opportunity.side,
            "max_spend_usd": max_spend_usd,
            "max_price": max_price})
        stub.orders_placed += 1
        return {"status": "matched",
                "orderID": f"{venue_id}-{len(stub.order_log)}",
                "filled_usd": max_spend_usd, "filled_price": max_price}

    stub.place_order = place_order
    return stub


def _build(tmp_path, monkeypatch, arb=None, live=False, ask_sizes=None):
    """
    A real agent on a fresh database, with only the venue ADAPTERS stubbed.

    Both legs need a venue, so a stub is instantiated once per venue: paper
    venues that simulate their fills, or live-capable ones that fill the
    request, depending on what the test is proving. Everything else in the cycle
    is production code - the executor, the state machine, the recorder, the
    ledger and the learning chain.
    """
    import tests.test_full_cycle_from_discovery_to_allocation as base

    agent, _ = base.build_agent(tmp_path, dry_run=False)
    pair = arb or _pair()
    # The venue's YES price: leg A buys the YES token at A's ask, leg B buys the
    # NO token, which costs 1 - B's YES bid.
    prices = {VENUE_A: pair.price_a, VENUE_B: pair.price_b}
    venues = {}
    adapters = []
    ask_sizes = ask_sizes or {}
    for venue_id in (VENUE_A, VENUE_B):
        stub = (_live_venue(venue_id) if live
                else _paper_venue(venue_id, prices[venue_id],
                                   ask_size=ask_sizes.get(venue_id, "5000")))
        venues[venue_id] = stub
        adapters.append(stub)
    registry = base.stub_registry(*adapters)
    agent.venue_registry = registry
    agent.multi_venue_executor.registry = registry
    agent.account_health_engine.venue_registry = registry
    agent.capability_engine.venue_registry = registry
    agent.settlement_engine.venue_registry = registry
    _inject_pair(agent, monkeypatch, pair)
    return agent, venues


def _orders(venue) -> List[Dict[str, Any]]:
    return venue.order_log


class TestThePairIsExecuted:
    def test_a_discovered_pair_reaches_both_venues(self, tmp_path, monkeypatch):
        agent, venues = _build(tmp_path, monkeypatch)

        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        assert _orders(venues[VENUE_A]), (
            "the pair was discovered and never sent: the arb lane is not wired")
        assert _orders(venues[VENUE_B]), (
            "only one leg was sent - an UNHEDGED one-sided position is worse "
            "than no trade at all")
        assert _orders(venues[VENUE_A])[0]["side"] == "YES"
        assert _orders(venues[VENUE_B])[0]["side"] == "NO"

        lane = report["arbitrage"]["execution"]
        assert lane["attempted"] == 1, lane
        assert lane["legs_filled"] == 2, lane
        assert report["arbitrage"]["tradeable"] == 1

    def test_the_legs_become_positions_the_ledger_can_settle(self, tmp_path,
                                                             monkeypatch):
        agent, venues = _build(tmp_path, monkeypatch)
        asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        positions = agent.storage.get_open_positions()
        markets = {p["market_id"] for p in positions}
        assert markets == {"PAIR-A", "PAIR-B"}, (
            f"the legs did not become positions: {markets} - a pair whose legs "
            f"are not in the ledger can never be settled or learned from")
        assert all(p["execution_mode"] == "paper" for p in positions), (
            "an adapter that cannot place real orders must record paper, never "
            "live")
        assert all(p["status"] == "paper" for p in positions)

    def test_a_matched_pair_is_recorded_as_matched(self, tmp_path, monkeypatch):
        agent, venues = _build(tmp_path, monkeypatch)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        arb_entries = [e for e in report["execution"]
                       if e.get("lane") == "arbitrage"]
        assert len(arb_entries) == 2, (
            f"the legs were not reported as an arbitrage lane: "
            f"{[e.get('lane') for e in report['execution']]}")
        assert all(e["position_recorded"] is True for e in arb_entries)
        assert "PAIR MATCHED" in arb_entries[0]["executor_result"]["reasoning"], (
            arb_entries[0]["executor_result"]["reasoning"])

    def test_the_legs_fill_the_same_number_of_shares(self, tmp_path, monkeypatch):
        """
        The economics of the pair, asserted on what reached the venues: one
        share of YES plus one share of NO pays exactly $1, so equal SHARES is
        what makes it riskless - equal dollars buys a different count on each
        side.
        """
        agent, venues = _build(tmp_path, monkeypatch)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        a_order = _orders(venues[VENUE_A])[0]
        b_order = _orders(venues[VENUE_B])[0]
        leg_a = next(e for e in report["execution"]
                     if e.get("market_id") == "PAIR-A")
        leg_b = next(e for e in report["execution"]
                     if e.get("market_id") == "PAIR-B")
        a_shares = leg_a["executor_result"]["filled_shares"]
        b_shares = leg_b["executor_result"]["filled_shares"]
        assert a_shares > 0 and b_shares > 0
        assert a_shares == pytest.approx(b_shares, rel=1e-6), (
            f"leg A holds {a_shares:.4f} shares and leg B {b_shares:.4f}: the "
            f"pair is not matched, so what was bought is a directional "
            f"position with a hedge-shaped name")
        # ...and B was sized on the price it was quoted, not on its limit:
        # dollars at the limit would have bought fewer shares than A holds.
        assert b_order["max_spend_usd"] == pytest.approx(a_shares * 0.50,
                                                         rel=1e-3), (
            f"leg B was sent for ${b_order['max_spend_usd']:.4f} to balance "
            f"{a_shares:.4f} shares at a 0.50 quote")

        positions = {p["market_id"]: p for p in agent.storage.get_open_positions()}
        held = {mid: row["position_size_usd"] / row["token_price_at_entry"]
                for mid, row in positions.items()}
        assert held["PAIR-A"] == pytest.approx(held["PAIR-B"], rel=1e-6), (
            f"the ledger holds {held} - the pair it recorded is not matched")

    def test_the_outcomes_reach_the_learning_chain(self, tmp_path, monkeypatch):
        """
        A leg that is not recorded teaches nothing, and the venue matrix is how
        a venue earns its place. Both legs must be in the outcome table.
        """
        agent, venues = _build(tmp_path, monkeypatch)
        asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        rows = agent.storage.conn.execute(
            "SELECT venue_id, execution_mode, amount_usd FROM trade_outcomes").fetchall()
        recorded = {r[0] for r in rows}
        assert recorded == {VENUE_A, VENUE_B}, (
            f"the recorded outcomes cover {recorded}, so one leg of the pair "
            f"would never be learned from")
        assert {r[1] for r in rows} == {"paper"}
        assert all(r[2] and r[2] > 0 for r in rows), (
            "an outcome row with no amount is not evidence of anything")


class TestALivePairRespectsTheOneVenueRule:
    def test_a_cross_venue_live_pair_is_refused_rather_than_funded(self, tmp_path,
                                                                   monkeypatch):
        """
        Money cannot move between venues by itself - a withdrawal and a deposit
        are the operator's actions - so exactly one venue holds the live capital.
        A cross-venue pair therefore has a leg that would move real money on a
        venue that is NOT authorised, and it cannot be funded.

        The honest outcome is the refusal, with the reason on the record. Sending
        leg A and not leg B would leave a one-sided live position, which is the
        worst of the available options: it is the exposure the hedge exists to
        avoid, taken at a venue nobody chose to fund.
        """
        agent, venues = _build(tmp_path, monkeypatch, live=True)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        assert _orders(venues[VENUE_A]) == [] and _orders(venues[VENUE_B]) == [], (
            "a live cross-venue pair reached the venues under a one-venue rule")
        lane = report["arbitrage"]["execution"]
        assert lane["attempted"] == 0, lane
        assert lane["results"], "the refusal was not reported"
        reason = lane["results"][0]["reason"]
        assert "not the selected live venue" in reason, reason


class TestTheGatesApplyToThePair:
    def test_a_pair_that_should_not_trade_is_not_sent(self, tmp_path, monkeypatch):
        agent, venues = _build(tmp_path, monkeypatch,
                               arb=_pair(should_trade=False))
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        assert _orders(venues[VENUE_A]) == []
        assert _orders(venues[VENUE_B]) == []
        assert report["arbitrage"]["execution"]["attempted"] == 0
        assert report["arbitrage"]["tradeable"] == 0

    def test_a_mock_leg_is_refused(self, tmp_path, monkeypatch):
        arb = _pair()
        arb.market_b.is_mock = True
        agent, venues = _build(tmp_path, monkeypatch, arb=arb)
        report = asyncio.run(agent.run_cycle(target_per_venue=5, max_trades=0))

        assert _orders(venues[VENUE_A]) == [] and _orders(venues[VENUE_B]) == [], (
            "a pair with a MOCK leg reached execution")
        blocked = [e for e in report["execution"]
                   if e.get("lane") == "arbitrage" and e.get("status") == "blocked"]
        assert len(blocked) == 1, (
            f"the refusal was not reported: "
            f"{[e.get('status') for e in report['execution']]}")
        assert "MOCK_DATA" in blocked[0]["reason"], blocked[0]["reason"]
