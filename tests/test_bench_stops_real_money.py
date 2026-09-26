"""
End to end: a rule the record has refused does not spend real money.

`tests/test_rule_bench.py` pins the bench as a decision. This file pins it as a
fact about the order path - the loop, with a live venue selected and a budget
authorised, refuses real money for an entry that only refused rules carry and
still places the order in paper.

The record here is generated, not discovered: 400 settled paper trades with a
strong edge for the first 240 and nothing after it. That is exactly the shape
walk-forward is built to catch - it looks real on the discovery folds and is gone
on the fresh holdout - and it is the one case where acting on the rule is a
mistake the record has already caught.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from tests.test_full_cycle_from_discovery_to_allocation import (
    VENUE,
    _cycle,
    _force_qualified,
    _inject_opportunity,
    _simulating_place_order,
    build_agent,
)


def _record_with_an_edge_that_stops(agent, *, rows: int = 400,
                                    stop_at: int = 240) -> None:
    """
    Seed settled outcomes: a strong edge that works, then stops.

    Written through the tracker, in the shape the loop writes, so the validator
    reads a real record rather than a fixture shaped for it.
    """
    tracker = agent.trade_outcome_tracker
    storage = agent.storage
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rnd = random.Random(3)
    for i in range(rows):
        big = (i % 2 == 1) and i < stop_at
        edge = 0.35 if big else 0.05
        prob = 0.5 + edge
        won = rnd.random() < prob
        tracker.record_trade(
            trade_id=f"SEED{i}", market_id=f"SEEDM{i}", venue_id=VENUE,
            strategy="value", category="politics", forecast_prob=prob,
            market_price=0.5, edge=edge, side="YES", amount_usd=3.0,
            execution_mode="paper", yes_price=0.5)
        tracker.record_resolution(
            f"SEED{i}", 1.0 if won else 0.0, 3.0 if won else -3.0)
        storage.conn.execute(
            "UPDATE trade_outcomes SET resolved_at=? WHERE trade_id=?",
            ((base + timedelta(days=i)).isoformat(), f"SEED{i}"))
    storage.conn.commit()


def test_a_record_that_refuses_the_edge_rules_runs_the_trade_in_paper(
        tmp_path, monkeypatch):
    agent, adapter = build_agent(tmp_path)
    _force_qualified(agent, monkeypatch)
    _inject_opportunity(agent, adapter._market(), monkeypatch, edge=0.15)
    # The venue simulates the fill, the way the real adapter does in dry run.
    # The adapter is created ARMED (dry_run=False), so the point of the check
    # below is that every call reached it in dry run.
    saw_armed = []
    inner = _simulating_place_order(adapter)

    async def place_order(opportunity, max_spend_usd, max_price):
        saw_armed.append(bool(adapter.dry_run))
        return await inner(opportunity, max_spend_usd, max_price)

    adapter.place_order = place_order
    try:
        _record_with_an_edge_that_stops(agent)
        # The verdict the loop records at the start of THIS cycle is what the
        # bench acts on - not a verdict handed to the test.
        result = _cycle(agent)

        assert agent._cycle_benched, (
            "the record says the edge rules failed their holdout and nothing "
            "was benched")
        entry = result["execution"][0]
        assert entry["execution_lane"] == "paper", entry
        assert "out-of-sample bench" in entry["paper_because"], entry["paper_because"]
        # ...and the live venue is still the live venue: the money is refused,
        # not the venue forgotten.
        assert agent._cycle_live_venue == VENUE

        # The trade ran. A refusal that drops the order would teach the agent
        # nothing and is the bug V42 fixed.
        assert saw_armed, "no order reached the venue at all"
        assert all(saw_armed), (
            "the bench refused real money and the order was sent ARMED anyway")
        assert entry["position_recorded"] is True
        assert adapter.orders_placed >= 1
        positions = agent.storage.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["execution_mode"] == "paper", (
            "the bench refused real money and the position came out live anyway")
    finally:
        agent.storage.close()


def test_without_that_record_the_same_order_is_live(tmp_path, monkeypatch):
    # The control: same fixture, same opportunity, no refused rules - so the
    # order goes where the operator's authorisation says it goes. Without this,
    # the test above could pass because the lane was never live in the first
    # place.
    agent, adapter = build_agent(tmp_path)
    _force_qualified(agent, monkeypatch)
    _inject_opportunity(agent, adapter._market(), monkeypatch, edge=0.15)
    try:
        result = _cycle(agent)
        assert agent._cycle_benched == {}, agent._cycle_benched
        entry = result["execution"][0]
        assert entry["execution_lane"] == "live", entry
        assert "paper_because" not in entry
        assert entry["execution_mode"] == "live", entry
    finally:
        agent.storage.close()


def test_the_bench_does_not_widen_the_entry_filters(tmp_path, monkeypatch):
    """
    The refused rule keeps choosing which trades are candidates.

    An entry the gates rejected must stay rejected when a rule is benched: the
    bench subtracts the authorisation to risk real money and never adds a trade.
    Here the opportunity's edge is 2%, under the 8% floor the sizing rule
    enforces, so nothing reaches the venue - with the edge rules benched and
    without them.
    """
    agent, adapter = build_agent(tmp_path)
    _force_qualified(agent, monkeypatch)
    # edge 0.02: under the 8% floor, so the gates must refuse it before the
    # bench is ever consulted.
    _inject_opportunity(agent, adapter._market(), monkeypatch, edge=0.02)
    try:
        _record_with_an_edge_that_stops(agent)
        _cycle(agent)
        placed = int(getattr(adapter, "orders_placed", 0) or 0)
        assert placed == 0, (
            f"the bench let a below-floor entry through: {placed} order(s)")
        # ...and no refused rule was needed to reach that outcome: the entry
        # gates are what refuse it, and they still run.
        assert agent._cycle_benched, agent._cycle_benched
    finally:
        agent.storage.close()


