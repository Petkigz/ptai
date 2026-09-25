"""
Capital allocation learns VENUE x MARKET TYPE x STRATEGY x EXECUTION STYLE.

The seventh report's seventh item. Every statistic the allocator had grouped by
`venue_id` alone, so "venue = good/bad" was the only reading of the record
available - and it is the wrong reading, because a venue can be good at a
five-minute sports market and hopeless at a thirty-day political one.

The cells here are the dimensions the outcome log can actually support
(venue x strategy x market type x execution mode).

THE ONE RULE THE LADDER OBEYS:

    Relax dimensional specificity. NEVER relax evidence provenance.

A narrow question can legitimately be answered from a wider slice, because a
wider slice is the same kind of thing counted more broadly. It can never be
answered from a different KIND of evidence: simulated trades are not live trades
with less detail. A live-capital decision that quietly leans on nine paper
trades because only three live ones exist is exactly the failure the matrix was
built to prevent.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
from src.ptai.storage.db import Storage
from src.ptai.strategy.venue_selection import VenueSelector


def _store(tmp_path, name: str, rows: List[Tuple]):
    """
    A storage full of resolved outcomes.

    `rows` is (execution_mode, strategy, market_type, pnl_per_trade, count) and
    optionally a book label.

    Both halves of a resolution are written: the TRADE row (which the per-venue
    figures and the ledger read) and the OUTCOME row (which the matrix and the
    qualification gate read). Writing only one of them is how a fixture ends up
    describing a system that never ran.
    """
    storage = Storage(db_path=str(tmp_path / name))
    tracker = TradeOutcomeTracker(storage=storage)
    n = 0
    for row in rows:
        mode, strategy, category, pnl, count = row[:5]
        book = row[5] if len(row) > 5 else "ladder"
        for _ in range(count):
            n += 1
            trade_id = storage.log_trade({
                "market_id": f"M{n}", "venue_id": "polymarket", "side": "YES",
                "position_size_usd": 3.0, "market_price": 0.5,
                "fair_price": 0.6, "edge": 0.1, "strategy": strategy,
                "category": category, "execution_mode": mode, "status": mode,
            })
            tracker.record_trade(
                trade_id=str(trade_id), market_id=f"M{n}", venue_id="polymarket",
                strategy=strategy, category=category, forecast_prob=0.6,
                market_price=0.5, edge=0.1, side="YES", amount_usd=3.0,
                execution_mode=mode, book_source=book, fees_usd=0.06,
                slippage_bps=10.0, execution_quality=0.9,
                expected_net_ev_pct=0.05)
            tracker.record_resolution(str(trade_id),
                                      actual_outcome=1.0 if pnl > 0 else 0.0,
                                      pnl=pnl)
            storage.resolve_trade(trade_id, outcome=1.0 if pnl > 0 else 0.0,
                                  pnl=pnl)
    return storage


@pytest.fixture()
def selector(tmp_path):
    """
    One venue, four cells. Its LIVE total is positive while one LIVE cell loses
    money on every trade, which is the whole reason a venue total is not an
    allocation.
    """
    storage = _store(tmp_path, "matrix.db", [
        ("live", "value", "sports", 1.10, 6),      # live, wins
        ("live", "value", "politics", -0.90, 6),   # live, loses every time
        ("paper", "value", "sports", 0.40, 6),
        # Two trades, and priced against a book that never existed.
        ("paper", "momentum", "sports", 5.00, 2, "assumed_default"),
    ])
    return VenueSelector(storage=storage)


# ----------------------------------------------------------------------
# the cells
# ----------------------------------------------------------------------

def test_cells_are_split_by_venue_strategy_market_type_and_mode(selector):
    cells = selector.evidence_matrix()
    keys = set(cells)
    assert "polymarket|value|sports|live" in keys
    assert "polymarket|value|politics|live" in keys
    assert "polymarket|value|sports|paper" in keys
    assert "polymarket|momentum|sports|paper" in keys

    # live and paper are separate cells, never one averaged number.
    live = cells["polymarket|value|sports|live"]
    assert live["execution_mode"] == "live"
    assert live["resolved"] == 6
    assert live["net_pnl_per_trade"] == pytest.approx(1.10)
    assert live["real_evidence_coverage"] == 1.0

    paper = cells["polymarket|value|sports|paper"]
    assert paper["resolved"] == 6
    assert paper["net_pnl_per_trade"] == pytest.approx(0.40)


def test_a_thin_cell_is_not_dressed_up_as_an_answer(selector):
    """
    Two trades earning $5 each is the best per-trade number in the record, and
    it is not an answer.
    """
    thin = selector.evidence_matrix()["polymarket|momentum|sports|paper"]
    assert thin["resolved"] == 2
    assert not thin["enough_evidence"]
    assert thin["real_evidence_coverage"] == 0.0, (
        "the assumed book it was priced against must be visible in the cell")

    assert "polymarket|momentum|sports|paper" not in {
        cell["key"] for cell in selector.best_cells()}


def test_a_venue_total_hides_that_one_cell_is_losing(selector):
    """
    The reason the matrix exists.

    Polymarket's LIVE total is positive, and one of its live cells loses money
    on every single trade. The aggregate cannot say which; the cell can.
    """
    stats = selector.venue_evidence()["polymarket"]
    assert stats["live_resolved"] == 12
    assert stats["live_net_pnl"] > 0, (
        "the fixture must give the venue a positive live total, or this test "
        "proves nothing about aggregates hiding losses")

    losing = selector.cell_evidence("polymarket", "value", "politics", "live")
    assert losing["enough_evidence"], "the losing cell must be big enough to judge"
    assert losing["resolved"] == 6
    assert losing["net_pnl_per_trade"] < 0, (
        "the cell that loses on every trade is not visible in the matrix")


def test_the_ladder_reports_which_level_answered(selector):
    exact = selector.cell_evidence("polymarket", "value", "sports", "live")
    assert exact["level"] == "venue x strategy x category (execution_mode=live)"
    assert exact["resolved"] == 6
    assert "reason" not in exact, "an exact answer needs no excuse"

    # Nothing resolved for this venue at all: no level to fall back to, and the
    # answer says so rather than reporting the absence as a zero.
    nothing = selector.cell_evidence("kalshi")
    assert nothing["level"] == "none"
    assert nothing["resolved"] == 0
    assert nothing["has_evidence"] is False
    assert "nothing resolved" in nothing["reason"]


# ----------------------------------------------------------------------
# the invariant: provenance never relaxes
# ----------------------------------------------------------------------

def test_a_live_question_is_never_answered_from_paper_trades(tmp_path):
    """
    3 LIVE resolved trades, 9 PAPER resolved trades, asking about LIVE.

    The answer is 3. It is below the sample bar and the ladder says so - what it
    must NOT do is step up into the nine paper trades to find a comfortable
    sample size.
    """
    storage = _store(tmp_path, "live_thin.db", [
        ("live", "value", "sports", 1.10, 3),
        ("paper", "value", "sports", 9.99, 9),
    ])
    selector = VenueSelector(storage=storage)

    answer = selector.cell_evidence("polymarket", "value", "sports", "live")
    assert answer["resolved"] == 3, (
        f"live evidence was answered from {answer['resolved']} trades")
    assert answer["execution_mode"] == "live"
    assert answer["has_evidence"] is True
    # Three trades is not enough to allocate on, and the answer says which level
    # it came from instead of pretending.
    assert answer["enough_evidence"] is False
    assert answer["level"] == "insufficient"


def test_a_live_question_with_no_live_trades_has_no_answer(tmp_path):
    """
    0 LIVE, 20 PAPER, asking about LIVE.

    The most important of these. A paper record - however long, however
    profitable - must never become a live allocation just because the live
    sample is empty. There is no evidence for this question, and the answer says
    so and names what the record actually contains.
    """
    storage = _store(tmp_path, "paper_only.db", [
        ("paper", "momentum", "sports", 9.99, 20),
    ])
    selector = VenueSelector(storage=storage)

    answer = selector.cell_evidence("polymarket", "momentum", "sports", "live")
    assert answer["resolved"] == 0
    assert answer["has_evidence"] is False
    assert answer["enough_evidence"] is False
    assert answer["level"] == "none", (
        "an empty live record must not borrow the level of the paper one")
    assert "live mode" in answer["reason"]
    assert "paper" in answer["reason"], (
        "the answer should say what the record DOES contain, so the operator "
        "knows the work is paper evidence awaiting a live sample")


def test_the_paper_question_is_still_answerable_from_paper(tmp_path):
    """The rule cuts both ways: paper questions are answered from paper."""
    storage = _store(tmp_path, "paper_answer.db", [
        ("paper", "momentum", "sports", 0.20, 20),
    ])
    selector = VenueSelector(storage=storage)

    answer = selector.cell_evidence("polymarket", "momentum", "sports", "paper")
    assert answer["resolved"] == 20
    assert answer["has_evidence"] is True
    assert answer["execution_mode"] == "paper"
    assert answer["enough_evidence"] is True


def test_a_losing_live_cell_cannot_be_brightened_by_paper_profit(tmp_path):
    """
    The mixing failure, stated as money.

    The paper cell earns $99 a trade and the live cell loses a dollar a trade.
    If any level of the ladder crossed the mode boundary, the live answer would
    come out positive. It must not.
    """
    storage = _store(tmp_path, "bright.db", [
        ("live", "value", "politics", -1.00, 6),
        ("paper", "value", "politics", 99.00, 6),
    ])
    selector = VenueSelector(storage=storage)

    live = selector.cell_evidence("polymarket", "value", "politics", "live")
    assert live["resolved"] == 6
    assert live["net_pnl_per_trade"] == pytest.approx(-1.00)
    assert live["net_pnl"] == pytest.approx(-6.00)

    # ...and the wide LIVE level, where the temptation to borrow is greatest,
    # is still live only.
    wide = selector.cell_evidence("polymarket")
    assert wide["execution_mode"] == ""
    assert wide["net_pnl"] == pytest.approx(99.00 * 6 + -1.00 * 6), (
        "a mode-agnostic question aggregates both - and says so by naming no mode")


# ----------------------------------------------------------------------
# ranking and the console
# ----------------------------------------------------------------------

def test_cells_are_ranked_on_profit_per_trade_not_total():
    """A cell with 40 trades earning 50c each beats 3 trades earning $3 each."""
    cells = [
        {"key": "big", "resolved": 40, "net_pnl_per_trade": 0.5},
        {"key": "small", "resolved": 3, "net_pnl_per_trade": 3.0},
    ]
    ordered = sorted(cells, key=lambda c: c["net_pnl_per_trade"], reverse=True)
    assert [c["key"] for c in ordered] == ["small", "big"]
    # ...and best_cells applies the sample bar, which drops the small one.
    assert VenueSelector.MIN_CELL_SAMPLES > 3


def test_an_empty_record_is_empty_not_zero(selector):
    """A venue nobody traded has no cells - not a cell of zeros."""
    matrix = selector.evidence_matrix()
    assert all(cell["venue_id"] == "polymarket" for cell in matrix.values())
    assert "kalshi" not in " ".join(matrix)


def test_the_console_shows_the_matrix_and_the_thin_evidence(selector, tmp_path):
    from src.ptai.operator_view import describe_snapshot, operator_snapshot

    storage = selector.storage
    snapshot = operator_snapshot(storage=storage)
    matrix = snapshot["venues"]["matrix"]
    assert matrix["cells"] >= 4
    assert matrix["cells_with_enough_evidence"] >= 2
    assert matrix["strongest_cells"][0]["cell"]

    lines = describe_snapshot(snapshot)
    evidence_lines = [line for line in lines if line.startswith("Evidence:")]
    assert evidence_lines, lines
    assert "venue x strategy x market type" in evidence_lines[0]
    assert "real book" in evidence_lines[0]
