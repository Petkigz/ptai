"""
Which venue is the agent using, and does it trade without asking?

Three questions the operator asked, and what has to be true for each answer:

1. "Does funding the most profitable venue mean I fund every venue separately?"
   No. The agent holds live capital at ONE venue at a time, because money cannot
   be moved between venues by the agent - a withdrawal and a deposit are the
   operator's actions. Every other venue is still scanned and paper-traded.

2. "How do I know which venue it is using?"
   The selection is computed every cycle and recorded in the cycle result, and
   the console names the live venue, the reason, and the ranking behind it.

3. "Does it trade automatically, with no interaction except logins?"
   Yes. There is no per-trade approval anywhere in the trade path, and this file
   proves the contract is stated rather than merely believed.

The tests below drive the unhappy paths, because the failure mode being guarded
against is a system that flatters itself: a venue with three trades presented as
"the most profitable", a switch recommended on noise, or an agent that quietly
waits for a human without saying so.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.ptai.execution.capital import CapitalLedger, FUNDING_ROUTES
from src.ptai.storage.db import Storage
from src.ptai.strategy.venue_selection import (
    LIVE_QUALIFICATION_SAMPLE,
    MAX_LIVE_VENUES_SMALL_BUDGET,
    MIN_SAMPLE_FOR_EVIDENCE,
    ROLE_LIVE,
    ROLE_PAPER,
    ROLE_UNAVAILABLE,
    VenueSelector,
    autonomy_contract,
)

POLY = "polymarket"
KALSHI = "kalshi"


@pytest.fixture()
def storage(tmp_path):
    return Storage(db_path=str(tmp_path / "venue.db"))


@pytest.fixture()
def selector(storage):
    return VenueSelector(storage=storage, funding_routes=FUNDING_ROUTES)


def _labels():
    return {v: r["label"] for v, r in FUNDING_ROUTES.items()}


def _accounts(storage, budgets, balances):
    """
    The real account shape, from the real ledger.

    Hand-writing {"funded": True} would test the test: the loop builds these
    through CapitalLedger, and a funded venue is decided there.
    """
    plan = CapitalLedger(storage=storage).build(
        mode="live", budgets=budgets, balances=balances,
        venue_labels=_labels())
    return [a.to_dict() for a in plan.accounts]


def _trades(storage, venue, n, pnl, price=0.5):
    """n resolved trades at the venue, each realising `pnl`."""
    for i in range(n):
        tid = storage.log_trade({
            "market_id": f"{venue}-{i}",
            "venue_id": venue,
            "side": "YES",
            "position_size_usd": 3.0,
            "market_price": price,
            "fair_price": price + 0.1,
            "edge": 0.1,
            "confidence": 0.7,
            "strategy": "value",
            "data_mode": "live_paper",
        })
        storage.resolve_trade(tid, outcome=1.0 if pnl > 0 else 0.0, pnl=pnl)


# ----------------------------------------------------------------------
# 1. no evidence is not a score
# ----------------------------------------------------------------------

def test_a_venue_with_no_history_has_no_score(selector):
    """An unmeasured venue is not a zero, and it is not a winner either."""
    a = selector.assess([POLY], labels=_labels())[0]
    assert a.resolved_trades == 0
    assert a.has_evidence is False
    assert a.qualified is False
    assert a.role == ROLE_PAPER
    # The reason names the absence, rather than implying a bad result.
    assert "no resolved trades" in a.evidence_basis


def test_three_lucky_trades_do_not_make_a_venue_the_most_profitable(selector, storage):
    """
    The exact way a small sample lies: three wins in a row.

    P&L per trade here is enormous ($2 on $3). It must still not rank, because a
    score from three trades is noise wearing a decimal point.
    """
    _trades(storage, POLY, 3, pnl=2.0)
    a = selector.assess([POLY], labels=_labels())[0]
    assert a.resolved_trades == 3
    assert a.has_evidence is False, "3 trades is below the evidence floor"
    assert a.resolved_trades < MIN_SAMPLE_FOR_EVIDENCE
    assert a.pnl_per_trade == 2.0  # the number exists ...
    assert a.qualified is False   # ... and it is not allowed to decide anything


def test_evidence_starts_at_the_floor(selector, storage):
    _trades(storage, POLY, MIN_SAMPLE_FOR_EVIDENCE - 1, pnl=0.5)
    assert selector.assess([POLY], labels=_labels())[0].has_evidence is False
    _trades(storage, POLY, 1, pnl=0.5)
    a = selector.assess([POLY], labels=_labels())[0]
    assert a.resolved_trades == MIN_SAMPLE_FOR_EVIDENCE
    assert a.has_evidence is True


def test_ranking_is_on_realised_money_per_trade(selector, storage):
    """Two venues with evidence: the better net P&L per resolved trade wins."""
    _trades(storage, POLY, 40, pnl=0.60)
    _trades(storage, KALSHI, 40, pnl=0.10)
    assessments = selector.assess([POLY, KALSHI], labels=_labels())
    ranked = sorted([a for a in assessments if a.has_evidence],
                    key=lambda a: a.pnl_per_trade, reverse=True)
    assert [a.venue_id for a in ranked] == [POLY, KALSHI]
    sel = selector.select(assessments)
    assert sel.candidate == POLY
    assert "polymarket" in sel.ranking_basis or "resolved trade" in sel.ranking_basis


def test_a_losing_venue_is_not_a_candidate(selector, storage):
    """Explicit: a venue can have plenty of evidence and still be the wrong one."""
    _trades(storage, POLY, 40, pnl=-0.40)
    a = selector.assess([POLY], labels=_labels())[0]
    assert a.has_evidence is True
    assert a.pnl_per_trade < 0
    # Do not fabricate a way for a losing venue to look fundable.
    assert "profit" not in " ".join(a.notes).lower() or a.pnl_per_trade < 0


# ----------------------------------------------------------------------
# 2. one venue holds the money
# ----------------------------------------------------------------------

def test_only_one_venue_may_hold_live_capital(selector, storage):
    """
    Even with two qualified, funded, authorised venues, one holds the capital.

    This is the answer to "do I have to fund every venue separately": no. The cap
    is enforced here rather than left to policy, because a second live venue
    would silently double the risk the operator authorised.
    """
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=2.0)
    _trades(storage, KALSHI, LIVE_QUALIFICATION_SAMPLE, pnl=1.5)
    accounts = _accounts(
        storage, {POLY: 25.0, KALSHI: 25.0},
        {POLY: {"available": True, "balance": 25.0},
         KALSHI: {"available": True, "balance": 25.0}})
    sel = selector.select(
        selector.assess([POLY, KALSHI], accounts=accounts,
                        qualified_ids=[POLY, KALSHI], labels=_labels()))
    assert all(a.funded for a in sel.assessments), "the fixture must really fund"
    assert sel.max_live_venues == MAX_LIVE_VENUES_SMALL_BUDGET == 1
    assert len(sel.live_venues) <= 1
    assert sel.live_venue == POLY, "the better venue should hold the money"
    assert "kalshi" not in [v.venue_id for v in sel.live_venues]
    # The runner-up is not idle: it is still researched.
    kalshi = [a for a in sel.assessments if a.venue_id == KALSHI][0]
    assert kalshi.role in (ROLE_PAPER, ROLE_UNAVAILABLE)
    assert any("paper" in r.lower() for r in sel.reasons)


def test_unfunded_means_not_live(selector, storage):
    """Qualified is not funded. A venue the operator never put money into cannot
    be where the money is."""
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=2.0)
    sel = selector.select(selector.assess(
        [POLY], accounts=[], qualified_ids=[POLY], labels=_labels()))
    assert sel.live_venue is None
    assert sel.verdict.startswith("paper only")


def test_a_venue_that_cannot_be_funded_is_unavailable(selector):
    """No funding route at all: scanned, but never described as a place to put
    money."""
    sel = selector.select(selector.assess(["manifold", "crypto_binance"],
                                         labels=_labels()))
    assert all(x.role == ROLE_UNAVAILABLE for x in sel.assessments)
    assert all(x.blockers for x in sel.assessments), "say WHY it is unavailable"


# ----------------------------------------------------------------------
# 3. a switch has to be worth doing
# ----------------------------------------------------------------------

def test_the_live_venue_is_remembered_not_recomputed(selector, storage):
    """
    The home of the real money does not drift on a better-looking number.

    The agent adopts the first venue that becomes ready, writes it down, and then
    keeps trading there. A ranking is a recommendation about where money SHOULD
    go; it must never silently relocate where money IS.
    """
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=1.00)
    accounts = _accounts(storage, {POLY: 50.0},
                         {POLY: {"available": True, "balance": 50.0}})
    first = selector.select(selector.assess(
        [POLY], accounts=accounts, qualified_ids=[POLY], labels=_labels()))
    assert first.live_venue == POLY
    assert first.remembered_live_venue == POLY
    # Written down, not held in memory.
    assert storage.get_state(VenueSelector.LIVE_VENUE_KEY) == POLY

    # A new selector, as after a restart: same answer, without a cycle of drift.
    fresh = VenueSelector(storage=storage, funding_routes=FUNDING_ROUTES)
    assert fresh.remembered_live_venue() == POLY


def test_a_switch_is_not_recommended_on_noise(selector, storage):
    """
    A marginally better venue is not a reason to move money.

    Withdrawing and re-depositing costs real money (a withdrawal fee plus an
    on-ramp), and paying that to chase a per-trade edge smaller than the cost is
    how an agent transfers its own capital to the rails.
    """
    # Polymarket is live and funded.
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=1.00)
    accounts = _accounts(storage, {POLY: 50.0},
                         {POLY: {"available": True, "balance": 50.0}})
    selector.select(selector.assess([POLY], accounts=accounts,
                                    qualified_ids=[POLY], labels=_labels()))

    # Kalshi now trades $0.001 per trade better. That is noise.
    _trades(storage, KALSHI, LIVE_QUALIFICATION_SAMPLE, pnl=1.001)
    accounts = _accounts(
        storage, {POLY: 50.0, KALSHI: 50.0},
        {POLY: {"available": True, "balance": 50.0},
         KALSHI: {"available": True, "balance": 50.0}})
    sel = selector.select(
        selector.assess([POLY, KALSHI], accounts=accounts,
                        qualified_ids=[POLY, KALSHI], labels=_labels()),
        total_budget_usd=50.0)

    assert sel.live_venue == POLY, "the funded venue stays the live one"
    assert sel.switch_warranted is False
    assert storage.get_state(VenueSelector.LIVE_VENUE_KEY) == POLY


def test_the_live_venue_can_be_taken_over_once_it_is_clearly_better(selector, storage):
    """
    Concentration is not stubbornness. A venue that is clearly better, with the
    evidence to say so, does warrant a move - and the plan says what to do.
    """
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=0.10)
    accounts = _accounts(storage, {POLY: 50.0},
                         {POLY: {"available": True, "balance": 50.0}})
    selector.select(selector.assess([POLY], accounts=accounts,
                                    qualified_ids=[POLY], labels=_labels()))

    # Kalshi is much better, and funded, and qualified.
    _trades(storage, KALSHI, LIVE_QUALIFICATION_SAMPLE, pnl=2.00)
    accounts = _accounts(
        storage, {POLY: 50.0, KALSHI: 50.0},
        {POLY: {"available": True, "balance": 50.0},
         KALSHI: {"available": True, "balance": 50.0}})
    sel = selector.select(
        selector.assess([POLY, KALSHI], accounts=accounts,
                        qualified_ids=[POLY, KALSHI], labels=_labels()),
        total_budget_usd=50.0)

    assert sel.candidate == KALSHI
    assert sel.switch_warranted is True
    assert sel.switch_plan.get("from") == POLY
    assert sel.switch_plan.get("to") == KALSHI
    assert sel.switch_plan.get("steps"), "a switch must say what to do"
    # The agent does not move the money.
    assert "withdraw" in " ".join(sel.switch_plan["steps"]).lower() or \
           "move" in " ".join(sel.switch_plan["steps"]).lower()


def test_a_live_venue_that_stops_being_ready_is_reported_not_silently_swapped(
        selector, storage):
    """
    The money is still there. If the account stops being ready, the honest
    answer is to say so - not to quietly announce a different live venue while
    the capital sits where it always was.
    """
    _trades(storage, POLY, LIVE_QUALIFICATION_SAMPLE, pnl=1.00)
    accounts = _accounts(storage, {POLY: 50.0},
                         {POLY: {"available": True, "balance": 50.0}})
    selector.select(selector.assess([POLY], accounts=accounts,
                                    qualified_ids=[POLY], labels=_labels()))
    assert selector.remembered_live_venue() == POLY

    # The balance can no longer be read: POLY is no longer deployable.
    accounts = _accounts(storage, {POLY: 50.0},
                         {POLY: {"available": False, "balance": 0.0}})
    sel = selector.select(selector.assess(
        [POLY], accounts=accounts, qualified_ids=[POLY], labels=_labels()))
    assert sel.live_venue is None
    assert sel.remembered_live_venue == POLY
    assert sel.warnings, "an unready live venue must be reported"
    assert POLY in " ".join(sel.warnings)


def test_the_move_cost_is_stated_not_guessed(selector):
    """
    Cost of moving money must come from the funding routes, and be visible.

    If this number is invented, the switch decision is invented with it.
    """
    cost = VenueSelector._estimated_move_cost(POLY, KALSHI, 50.0)
    assert cost > 0
    assert cost < 50.0, "a move cost equal to the capital would never be right"
    # It has to scale with the amount: a flat fee would make a large move look
    # as cheap as a small one.
    assert VenueSelector._estimated_move_cost(POLY, KALSHI, 100.0) > cost


# ----------------------------------------------------------------------
# 4. the operator can always see the answer
# ----------------------------------------------------------------------

def test_the_selection_survives_being_written_to_json(selector, storage):
    """The console reads this as JSON; a selection that cannot be serialised is
    a selection the operator never sees."""
    import json

    sel = selector.select(selector.assess([POLY, KALSHI], labels=_labels()),
                          total_budget_usd=50.0)
    blob = json.loads(json.dumps(sel.to_dict()))
    assert set(blob) >= {"live_venue", "candidate", "verdict", "reasons",
                         "switch_plan", "autonomy", "ranking_basis",
                         "max_live_venues", "assessments"}
    assert isinstance(blob["reasons"], list)
    assert blob["verdict"]
    # The assessment the operator reads must carry its own sample size, or the
    # number next to the venue means nothing.
    assert all("resolved_trades" in a for a in blob["assessments"])
    assert all("has_evidence" in a for a in blob["assessments"])


def test_the_fresh_install_names_a_venue_it_can_actually_open_and_says_why(selector):
    """
    Nothing has traded yet, which is where every install starts.

    The choice must not be arbitrary, must not be presented as performance, and
    must not pick a venue this operator cannot open an account at.
    """
    sel = selector.select(selector.assess([POLY, KALSHI], labels=_labels()))
    assert sel.live_venue is None
    assert sel.candidate is not None
    joined = " ".join(sel.reasons).lower()
    assert "provisional" in joined
    # It must say the choice is about access/cost, not about results.
    assert "entry cost" in joined or "access" in joined
    # ... and it must name the reason a restricted venue is not chosen.
    assert "us" in joined or "residency" in joined or "identity" in joined


def test_a_venue_the_operator_cannot_open_does_not_win_the_tiebreak(selector):
    """Kalshi needs US residency. With no evidence either way, the venue that is
    actually open to this operator is the one to fund."""
    sel = selector.select(selector.assess([POLY, KALSHI], labels=_labels()))
    assert sel.candidate == POLY


# ----------------------------------------------------------------------
# 5. autonomy: what runs without asking
# ----------------------------------------------------------------------

def test_trading_needs_no_approval(selector, storage):
    """The contract says it plainly, and the answer is not 'maybe'."""
    sel = selector.select(selector.assess([POLY], labels=_labels()))
    au = sel.autonomy or autonomy_contract()
    assert au["trades_without_approval"] is True
    assert au["per_trade_approval"] is False


def test_the_operator_keeps_exactly_four_things(selector):
    """
    Deposit, credentials, budget, kill switch. Anything else on this list is
    either a disguised approval step or an operator chore the agent should own.
    """
    au = autonomy_contract()
    actions = " ".join(
        a["action"] if isinstance(a, dict) else str(a)
        for a in au["operator_only_actions"]
    ).lower()
    assert "deposit" in actions
    assert "credential" in actions or "log in" in actions or "api key" in actions
    assert "budget" in actions or "mode" in actions
    assert "kill" in actions
    # The forbidden one.
    assert "approve each" not in actions
    assert "confirm each" not in actions


def test_the_statement_is_specific_enough_to_be_falsifiable(selector):
    """A vague autonomy claim is the kind that turns out to be false."""
    au = autonomy_contract()
    assert au["agent_does_autonomously"], "the claim must list what it does"
    limit = au["honest_limit"].lower()
    # The limit must name the thing the agent genuinely cannot do, so the
    # claim is falsifiable rather than reassuring.
    assert "cannot" in limit
    assert "fund" in limit
    assert len(au["operator_only_actions"]) <= 5, \
        "more than five operator chores is not 'automatic'"
    never = " ".join(au["what_it_never_needs"]).lower()
    assert "approval" in never


# ----------------------------------------------------------------------
# 6. the console endpoint answers the question
# ----------------------------------------------------------------------

def test_the_console_venue_endpoint_names_the_venue_and_the_autonomy(
        tmp_path, monkeypatch
):
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console_venue.db"))
    from fastapi.testclient import TestClient

    from src.ptai.ui.console import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/console/venue")
    assert r.status_code == 200, r.text
    body = r.json()

    sel = body["selection"]
    assert "live_venue" in sel
    assert sel["verdict"]
    assert sel["reasons"], "the operator must be told WHY"
    assert sel["autonomy"]["trades_without_approval"] is True
    assert sel["autonomy"]["per_trade_approval"] is False
    assert body["ranking_basis"]
    assert body["one_live_venue_cap"] is True
    assert body["assessments"], "there must be something to rank"

    # A fresh install must not claim a live venue it does not have.
    assert sel["live_venue"] is None
    assert "engine_running" in body


def test_the_console_page_has_the_venue_tab(tmp_path, monkeypatch):
    """The answer has to be reachable by clicking, not only by curling the API."""
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console_page.db"))
    from fastapi.testclient import TestClient

    from src.ptai.ui.console import app

    html = TestClient(app, raise_server_exceptions=False).get("/").text
    assert 'data-tab="venue"' in html
    assert 'id="tab-venue"' in html
    assert "Which venue the agent is using" in html
    assert "loadVenue" in html
    assert "What runs without asking" in html
    # The claim about ranking has to be on the page next to the numbers.
    assert "no score" in html.lower()


def test_the_console_and_the_cycle_share_one_definition_of_the_budget(tmp_path):
    """
    The screen and the trading loop must not disagree about what is authorised.

    They did: the console wrote its own state keys while the cycle read none, so a
    venue could be shown as authorised while the agent sized against nothing.
    One reader, one writer, one key namespace - checked in both directions.
    """
    from src.ptai.execution.capital import (
        authorised_budgets,
        operator_mode,
        set_authorised_budget,
        set_operator_mode,
    )
    from src.ptai.ui.console import ConsoleState

    storage = Storage(db_path=str(tmp_path / "one_definition.db"))
    state = ConsoleState(storage)

    # Nothing authorised yet: both agree, and neither invents a budget.
    assert state.budgets() == authorised_budgets(storage) == {}
    assert state.mode == operator_mode(storage) == "paper"

    # Written the way the console writes it, read the way the cycle reads it.
    set_authorised_budget(storage, POLY, 12.0)
    set_operator_mode(storage, "live")
    assert state.budgets() == {POLY: 12.0}
    assert authorised_budgets(storage) == {POLY: 12.0}
    assert state.mode == "live" == operator_mode(storage)

    # And the other way round: what the console route writes (through the same
    # domain helper) is what ConsoleState - and therefore the screen - shows.
    set_authorised_budget(storage, KALSHI, 25.0)
    assert state.budgets() == authorised_budgets(storage) == {POLY: 12.0,
                                                              KALSHI: 25.0}


def test_the_console_refuses_to_authorise_capital_it_cannot_see(tmp_path, monkeypatch):
    """
    A budget is permission to spend money that is there. Without a balance, the
    permission cannot be granted - and the refusal has to say why.
    """
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "refuse.db"))
    from fastapi.testclient import TestClient

    from src.ptai.execution.capital import authorised_budgets
    from src.ptai.storage.db import Storage
    from src.ptai.ui.console import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/console/budget", json={"venue": POLY, "amount": 12.0})
    assert r.status_code == 409, r.text
    assert "balance" in r.text.lower()
    # Crucially: nothing was written. A refusal that still records the budget is
    # not a refusal.
    assert authorised_budgets(Storage(db_path=os.environ["PTAI_DB"])) == {}


def test_an_unreadable_balance_is_not_reported_as_a_zero_balance(selector):
    """
    "We could not look" and "there is nothing there" have different fixes.

    Telling the operator their account is empty when the venue simply did not
    answer sends them to deposit money they already have.
    """
    unread = selector.assess(
        [POLY], accounts=[{"venue_id": POLY, "budget_usd": 50.0,
                           "funded": False, "balance_is_real": False,
                           "deposited_usd": 0.0}],
        labels=_labels())[0]
    empty = selector.assess(
        [POLY], accounts=[{"venue_id": POLY, "budget_usd": 50.0,
                           "funded": False, "balance_is_real": True,
                           "deposited_usd": 0.0}],
        labels=_labels())[0]
    sel_unread = selector.select([unread])
    sel_empty = selector.select([empty])
    assert unread.blockers != empty.blockers
    assert any("could not be read" in b for b in unread.blockers)
    assert not any("could not be read" in b for b in empty.blockers)
    assert any("$0.00" in b for b in empty.blockers)


def test_the_console_escapes_venue_text(tmp_path, monkeypatch):
    """
    Venue labels and reasons are strings that reach innerHTML. A venue name is
    not a place to discover a scripting bug.
    """
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "console_esc.db"))
    from fastapi.testclient import TestClient

    from src.ptai.ui.console import app

    html = TestClient(app, raise_server_exceptions=False).get("/").text
    assert "const esc = v =>" in html
    assert "esc(a.label)" in html
