"""
The hard limits, and the fact that they are in the way of an order.

PTAI advertised four kinds of wipeout protection and enforced none of them:

  * `Settings.max_daily_loss_pct` - read by nothing,
  * `CircuitBreaker(daily_loss_limit=...)` - constructed by a dashboard, never
    consulted by the trading loop,
  * `DrawdownManager` - constructed in the loop and used by nothing after that,
  * `KillSwitch.check_daily_loss(...)` and `check_drawdown(...)` - defined, and
    called by nothing, so two of the twelve triggers the module lists could not
    fire whatever happened to the account.

The sibling avt-bot project consults its bankroll guard BEFORE every stake and
persists the limits across restarts. These tests pin the same property here:
a limit that cannot be found in the path of an order does not exist.

The two design rules worth remembering:

  * limits are DERIVED from settled outcomes in the trade log, never a counter
    in memory - that is what makes a restart unable to reset them; and
  * they are SCOPED to the day and the session, and lift on their own. The
    latched kill switch stays a separate thing, because a loss limit that only a
    human can clear is a loss limit that stops an unattended agent forever.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ptai.learning.trade_outcomes import TradeOutcomeTracker
from src.ptai.risk.money_guard import (
    LANE_LIVE,
    LANE_PAPER,
    MoneyGuard,
    default_guard,
    guard_snapshot,
)
from src.ptai.storage.db import Storage

BANKROLL = 50.0


@pytest.fixture()
def storage(tmp_path):
    store = Storage(db_path=str(tmp_path / "guard.db"))
    store.set_bankroll(BANKROLL)
    yield store
    store.close()


def _settle(storage, pnl: float, *, lane: str = "paper", i: int = 0,
            venue: str = "polymarket", amount: float = 3.0,
            resolved_at: str | None = None):
    """One settled outcome, written the way the settlement pass writes it."""
    tracker = TradeOutcomeTracker(storage=storage)
    trade_id = f"{lane}-{venue}-{i}"
    tracker.record_trade(
        trade_id=trade_id, market_id=f"M{i}", venue_id=venue, strategy="value",
        category="politics", forecast_prob=0.6, market_price=0.5, edge=0.1,
        side="YES", amount_usd=amount, execution_mode=lane, yes_price=0.5,
    )
    tracker.record_resolution(trade_id, actual_outcome=0.0 if pnl < 0 else 1.0,
                              pnl=pnl)
    if resolved_at:
        storage.conn.execute("UPDATE trade_outcomes SET resolved_at = ? "
                             "WHERE trade_id = ?", (resolved_at, trade_id))
        storage.conn.commit()
    return trade_id


# ---------------------------------------------------------------------------
# 1. the limit is real, and it is measured from the record
# ---------------------------------------------------------------------------

class TestTheLimitIsMeasuredFromTheTradeLog:
    def test_nothing_settled_means_nothing_used(self, storage):
        guard = MoneyGuard(storage)
        usage = guard.usage(LANE_PAPER, BANKROLL)

        assert usage.daily_pnl_usd == 0.0
        assert usage.daily_used_pct == 0.0
        assert usage.halted is False
        assert usage.daily_loss_limit_usd == pytest.approx(BANKROLL * 0.15)

    def test_losses_accumulate_against_the_daily_limit(self, storage):
        guard = MoneyGuard(storage)
        for i in range(2):
            _settle(storage, -3.0, i=i)

        usage = guard.usage(LANE_PAPER, BANKROLL)
        assert usage.daily_pnl_usd == pytest.approx(-6.0)
        assert usage.daily_used_pct == pytest.approx(6.0 / 7.5)
        assert usage.halted is False, "two losing trades is not a wipeout"

    def test_the_limit_stops_the_lane(self, storage):
        guard = MoneyGuard(storage)
        for i in range(3):
            _settle(storage, -3.0, i=i)

        decision = guard.check(3.0, LANE_PAPER, BANKROLL)
        assert decision.refused is True
        assert decision.approved_usd == 0.0
        assert decision.binding == "daily"
        assert "daily limit" in decision.reason
        assert "lifts when the day rolls" in decision.reason

    def test_a_win_does_not_buy_back_the_limit(self, storage):
        """
        A loss limit measures LOSSES. A profitable morning must not licence a
        losing afternoon of the same size - the fraction is of the loss, not of
        the net, and the day's ceiling does not rise because earlier trades won.
        """
        guard = MoneyGuard(storage)
        _settle(storage, +12.0, i=0)
        for i in range(1, 3):
            _settle(storage, -3.0, i=i)

        usage = guard.usage(LANE_PAPER, BANKROLL)
        assert usage.daily_pnl_usd == pytest.approx(+6.0)
        assert usage.daily_used_pct == pytest.approx(0.0), (
            "the limit is about losses; a net-positive day has used none of it")
        assert usage.halted is False

    def test_the_limit_lifts_when_the_day_rolls(self, storage):
        guard = MoneyGuard(storage)
        now = datetime(2026, 9, 27, 23, 0, tzinfo=timezone.utc)
        for i in range(3):
            _settle(storage, -3.0, i=i,
                    resolved_at=(now - timedelta(hours=1)).isoformat())

        assert guard.usage(LANE_PAPER, BANKROLL, now=now).halted is True
        tomorrow = now + timedelta(hours=3)
        assert guard.usage(LANE_PAPER, BANKROLL, now=tomorrow).halted is False, (
            "the daily stop is scoped to the day and must clear itself")

    def test_a_restart_does_not_reset_the_limit(self, storage, tmp_path):
        """The property the whole design exists for."""
        db = str(storage.db_path)
        for i in range(3):
            _settle(storage, -3.0, i=i)
        storage.conn.commit()

        # A second process, reading the same database.
        reopened = Storage(db_path=db)
        try:
            usage = MoneyGuard(reopened).usage(LANE_PAPER, BANKROLL)
        finally:
            reopened.close()

        assert usage.halted is True, (
            "restarting the agent must not hand the strategy a fresh loss budget")
        assert usage.daily_pnl_usd == pytest.approx(-9.0)

    def test_the_session_and_the_day_are_separate(self, storage):
        """
        A loss yesterday is not a loss today, but it is still this session's.

        The session begins when the operator starts one, so a session that spans
        midnight keeps its own ledger - which is the difference between "stop for
        the day" and "stop this run".
        """
        guard = MoneyGuard(storage)
        now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
        guard.session_started_at(LANE_PAPER, now=now - timedelta(days=1))
        for i in range(3):
            _settle(storage, -3.0, i=i,
                    resolved_at=(now - timedelta(hours=20)).isoformat())

        usage = guard.usage(LANE_PAPER, BANKROLL, now=now)
        assert usage.daily_pnl_usd == pytest.approx(0.0), "yesterday's loss"
        assert usage.session_pnl_usd == pytest.approx(-9.0), "still this session's"
        assert usage.session_used_pct == pytest.approx(9.0 / 15.0)
        assert usage.halted is False, (
            "two thirds of the session limit is not a stop; the session stop "
            "comes at the 30% of capital the limit names")

        for i in range(3, 5):
            _settle(storage, -3.0, i=i,
                    resolved_at=(now - timedelta(hours=20)).isoformat())
        halted = guard.usage(LANE_PAPER, BANKROLL, now=now)
        assert halted.session_pnl_usd == pytest.approx(-15.0)
        assert halted.halted is True
        assert halted.daily_used_pct == pytest.approx(0.0), (
            "the day is untouched - it is the SESSION that is spent")
        assert "session limit" in halted.halt_reason

    def test_the_lanes_do_not_share_a_limit(self, storage):
        """A paper loss must not spend the live loss budget, or hide in it."""
        guard = MoneyGuard(storage)
        for i in range(4):
            _settle(storage, -3.0, lane="paper", i=i)

        assert guard.usage(LANE_PAPER, BANKROLL).halted is True
        live = guard.usage(LANE_LIVE, BANKROLL)
        assert live.halted is False
        assert live.daily_pnl_usd == 0.0


# ---------------------------------------------------------------------------
# 2. what the guard allows, and what it refuses
# ---------------------------------------------------------------------------

class TestTheDecisionIsAnAnswerNotAWarning:
    def test_a_clean_record_is_approved(self, storage):
        decision = MoneyGuard(storage).check(3.0, LANE_PAPER, BANKROLL)
        assert decision.refused is False
        assert decision.approved_usd == pytest.approx(3.0)

    def test_headroom_trims_rather_than_refuses(self, storage):
        """
        A stake bigger than the remaining budget is trimmed to it.

        The trade is still worth taking, just smaller - the same reasoning that
        trims to the authorised capital cap at dispatch. Refusing here would drop
        a trade the account can afford at a smaller size.
        """
        guard = MoneyGuard(storage)
        for i in range(2):
            _settle(storage, -3.0, i=i)  # $6 of the $7.50 daily budget used

        decision = guard.check(3.0, LANE_PAPER, BANKROLL)
        assert decision.refused is False
        assert decision.approved_usd == pytest.approx(1.5, abs=0.01)
        assert "trimmed" in " ".join(decision.notes)

    def test_below_the_minimum_order_it_is_a_refusal(self, storage):
        guard = MoneyGuard(storage)
        for i in range(2):
            _settle(storage, -3.0, i=i)
        _settle(storage, -0.9, i=99)  # $6.90 of $7.50 used, $0.60 left

        decision = guard.check(3.0, LANE_PAPER, BANKROLL)
        assert decision.refused is True
        assert "minimum order size" in decision.reason

    def test_a_zero_proposal_is_refused(self, storage):
        decision = MoneyGuard(storage).check(0.0, LANE_PAPER, BANKROLL)
        assert decision.refused is True
        assert decision.binding == "stake"

    def test_an_unknown_bankroll_cannot_justify_a_size(self, storage):
        decision = MoneyGuard(storage).check(3.0, LANE_LIVE, 0.0, proven=True)
        assert decision.refused is True
        assert "bankroll is unknown" in decision.reason

    def test_an_unreadable_record_refuses_real_money(self, tmp_path):
        """
        Fail closed on the lane that spends, open on the lane that learns.

        An unreadable loss record is a reason not to spend real money. It is not
        a reason to stop the simulation, which risks nothing but inference - and
        stopping it would mean a database hiccup quietly ends the paper record
        that earns the qualification.
        """
        storage = Storage(db_path=str(tmp_path / "broken.db"))
        try:
            storage.conn.execute("DROP TABLE trade_outcomes")
            storage.conn.commit()
            guard = MoneyGuard(storage)

            live = guard.check(3.0, LANE_LIVE, BANKROLL, proven=True)
            assert live.refused is True
            assert live.binding == "unknown"
            assert "unknown limit" in live.reason

            paper = guard.usage(LANE_PAPER, BANKROLL)
            assert paper.unknown is True
            assert paper.halted is False, (
                "an unreadable record is not a reached limit")
            decision = guard.check(3.0, LANE_PAPER, BANKROLL)
            assert decision.refused is False, (
                "the simulation must survive a read failure")
            assert decision.notes, "the simulation says what it could not read"
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# 3. first live money is small
# ---------------------------------------------------------------------------

class TestTheFirstLiveMoneyIsCapped:
    def test_live_is_micro_until_it_has_settled_real_trades(self, storage):
        guard = MoneyGuard(storage)
        assert guard.is_proven_live("polymarket") is False

        decision = guard.check(6.0, LANE_LIVE, BANKROLL, venue_id="polymarket")
        assert decision.tier == "micro"
        assert decision.approved_usd == pytest.approx(BANKROLL * 0.02)
        assert "first live money" in " ".join(decision.notes)

    def test_paper_is_not_held_back_by_the_live_cap(self, storage):
        """Paper sizing is unchanged: simulation is not the risky lane."""
        decision = MoneyGuard(storage).check(3.0, LANE_PAPER, BANKROLL)
        assert decision.approved_usd == pytest.approx(3.0)
        assert decision.tier == "armed"

    def test_a_settled_live_record_earns_the_full_cap(self, storage):
        guard = MoneyGuard(storage)
        for i in range(guard.live_proving_trades):
            _settle(storage, +0.5, lane="live", i=i)

        assert guard.is_proven_live("polymarket") is True
        decision = guard.check(6.0, LANE_LIVE, BANKROLL, venue_id="polymarket")
        assert decision.tier == "armed"
        assert decision.approved_usd == pytest.approx(3.0), (
            "6% of $50 per order")

    def test_a_proving_record_at_another_venue_does_not_count(
            self, storage):
        guard = MoneyGuard(storage)
        for i in range(guard.live_proving_trades):
            _settle(storage, +0.5, lane="live", i=i, venue="kalshi")

        assert guard.is_proven_live("polymarket") is False, (
            "proving the path at one venue says nothing about another")


# ---------------------------------------------------------------------------
# 4. it is wired into the order path, and the operator can see it
# ---------------------------------------------------------------------------

class TestItIsInTheWayOfTheOrder:
    def test_the_loop_consults_the_guard_before_dispatch(self):
        source = Path("src/ptai/agent/v3_loop.py").read_text()
        dispatch = source.index("lane, lane_reason = self._money_lane(")
        guard_call = source.index("guard_decision = self.money_guard.check(")
        executor = source.index("_execute_with_side_aware_cap(", dispatch)
        assert dispatch < guard_call < executor, (
            "the guard must be asked between choosing the purse and placing the "
            "order; anywhere else and it is advice, not a limit")

    def test_a_refused_live_order_still_runs_in_paper(self):
        """
        The boundary forbids REAL money, it does not forbid the trade.

        A guard that dropped the order would stop the paper record on the day the
        account had a bad run - exactly when the simulation is most informative.
        """
        source = Path("src/ptai/agent/v3_loop.py").read_text()
        block = source[source.index("if guard_decision.refused:") :]
        block = block[: block.index("elif guard_decision.approved_usd")]
        assert 'lane = "paper"' in block
        assert "paper_because = f\"money guard:" in block
        assert "blocked_loss_limit" in block, (
            "a stopped PAPER lane is a refusal, and says so")

    def test_the_paper_lane_can_be_stopped_and_the_order_is_not_placed(self):
        source = Path("src/ptai/agent/v3_loop.py").read_text()
        block = source[source.index("if guard_decision.refused:") :]
        block = block[: block.index("elif guard_decision.approved_usd")]
        assert "continue" in block, (
            "a stopped lane must not fall through to placing the order")

    def test_the_kill_switch_triggers_that_were_never_called_now_are(self):
        source = Path("src/ptai/agent/v3_loop.py").read_text()
        flat = " ".join(source.split())
        assert "self.kill_switch.check_drawdown(live.loss_pct_of_bankroll)" in flat
        # ...and the daily trigger stays unfed on purpose: the guard scopes it
        # and lifts it, while the kill switch latches until a human resets it.
        assert "check_daily_loss(live.daily_used_pct)" not in flat
        assert "already stops the live lane for the day" in flat

    def test_the_operator_payload_carries_the_limits(self, storage):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        snapshot = operator_snapshot(storage)
        assert snapshot["limits"]["paper"]["daily_loss_limit_usd"] == pytest.approx(7.5)

        lines = describe_snapshot(snapshot)
        assert any(line.startswith("Loss limit: paper") for line in lines)

    def test_the_operator_lines_shout_when_a_lane_is_stopped(self, storage):
        from src.ptai.operator_view import describe_snapshot, operator_snapshot

        for i in range(3):
            _settle(storage, -3.0, i=i)

        lines = describe_snapshot(operator_snapshot(storage))
        stopped = [line for line in lines if line.startswith("Loss limit:")]
        assert stopped and "STOPPED" in stopped[0]
        assert "PAPER" in stopped[0]

    def test_the_doctor_asks_the_same_guard(self, tmp_path):
        from src.ptai.doctor import _limits_check

        check = _limits_check(tmp_path / "does-not-exist.db")
        assert check.name == "Loss limits"
        assert "enforced before every order" in check.detail

    def test_the_console_panel_renders_the_limits(self):
        from src.ptai.ui import console as console_module

        html = console_module.CONSOLE_HTML
        assert 'id="limits"' in html
        assert "Loss limits" in html
        script = html.split("<script>")[1]
        assert "body.limits" in script
        assert "Daily loss limit" in script and "Session loss limit" in script


# ---------------------------------------------------------------------------
# 5. the session belongs to the operator
# ---------------------------------------------------------------------------

class TestTheSessionIsTheOperatorsToRestart:
    def test_authorising_a_budget_starts_a_fresh_live_session(self, storage):
        from src.ptai.execution.capital import set_authorised_budget
        from src.ptai.risk.money_guard import MoneyGuard as Guard

        old = Guard(storage).session_started_at(LANE_LIVE)
        set_authorised_budget(storage, "polymarket", 50.0)
        new = Guard(storage).session_started_at(LANE_LIVE)
        assert new > old, (
            "the operator saying 'start from here' is the one act that may reset "
            "a limit a restart cannot")

    def test_switching_mode_starts_a_fresh_session_for_that_lane(self, storage):
        from src.ptai.execution.capital import set_operator_mode
        from src.ptai.risk.money_guard import MoneyGuard as Guard

        before = Guard(storage).session_started_at(LANE_PAPER)
        set_operator_mode(storage, "live")
        set_operator_mode(storage, "paper")
        after = Guard(storage).session_started_at(LANE_PAPER)
        assert after > before

    def test_the_session_start_is_persisted(self, storage, tmp_path):
        guard = MoneyGuard(storage)
        started = guard.session_started_at(LANE_LIVE)
        storage.conn.commit()

        reopened = Storage(db_path=str(storage.db_path))
        try:
            again = MoneyGuard(reopened).session_started_at(LANE_LIVE)
        finally:
            reopened.close()
        assert again == started

    def test_the_settings_value_is_the_one_used(self, tmp_path):
        """
        `max_daily_loss_pct` was read by nothing. A limit that exists only in
        configuration is worse than no limit, because the setting says the
        account is protected.
        """
        class Settings:
            max_daily_loss_pct = 0.05

        storage = Storage(db_path=str(tmp_path / "cfg.db"))
        try:
            guard = default_guard(storage, Settings())
            assert guard.daily_loss_pct == pytest.approx(0.05)
            assert guard_snapshot(storage, settings=Settings())["daily_loss_pct"] \
                == pytest.approx(0.05)
        finally:
            storage.close()
