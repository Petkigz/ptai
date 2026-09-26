"""
The money guard: the hard limits that stand between a bad run and a wiped account.

PTAI had all of this on paper and none of it in the path of an order:

  * `CircuitBreaker` carried a `daily_loss_limit` and had no production caller,
  * `DrawdownManager` was constructed in the trading loop and used by nothing,
  * `KillSwitch.check_daily_loss(...)` and `check_drawdown(...)` existed and were
    never called - so two of the twelve triggers the module advertises could not
    fire, whatever happened to the account,
  * and `max_daily_loss_pct` sat in Settings, read by nobody.

The sibling avt-bot project does the opposite: its bankroll guard is consulted
BEFORE every stake, its session and daily loss limits are persisted across
restarts, and `approveStake()` returns the capped amount (or zero) rather than a
warning. This module is that idea, built on PTAI's own record.

Two rules shape it:

  1. **Derived, not remembered.** The limits are computed from the settled
     outcomes already in the database - per lane, per day - instead of an
     in-memory counter. A restart, a crash or a `Ctrl+C` cannot reset a loss
     limit, and there is no second copy of the P&L that can disagree with the
     trade log.

  2. **Scoped, not latched.** The guard refuses for the day or for the session it
     belongs to, and lifts on its own when the day rolls or the operator starts a
     new session. The latched, human-reset kill switch stays a separate thing,
     because a paper loss must never leave the system stopped forever waiting for
     somebody to notice - that would be the opposite of running unattended.

One consequence worth stating plainly: the limits apply to BOTH lanes. Real
money is the point, but a paper record that ignores the stop real money would
have hit is not evidence about the strategy - it is evidence about a simulation
with the safety rails removed. The paper bankroll gets the same limits, so a
paper run that would have been stopped, stops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from loguru import logger

# The two purses. Real money may only ever come from LIVE; everything else is
# simulated, and the guard tracks them apart so a paper loss can never consume a
# live limit or hide behind one.
LANE_LIVE = "live"
LANE_PAPER = "paper"

# Stake policy. `max_stake_pct` is the per-order cap the allocator already used
# (6% of the bankroll - the rule the operator set). `micro_stake_pct` is what the
# FIRST live money is limited to: one real venue through the full live cycle on
# very small capital comes before sizing live like a proven account.
DEFAULT_MAX_STAKE_PCT = 0.06
DEFAULT_MICRO_STAKE_PCT = 0.02
DEFAULT_MIN_STAKE_USD = 1.0
DEFAULT_SESSION_LOSS_PCT = 0.30
DEFAULT_DAILY_LOSS_PCT = 0.15
# How many settled LIVE trades a venue gets before it is no longer treated as
# unproven for the purpose of live sizing.
LIVE_PROVING_TRADES = 10


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class LaneUsage:
    """How much of each limit one lane has used, and whether it is stopped."""

    lane: str
    bankroll_usd: float = 0.0
    session_pnl_usd: float = 0.0
    session_trades: int = 0
    daily_pnl_usd: float = 0.0
    daily_trades: int = 0
    open_risk_usd: float = 0.0
    session_loss_limit_usd: Optional[float] = None
    daily_loss_limit_usd: Optional[float] = None
    session_used_pct: float = 0.0
    daily_used_pct: float = 0.0
    # The loss since the session began as a fraction of the bankroll. This is the
    # figure a drawdown trigger means - a fraction of capital, not a fraction of a
    # limit - so the kill switch's `drawdown_pct` threshold compares against
    # something in its own units.
    loss_pct_of_bankroll: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    unknown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "bankroll_usd": round(self.bankroll_usd, 2),
            "session_pnl_usd": round(self.session_pnl_usd, 2),
            "session_trades": self.session_trades,
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
            "daily_trades": self.daily_trades,
            "open_risk_usd": round(self.open_risk_usd, 2),
            "session_loss_limit_usd": (round(self.session_loss_limit_usd, 2)
                                       if self.session_loss_limit_usd else None),
            "daily_loss_limit_usd": (round(self.daily_loss_limit_usd, 2)
                                     if self.daily_loss_limit_usd else None),
            "session_used_pct": round(self.session_used_pct, 4),
            "daily_used_pct": round(self.daily_used_pct, 4),
            "loss_pct_of_bankroll": round(self.loss_pct_of_bankroll, 4),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "unknown": self.unknown,
        }


@dataclass
class GuardDecision:
    """What the guard will allow, and why - the answer, not a warning."""

    approved_usd: float
    lane: str
    tier: str = "micro"
    refused: bool = False
    reason: str = ""
    binding: str = ""            # which limit bound: session | daily | stake | none
    usage: Optional[LaneUsage] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved_usd": round(self.approved_usd, 2),
            "lane": self.lane,
            "tier": self.tier,
            "refused": self.refused,
            "reason": self.reason,
            "binding": self.binding,
            "usage": self.usage.to_dict() if self.usage else None,
        }


class MoneyGuard:
    """
    Session and daily loss limits, per lane, reconstructed from the trade log.

    `storage` is the one source of settled P&L. The guard never keeps a running
    total of its own, which is the property that makes it survive a restart: the
    numbers it compares are read fresh from the outcomes the settlement pass
    wrote.
    """

    SESSION_KEY = "risk.session_started_at.{lane}"

    def __init__(self, storage=None, *,
                 session_loss_pct: float = DEFAULT_SESSION_LOSS_PCT,
                 daily_loss_pct: float = DEFAULT_DAILY_LOSS_PCT,
                 max_stake_pct: float = DEFAULT_MAX_STAKE_PCT,
                 micro_stake_pct: float = DEFAULT_MICRO_STAKE_PCT,
                 min_stake_usd: float = DEFAULT_MIN_STAKE_USD,
                 live_proving_trades: int = LIVE_PROVING_TRADES):
        self.storage = storage
        self.session_loss_pct = float(session_loss_pct)
        self.daily_loss_pct = float(daily_loss_pct)
        self.max_stake_pct = float(max_stake_pct)
        self.micro_stake_pct = float(micro_stake_pct)
        self.min_stake_usd = float(min_stake_usd)
        self.live_proving_trades = int(live_proving_trades)

    # ------------------------------------------------------------------
    # the record the limits are measured against
    # ------------------------------------------------------------------

    def _state_get(self, key: str) -> Optional[str]:
        try:
            return self.storage.get_state(key) if self.storage else None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Money guard could not read state {key}: {e}")
            return None

    def _state_set(self, key: str, value: str) -> None:
        try:
            if self.storage:
                self.storage.set_state(key, value)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Money guard could not write state {key}: {e}")

    def session_started_at(self, lane: str, *, now: Optional[datetime] = None) -> datetime:
        """
        When this lane's session began.

        Persisted, so a restart resumes the same session instead of handing the
        strategy a fresh loss budget - which is exactly how a loss limit gets
        quietly reset by the thing it is supposed to survive. Started by the
        operator's own acts (choosing a mode, authorising a budget), never by the
        clock.
        """
        key = self.SESSION_KEY.format(lane=lane)
        stored = _as_dt(self._state_get(key))
        if stored is not None:
            return stored
        started = now or _utcnow()
        self._state_set(key, started.isoformat())
        logger.info(f"Money guard: {lane} session started {started.isoformat()}")
        return started

    def start_session(self, lane: str, reason: str = "") -> datetime:
        """Begin a new session for a lane - an operator act, not an automatic one."""
        started = _utcnow()
        self._state_set(self.SESSION_KEY.format(lane=lane), started.isoformat())
        logger.info(f"Money guard: {lane} session restarted"
                    + (f" ({reason})" if reason else ""))
        return started

    def _settled(self, lane: str, since: Optional[datetime] = None,
                 day: Optional[str] = None) -> Dict[str, Any]:
        """
        Settled outcomes for one lane, from the trade log.

        Returns `unknown=True` when the record cannot be read. Unknown loss
        state must never authorise real money, so the caller fails closed - but
        a database hiccup must not silently stop the simulation either.
        """
        if self.storage is None:
            return {"unknown": True, "reason": "no storage"}
        clauses = ["execution_mode = ?", "resolved_at IS NOT NULL",
                   "pnl IS NOT NULL"]
        params: list = [lane]
        if since is not None:
            clauses.append("resolved_at >= ?")
            params.append(since.isoformat())
        if day is not None:
            clauses.append("substr(resolved_at, 1, 10) = ?")
            params.append(day)
        sql = ("SELECT COALESCE(SUM(pnl), 0.0), COUNT(*) FROM trade_outcomes "
               "WHERE " + " AND ".join(clauses))
        try:
            row = self.storage.conn.execute(sql, params).fetchone()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Money guard could not read {lane} outcomes: "
                         f"{type(e).__name__}: {e}")
            return {"unknown": True, "reason": f"{type(e).__name__}: {e}"}
        return {"unknown": False, "pnl_usd": float(row[0] or 0.0),
                "trades": int(row[1] or 0), "since": since, "day": day}

    def _open_risk_usd(self, lane: str) -> float:
        """
        Money committed to positions that have not settled.

        A loss limit that only counts settled losses is a limit that is always one
        settlement behind: three open positions can all be losers at once. This is
        what makes the guard bind on exposure rather than on history.
        """
        if self.storage is None:
            return 0.0
        try:
            row = self.storage.conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0.0) FROM trade_outcomes "
                "WHERE execution_mode = ? AND resolved_at IS NULL",
                (lane,)).fetchone()
            return float(row[0] or 0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Money guard could not total open {lane} risk: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # usage
    # ------------------------------------------------------------------

    def usage(self, lane: str, bankroll_usd: float,
              *, now: Optional[datetime] = None) -> LaneUsage:
        """How much of the session and daily limits this lane has used."""
        now = now or _utcnow()
        started = self.session_started_at(lane, now=now)
        day = now.date().isoformat()

        session = self._settled(lane, since=started)
        daily = self._settled(lane, day=day)
        unknown = bool(session.get("unknown") or daily.get("unknown"))

        bankroll = float(bankroll_usd or 0.0)
        session_limit = (self.session_loss_pct * bankroll) if bankroll > 0 else None
        daily_limit = (self.daily_loss_pct * bankroll) if bankroll > 0 else None

        usage = LaneUsage(
            lane=lane,
            bankroll_usd=bankroll,
            session_pnl_usd=float(session.get("pnl_usd") or 0.0),
            session_trades=int(session.get("trades") or 0),
            daily_pnl_usd=float(daily.get("pnl_usd") or 0.0),
            daily_trades=int(daily.get("trades") or 0),
            open_risk_usd=self._open_risk_usd(lane),
            session_loss_limit_usd=session_limit,
            daily_loss_limit_usd=daily_limit,
            unknown=unknown,
        )

        # Losses are what a loss limit measures: a positive day does not bank
        # credit against tomorrow's limit, and it must not dilute the fraction.
        session_loss = max(0.0, -usage.session_pnl_usd)
        daily_loss = max(0.0, -usage.daily_pnl_usd)
        if bankroll > 0:
            usage.loss_pct_of_bankroll = session_loss / bankroll
        if session_limit:
            usage.session_used_pct = min(1.0, session_loss / session_limit)
        if daily_limit:
            usage.daily_used_pct = min(1.0, daily_loss / daily_limit)

        if session_limit is not None and session_loss >= session_limit:
            usage.halted = True
            usage.halt_reason = (
                f"{lane} session loss {usage.session_pnl_usd:+.2f} has reached the "
                f"{self.session_loss_pct * 100:.0f}% session limit "
                f"(-${session_limit:.2f} of ${bankroll:.2f})")
        if daily_limit is not None and daily_loss >= daily_limit:
            usage.halted = True
            usage.halt_reason = (
                f"{lane} loss today {usage.daily_pnl_usd:+.2f} has reached the "
                f"{self.daily_loss_pct * 100:.0f}% daily limit "
                f"(-${daily_limit:.2f} of ${bankroll:.2f}); it lifts when the day "
                f"rolls, and nothing else about the agent changes")
        return usage

    # ------------------------------------------------------------------
    # the decision
    # ------------------------------------------------------------------

    def stake_cap(self, lane: str, bankroll_usd: float, *,
                  proven: bool = True) -> float:
        """
        The largest single order this lane may place, before the limits.

        The first live money is deliberately smaller: an account with no settled
        live trades has not shown that its orders fill, so it gets the micro cap
        until it has.
        """
        bankroll = float(bankroll_usd or 0.0)
        if bankroll <= 0:
            return 0.0
        pct = self.micro_stake_pct if (lane == LANE_LIVE and not proven) else self.max_stake_pct
        return round(bankroll * pct, 2)

    def is_proven_live(self, venue_id: Optional[str] = None) -> bool:
        """Has live money at this venue completed enough cycles to size normally?"""
        if self.storage is None:
            return False
        try:
            if venue_id:
                row = self.storage.conn.execute(
                    "SELECT COUNT(*) FROM trade_outcomes WHERE execution_mode = ? "
                    "AND venue_id = ? AND resolved_at IS NOT NULL",
                    (LANE_LIVE, venue_id)).fetchone()
            else:
                row = self.storage.conn.execute(
                    "SELECT COUNT(*) FROM trade_outcomes WHERE execution_mode = ? "
                    "AND resolved_at IS NOT NULL", (LANE_LIVE,)).fetchone()
            return int(row[0] or 0) >= self.live_proving_trades
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Money guard could not count live trades: {e}")
            return False

    def check(self, amount_usd: float, lane: str, bankroll_usd: float, *,
              venue_id: Optional[str] = None, proven: Optional[bool] = None,
              now: Optional[datetime] = None) -> GuardDecision:
        """
        The answer for one proposed order: what is approved, or why nothing is.

        Fails closed for real money and open for simulation - an unreadable loss
        record is a reason not to spend, never a reason to stop learning. The
        refusal is scoped to this cycle; the next cycle asks again.
        """
        lane = LANE_LIVE if str(lane).lower() == LANE_LIVE else LANE_PAPER
        usage = self.usage(lane, bankroll_usd, now=now)
        decision = GuardDecision(approved_usd=0.0, lane=lane, usage=usage)

        if proven is None:
            proven = (True if lane == LANE_PAPER
                      else self.is_proven_live(venue_id))

        if usage.unknown:
            if lane == LANE_LIVE:
                decision.refused = True
                decision.binding = "unknown"
                decision.reason = (
                    "the settled-loss record could not be read, so the loss "
                    "limits cannot be checked; real money is not spent on an "
                    "unknown limit")
                return decision
            decision.notes.append(
                "the settled-loss record could not be read; running the "
                "simulation, which risks nothing but inference")
        elif usage.halted:
            decision.refused = True
            decision.binding = "daily" if usage.daily_used_pct >= 1.0 else "session"
            decision.reason = usage.halt_reason
            return decision

        try:
            proposed = float(amount_usd or 0.0)
        except (TypeError, ValueError):
            proposed = 0.0
        if proposed <= 0:
            decision.refused = True
            decision.binding = "stake"
            decision.reason = "no positive amount was proposed"
            return decision

        if usage.bankroll_usd <= 0:
            decision.refused = True
            decision.binding = "stake"
            decision.reason = (f"the {lane} bankroll is unknown, so no size can "
                               f"be justified against it")
            return decision

        cap = self.stake_cap(lane, usage.bankroll_usd, proven=proven)
        tier = "armed" if proven else "micro"
        decision.tier = tier
        # Headroom is the part of the limit that has not been used yet. A stake
        # larger than the remaining budget is trimmed to it rather than refused:
        # the trade is still worth taking, just smaller - the same reasoning that
        # trims to the authorised capital cap at dispatch.
        headroom = None
        if usage.session_loss_limit_usd:
            headroom = max(0.0, usage.session_loss_limit_usd
                           - max(0.0, -usage.session_pnl_usd))
        if usage.daily_loss_limit_usd:
            daily_headroom = max(0.0, usage.daily_loss_limit_usd
                                 - max(0.0, -usage.daily_pnl_usd))
            headroom = daily_headroom if headroom is None else min(headroom, daily_headroom)

        approved = min(proposed, cap)
        if headroom is not None and approved > headroom:
            approved = headroom
            decision.binding = "daily" if (usage.daily_used_pct
                                           >= usage.session_used_pct) else "session"
            decision.notes.append(
                f"trimmed to ${headroom:.2f}, the part of the loss limit still "
                f"unused")
        if lane == LANE_LIVE and not proven:
            decision.notes.append(
                f"first live money: capped at {self.micro_stake_pct * 100:.0f}% "
                f"of capital until {self.live_proving_trades} live trade(s) at "
                f"this venue have settled")

        if approved < self.min_stake_usd:
            decision.refused = True
            decision.binding = decision.binding or "stake"
            decision.reason = (
                f"${approved:.2f} is below the ${self.min_stake_usd:.2f} minimum "
                f"order size (cap ${cap:.2f}"
                + (f", ${headroom:.2f} of loss limit left" if headroom is not None
                   else "") + ")")
            return decision

        decision.approved_usd = round(approved, 2)
        return decision

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def snapshot(self, *, live_bankroll_usd: float = 0.0,
                 paper_bankroll_usd: float = 0.0,
                 venue_id: Optional[str] = None) -> Dict[str, Any]:
        """Both lanes, for the console and `ptai status`."""
        return {
            "source": "trade_outcomes (settled P&L), per lane and per day",
            "session_loss_pct": self.session_loss_pct,
            "daily_loss_pct": self.daily_loss_pct,
            "max_stake_pct": self.max_stake_pct,
            "micro_stake_pct": self.micro_stake_pct,
            "live_proving_trades": self.live_proving_trades,
            "live": self.usage(LANE_LIVE, live_bankroll_usd).to_dict(),
            "paper": self.usage(LANE_PAPER, paper_bankroll_usd).to_dict(),
            "live_proven": self.is_proven_live(venue_id),
            "note": ("limits are measured against settled outcomes in the trade "
                     "log, so no restart resets them; they lift when the day "
                     "rolls or the operator starts a new session"),
        }


def default_guard(storage=None, settings=None) -> MoneyGuard:
    """
    A guard with the project's own thresholds, where they exist.

    `Settings.max_daily_loss_pct` was read by nothing before this; a limit that
    exists in configuration and nowhere else is worse than no limit, because the
    setting says the account is protected.
    """
    daily = DEFAULT_DAILY_LOSS_PCT
    if settings is not None:
        configured = getattr(settings, "max_daily_loss_pct", None)
        if isinstance(configured, (int, float)) and configured > 0:
            daily = float(configured)
    return MoneyGuard(storage, daily_loss_pct=daily)


def guard_snapshot(storage, *, live_bankroll_usd: float = 0.0,
                   paper_bankroll_usd: float = 0.0,
                   settings=None, venue_id: Optional[str] = None) -> Dict[str, Any]:
    """One call for the readers that only want the numbers."""
    guard = default_guard(storage, settings)
    return guard.snapshot(live_bankroll_usd=live_bankroll_usd,
                          paper_bankroll_usd=paper_bankroll_usd,
                          venue_id=venue_id)
