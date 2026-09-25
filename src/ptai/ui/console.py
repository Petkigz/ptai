"""
PTAI Console - the control panel for the live and paper system.

What this screen is FOR, in one sentence: to show the operator what the agent has
done with their money, and to let them change the two things that decide how much
is at risk - the mode, and the budget per venue.

Design rules, each one a reaction to a specific way this goes wrong:

  * LIVE and PAPER are never ambiguous. The mode is a banner in the header, not a
    setting buried in a form. An operator should never have to wonder which one
    is running.
  * Paper and live money are shown SEPARATELY. A paper position must not consume
    live budget, and a live position must not hide behind paper ones.
  * RESERVED capital is its own row. A resting order has committed cash without
    buying anything, so it is not a position and it is not available either.
  * Nothing is shown as verified unless it was verified. A balance that was not
    read is "unread", not zero. A venue with no probe is "unproven", not ready.
  * The funding panel explains how to put money in. See execution/capital.py for
    why the answer is "deposit at the venue" and not "send the agent money".
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from ..execution.capital import (
    CapitalLedger,
    FUNDING_ROUTES,
    UNFUNDABLE_SMALL,
    authorised_budgets,
    operator_mode,
    plan_for_budget,
    set_authorised_budget,
    set_operator_mode,
)
from ..storage.db import Storage
from ..strategy.venue_selection import MIN_SAMPLE_FOR_EVIDENCE, VenueSelector

app = FastAPI(title="PTAI Console", version="console-1")


# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------

class ConsoleState:
    """
    The operator's choices, persisted.

    Mode and budgets live in the database, not in a module global, because a
    restart must not silently revert a live account to paper - or worse, a paper
    account to live.
    """

    # Mode and budget are read and written through execution.capital, which is
    # the one definition of both, shared with the trading loop. This class no
    # longer owns a key namespace: a second copy of these key names is how the
    # screen ends up showing $50 authorised while the cycle sizes against $0.

    def __init__(self, storage: Storage):
        self.storage = storage

    @property
    def mode(self) -> str:
        return operator_mode(self.storage)

    @mode.setter
    def mode(self, value: str) -> None:
        set_operator_mode(self.storage, value)

    def budgets(self) -> Dict[str, float]:
        return authorised_budgets(self.storage)


def get_storage() -> Storage:
    import os
    return Storage(db_path=os.getenv("PTAI_DB", "./data/ptai.db"))


# ----------------------------------------------------------------------
# capital
# ----------------------------------------------------------------------

# A short cache. The console polls every 15 seconds and asks every venue for a
# balance; without this a browser tab left open becomes a sustained load on every
# venue's API, which is how an account gets rate limited for no reason.
_BALANCE_CACHE: Dict[str, Any] = {"at": 0.0, "values": {}}
_BALANCE_TTL_SECONDS = 20.0


async def _venue_balances(agent=None, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Ask each venue what the balance is. Never guesses.

    "Available: False" means the venue did not answer, and downstream that is
    treated as NOT FUNDED. The alternative - defaulting to whatever the operator
    typed into the budget box - is an agent that believes it has money because
    someone filled in a form.
    """
    import time as _time

    now = _time.monotonic()
    if not force and _BALANCE_CACHE["values"] and \
            now - _BALANCE_CACHE["at"] < _BALANCE_TTL_SECONDS:
        return dict(_BALANCE_CACHE["values"])

    balances: Dict[str, Dict[str, Any]] = {}
    registry = getattr(agent, "venue_registry", None)
    adapters = getattr(registry, "adapters", {}) if registry is not None else {}
    for venue_id, adapter in adapters.items():
        getter = getattr(adapter, "get_portfolio", None)
        if getter is None:
            continue
        try:
            result = getter()
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            balances[venue_id] = {"available": False, "balance": 0.0,
                                  "reason": f"{type(e).__name__}: {e}"}
            continue
        if isinstance(result, dict):
            balances[venue_id] = {
                "available": bool(result.get("available")),
                "balance": float(result.get("balance") or 0.0),
                "source": result.get("source", ""),
            }
    _BALANCE_CACHE["at"] = _time.monotonic()
    _BALANCE_CACHE["values"] = dict(balances)
    return balances


def _build_plan(state: ConsoleState, storage: Storage,
                balances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    budgets = state.budgets()
    ledger = CapitalLedger(storage=storage)
    plan = ledger.build(mode=state.mode, budgets=budgets,
                        venue_labels={v: r["label"] for v, r in FUNDING_ROUTES.items()},
                        balances=balances)
    return plan.to_dict()


@app.get("/api/console/capital")
async def api_capital() -> JSONResponse:
    storage = get_storage()
    state = ConsoleState(storage)
    balances = await _venue_balances(_agent())
    plan = _build_plan(state, storage, balances)
    plan["balances_read"] = {k: bool(v.get("available")) for k, v in balances.items()}
    plan["session"] = {"mode": state.mode, "budgets": state.budgets()}
    return JSONResponse(plan)


@app.post("/api/console/mode")
async def api_set_mode(request: Request) -> JSONResponse:
    """
    Switch between paper and live.

    Live mode is REFUSED unless something can actually be traded live: a funded,
    authorised venue. Letting the switch flip into live while nothing is
    verified would present an armed system that cannot fire, and worse, one the
    operator believes is trading.

    Live mode with no signer is refused for a second reason: without a key the
    agent cannot place, cancel or redeem anything, so "live" would mean watching
    a system that does nothing.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json()
    mode = str(body.get("mode", "")).lower()
    if mode not in ("paper", "live"):
        return JSONResponse(status_code=400, content={"error": "mode must be paper or live"})

    if mode == "live":
        balances = await _venue_balances(_agent())
        plan = _build_plan(state, storage, balances)
        blockers = []
        if not plan["live_venues"]:
            if not plan["accounts"]:
                blockers.append("no venue has a budget set")
            for account in plan["accounts"]:
                if not account["balance_is_real"]:
                    blockers.append(
                        f"{account['venue_id']}: the venue balance could not be read "
                        f"(no credentials, or the venue did not answer)")
                elif account["budget_usd"] <= 0:
                    blockers.append(f"{account['venue_id']}: no budget authorised")
        if blockers:
            return JSONResponse(status_code=409, content={
                "error": "live mode refused: nothing is ready to trade live",
                "blockers": blockers,
                "note": ("Paper mode needs none of this and is where the strategy "
                         "has to prove itself first. Staying in paper is not a "
                         "downgrade."),
                "plan": plan,
            })

    state.mode = mode
    return JSONResponse({"mode": mode,
                         "note": ("no order can be sent in paper mode"
                                  if mode == "paper" else
                                  "live mode: capital can be deployed where the "
                                  "venue is funded and authorised")})


@app.post("/api/console/budget")
async def api_set_budget(request: Request) -> JSONResponse:
    """
    Record how much of an account the agent may use.

    This moves no money. It is a permission, not a transfer - the money is
    already at the venue. Setting it above the venue balance is refused.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json()
    venue = str(body.get("venue", "")).lower()
    if venue not in FUNDING_ROUTES:
        return JSONResponse(status_code=400, content={
            "error": f"{venue} has no funding route; only "
                     f"{sorted(FUNDING_ROUTES)} can be funded"})
    try:
        amount = float(body.get("amount", 0))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "amount must be a number"})
    if amount < 0:
        return JSONResponse(status_code=400, content={"error": "amount cannot be negative"})

    balances = await _venue_balances(_agent())
    reported = balances.get(venue) or {}
    if amount > 0 and not reported.get("available"):
        return JSONResponse(status_code=409, content={
            "error": (f"cannot authorise capital at {venue}: the venue's balance "
                      f"was not read, so there is no evidence the account is funded"),
            "reason": reported.get("reason", "venue did not answer"),
            "note": ("Deposit at the venue first, then authorize a budget here. "
                     "A budget is permission to use money that already exists - "
                     "it is not a way to put money in."),
        })
    if amount > float(reported.get("balance") or 0.0) + 1e-9:
        return JSONResponse(status_code=409, content={
            "error": (f"cannot authorise ${amount:.2f}: {venue} reports "
                      f"${float(reported.get('balance') or 0.0):.2f}"),
            "note": "Authorise no more than the account holds.",
        })

    set_authorised_budget(storage, venue, amount)
    plan = _build_plan(state, storage, balances)
    return JSONResponse({"budget": amount, "venue": venue, "plan": plan})


@app.get("/api/console/funding")
async def api_funding(total: float = 50.0) -> JSONResponse:
    """How money gets in, and what this amount will actually buy."""
    return JSONResponse({
        "routes": FUNDING_ROUTES,
        "unfundable": UNFUNDABLE_SMALL,
        "plan": plan_for_budget(total, mode="live"),
        "principle": (
            "There is no account to send money to. The agent trades the venue "
            "account you fund, so capital goes in at the venue - and the agent's "
            "budget is a limit on that account, not a transfer."
        ),
    })


# ----------------------------------------------------------------------
# the running system
# ----------------------------------------------------------------------

_agent_cache: Dict[str, Any] = {}


def _agent():
    return _agent_cache.get("agent")


# A short cache for the local model check, for the same reason the balances
# have one: the console polls, and every poll must not become a fresh HTTP
# request to LM Studio. The model list changes when the operator changes it,
# not between two refreshes.
_BRAIN_CACHE: Dict[str, Any] = {"at": 0.0, "value": {}}
_BRAIN_TTL_SECONDS = 15.0


def _brain_status(force: bool = False) -> Dict[str, Any]:
    """
    The local model the agent reasons with.

    Read through the SAME detector the dashboard uses, so the two screens cannot
    disagree about which model is loaded, which one the agent will call, and
    whether it is an R1-style model that would make a cycle take hours.

    Deliberately NOT part of the operator snapshot: the snapshot is storage
    facts that the CLI prints, and it must not start requiring LM Studio to be
    running in order to describe the account.
    """
    import time as _time

    now = _time.monotonic()
    if not force and _BRAIN_CACHE["value"] and \
            now - _BRAIN_CACHE["at"] < _BRAIN_TTL_SECONDS:
        return dict(_BRAIN_CACHE["value"])
    try:
        from ..dashboard import check_lm_studio, read_env_file
        env = read_env_file()
        value = check_lm_studio(env.get("LM_STUDIO_HOST", "http://localhost:1234"),
                                configured_model=env.get("LM_STUDIO_MODEL"))
        value["env_model"] = env.get("LM_STUDIO_MODEL", "local-model")
        value["pinned"] = bool(value["env_model"] and
                               value["env_model"] not in ("local-model", "", "auto"))
        try:
            value["timeout_seconds"] = float(
                env.get("LLM_TIMEOUT_SECONDS") or 180.0)
        except (TypeError, ValueError):
            value["timeout_seconds"] = 180.0
        value["available"] = True
    except Exception as e:
        value = {"available": False, "connected": False, "models": [],
                 "active_model": None, "error": f"{type(e).__name__}: {e}"}
    _BRAIN_CACHE["at"] = _time.monotonic()
    _BRAIN_CACHE["value"] = dict(value)
    return value


@app.get("/api/console/agent")
async def api_agent() -> JSONResponse:
    """
    The agent, in one payload: what it is, what it is doing, what the money is
    doing, and the one thing to do next.

    This is the front page's data. It is the operator snapshot - the same
    payload `ptai status` prints - plus the two things only a local console can
    read: the model on this machine, and (through the copy above) nothing else.
    A blocker list is not duplicated here: the snapshot computes it, and this
    route only adds the one blocker the snapshot cannot see, which is a local
    model that is not answering.
    """
    from ..operator_view import operator_snapshot

    storage = get_storage()
    try:
        snapshot = operator_snapshot(storage)
    finally:
        storage.close()

    blockers = list(snapshot.get("blockers") or [])
    brain = _brain_status()
    if brain.get("available") and not brain.get("connected"):
        entry = {
            "id": "brain_offline", "severity": "critical",
            "what": "The agent's local model server is not answering, so it "
                    "cannot reason about fair value.",
            "evidence": f"{brain.get('host', 'http://localhost:1234')}: "
                        f"{brain.get('error') or 'no response'}",
            "clear": "Start LM Studio and load a model with the server on "
                     "(Developer -> Start Server), then refresh this page.",
        }
        # Right behind a dead process: a live agent that cannot think is still
        # not earning.
        at = 1 if blockers and blockers[0].get("id") == "agent_not_running" else 0
        blockers.insert(at, entry)
    elif brain.get("available") and brain.get("is_r1"):
        blockers.append({
            "id": "brain_slow", "severity": "warning",
            "what": "The model the agent will call is an R1-style reasoning "
                    "model, so a cycle takes minutes per market.",
            "evidence": f"active model: {brain.get('active_model')}",
            "clear": "Pin a fast model below, or in Setup -> Brain. The agent "
                     "uses it from its next start.",
        })

    return JSONResponse({
        "generated_at": snapshot.get("generated_at"),
        "mode": snapshot.get("mode"),
        "headline": snapshot.get("headline"),
        "agent": snapshot.get("agent"),
        "capital": snapshot.get("capital"),
        "profit": snapshot.get("profit"),
        "positions": snapshot.get("positions"),
        "venues": snapshot.get("venues"),
        "strategies": snapshot.get("strategies"),
        "risk": snapshot.get("risk"),
        "last_cycle": snapshot.get("last_cycle"),
        "blockers": blockers,
        "next_action": (blockers[0].get("clear") if blockers else None),
        "brain": brain,
    })


@app.post("/api/console/brain")
async def api_pin_brain(request: Request) -> JSONResponse:
    """
    Pin the model the agent will call, from the screen.

    The operator asked not to edit .env for this, and they should not have to:
    the model the agent calls is a product decision, not a config file. Only a
    model the server actually reports may be pinned - pinning a name that is not
    loaded would make the agent fall back to "first loaded model" while the
    screen claimed otherwise, which is the exact confusion this endpoint exists
    to end.
    """
    body = await request.json() if await request.body() else {}
    from ..dashboard import write_env_file

    # The agent's wait for ONE model call. Editable here for the same reason the
    # model is: it is a product setting, not a config file. A call that takes
    # nine minutes is what made a 10-minute cycle unable to finish, and the
    # operator should be able to bound it without opening .env.
    if body.get("timeout_seconds") is not None:
        try:
            seconds = float(body["timeout_seconds"])
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={
                "error": "the model call limit has to be a number of seconds"})
        if not (10 <= seconds <= 3600):
            return JSONResponse(status_code=400, content={
                "error": "the model call limit has to be between 10 and 3600 "
                         "seconds",
                "note": "Below 10s no local model finishes a forecast; above an "
                        "hour it can hold a cycle past its interval."})
        write_env_file({"LLM_TIMEOUT_SECONDS": str(int(seconds))})
        _BRAIN_CACHE["at"] = 0.0
        return JSONResponse({
            "timeout_seconds": int(seconds),
            "note": (f"The agent will wait at most {int(seconds)}s for one "
                     f"market's forecast from its next start. A slower model then "
                     f"produces no forecast for that market - which it says in the "
                     f"log - instead of stalling the cycle."),
        })

    model = str(body.get("model") or "").strip()
    if not model:
        return JSONResponse(status_code=400,
                            content={"error": "no model was named"})
    status = _brain_status(force=True)
    if not status.get("connected"):
        return JSONResponse(status_code=409, content={
            "error": "LM Studio is not answering, so there is nothing to pin.",
            "reason": status.get("error"),
            "note": "Start the server in LM Studio, refresh, and pin again.",
        })
    loaded = list(status.get("models") or [])
    if model not in loaded:
        return JSONResponse(status_code=409, content={
            "error": f"{model} is not one of the models this server has loaded.",
            "models": loaded,
            "note": "Pin one of the models in the list, so the agent calls what "
                    "this screen says it calls.",
        })
    write_env_file({"LM_STUDIO_MODEL": model})
    _BRAIN_CACHE["at"] = 0.0
    return JSONResponse({
        "pinned": model,
        "note": ("Pinned. The agent calls exactly this model from its next "
                 "start; a cycle already in flight is using the old one."),
    })


@app.get("/api/console/status")
async def api_status() -> JSONResponse:
    """
    Where the whole loop stands, in the order it runs.

    Each step reports its own state rather than a single green tick, because
    "configured" is not "ready to trade" and the difference is exactly the step
    that is missing. The steps are read from the operator snapshot - the payload
    the CLI prints - so the console cannot describe the agent differently from
    the agent's own report.
    """
    from ..operator_view import operator_snapshot

    storage = get_storage()
    state = ConsoleState(storage)
    agent = _agent()
    out: Dict[str, Any] = {
        "mode": state.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {"running": agent is not None},
        "steps": [],
        "storage": {},
    }
    balances = await _venue_balances(agent)
    plan = _build_plan(state, storage, balances)
    try:
        snapshot = operator_snapshot(storage)
    except Exception as e:
        snapshot = {}
        out["snapshot_error"] = f"{type(e).__name__}: {e}"
    answering = sum(1 for b in balances.values() if b.get("available"))

    agt = snapshot.get("agent") or {}
    profit = snapshot.get("profit") or {}
    venues = snapshot.get("venues") or {}
    matrix = venues.get("matrix") or {}
    paper = profit.get("paper") or {}
    out["storage"] = {
        "bankroll": profit.get("bankroll_usd"),
        "total_trades": profit.get("total_trades"),
        "resolved_trades": profit.get("live_resolved_trades"),
        "win_rate": profit.get("win_rate_pct"),
        "open_positions": (snapshot.get("positions") or {}).get("live_count"),
    }
    try:
        storage.close()
    except Exception:
        pass

    out["steps"] = [
        {"step": "agent", "label": "Agent running",
         "ok": bool(agt.get("running")),
         "detail": (f"{agt.get('state', 'unknown').replace('_', ' ')} - "
                    f"{agt.get('evidence')}" if agt.get("evidence")
                    else "the agent has not been seen in this database yet")},
        {"step": "capital", "label": "Capital authorised",
         "ok": bool(plan["live_venues"]),
         "detail": (f"${plan['total_available_usd']:.2f} available across "
                    f"{len(plan['live_venues'])} venue(s)"
                    if plan["live_venues"] else
                    "no funded, authorised venue: paper only")},
        {"step": "data", "label": "Market data",
         "ok": answering > 0,
         "detail": ((f"{answering} of {len(balances)} venue(s) reported a balance"
                     if balances else "no venue was reachable from this process")
                    + ("" if answering else
                       " - asked, but none answered. A venue that does not answer "
                       "is not a venue with a zero balance."))},
        {"step": "qualification", "label": "Venue qualified",
         "ok": bool(venues.get("best_validated_venue")),
         "detail": (f"{venues.get('best_validated_venue')} passed the gate on "
                    f"real resolved trades"
                    if venues.get("best_validated_venue") else
                    (f"{int(matrix.get('cells_with_enough_evidence') or 0)} of "
                     f"{int(matrix.get('cells') or 0)} venue x strategy x market "
                     f"combination(s) have enough evidence; the gate needs real "
                     f"fills and resolved outcomes, not a score. Nothing is "
                     f"qualified on a fresh install, and that is the gate "
                     f"working."))},
        {"step": "evidence", "label": "Paper evidence",
         "ok": bool(int(paper.get("settled_trades") or 0)),
         "detail": (f"{int(paper.get('settled_trades') or 0)} settled simulated "
                    f"trade(s), ${float(paper.get('net_pnl') or 0.0):+.2f}"
                    if int(paper.get("settled_trades") or 0)
                    else "no simulated trade has settled yet")},
    ]
    out["capital"] = plan
    return JSONResponse(out)


@app.get("/api/console/results")
async def api_results(limit: int = 50) -> JSONResponse:
    """
    What the system has actually achieved.

    These are the eight figures the operator is owed: equity, realised P&L, free
    capital, reserved capital, net return, drawdown, the best validated strategy
    and venue, and the risk state. A figure that cannot be computed is reported
    as null with a reason, never as a zero that reads like a result.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    out: Dict[str, Any] = {"mode": state.mode, "figures": {}, "unavailable": {}}

    try:
        perf = storage.get_performance_summary()
    except Exception as e:
        perf = {}
        out["unavailable"]["performance"] = f"{type(e).__name__}: {e}"

    balances = await _venue_balances(_agent())
    plan = _build_plan(state, storage, balances)

    def figure(key: str, value: Any, reason: str = ""):
        if value is None:
            out["unavailable"][key] = reason or "not computed"
        else:
            out["figures"][key] = value

    figure("equity_usd", perf.get("bankroll"), "no bankroll recorded")
    figure("realised_pnl_usd", perf.get("net_pnl"),
           "no resolved trades, so no realised P&L exists yet")
    figure("free_capital_usd", plan["total_available_usd"])
    figure("reserved_capital_usd", plan["total_reserved_usd"])
    figure("in_positions_usd", plan["total_in_positions_usd"])
    figure("open_positions", storage.count_open_positions())
    figure("win_rate", perf.get("win_rate"), "needs resolved trades")
    figure("total_trades", perf.get("total_trades"))
    figure("resolved_trades", perf.get("resolved_trades"))

    # Return and drawdown need a resolved history. On a fresh account they are
    # not zero, they are undefined - and presenting 0.0% as a return is a claim
    # about performance that has not happened.
    out["unavailable"].setdefault(
        "net_return_30d_pct",
        "undefined until there is a resolved-trade history spanning 30 days")
    out["unavailable"].setdefault(
        "max_drawdown_pct",
        "undefined until there is an equity curve to draw down")
    out["unavailable"].setdefault(
        "best_validated_strategy",
        "no strategy has a validated out-of-sample sample yet")
    out["unavailable"].setdefault(
        "best_validated_venue",
        "no venue has passed the 100-trade qualification gate yet")

    out["risk_state"] = {
        "mode": state.mode,
        "live_deployable": plan["is_deployable"],
        "venues_live": plan["live_venues"],
        "warnings": plan["warnings"],
    }
    try:
        trades = storage.get_recent_trades(limit=limit)
        for trade in trades:
            for key in ("pnl", "position_size_usd", "market_price", "edge"):
                if isinstance(trade.get(key), (int, float)):
                    trade[key] = round(trade[key], 4)
        out["recent_trades"] = trades
    except Exception as e:
        out["recent_trades"] = []
        out["unavailable"]["recent_trades"] = f"{type(e).__name__}: {e}"
    return JSONResponse(out)


@app.get("/api/console/orders")
async def api_orders() -> JSONResponse:
    """Working orders and what they are holding. Live against the venue."""
    storage = get_storage()
    agent = _agent()
    local = storage.get_open_orders()
    out: Dict[str, Any] = {
        "local_open_orders": len(local),
        "local_reserved_usd": round(storage.resting_capital_usd(), 4),
        "orders": local,
        "venue_view": None,
        "note": ("A working order has bought nothing, so it is not a position. "
                 "It has committed cash, so it is not available either."),
    }
    registry = getattr(agent, "venue_registry", None)
    adapters = getattr(registry, "adapters", {}) if registry is not None else {}
    adapter = adapters.get("polymarket")
    getter = getattr(adapter, "get_open_orders", None)
    if getter is not None:
        try:
            result = getter()
            if asyncio.iscoroutine(result):
                result = await result
            out["venue_view"] = result
        except Exception as e:
            out["venue_view"] = {"available": False,
                                 "reason": f"{type(e).__name__}: {e}"}
    else:
        out["venue_view"] = {"available": False,
                             "reason": "no adapter with order reads is loaded"}
    return JSONResponse(out)


@app.get("/api/console/venue")
async def api_venue() -> JSONResponse:
    """
    Which venue the agent is using right now, and why.

    The agent holds live capital at exactly one venue, because money cannot be
    moved between venues by the agent - a withdrawal and a deposit are the
    operator's actions. Everything else is scanned and paper-traded for free.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    agent = _agent()
    balances = await _venue_balances(agent)
    plan = _build_plan(state, storage, balances)

    qualified = list(getattr(agent, "_last_qualified_venue_ids", []) or [])
    labels = {v: r["label"] for v, r in FUNDING_ROUTES.items()}
    adapters = getattr(getattr(agent, "venue_registry", None), "adapters", {}) or {}
    selector = VenueSelector(storage=storage, funding_routes=FUNDING_ROUTES)
    # With the engine running, rank everything it has registered. Without it,
    # rank the venues that have a funding route or a trade history - the set that
    # could actually hold money or already has evidence.
    registered = list(adapters) or sorted(set(labels) | set(selector.known_venues()))
    assessments = selector.assess(
        registered,
        accounts=plan.get("accounts", []),
        qualified_ids=qualified,
        labels=labels,
    )
    selection = selector.select(assessments, total_budget_usd=plan.get("total_budget_usd") or 0.0)

    return JSONResponse({
        "mode": state.mode,
        "selection": selection.to_dict(),
        "assessments": [a.to_dict() for a in assessments],
        "ranking_basis": selection.ranking_basis,
        "min_sample_for_evidence": MIN_SAMPLE_FOR_EVIDENCE,
        "one_live_venue_cap": True,
        "balances_read": sum(1 for b in balances.values() if b.get("available")),
        "venues_asked": len(balances),
        # Distinguishes "every venue stayed silent" from "nothing was asked".
        # They read the same on a dashboard and mean opposite things.
        "engine_running": agent is not None,
    })


@app.post("/api/console/run-cycle")
async def api_run_cycle(request: Request) -> JSONResponse:
    """
    Run one cycle in the CURRENT mode, and refuse if live is not ready.

    This route exists so a first-time operator cannot accidentally open a live
    position from a button they thought was a preview.
    """
    storage = get_storage()
    state = ConsoleState(storage)
    body = await request.json() if await request.body() else {}
    requested_mode = str(body.get("mode") or state.mode).lower()

    if requested_mode == "live":
        balances = await _venue_balances(_agent())
        plan = _build_plan(state, storage, balances)
        if not plan["is_deployable"]:
            return JSONResponse(status_code=409, content={
                "error": "live cycle refused: no funded, authorised venue is ready",
                "warnings": plan["warnings"],
            })

    agent = _agent()
    if agent is None:
        if requested_mode == "live":
            # A live engine must be the supervised process, not something this
            # web handler constructs. Two things would otherwise become real
            # money: the button, and whatever loads the page.
            return JSONResponse(status_code=503, content={
                "error": ("live cycles run in the engine process, not in the "
                          "console. Start it with `python -m src.ptai.cli run "
                          "--no-dry-run`; it appears here when it is running."),
                "mode": requested_mode,
            })
        # PAPER cycles are safe to construct here, and deliberately so: the
        # agent is built with dry_run=True, which propagates to every adapter,
        # so no order can be sent. No credentials are needed and no capital is
        # at risk, which is what makes paper mode the right place to start.
        try:
            from ..agent.v3_loop import TradingAgentV3

            agent = TradingAgentV3(country_code="UG", dry_run=True)
            _agent_cache["agent"] = agent
            logger.info("Paper engine constructed in-process for the console")
        except Exception as e:
            logger.error(f"Could not start a paper engine: {type(e).__name__}: {e}")
            return JSONResponse(status_code=500, content={
                "error": f"could not start a paper engine: {type(e).__name__}: {e}",
                "mode": "paper",
            })
        # A paper engine constructed on demand must not be mistaken for a
        # supervised live one.
        if getattr(agent, "dry_run", None) is not True:
            _agent_cache.pop("agent", None)
            return JSONResponse(status_code=500, content={
                "error": ("refusing to run: the engine built here was not in "
                          "dry run, so a paper cycle could have sent a real order"),
                "mode": "paper",
            })

    try:
        result = await agent.run_cycle()
    except Exception as e:
        logger.error(f"Console cycle failed: {type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    execution = result.get("execution") or []
    return JSONResponse({
        "mode": requested_mode,
        "status": result.get("status"),
        "markets_scanned": len((result.get("markets") or {})) or None,
        "executed": len([e for e in execution if e.get("position_recorded")]),
        "orders_tracked": len([e for e in execution if e.get("order_recorded")]),
        "settlement": result.get("settlement"),
        "reconciliation": result.get("reconciliation"),
        "redemption": result.get("redemption"),
        "detail": result,
    })


@app.get("/", response_class=HTMLResponse)
async def console() -> str:
    return CONSOLE_HTML


CONSOLE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PTAI Console</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121821; --panel2:#0f151d; --line:#243040;
  --text:#e6edf6; --dim:#8b9bb0; --dimmer:#5a6a7e;
  --green:#2ecc71; --green-dim:#16351f;
  --red:#ef4444; --red-dim:#3a1616;
  --amber:#f5a524; --amber-dim:#3a2c10;
  --blue:#4c8dff; --paper:#a78bfa;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--line);
  padding:14px 22px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{font-weight:700;font-size:17px;letter-spacing:.4px}
.mode{display:flex;align-items:center;gap:9px;padding:6px 14px;border-radius:999px;font-weight:700;
  font-size:13px;letter-spacing:.9px}
.mode.paper{background:rgba(167,139,250,.13);color:var(--paper);border:1px solid rgba(167,139,250,.4)}
.mode.live{background:var(--red-dim);color:#ff8080;border:1px solid rgba(239,68,68,.55)}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor}
.dot.live{animation:pulse 1.4s infinite}
@keyframes pulse{50%{opacity:.25}}
.spacer{flex:1}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;
  padding:8px 15px;font-size:13px;cursor:pointer;font-weight:500}
button:hover{border-color:var(--blue);color:#fff}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
button.primary:hover{opacity:.9}
button.danger{background:var(--red-dim);border-color:rgba(239,68,68,.5);color:#ff9b9b}
button:disabled{opacity:.45;cursor:not-allowed}
main{max-width:1240px;margin:0 auto;padding:22px}
.grid{display:grid;gap:16px}
.cols-4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.cols-2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:17px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:1.1px;color:var(--dim);
  margin-bottom:14px;font-weight:600}
.kpi .v{font-size:26px;font-weight:700;letter-spacing:-.5px}
.kpi .k{font-size:12px;color:var(--dim);margin-bottom:6px}
.kpi .sub{font-size:11.5px;color:var(--dimmer);margin-top:5px}
.pos{color:var(--green)} .neg{color:var(--red)} .warn{color:var(--amber)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.7px;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid rgba(36,48,64,.5)}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
.pill.ok{background:var(--green-dim);color:var(--green)}
.pill.no{background:var(--red-dim);color:#ff9b9b}
.pill.wait{background:var(--amber-dim);color:var(--amber)}
.pill.dim{background:#1b2430;color:var(--dim)}
.empty{color:var(--dimmer);font-size:13px;padding:12px 0}
.step{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)}
.step:last-child{border-bottom:none}
.step .mark{width:20px;height:20px;border-radius:50%;flex:0 0 20px;display:grid;place-items:center;
  font-size:11px;font-weight:700;margin-top:1px}
.step .mark.ok{background:var(--green-dim);color:var(--green)}
.step .mark.no{background:var(--amber-dim);color:var(--amber)}
.step .lbl{font-weight:600;font-size:13.5px}
.step .det{color:var(--dim);font-size:12.5px;margin-top:2px}
.modebox{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.modeopt{flex:1;min-width:150px;border:1.5px solid var(--line);border-radius:10px;padding:13px;
  cursor:pointer;background:var(--panel2)}
.modeopt.sel-paper{border-color:var(--paper);background:rgba(167,139,250,.09)}
.modeopt.sel-live{border-color:var(--red);background:rgba(239,68,68,.09)}
.modeopt .t{font-weight:700;font-size:14px;margin-bottom:4px}
.modeopt .d{font-size:12px;color:var(--dim);line-height:1.4}
input{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:7px;
  padding:9px 11px;font-size:13px;width:100%}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:5px}
.note{font-size:12.5px;color:var(--dim);line-height:1.55}
.warnbox{background:var(--amber-dim);border:1px solid rgba(245,165,36,.35);border-radius:9px;
  padding:11px 13px;font-size:12.5px;color:#f7c46c;margin-bottom:12px}
.errbox{background:var(--red-dim);border:1px solid rgba(239,68,68,.4);border-radius:9px;
  padding:11px 13px;font-size:12.5px;color:#ff9b9b;margin-bottom:12px}
ol{margin:9px 0 0 18px} ol li{margin-bottom:7px;font-size:13px;line-height:1.5}
details{margin-top:10px} summary{cursor:pointer;color:var(--blue);font-size:13px}
.bar{height:7px;border-radius:4px;background:var(--panel2);overflow:hidden;margin-top:9px;display:flex}
.bar i{display:block;height:100%}
.tabs{display:flex;gap:6px;margin-top:11px;flex-wrap:wrap}
.tab{padding:7px 15px;border-radius:8px;font-size:13px;cursor:pointer;color:var(--dim);
  border:1px solid transparent}
.tab.on{background:var(--panel);border-color:var(--line);color:var(--text);font-weight:600}
.hero-pill{display:inline-block;font-size:12px;font-weight:700;letter-spacing:1.2px;
  padding:4px 11px;border-radius:999px;background:var(--panel2);color:var(--dim);
  border:1px solid var(--line);margin-bottom:9px}
.hero-pill.ok{background:var(--green-dim);color:var(--green);border-color:rgba(46,204,113,.4)}
.hero-pill.wait{background:var(--amber-dim);color:var(--amber);border-color:rgba(245,165,36,.35)}
.hero-pill.no{background:var(--red-dim);color:#ff9b9b;border-color:rgba(239,68,68,.4)}
.headline{font-size:19px;font-weight:650;line-height:1.35;letter-spacing:-.2px}
.blocker{padding:11px 0;border-bottom:1px solid rgba(36,48,64,.5)}
.blocker:last-child{border-bottom:none}
.blocker .clear{color:var(--blue);font-size:12.5px;line-height:1.5}
code{background:var(--panel2);padding:1.5px 6px;border-radius:5px;font-size:12.5px}
.foot{color:var(--dimmer);font-size:11.5px;text-align:center;padding:26px 0 12px}
.section{margin-top:26px}
.section-head{padding:16px 0 2px;border-top:1px solid var(--line)}
.section-head h1{font-size:15px;font-weight:700;letter-spacing:.2px}
.section-head p{color:var(--dim);font-size:12.5px;margin-top:3px}
section[id]{scroll-margin-top:132px}
</style>
</head>
<body>
<header>
  <div class="brand">PTAI <span style="color:var(--dim);font-weight:500">&middot; one agent, trading your money</span></div>
  <div id="modeBadge" class="mode paper"><span class="dot"></span><span id="modeText">PAPER</span></div>
  <div id="agentPill" class="pill dim">checking&hellip;</div>
  <div class="mono" style="font-size:12.5px;color:var(--dim)" id="hdrCapital"></div>
  <div class="spacer"></div>
  <button onclick="loadAll()">Refresh</button>
  <button id="runBtn" onclick="runCycle()">Run one cycle</button>
  <!--
    One page, jump links. The sections are all on this page and all visible;
    these only scroll to them, so nothing can be hidden from the operator by a
    navigation state they forgot they set. The active one is highlighted as the
    page scrolls.
  -->
  <nav class="tabs" style="flex-basis:100%">
    <div class="tab on" data-tab="agent" onclick="goTo('agent')">Agent</div>
    <div class="tab" data-tab="money" onclick="goTo('money')">Money</div>
    <div class="tab" data-tab="venue" onclick="goTo('venue')">Venue</div>
    <div class="tab" data-tab="orders" onclick="goTo('orders')">Orders</div>
    <div class="tab" data-tab="activity" onclick="goTo('activity')">Activity</div>
    <div class="tab" data-tab="setup" onclick="goTo('setup')">Setup</div>
  </nav>
</header>

<main>
  <!-- AGENT: the state of the one agent, in the order the operator asks -->
  <section id="tab-agent">
    <div class="section-head">
      <h1>Agent</h1>
      <p>Is it running, what is it doing right now, and what is the money doing.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:1;min-width:280px">
          <div id="agentState" class="hero-pill">reading the agent&hellip;</div>
          <div id="agentHeadline" class="headline">&nbsp;</div>
          <div id="agentDoing" class="note" style="margin-top:9px"></div>
        </div>
        <div id="agentVitals" class="note mono"
             style="font-size:12px;min-width:210px;text-align:right"></div>
      </div>
    </div>
    <div class="grid cols-4" id="agentKpis" style="margin-top:16px"></div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>What stands in the way</h2>
        <div id="blockers"></div>
      </div>
      <div class="card">
        <h2>Last cycle</h2>
        <div id="lastCycle"></div>
      </div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>The loop, step by step</h2>
        <div id="steps"></div>
      </div>
      <div class="card">
        <h2>Brain &mdash; the model it reasons with</h2>
        <div id="brainBox"></div>
      </div>
    </div>
  </section>

  <!-- MONEY -->
  <section id="tab-money" class="section">
    <div class="section-head">
      <h1>Money</h1>
      <p>Where it is, what it will buy, and how it gets in. A budget is
        permission to use money already in a venue account.</p>
    </div>
    <div class="grid cols-2" style="margin-top:14px">
      <div class="card">
        <h2>Where the money is</h2>
        <div id="accounts"></div>
      </div>
      <div class="card">
        <h2>Setting a budget</h2>
        <div class="warnbox">
          A budget moves no money. It is permission to use money that is already
          in the account - there is nothing to send to the agent, and the agent
          has no account of its own.
        </div>
        <label>Venue</label>
        <select id="budgetVenue" onchange="loadFunding()"
          style="background:var(--panel2);border:1px solid var(--line);color:var(--text);
                 border-radius:7px;padding:9px 11px;width:100%;margin-bottom:12px">
          <option value="polymarket">Polymarket</option>
          <option value="kalshi">Kalshi</option>
        </select>
        <label>Amount the agent may use (USD)</label>
        <input id="budgetAmount" type="number" min="0" step="1" placeholder="50">
        <div style="margin-top:12px;display:flex;gap:9px">
          <button class="primary" onclick="saveBudget()">Authorise budget</button>
          <button onclick="clearBudget()">Set to 0</button>
        </div>
        <div id="budgetMsg" style="margin-top:13px"></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>How capital gets in</h2>
      <div id="funding"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>What a limited budget will not buy</h2>
      <div id="unfundable"></div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Mode</h2>
        <div class="modebox">
          <div class="modeopt" id="modePaper" onclick="setMode('paper')">
            <div class="t">Paper</div>
            <div class="d">Simulates the whole system against the real orderbook.
              No order is sent, no money moves. Needs no capital and no
              credentials.</div>
          </div>
          <div class="modeopt" id="modeLive" onclick="setMode('live')">
            <div class="t">Live</div>
            <div class="d">Places real orders with real money on venues that are
              funded and authorised. Refused unless a venue is actually ready.</div>
          </div>
        </div>
        <div id="modeMsg"></div>
        <div class="note" style="margin-top:12px">
          The switch is refused, not merely warned about, when nothing is ready:
          an armed system that cannot fire reads as progress when it is not.
        </div>
      </div>
      <div class="card">
        <h2>Not yet measurable</h2>
        <div id="unavailable" class="note"></div>
      </div>
    </div>
  </section>

  <!-- VENUE -->
  <section id="tab-venue" class="section">
    <div class="section-head">
      <h1>Venue</h1>
      <p>One venue holds live capital at a time, named with the reason. Every
        other venue is still scanned and paper-traded.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Which venue the agent is using</h2>
      <div id="venueNow"></div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Why this one</h2>
        <div id="venueWhy" class="note"></div>
      </div>
      <div class="card">
        <h2>Ranking</h2>
        <div id="venueRank"></div>
        <div class="note" style="margin-top:10px">
          Ranked on realised net P&amp;L per resolved trade, with the sample size
          shown next to it. Below the evidence floor a venue has no score at all -
          an unmeasured venue is not a zero, and it is not a winner either.
        </div>
      </div>
    </div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>Moving to another venue</h2>
        <div id="venueSwitch"></div>
      </div>
      <div class="card">
        <h2>What runs without asking</h2>
        <div id="venueAutonomy"></div>
      </div>
    </div>
  </section>

  <!-- ORDERS -->
  <section id="tab-orders" class="section">
    <div class="section-head">
      <h1>Orders</h1>
      <p>What is working, what it is holding, and whether the venue agrees.</p>
    </div>
    <div class="grid cols-4" id="orderKpis" style="margin-top:14px"></div>
    <div class="card" style="margin-top:16px">
      <h2>Working orders</h2>
      <div id="orders"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>What the venue says</h2>
      <div id="venueView" class="note"></div>
    </div>
  </section>

  <!-- ACTIVITY -->
  <section id="tab-activity" class="section">
    <div class="section-head">
      <h1>Activity</h1>
      <p>Every trade the agent has taken, and the full result of the last cycle.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Recent trades</h2>
      <div id="trades"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Last cycle result</h2>
      <div id="cycleOut" class="note">No cycle run from this page yet.</div>
    </div>
  </section>

  <!-- SETUP -->
  <section id="tab-setup" class="section">
    <div class="section-head">
      <h1>Setup</h1>
      <p>How it runs, the model it thinks with, and where the diagnostics live.</p>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>How it runs</h2>
      <div class="note">
        <p style="margin-bottom:9px">One runner starts everything:
          <code>run_ptai.bat</code>. It opens the agent window (the loop that
          trades), this console, and nothing else. The agent cycles every
          <span id="setupInterval">10</span> minutes on its own; closing its
          window stops the trading, and closing this page changes nothing.</p>
        <p style="margin-bottom:9px">There is no approval step and no button you
          must press between cycles. The buttons here exist to look and to test,
          not to keep it alive: <b>Run one cycle</b> runs a single paper cycle in
          this process, which is why it refuses to run live.</p>
        <p>Everything the agent decides is written to the database as it goes, so
          this screen can be closed and reopened without losing the story.</p>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Brain &mdash; pin the model the agent calls</h2>
      <div id="brainSetup"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Advanced tools (diagnostics)</h2>
      <div class="note">
        <p style="margin-bottom:9px">This console is deliberately one screen
          about one agent. The older diagnostic dashboard &mdash; raw logs, the
          V2/V3 internals, the scan history, backtests, wallet linking, the API
          config &mdash; is still on disk and is not part of the daily loop. It
          is a second process, started on purpose:</p>
        <p class="mono" style="margin-bottom:9px">&#62; set PYTHONPATH=src<br>
           &#62; set PTAI_DASHBOARD_PORT=8020<br>
           &#62; python -m ptai.dashboard</p>
        <p>Use it when something needs diagnosing. Nothing on the Agent tab
          depends on it.</p>
      </div>
    </div>
  </section>

  <div class="foot">
    Local only. No cloud. The agent trades the accounts you fund and never holds
    your seed phrase.
  </div>
</main>

<script>
let STATE = {mode:'paper'};
const $ = id => document.getElementById(id);
const money = v => (v===null||v===undefined) ? '&mdash;' : '$' + Number(v).toFixed(2);
const pct = v => (v===null||v===undefined) ? '&mdash;' : Number(v).toFixed(1) + '%';

const esc = v => String(v===null||v===undefined?'':v)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// The sections, in the order they are stacked on the page. Every one is loaded
// when the page opens and refreshed on the timer: "always visible" and "only
// loaded when you click" are the same bug in different clothes.
const SECTIONS = ['agent','money','venue','orders','activity','setup'];

function goTo(name){
  const el = $('tab-'+name);
  if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on', t.dataset.tab===name));
}

// Which section is on screen, so the link the operator is looking at is the one
// highlighted. Read from scroll position rather than a state variable, because
// the page can be scrolled without ever pressing a link.
function spyScroll(){
  const line = 150;
  let current = SECTIONS[0];
  SECTIONS.forEach(n=>{
    const el = $('tab-'+n);
    if(el && el.getBoundingClientRect().top <= line) current = n;
  });
  document.querySelectorAll('.tab').forEach(t=>
    t.classList.toggle('on', t.dataset.tab===current));
}
window.addEventListener('scroll', spyScroll, {passive:true});

async function api(path, opts){
  const r = await fetch(path, Object.assign({headers:{'Content-Type':'application/json'}}, opts||{}));
  let body = null;
  try { body = await r.json(); } catch(e){ body = {}; }
  return {ok:r.ok, status:r.status, body};
}

function setBadge(mode){
  STATE.mode = mode;
  const b = $('modeBadge');
  b.className = 'mode ' + mode;
  $('modeText').textContent = mode.toUpperCase();
  $('modeBadge').querySelector('.dot').className = 'dot' + (mode==='live'?' live':'');
  $('modePaper').className = 'modeopt' + (mode==='paper'?' sel-paper':'');
  $('modeLive').className = 'modeopt' + (mode==='live'?' sel-live':'');
}

// ---- overview ----
async function loadStatus(){
  const {body} = await api('/api/console/status');
  setBadge(body.mode || 'paper');

  const cap = body.capital || {};
  const st = body.storage || {};
  $('hdrCapital').textContent = money(cap.total_available_usd) + ' available \u00b7 '
    + money(cap.total_reserved_usd) + ' reserved';

  // The KPI row lives on the Agent tab, fed by /api/console/agent - which reads
  // the same snapshot these steps do. Rendering it twice is how two screens end
  // up disagreeing about the same number.
  $('steps').innerHTML = (body.steps||[]).map(s=>`
    <div class="step">
      <div class="mark ${s.ok?'ok':'no'}">${s.ok?'\u2713':'\u2022'}</div>
      <div><div class="lbl">${s.label}</div><div class="det">${s.detail}</div></div>
    </div>`).join('');

  const un = body.unavailable || {};
  // Capital has its own availability; the rest is what "not measurable" means.
  const keys = Object.keys(un).filter(k=>k!=='performance');
  $('unavailable').innerHTML = keys.length
    ? '<ul style="margin-left:16px">' + keys.map(k=>
        `<li class="mono" style="margin-bottom:5px">${k}: <span style="color:var(--dim)">${un[k]}</span></li>`
      ).join('') + '</ul>'
    : '<span style="color:var(--green)">Everything on this page is measured from recorded data.</span>';
}

async function setMode(mode){
  $('modeMsg').innerHTML = '';
  const {ok, status, body} = await api('/api/console/mode', {method:'POST', body:JSON.stringify({mode})});
  if(!ok){
    const bs = (body.blockers||[]).map(b=>`<li>${b}</li>`).join('');
    $('modeMsg').innerHTML = `<div class="errbox"><b>${body.error||'refused'}</b>`
      + (bs?`<ul style="margin-left:16px;margin-top:7px">${bs}</ul>`:'')
      + `<div style="margin-top:8px">${body.note||''}</div></div>`;
    return;
  }
  setBadge(body.mode);
  $('modeMsg').innerHTML = `<div class="note">${body.note||''}</div>`;
  loadStatus(); loadAgent();
}

// ---- capital ----
async function loadVenue(){
  const {body} = await api('/api/console/venue');
  const sel = body.selection || {};
  const live = sel.live_venue;
  const candidate = sel.candidate;
  const byId = {};
  (body.assessments||[]).forEach(a=>{ byId[a.venue_id]=a; });

  // ---- which venue, stated plainly ----
  const liveLabel = live ? ((byId[live]||{}).label || live) : null;
  const candLabel = candidate ? ((byId[candidate]||{}).label || candidate) : null;
  let headline;
  if(liveLabel){
    headline = `<div class="pill ok" style="font-size:13px">TRADING LIVE ON
      ${esc(liveLabel).toUpperCase()}</div>`;
  } else {
    headline = `<div class="pill wait" style="font-size:13px">NO VENUE IS LIVE YET
      &mdash; PAPER ONLY</div>`;
  }
  $('venueNow').innerHTML = headline + `
    <div class="note" style="margin-top:10px">${esc(sel.verdict||'')}</div>
    <table style="margin-top:12px">
      <tr><td style="color:var(--dim);width:190px">Venue holding live capital</td>
          <td class="mono">${liveLabel?esc(liveLabel):'<span class="warn">none</span>'}</td></tr>
      <tr><td style="color:var(--dim)">Next venue to fund</td>
          <td class="mono">${candLabel?esc(candLabel):'&mdash;'}</td></tr>
      <tr><td style="color:var(--dim)">How many may hold capital</td>
          <td class="mono">1 &mdash; the agent cannot move money between venues</td></tr>
      <tr><td style="color:var(--dim)">Balances actually read</td>
          <td class="mono">${ body.engine_running
              ? `${body.balances_read||0} of ${body.venues_asked||0} venue(s) answered`
              : 'no engine is running in this process yet, so none was asked' }</td></tr>
    </table>
    <div class="note" style="margin-top:10px">
      Every other venue is still scanned and paper-traded. It is only the money
      that sits in one place.
    </div>`;

  // ---- why ----
  $('venueWhy').innerHTML = (sel.reasons||[]).length
    ? '<ul style="margin:0;padding-left:18px">' + sel.reasons.map(r=>
        `<li style="margin-bottom:7px">${esc(r)}</li>`).join('') + '</ul>'
    : '<span class="warn">No reason was recorded for the current selection.</span>';

  // ---- ranking, with the sample size next to the score ----
  const rows = (body.assessments||[]);
  $('venueRank').innerHTML = `<table>
    <tr><th>Venue</th><th>Role</th><th>Resolved</th><th>Net P&amp;L / trade</th>
        <th>Total net P&amp;L</th></tr>` +
    rows.map(a=>{
      const noEv = !a.has_evidence;
      const pnl = noEv ? '<span style="color:var(--dim)">no score &mdash; too few trades</span>'
                       : `<span class="${a.pnl_per_trade>=0?'pos':'neg'} mono">${money(a.pnl_per_trade)}</span>`;
      const tot = noEv ? '&mdash;'
                       : `<span class="mono ${a.net_pnl_usd>=0?'pos':'neg'}">${money(a.net_pnl_usd)}</span>`;
      const roleTxt = a.role==='live' ? '<span class="pill ok">live</span>'
                    : a.role==='paper' ? '<span class="pill dim">paper</span>'
                    : '<span class="pill wait">unavailable</span>';
      return `<tr>
        <td><b>${esc(a.label)}</b>${a.qualified?' <span class="pill ok">qualified</span>':''}</td>
        <td>${roleTxt}</td>
        <td class="mono">${a.resolved_trades}${noEv?` <span style="color:var(--dim)">/ ${body.min_sample_for_evidence} to score</span>`:''}</td>
        <td>${pnl}</td><td>${tot}</td>
      </tr>`;
    }).join('') + '</table>';

  // ---- switching ----
  const sp = sel.switch_plan || {};
  const stepList = (sp.steps||[]).map(x=>`<li style="margin-bottom:5px">${esc(x)}</li>`).join('');
  $('venueSwitch').innerHTML = sp.reason
    ? `<div class="note">${esc(sp.reason)}</div>` +
      (stepList ? `<ol style="margin:10px 0 0;padding-left:18px">${stepList}</ol>` : '')
    : '<span class="warn">No switch plan was recorded.</span>';

  // ---- autonomy ----
  const au = sel.autonomy || {};
  const act = (au.operator_only_actions||[]).map(x=>
    `<li style="margin-bottom:6px"><b>${esc(x.action)}</b> <span style="color:var(--dim)">
     &mdash; ${esc(x.how_often)}</span><br><span class="note">${esc(x.why)}</span></li>`).join('');
  $('venueAutonomy').innerHTML = `
    <table>
      <tr><td style="color:var(--dim);width:170px">Trades without approval</td>
          <td class="mono ${au.trades_without_approval?'pos':'neg'}">${
            au.trades_without_approval?'YES':'no'}</td></tr>
      <tr><td style="color:var(--dim)">Approval per trade</td>
          <td class="mono">${au.per_trade_approval?'required':'none'}</td></tr>
      <tr><td style="color:var(--dim)">Cycle</td>
          <td class="mono">every ${au.cycle_minutes||10} minutes, indefinitely</td></tr>
    </table>
    <div style="margin-top:12px"><b style="font-size:12.5px">Done on its own</b>
      <ul style="margin:7px 0 0;padding-left:18px">${
        (au.agent_does_autonomously||[]).map(x=>
          `<li style="margin-bottom:4px">${esc(x)}</li>`).join('')}</ul></div>
    <div style="margin-top:12px"><b style="font-size:12.5px">Never needs</b>
      <ul style="margin:7px 0 0;padding-left:18px">${
        (au.what_it_never_needs||[]).map(x=>
          `<li style="margin-bottom:4px">${esc(x)}</li>`).join('')}</ul></div>
    <div style="margin-top:12px"><b style="font-size:12.5px">Your actions only</b>
      <ol style="margin:7px 0 0;padding-left:18px">${act}</ol></div>
    ${au.honest_limit?`<div class="errbox" style="margin-top:12px;margin-bottom:0">${
      esc(au.honest_limit)}</div>`:''}`;
}

async function loadCapital(){
  const {body} = await api('/api/console/capital');
  const accts = body.accounts || [];
  if(!accts.length){
    $('accounts').innerHTML = `<div class="empty">No venue has a budget yet.
      Set one below, and run in paper mode in the meantime - it needs no capital.</div>`;
    return;
  }
  $('accounts').innerHTML = accts.map(a=>{
    const deployed = a.deposited_usd>0 ? Math.min(100, a.deployment_pct) : 0;
    const posPart = a.deposited_usd>0 ? (a.in_positions_usd/a.deposited_usd*100) : 0;
    const resPart = a.deposited_usd>0 ? (a.reserved_usd/a.deposited_usd*100) : 0;
    return `<div style="padding:13px 0;border-bottom:1px solid rgba(36,48,64,.5)">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <b style="font-size:14.5px">${a.venue_label}</b>
        <span class="pill ${a.balance_is_real?'ok':'wait'}">${
          a.balance_is_real ? 'balance read' : 'balance unread'}</span>
        <span class="pill ${a.can_deploy_live?'ok':'dim'}">${
          a.can_deploy_live ? 'live enabled' : 'paper only'}</span>
        <span class="spacer" style="flex:1"></span>
        <span class="mono" style="color:var(--dim);font-size:12.5px">
          authorised ${money(a.budget_usd)}</span>
      </div>
      <div class="bar" title="committed vs deposited">
        <i style="width:${posPart}%;background:var(--blue)"></i>
        <i style="width:${resPart}%;background:var(--amber)"></i>
      </div>
      <div class="grid cols-4" style="margin-top:11px;gap:10px">
        <div><div class="k" style="font-size:11px;color:var(--dim)">Deposited</div>
             <div class="mono">${money(a.deposited_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">In positions</div>
             <div class="mono">${money(a.in_positions_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">Reserved</div>
             <div class="mono warn">${money(a.reserved_usd)}</div></div>
        <div><div style="font-size:11px;color:var(--dim)">Available</div>
             <div class="mono pos">${money(a.available_usd)}</div></div>
      </div>
      ${(a.notes||[]).length?`<div class="note" style="margin-top:8px">${a.notes.join(' &middot; ')}</div>`:''}
      ${(a.warnings||[]).length?`<div class="errbox" style="margin-top:9px;margin-bottom:0">${
        a.warnings.join('<br>')}</div>`:''}
      <div class="note" style="margin-top:6px">${deployed.toFixed(0)}% of deposited capital is committed.</div>
    </div>`;
  }).join('');
}

async function loadFunding(){
  const venue = $('budgetVenue').value;
  const total = parseFloat($('budgetAmount').value) || 0;
  const {body} = await api('/api/console/funding?total=' + encodeURIComponent(total));
  const r = body.routes[venue];
  if(!r){ $('funding').innerHTML = '<div class="empty">No route recorded.</div>'; return; }
  $('funding').innerHTML = `
    <div class="note" style="margin-bottom:12px">${body.principle}</div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">
      <div><div style="font-size:11px;color:var(--dim)">Currency</div>
        <div style="font-size:13px">${r.currency}</div></div>
      <div><div style="font-size:11px;color:var(--dim)">Minimum deposit</div>
        <div class="mono">${money(r.minimum_deposit_usd)}</div></div>
      <div><div style="font-size:11px;color:var(--dim)">Smallest practical</div>
        <div class="mono">${money(r.smallest_practical_usd)}</div></div>
    </div>
    <ol>${r.deposit_steps.map(s=>`<li>${s}</li>`).join('')}</ol>
    <div class="note" style="margin-top:12px"><b>Withdrawing.</b> ${r.withdraw_note}</div>
    <div class="note" style="margin-top:7px"><b>Fees.</b> ${r.fees_note}</div>
    <div class="note" style="margin-top:7px"><b>The key.</b> ${r.agent_key_explanation}</div>
    ${(body.plan.warnings||[]).length
      ? `<div class="warnbox" style="margin-top:13px;margin-bottom:0">${
          body.plan.warnings.join('<br>')}</div>`
      : `<div class="note" style="margin-top:13px;color:var(--green)">${
          (body.plan.notes||[]).join(' ')}</div>`}`;

  $('unfundable').innerHTML = Object.entries(body.unfundable||{}).map(([k,v])=>
    `<div style="padding:8px 0;border-bottom:1px solid rgba(36,48,64,.4)">
       <span class="mono" style="font-size:12.5px">${k}</span>
       <span style="color:var(--dim);font-size:12.5px"> &mdash; ${v}</span></div>`).join('')
    + `<div class="note" style="margin-top:11px">These are still scanned for
       opportunities and still run in paper mode. They cost nothing there; they
       just cannot hold your $50.</div>`;
}

async function saveBudget(){
  const venue = $('budgetVenue').value;
  const amount = parseFloat($('budgetAmount').value);
  if(isNaN(amount) || amount < 0){
    $('budgetMsg').innerHTML = '<div class="errbox">Enter an amount of 0 or more.</div>';
    return;
  }
  const {ok, body} = await api('/api/console/budget', {method:'POST',
    body:JSON.stringify({venue, amount})});
  if(!ok){
    $('budgetMsg').innerHTML = `<div class="errbox"><b>${body.error}</b>`
      + (body.reason?`<div style="margin-top:7px">${body.reason}</div>`:'')
      + (body.note?`<div style="margin-top:7px">${body.note}</div>`:'') + `</div>`;
    return;
  }
  $('budgetMsg').innerHTML = `<div class="note pos">Authorised ${money(amount)} at ${venue}.
    Nothing was moved: the money is already in that account.</div>`;
  loadCapital(); loadStatus();
}

function clearBudget(){ $('budgetAmount').value = 0; saveBudget(); }

// ---- orders ----
async function loadOrders(){
  const {body} = await api('/api/console/orders');
  $('orderKpis').innerHTML = [
    ['Working orders', body.local_open_orders, 'not yet filled or cancelled'],
    ['Cash reserved', money(body.local_reserved_usd), 'locked behind those orders'],
  ].map(([k,v,s])=>`<div class="card kpi"><div class="k">${k}</div>
     <div class="v">${v}</div><div class="sub">${s}</div></div>`).join('');

  const rows = body.orders || [];
  $('orders').innerHTML = rows.length ? `<table><thead><tr>
      <th>Order</th><th>Market</th><th>Side</th><th>Limit</th>
      <th>Requested</th><th>Matched</th><th>Status</th></tr></thead><tbody>`
    + rows.map(o=>`<tr>
        <td class="mono" style="font-size:12px">${o.order_id||'&mdash;'}</td>
        <td class="mono" style="font-size:12px">${o.market_id||''}</td>
        <td>${o.side||''}</td>
        <td class="mono">${o.limit_price!=null?Number(o.limit_price).toFixed(3):'&mdash;'}</td>
        <td class="mono">${money(o.requested_usd)}</td>
        <td class="mono">${money(o.matched_usd)}</td>
        <td><span class="pill wait">${o.status||''}</span></td></tr>`).join('')
    + `</tbody></table>`
    : `<div class="empty">No working orders. ${body.note}</div>`;

  const view = body.venue_view;
  $('venueView').innerHTML = (view && view.available)
    ? `<div class="mono" style="font-size:12.5px">${JSON.stringify(view, null, 2)}</div>`
    : `<span class="warn">The venue's own order list was not read:</span>
       ${(view&&view.reason)||'no adapter'} &mdash; local orders stay reserved
       until the venue answers.`;
}

// ---- activity ----
async function loadResults(){
  const {body} = await api('/api/console/results');
  const trades = body.recent_trades || [];
  $('trades').innerHTML = trades.length ? `<table><thead><tr>
      <th>Market</th><th>Side</th><th>Size</th><th>Entry</th>
      <th>P&amp;L</th><th>Status</th></tr></thead><tbody>`
    + trades.map(t=>{
        const pnl = t.pnl;
        const cls = (pnl==null) ? '' : (pnl>=0?'pos':'neg');
        return `<tr>
          <td class="mono" style="font-size:12px">${(t.market_id||'').slice(0,22)}</td>
          <td>${t.side||''}</td>
          <td class="mono">${money(t.position_size_usd)}</td>
          <td class="mono">${t.market_price!=null?Number(t.market_price).toFixed(3):'&mdash;'}</td>
          <td class="mono ${cls}">${pnl==null?'&mdash;':(pnl>=0?'+':'')+Number(pnl).toFixed(2)}</td>
          <td><span class="pill ${t.status==='paper'?'dim':'ok'}">${t.status||''}</span></td>
        </tr>`;}).join('') + `</tbody></table>`
    : `<div class="empty">No trades recorded yet. Paper mode fills are recorded the
       same way live ones are, so this table is where the strategy earns the right
       to real money.</div>`;
}

async function runCycle(){
  const btn = $('runBtn');
  btn.disabled = true; btn.textContent = 'Running\u2026';
  const {ok, body} = await api('/api/console/run-cycle', {method:'POST',
    body:JSON.stringify({mode:STATE.mode})});
  btn.disabled = false; btn.textContent = 'Run one cycle';
  if(!ok){
    $('cycleOut').innerHTML = `<div class="errbox"><b>${body.error||'cycle refused'}</b>`
      + ((body.warnings||[]).length?`<ul style="margin-left:16px;margin-top:7px">${
          body.warnings.map(w=>`<li>${w}</li>`).join('')}</ul>`:'') + '</div>';
    return;
  }
  $('cycleOut').innerHTML = `
    <div><b>${body.status||''}</b> &mdash; ${body.executed||0} position(s) recorded,
      ${body.orders_tracked||0} order(s) tracked for reconciliation.</div>
    <div class="note" style="margin-top:7px">
      settlement: ${JSON.stringify(body.settlement||null)} &middot;
      reconciliation: ${JSON.stringify(body.reconciliation||null)} &middot;
      redemption: ${JSON.stringify(body.redemption||null)}
    </div>`;
  // The cycle just wrote its own phase, heartbeat and scan row: show them, so
  // the front page cannot lag behind a cycle the operator just ran by hand.
  loadStatus(); loadAgent();
}

// ---- the agent: the front page, and the only question that matters ----

const SEVERITY_PILL = {critical:'no', loss:'no', next_action:'wait',
                       next_step:'wait', gate:'wait', warning:'wait',
                       waiting:'dim', ok:'ok'};
const STATE_PILL = {running:'ok', working:'wait', blocked:'wait',
                    not_running:'no', unknown:'dim'};
const STATE_WORD = {running:'RUNNING', working:'WORKING', blocked:'BLOCKED',
                    not_running:'STOPPED', unknown:'UNKNOWN'};

function ageText(seconds){
  if(seconds===null || seconds===undefined) return 'never';
  seconds = Math.max(0, Number(seconds));
  if(seconds < 90) return Math.round(seconds) + 's';
  if(seconds < 5400) return Math.round(seconds/60) + ' min';
  return (seconds/3600).toFixed(1) + 'h';
}

async function loadAgent(){
  const {body} = await api('/api/console/agent');
  if(!body || body.headline===undefined){
    $('agentHeadline').textContent = 'The agent state could not be read.';
    return;
  }
  const agt = body.agent || {};
  const cap = body.capital || {};
  const prof = body.profit || {};
  const paper = prof.paper || {};
  const risk = body.risk || {};
  const ven = body.venues || {};
  const st = body.strategies || {};

  STATE.mode = body.mode || STATE.mode;
  setBadge(STATE.mode);

  // header pill
  const pill = $('agentPill');
  pill.className = 'pill ' + (STATE_PILL[agt.state] || 'dim');
  pill.textContent = STATE_WORD[agt.state] || String(agt.state||'unknown').toUpperCase();

  // hero
  const heroPill = $('agentState');
  heroPill.className = 'hero-pill ' + (STATE_PILL[agt.state] || 'dim');
  heroPill.textContent = (STATE_WORD[agt.state] || 'UNKNOWN')
    + (agt.state==='running' && agt.last_scan_ago_seconds!=null
       ? ' \u00b7 LAST CYCLE ' + ageText(agt.last_scan_ago_seconds) + ' AGO'
       : agt.evidence ? ' \u00b7 ' + ageText(agt.phase_seconds || agt.heartbeat_ago_seconds || agt.last_scan_ago_seconds) + ' AGO' : '');
  $('agentHeadline').textContent = body.headline || '';
  $('agentDoing').innerHTML = agt.running
    ? '<b>Right now:</b> ' + esc(agt.doing || 'between cycles')
      + (agt.next_cycle_at ? ' \u00b7 next cycle ' + esc(String(agt.next_cycle_at).slice(11,16)) + ' UTC' : '')
    : '<b>Nothing is running.</b> ' + esc(agt.evidence || 'No agent process has left a mark in this database.')
      + ' Start <code>run_ptai.bat</code> on the machine that trades.';

  // vitals, in the operator's words
  const validated = ven.best_validated_venue
    ? esc(ven.best_validated_venue) + ' (' + esc(ven.best_validated_on||'') + ')'
    : 'none yet';
  const bestStrat = st.best_validated_strategy ? esc(st.best_validated_strategy) : 'none yet';
  $('agentVitals').innerHTML =
      'cycle every ' + esc(String(agt.interval_min||10)) + ' min<br>'
    + 'liveness window ' + Math.round((agt.window_seconds||900)/60) + ' min<br>'
    + 'heartbeat: ' + esc(agt.heartbeat_status || '\u2014') + ' ' + ageText(agt.heartbeat_ago_seconds) + '<br>'
    + 'last completed cycle: ' + ageText(agt.last_scan_ago_seconds) + '<br>'
    + 'validated venue: ' + validated + '<br>'
    + 'validated strategy: ' + bestStrat + '<br>'
    + 'risk: ' + (risk.trading_halted ? '<span class="neg">halted</span>'
        : 'kill switch ' + esc(String(risk.kill_switch_level==null?'none':risk.kill_switch_level)));

  // the money
  const pnl = cap.realised_pnl_usd;
  const pnlCls = (typeof pnl==='number') ? (pnl>=0?'pos':'neg') : '';
  const kpis = [
    ['Equity', money(cap.equity_usd), (cap.account==='paper'?'paper account':(cap.account||'')+' account')],
    ['Realised P&amp;L', money(pnl), (prof.live_resolved_trades||0) + ' resolved live trade(s), class:pnlCls'],
    ['Free capital', money(cap.free_cash_usd), 'not committed anywhere'],
    ['Reserved', money(cap.reserved_capital_usd), 'locked behind working orders'],
    ['Deployed', (cap.deployment_pct==null?'\u2014':Number(cap.deployment_pct).toFixed(1)+'%'),
      money(cap.deployed_usd) + ' working'],
    ['30-day net return', (prof.return_30d_pct==null?'\u2014':Number(prof.return_30d_pct).toFixed(2)+'%'),
      (prof.return_30d_pct==null?'undefined until a resolved history exists':'on the live account')],
    ['Max drawdown', (prof.max_drawdown_pct==null?'\u2014':Number(prof.max_drawdown_pct).toFixed(2)+'%'),
      (prof.max_drawdown_pct==null?'no equity curve to draw down yet':'worst peak-to-trough')],
    ['Paper (kept apart)', money(paper.net_pnl),
      (paper.settled_trades||0) + ' settled simulated trade(s)'],
  ];
  $('agentKpis').innerHTML = kpis.map(([k,v,s])=>{
    const cls = (s||'').indexOf('class:pnlCls')>=0 ? ' '+pnlCls : '';
    return `<div class="card kpi"><div class="k">${k}</div>
      <div class="v${cls}">${v}</div>
      <div class="sub">${(s||'').replace(' class:pnlCls','')}</div></div>`;
  }).join('');

  // what stands in the way - the reason this screen exists
  $('blockers').innerHTML = (body.blockers||[]).length
    ? body.blockers.map(b=>`
      <div class="blocker">
        <div><span class="pill ${SEVERITY_PILL[b.severity]||'dim'}">${esc(String(b.severity||'').replace('_',' '))}</span>
             <b style="margin-left:8px">${esc(b.what)}</b></div>
        <div class="note" style="margin-top:5px">${esc(b.evidence||'')}</div>
        <div class="clear" style="margin-top:6px">&#8594; ${esc(b.clear)}</div>
      </div>`).join('')
    : '<div class="empty">No blockers were computed.</div>';

  // last cycle
  const lc = body.last_cycle || {};
  if(!lc.available){
    $('lastCycle').innerHTML = `<div class="empty">${esc(lc.note||'No completed cycle is recorded.')}</div>`;
  } else {
    const dec = lc.decided || {};
    const orders = lc.orders || {};
    $('lastCycle').innerHTML = `
      <div><span class="pill ${lc.verdict==='DEPLOYED'?'ok':'dim'}">${esc(lc.verdict||'')}</span>
        <span class="note" style="margin-left:9px">${esc(String(lc.at||'').replace('T',' ').slice(0,19))} UTC
        &middot; took ${esc(String(lc.cycle_seconds||'?'))}s</span></div>
      <table style="margin-top:10px">
        <tr><td style="color:var(--dim)">Markets scanned</td>
            <td class="mono">${esc(String(lc.markets_scanned||0))} across ${esc(String(lc.venues_searched||0))} venue(s)</td></tr>
        <tr><td style="color:var(--dim)">Candidates</td>
            <td class="mono">${esc(String(lc.candidates||0))}</td></tr>
        <tr><td style="color:var(--dim)">Positions recorded</td>
            <td class="mono">${esc(String(orders.positions_recorded||0))}</td></tr>
        <tr><td style="color:var(--dim)">Blocked by capital boundary</td>
            <td class="mono">${esc(String(orders.blocked_live_capital||0))}${orders.blocked_reason?` <span style="color:var(--dim)">&mdash; ${esc(orders.blocked_reason)}</span>`:''}</td></tr>
        <tr><td style="color:var(--dim)">Best it found</td>
            <td class="mono">${dec.venue?esc(dec.venue)+' &middot; '+esc(dec.strategy||''):'&mdash;'}</td></tr>
        <tr><td style="color:var(--dim)">Edge on it</td>
            <td class="mono">${dec.edge!=null?esc(String(dec.edge)):'&mdash;'}</td></tr>
      </table>
      ${lc.why?`<div class="note" style="margin-top:10px"><b>Why:</b> ${esc(lc.why)}</div>`:''}`;
  }

  renderBrain($('brainBox'), body.brain||{}, false);
  window.__BRAIN = body.brain || {};
  window.__INTERVAL = agt.interval_min || 10;
}

function renderBrain(el, b, withPicker){
  if(!el) return;
  const active = b.active_model;
  const pinNote = b.pinned
    ? 'pinned in .env &mdash; this is the model the agent calls'
    : 'auto: whichever model LM Studio lists first';
  const head = b.connected
    ? `<div class="pill ok">CONNECTED</div>
       <span class="note" style="margin-left:9px">${esc(String(b.models.length))} model(s) loaded${b.latency_ms!=null?' &middot; '+esc(String(b.latency_ms))+' ms':''}</span>`
    : `<div class="pill no">NO ANSWER</div>
       <span class="note" style="margin-left:9px">${esc(b.host||'http://localhost:1234')}: ${esc(b.error||'not reachable')}</span>`;
  const table = `
    <table style="margin-top:11px">
      <tr><td style="color:var(--dim);width:170px">Agent will use</td>
          <td class="mono ${active?'pos':''}">${active?esc(active):'nothing - no model is loaded'}</td></tr>
      <tr><td style="color:var(--dim)">How it was chosen</td>
          <td class="note">${esc(b.model_reason || pinNote)}</td></tr>
      <tr><td style="color:var(--dim)">Speed</td>
          <td class="mono">${b.is_r1?'<span class="neg">SLOW - minutes per market (R1-style)</span>'
              :(b.connected?'<span class="pos">full speed - no long thinking phase</span>':'&mdash;')}</td></tr>
      <tr><td style="color:var(--dim)">One call waits at most</td>
          <td class="mono">${b.timeout_seconds?Number(b.timeout_seconds).toFixed(0)+'s':'&mdash;'}
            <span class="note"> - a slower model then gives no forecast for that market instead of holding the cycle</span></td></tr>
    </table>`;
  const picker = (withPicker && b.connected && b.models.length) ? `
    <div style="margin-top:13px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="modelSelect" style="max-width:340px;width:auto">${
        b.models.map(m=>`<option value="${esc(m)}"${m===active?' selected':''}>${esc(m)}</option>`).join('')}</select>
      <button class="primary" onclick="pinModel()">Pin this model</button>
      <label style="margin:0 0 0 8px;color:var(--dim);font-size:12px">Model call limit (s)</label>
      <input id="timeoutSeconds" type="number" min="10" max="3600" step="10"
             value="${b.timeout_seconds?Number(b.timeout_seconds).toFixed(0):180}"
             style="width:90px">
      <button onclick="saveTimeout()">Save</button>
    </div>
    <div id="pinMsg" class="note" style="margin-top:9px"></div>
    <div class="note" style="margin-top:9px">Pinning writes <code>LM_STUDIO_MODEL</code> to
      <code>.env</code> so the agent stops taking whichever model is listed first.
      The running agent picks it up on its <b>next start</b>.</div>`
    : (withPicker ? `<div class="note" style="margin-top:11px">Start LM Studio and load a
        model with the local server on, then refresh, to pin one from here.</div>` : '');
  el.innerHTML = head + table + picker;
}

async function pinModel(){
  const sel = $('modelSelect');
  if(!sel){ return; }
  const {ok, body} = await api('/api/console/brain', {method:'POST',
    body:JSON.stringify({model: sel.value})});
  $('pinMsg').innerHTML = ok
    ? `<span class="pos">${esc(body.note||'pinned')}</span>`
    : `<span class="neg">${esc(body.error||'could not pin')}</span> ${esc(body.note||'')}`;
  if(ok){ loadBrainSetup(); loadAgent(); }
}

async function saveTimeout(){
  const raw = $('timeoutSeconds').value;
  const {ok, body} = await api('/api/console/brain', {method:'POST',
    body:JSON.stringify({timeout_seconds: Number(raw)})});
  $('pinMsg').innerHTML = ok
    ? `<span class="pos">${esc(body.note||'saved')}</span>`
    : `<span class="neg">${esc(body.error||'could not save')}</span> ${esc(body.note||'')}`;
  if(ok) loadBrainSetup();
}

async function loadBrainSetup(){
  const {body} = await api('/api/console/agent');
  window.__BRAIN = body.brain || {};
  renderBrain($('brainSetup'), body.brain||{}, true);
  const iv = $('setupInterval');
  if(iv && body.agent) iv.textContent = String(body.agent.interval_min||10);
}

async function loadAll(){
  await Promise.all([
    loadAgent(), loadStatus(), loadBrainSetup(),
    loadVenue(), loadCapital(), loadFunding(), loadOrders(), loadResults(),
  ]);
  spyScroll();
}

// Everything, on open and on the timer. The panels are read-only views of the
// database; the only writable thing on the page is the budget box, and its
// input is never re-rendered by a refresh.
loadAll();
setInterval(loadAll, 15000);
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8101) -> None:
    """
    Serve the console - the one screen the product has.

    The port comes from the environment first, because the runner sets it in one
    place (`PTAI_DASHBOARD_PORT`, the name already in the operator's .bat and in
    the docs) and every process the runner starts has to agree on it. PTAI is
    local-only either way: this binds on the machine that runs it.
    """
    import os
    chosen = (os.environ.get("PTAI_CONSOLE_PORT")
              or os.environ.get("PTAI_DASHBOARD_PORT") or port)
    try:
        port = int(chosen)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring unusable port {chosen!r}; using {port}")
    import uvicorn
    logger.info(f"PTAI Console on http://localhost:{port}")
    logger.info("Paper mode is the default and needs no capital or credentials.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
