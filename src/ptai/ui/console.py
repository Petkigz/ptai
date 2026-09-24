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

from ..execution.capital import CapitalLedger, FUNDING_ROUTES, UNFUNDABLE_SMALL, plan_for_budget
from ..storage.db import Storage

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

    KEYS = ("mode", "budget.polymarket", "budget.kalshi")

    def __init__(self, storage: Storage):
        self.storage = storage

    def get(self, key: str, default: str = "") -> str:
        value = self.storage.get_state(f"console.{key}")
        return default if value is None else value

    def set(self, key: str, value: str) -> None:
        self.storage.set_state(f"console.{key}", str(value))

    @property
    def mode(self) -> str:
        mode = (self.get("mode", "paper") or "paper").lower()
        return mode if mode in ("paper", "live") else "paper"

    @mode.setter
    def mode(self, value: str) -> None:
        mode = str(value).lower()
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be paper or live, got {value!r}")
        self.set("mode", mode)

    def budgets(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in ("polymarket", "kalshi"):
            raw = self.get(f"budget.{key}", "0")
            try:
                value = float(raw)
            except ValueError:
                value = 0.0
            if value > 0:
                out[key] = value
        return out


def get_storage() -> Storage:
    import os
    return Storage(db_path=os.getenv("PTAI_DB", "./data/ptai.db"))


# ----------------------------------------------------------------------
# capital
# ----------------------------------------------------------------------

async def _venue_balances(agent=None) -> Dict[str, Dict[str, Any]]:
    """
    Ask each venue what the balance is. Never guesses.

    "Available: False" means the venue did not answer, and downstream that is
    treated as NOT FUNDED. The alternative - defaulting to whatever the operator
    typed into the budget box - is an agent that believes it has money because
    someone filled in a form.
    """
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

    state.set(f"budget.{venue}", f"{amount:.2f}")
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


@app.get("/api/console/status")
async def api_status() -> JSONResponse:
    """
    Where the whole loop stands, in the order it runs.

    Each step reports its own state rather than a single green tick, because
    "configured" is not "ready to trade" and the difference is exactly the step
    that is missing.
    """
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
    try:
        perf = storage.get_performance_summary()
        out["storage"] = {
            "bankroll": perf.get("bankroll"),
            "total_trades": perf.get("total_trades"),
            "resolved_trades": perf.get("resolved_trades"),
            "win_rate": perf.get("win_rate"),
            "open_positions": storage.count_open_positions(),
        }
    except Exception as e:
        out["storage"] = {"error": f"{type(e).__name__}: {e}"}

    balances = await _venue_balances(agent)
    plan = _build_plan(state, storage, balances)

    out["steps"] = [
        {"step": "capital", "label": "Capital authorised",
         "ok": bool(plan["live_venues"]),
         "detail": (f"${plan['total_available_usd']:.2f} available across "
                    f"{len(plan['live_venues'])} venue(s)"
                    if plan["live_venues"] else
                    "no funded, authorised venue: paper only")},
        {"step": "account", "label": "Account verified (auth, funds, permission)",
         "ok": False,
         "detail": ("not verified this session - the order probe needs a real "
                    "account, and it has not been run since startup")},
        {"step": "data", "label": "Market data",
         "ok": bool(balances),
         "detail": (f"{len(balances)} venue(s) answered"
                    if balances else "no venue was reachable from this process")},
        {"step": "qualification", "label": "Venue qualified",
         "ok": False,
         "detail": ("needs 100+ resolved trades per venue: win rate, Brier, "
                    "profit factor. Nothing is qualified on a fresh install, and "
                    "that is the gate working.")},
        {"step": "paper", "label": "Paper evidence",
         "ok": bool(out["storage"].get("total_trades")),
         "detail": (f"{out['storage'].get('total_trades')} simulated trade(s) "
                    f"recorded" if out["storage"].get("total_trades")
                    else "no simulated trades yet")},
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
        return JSONResponse(status_code=503, content={
            "error": ("the engine is not running in this process. Start it with "
                      "`python -m ptai.cli run` and it will report here."),
            "mode": requested_mode,
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
.tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:7px 15px;border-radius:8px;font-size:13px;cursor:pointer;color:var(--dim);
  border:1px solid transparent}
.tab.on{background:var(--panel);border-color:var(--line);color:var(--text);font-weight:600}
.hide{display:none}
code{background:var(--panel2);padding:1.5px 6px;border-radius:5px;font-size:12.5px}
.foot{color:var(--dimmer);font-size:11.5px;text-align:center;padding:26px 0 12px}
</style>
</head>
<body>
<header>
  <div class="brand">PTAI</div>
  <div id="modeBadge" class="mode paper"><span class="dot"></span><span id="modeText">PAPER</span></div>
  <div class="mono" style="font-size:12.5px;color:var(--dim)" id="hdrCapital"></div>
  <div class="spacer"></div>
  <button onclick="loadAll()">Refresh</button>
  <button id="runBtn" onclick="runCycle()">Run one cycle</button>
</header>

<main>
  <div class="tabs">
    <div class="tab on" data-tab="overview" onclick="showTab('overview')">Overview</div>
    <div class="tab" data-tab="capital" onclick="showTab('capital')">Capital &amp; Funding</div>
    <div class="tab" data-tab="orders" onclick="showTab('orders')">Orders</div>
    <div class="tab" data-tab="activity" onclick="showTab('activity')">Activity</div>
  </div>

  <!-- OVERVIEW -->
  <section id="tab-overview">
    <div class="grid cols-4" id="kpis"></div>
    <div class="grid cols-2" style="margin-top:16px">
      <div class="card">
        <h2>The loop, step by step</h2>
        <div id="steps"></div>
      </div>
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
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Not yet measurable</h2>
      <div id="unavailable" class="note"></div>
    </div>
  </section>

  <!-- CAPITAL -->
  <section id="tab-capital" class="hide">
    <div class="grid cols-2">
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
  </section>

  <!-- ORDERS -->
  <section id="tab-orders" class="hide">
    <div class="grid cols-4" id="orderKpis"></div>
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
  <section id="tab-activity" class="hide">
    <div class="card">
      <h2>Recent trades</h2>
      <div id="trades"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>Last cycle result</h2>
      <div id="cycleOut" class="note">No cycle run from this page yet.</div>
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

function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on', t.dataset.tab===name));
  ['overview','capital','orders','activity'].forEach(n=>
    $('tab-'+n).classList.toggle('hide', n!==name));
  if(name==='capital'){ loadCapital(); loadFunding(); }
  if(name==='orders') loadOrders();
  if(name==='activity') loadResults();
}

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

  const kpi = [];
  kpi.push(['Equity', money(st.bankroll), (st.bankroll==null?'no bankroll recorded':'')]);
  kpi.push(['Free capital', money(cap.total_available_usd),
            'money not committed anywhere']);
  kpi.push(['Reserved', money(cap.total_reserved_usd),
            'locked behind working orders']);
  kpi.push(['In positions', money(cap.total_in_positions_usd),
            (st.open_positions||0) + ' open']);
  $('kpis').innerHTML = kpi.map(([k,v,s])=>`
    <div class="card kpi"><div class="k">${k}</div>
      <div class="v">${v}</div><div class="sub">${s||''}</div></div>`).join('');

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
  loadStatus();
}

// ---- capital ----
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
  loadStatus();
}

async function loadAll(){ await loadStatus(); }

loadAll();
setInterval(loadStatus, 15000);
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8101) -> None:
    import uvicorn
    logger.info(f"PTAI Console on http://{host}:{port}")
    logger.info("Paper mode is the default and needs no capital or credentials.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
