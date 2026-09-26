"""
Every venue PTAI knows about, and what it can actually do with each one.

The console could only list venues it had a reason to rank: the fundable ones,
or the ones with a trade history. On a fresh install that is two - so an
operator looking at the venue panel saw "Polymarket" and "Kalshi" and had no way
to tell whether the other seventeen adapters existed, were broken, or were
merely invisible. They are none of those things: most of them are reading live
public market data every cycle and being paper-traded for free, and the operator
had no way to find that out from the product.

The distinction this module exists to make, per venue:

  * CAN IT BE READ NOW   - does it return live markets with no login?
  * CAN PTAI FILL THERE  - every venue can be paper-filled; that is the point of
                           paper mode, and it is how a venue earns its record.
  * CAN IT HOLD REAL MONEY - does the adapter have a submission path at all, and
                           what would arming it require?
  * WHAT DOES IT NEED    - credentials, a funding rail, or code that has not
                           been written yet.

Nothing here is a judgement about profitability. It is the answer to "can I run
this one too?", which is a different question and was previously unanswerable
from the UI.

Written to the state store by the agent at the start of every cycle, because the
agent is the process that HAS the registry. The console runs in a different
process (the .bat starts them separately), so it reads this record - and says
"the agent has not run yet" when there is nothing to read, rather than showing
two venues as though that were the whole list.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .adapter import STATUS_LIVE, STATUS_SCANNER, STATUS_UNIMPLEMENTED

# The state key the agent writes and the console reads. One key, one record.
VENUE_INVENTORY_KEY = "operator.venue_inventory"

# What each row's `use` field can be. These are the answers to the operator's
# question, so the set is closed and the UI switches on it.
USE_REAL_MONEY = "real_money"      # has a submission path; needs arming + funding
USE_PAPER_ONLY = "paper_only"      # reads live data, fills in paper, never real
USE_NEEDS_LOGIN = "needs_login"    # no credentials means no data either
USE_NO_CLIENT = "no_client"        # no client written; must not pretend
USE_SCANNER = "scanner"            # read-only aggregator


def _funding_note(venue_id: str) -> Dict[str, Any]:
    """How this venue would be funded, if it can be at all."""
    try:
        from ..execution.capital import FUNDING_ROUTES, UNFUNDABLE_SMALL

        route = FUNDING_ROUTES.get(venue_id)
        if route:
            return {
                "fundable": True,
                "currency": route.get("currency"),
                "minimum_deposit_usd": route.get("minimum_deposit_usd"),
                "recommended_deposit_usd": route.get("recommended_deposit_usd"),
                "available_from": route.get("available_from"),
                "reason_unfundable": None,
            }
        return {
            "fundable": False,
            "reason_unfundable": UNFUNDABLE_SMALL.get(
                venue_id, "no funding route is recorded for this venue"),
        }
    except Exception as e:  # noqa: BLE001 - the panel must still render
        logger.warning(f"Funding routes unavailable for {venue_id}: {e}")
        return {"fundable": False, "reason_unfundable": "funding route unknown"}


# What a venue is called on screen. Adapters carry no display name of their own
# - the funding routes do, because that is where the operator meets them - and
# anything without one is shown under its id rather than invented.
_LABELS = {
    "polymarket": "Polymarket",
    "kalshi": "Kalshi",
    "manifold": "Manifold",
    "predictit": "PredictIt",
    "crypto_binance": "Binance (crypto)",
    "whitebit": "WhiteBIT",
    "betfair": "Betfair Exchange",
    "simmer": "Simmer",
    "cymetica": "Cymetica",
    "afx_dex": "AFX DEX",
    "grvt": "GRVT",
    "pionex": "Pionex",
    "stock_mock": "Stocks (simulated)",
    "betdaq": "Betdaq",
    "betconnect": "BetConnect",
    "ccxt_unified": "CCXT (unified)",
    "veynor": "Veynor",
    "openpx": "OpenPX",
    "apify": "Apify scanner",
}


def _label(venue_id: str, adapter: Any) -> str:
    named = getattr(adapter, "label", None)
    if named:
        return str(named)
    try:
        from ..execution.capital import FUNDING_ROUTES

        route = FUNDING_ROUTES.get(venue_id) or {}
        if route.get("label"):
            return str(route["label"])
    except Exception:  # noqa: BLE001
        pass
    return _LABELS.get(venue_id, venue_id)


def adapter_row(venue_id: str, adapter: Any) -> Dict[str, Any]:
    """
    One venue, described by what its code can do - not by what it might.

    `implementation_status` is the adapter's own claim about its client, and the
    flags beside it are read from the same AdapterCapability the runtime gates
    consult, so this row cannot say "can trade" while `can_place_real_orders`
    says otherwise.
    """
    caps = getattr(adapter, "capabilities", None)
    status = getattr(caps, "implementation_status", STATUS_UNIMPLEMENTED)
    reads_now = (status != STATUS_UNIMPLEMENTED
                 and bool(getattr(caps, "supports_market_discovery", False))
                 and not bool(getattr(caps, "requires_credentials", False)))
    needs_login = bool(getattr(caps, "requires_credentials", False))
    real_path = bool(getattr(caps, "real_order_path", False))
    armed = bool(getattr(adapter, "can_place_real_orders", False))

    if status == STATUS_UNIMPLEMENTED:
        use = USE_NO_CLIENT
    elif status == STATUS_SCANNER:
        use = USE_SCANNER
    elif real_path:
        use = USE_REAL_MONEY
    elif needs_login:
        use = USE_NEEDS_LOGIN
    else:
        use = USE_PAPER_ONLY

    funding = _funding_note(venue_id)

    if use == USE_NO_CLIENT:
        can_run_today = False
        what_it_needs = (getattr(caps, "implementation_note", "")
                         or "no client implementation")
        why = ("PTAI has no client for this venue, so it returns no markets. "
               "It is listed so you can see it exists and has not been built.")
    elif use == USE_SCANNER:
        can_run_today = True
        what_it_needs = "an Apify key, to read its ranked opportunities"
        why = ("Read-only aggregator. It ranks arbitrage opportunities it has "
               "already found; it cannot hold a position.")
    elif use == USE_REAL_MONEY:
        can_run_today = True
        if armed:
            what_it_needs = "nothing further - it is armed and funded"
        else:
            what_it_needs = ("venue credentials and an authorised budget; it is "
                             "the only adapter that can submit a real order")
        why = ("Reads live markets now and is paper-traded like every other "
               "venue. It is also the only venue whose orders can be real, "
               "which is what a funded, armed account unlocks.")
    elif use == USE_NEEDS_LOGIN:
        can_run_today = False
        what_it_needs = ("an account and credentials; it returns no markets "
                         "until it is logged in")
        why = ("Its public feed is closed. Credentials let it be scanned and "
               "paper-traded; it still cannot place a real order, because no "
               "submission path exists in the adapter.")
    else:  # USE_PAPER_ONLY
        can_run_today = True
        what_it_needs = "nothing - it reads public data with no account"
        why = ("Scanned and paper-traded every cycle at no cost. It cannot hold "
               "real money: the adapter has no way to submit an order.")

    return {
        "venue_id": venue_id,
        "label": _label(venue_id, adapter),
        "venue_type": getattr(getattr(adapter, "venue_type", None), "value", None),
        "implementation_status": status,
        "reads_live_markets_now": reads_now,
        "needs_credentials": needs_login,
        # "Paper-tradable" means it can be filled in simulation TODAY: a real
        # market feed, reachable without a login. A venue that needs credentials
        # before it returns a single market is not tradable yet - it is
        # available once you log in, which is what `what_it_needs` says.
        "paper_tradable": (status == STATUS_LIVE
                           and bool(getattr(caps, "supports_market_discovery", False))
                           and not needs_login),
        "can_place_real_orders": armed,
        "real_order_path": real_path,
        "can_run_today": can_run_today,
        "use": use,
        "why": why,
        "what_it_needs": what_it_needs,
        "min_order_usd": getattr(caps, "min_order_usd", None),
        **funding,
    }


def build_inventory(registry: Any) -> Dict[str, Any]:
    """Every registered adapter, as the operator's answer table."""
    adapters = getattr(registry, "adapters", None) or {}
    rows = {venue_id: adapter_row(venue_id, adapter)
            for venue_id, adapter in sorted(adapters.items())}

    def count(pred) -> int:
        return sum(1 for row in rows.values() if pred(row))

    counts = {
        "registered": len(rows),
        "readable_now": count(lambda r: r["reads_live_markets_now"]),
        "paper_tradable": count(lambda r: r["paper_tradable"]),
        "can_place_real_orders": count(lambda r: r["can_place_real_orders"]),
        "real_order_path": count(lambda r: r["real_order_path"]),
        "need_credentials": count(lambda r: r["needs_credentials"]),
        "no_client": count(lambda r: r["use"] == USE_NO_CLIENT),
    }
    return {
        "available": bool(rows),
        "at": datetime.now(timezone.utc).isoformat(),
        "source": "the live registry the agent built at startup",
        "counts": counts,
        "venues": rows,
    }


def record_inventory(storage: Any, registry: Any) -> Optional[Dict[str, Any]]:
    """
    Persist the inventory for the console process to read.

    Called by the agent's own loop, never by a doctor or a screen: a check that
    writes is not a check, and a console that builds its own registry is a second
    answer to "what is registered".
    """
    if storage is None or registry is None:
        return None
    try:
        payload = build_inventory(registry)
        storage.set_state(VENUE_INVENTORY_KEY, json.dumps(payload))
        return payload
    except Exception as e:  # noqa: BLE001 - never break a cycle over a screen
        logger.warning(f"Could not record the venue inventory: "
                       f"{type(e).__name__}: {e}")
        return None


def load_inventory(storage: Any) -> Dict[str, Any]:
    """
    The inventory the agent recorded, with an explicit "not yet" state.

    An empty table and an unknown table look identical on a dashboard and mean
    opposite things - one says "nothing is registered", the other says "ask
    again once the agent has started". The caller gets `available: False` and a
    reason in the second case.
    """
    block: Dict[str, Any] = {
        "available": False,
        "source": f"state key {VENUE_INVENTORY_KEY}",
        "venues": {},
    }
    if storage is None:
        block["reason"] = "no storage"
        return block
    try:
        raw = storage.get_state(VENUE_INVENTORY_KEY)
    except Exception as e:  # noqa: BLE001
        block["reason"] = f"{type(e).__name__}: {e}"
        return block
    if not raw:
        block["reason"] = ("the agent has not recorded its venue list yet - "
                            "start it once and every venue it knows about "
                            "appears here")
        return block
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as e:
        block["reason"] = f"unreadable inventory ({e})"
        return block
    block.update(payload)
    block["available"] = bool(payload.get("venues"))
    if not block["available"]:
        block["reason"] = "the recorded inventory is empty"
    return block


def inventory_line(inventory: Dict[str, Any]) -> str:
    """One sentence for the CLI and the panel headline."""
    if not inventory.get("available"):
        return f"venue list: {inventory.get('reason') or 'not recorded'}"
    counts = inventory.get("counts") or {}
    ordered = [v for v in (inventory.get("venues") or {}).values()
               if v.get("use") == USE_REAL_MONEY]
    can_be_real = ", ".join(v["label"] for v in ordered) or "none"
    return (f"{counts.get('registered', 0)} venues registered: "
            f"{counts.get('readable_now', 0)} readable now, "
            f"{counts.get('paper_tradable', 0)} paper-tradable, "
            f"{counts.get('can_place_real_orders', 0)} armed for real orders "
            f"(can ever: {can_be_real}), "
            f"{counts.get('no_client', 0)} with no client written")


def data_dir_for(storage: Any) -> str:
    """The data directory beside a storage's database, for sibling readers."""
    try:
        return str(Path(getattr(storage, "db_path")).parent)
    except TypeError:
        return "./data"
