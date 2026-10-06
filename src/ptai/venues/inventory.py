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
                "residency_required": route.get("residency_required"),
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


#: The three layers, in the order they have to be built. A venue "reaches
#: Polymarket" when all three are present - and the last one is the only one that
#: can spend money.
REACH_LAYERS = ("reads_markets", "reads_account", "places_real_orders")

_LAYER_LABEL = {
    "reads_markets": "read live markets",
    "reads_account": "read the account (balance and positions)",
    "places_real_orders": "place a real order",
}


def _reach_layers(caps: Any, status: str, note: str, requires_creds: bool,
                  login_configured: bool, login: Dict[str, Any],
                  funding: Dict[str, Any], can_run_today: bool,
                  fundable_from_here: Optional[bool] = None,
                  eligibility: Optional[Dict[str, Any]] = None,
                  country_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Which of the three layers this venue has, and what the next piece of work is.

    Deliberately derived from the adapter's OWN capability declarations - the same
    flags the runtime gates read - so a venue cannot be described as closer to
    Polymarket here than it is in the code that would have to submit the order.
    """
    supported = bool(getattr(caps, "supports_market_discovery", False))
    layers = {
        "reads_markets": bool(status != STATUS_UNIMPLEMENTED and supported),
        "reads_account": bool(getattr(caps, "supports_portfolio", False)),
        "places_real_orders": bool(getattr(caps, "real_order_path", False)),
    }
    missing = [name for name in REACH_LAYERS if not layers[name]]

    if not layers["reads_markets"]:
        next_step = (note or "write the client: this venue returns no markets yet")
    elif not layers["reads_account"]:
        # Two different reasons share one flag, and the difference matters to the
        # operator: an adapter with no portfolio call is work for the code, an
        # adapter whose portfolio call needs a login is a form to fill in. The
        # login named on this row is what tells them apart.
        _login_name = login.get("label") or login.get("tool") or ""
        if _login_name:
            next_step = (f"save the {_login_name} login: the account read needs it "
                         f"before the venue will report a balance")
        else:
            next_step = ("write the account read (balance and open positions) - the "
                         "adapter has no portfolio call")
    elif not layers["places_real_orders"]:
        next_step = (note or "write the order path: the adapter has no submission "
                             "path, so it can be read and paper-traded only")
    else:
        next_step = "nothing - all three layers are written"

    # The two things code cannot supply. Said here rather than left to the
    # operator to discover after the next venue is built.
    if requires_creds and not login_configured:
        where = login.get("label") or login.get("tool") or ""
        next_step += (f"; then save the {where} login" if where
                      else "; then it needs a login PTAI can save")
    _country = (country_code or "").upper()
    if fundable_from_here is False:
        why = (funding.get("reason_unfundable")
               or (funding.get("residency_required")
                   and f"funding needs {funding['residency_required']}")
               or ((eligibility or {}).get("means"))
               or "no funding route is recorded")
        next_step += f"; and it cannot hold real money from here: {why}"
    elif fundable_from_here is None:
        next_step += (f"; and funding from {_country or 'here'} is unverified - "
                      f"check with the venue before relying on it")

    return {
        "reach": {
            "reference": "polymarket",
            "layers": layers,
            "missing": missing,
            "distance": len(missing),
            "next_step": next_step,
            "runs_today": bool(can_run_today),
            "layer_labels": {k: _LAYER_LABEL[k] for k in REACH_LAYERS},
        },
    }


def _login_state(venue_id: str, data_dir: Optional[str]) -> Dict[str, Any]:
    """
    The login this venue's adapter reads, and whether the operator has saved it.

    Betfair publishes no public market feed: with no login it returns nothing at
    all. The classification used to ask only "does this adapter require
    credentials" - so a venue whose login HAD been saved still read "needs a
    login" on the panel, and a saved login looked like it had done nothing. The
    operator's report was exactly that: "i cant even connect my credentials to
    most of them". Whether a login is still NEEDED is a fact about the vault, not
    about the adapter.
    """
    state: Dict[str, Any] = {"tool": "", "label": "", "configured": False}
    try:
        from .credentials import TOOL_FOR_VENUE, describe  # noqa: WPS433
    except Exception as e:  # noqa: BLE001 - the panel must still render
        logger.debug(f"Credential module unavailable: {type(e).__name__}: {e}")
        return state
    tool = TOOL_FOR_VENUE.get(venue_id)
    if not tool:
        return state
    state["tool"] = tool
    if not data_dir:
        # A caller with no vault to read: name the login, claim nothing about it.
        return state
    try:
        info = describe(tool, data_dir)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not read the {tool} login state: {type(e).__name__}: {e}")
        return state
    state["label"] = info.get("label") or tool
    state["configured"] = bool(info.get("configured"))
    return state


def _eligibility(adapter: Any, country_code: Optional[str]) -> Dict[str, Any]:
    """
    The venue's own answer about this operator's country.

    Read from `adapter.check_eligibility` - the same call the runtime consults -
    rather than from a table maintained here, so the row cannot claim access the
    adapter would refuse. An adapter that cannot answer says so.
    """
    out: Dict[str, Any] = {"country": (country_code or "").upper(),
                           "status": "", "means": ""}
    if not country_code:
        return out
    try:
        status = adapter.check_eligibility(str(country_code).upper())
    except Exception as e:  # noqa: BLE001 - a screen must still render
        logger.debug(f"{getattr(adapter, 'venue_id', '?')} eligibility failed: "
                     f"{type(e).__name__}: {e}")
        return out
    value = getattr(status, "value", status)
    out["status"] = str(value)
    out["means"] = {
        "eligible": "the venue accepts operators here",
        "restricted": ("the venue refuses operators here, so real money cannot be "
                       "deployed - it can still be read and paper-traded"),
        "requires_verification": ("the venue has to verify the operator before it "
                                  "will take real money"),
    }.get(str(value), "the adapter does not know, so this is unverified")
    return out


def adapter_row(venue_id: str, adapter: Any,
                login: Optional[Dict[str, Any]] = None,
                country_code: Optional[str] = None) -> Dict[str, Any]:
    """
    One venue, described by what its code can do - not by what it might.

    `implementation_status` is the adapter's own claim about its client, and the
    flags beside it are read from the same AdapterCapability the runtime gates
    consult, so this row cannot say "can trade" while `can_place_real_orders`
    says otherwise.

    `login` is the saved-credential state from `_login_state`. It decides whether
    a venue that requires credentials is still BLOCKED by them or merely WAS:
    a saved login is what turns "needs a login" into "runs today".
    """
    caps = getattr(adapter, "capabilities", None)
    status = getattr(caps, "implementation_status", STATUS_UNIMPLEMENTED)
    requires_creds = bool(getattr(caps, "requires_credentials", False))
    login = dict(login or {})
    login_configured = bool(login.get("configured"))
    # A login the operator has already saved is not still needed.
    needs_login = requires_creds and not login_configured
    supported = bool(getattr(caps, "supports_market_discovery", False))
    reads_now = (status != STATUS_UNIMPLEMENTED and supported
                 and not needs_login)
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
    eligibility = _eligibility(adapter, country_code)
    # CAN MONEY REACH THIS VENUE FROM WHERE THE OPERATOR IS. A funding route that
    # exists is not the same as a funding route this operator can use: Kalshi's
    # route is a US bank account, and its own adapter says "restricted" for UG.
    # True / False / None (unverified) - never a hopeful yes.
    if not funding.get("fundable"):
        fundable_from_here: Optional[bool] = False
    elif eligibility.get("status") == "restricted":
        fundable_from_here = False
    elif eligibility.get("status") == "eligible":
        fundable_from_here = True
    else:
        fundable_from_here = None

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
        # An order path does not imply a readable feed. Betfair has both an
        # adapter that can submit AND a closed market feed: saying "reads live
        # markets now and is paper-traded like every other venue" about it would
        # describe a venue that returns nothing until its login is saved.
        can_run_today = not needs_login
        _login_name = login.get("label") or login.get("tool") or ""
        if needs_login:
            what_it_needs = (
                f"an account and credentials for the {_login_name} login before "
                f"it returns a single market, then venue credentials and an "
                f"authorised budget to submit" if _login_name else
                "an account and credentials before it returns a single market, "
                "then an authorised budget to submit")
            why = ("Its public feed is closed and its adapter has a real "
                   "submission path: it returns no markets until the login is "
                   "saved, and once it is, it can be scanned, paper-traded and - "
                   "with a funded account that passes the health check - "
                   "submitted for real.")
        elif armed:
            what_it_needs = "nothing further - it is armed and funded"
            why = ("Reads live markets now and is paper-traded like every other "
                   "venue. Its adapter can also submit a real order once the login "
                   "is saved, the account is funded, and the account health check "
                   "has proven an order can be placed.")
        else:
            what_it_needs = ("venue credentials and an authorised budget; its "
                             "adapter has a real submission path")
            why = ("Reads live markets now and is paper-traded like every other "
                   "venue. Its adapter can also submit a real order once the login "
                   "is saved, the account is funded, and the account health check "
                   "has proven an order can be placed.")
    elif use == USE_NEEDS_LOGIN:
        can_run_today = False
        _login_name = login.get("label") or login.get("tool") or ""
        what_it_needs = (
            f"an account and credentials for the {_login_name} login; it "
            f"returns no markets until it is logged in" if _login_name else
            "an account and credentials; it returns no markets until it is "
            "logged in")
        why = ("Its public feed is closed. Credentials let it be scanned and "
               "paper-traded; it still cannot place a real order, because no "
               "submission path exists in the adapter.")
    else:  # USE_PAPER_ONLY
        can_run_today = True
        _login_name = login.get("label") or login.get("tool") or ""
        if login_configured:
            what_it_needs = ("nothing - the saved login is read at the start of "
                             "every cycle")
        elif _login_name:
            # Its markets are free and its login is optional, but it exists - and
            # "nothing" would hide a form the operator can use for the account read.
            what_it_needs = (f"nothing for its markets - the optional "
                             f"{_login_name} login adds the account read")
        else:
            what_it_needs = "nothing - it reads public data with no account"
        why = ("Scanned and paper-traded every cycle at no cost. It cannot hold "
               "real money: the adapter has no way to submit an order.")

    return {
        "venue_id": venue_id,
        "label": _label(venue_id, adapter),
        "venue_type": getattr(getattr(adapter, "venue_type", None), "value", None),
        "implementation_status": status,
        # HOW FAR THIS VENUE IS FROM POLYMARKET, layer by layer. "Can hold real
        # money" was one flag; the work is three layers and the operator asking
        # "let's start giving the venues enough code to reach Polymarket" needs to
        # see which layer is missing on each one, in a fixed order:
        #
        #   1. READS MARKETS  - a client that returns live markets at all
        #   2. READS ACCOUNT  - the venue will tell PTAI the balance and positions
        #   3. PLACES ORDERS  - a submission path that can reach the venue
        #
        # Plus the two things the code cannot supply: a login, and a way to fund
        # the account from where the operator is. Both are named rather than
        # implied, so no venue looks one small step away when it is three.
        "eligibility": eligibility,
        "fundable_from_here": fundable_from_here,
        **_reach_layers(caps, status, note=getattr(caps, "implementation_note", "") or "",
                        requires_creds=requires_creds, login_configured=login_configured,
                        login=login, funding=funding, can_run_today=can_run_today,
                        fundable_from_here=fundable_from_here,
                        eligibility=eligibility, country_code=country_code),
        "reads_live_markets_now": reads_now,
        "needs_credentials": needs_login,
        # The login this venue reads, and whether it is already saved. Present on
        # every row - an empty tool means this venue reads no login at all - so
        # "why can I not connect this one?" is answerable per venue rather than
        # guessed at from a panel that lists seven logins beside nineteen venues.
        "login": login,
        "logged_in": login_configured,
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


def build_inventory(registry: Any,
                    data_dir: Optional[str] = None,
                    country_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Every registered adapter, as the operator's answer table.

    `data_dir` is where the saved logins live. With it, a venue whose login the
    operator saved is classified by what it can DO now rather than by what its
    adapter needs in principle. Without it (a caller that only wants the code's
    own view) the login named on the row is unclaimed, exactly as before.
    """
    adapters = getattr(registry, "adapters", None) or {}
    country_code = country_code or getattr(registry, "country_code", None)
    rows = {venue_id: adapter_row(venue_id, adapter,
                                  login=_login_state(venue_id, data_dir),
                                  country_code=country_code)
            for venue_id, adapter in sorted(adapters.items())}

    def count(pred) -> int:
        return sum(1 for row in rows.values() if pred(row))

    counts = {
        "registered": len(rows),
        "readable_now": count(lambda r: r["reads_live_markets_now"]),
        "paper_tradable": count(lambda r: r["paper_tradable"]),
        "can_place_real_orders": count(lambda r: r["can_place_real_orders"]),
        "real_order_path": count(lambda r: r["real_order_path"]),
        # Venues STILL blocked by a login (not: venues whose adapter could read
        # one), and venues whose login is already saved and in use.
        "need_credentials": count(lambda r: r["needs_credentials"]),
        "logins_configured": count(lambda r: r["logged_in"]),
        "no_client": count(lambda r: r["use"] == USE_NO_CLIENT),
        # Venues where the operator could actually put money AND PTAI could
        # actually submit the order. This is the number that matters to the
        # question "let's give the venues enough code": everything else is work
        # that cannot yet become a trade.
        "can_hold_real_money_from_here": count(
            lambda r: (r.get("reach") or {}).get("layers", {})
            .get("places_real_orders") is True
            and r.get("fundable_from_here") is True),
        "fundable_from_here": count(lambda r: r.get("fundable_from_here") is True),
    }
    # WHERE THE NEXT PIECE OF VENUE CODE SHOULD GO. The operator's question was
    # "all at once or one by one" - this orders the answer: venues that already
    # read and only lack the order path first, then the ones missing two layers,
    # then the ones with no client at all (which are the most work and the least
    # certain). Within a distance, fundable venues come first: a venue nobody can
    # put money into is a paper exercise whatever is written for it.
    def _rank(row: Dict[str, Any]) -> tuple:
        reach = row.get("reach") or {}
        # Fundable-from-here is ranked ABOVE unfundable: a venue whose deposit
        # route this operator cannot use is a paper exercise whatever is written
        # for it, and the queue for real work should say so.
        return (int(reach.get("distance", 99)),
                row.get("fundable_from_here") is not True,
                str(row.get("venue_id")))

    ordered = sorted(rows.values(), key=_rank)
    return {
        "available": bool(rows),
        "at": datetime.now(timezone.utc).isoformat(),
        "source": "the live registry the agent built at startup",
        "counts": counts,
        "venues": rows,
        "reach": {
            "reference": "polymarket",
            "real_orders": [r["venue_id"] for r in ordered
                            if (r.get("reach") or {}).get("layers", {})
                            .get("places_real_orders")],
            "next_steps": [
                {"venue_id": r["venue_id"], "label": r["label"],
                 "distance": (r.get("reach") or {}).get("distance"),
                 "missing": (r.get("reach") or {}).get("missing"),
                 "next_step": (r.get("reach") or {}).get("next_step"),
                 "fundable": bool(r.get("fundable")),
                 "fundable_from_here": r.get("fundable_from_here"),
                 "can_run_today": bool(r.get("can_run_today"))}
                for r in ordered if (r.get("reach") or {}).get("distance")
            ],
        },
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
        payload = build_inventory(registry, data_dir=data_dir_for(storage))
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
    real = [v["label"] for v in (inventory.get("venues") or {}).values()
            if (v.get("reach") or {}).get("layers", {}).get("places_real_orders")]
    return (f"{counts.get('registered', 0)} venues registered, "
            f"{len(real)} can place a real order ({', '.join(real) or 'none'}): "
            f"{counts.get('readable_now', 0)} readable now, "
            f"{counts.get('paper_tradable', 0)} paper-tradable, "
            f"{counts.get('need_credentials', 0)} still needs a login "
            f"({counts.get('logins_configured', 0)} login(s) saved), "
            f"{counts.get('can_place_real_orders', 0)} armed for real orders "
            f"(can ever: {can_be_real}), "
            f"{counts.get('no_client', 0)} with no client written")


def data_dir_for(storage: Any) -> str:
    """The data directory beside a storage's database, for sibling readers."""
    try:
        return str(Path(getattr(storage, "db_path")).parent)
    except TypeError:
        return "./data"
