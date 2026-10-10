"""
The logins PTAI needs, where they are kept, and whether the agent will use them.

The operator's request was plain: "my logins need to be saved somewhere so I don't
always have to log in when the system is running automatically". Before this
module the answer was no:

  * a `Vault` existed and wrote `data/vault.json`, but nothing in the console could
    put anything INTO it. The only way in was a Python call;
  * the Betfair adapter - the sports exchange that carries goals, corners and
    cards - was constructed from `os.getenv("BETFAIR_USERNAME")` in the middle of
    the trading loop. No env var, no login, and no way to set one from the product;
  * Kalshi was constructed as `KalshiAdapter()` with no arguments at all, so a key
    the operator owned could never reach it;
  * `sports_data` read `settings.the_odds_api_key` and
    `settings.football_data_token`, which were not fields on `Settings` - they could
    only ever be empty, while the engine's own error message told the operator to
    "add THE_ODDS_API_KEY or FOOTBALL_DATA_TOKEN in Setup". There was no Setup.

So this module is the one place that answers three questions per login:

  1. WHAT IS IT FOR - which venue or feed it unlocks, and what PTAI can do with
     that which it cannot do without;
  2. WHERE IS IT KEPT - the operator's saved copy in the encrypted local vault,
     or the environment they set up before this existed. Saved wins, because the
     operator typing it into the product is the most recent instruction, and
     `.env` keeps working so nobody's existing setup breaks. The source is
     reported per field so the screen never has to guess;
  3. WILL THE AGENT USE IT - `apply_to_settings` fills the settings the adapters
     read, and `refresh_adapters` pushes credentials into adapters that are
     ALREADY RUNNING, so a login saved while the agent is up takes effect on the
     next cycle instead of waiting for a restart nobody mentioned.

Secrets are handed to the vault with an explicit `secret_fields` list. The vault's
own rule was "encrypt the values whose KEY NAME contains PRIVATE/KEY/TOKEN", which
would have written a Betfair password and a Kalshi `api_secret` to disk in
plaintext - the two credentials most likely to be reused elsewhere. Names are not
a security policy; the caller says which fields are secret and the vault obeys.

Nothing here ever returns a secret to a caller that only wants to display it:
`describe()` returns masked values and the source, never the value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# The vault teammate name the trading agent already stores under. Kept as one
# constant so the console and the agent cannot disagree about where a login lives.
VAULT_OWNER = "Trader"

# The tool name the Polymarket credentials were already stored under before this
# module existed (v3_loop read "polymarket_clob"). Kept as an alias so an operator
# who saved them the old way is not asked to type a private key twice.
ALIASES: Dict[str, Tuple[str, ...]] = {"polymarket": ("polymarket_clob",)}


@dataclass(frozen=True)
class Field:
    """One piece of a login."""

    name: str
    label: str
    env: str = ""
    # Whether the vault must encrypt it. True for anything reusable as a secret;
    # a public wallet address is not, and encrypting it would only make the vault
    # harder to read.
    secret: bool = True
    required: bool = True
    hint: str = ""
    example: str = ""
    # Which `Settings` attribute the adapters read, when they read settings at
    # all. The bridge from "saved" to "the agent is using it".
    setting: str = ""


@dataclass(frozen=True)
class Tool:
    """A login, what it unlocks, and the adapter attribute it lands on."""

    name: str
    label: str
    kind: str                     # venue | feed | sentiment
    unlocks: str
    fields: Tuple[Field, ...]
    venue_id: str = ""
    # What is still missing after the login, so the screen can be honest rather
    # than implying that saving a key makes a venue fundable.
    then: str = ""
    docs: str = ""


TOOLS: Dict[str, Tool] = {
    "polymarket": Tool(
        name="polymarket",
        label="Polymarket",
        kind="venue",
        venue_id="polymarket",
        unlocks=("the only venue PTAI can place a REAL order on: signing key "
                 "and the funding wallet"),
        then=("funding happens at Polymarket (in-app card, or USDC on Polygon); "
              "then authorise a budget on the Money tab"),
        fields=(
            Field("private_key", "Signing private key (Polygon wallet)",
                  env="POLYMARKET_PRIVATE_KEY", secret=True,
                  hint="0x + 64 hex characters. Used to sign orders locally; it "
                       "never leaves this machine."),
            Field("funder", "Wallet address holding the USDC (funder)",
                  env="POLYMARKET_FUNDER_ADDRESS", secret=False,
                  hint="0x + 40 hex characters. The address your money sits on."),
        ),
    ),
    "kalshi": Tool(
        name="kalshi",
        label="Kalshi",
        kind="venue",
        venue_id="kalshi",
        unlocks=("the account read (balance, positions) and the real order path: "
                 "Kalshi signs every request with the RSA key that belongs to the "
                 "API key you create"),
        then=("money still has to reach Kalshi through their own USD rails (US "
              "bank account, ACH), and their adapter marks Uganda restricted - so "
              "an order path alone does not make it fundable from here"),
        docs="https://docs.kalshi.com/getting_started/api_keys",
        fields=(
            Field("api_key", "API key id", env="KALSHI_API_KEY",
                  hint="Shown next to the key in your Kalshi account (API keys)."),
            Field("private_key", "RSA private key (PEM)", env="KALSHI_PRIVATE_KEY",
                  required=False,
                  hint=("Paste the whole key, -----BEGIN to -----END. Kalshi "
                        "shows it once when the API key is created; it is what "
                        "signs every order, so trading needs it.")),
            Field("api_secret", "API secret (legacy)", env="KALSHI_API_SECRET",
                  required=False),
            Field("member_id", "Member id", env="KALSHI_MEMBER_ID",
                  secret=False, required=False),
            Field("environment", "Environment (production or demo)",
                  env="KALSHI_ENVIRONMENT", secret=False, required=False,
                  hint=("'demo' uses Kalshi's demo exchange (separate keys, mock "
                        "funds) - the safe place to prove the order path first.")),
        ),
    ),
    "betfair": Tool(
        name="betfair",
        label="Betfair Exchange",
        kind="venue",
        venue_id="betfair",
        unlocks=("the sports exchange feed: football fixtures priced across "
                 "match odds, goals, corners and cards"),
        then=("reads and prices those markets, reads the account balance and open "
              "orders, and can submit a real back bet and cancel it - so the "
              "account-health check can prove order permission instead of assuming "
              "it. Whether Betfair accepts you as a customer is the venue's own "
              "decision: an account in a jurisdiction it does not serve cannot be "
              "funded, and the venue panel says so."),
        docs="https://www.betfair.com/exchange (app key via your Betfair account)",
        fields=(
            Field("username", "Username", env="BETFAIR_USERNAME", secret=False),
            Field("password", "Password", env="BETFAIR_PASSWORD"),
            Field("app_key", "Application key", env="BETFAIR_APP_KEY",
                  hint="From your Betfair developer account."),
        ),
    ),
    "the_odds_api": Tool(
        name="the_odds_api",
        label="The Odds API (sports odds feed)",
        kind="feed",
        unlocks=("a genuine bookmaker consensus - many books, de-vigged - for the "
                 "sports lane, instead of one feed that can block your machine"),
        then="free tier is plenty for a handful of leagues per cycle",
        fields=(
            Field("api_key", "API key", env="THE_ODDS_API_KEY",
                  setting="the_odds_api_key"),
        ),
    ),
    "football_data": Tool(
        name="football_data",
        label="football-data.org (fixtures + standings)",
        kind="feed",
        unlocks="football fixtures, results and standings without HTML scraping",
        then="free tier covers the main European leagues",
        fields=(
            Field("token", "API token", env="FOOTBALL_DATA_TOKEN",
                  setting="football_data_token"),
        ),
    ),
    "x_api": Tool(
        name="x_api",
        label="X (Twitter) API",
        kind="sentiment",
        unlocks="sentiment without scraping, which X keeps blocking",
        then="optional. With no token PTAI says so and scores without X",
        fields=(
            Field("bearer_token", "Bearer token", env="X_BEARER_TOKEN",
                  secret=True, required=False, setting="x_bearer_token"),
        ),
    ),
    "manifold": Tool(
        name="manifold",
        label="Manifold Markets (play money)",
        kind="venue",
        venue_id="manifold",
        unlocks=("the Mana account read: the balance behind the market feed, and "
                 "a real outcome for every paper trade this venue resolves"),
        then=("optional. Manifold's markets are read without any key, and its "
              "resolution is too, so this only adds the account read. Mana "
              "cannot be cashed out: this venue can never hold your money."),
        docs="https://docs.manifold.markets/api (API key in your profile, edit)",
        fields=(
            Field("api_key", "API key", env="MANIFOLD_API_KEY", required=False),
        ),
    ),
    "simmer": Tool(
        name="simmer",
        label="Simmer (free virtual-currency venue)",
        kind="venue",
        venue_id="simmer",
        unlocks=("the whole venue: its synthetic $SIM venue, the only place PTAI "
                 "can submit an order today. No card, no wallet, no deposit - the "
                 "key is all it needs"),
        then=("nothing further for the $SIM venue. Simmer's REAL-money venues "
              "(Polymarket, Kalshi through Simmer) are not used here: they need a "
              "signed wallet, and PTAI talks to Polymarket directly instead."),
        docs="https://simmer.markets/dashboard (free API key)",
        fields=(
            Field("api_key", "API key", env="SIMMER_API_KEY", required=False),
        ),
    ),
    "betdaq": Tool(
        name="betdaq",
        label="Betdaq (play-money betting exchange)",
        kind="venue",
        venue_id="betdaq",
        unlocks=("the whole venue: its play-money markets, where real questions "
                 "are matched by the venue's own engine and settled by the venue. "
                 "The login is all it needs - play money needs no card, no wallet "
                 "and no deposit"),
        then=("nothing further for the play markets. Betdaq's REAL-money markets "
              "are not used here: this adapter is pinned to the venue's play "
              "markets, so no real order can be placed and the venue can never "
              "hold your money through it."),
        docs="https://api.betdaq.com/v2.0/Docs (Betdaq API; login from your Betdaq account)",
        fields=(
            Field("username", "Username", env="BETDAQ_USERNAME", secret=False),
            Field("password", "Password", env="BETDAQ_PASSWORD"),
        ),
    ),
    "apify": Tool(
        name="apify",
        label="Apify (paid arb scanner)",
        kind="feed",
        unlocks="ranked cross-venue opportunities as an extra input",
        then="paid per 1000 matched pairs - optional at $50 of capital",
        fields=(
            Field("api_token", "API token", env="APIFY_API_TOKEN", required=False),
        ),
    ),
}

#: venue id -> tool name, so a venue row can offer its own login.
TOOL_FOR_VENUE: Dict[str, str] = {
    tool.venue_id: tool.name for tool in TOOLS.values() if tool.venue_id
}


# ----------------------------------------------------------------------
# the vault
# ----------------------------------------------------------------------

def vault_path(data_dir: str = "./data") -> str:
    return os.path.join(data_dir, "vault.json")


def get_vault(data_dir: str = "./data"):
    """The vault, or None with a reason if it cannot be opened."""
    from ..vault.vault import Vault  # noqa: WPS433

    return Vault(vault_path=vault_path(data_dir))


def _vault_read(tool: Tool, data_dir: str) -> Dict[str, Any]:
    """
    What the operator saved, under this tool's name or any alias of it.

    A vault that cannot be read returns {} rather than raising: "no login saved"
    and "the vault is unreadable" are different facts, and the caller reports
    which one it has.
    """
    names = (tool.name,) + ALIASES.get(tool.name, ())
    try:
        vault = get_vault(data_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not open the credential vault: "
                       f"{type(e).__name__}: {e}")
        return {}
    for name in names:
        try:
            stored = vault.get_tool_credentials(VAULT_OWNER, name) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read {name} credentials from the vault: "
                           f"{type(e).__name__}: {e}")
            continue
        if stored:
            return dict(stored)
    return {}


def _env_read(field: Field) -> Optional[str]:
    """
    The environment's value, through Settings when the field maps to one.

    Settings is the reader the adapters use, so reading the same object here means
    "saved or from .env" cannot disagree with what the agent actually consumed.
    """
    if field.setting:
        try:
            from ..config import get_settings  # noqa: WPS433

            value = getattr(get_settings(), field.setting, None)
            if value:
                return str(value)
        except Exception:  # noqa: BLE001 - fall through to the raw environment
            pass
    if field.env:
        value = os.environ.get(field.env)
        if value:
            return value
    return None


def resolve(tool_name: str, data_dir: str = "./data") -> Dict[str, str]:
    """
    The value of every FIELD for a tool: saved first, environment second.

    Saved wins. The operator typing a new key into the console is a later
    instruction than a line in `.env` they may have forgotten about, and the
    screen reports which one is in force.
    """
    tool = TOOLS.get(tool_name)
    if tool is None:
        return {}
    saved = _vault_read(tool, data_dir)
    values: Dict[str, str] = {}
    for f in tool.fields:
        value = saved.get(f.name)
        if not value:
            value = _env_read(f)
        if value:
            values[f.name] = str(value).strip()
    return values


def _validate(tool: Tool, values: Dict[str, str]) -> List[str]:
    """Format checks that exist, applied so a typo is caught while typing."""
    problems: List[str] = []
    if tool.name == "polymarket":
        from ..security import validate_funder_address, validate_private_key  # noqa: WPS433

        if values.get("private_key"):
            try:
                validate_private_key(values["private_key"])
            except ValueError as e:
                problems.append(f"private_key: {e}")
        if values.get("funder"):
            try:
                validate_funder_address(values["funder"])
            except ValueError as e:
                problems.append(f"funder: {e}")
    return problems


def save(tool_name: str, values: Dict[str, Any],
         data_dir: str = "./data") -> Dict[str, Any]:
    """
    Save a login. Returns (ok, saved fields, problems) - never the values.

    An empty string for a REQUIRED field is refused rather than stored: a login
    with a blank password reads as "configured" to every downstream check and then
    fails at the venue, which is the least debuggable state a credential can be in.
    """
    tool = TOOLS.get(tool_name)
    if tool is None:
        return {"ok": False, "error": f"unknown login {tool_name!r}",
                "saved": [], "problems": []}

    existing = _vault_read(tool, data_dir)
    submitted = {f.name: str(values.get(f.name) or "").strip() for f in tool.fields}

    # MERGE FIRST, then validate what the login will actually BE. An operator
    # editing one field of a saved login must not be told the OTHER fields are
    # missing - they are already stored. A blank optional field keeps what is
    # there rather than deleting it by accident, and a blank required field is
    # only a problem when nothing is stored for it either.
    record: Dict[str, str] = {}
    saved_names: List[str] = []
    for f in tool.fields:
        value = submitted[f.name] or str(existing.get(f.name) or "")
        if not value:
            continue
        record[f.name] = value
        if submitted[f.name]:
            saved_names.append(f.name)

    missing = [f.label for f in tool.fields
               if f.required and not record.get(f.name)]
    if missing:
        return {"ok": False, "saved": [],
                "error": "still needed: " + ", ".join(missing),
                "problems": []}
    if not record:
        return {"ok": False, "saved": [], "problems": [],
                "error": "nothing to save"}

    problems = _validate(tool, {f.name: record.get(f.name, "") for f in tool.fields})
    if problems:
        return {"ok": False, "saved": [], "error": "check the values",
                "problems": problems}

    secret_fields = [f.name for f in tool.fields if f.secret and record.get(f.name)]
    try:
        vault = get_vault(data_dir)
        # The alias is honoured on write too: a login that was stored as
        # "polymarket_clob" is updated there rather than leaving two copies.
        names = (tool.name,) + ALIASES.get(tool.name, ())
        target = tool.name
        for name in names[:-1]:
            if _vault_read(TOOLS[tool.name], data_dir) and vault.get_tool_credentials(
                    VAULT_OWNER, name):
                target = name
                break
        vault.store_tool_credentials(
            VAULT_OWNER, target, record, secret_fields=secret_fields)
    except Exception as e:  # noqa: BLE001
        from ..vault.vault import Vault  # noqa: WPS433,F401 - type clarity only

        if isinstance(e, TypeError):
            # A vault from before `secret_fields` existed. Refusing to write is
            # the only safe answer: storing a password unencrypted without saying
            # so would be worse than an error the operator can read.
            return {"ok": False, "saved": [], "problems": [],
                    "error": ("this PTAI's vault does not support marking secret "
                              "fields, so the login was NOT saved - update the "
                              "vault module and try again")}
        logger.error(f"Could not save {tool_name} credentials: "
                     f"{type(e).__name__}: {e}")
        return {"ok": False, "saved": [], "problems": [],
                "error": f"{type(e).__name__}: {e}"}

    logger.info(f"Saved {tool_name} login ({', '.join(saved_names)}) to the "
                f"local vault: {vault_path(data_dir)}")
    return {"ok": True, "saved": saved_names, "problems": [], "error": "",
            "tools": describe_all(data_dir)}


def forget(tool_name: str, data_dir: str = "./data") -> Dict[str, Any]:
    """Delete a saved login (every alias of it). Env values are untouched."""
    tool = TOOLS.get(tool_name)
    if tool is None:
        return {"ok": False, "error": f"unknown login {tool_name!r}"}
    removed = []
    try:
        vault = get_vault(data_dir)
        for name in (tool.name,) + ALIASES.get(tool.name, ()):
            try:
                if vault.revoke(VAULT_OWNER, name):
                    removed.append(name)
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    logger.info(f"Forgot {tool_name} login ({', '.join(removed) or 'nothing saved'})")
    return {"ok": True, "removed": removed, "tools": describe_all(data_dir)}


# ----------------------------------------------------------------------
# what the screen shows
# ----------------------------------------------------------------------

def _mask(value: str) -> str:
    if not value:
        return ""
    tail = value[-4:] if len(value) > 4 else ""
    return f"••••{tail}"


def describe(tool_name: str, data_dir: str = "./data") -> Dict[str, Any]:
    """
    The tool's state for display: which fields are set, from where, and what the
    login unlocks. Masked - a secret never leaves this function in the clear.
    """
    tool = TOOLS[tool_name]
    saved = _vault_read(tool, data_dir)
    fields: List[Dict[str, Any]] = []
    configured = True
    any_set = False
    for f in tool.fields:
        saved_value = saved.get(f.name)
        env_value = None if saved_value else _env_read(f)
        value = saved_value or env_value
        any_set = any_set or bool(value)
        if f.required and not value:
            configured = False
        fields.append({
            "name": f.name, "label": f.label, "required": f.required,
            "secret": f.secret, "hint": f.hint, "example": f.example,
            "set": bool(value),
            "source": ("saved" if saved_value else
                       ("environment" if env_value else "")),
            # Masked for secrets, and for a non-secret field only when it came
            # from the environment - the wallet address the operator typed is
            # theirs to see, and seeing it is how they check they typed it right.
            "shown": (_mask(str(value)) if (f.secret and value)
                      else (str(value) if value and saved_value else
                            (_mask(str(value)) if value else ""))),
        })
    optional = bool(tool.fields) and not any(f.required for f in tool.fields)
    # A login with no REQUIRED field used to read as "configured" from the moment
    # it existed, because the loop never saw a missing requirement. An optional
    # key that nobody typed in is not a key, so it must not be reported as saved.
    configured = configured and (any_set or not tool.fields)
    return {
        "name": tool.name, "label": tool.label, "kind": tool.kind,
        "venue_id": tool.venue_id, "unlocks": tool.unlocks, "then": tool.then,
        "docs": tool.docs, "fields": fields,
        "optional": optional,
        "configured": configured,
        "how_it_gets_used": ("the agent reads saved logins when it starts and "
                             "refreshes them at the top of every cycle"),
    }


def describe_all(data_dir: str = "./data") -> Dict[str, Any]:
    tools = {name: describe(name, data_dir) for name in TOOLS}
    needed = [name for name, t in tools.items() if t["fields"] is not None]
    return {
        "vault": vault_path(data_dir),
        "tools": tools,
        "configured": sorted(n for n in needed if tools[n]["configured"]),
        "needs_attention": sorted(n for n in needed if not tools[n]["configured"]),
        "note": ("Saved credentials live in the file above, encrypted, on this "
                 "machine only. Nothing is sent anywhere."),
    }


# ----------------------------------------------------------------------
# which venues these logins even cover
# ----------------------------------------------------------------------

def coverage(venues: Dict[str, Dict[str, Any]],
             data_dir: str = "./data") -> Dict[str, Any]:
    """
    Every venue, against the logins that exist: connected, needed, or not used.

    The operator's report: "i have alot of venues but all of them except two are
    saying unavailable even in paper mode which doesnt make sense unless they
    require login but it seems i cant even connect my credentials to most of
    them". Two separate truths sit behind that sentence, and neither was visible:

      * most venues take NO login at all - their markets are public, and PTAI
        reads and paper-trades them with no account. There is nothing to connect
        because nothing is missing;
      * some venues have NO client written yet, so a login would unlock nothing.
        Offering a form for them would be theatre;
      * and exactly one venue in this build (Betfair) is closed without a login -
        a form that does exist, and that the operator can use.

    So "no form" and "broken" are not the same thing, and this function is what
    lets the screen say which is which instead of leaving the operator to guess.
    `venues` is the recorded inventory keyed by venue id (see inventory.py).

    Returns counts plus one row per venue. `missing_form` is the only bucket that
    is ever a defect: a venue that REQUIRES credentials and has no way to enter
    them. It is reported rather than silently dropped.
    """
    rows: List[Dict[str, Any]] = []
    for venue_id, row in sorted((venues or {}).items()):
        use = str((row or {}).get("use") or "")
        needs = bool((row or {}).get("needs_credentials"))
        label = str((row or {}).get("label") or venue_id)
        tool = TOOL_FOR_VENUE.get(venue_id, "")
        configured = False
        if tool:
            try:
                configured = bool(describe(tool, data_dir).get("configured"))
            except Exception as e:  # noqa: BLE001 - a screen must still render
                logger.debug(f"Could not read the {tool} login: "
                             f"{type(e).__name__}: {e}")

        if use == "no_client":
            state = "no_client"
            why = ("PTAI has no client for this venue yet, so there is nothing "
                   "for a login to unlock.")
        elif tool and configured:
            state = "login_saved"
            why = "Your saved login is read at the start of every cycle."
        elif tool and needs:
            state = "login_required"
            why = (f"Closed without a login: save the {label} login here and it "
                   f"returns markets on the next cycle.")
        elif tool and bool((row or {}).get("real_order_path")):
            state = "login_available"
            why = ("Reads public data with no account; the login is for "
                   "authenticated reads and real orders.")
        elif tool:
            # The login exists but unlocks nothing transactional here: this
            # venue's markets - and Manifold's resolutions - are public. An
            # optional key that adds the account read is not a missing login, and
            # saying "login needed" would send the operator to fill in a form
            # that changes nothing about what the venue can do.
            state = "no_login_needed"
            why = ((f"Reads public data with no account and is paper-traded for "
                    f"free. An optional {label} login adds the account read only."))
        elif needs:
            state = "missing_form"
            why = ("This adapter requires credentials and no login form exists "
                   "for it - a defect, not a setting.")
        else:
            state = "no_login_needed"
            why = ("Reads public data with no account and is paper-traded for "
                   "free. There is nothing to connect.")
        rows.append({"venue_id": venue_id, "label": label, "tool": tool,
                     "configured": configured, "state": state, "why": why})

    def _n(state: str) -> int:
        return sum(1 for r in rows if r["state"] == state)

    return {
        "venues": rows,
        "counts": {
            "registered": len(rows),
            "logins_saved": _n("login_saved"),
            "logins_required": _n("login_required"),
            "logins_available": _n("login_available"),
            "no_login_needed": _n("no_login_needed"),
            "no_client": _n("no_client"),
            "missing_form": _n("missing_form"),
        },
        "note": ("A login exists only where an adapter reads one. Venues with no "
                 "client built yet say so rather than offering a form that would "
                 "unlock nothing."),
    }


# ----------------------------------------------------------------------
# handing them to the agent
# ----------------------------------------------------------------------

def apply_to_settings(settings, data_dir: str = "./data") -> Dict[str, bool]:
    """
    Fill the settings the adapters read. Returns tool -> configured.

    Applied at agent construction AND at the top of every cycle, so an operator
    who pastes a key into the console does not have to find and restart a window
    to make it count.
    """
    state: Dict[str, bool] = {}
    for name, tool in TOOLS.items():
        values = resolve(name, data_dir)
        state[name] = all(values.get(f.name) for f in tool.fields if f.required)
        for f in tool.fields:
            if not f.setting:
                continue
            value = values.get(f.name)
            if value is None:
                continue
            try:
                setattr(settings, f.setting, value)
            except Exception as e:  # noqa: BLE001 - a read-only field is not fatal
                logger.warning(f"Could not set settings.{f.setting} from the "
                               f"{name} login: {type(e).__name__}: {e}")
    return state


def refresh_adapters(registry, settings=None, data_dir: str = "./data") -> Dict[str, Any]:
    """
    Push saved logins into adapters that are ALREADY CONSTRUCTED.

    The agent builds its adapters once, at startup. Without this, a login saved
    while the agent is running sits in the vault until somebody restarts a window
    they were never told about - which is exactly the "I have to log in again"
    loop this module exists to end. Returns what changed, for the log and the
    console.
    """
    if registry is None:
        return {}
    from ..config import get_settings  # noqa: WPS433

    settings = settings or get_settings()
    appliers = {
        "polymarket": lambda adapter: _apply_polymarket(adapter, data_dir),
        "kalshi": lambda adapter: _apply_kalshi(adapter, data_dir),
        "betfair": lambda adapter: _apply_betfair(adapter, data_dir),
        "manifold": lambda adapter: _apply_manifold(adapter, data_dir),
        "simmer": lambda adapter: _apply_simmer(adapter, data_dir),
        "betdaq": lambda adapter: _apply_betdaq(adapter, data_dir),
    }
    changed: Dict[str, Any] = {}
    for venue_id, apply_fn in appliers.items():
        adapter = None
        try:
            adapter = registry.get_adapter_for_venue_id(venue_id)
        except Exception:  # noqa: BLE001
            adapter = None
        if adapter is None:
            continue
        try:
            result = apply_fn(adapter)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not refresh {venue_id} credentials on the "
                           f"running adapter: {type(e).__name__}: {e}")
            continue
        if result:
            changed[venue_id] = result
    if changed:
        logger.info("Credential refresh applied to running adapters: "
                    + ", ".join(f"{k}: {', '.join(v)}" for k, v in changed.items()))
    return changed


def _apply_manifold(adapter, data_dir: str) -> List[str]:
    """
    A Mana API key saved while the agent is running, applied to the client.

    The key is a header, so the session is rebuilt with it and the capability
    flag - computed at construction from what was known then - is raised, or the
    account read would keep reporting "no key" until a restart.
    """
    values = resolve("manifold", data_dir)
    key = values.get("api_key")
    if not key or getattr(adapter, "api_key", None) == key:
        return []
    applied = []
    adapter.api_key = key
    session = getattr(adapter, "session", None)
    if session is not None:
        session.headers.update({"Authorization": f"Key {key}"})
        applied.append("session_header")
    if hasattr(adapter, "capabilities"):
        if not adapter.capabilities.supports_portfolio:
            adapter.capabilities.supports_portfolio = True
            applied.append("supports_portfolio")
    if hasattr(adapter, "last_error"):
        adapter.last_error = ""
    applied.append("api_key")
    return applied


def _apply_simmer(adapter, data_dir: str) -> List[str]:
    """
    A Simmer key saved while the agent is running, applied to the adapter.

    The Simmer client is built lazily (`adapter._client()`), so the key landing
    on the adapter is what matters - and the capability flags that were computed
    from what was known at construction have to be raised, or the venue keeps
    reporting "needs a login" and no account read until the next restart.
    """
    values = resolve("simmer", data_dir)
    key = values.get("api_key")
    if not key or getattr(adapter, "api_key", None) == key:
        return []
    applied = ["api_key"]
    adapter.api_key = key
    # The cached client, if one was built without the key, is dropped rather
    # than patched: the SDK client carries the key inside its own session.
    if getattr(adapter, "_client_obj", None) is not None:
        adapter._client_obj = None
        applied.append("client_rebuilt")
    caps = getattr(adapter, "capabilities", None)
    if caps is not None:
        if not getattr(caps, "requires_credentials", False):
            caps.requires_credentials = True
            applied.append("requires_credentials")
        if not getattr(caps, "supports_portfolio", False):
            caps.supports_portfolio = True
            applied.append("supports_portfolio")
    if hasattr(adapter, "last_error"):
        adapter.last_error = ""
    return applied


def _apply_betdaq(adapter, data_dir: str) -> List[str]:
    """
    A Betdaq login saved while the agent is running, applied to the adapter.

    The client is built lazily (`adapter._client()`), so the login landing on
    the adapter is what matters - and the capability flags that were computed
    from what was known at construction have to be raised, or the venue keeps
    reporting "needs a login" and no account read until the next restart.
    """
    values = resolve("betdaq", data_dir)
    username = values.get("username")
    password = values.get("password")
    if not username or not password:
        return []
    if getattr(adapter, "username", "") == username \
            and getattr(adapter, "password", "") == password:
        return []
    applied = ["username", "password"]
    adapter.username = username
    adapter.password = password
    # The cached client, if one was built without the login, is dropped rather
    # than patched: the SDK client carries the login inside its own session.
    if getattr(adapter, "_client_obj", None) is not None:
        adapter._client_obj = None
        applied.append("client_rebuilt")
    caps = getattr(adapter, "capabilities", None)
    if caps is not None:
        if not getattr(caps, "supports_portfolio", False):
            caps.supports_portfolio = True
            applied.append("supports_portfolio")
    if hasattr(adapter, "last_error"):
        adapter.last_error = ""
    return applied


def _apply_polymarket(adapter, data_dir: str) -> List[str]:
    values = resolve("polymarket", data_dir)
    applied = []
    for attr, key in (("private_key", "private_key"), ("funder", "funder")):
        value = values.get(key)
        if value and getattr(adapter, attr, None) != value:
            setattr(adapter, attr, value)
            applied.append(attr)
    return applied


def _apply_kalshi(adapter, data_dir: str) -> List[str]:
    values = resolve("kalshi", data_dir)
    applied = []
    for attr, key in (("api_key", "api_key"), ("api_secret", "api_secret"),
                      ("member_id", "member_id"),
                      ("private_key_pem", "private_key"),
                      ("environment", "environment")):
        value = values.get(key)
        if not value:
            continue
        if getattr(adapter, attr, None) == value:
            continue
        setattr(adapter, attr, value)
        applied.append(attr)
    # The RSA key is what turns the order path on, and the capability flag is
    # computed at construction from what was known then. A login saved while the
    # agent is running has to raise it, or the venue keeps reporting "no order
    # path" until the next restart - the exact stale-flag loop this module ends.
    if getattr(adapter, "private_key_pem", None) and hasattr(adapter, "capabilities"):
        adapter.capabilities.supports_trading = bool(
            getattr(adapter, "api_key", None) and adapter.private_key_pem)
        applied.append("supports_trading")
    return applied


def _apply_betfair(adapter, data_dir: str) -> List[str]:
    values = resolve("betfair", data_dir)
    if not values:
        return []
    applied = []
    # BetfairExchangeAdapter holds a BetfairClient, and the client holds the
    # credentials it will log in with. Both are updated: setting only the
    # outer attributes would leave the client logging in with the old values.
    targets = [adapter]
    client = getattr(adapter, "client", None)
    if client is not None:
        targets.append(client)
    for target in targets:
        for attr, key in (("username", "username"), ("password", "password"),
                          ("app_key", "app_key")):
            value = values.get(key)
            if value and getattr(target, attr, None) != value:
                setattr(target, attr, value)
                applied.append(f"{type(target).__name__}.{attr}")
    # A login saved while the agent is running has to take effect now: the client
    # object holds the session that was built with the old credentials, and the
    # capability flag was computed at construction from what was known then.
    # Without this the venue keeps reporting "cannot trade" until a restart, and
    # the operator who just saved the login sees the old answer.
    if client is not None and hasattr(client, "_client"):
        if getattr(client, "_client", None) is not None:
            client._client = None
            applied.append("client_session_reset")
        client._logged_in = False
    configured = all(getattr(target, field, None)
                     for target in targets
                     for field in ("username", "password", "app_key"))
    if hasattr(adapter, "capabilities"):
        caps = adapter.capabilities
        if bool(caps.supports_trading) != bool(configured):
            caps.supports_trading = bool(configured)
            applied.append("supports_trading")
    return applied
