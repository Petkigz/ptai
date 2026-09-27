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
        unlocks="authenticated market reads and account/balance reads",
        then=("PTAI has no Kalshi order path yet (the adapter says so itself), so "
              "it stays paper even with a key - the key is what earns it the read "
              "quota to build a record"),
        fields=(
            Field("api_key", "API key", env="KALSHI_API_KEY"),
            Field("api_secret", "API secret", env="KALSHI_API_SECRET", required=False),
            Field("member_id", "Member id", env="KALSHI_MEMBER_ID",
                  secret=False, required=False),
        ),
    ),
    "betfair": Tool(
        name="betfair",
        label="Betfair Exchange",
        kind="venue",
        venue_id="betfair",
        unlocks=("the sports exchange feed: football fixtures priced across "
                 "match odds, goals, corners and cards"),
        then=("PTAI's Betfair adapter reads and prices; order placement is not "
              "written yet, so it runs in paper"),
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
    for f in tool.fields:
        saved_value = saved.get(f.name)
        env_value = None if saved_value else _env_read(f)
        value = saved_value or env_value
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
    return {
        "name": tool.name, "label": tool.label, "kind": tool.kind,
        "venue_id": tool.venue_id, "unlocks": tool.unlocks, "then": tool.then,
        "docs": tool.docs, "fields": fields,
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
                      ("member_id", "member_id")):
        value = values.get(key)
        if value and getattr(adapter, attr, None) != value:
            setattr(adapter, attr, value)
            applied.append(attr)
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
    return applied
