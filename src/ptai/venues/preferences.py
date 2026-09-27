"""
Which venues the agent may use, switched on and off by the operator.

PTAI registers nineteen adapters and scans them all. That is right for a fresh
install - paper trading costs nothing and the record is what earns a venue the
right to real money - but it left the operator with no answer to "I have a Kalshi
account, can I stop it scanning the eleven venues with no client written yet", or
"do not touch crypto, I only do politics".

So: an explicit, persisted choice per venue, and no other meaning attached to it.

  * DISABLED means the agent does not ask that venue for markets and does not
    trade it. It is the operator's instruction, and the log says so in those
    words - an off venue reported as a failure would make the operator think the
    product is broken;
  * ENABLED is the default for everything new. A venue nobody has an opinion
    about keeps being paper-traded, because that is how it earns a record;
  * it never overrides a safety check. A venue switched on is still refused live
    capital if it is not the selected live venue, not funded, not qualified, or
    if the money guard or the bench is stopping it. This switch only ever
    SUBTRACTS.

Kept in the state store rather than in `.env` because the console writes it and
the agent (a different process) reads it, and because the choice must survive a
restart - "it started scanning crypto again after the update" would be a bug the
operator cannot reproduce.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

STATE_KEY = "venues.preferences"


def load(storage) -> Dict[str, Any]:
    """
    The saved choices. Never raises: an unreadable store means "no opinions", and
    the agent keeps doing what it did before rather than refusing to scan.
    """
    empty: Dict[str, Any] = {"disabled": [], "enabled": [], "updated_at": ""}
    if storage is None:
        return empty
    try:
        raw = storage.get_state(STATE_KEY)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read venue preferences: "
                       f"{type(e).__name__}: {e}")
        return empty
    if not raw:
        return empty
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.warning(f"Venue preferences are unreadable ({e}); treating every "
                       f"venue as enabled")
        return empty
    if not isinstance(data, dict):
        return empty
    return {
        "disabled": [str(v) for v in (data.get("disabled") or [])],
        "enabled": [str(v) for v in (data.get("enabled") or [])],
        "updated_at": str(data.get("updated_at") or ""),
    }


def save(storage, *, disabled: List[str], enabled: List[str]) -> Dict[str, Any]:
    from datetime import datetime, timezone

    payload = {
        "disabled": sorted({str(v) for v in disabled if v}),
        "enabled": sorted({str(v) for v in enabled if v}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        storage.set_state(STATE_KEY, json.dumps(payload))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not save venue preferences: "
                     f"{type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, **payload}


def set_enabled(storage, venue_id: str, enabled: bool) -> Dict[str, Any]:
    """Switch one venue on or off, keeping the other choices."""
    current = load(storage)
    disabled = [v for v in current["disabled"] if v != venue_id]
    explicit = [v for v in current["enabled"] if v != venue_id]
    if enabled:
        explicit.append(venue_id)
    else:
        disabled.append(venue_id)
    result = save(storage, disabled=disabled, enabled=explicit)
    if result.get("ok"):
        logger.info(f"Venue {venue_id} switched "
                    f"{'ON' if enabled else 'OFF'} by the operator"
                    + ("" if enabled else
                       " - the agent will not ask it for markets or trade it"))
    return result


def is_enabled(storage, venue_id: str, preferences: Optional[Dict[str, Any]] = None) -> bool:
    prefs = preferences if preferences is not None else load(storage)
    # An explicit OFF always wins, even if the same id was also switched on at
    # some point: the operator's last instruction is the one stored in `disabled`,
    # and a stale `enabled` entry must not undo it.
    return str(venue_id) not in set(prefs.get("disabled") or [])


def filter_adapters(storage, adapters: List[Any],
                    preferences: Optional[Dict[str, Any]] = None):
    """
    (adapters the operator allows, [venue ids skipped]).

    Returned as a pair because the skipped list is shown to the operator: a venue
    that is missing from the scan must be visible as their own choice.
    """
    prefs = preferences if preferences is not None else load(storage)
    disabled = set(prefs.get("disabled") or [])
    allowed, skipped = [], []
    for adapter in adapters:
        venue_id = str(getattr(adapter, "venue_id", "") or "")
        if venue_id and venue_id in disabled:
            skipped.append(venue_id)
            continue
        allowed.append(adapter)
    return allowed, skipped
