"""
Account Health Engine - V10 FIX #4, rewritten in V11q

Venue qualification is not the same as "this venue is actually ready to trade",
and neither is "credentials are configured".

The previous version of this module looked like it checked eight things and in
fact proved none of them:

  * the portfolio check read `adapter.capabilities.supports_portfolio` - a
    capability FLAG - with the comment "Don't actually call for now to avoid
    rate limits". A flag says the adapter could fetch a portfolio; it says
    nothing about whether the API accepts this account's credentials.
  * `has_funds = True  # simplified` for any venue with credentials, so
    "configured" was indistinguishable from "funded".
  * whitebit and crypto_binance were set `healthy = True` unconditionally, no
    credentials required at all. `v3_loop`'s execution guard tests
    `account_health.healthy`, so a hardcoded True was a path by which a real
    order could be placed against an account nobody had ever authenticated.
  * the report advertised "Can place orders" and "Can cancel orders" as checks.
    Neither existed anywhere in the codebase.

This module now reports a READINESS LADDER, and each rung requires evidence
rather than a declaration:

    NOT_CONFIGURED -> CONFIGURED -> AUTHENTICATED -> FUNDED -> TRADE_PERMITTED

`ready_to_trade` is True only at TRADE_PERMITTED, which requires an actual order
to have been submitted and withdrawn. Nothing sets that today: no adapter
implements an order probe, so for every venue the honest answer is "not yet
verified" rather than "healthy". That is the point. The rung is recorded as a
named blocker so it appears as work to do instead of as a silent assumption.

The same provenance rule used for spreads applies to balances: an adapter that
returns `{"balance": 0}` without saying where the number came from has not told
us the account is empty, it has told us it does not know.
"""

from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import time

from ..venues.adapter import EligibilityStatus
from ..venues.registry import VenueRegistry


class TradeReadiness(str, Enum):
    """
    Ordered ladder of what has actually been PROVEN about a venue account.

    Ordering is meaningful: each rung implies every rung below it.
    """
    NOT_CONFIGURED = "not_configured"    # no credentials, nothing to prove
    CONFIGURED = "configured"            # credentials present but unverified
    AUTHENTICATED = "authenticated"      # a real authenticated call succeeded
    FUNDED = "funded"                    # balance read from that call, and > 0
    TRADE_PERMITTED = "trade_permitted"  # an order round trip succeeded


_READINESS_ORDER = [
    TradeReadiness.NOT_CONFIGURED,
    TradeReadiness.CONFIGURED,
    TradeReadiness.AUTHENTICATED,
    TradeReadiness.FUNDED,
    TradeReadiness.TRADE_PERMITTED,
]


def readiness_rank(readiness: TradeReadiness) -> int:
    return _READINESS_ORDER.index(readiness)


# Provenance tokens that mean "a VENUE API answered us". Only these prove the
# venue accepted our credentials.
#
# This list is deliberately a positive allowlist drawn from the adapters that
# really issue venue requests, rather than a "not obviously fake" heuristic.
# PolymarketAdapter.get_portfolio() reads the LOCAL database first and returns
# is_real=True with source="storage" - a true statement about its own storage,
# and no evidence whatsoever that Polymarket authenticated us. Accepting it
# would have made the auth rung certify a local file read.
#
# Source of each entry: grep the adapters for the provenance strings they
# actually emit after a successful venue request, and list those. An allowlist
# that does not contain the string a real adapter emits refuses real evidence -
# which is what happened to Polymarket: the executor's authenticated
# get_balance_allowance() answered with "clob_balance_allowance", the allowlist
# only knew "clob_real", and so a genuinely venue-confirmed balance was rejected
# as unverified. The capability existed on both sides and the two names never
# met.
_VENUE_AUTH_SOURCES = frozenset({
    "betfair_account_api",
    "betfair_exchange_live",
    "kalshi_api_real",
    "clob_real",
    # PolymarketExecutor.get_balance_allowance() - an authenticated CLOB read of
    # the account's collateral and allowance. The venue's own number.
    "clob_balance_allowance",
    "whitebit_api_real",
    "polymarket_data_api",
    "binance_api_real",
})

# Tokens that mean the producer is guessing, replaying local state, or falling
# back. Any of these disqualifies a source even if it also carries a venue token.
_NON_VENUE_SOURCE_MARKERS = (
    "storage", "local", "database", "cache",
    "mock", "fallback", "placeholder", "estimat", "attempted", "assumed",
)


def is_venue_side_provenance(provenance: str) -> bool:
    """Does this provenance prove a venue API answered, rather than our own DB?"""
    if not provenance:
        return False
    tokens = [t.strip().lower() for t in str(provenance).split("+") if t.strip()]
    if not tokens:
        return False
    if any(marker in token for token in tokens for marker in _NON_VENUE_SOURCE_MARKERS):
        return False
    return any(token in _VENUE_AUTH_SOURCES for token in tokens)


def read_balance(portfolio: Any) -> Tuple[Optional[float], bool, str]:
    """
    Read a balance out of a portfolio response, and say whether it is real.

    Mirrors `markets.orderbook.read_spread`: a number is only usable if the
    producer said where it came from. Returns (balance, is_real, provenance).

    is_real is False when:
      * the portfolio is not a mapping
      * there is no balance field at all
      * the response explicitly marks itself as a fallback / placeholder
        (`is_real: False`)
      * the venue said the account is not available (`available: False`)
      * the balance is not a number

    A genuine 0.0 from a real source IS real: it means the account is empty,
    which is information, not ignorance.
    """
    if not isinstance(portfolio, dict):
        return None, False, "not_a_mapping"

    # A producer that declares itself un-real is not a source.
    if portfolio.get("is_real") is False:
        return None, False, "declared_not_real"
    if portfolio.get("available") is False:
        return None, False, "account_unavailable"

    provenance = None
    for key in ("source", "balance_source", "provenance"):
        value = portfolio.get(key)
        if isinstance(value, str) and value.strip():
            provenance = value.strip()
            break

    if portfolio.get("is_real") is True and provenance is None:
        provenance = "declared_real"

    # A balance with no provenance is an unknown, not a zero. Several adapters
    # return exactly {"balance": 0, "positions": []} as a stub.
    if provenance is None:
        return None, False, "no_provenance"

    for key in ("balance", "available_balance", "actual_balance", "cash"):
        value = portfolio.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value), True, provenance

    # No usable number, but the provenance is still whatever the producer
    # declared - "clob_real" with no balance means the venue answered and told
    # us nothing about funds. Collapsing that into "no_balance_field" would
    # throw away the authentication evidence.
    return None, False, provenance


@dataclass
class AccountHealthResult:
    venue_id: str
    healthy: bool                 # kept for callers; now means ready_to_trade
    paper_trading_ok: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    checks: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    readiness: TradeReadiness = TradeReadiness.NOT_CONFIGURED
    ready_to_trade: bool = False
    blockers: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "healthy": self.healthy,
            "ready_to_trade": self.ready_to_trade,
            "readiness": self.readiness.value,
            "paper_trading_ok": self.paper_trading_ok,
            "reason": self.reason,
            "details": self.details,
            "checks": self.checks,
            "checks_failed": self.checks_failed,
            "blockers": self.blockers,
            "evidence": self.evidence,
        }


class AccountHealthEngine:
    """
    V10 FIX #4: an account readiness check, not a capability-flag echo.
    """

    # Venues whose credential fields we know how to look for. Anything not
    # listed is reported as "not configured" rather than assumed configured.
    _CREDENTIAL_FIELDS = {
        "polymarket": (("private_key", "funder"), "private_key+funder"),
        "kalshi": (("api_key",), "api_key"),
        "whitebit": (("api_key", "api_secret"), "api_key+secret"),
        "binance": (("api_key", "api_secret"), "api_key+secret"),
        "crypto_binance": (("api_key", "api_secret"), "api_key+secret"),
        "pionex": (("api_key", "api_secret"), "api_key+secret"),
        "grvt": (("api_key", "api_secret"), "api_key+secret"),
        "betfair": (("username", "app_key"), "username+app_key(+certs)"),
        "betdaq": (("username", "app_key"), "username+app_key"),
        "manifold": (("api_key",), "api_key"),
        "chatgpt": (("api_key",), "api_key"),
    }

    def __init__(self, venue_registry: VenueRegistry = None, bankroll: float = 50.0,
                 storage=None):
        self.venue_registry = venue_registry
        self.bankroll = bankroll
        # Where a completed identity/eligibility verification is recorded. Read
        # only for venues whose eligibility is REQUIRES_VERIFICATION - see the
        # geography rung.
        self.storage = storage
        self.health_cache: Dict[str, AccountHealthResult] = {}
        self.cache_ttl_seconds = 300  # 5 min cache
        self.last_check: Dict[str, float] = {}

    # -- verification evidence, for venues that require it ------------------

    def _verification_evidence(self, venue_id: str, adapter) -> Tuple[bool, str]:
        """
        Is there evidence that this account's verification was COMPLETED?

        Two sources, both requiring a positive record rather than an absence of
        complaints:

          * the adapter reporting it, if the venue exposes such a state;
          * the operator attesting to it, via record_verification(), because the
            KYC step happens at the venue and in the operator's name - exactly
            like the deposit.

        Silence is not evidence. No record means unverified, which caps the venue
        below live.
        """
        checker = getattr(adapter, "verification_evidence", None)
        if callable(checker):
            try:
                note = checker()
                if note:
                    return True, str(note)[:200]
            except Exception as e:
                logger.warning(
                    f"{venue_id}: venue verification check failed "
                    f"({type(e).__name__}: {e}) - treating as unverified")
        if self.storage is not None:
            try:
                recorded = self.storage.get_state(f"verification.{venue_id}")
                if recorded:
                    return True, str(recorded)[:200]
            except Exception as e:
                logger.error(
                    f"Could not read the verification record for {venue_id}: "
                    f"{type(e).__name__}: {e}. Treating as unverified.")
        return False, ""

    # -- rung 3: does the API actually accept these credentials? ------------

    async def _probe_authenticated(self, adapter) -> Tuple[bool, Optional[float], str, str]:
        """
        Make a REAL call and see whether the account answers.

        Returns (authenticated, balance, provenance, note).

        This is the rung the old implementation faked by reading
        `capabilities.supports_portfolio`. The rate-limit concern that
        motivated the flag is real but already handled: results are cached for
        300 seconds, and account health runs once per cycle, not per market.
        """
        try:
            portfolio = await adapter.get_portfolio()
        except Exception as e:
            return False, None, "", f"get_portfolio raised {type(e).__name__}: {e}"

        balance, _is_real, provenance = read_balance(portfolio)

        # Authentication and funding are SEPARATE rungs. Authentication is
        # decided by where the response came from, not by whether it contained a
        # number: a venue that answers "your balance is unavailable" has still
        # proven it accepts our credentials.
        if not is_venue_side_provenance(provenance):
            if provenance == "no_provenance":
                note = (
                    "portfolio response carried no provenance at all; cannot "
                    "confirm the venue accepted these credentials"
                )
            else:
                note = (
                    f"portfolio came from '{provenance}', which is not a "
                    f"venue-side read; credentials are still unverified. A local "
                    f"record is not proof the venue accepts these credentials."
                )
            return False, None, provenance, note

        return True, balance, provenance, f"venue responded, via {provenance}"

    # -- rung 5: may this account actually submit an order? ----------------

    async def _probe_order_permission(self, adapter, opportunity=None) -> Tuple[bool, str]:
        """
        Prove an order can be submitted, by submitting one.

        Requires the adapter to declare `supports_order_probe`, which means it
        can place a minimum-size order and cancel it. Polymarket does;
        everything else does not yet, and an adapter that does not declare it is
        reported as unproven rather than assumed - assuming it is how
        "configured" quietly becomes "ready".

        `opportunity` is passed through because a probe needs a specific
        tradeable market to test against. Probing "this account" in the abstract
        is not possible: permission is demonstrated on a real market or not at
        all.
        """
        caps = getattr(adapter, "capabilities", None)
        if caps is None or not getattr(caps, "supports_order_probe", False):
            return False, (
                "order submission is NOT verified: this adapter declares no "
                "order probe (place minimum order then cancel), so permission "
                "to trade is unproven"
            )

        probe = getattr(adapter, "probe_order_permission", None)
        if probe is None:
            return False, (
                "adapter declares supports_order_probe but implements no "
                "probe_order_permission()"
            )
        try:
            ok = await probe(opportunity)
        except TypeError:
            # An adapter whose probe takes no opportunity.
            try:
                ok = await probe()
            except Exception as e:
                return False, f"order probe raised {type(e).__name__}: {e}"
        except Exception as e:
            return False, f"order probe raised {type(e).__name__}: {e}"
        if ok:
            return True, "order submission verified by live round trip"
        return False, "order probe ran but did not confirm submission"

    # -- main ---------------------------------------------------------------

    async def check_venue_health(self, venue_id: str,
                                 opportunity=None) -> AccountHealthResult:
        """
        Determine how far up the readiness ladder this venue has been proven.

        `opportunity` is used only by the order probe, which needs a market to
        test against.
        """
        venue_id = venue_id.lower().strip()

        if venue_id in self.health_cache:
            last = self.last_check.get(venue_id, 0)
            if time.time() - last < self.cache_ttl_seconds:
                return self.health_cache[venue_id]

        checks: List[str] = []
        checks_failed: List[str] = []
        blockers: List[str] = []
        evidence: Dict[str, Any] = {}
        details: Dict[str, Any] = {"bankroll": self.bankroll}

        adapter = None
        if self.venue_registry and venue_id in self.venue_registry.adapters:
            adapter = self.venue_registry.adapters[venue_id]

        def finish(readiness: TradeReadiness, reason: str, paper_ok: bool) -> AccountHealthResult:
            ready = readiness == TradeReadiness.TRADE_PERMITTED
            result = AccountHealthResult(
                venue_id=venue_id,
                healthy=ready,
                paper_trading_ok=paper_ok,
                reason=reason,
                details=details,
                checks=checks,
                checks_failed=checks_failed,
                readiness=readiness,
                ready_to_trade=ready,
                blockers=blockers,
                evidence=evidence,
            )
            self.health_cache[venue_id] = result
            self.last_check[venue_id] = time.time()
            logger.info(
                f"Account health {venue_id}: readiness={readiness.value} "
                f"ready_to_trade={ready} paper_ok={paper_ok} - {reason}"
            )
            return result

        if adapter is None:
            checks_failed.append(f"No adapter for {venue_id}")
            blockers.append("no_adapter")
            return finish(
                TradeReadiness.NOT_CONFIGURED,
                f"No adapter for {venue_id} - paper/shadow only",
                True,
            )

        details["has_adapter"] = True
        caps = adapter.capabilities
        details["implementation_status"] = caps.implementation_status
        details["supports_trading_claim"] = caps.supports_trading

        # -- rung 1: is this venue implemented at all? --
        if not caps.is_implemented:
            blockers.append("not_implemented")
            checks_failed.append(
                f"{venue_id} has no client implementation ({caps.implementation_status}); "
                f"{caps.implementation_note or 'no note'}"
            )
            return finish(
                TradeReadiness.NOT_CONFIGURED,
                f"{venue_id} is not implemented - nothing to authenticate",
                True,
            )
        checks.append(f"Adapter implementation status is {caps.implementation_status}")

        # -- rung 2: are credentials present? --
        fields, credential_type = self._CREDENTIAL_FIELDS.get(venue_id, ((), "none"))
        if not fields and caps.supports_trading:
            # A trading venue we have no credential model for. Report the gap
            # rather than guessing at attribute names.
            blockers.append("credential_model_unknown")
            checks_failed.append(
                f"{venue_id} declares trading but has no known credential "
                f"fields, so credentials cannot be checked"
            )
            return finish(
                TradeReadiness.NOT_CONFIGURED,
                f"{venue_id}: credential model unknown - cannot confirm configuration",
                True,
            )

        present = {f: bool(getattr(adapter, f, None)) for f in fields}
        details["credentials"] = present
        details["credential_type"] = credential_type
        has_credentials = bool(fields) and all(present.values())

        if not fields:
            # Not a venue that takes credentials (scanner / paper venue).
            checks.append(f"{venue_id} requires no credentials")
            return finish(
                TradeReadiness.CONFIGURED,
                f"{venue_id} is a data/scanner venue; no trading account to verify",
                True,
            )

        if not has_credentials:
            missing = [f for f, ok in present.items() if not ok]
            checks_failed.append(f"Missing credentials: {', '.join(missing)}")
            blockers.append("credentials_missing")
            return finish(
                TradeReadiness.NOT_CONFIGURED,
                f"{venue_id}: missing {', '.join(missing)} - paper only",
                True,
            )
        checks.append(f"Credentials present ({credential_type})")

        # -- rung 3: does the API accept them? --
        authenticated, balance, provenance, note = await self._probe_authenticated(adapter)
        evidence["auth_probe"] = note
        if not authenticated:
            checks_failed.append(note)
            blockers.append("auth_unverified")
            return finish(
                TradeReadiness.CONFIGURED,
                f"{venue_id}: credentials configured but NOT verified - {note}",
                True,
            )
        checks.append(note)

        # -- rung 4: is there money to trade with? --
        min_order = float(getattr(caps, "min_order_usd", 1.0) or 0.0)
        details["min_order_usd"] = min_order
        evidence["balance_usd"] = balance
        evidence["balance_provenance"] = provenance

        if balance is None:
            blockers.append("balance_unreadable")
            checks_failed.append("Authenticated but no balance could be read")
            return finish(
                TradeReadiness.AUTHENTICATED,
                f"{venue_id}: authenticated, but balance unreadable",
                True,
            )

        details["balance_usd"] = balance
        if balance < min_order:
            blockers.append("insufficient_funds")
            checks_failed.append(
                f"Balance ${balance:.2f} < venue minimum order ${min_order:.2f}"
            )
            return finish(
                TradeReadiness.AUTHENTICATED,
                f"{venue_id}: authenticated but underfunded (${balance:.2f} < ${min_order:.2f})",
                True,
            )
        checks.append(f"Balance ${balance:.2f} (via {provenance}) >= min order ${min_order:.2f}")

        # -- rung 5: can an order be submitted? --
        permitted, probe_note = await self._probe_order_permission(adapter, opportunity)
        evidence["order_probe"] = probe_note
        if not permitted:
            checks_failed.append(probe_note)
            blockers.append("order_permission_unproven")
            return finish(
                TradeReadiness.FUNDED,
                f"{venue_id}: authenticated and funded (${balance:.2f}) but {probe_note}",
                True,
            )
        checks.append(probe_note)

        # -- geography still applies at the top of the ladder --
        try:
            eligibility = adapter.check_eligibility("UG")
            details["eligibility_ug"] = eligibility.value
            if eligibility == EligibilityStatus.REQUIRES_VERIFICATION:
                # "Requires verification" is not a soft preference - it means the
                # venue will not serve this account until a check has been
                # completed, and an authenticated, funded account with a working
                # order probe is NOT evidence that the check was done. Treating it
                # as one let REQUIRES_VERIFICATION slide into TRADE_PERMITTED with
                # nothing but the word "requires" distinguishing it from READY.
                verified, note = self._verification_evidence(venue_id, adapter)
                if not verified:
                    blockers.append("verification_unproven")
                    checks_failed.append(
                        f"Eligibility for UG is {eligibility.value}, and no "
                        f"completed verification is recorded")
                    return finish(
                        TradeReadiness.FUNDED,
                        f"{venue_id}: authenticated and funded (${balance:.2f}), "
                        f"but eligibility for UG is '{eligibility.value}' and "
                        f"there is no recorded evidence that verification was "
                        f"completed. Funded is not permitted.",
                        True,
                    )
                checks.append(f"Verification recorded for {venue_id}: {note}")
            if eligibility == EligibilityStatus.RESTRICTED:
                blockers.append("restricted_for_country")
                checks_failed.append("Restricted for UG - cannot trade live")
                return finish(
                    TradeReadiness.FUNDED,
                    f"{venue_id}: account is ready but the venue is restricted for UG",
                    True,
                )
            checks.append(f"Eligible for UG: {eligibility.value}")
        except Exception as e:
            details["eligibility_error"] = str(e)
            blockers.append("eligibility_unchecked")
            return finish(
                TradeReadiness.FUNDED,
                f"{venue_id}: ready but eligibility could not be confirmed ({e})",
                True,
            )

        if min_order > self.bankroll:
            blockers.append("min_order_exceeds_bankroll")
            checks_failed.append(
                f"Min order ${min_order:.2f} > bankroll ${self.bankroll:.2f}"
            )
            return finish(
                TradeReadiness.FUNDED,
                f"{venue_id}: ready but min order ${min_order:.2f} exceeds bankroll",
                True,
            )

        return finish(
            TradeReadiness.TRADE_PERMITTED,
            f"{venue_id}: READY - authenticated, funded ${balance:.2f}, "
            f"order submission verified, eligible for UG",
            True,
        )

    async def check_all_venues(self) -> Dict[str, AccountHealthResult]:
        """Check all venues."""
        results = {}
        if not self.venue_registry:
            return results

        for venue_id in self.venue_registry.adapters.keys():
            try:
                results[venue_id] = await self.check_venue_health(venue_id)
            except Exception as e:
                logger.error(f"Account health check failed for {venue_id}: {e}")
                results[venue_id] = AccountHealthResult(
                    venue_id=venue_id,
                    healthy=False,
                    paper_trading_ok=True,
                    reason=f"Health check error: {e}",
                    checks_failed=[str(e)],
                    readiness=TradeReadiness.NOT_CONFIGURED,
                    ready_to_trade=False,
                    blockers=["health_check_error"],
                )
        return results

    def get_report(self) -> Dict[str, Any]:
        by_readiness: Dict[str, List[str]] = {}
        for vid, h in self.health_cache.items():
            by_readiness.setdefault(h.readiness.value, []).append(vid)

        return {
            "engine": "AccountHealthEngine - V11q readiness ladder",
            "principle": (
                "A configured account is not a ready account. Each rung needs "
                "evidence: credentials present, then a real authenticated call, "
                "then a balance read from that call, then a proven order round "
                "trip. Nothing is assumed from a capability flag."
            ),
            "ladder": [r.value for r in _READINESS_ORDER],
            "checks_that_require_evidence": [
                "credentials_present (attribute inspection)",
                "authenticated (real get_portfolio call that reports provenance)",
                "funded (non-zero balance read from that call)",
                "trade_permitted (minimum-size order placed then cancelled)",
                "eligibility for the operating country",
                "min order vs bankroll",
            ],
            "checks_not_implemented": [
                "withdrawal/deposit capability",
                "rate limit discovery",
            ],
            "cached_venues": list(self.health_cache.keys()),
            "by_readiness": by_readiness,
            "ready_to_trade_venues": [
                vid for vid, h in self.health_cache.items() if h.ready_to_trade
            ],
            "paper_only_venues": [
                vid for vid, h in self.health_cache.items()
                if not h.ready_to_trade and h.paper_trading_ok
            ],
            "blockers_by_venue": {
                vid: h.blockers for vid, h in self.health_cache.items() if h.blockers
            },
        }


def record_verification(storage, venue_id: str, note: str = "") -> None:
    """
    Record that an account's verification with the venue was completed.

    This is an OPERATOR action, like the deposit: the identity check happens at
    the venue, in the operator's name, and the agent has no way to perform it or
    to observe it. Without a record, a venue whose eligibility is
    REQUIRES_VERIFICATION stops at FUNDED and cannot trade live.

    What this must NOT become is a box that gets ticked to make a warning go
    away. The note is kept and shown with the readiness evidence, so whatever is
    written here is what the operator is asserting happened.
    """
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    storage.set_state(f"verification.{venue_id}",
                      (note or "confirmed") + f" (recorded {stamp})")


def verification_status(storage, venue_id: str) -> str:
    """What is recorded, or an explicit 'not recorded'."""
    if storage is None:
        return "not recorded"
    try:
        return storage.get_state(f"verification.{venue_id}") or "not recorded"
    except Exception as e:
        logger.error(f"Could not read verification status for {venue_id}: {e}")
        return "not recorded"
