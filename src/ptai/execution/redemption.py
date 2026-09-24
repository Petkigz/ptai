"""
Redemption - turning a settled win into money.

A winning position is not profit until it is redeemed. The CTF holds the shares;
the collateral only comes back when `redeemPositions` is called on-chain. An
agent that settles a market in its own ledger, marks the P&L, and never redeems
has a bookkeeping win and an empty wallet - the collateral stays locked in the
contract and the "profit" is unusable.

That failure is invisible in every log line the agent currently writes. The
position looks settled, the P&L looks realised, and the balance never moves. So
this module reports redemption as three separate facts:

    redeemable  - what the venue says can be claimed, read from the Data API
    claimed     - what was actually submitted and confirmed on-chain
    unclaimed   - everything else, with the reason it was not claimed

Nothing here guesses. If the relayer is not configured, every position is
reported unclaimed with the reason, because a redemption nobody performed is not
a redemption.

Routing (three cases, from the venue's own builder docs):

  * standard market        -> CTF `redeemPositions(address,bytes32,bytes32,uint256[])`
  * neg-risk, pUSD output  -> NEG_RISK_COLLATERAL_ADAPTER, same four-arg shape
  * legacy neg-risk        -> the two-arg `redeemPositions(bytes32,uint256[])`

`indexSets=[1, 2]` covers both outcomes of a binary market; the losing side
redeems to zero, so redeeming both is safe and avoids having to know which
outcome we hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import requests
from loguru import logger

# Chain 137 (Polygon).
#
# pUSD, the V2 collateral token. Agrees between the V2 client's own
# get_contract_config(137) and the exchange-v2 deployment table.
COLLATERAL_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"

# The raw Conditional Tokens Framework contract.
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# On V2 the redemption target is a COLLATERAL ADAPTER, not the raw CTF: the
# adapter redeems through the CTF and wraps the result into pUSD for the caller.
# It exposes the same four-argument redeemPositions signature, so one calldata
# template covers standard and neg-risk markets.
#
# SOURCES DISAGREE, so both are overridable and neither is treated as certain:
#   docs.polymarket.com/resources/contracts
#   docs.polymarket.com/trading/positions/manage
#   docs.polymarket.com/changelog            -> 0xAdA100Db.. / 0xadA20056.. below
#   github.com/Polymarket/ctf-exchange-v2 readme -> 0xADa10087.. / 0xAdA20000..
# The venue's own documentation is used as the default because three separate
# doc pages agree with each other; the deployment table is recorded here so the
# difference is visible rather than lost. Verify on-chain before live money.
CTF_COLLATERAL_ADAPTER = "0xAdA100Db00Ca00073811820692005400218FcE1f"
NEG_RISK_COLLATERAL_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"
CTF_COLLATERAL_ADAPTER_ALT = "0xADa100874d00e3331D00F2007a9c336a65009718"
NEG_RISK_COLLATERAL_ADAPTER_ALT = "0xAdA200001000ef00D07553cEE7006808F895c6F1"

# Kept only for the legacy two-argument path: markets created before the
# migration. The venue retired this adapter for new integrations.
LEGACY_NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

DATA_API = "https://data-api.polymarket.com"
DATA_API_V2 = "https://data-api.polymarket.com/v2"
RELAYER_URL = "https://relayer-v2.polymarket.com"
# eth_abi encodes bytes, not hex strings: a "0x00.." literal is rejected by
# BytesEncoder, which is why this is raw bytes.
ZERO_BYTES32 = b"\x00" * 32

# `redeemPositions` on the CTF and the collateral adapters share this signature.
REDEEM_SELECTOR_ARGS = "redeemPositions(address,bytes32,bytes32,uint256[])"
# The legacy neg-risk adapter kept the original two-argument form.
LEGACY_REDEEM_SELECTOR_ARGS = "redeemPositions(bytes32,uint256[])"

# Both outcomes of a binary market. The losing side redeems to zero.
BINARY_INDEX_SETS = (1, 2)


@dataclass
class RedeemablePosition:
    """One claimable position, as the venue reports it."""

    condition_id: str
    asset: str
    outcome: str
    size: float
    current_value_usd: float
    neg_risk: bool = False
    title: str = ""
    slug: str = ""
    event_slug: str = ""
    redeemable: bool = True
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.condition_id}:{self.asset}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "asset": self.asset,
            "outcome": self.outcome,
            "size": self.size,
            "current_value_usd": round(self.current_value_usd, 6),
            "neg_risk": self.neg_risk,
            "title": self.title,
            "slug": self.slug,
            "event_slug": self.event_slug,
        }


@dataclass
class RedemptionReport:
    """What could be claimed, and what actually was."""

    available: bool = False
    reason: str = ""
    source: str = ""
    positions_seen: int = 0
    attempted: int = 0
    claimed: int = 0
    failed: int = 0
    skipped: int = 0
    claimable_value_usd: float = 0.0
    claimed_value_usd: float = 0.0
    items: List[Dict[str, Any]] = field(default_factory=list)
    unclaimed: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def unclaimed_value_usd(self) -> float:
        return round(max(0.0, self.claimable_value_usd - self.claimed_value_usd), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "source": self.source,
            "positions_seen": self.positions_seen,
            "attempted": self.attempted,
            "claimed": self.claimed,
            "failed": self.failed,
            "skipped": self.skipped,
            "claimable_value_usd": round(self.claimable_value_usd, 6),
            "claimed_value_usd": round(self.claimed_value_usd, 6),
            "unclaimed_value_usd": self.unclaimed_value_usd,
            "items": self.items,
            "unclaimed": self.unclaimed,
        }


# ----------------------------------------------------------------------
# calldata
# ----------------------------------------------------------------------

def redeem_calldata(condition_id: str, index_sets: Tuple[int, ...] = BINARY_INDEX_SETS,
                    collateral: str = COLLATERAL_ADDRESS,
                    legacy: bool = False) -> str:
    """
    ABI-encode a `redeemPositions` call.

    Two shapes are supported because the venue's own docs name both: the
    four-argument form used by the CTF and both collateral adapters, and the
    two-argument form kept by the legacy neg-risk adapter for markets created
    before the migration. The four-argument form takes the condition id as a
    bytes32; the legacy form takes it as the first of two bytes32 arguments.
    """
    from eth_abi import encode as abi_encode
    from eth_utils import keccak

    condition = _as_bytes32(condition_id)
    index_sets = [int(i) for i in index_sets]

    if legacy:
        selector = keccak(text=LEGACY_REDEEM_SELECTOR_ARGS)[:4]
        encoded = abi_encode(["bytes32", "uint256[]"], [condition, index_sets])
    else:
        selector = keccak(text=REDEEM_SELECTOR_ARGS)[:4]
        encoded = abi_encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [collateral, ZERO_BYTES32, condition, index_sets],
        )
    return "0x" + (selector + encoded).hex()


def _as_bytes32(value: str) -> bytes:
    """A condition id as 32 bytes, from either a hex string or a bare hash."""
    if not value:
        raise ValueError("condition id is required to build redeem calldata")
    text = str(value).strip()
    if text.startswith("0x"):
        text = text[2:]
    raw = bytes.fromhex(text)
    if len(raw) == 32:
        return raw
    if len(raw) < 32:
        return raw.rjust(32, b"\x00")
    raise ValueError(f"condition id is {len(raw)} bytes, expected at most 32")


# ----------------------------------------------------------------------
# the redeemer
# ----------------------------------------------------------------------

class Redeemer:
    """
    Read claimable positions from the venue and claim them.

    The venue's `redeemable` flag is the authority on WHAT can be claimed: the
    agent's own ledger knows a position won, but only the venue knows whether
    the collateral is still in the contract. The relayer is the authority on
    WHETHER a claim happened: a submitted transaction is not a redeemed one
    until the venue reports it CONFIRMED, and only `claimed` counts.

    Claiming is idempotent at the venue - redeeming an already-redeemed
    condition reverts to nothing and returns no collateral - so a failed claim
    is retried next cycle rather than being recorded as done.
    """

    def __init__(self, funder: Optional[str] = None, private_key: Optional[str] = None,
                 relayer_url: str = RELAYER_URL, chain_id: int = 137,
                 builder_key: Optional[str] = None, builder_secret: Optional[str] = None,
                 builder_passphrase: Optional[str] = None,
                 session: Optional[requests.Session] = None, dry_run: bool = True,
                 standard_adapter: Optional[str] = None,
                 neg_risk_adapter: Optional[str] = None):
        # Overridable: the venues's own docs and its contracts repo disagree on
        # these two addresses, so an operator who has verified them on-chain must
        # be able to set them without editing this file.
        self.standard_adapter = standard_adapter or CTF_COLLATERAL_ADAPTER
        self.neg_risk_adapter = neg_risk_adapter or NEG_RISK_COLLATERAL_ADAPTER
        self.funder = funder
        self.private_key = private_key
        self.relayer_url = relayer_url
        self.chain_id = chain_id
        self.builder_key = builder_key
        self.builder_secret = builder_secret
        self.builder_passphrase = builder_passphrase
        self.session = session or requests.Session()
        self.dry_run = dry_run
        self._relayer = None
        self._relayer_error: Optional[str] = None
        # Conditions already attempted this run, so a transient failure is not
        # retried in a tight loop inside one cycle.
        self._attempted: set = set()

    # -- capability ----------------------------------------------------

    @property
    def can_redeem(self) -> bool:
        """
        Is there a signer? Without one, no claim can be submitted, and saying
        otherwise would report a redemption that never happened.
        """
        return bool(self.private_key and self.funder)

    def _get_relayer(self):
        if self._relayer is not None or self._relayer_error is not None:
            return self._relayer
        if not self.can_redeem:
            self._relayer_error = ("no private key/funder: redemption needs a signer "
                                   "to submit the claim")
            return None
        try:
            from py_builder_relayer_client.client import RelayClient
            from py_builder_signing_sdk.config import BuilderConfig
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds

            builder_config = None
            if self.builder_key and self.builder_secret and self.builder_passphrase:
                builder_config = BuilderConfig(local_builder_creds=BuilderApiKeyCreds(
                    key=self.builder_key, secret=self.builder_secret,
                    passphrase=self.builder_passphrase))
            self._relayer = RelayClient(
                relayer_url=self.relayer_url, chain_id=self.chain_id,
                private_key=self.private_key, builder_config=builder_config)
        except Exception as e:
            self._relayer_error = f"relayer client unavailable: {type(e).__name__}: {e}"
            logger.error(f"Redemption disabled: {self._relayer_error}")
        return self._relayer

    # -- reading what can be claimed ----------------------------------

    def read_redeemable(self, user: Optional[str] = None,
                        limit: int = 500) -> Tuple[List[RedeemablePosition], bool, str]:
        """
        Positions the venue marks redeemable, for our wallet.

        Returns (positions, available, reason). `sizeThreshold=0` includes dust,
        because a dust position left unredeemed is a small loss repeated forever.
        An unreachable or erroring API returns available=False - the caller must
        be able to tell "nothing to claim" from "could not find out".
        """
        user = user or self.funder
        if not user:
            return [], False, "no funder address, so no position list can be requested"

        last_error = ""
        for base in (DATA_API_V2, DATA_API):
            for attempt in range(3):
                try:
                    response = self.session.get(
                        f"{base}/positions",
                        params={"user": user, "redeemable": "true",
                                "sizeThreshold": 0, "limit": min(500, limit)},
                        timeout=15,
                    )
                    if response.status_code == 429:
                        # The venue rate-limits with 429/1015; back off rather
                        # than hammering it into a longer ban.
                        wait = 1.5 * (attempt + 1)
                        logger.warning(f"positions API rate limited; backing off {wait}s")
                        time.sleep(wait)
                        continue
                    if response.status_code >= 400:
                        last_error = f"HTTP {response.status_code} from {base}"
                        break
                    payload = response.json()
                    rows = payload.get("data") if isinstance(payload, dict) else payload
                    if rows is None:
                        rows = []
                    if not isinstance(rows, list):
                        last_error = f"unexpected payload shape from {base}: {type(rows).__name__}"
                        break
                    positions = [p for p in (self._parse_position(r) for r in rows) if p]
                    logger.info(f"Redeemable positions: {len(positions)} from {base}")
                    return positions, True, ""
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    time.sleep(0.5 * (attempt + 1))
        logger.warning(f"Could not read redeemable positions: {last_error}")
        return [], False, last_error

    @staticmethod
    def _parse_position(row: Dict[str, Any]) -> Optional[RedeemablePosition]:
        if not isinstance(row, dict):
            return None
        condition_id = (row.get("conditionId") or row.get("condition_id")
                        or row.get("market"))
        asset = row.get("asset") or row.get("tokenId") or row.get("token_id") or ""
        if not condition_id:
            return None
        size = _first_float(row, "size", "shares", "quantity")
        value = _first_float(row, "currentValue", "current_value_usd", "value_usd")
        return RedeemablePosition(
            condition_id=str(condition_id),
            asset=str(asset),
            outcome=str(row.get("outcome") or ""),
            size=size,
            current_value_usd=value,
            neg_risk=bool(row.get("negativeRisk", row.get("neg_risk", False))),
            title=str(row.get("title") or ""),
            slug=str(row.get("slug") or ""),
            event_slug=str(row.get("eventSlug") or row.get("event_slug") or ""),
            redeemable=bool(row.get("redeemable", True)),
            raw=row,
        )

    # -- claiming ------------------------------------------------------

    def redeem(self, positions: List[RedeemablePosition],
               dry_run: Optional[bool] = None) -> RedemptionReport:
        """
        Claim every redeemable position, and report exactly what was claimed.

        A position is counted as claimed only when the relayer reports the
        transaction CONFIRMED. Submitted-but-unconfirmed, failed, and skipped
        claims all appear in `unclaimed` with their reason, so the agent can
        never present locked collateral as redeemed profit.
        """
        dry_run = self.dry_run if dry_run is None else dry_run
        report = RedemptionReport(source="data_api")
        report.positions_seen = len(positions)
        report.claimable_value_usd = round(
            sum(p.current_value_usd for p in positions), 6)

        if not positions:
            report.available = True
            report.reason = "no redeemable positions"
            return report

        unclaimable = [p for p in positions if not self._claimable(p)]
        for p in unclaimable:
            reason = self._unclaimable_reason(p)
            report.skipped += 1
            report.items.append({**p.to_dict(), "outcome": "skipped", "reason": reason})
            report.unclaimed.append({**p.to_dict(), "reason": reason})

        if dry_run:
            for p in positions:
                if self._claimable(p):
                    report.items.append({**p.to_dict(), "outcome": "dry_run",
                                         "reason": "dry run: no claim submitted"})
                    report.unclaimed.append({**p.to_dict(),
                                             "reason": "dry run: no claim submitted"})
            report.available = True
            report.reason = (f"dry run: {len(positions)} position(s) would be claimed "
                             f"for ${report.claimable_value_usd:.4f}")
            if self.can_redeem:
                logger.info(report.reason)
            else:
                logger.warning(f"{report.reason} - and no signer is configured, so "
                               f"they could not be claimed even without dry run")
            return report

        relayer = self._get_relayer()
        if relayer is None:
            report.available = False
            report.reason = self._relayer_error or "relayer unavailable"
            for p in positions:
                report.unclaimed.append({**p.to_dict(), "reason": report.reason})
            logger.error(f"Redemption could not run: {report.reason}. "
                         f"${report.claimable_value_usd:.4f} stays locked in the contract.")
            return report

        report.available = True
        for position in positions:
            if not self._claimable(position):
                continue
            if position.condition_id in self._attempted:
                report.skipped += 1
                report.unclaimed.append({**position.to_dict(),
                                         "reason": "already attempted this run"})
                continue
            self._attempted.add(position.condition_id)
            report.attempted += 1
            outcome, reason, tx_id = self._submit(relayer, position)
            item = {**position.to_dict(), "outcome": outcome, "reason": reason,
                    "transaction_id": tx_id}
            report.items.append(item)
            if outcome == "claimed":
                report.claimed += 1
                report.claimed_value_usd += position.current_value_usd
            else:
                report.failed += 1
                report.unclaimed.append({**position.to_dict(), "reason": reason})

        report.claimed_value_usd = round(report.claimed_value_usd, 6)
        logger.info(
            f"Redemption: {report.claimed}/{report.attempted} claimed "
            f"(${report.claimed_value_usd:.4f}); ${report.unclaimed_value_usd:.4f} "
            f"left unclaimed; {report.skipped} skipped")
        if report.unclaimed_value_usd > 0 and report.claimable_value_usd > 0:
            logger.warning(
                f"${report.unclaimed_value_usd:.4f} of settled winnings is still in "
                f"the contract and cannot be traded with")
        return report

    @staticmethod
    def _claimable(position: RedeemablePosition) -> bool:
        return bool(position.condition_id) and position.redeemable

    @staticmethod
    def _unclaimable_reason(position: RedeemablePosition) -> str:
        if not position.condition_id:
            return "no condition id: a claim cannot be built without one"
        return "the venue does not mark this position redeemable"

    def _submit(self, relayer, position: RedeemablePosition
                ) -> Tuple[str, str, str]:
        """Submit one claim and wait for the venue to confirm it."""
        try:
            from py_builder_relayer_client.models import (
                RelayerTransactionState,
                Transaction,
            )

            legacy = bool(position.raw.get("legacyNegRisk")
                          or position.raw.get("legacy_neg_risk"))
            if legacy:
                # The pre-migration neg-risk adapter keeps the two-argument
                # signature. Its address is not overridable because there was
                # only ever one of it.
                target = LEGACY_NEG_RISK_ADAPTER
            else:
                target = (self.neg_risk_adapter if position.neg_risk
                          else self.standard_adapter)
            data = redeem_calldata(position.condition_id,
                                   collateral=COLLATERAL_ADDRESS, legacy=legacy)
            tx = Transaction(to=target, data=data, value="0")

            response = relayer.execute([tx], "Redeem positions")
            transaction_id = ""
            for attr in ("transaction_id", "transactionID", "id"):
                value = getattr(response, attr, None)
                if value:
                    transaction_id = str(value)
                    break
            if not transaction_id and isinstance(response, dict):
                transaction_id = str(response.get("transactionID")
                                     or response.get("transaction_id")
                                     or response.get("id") or "")
            if not transaction_id:
                return "failed", f"relayer accepted no transaction id: {response}", ""

            confirmed = relayer.poll_until_state(
                transaction_id,
                states=[RelayerTransactionState.STATE_CONFIRMED.value,
                        RelayerTransactionState.STATE_MINED.value],
                fail_state=RelayerTransactionState.STATE_FAILED.value,
                max_polls=30, poll_frequency=2,
            )
            if confirmed is None:
                return ("failed",
                        f"claim {transaction_id} did not confirm in time; it may "
                        f"still land, and will be re-read next cycle", transaction_id)
            return "claimed", f"confirmed on-chain via {transaction_id}", transaction_id
        except Exception as e:
            logger.error(f"Redemption of {position.condition_id} raised "
                         f"{type(e).__name__}: {e}")
            return "failed", f"{type(e).__name__}: {e}", ""

    # -- recording -----------------------------------------------------

    def record(self, storage, report: RedemptionReport,
               positions: Optional[List[RedeemablePosition]] = None) -> None:
        """
        Record the claim against the position it closes.

        Without this a redeemed win looks like an unclaimed one, and the agent
        re-attempts a claim the contract has already paid out - or worse, counts
        the winnings as still locked and under-reports its own capital.
        """
        if storage is None or not report.claimed:
            return
        by_condition = {p.condition_id: p for p in (positions or [])}
        for item in report.items:
            if item.get("outcome") != "claimed":
                continue
            position = by_condition.get(item.get("condition_id"))
            value = float(item.get("current_value_usd") or 0.0)
            try:
                storage.conn.execute(
                    "INSERT INTO redemptions (condition_id, asset, outcome, size, "
                    "value_usd, transaction_id, redeemed_at) VALUES (?,?,?,?,?,?,?)",
                    (item.get("condition_id"), item.get("asset"),
                     item.get("outcome"), position.size if position else None,
                     value, item.get("transaction_id"),
                     datetime.now(timezone.utc).isoformat()))
                storage.conn.commit()
            except Exception as e:
                # Never silent: an unrecorded redemption is a claim that will be
                # re-attempted forever.
                logger.error(f"Could not record redemption of "
                             f"{item.get('condition_id')}: {type(e).__name__}: {e}")

    def already_redeemed(self, storage, condition_id: str) -> bool:
        try:
            row = storage.conn.execute(
                "SELECT 1 FROM redemptions WHERE condition_id = ? LIMIT 1",
                (condition_id,)).fetchone()
            return row is not None
        except Exception as e:
            logger.warning(f"Could not check redemption history: {e}")
            return False


def _first_float(row: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0
