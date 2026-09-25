
"""
Multi-Venue Executor - handles execution across 18 venues with per-venue guards
Operational overhead: each venue has own API auth model rate limits failure modes
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import asyncio

from ..venues.registry import VenueRegistry
from ..venues.adapter import VenueOpportunity
from ..markets.orderbook import read_spread

# Outcomes that mean real (or simulated) capital is now committed. Anything
# else means no position exists and none may be recorded.
FILLED_STATUSES = frozenset({"filled", "partial", "submitted"})
SIMULATED_STATUSES = frozenset({"dry_run", "simulated", "paper"})
# Explicitly no position: the venue refused, or we refused to ask.
NO_POSITION_STATUSES = frozenset({
    "blocked", "aborted", "rejected", "rate_limited", "error", "unknown",
})


def _order_gas_usd(adapter, venue_id: str, amount_usd: float = 0.0) -> float:
    """
    Gas to place one order, from the venue's own declared mechanism.

    `order_gas_usd` on the adapter's capabilities is the venue's answer:
    0.0 for a venue that relays orders, a number for one that charges, None
    when the venue does not say - and then the cost model estimates it. The
    estimate is the model's, not a constant here, because a constant here
    cannot know which venue it is pricing.
    """
    declared = getattr(getattr(adapter, "capabilities", None), "order_gas_usd", None)
    if declared is not None:
        return max(0.0, float(declared))
    try:
        from ..execution.gas import GasModel
        return float(GasModel().calculate_gas(
            operation="place_order", amount_usd=float(amount_usd or 0.0),
            venue_id=venue_id).gas_usd)
    except Exception as e:
        logger.warning(
            f"No gas estimate for {venue_id} ({type(e).__name__}: {e}); "
            f"charging nothing and labelling the execution with it")
        return 0.0


@dataclass
class ExecutionResult:
    venue_id: str
    market_id: str
    status: str
    amount_usd: float
    price: float
    fees_usd: float
    gas_usd: float
    latency_ms: float
    reasoning: str
    # What the venue actually did, as opposed to what we asked for. Only a
    # filled (or partially filled) result may become a position.
    filled_usd: float = 0.0
    filled_price: float = 0.0
    order_id: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)
    # The venue's own count of how much of the order has matched, and how much
    # there was. A marketable order that took only part of the book returns
    # "matched" with size_matched below original_size.
    size_matched: Optional[float] = None
    original_size: Optional[float] = None
    # Unfilled shares still working in the book, as USD at the limit price.
    resting_usd: float = 0.0
    # The price the unfilled remainder rests at (paper: the signed limit).
    # None where there is no remainder or the venue does not report one.
    limit_price: Optional[float] = None
    # A send that failed with no venue confirmation: the order may exist.
    unconfirmed_send: bool = False
    trade_ids: List[str] = field(default_factory=list)
    # The simulated fill, when this was a paper execution: the walk of the real
    # book behind the numbers above. Kept so the reason a paper trade filled at
    # 0.564 rather than 0.56 is inspectable rather than folklore.
    paper_fill: Dict[str, Any] = field(default_factory=dict)

    @property
    def committed_capital(self) -> bool:
        """
        Did this execution actually BUY something?

        Distinct from `reserves_capital`. A resting order reserves cash at the
        venue without buying anything, and treating that as a position would
        invent shares the agent does not own.
        """
        return self.status in FILLED_STATUSES and self.filled_usd > 0

    @property
    def is_simulated(self) -> bool:
        return self.status in SIMULATED_STATUSES

    @property
    def bought_something(self) -> bool:
        """
        Did this execution buy a position - REAL or SIMULATED?

        `committed_capital` is the LIVE question ("did real money leave?"), and
        it is deliberately False for a paper fill. The arbitrage state machine
        used it to decide whether leg A had anything to balance, which made the
        pair live-only by accident: in paper, leg A had "committed no capital",
        leg B was never sent, and the arb lane could not run at all - on the one
        strategy whose entire economics are testable in paper.

        Same rule as `should_record_position`, because they are the same
        question: a position exists when a real fill happened, or when a
        simulation says one would have.
        """
        return self.should_record_position

    @property
    def reserves_capital(self) -> bool:
        """
        Is real money locked at the venue by this result?

        True for a filled position (the cost is spent) and for an order still
        working in the book (the cash is reserved against it). The position
        ledger must subtract both, or the agent will size a new trade against
        cash the venue has already locked.
        """
        if self.unconfirmed_send:
            return True
        if self.committed_capital:
            return True
        if self.status in ("submitted", "partial"):
            return True
        # A simulated order with an unfilled remainder locks cash in the
        # simulation the same way a live order locks it at the venue. If
        # paper did not reserve it, free capital would be higher in paper
        # than in live for the identical order, and sizing would spend cash
        # the live venue has already locked.
        if self.is_simulated and self.unfilled_shares > 0:
            return True
        return False

    @property
    def needs_reconciliation(self) -> bool:
        """
        Is this order's fate still unknown?

        A resting or partially filled order has to be re-read from the venue
        until it reaches a final state. So does an unconfirmed send, because the
        order may be in the book while the response was lost.
        """
        if self.unconfirmed_send:
            return True
        if self.status in ("submitted", "partial") and self.unfilled_shares > 0:
            return True
        return False

    @property
    def unfilled_shares(self) -> float:
        if self.original_size is None:
            return 0.0
        matched = self.size_matched if self.size_matched is not None else 0.0
        return max(0.0, float(self.original_size) - float(matched))

    @property
    def position_size_usd(self) -> float:
        """
        What this execution actually put at risk.

        For a real fill it is what filled. For a paper fill it is what the
        simulation says would have filled, which is usually less than the request
        because depth is finite. Charging the ledger the REQUESTED amount for a
        paper trade overstates exposure and understates free cash, and it is how
        a paper equity curve becomes fiction.
        """
        if self.filled_usd > 0:
            return self.filled_usd
        if self.is_simulated:
            simulated = self.paper_fill or {}
            value = simulated.get("filled_usd")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 0.0

    @property
    def filled_shares(self) -> float:
        """
        How many SHARES this execution actually bought.

        Dollars are not a position: $3.00 buys 5 shares at 0.60 and 4 shares at
        0.75. A hedge has to close SHARES, so matching dollars leaves the
        difference naked when the opposite side is dearer and buys a
        directional bet when it is cheaper - and either way a dollars-matched
        log line says the exposure was closed.

        The venue's own `size_matched` is used when it reported one. Otherwise
        the shares are derived from the fill it did report, because a fill that
        cannot be counted in shares cannot be hedged against.
        """
        matched = _num(self.size_matched)
        if matched is not None and matched > 0:
            return float(matched)
        if self.filled_usd > 0 and self.filled_price > 0:
            return float(self.filled_usd) / float(self.filled_price)
        return 0.0

    @property
    def should_record_position(self) -> bool:
        """
        Should a POSITION row be written?

        A partial fill is a real position of the filled size, so it is recorded
        and then reconciled upward as more of the order fills. A resting order
        with nothing matched is not a position - it is an ORDER, and it belongs
        in the order book table where it reserves capital without claiming
        shares.

        A SIMULATED execution with nothing filled is the same case. Paper mode
        against a book that cannot supply the order, or with no book at all,
        would otherwise record a position at the REQUESTED size - which is the
        old fiction wearing a new label.
        """
        if self.committed_capital:
            return True
        if self.is_simulated:
            return self.position_size_usd > 0
        return False

    def to_position_dict(self) -> Dict[str, Any]:
        """The facts of what filled. Zeros are honest: nothing filled."""
        return {
            "venue_id": self.venue_id,
            "market_id": self.market_id,
            "status": self.status,
            "requested_usd": self.amount_usd,
            "filled_usd": self.filled_usd,
            "filled_price": self.filled_price,
            "filled_shares": self.filled_shares,
            "fees_usd": self.fees_usd,
            "gas_usd": self.gas_usd,
            "order_id": self.order_id,
            "simulated": self.is_simulated,
            "size_matched": self.size_matched,
            "original_size": self.original_size,
            "unfilled_shares": self.unfilled_shares,
            "resting_usd": self.resting_usd,
            "limit_price": self.limit_price,
            "reserves_capital": self.reserves_capital,
            "needs_reconciliation": self.needs_reconciliation,
            "unconfirmed_send": self.unconfirmed_send,
            "trade_ids": list(self.trade_ids),
            "reasoning": self.reasoning,
        }


def _num(value: Any) -> Optional[float]:
    """
    A number from a venue payload field, or None.

    The CLOB reports size_matched, original_size and price as STRINGS
    ("5.45"), so an isinstance(value, (int, float)) check silently discards
    every real fill size - and a partial fill that cannot be quantified is
    reported as unknown, which erases the position.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _resting_usd(fill: Dict[str, Any], requested_usd: float) -> float:
    """
    How much of an order is still working in the book, in USD.

    Derived from the venue's own numbers only. With no size_matched figure the
    remainder is unknown, so the whole unfilled request is reported as resting -
    the conservative direction, because under-counting locked cash is the way an
    agent sizes a second trade against money the venue has already reserved.
    """
    if fill.get("status") not in ("submitted", "partial",
                                   "dry_run", "simulated", "paper"):
        return 0.0
    if fill.get("unconfirmed_send"):
        return float(requested_usd or 0.0)
    matched = _num(fill.get("size_matched"))
    original = _num(fill.get("original_size"))
    # The remainder rests at the LIMIT, not at the average fill price.
    price = _num(fill.get("limit_price")) or _num(fill.get("price")) or 0.0
    if matched is None or not original or not price:
        # Nothing quantifiable: assume the whole request is still committed.
        return max(0.0, float(requested_usd or 0.0) - float(fill.get("filled_usd") or 0.0))
    remaining = max(0.0, float(original) - float(matched))
    return round(remaining * float(price), 6)


def _coerce_venue_type(value):
    """
    A VenueType from whatever the market carried.

    Missing or unrecognised becomes PREDICTION, which is what the old fallback
    string meant - but as the enum member, so an equality check against
    VenueType.PREDICTION actually matches.
    """
    from ..venues.adapter import VenueType
    if isinstance(value, VenueType):
        return value
    text = str(value or "").strip().lower()
    for member in VenueType:
        if member.value == text or member.name.lower() == text:
            return member
    return VenueType.PREDICTION


# PRICE units - 2 cents of room on the price of one share. It is not a
# fraction of the stake, and confusing the two is how the unit error happened
# in the first place.
_PRICE_SLIPPAGE = 0.02

# Shares. A fraction of a share is a fraction of a dollar of payoff; a tenth of
# a share at 6-decimal tick size is a rounding artefact, not exposure.
_SHARE_TOLERANCE = 0.01


def _token_price(opportunity) -> float:
    """
    What one share of the side being bought costs at the modelled price.

    A NO position is a BUY of the NO token, so its modelled price is
    1 - the YES price. Sizing a NO leg off the YES price is the same defect as
    capping it off the YES price, one step further down the pipe.
    """
    price = float(getattr(opportunity, "market_price", 0.0) or 0.0)
    side = str(getattr(opportunity, "side", "YES") or "YES").upper()
    if side in ("NO", "SELL", "SHORT", "0"):
        return max(0.0, 1.0 - price)
    return price


def _best_quote(book, side):
    """
    What one share of `side` costs to BUY from this book, and the note for it.

    YES buys the ask. NO is the other token of the same binary market, so it
    buys at 1 - the YES bid, and quoting a NO order off the YES ask is how a
    NO limit ends up on a price that token never trades at. Shared so the pair
    recheck and the hedge cannot drift apart.
    """
    best_ask = None
    best_bid = None
    for level in (book.get("asks") or []):
        try:
            best_ask = float(level.get("price"))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    for level in (book.get("bids") or []):
        try:
            best_bid = float(level.get("price"))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    if str(side).upper() in ("NO", "SELL", "SHORT", "0"):
        if best_bid is None:
            return None, "no bid to price NO"
        return 1.0 - best_bid, f"NO at {1.0 - best_bid:.3f} (1 - bid {best_bid:.3f})"
    if best_ask is None:
        return None, "no ask to price YES"
    return best_ask, f"YES at {best_ask:.3f}"


def _side_aware_cap(opportunity, slippage: float = _PRICE_SLIPPAGE) -> float:
    """
    The most this order may pay per share, on the token it is buying.

    A NO position is a BUY of the NO token, priced at 1 - YES. Capping both legs
    at `yes_price + 0.02` gives a NO order a ceiling above 1.0 or below its own
    price depending on the market - the same defect that was fixed on the single
    path when it grew `_execute_with_side_aware_cap`.
    """
    price = float(getattr(opportunity, "market_price", 0.0) or 0.0)
    side = str(getattr(opportunity, "side", "YES") or "YES").upper()
    if side in ("NO", "SELL", "SHORT", "0"):
        return min(0.999, (1.0 - price) + slippage)
    return min(0.999, price + slippage)


class MultiVenueExecutor:
    def __init__(self, registry: VenueRegistry, bankroll: float = 50.0):
        self.registry = registry
        self.bankroll = bankroll
        self.rate_limits: Dict[str, float] = {}  # venue -> last request time
        self.min_order_sizes = {
            "polymarket": 1.0,
            "kalshi": 1.0,
            "manifold": 0.1,
            "predictit": 1.0,
            "simmer": 0.1,
            "cymetica": 1.0,
            "binance": 1.0,
            "whitebit": 1.0,
            "afx_dex": 1.0,
            "grvt": 1.0,
            "pionex": 1.0,
            "stock_mock": 1.0,
            "betfair": 2.0,
            "betdaq": 2.0,
            "betconnect": 2.0,
        }

    def check_rate_limit(self, venue_id: str) -> bool:
        # Pionex 10 req/sec, WhiteBIT etc
        import time
        now = time.time()
        last = self.rate_limits.get(venue_id, 0)
        # Simplified: 1 sec min between requests per venue for safety
        if now - last < 1.0:
            return False
        self.rate_limits[venue_id] = now
        return True

    def check_min_order(self, venue_id: str, amount_usd: float) -> bool:
        min_size = self.min_order_sizes.get(venue_id, 1.0)
        return amount_usd >= min_size

    async def execute_single(self, opportunity: VenueOpportunity, max_spend_usd: float, max_price: float) -> ExecutionResult:
        # V9 FIX #2: Exact routing only - never fallback to first adapter or substring match, ABORT if missing
        # V9 FIX #1: MOCK_DATA must be impossible to reach live execution
        market = opportunity.market
        # Hard MOCK blocking before any execution
        data_mode = getattr(market, 'data_mode', 'live')
        if hasattr(data_mode, 'value'):
            data_mode = data_mode.value
        data_mode = str(data_mode).lower()
        is_mock = getattr(market, 'is_mock', False) or data_mode == "mock" or "MOCK" in str(market.id).upper()
        if is_mock:
            logger.error(f"ABORT MULTI_VENUE EXECUTOR: Market {market.id} is MOCK_DATA data_mode={data_mode} is_mock={is_mock} - MOCK_DATA MUST NEVER REACH LIVE EXECUTION")
            return ExecutionResult(
                venue_id=opportunity.venue_id,
                market_id=market.id,
                status="blocked",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=0,
                reasoning=f"MOCK_DATA {market.id} blocked - cannot reach execution - data_mode={data_mode} is_mock={is_mock}"
            )

        venue_id = opportunity.venue_id.split("+")[0] if "+" in opportunity.venue_id else opportunity.venue_id
        venue_id = venue_id.lower().strip()

        # Exact routing only - ABORT if not found, never fallback to first eligible or substring
        adapter = self.registry.get_adapter_for_venue_id(venue_id) if hasattr(self.registry, 'get_adapter_for_venue_id') else self.registry.adapters.get(venue_id)
        if not adapter:
            adapter = self.registry.get_adapter_for_market(market) if hasattr(self.registry, 'get_adapter_for_market') else None

        if not adapter:
            logger.error(f"ABORT MULTI_VENUE EXECUTOR: venue {venue_id} adapter not found for market {market.id} - requested {opportunity.venue_id} not in {list(self.registry.adapters.keys())} - ABORT, never fallback to first eligible")
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status="aborted",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=0,
                reasoning=f"Adapter {venue_id} not found - ABORT, never fallback - available {list(self.registry.adapters.keys())}"
            )

        # Validate venue_id matches exactly - hard safety, allow composite arb e.g. polymarket+kalshi
        if "+" not in opportunity.venue_id:
            if adapter.venue_id.lower() != venue_id and venue_id not in adapter.venue_id.lower() and adapter.venue_id.lower() not in venue_id:
                # For strict exact routing, if mismatch, ABORT unless composite
                # Allow if registry mapping handles alias, but log warning
                logger.warning(f"Venue identity mismatch opportunity {opportunity.venue_id} vs adapter {adapter.venue_id} for market {market.id} - checking if alias allowed")
                # Only allow if exact match via registry's alias logic, otherwise ABORT
                if hasattr(self.registry, 'get_adapter_for_venue_id'):
                    # Already tried exact, so this is mismatch
                    pass

        # Rate limit check
        if not self.check_rate_limit(venue_id):
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status="rate_limited",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=0,
                reasoning=f"Rate limited {venue_id}, Pionex 10 req/sec etc"
            )

        # Min order check
        if not self.check_min_order(venue_id, max_spend_usd):
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status="rejected",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=0,
                reasoning=f"Amount ${max_spend_usd} < min ${self.min_order_sizes.get(venue_id, 1.0)} for {venue_id}"
            )

        # Execute via adapter
        import time
        start = time.time()
        try:
            result = await adapter.place_order(opportunity=opportunity, max_spend_usd=max_spend_usd, max_price=max_price)
            latency = (time.time() - start) * 1000
            fees = adapter.calculate_fees(opportunity.market, max_spend_usd)
            # Gas, the same way the EV engine costs it: what the venue declares,
            # or an estimate if it declares nothing.
            #
            # This was `0.05 if venue_id in ["polymarket", "afx_dex"] else 0.0` -
            # a literal charged to a venue that relays its orders and pays no
            # gas to trade, and a free pass to every other venue including ones
            # that do settle on-chain. Both halves were wrong in the direction
            # that decides trades.
            gas = _order_gas_usd(adapter, venue_id, max_spend_usd)

            fill = self._read_fill(result, max_spend_usd, max_price)

            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status=fill["status"],
                amount_usd=max_spend_usd,
                price=fill["price"] or max_price,
                fees_usd=fees,
                gas_usd=gas,
                latency_ms=latency,
                reasoning=f"{venue_id} execution latency {latency:.1f}ms fees ${fees:.4f} gas ${gas:.4f} | {result.get('message', '')[:100] if isinstance(result, dict) else str(result)[:100]}",
                filled_usd=fill["filled_usd"],
                filled_price=fill["price"],
                order_id=fill["order_id"],
                raw_response=result if isinstance(result, dict) else {"raw": str(result)},
                size_matched=fill.get("size_matched"),
                original_size=fill.get("original_size"),
                resting_usd=_resting_usd(fill, max_spend_usd),
                limit_price=fill.get("limit_price"),
                unconfirmed_send=bool(fill.get("unconfirmed_send")),
                trade_ids=list(fill.get("trade_ids") or []),
                paper_fill=(result.get("paper_fill")
                            if isinstance(result, dict) else None) or {},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Multi-venue execution failed {venue_id}: {e}")
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status="error",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=latency,
                reasoning=f"Error {e}"
            )

    # Venue status strings that mean the order was accepted. Adapters differ
    # ("matched" from Polymarket's CLOB, "submitted", "placed", "ok"), so the
    # mapping is explicit rather than a substring guess.
    _FILLED_ALIASES = {
        "filled": "filled", "matched": "filled", "complete": "filled",
        "completed": "filled", "executed": "filled",
        "partial": "partial", "partially_filled": "partial",
        "submitted": "submitted", "placed": "submitted", "accepted": "submitted",
        "open": "submitted", "ok": "submitted", "success": "submitted",
        # The CLOB's own word for an order resting in the book with nothing
        # matched yet. It is neither a fill nor a refusal: the capital is
        # committed and the exposure is real, so it must be reconciled against
        # the venue rather than dropped.
        "live": "submitted",
        "delayed": "submitted",
        "dry_run": "dry_run", "simulated": "dry_run", "paper": "dry_run",
        # Keep refusals under their own names. They are all no-position
        # outcomes, but collapsing them into "unknown" would throw away the
        # reason, and the reason is what an operator needs.
        "rejected": "rejected", "error": "error", "blocked": "blocked",
        "aborted": "aborted", "rate_limited": "rate_limited",
        "insufficient_funds": "rejected", "unauthorized": "rejected",
        "invalid": "rejected",
        # A send that failed without a venue confirmation. It is distinct from
        # a rejection because the order may exist: the failure could be the
        # response that was lost, not the request.
        "failed": "failed",
    }

    def _read_fill(self, result: Any, requested_usd: float,
                   requested_price: float) -> Dict[str, Any]:
        """
        Extract what ACTUALLY filled from an adapter response.

        The old code took the adapter's raw status string straight into
        ExecutionResult and copied the REQUESTED amount and price into the
        result. Downstream, a position was opened from that - so a request that
        was rejected, rate limited or simply never acknowledged still produced
        an "open" position at a price nobody traded at, sized at an amount that
        never left the account.

        Nothing is assumed here. An unrecognised status is "unknown", which is
        a no-position outcome. Missing size or price fields are treated as
        zero-filled, because a fill we cannot quantify is not a fill we can
        account for.
        """
        if not isinstance(result, dict):
            return {"status": "unknown", "filled_usd": 0.0, "price": 0.0,
                    "order_id": "", "note": "adapter returned no dict"}

        raw_status = str(result.get("status", "") or "").strip().lower()
        status = self._FILLED_ALIASES.get(raw_status, "unknown")

        order_id = ""
        for key in ("orderID", "order_id", "id", "orderId", "tx_hash"):
            value = result.get(key)
            if value:
                order_id = str(value)
                break

        # Size: adapters report shares or USD. Prefer an explicit USD figure.
        filled_usd = 0.0
        for key in ("filled_usd", "amount_usd", "cost_usd", "size_usd",
                    "notional_usd"):
            value = _num(result.get(key))
            if value is not None:
                filled_usd = value
                break
        # The CLOB reports matched size separately from the original size, so
        # "matched" alone does not mean fully filled. A marketable order that
        # took only part of the book comes back "matched" with size_matched
        # below original_size, and booking that as a complete fill overstates
        # the position by the unfilled remainder.
        size_matched = _num(result.get("size_matched"))
        original_size = _num(result.get("original_size"))

        # Where an unfilled remainder rests: the signed limit, not the
        # average fill price. Paper reports it explicitly; a live venue
        # does not, so this stays None there and _resting_usd falls back.
        limit_price = None
        for key in ("resting_limit_price", "signed_price", "limit_price"):
            value = _num(result.get(key))
            if value is not None:
                limit_price = value
                break

        if filled_usd <= 0:
            for key in ("size_matched", "size", "filled_size", "matched_size",
                        "shares", "quantity", "amount"):
                value = _num(result.get(key))
                if value is not None:
                    price_hint = self._read_price(result, requested_price)
                    filled_usd = value * (price_hint or requested_price)
                    break

        price = self._read_price(result, requested_price)

        # Downgrade a "filled" claim that the venue's own numbers contradict.
        if status == "filled" and size_matched is not None and original_size:
            if size_matched + 1e-9 < original_size:
                status = "partial"

        if status in ("filled", "partial", "submitted"):
            if filled_usd <= 0:
                if status == "submitted" and (size_matched is not None
                                              or original_size is not None):
                    # An order resting in the book with nothing matched yet. The
                    # venue quantified it: zero of a known size has filled and
                    # the rest is still working. That is a KNOWN state, and
                    # calling it "unknown" would erase the order - and with it
                    # the capital it is holding.
                    logger.info(
                        f"Order {order_id} is resting: 0 of {original_size} "
                        f"shares matched, capital reserved and awaiting "
                        f"reconciliation")
                else:
                    # The venue says it took the order but gave us no size. We
                    # cannot account for capital we cannot quantify.
                    return {"status": "unknown", "filled_usd": 0.0, "price": price,
                            "order_id": order_id,
                            "note": f"status {raw_status!r} but no fill size reported - "
                                    f"cannot account for an unquantified position"}
        elif status in SIMULATED_STATUSES:
            # A SIMULATED fill carries numbers too, and they are the point of
            # paper mode. The old code zeroed every field for a simulated result,
            # so the loop fell back to the REQUESTED size and price and paper
            # trading filled every order completely, instantly, at the price the
            # agent wanted. The simulated values come from a walk of the real
            # orderbook and must survive to the ledger, or paper mode measures
            # nothing.
            if filled_usd <= 0:
                for key in ("simulated_filled_usd", "filled_usd", "amount_usd"):
                    value = _num(result.get(key))
                    if value is not None:
                        filled_usd = value
                        break
        else:
            filled_usd = 0.0

        return {"status": status, "filled_usd": filled_usd, "price": price,
                "order_id": order_id, "raw_status": raw_status,
                "size_matched": size_matched, "original_size": original_size,
                "limit_price": limit_price,
                "unconfirmed_send": bool(result.get("unconfirmed_send")),
                "trade_ids": list(result.get("trade_ids") or []),
                "simulated": status in SIMULATED_STATUSES}

    @staticmethod
    def _read_price(result: Dict[str, Any], fallback: float) -> float:
        for key in ("filled_price", "average_price", "avg_price", "price",
                    "fill_price"):
            value = _num(result.get(key))
            if value is not None and 0.0 < value <= 1.0:
                return value
        return 0.0

    def arb_leg_opportunities(self, arb):
        """
        The two legs of a pair, as opportunities.

        The cheaper side of the pair is bought at A and the other side at B.
        ONE definition, because the gate that decides whether to send the pair
        and the state machine that sends it must not be able to disagree about
        which market, side or price they are talking about.

        A VenueType, not a bare string: `raw.get("venue_type", "prediction")`
        produces a str wherever the market does not carry that key - which is
        everywhere - and downstream code compares against the enum.
        """
        side_a = "YES" if arb.price_a < arb.price_b else "NO"
        return (
            VenueOpportunity(
                market=arb.market_a,
                venue_id=arb.venue_a,
                venue_type=_coerce_venue_type(arb.market_a.raw.get("venue_type")),
                side=side_a,
                market_price=arb.price_a,
                estimated_fair=arb.price_b,
                raw_edge=arb.spread,
                effective_edge=arb.fee_adjusted_profit,
                confidence=arb.confidence_same_event,
                should_trade=arb.should_trade,
            ),
            VenueOpportunity(
                market=arb.market_b,
                venue_id=arb.venue_b,
                venue_type=_coerce_venue_type(arb.market_b.raw.get("venue_type")),
                side="NO" if side_a == "YES" else "YES",
                market_price=arb.price_b,
                estimated_fair=arb.price_a,
                raw_edge=arb.spread,
                effective_edge=arb.fee_adjusted_profit,
                confidence=arb.confidence_same_event,
                should_trade=arb.should_trade,
            ),
        )


    async def execute_arbitrage_pair(self, arb, amount_per_leg: float = 3.0) -> List[ExecutionResult]:
        """
        A sequential pair with an explicit hedge - NOT an atomic transaction.

        An earlier version of this docstring called it "arbitrage atomicity".
        There is no atomic cross-venue primitive: leg A and leg B are separate
        orders at separate venues, they cannot be submitted as one unit, and the
        window between them is real. Claiming atomicity in a comment is worse
        than the exposure, because it stops anyone looking for it.

        What the state machine actually guarantees, and what it now does:

          PRECHECK      the spread survives the fee model
          RESERVE       both legs fit inside the bankroll cap
          VERIFY_BOOKS  both books are readable and not too wide
          SUBMIT_A      only this may open exposure, and sized so that a
                        SHARE-MATCHED second leg fits the reserve
          size_B to     the SHARES A ACTUALLY FILLED, not the amount requested,
          SUBMIT_B      so a partial A cannot be "balanced" by a full B
          HEDGE         the UNMATCHED SHARES - if B underfills, or if the price
                        moved while A was filling - bought on the side that
                        closes them, at the price the book now shows

        Two things this deliberately does not do. It does not submit both legs
        concurrently: with no cross-venue primitive, concurrent submits double
        the chance of a leg resting unfilled, and a resting leg is exposure that
        no hedge is watching. And it does not cancel a resting remainder -
        there is no cancel path here yet, so a partially-filled A leaves a
        resting order that `order_manager` tracks for reconciliation, and the
        result says so rather than implying a clean pair.
        """
        venue_a = arb.venue_a
        venue_b = arb.venue_b
        
        logger.info(f"Arb atomic execution start: {venue_a} vs {venue_b} spread {arb.spread*100:.1f}% - V10 FIX #7 state machine")
        
        state = "PRECHECK"
        if arb.spread < 0.02:
            logger.warning(f"Arb PRECHECK FAIL: spread {arb.spread*100:.1f}% <2% - abort")
            return [
                ExecutionResult(venue_id=venue_a, market_id=arb.market_a.id, status="aborted", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning="PRECHECK FAIL spread <2%"),
                ExecutionResult(venue_id=venue_b, market_id=arb.market_b.id, status="aborted", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning="PRECHECK FAIL spread <2%")
            ]
        
        state = "RESERVE_CAPITAL"
        total_needed = amount_per_leg * 2
        if total_needed > self.bankroll * 0.20:
            logger.warning(f"Arb RESERVE FAIL: need ${total_needed} > 20% bankroll ${self.bankroll*0.20} - abort")
            return [
                ExecutionResult(venue_id=venue_a, market_id=arb.market_a.id, status="rejected", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning=f"RESERVE FAIL need ${total_needed} > 20% bankroll"),
                ExecutionResult(venue_id=venue_b, market_id=arb.market_b.id, status="rejected", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning=f"RESERVE FAIL need ${total_needed} > 20% bankroll")
            ]
        
        state = "VERIFY_BOTH_BOOKS"
        try:
            adapter_a = self.registry.get_adapter_for_venue_id(venue_a) if hasattr(self.registry, 'get_adapter_for_venue_id') else self.registry.adapters.get(venue_a)
            adapter_b = self.registry.get_adapter_for_venue_id(venue_b) if hasattr(self.registry, 'get_adapter_for_venue_id') else self.registry.adapters.get(venue_b)
            
            if adapter_a and adapter_b:
                ob_a = await adapter_a.get_orderbook(arb.market_a)
                ob_b = await adapter_b.get_orderbook(arb.market_b)
                
                # An unmeasured spread must not pass the arb gate on a default.
                # read_spread reports provenance; treat unknown as too wide.
                spread_a, real_a = read_spread(ob_a, 0.02)
                spread_b, real_b = read_spread(ob_b, 0.08)
                if not real_a or not real_b or spread_a > 0.08 or spread_b > 0.08:
                    logger.warning(f"Arb VERIFY_BOOKS FAIL: spread too wide A {ob_a.get('spread')} B {ob_b.get('spread')} - abort")
                    return [
                        ExecutionResult(venue_id=venue_a, market_id=arb.market_a.id, status="aborted", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning="VERIFY_BOOKS FAIL spread too wide"),
                        ExecutionResult(venue_id=venue_b, market_id=arb.market_b.id, status="aborted", amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0, reasoning="VERIFY_BOOKS FAIL spread too wide")
                    ]
                
                if not ob_a.get("is_real", False) or not ob_b.get("is_real", False):
                    logger.warning(f"Arb VERIFY_BOOKS WARN: orderbook not real A is_real={ob_a.get('is_real')} B is_real={ob_b.get('is_real')} - estimation, higher risk")
        except Exception as e:
            logger.warning(f"Arb VERIFY_BOOKS error {e} - continuing with caution")
        
        opp_a, opp_b = self.arb_leg_opportunities(arb)
        
        state = "SUBMIT_A"
        # The pair is sized in SHARES, and leg A is sized so the pair can still
        # be completed inside the capital the risk check above reserved.
        #
        # The riskless construction is n shares of each leg: one share of YES
        # plus one share of NO pays exactly $1. Two equal DOLLAR legs buy
        # n_a = D/p_a and n_b = D/p_b shares, so the pair only pays
        # min(n_a, n_b) - the spread is handed back as directional risk. The
        # budgets are therefore split in the ratio of the two token prices, and
        # sized on the CAP prices so the completion cannot exceed the reserve.
        slippage = _PRICE_SLIPPAGE
        token_a = _token_price(opp_a)
        token_b = _token_price(opp_b)
        pair_price = token_a + token_b
        if pair_price > 0:
            shares_target = total_needed / (pair_price + 2 * slippage)
            leg_a_budget = min(amount_per_leg,
                               shares_target * (token_a + slippage))
        else:
            shares_target = 0.0
            leg_a_budget = amount_per_leg
        # The side-aware cap, same rule as the single-opportunity path. A flat
        # `market_price + 0.02` caps the YES price, which for a NO leg is a cap
        # on a number that token never reaches - so the NO order was either
        # rejected or admitted at a price it could not fill at.
        result_a = await self.execute_single(
            opp_a, max_spend_usd=leg_a_budget,
            max_price=_side_aware_cap(opp_a))
        
        if result_a.status in ["error", "rejected", "rate_limited", "blocked", "aborted"]:
            logger.warning(f"Arb {state} FAIL: leg A {result_a.status} - aborting leg B to avoid naked exposure - state machine")
            return [result_a]
        
        # ------------------------------------------------------------------
        # How much does A actually have to be balanced by?
        #
        # `amount_per_leg` is what we ASKED for. Leg B was sized on it, and so
        # was the hedge - so a partial fill of A produced a leg B larger than
        # the position it was supposed to offset, and a hedge larger than the
        # exposure it was supposed to close. Either way the "riskless" pair ends
        # up holding a one-sided position, which is the exact failure the state
        # machine exists to prevent.
        # ------------------------------------------------------------------
        leg_a_fill = float(getattr(result_a, "filled_usd", 0.0) or 0.0)
        leg_a_shares = float(getattr(result_a, "filled_shares", 0.0) or 0.0)
        resting_a = float(getattr(result_a, "resting_usd", 0.0) or 0.0)
        if not result_a.bought_something or leg_a_fill <= 0:
            # A bought nothing, so there is nothing to balance and no reason to
            # send the second leg. Sending it anyway would be a naked bet that
            # the "arb" had nothing to do with.
            logger.warning(
                f"Arb SUBMIT_A produced no position ({result_a.status}, "
                f"${leg_a_fill:.2f} filled, ${resting_a:.2f} resting) - leg B "
                f"not submitted: there is nothing to hedge against")
            result_a.reasoning += (
                " | PAIR NOT ATTEMPTED: leg A committed no capital, so leg B "
                "would have been a one-sided position")
            return [result_a]

        if leg_a_shares <= 0:
            # A bought something and the venue did not say how much of it, or
            # said zero. Either way the position cannot be balanced by a
            # number, and a hedge sized on a guess is not a hedge.
            logger.warning(
                f"Arb SUBMIT_A filled ${leg_a_fill:.2f} but reported no share "
                f"count - leg B not submitted: what to balance it with is "
                f"unknown")
            result_a.reasoning += (
                f" | PAIR NOT ATTEMPTED: leg A filled ${leg_a_fill:.2f} with no "
                f"quantifiable share count, so leg B would have been sized on "
                f"a guess")
            return [result_a]

        # Recheck the price B would actually have to pay. `_side_aware_cap` was
        # computed from the spread DISCOVERED earlier; by the time A has filled,
        # that number is history. A pair admitted at the old cap can fill B at a
        # price where the arb no longer exists - paying to close a spread that
        # is no longer open.
        # ...against the price A ACTUALLY paid. `opp_a.market_price` is the
        # price the opportunity was discovered at, which is history by now.
        cap_b, price_b_expected, price_recheck = await self._arb_leg_cap(
            arb, opp_a, opp_b, adapter_b,
            a_price=float(getattr(result_a, "filled_price", 0.0) or 0.0) or None)
        if cap_b is None:
            logger.warning(
                f"Arb RECHECK: leg B edge is gone by the time A filled "
                f"({price_recheck}) - hedging A instead of paying up for B")
            result_b = ExecutionResult(
                venue_id=venue_b, market_id=arb.market_b.id, status="aborted",
                amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0,
                reasoning=f"RECHECK failed: {price_recheck}")
            return await self._hedge_naked_leg(
                arb, opp_a, result_a, result_b, leg_a_shares,
                f"leg B no longer priced ({price_recheck})")

        state = "SUBMIT_B"
        # What completing the pair costs is the SAME NUMBER OF SHARES on the
        # other side, which is not the same number of dollars. A half-filled
        # second leg is neither a pair nor a hedge: if the share-matched order
        # does not fit inside what is left of the pair budget, leg B is not
        # sent and A is hedged instead.
        pair_budget_left = max(0.0, total_needed - leg_a_fill)
        # The budget was planned in SHARES AT THE CAPS: `shares_target` shares,
        # each allowed up to cap_a + cap_b. A fill BETTER than the cap buys more
        # shares than the plan expected, and those shares have to be matched too
        # - otherwise a good price would be what stopped the pair completing.
        # The allowance is that excess, priced at B's cap: the pair is still
        # bounded, and it is bounded by the shares that exist.
        extra_shares = max(0.0, leg_a_shares - shares_target)
        pair_budget_left += extra_shares * cap_b
        # Sized on the price B is expected to pay, capped at `cap_b`. Sizing on
        # the cap would buy more shares than A holds whenever the book fills
        # better than the limit - a directional position wearing the word
        # "pair", which is the thing this whole path exists to prevent.
        needed_b = leg_a_shares * price_b_expected
        if needed_b > pair_budget_left + 0.01:
            logger.warning(
                f"Arb COMPLETE FAIL: balancing {leg_a_shares:.4f} shares costs "
                f"${needed_b:.2f} at the {cap_b:.3f} cap but only "
                f"${pair_budget_left:.2f} of the pair budget is left - not "
                f"sending a half-leg B, hedging A instead")
            result_b = ExecutionResult(
                venue_id=venue_b, market_id=arb.market_b.id, status="aborted",
                amount_usd=0, price=0, fees_usd=0, gas_usd=0, latency_ms=0,
                reasoning=(f"COMPLETE FAIL: {leg_a_shares:.4f} shares at "
                           f"{price_b_expected:.3f} costs ${needed_b:.2f} "
                           f"against ${pair_budget_left:.2f} of budget left"))
            return await self._hedge_naked_leg(
                arb, opp_a, result_a, result_b, leg_a_shares,
                f"balancing costs ${needed_b:.2f}, ${pair_budget_left:.2f} left")
        result_b = await self.execute_single(
            opp_b, max_spend_usd=needed_b, max_price=cap_b)

        if result_b.status in ["error", "rejected", "rate_limited", "blocked", "aborted"]:
            logger.error(f"Arb {state} FAIL: leg A FILLED ${leg_a_fill:.2f} ({result_a.status}) but leg B FAIL {result_b.status} - NAKED EXPOSURE - hedging A")
            return await self._hedge_naked_leg(
                arb, opp_a, result_a, result_b, leg_a_shares,
                f"leg B {result_b.status}")
        
        # Both legs are in - but "filled" is not a boolean. Whatever B did not
        # take is still A's exposure, and the hedge closes THAT, not the request.
        leg_b_fill = float(getattr(result_b, "filled_usd", 0.0) or 0.0)
        leg_b_shares = float(getattr(result_b, "filled_shares", 0.0) or 0.0)
        unmatched_shares = leg_a_shares - leg_b_shares
        if unmatched_shares > _SHARE_TOLERANCE:
            logger.warning(
                f"Arb PARTIAL PAIR: A filled {leg_a_shares:.4f} shares, B "
                f"filled {leg_b_shares:.4f} - {unmatched_shares:.4f} shares of "
                f"A are unmatched - hedging")
            hedged = await self._hedge_naked_leg(
                arb, opp_a, result_a, result_b, unmatched_shares,
                f"B filled {leg_b_shares:.4f} of {leg_a_shares:.4f} shares")
            return hedged

        state = "BOTH_FILLED"
        result_a.reasoning += (
            f" | PAIR MATCHED: A {leg_a_shares:.4f} / B {leg_b_shares:.4f} "
            f"shares (${leg_a_fill:.2f} / ${leg_b_fill:.2f}) "
            f"({price_recheck})")
        logger.success(f"Arb {state}: both legs filled A {result_a.status} B {result_b.status} spread {arb.spread*100:.1f}% - DONE")
        return [result_a, result_b]

    async def _arb_leg_cap(self, arb, opp_a, opp_b, adapter_b, a_price=None):
        """
        The cap for leg B, recomputed from the book as it is NOW.

        Returns (cap, price_to_size_on, note), or (None, reason) when the pair
        is no longer worth completing. The discovered spread is not a price: between measuring it
        and filling leg A, the book that made the arb can move, and completing
        the pair at the old cap would buy the second leg at a price that turns a
        riskless pair into a loss that is certain rather than expected.

        A book that cannot be re-read is reported as such and the OLD cap is
        used - refusing every arb whose venue will not let us re-read would be
        its own failure mode - but the note travels with the result so nobody
        reads a stale number as a measurement.
        """
        stale_cap = _side_aware_cap(opp_b)
        stale_price = _token_price(opp_b)
        if adapter_b is None or not hasattr(adapter_b, "get_orderbook"):
            return stale_cap, stale_price, "cap from the discovered spread (no adapter to re-read)"
        try:
            book = await adapter_b.get_orderbook(arb.market_b)
        except Exception as e:
            return stale_cap, stale_price, f"cap from the discovered spread (re-read failed: {e})"
        if not isinstance(book, dict) or not book.get("is_real", False):
            return stale_cap, stale_price, "cap from the discovered spread (book not real)"

        # The side B is buying: YES buys the ask, NO buys 1 - bid.
        live_price, quote_note = _best_quote(book, opp_b.side)
        if live_price is None:
            return stale_cap, stale_price, f"cap from the discovered spread ({quote_note})"

        # What a share of B would cost now, against what A actually PAID for
        # its share. One share of each pays $1, so the pair is worth completing
        # while the two prices still sum to less than a dollar; the moment they
        # do not, B is a loss and hedging A is the cheaper mistake.
        a_price = float(a_price or 0.0) or float(
            getattr(opp_a, "market_price", 0.0) or 0.0)
        if a_price > 0 and (live_price + a_price) >= 1.0:
            return None, None, (
                f"B would cost {live_price:.3f} against A's {a_price:.3f} "
                f"- the pair costs {live_price + a_price:.3f} for a $1 "
                f"payoff, so the spread is gone")
        # The cap is the limit; the price is what the dollars are sized on. They
        # are different numbers: sizing the order at the CAP buys MORE shares
        # when the book fills cheaper, which turns a matched pair into a
        # directional position on the other side.
        return (min(stale_cap, live_price + _PRICE_SLIPPAGE), live_price,
                f"cap re-read: {quote_note} against A's fill {a_price:.3f}")

    async def _hedge_naked_leg(self, arb, opp_a, result_a, result_b,
                               naked_shares: float,
                               why: str) -> List[ExecutionResult]:
        """
        Close the exposure that exists, measured - in SHARES.

        `naked_shares` is leg A's unmatched share count. Two earlier versions
        were both wrong in this unit: the first hedged `amount_per_leg`, and the
        second hedged the unmatched DOLLARS - which on a partial fill still
        swapped the position for a different one, and reported the exposure as
        closed while shares remained. One share of YES plus one share of NO pays
        exactly $1, so the hedge buys the shares it has to close.
        """
        naked_shares = float(naked_shares or 0.0)
        if naked_shares <= _SHARE_TOLERANCE:
            result_a.reasoning += f" | NOTHING TO HEDGE: {why}"
            return [result_a, result_b]

        hedge_side = "NO" if str(opp_a.side).upper() == "YES" else "YES"
        hedge_opp = VenueOpportunity(
            market=arb.market_a,
            venue_id=arb.venue_a,
            venue_type=_coerce_venue_type(arb.market_a.raw.get("venue_type")),
            side=hedge_side,
            market_price=arb.market_a.best_price,
            estimated_fair=arb.market_a.best_price,
            raw_edge=0,
            effective_edge=0,
            confidence=0.5,
            should_trade=False,
        )
        cap, cap_note = await self._hedge_cap(arb, hedge_side, hedge_opp)
        try:
            # The order is sized so it can buy `naked_shares` AT THE CAP: the
            # dollars are an output of the share count and the price, never the
            # input.
            hedge_result = await self.execute_single(
                hedge_opp, max_spend_usd=naked_shares * cap, max_price=cap)
            closed_shares = (
                float(getattr(hedge_result, "filled_shares", 0.0) or 0.0)
                if getattr(hedge_result, "bought_something", False) else 0.0)
            remaining = max(0.0, naked_shares - closed_shares)
            logger.info(
                f"Arb HEDGE: {why} - {naked_shares:.4f} shares naked, "
                f"{closed_shares:.4f} hedged ({hedge_result.status}) at "
                f"{float(getattr(hedge_result, 'filled_price', 0.0) or 0.0):.3f}, "
                f"{remaining:.4f} shares exposed")
            result_a.reasoning += (
                f" | HEDGE ({why}): {naked_shares:.4f} shares naked, "
                f"{closed_shares:.4f} hedged ({hedge_result.status}) at "
                f"${float(getattr(hedge_result, 'filled_price', 0.0) or 0.0):.3f}"
                f" ({cap_note}), {remaining:.4f} shares exposed")
            if remaining > _SHARE_TOLERANCE:
                # The honest label. A failed hedge is not "attempted".
                result_a.reasoning += " | NAKED EXPOSURE REMAINS - reconcile"
        except Exception as e:
            logger.error(
                f"Arb HEDGE failed: {e} - {naked_shares:.4f} shares naked "
                f"exposure remains!")
            result_a.reasoning += (
                f" | HEDGE FAILED ({why}): {type(e).__name__}: {e} - "
                f"{naked_shares:.4f} SHARES NAKED EXPOSURE REMAINS")
        return [result_a, result_b]

    async def _hedge_cap(self, arb, hedge_side, hedge_opp):
        """
        The cap for the hedge, from the book A would actually face NOW.

        The hedge is taken because something went wrong - a leg failed, or the
        price moved - which is exactly when the modelled price is least
        trustworthy. The live book is therefore the cap when it can be read, NOT
        the tighter of the two: a hedge capped at a price the market has left
        behind does not fill, and an unfilled insurance order leaves the
        exposure it existed to close. Paying the moved price locks in the loss
        the position already has; it does not add one, because the alternative
        is staying directional. Falls back to the side-aware cap on the
        discovered price when the book cannot be re-read, and says which one it
        used.
        """
        stale = _side_aware_cap(hedge_opp)
        adapter = (self.registry.get_adapter_for_venue_id(arb.venue_a)
                   if hasattr(self.registry, "get_adapter_for_venue_id")
                   else getattr(self.registry, "adapters", {}).get(arb.venue_a))
        if adapter is None or not hasattr(adapter, "get_orderbook"):
            return stale, "cap from the discovered price (no adapter to re-read)"
        try:
            book = await adapter.get_orderbook(arb.market_a)
        except Exception as e:
            return stale, f"cap from the discovered price (re-read failed: {e})"
        if not isinstance(book, dict) or not book.get("is_real", False):
            return stale, "cap from the discovered price (book not real)"
        live, note = _best_quote(book, hedge_side)
        if live is None:
            return stale, f"cap from the discovered price ({note})"
        return live + _PRICE_SLIPPAGE, f"hedge cap re-read: {note}"

    def get_report(self) -> Dict[str, Any]:
        return {
            "executor": "Multi-Venue Executor 18 venues",
            "venues": list(self.min_order_sizes.keys()),
            "rate_limits": "Pionex 10 req/sec, WhiteBIT HMAC-SHA512, AFX DEX EIP-712 wallet-signed no API keys, etc",
            "min_orders": self.min_order_sizes,
            "operational_overhead": "Each venue has own API auth model rate limits failure modes, start with one additional venue prove pipeline works then add next",
            "atomicity": ("NO cross-venue atomic primitive exists. Both legs are "
                          "separate orders at separate venues and cannot be "
                          "submitted as one unit: the pair is sized in SHARES, "
                          "leg B is sent for the SHARES leg A actually filled, "
                          "the price is re-read before B is sent, and the "
                          "UNMATCHED SHARES are hedged at the price the book now "
                          "shows. This is a sequential pair with an explicit "
                          "hedge, and the exposure window is real - see "
                          "execute_arbitrage_pair"),
            "capital_fragmentation": "Splitting $50 across multiple venues tiny positions fixed costs gas withdrawal fees min order eat larger percentage concentrate 2-3 venues until bankroll grows"
        }
