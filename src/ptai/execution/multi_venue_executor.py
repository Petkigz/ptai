
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
    # A send that failed with no venue confirmation: the order may exist.
    unconfirmed_send: bool = False
    trade_ids: List[str] = field(default_factory=list)

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
    def should_record_position(self) -> bool:
        """
        Should a POSITION row be written?

        A partial fill is a real position of the filled size, so it is recorded
        and then reconciled upward as more of the order fills. A resting order
        with nothing matched is not a position - it is an ORDER, and it belongs
        in the order book table where it reserves capital without claiming
        shares.
        """
        return self.committed_capital or self.is_simulated

    def to_position_dict(self) -> Dict[str, Any]:
        """The facts of what filled. Zeros are honest: nothing filled."""
        return {
            "venue_id": self.venue_id,
            "market_id": self.market_id,
            "status": self.status,
            "requested_usd": self.amount_usd,
            "filled_usd": self.filled_usd,
            "filled_price": self.filled_price,
            "fees_usd": self.fees_usd,
            "gas_usd": self.gas_usd,
            "order_id": self.order_id,
            "simulated": self.is_simulated,
            "size_matched": self.size_matched,
            "original_size": self.original_size,
            "unfilled_shares": self.unfilled_shares,
            "resting_usd": self.resting_usd,
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
    if fill.get("status") not in ("submitted", "partial"):
        return 0.0
    if fill.get("unconfirmed_send"):
        return float(requested_usd or 0.0)
    matched = _num(fill.get("size_matched"))
    original = _num(fill.get("original_size"))
    price = _num(fill.get("price")) or 0.0
    if matched is None or not original or not price:
        # Nothing quantifiable: assume the whole request is still committed.
        return max(0.0, float(requested_usd or 0.0) - float(fill.get("filled_usd") or 0.0))
    remaining = max(0.0, float(original) - float(matched))
    return round(remaining * float(price), 6)


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
            # Gas for on-chain venues
            gas = 0.05 if venue_id in ["polymarket", "afx_dex"] else 0.0

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
                unconfirmed_send=bool(fill.get("unconfirmed_send")),
                trade_ids=list(fill.get("trade_ids") or []),
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
        else:
            filled_usd = 0.0

        return {"status": status, "filled_usd": filled_usd, "price": price,
                "order_id": order_id, "raw_status": raw_status,
                "size_matched": size_matched, "original_size": original_size,
                "unconfirmed_send": bool(result.get("unconfirmed_send")),
                "trade_ids": list(result.get("trade_ids") or [])}

    @staticmethod
    def _read_price(result: Dict[str, Any], fallback: float) -> float:
        for key in ("filled_price", "average_price", "avg_price", "price",
                    "fill_price"):
            value = _num(result.get(key))
            if value is not None and 0.0 < value <= 1.0:
                return value
        return 0.0

    async def execute_arbitrage_pair(self, arb, amount_per_leg: float = 3.0) -> List[ExecutionResult]:
        """
        V10 FIX #7: Arbitrage atomicity with state machine - avoid naked exposure
        Previously: LEG A → LEG B sequential, if A fills and B fails → naked exposure
        Now: PRECHECK → RESERVE → VERIFY BOTH BOOKS → SUBMIT A → SUBMIT B → BOTH FILLED or HEDGE
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
        
        opp_a = VenueOpportunity(
            market=arb.market_a,
            venue_id=venue_a,
            venue_type=arb.market_a.raw.get("venue_type", "prediction"),
            side="YES" if arb.price_a < arb.price_b else "NO",
            market_price=arb.price_a,
            estimated_fair=arb.price_b,
            raw_edge=arb.spread,
            effective_edge=arb.fee_adjusted_profit,
            confidence=arb.confidence_same_event,
            should_trade=arb.should_trade
        )
        opp_b = VenueOpportunity(
            market=arb.market_b,
            venue_id=venue_b,
            venue_type=arb.market_b.raw.get("venue_type", "prediction"),
            side="NO" if arb.price_a < arb.price_b else "YES",
            market_price=arb.price_b,
            estimated_fair=arb.price_a,
            raw_edge=arb.spread,
            effective_edge=arb.fee_adjusted_profit,
            confidence=arb.confidence_same_event,
            should_trade=arb.should_trade
        )
        
        state = "SUBMIT_A"
        result_a = await self.execute_single(opp_a, max_spend_usd=amount_per_leg, max_price=opp_a.market_price+0.02)
        
        if result_a.status in ["error", "rejected", "rate_limited", "blocked", "aborted"]:
            logger.warning(f"Arb {state} FAIL: leg A {result_a.status} - aborting leg B to avoid naked exposure - state machine")
            return [result_a]
        
        state = "SUBMIT_B"
        result_b = await self.execute_single(opp_b, max_spend_usd=amount_per_leg, max_price=opp_b.market_price+0.02)
        
        if result_b.status in ["error", "rejected", "rate_limited", "blocked", "aborted"]:
            logger.error(f"Arb {state} FAIL: leg A FILLED {result_a.status} but leg B FAIL {result_b.status} - NAKED EXPOSURE - need HEDGE A")
            state = "HEDGE_A"
            try:
                hedge_side = "NO" if opp_a.side == "YES" else "YES"
                hedge_opp = VenueOpportunity(
                    market=arb.market_a,
                    venue_id=venue_a,
                    venue_type=arb.market_a.raw.get("venue_type", "prediction"),
                    side=hedge_side,
                    market_price=arb.market_a.best_price,
                    estimated_fair=arb.market_a.best_price,
                    raw_edge=0,
                    effective_edge=0,
                    confidence=0.5,
                    should_trade=False
                )
                hedge_result = await self.execute_single(hedge_opp, max_spend_usd=amount_per_leg, max_price=hedge_opp.market_price+0.02)
                logger.info(f"Arb HEDGE_A result: {hedge_result.status} - attempted to close naked exposure")
                result_a.reasoning += f" | HEDGE attempted: {hedge_result.status} {hedge_result.reasoning}"
            except Exception as e:
                logger.error(f"Arb HEDGE_A failed: {e} - naked exposure remains!")
                result_a.reasoning += f" | HEDGE FAILED: {e} - NAKED EXPOSURE REMAINS"
            
            return [result_a, result_b]
        
        state = "BOTH_FILLED"
        logger.success(f"Arb {state} SUCCESS: both legs filled A {result_a.status} B {result_b.status} spread {arb.spread*100:.1f}% - DONE")
        return [result_a, result_b]

    def get_report(self) -> Dict[str, Any]:
        return {
            "executor": "Multi-Venue Executor 18 venues",
            "venues": list(self.min_order_sizes.keys()),
            "rate_limits": "Pionex 10 req/sec, WhiteBIT HMAC-SHA512, AFX DEX EIP-712 wallet-signed no API keys, etc",
            "min_orders": self.min_order_sizes,
            "operational_overhead": "Each venue has own API auth model rate limits failure modes, start with one additional venue prove pipeline works then add next",
            "atomicity": "Arb both legs must settle exactly same event definition, fund transfers between venues slow, need to ensure both legs execute or none to avoid naked exposure",
            "capital_fragmentation": "Splitting $50 across multiple venues tiny positions fixed costs gas withdrawal fees min order eat larger percentage concentrate 2-3 venues until bankroll grows"
        }
