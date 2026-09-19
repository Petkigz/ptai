
"""
Multi-Venue Executor - handles execution across 18 venues with per-venue guards
Operational overhead: each venue has own API auth model rate limits failure modes
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import asyncio

from ..venues.registry import VenueRegistry
from ..venues.adapter import VenueOpportunity

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
        venue_id = opportunity.venue_id.split("+")[0] if "+" in opportunity.venue_id else opportunity.venue_id
        # Find adapter
        adapter = self.registry.adapters.get(venue_id)
        if not adapter:
            # Try to find matching
            for vid, ad in self.registry.adapters.items():
                if vid in venue_id or venue_id in vid:
                    adapter = ad
                    venue_id = vid
                    break
        if not adapter:
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status="error",
                amount_usd=0,
                price=0,
                fees_usd=0,
                gas_usd=0,
                latency_ms=0,
                reasoning=f"Adapter {venue_id} not found"
            )

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
            return ExecutionResult(
                venue_id=venue_id,
                market_id=opportunity.market.id,
                status=result.get("status", "unknown"),
                amount_usd=max_spend_usd,
                price=max_price,
                fees_usd=fees,
                gas_usd=gas,
                latency_ms=latency,
                reasoning=f"{venue_id} execution latency {latency:.1f}ms fees ${fees:.4f} gas ${gas:.4f} | {result.get('message', '')[:100]}"
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

    async def execute_arbitrage_pair(self, arb, amount_per_leg: float = 3.0) -> List[ExecutionResult]:
        # Execute both legs of arb - must be atomic-ish, fund transfers slow
        # For $50 bankroll, need to ensure both legs execute or none
        venue_a = arb.venue_a
        venue_b = arb.venue_b
        # Create opportunities for each leg
        # Simplified: use market_a and market_b from arb
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
        # Execute sequentially with guard
        result_a = await self.execute_single(opp_a, max_spend_usd=amount_per_leg, max_price=opp_a.market_price+0.02)
        if result_a.status in ["error", "rejected", "rate_limited"]:
            logger.warning(f"Arb leg A failed, aborting leg B to avoid naked exposure")
            return [result_a]

        result_b = await self.execute_single(opp_b, max_spend_usd=amount_per_leg, max_price=opp_b.market_price+0.02)
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
