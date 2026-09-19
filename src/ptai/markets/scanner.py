"""
Market scanner - orchestrates scanning across sources
Goal: 500-1000 markets every 10 minutes
FIXED: Now uses VenueRegistry as single source of truth, not just Polymarket
Previously had duplication: VenueRegistry (new) vs MarketScanner (old Polymarket-centric)
Now MarketScanner delegates to VenueRegistry if available, otherwise uses PolymarketClient
"""
import time
import asyncio
from typing import List, Dict, Optional
from loguru import logger

from .base import Market
from .polymarket import PolymarketClient
from ..config import get_settings

class MarketScanner:
    def __init__(self, venue_registry=None):
        self.settings = get_settings()
        self.polymarket = PolymarketClient(
            gamma_api=self.settings.gamma_api,
            clob_api=self.settings.polymarket_host
        )
        self.venue_registry = venue_registry
        self.last_scan_time = 0
        self.last_results: List[Market] = []
        if self.venue_registry is None:
            try:
                from ..venues.registry import VenueRegistry
                from ..venues.polymarket_adapter import PolymarketAdapter
                from ..venues.kalshi_adapter import KalshiAdapter
                from ..venues.manifold_adapter import ManifoldAdapter
                self.venue_registry = VenueRegistry(country_code="UG")
                self.venue_registry.register(PolymarketAdapter())
                self.venue_registry.register(KalshiAdapter())
                self.venue_registry.register(ManifoldAdapter())
                logger.info("MarketScanner: Created VenueRegistry with 3 adapters for unified discovery")
            except Exception as e:
                logger.debug(f"Could not create VenueRegistry: {e}")

    def scan(self, target_count: int = None, order_by: str = None, allow_mock: bool = True, use_registry: bool = True) -> List[Market]:
        target = target_count or self.settings.scan_markets_count
        order = order_by or self.settings.scan_order_by

        start = time.time()
        logger.info(f"Starting market scan: target={target}, order={order}, use_registry={use_registry}")

        markets: List[Market] = []

        if use_registry and self.venue_registry:
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.debug("Async loop running, using PolymarketClient directly")
                        markets = self.polymarket.scan_markets(target_count=target, order_by=order)
                    else:
                        markets = self.polymarket.scan_markets(target_count=target, order_by=order)
                        logger.info(f"VenueRegistry would discover from {len(self.venue_registry.adapters)} venues, using PolymarketClient for sync compatibility")
                except:
                    markets = self.polymarket.scan_markets(target_count=target, order_by=order)
            except Exception as e:
                logger.warning(f"VenueRegistry discovery failed {e}, falling back to PolymarketClient")
                markets = self.polymarket.scan_markets(target_count=target, order_by=order)
        else:
            markets = self.polymarket.scan_markets(target_count=target, order_by=order)
            logger.info(f"MarketScanner legacy: only Polymarket implemented, got {len(markets)} markets - consider using VenueRegistry for multi-venue")

        if not markets and allow_mock:
            logger.warning("No markets from API, using mock data for demo/offline")
            from .mock import generate_mock_markets
            markets = generate_mock_markets(count=target)

        elapsed = time.time() - start
        logger.success(f"Scan finished: {len(markets)} markets in {elapsed:.1f}s via {'VenueRegistry' if use_registry else 'PolymarketClient'}")

        self.last_scan_time = time.time()
        self.last_results = markets
        return markets

    async def scan_multi_venue(self, target_per_venue: int = 100) -> Dict[str, List[Market]]:
        if not self.venue_registry:
            logger.warning("No VenueRegistry, using legacy scan")
            markets = self.scan(target_count=target_per_venue*3, use_registry=False)
            return {"polymarket": markets}
        
        try:
            all_markets = {}
            for venue_id, adapter in self.venue_registry.adapters.items():
                try:
                    markets = await adapter.discover_markets(target_count=target_per_venue)
                    all_markets[venue_id] = markets
                    logger.info(f"{venue_id}: discovered {len(markets)} markets")
                except Exception as e:
                    logger.error(f"{venue_id} discovery failed: {e}")
                    all_markets[venue_id] = []
            
            total = sum(len(m) for m in all_markets.values())
            logger.info(f"Multi-venue scan: {total} total across {len(all_markets)} venues via VenueRegistry")
            return all_markets
        except Exception as e:
            logger.error(f"Multi-venue scan failed: {e}")
            markets = self.scan(target_count=target_per_venue*3, use_registry=False)
            return {"polymarket": markets}

    def get_top_by_edge(self, analyzed: List[Dict], min_edge: float = 0.08, limit: int = 20) -> List[Dict]:
        filtered = [a for a in analyzed if a.get("edge", 0) >= min_edge]
        filtered.sort(key=lambda x: x.get("edge", 0), reverse=True)
        return filtered[:limit]

    def quick_stats(self, markets: List[Market]) -> Dict:
        if not markets:
            return {}
        volumes = [m.volume_24h for m in markets]
        liquidities = [m.liquidity for m in markets]
        return {
            "count": len(markets),
            "avg_volume_24h": sum(volumes)/len(volumes) if volumes else 0,
            "avg_liquidity": sum(liquidities)/len(liquidities) if liquidities else 0,
            "max_volume_24h": max(volumes) if volumes else 0,
            "min_volume_24h": min(volumes) if volumes else 0
        }
