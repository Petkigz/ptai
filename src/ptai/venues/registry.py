"""
Venue Registry - manages all adapters, learns which venues work
FIXED V7: Exact routing, no fallback to first eligible - hard safety
"""
from typing import Any, Dict, List, Optional
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity
from ..markets.base import Market


class VenueRegistry:
    """
    Registry of all venue adapters.
    Learns performance per venue/strategy/category.
    Concentrates research where demonstrated edge is strongest.
    FIXED V7: Exact routing - opportunity.venue_id -> Registry -> exact adapter -> exact market -> exact orderbook
    Never first eligible adapter
    """
    def __init__(self, country_code: str = "UG"):
        self.adapters: Dict[str, MarketAdapter] = {}
        self.country_code = country_code
        self.eligibility_cache: Dict[str, EligibilityStatus] = {}
        self.venue_performance: Dict[str, Dict] = {}

    def register(self, adapter: MarketAdapter):
        self.adapters[adapter.venue_id] = adapter
        logger.info(f"Registered venue {adapter.venue_id} type {adapter.venue_type.value}")

    def check_all_eligibility(self) -> Dict[str, EligibilityStatus]:
        results = {}
        for venue_id, adapter in self.adapters.items():
            status = adapter.check_eligibility(self.country_code)
            self.eligibility_cache[venue_id] = status
            results[venue_id] = status
            if status == EligibilityStatus.RESTRICTED:
                logger.warning(f"Venue {venue_id} RESTRICTED for {self.country_code} - will not trade")
        return results

    def get_eligible_adapters(self) -> List[MarketAdapter]:
        eligible = []
        for venue_id, adapter in self.adapters.items():
            status = self.eligibility_cache.get(venue_id)
            if status is None:
                status = adapter.check_eligibility(self.country_code)
                self.eligibility_cache[venue_id] = status
            if status == EligibilityStatus.ELIGIBLE:
                eligible.append(adapter)
            elif status == EligibilityStatus.RESTRICTED:
                logger.info(f"Skipping restricted venue {venue_id}")
        return eligible

    async def discover_all(self, target_per_venue: int = 500) -> List[Market]:
        """Discover markets from all eligible venues - exact venue_id immutable"""
        all_markets = []
        for adapter in self.get_eligible_adapters():
            try:
                markets = await adapter.discover_markets(target_count=target_per_venue)
                for m in markets:
                    m.venue_id = adapter.venue_id
                    m.raw["venue_id"] = adapter.venue_id
                    m.raw["discovery_source"] = f"VenueRegistry.discover_all->{adapter.venue_id}"
                all_markets.extend(markets)
                logger.info(f"{adapter.venue_id}: discovered {len(markets)} markets - venue_id immutable")
            except Exception as e:
                logger.error(f"{adapter.venue_id} discovery failed: {e}")
        return all_markets

    async def discover_all_including_verification(self, target_per_venue: int = 500) -> List[Market]:
        """Discover from eligible + requires_verification for paper trading learning"""
        all_markets = []
        for venue_id, adapter in self.adapters.items():
            status = self.eligibility_cache.get(venue_id)
            if status is None:
                status = adapter.check_eligibility(self.country_code)
                self.eligibility_cache[venue_id] = status
            if status in [EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_VERIFICATION]:
                try:
                    markets = await adapter.discover_markets(target_count=target_per_venue)
                    for m in markets:
                        m.venue_id = adapter.venue_id
                        m.raw["venue_id"] = adapter.venue_id
                        m.raw["discovery_source"] = f"VenueRegistry.discover_all_including_verification->{adapter.venue_id}"
                        m.raw["eligibility"] = status.value
                    all_markets.extend(markets)
                    logger.info(f"{adapter.venue_id} ({status.value}): discovered {len(markets)} markets for paper trading")
                except Exception as e:
                    logger.error(f"{adapter.venue_id} discovery failed: {e}")
            else:
                logger.info(f"Skipping {venue_id} status {status.value} - restricted")
        return all_markets

    def get_adapter_for_market(self, market: Market) -> Optional[MarketAdapter]:
        """
        FIXED V7: Exact routing, never first eligible
        Previously: eligible[0].get_orderbook(market) - dangerous if Kalshi market asked to Polymarket adapter
        Now: opportunity.venue_id -> VenueRegistry -> exact adapter -> exact market -> exact orderbook
        Never first eligible adapter
        """
        venue_id = getattr(market, 'venue_id', None)
        if not venue_id:
            venue_id = market.raw.get("venue_id") if hasattr(market, 'raw') else None
        if not venue_id:
            source = getattr(market, 'source', None)
            if source:
                venue_id = source.value if hasattr(source, 'value') else str(source)
            else:
                venue_id = "unknown"
        
        if hasattr(venue_id, 'value'):
            venue_id = venue_id.value
        venue_id = str(venue_id).lower()
        
        adapter = self.adapters.get(venue_id)
        if adapter:
            logger.debug(f"Routing market {market.id} venue_id {venue_id} -> exact adapter {adapter.venue_id}")
            return adapter
        
        logger.error(f"Routing FAILED: market {market.id} venue_id {venue_id} not found in adapters {list(self.adapters.keys())} - ABORT, never fallback to first eligible")
        return None

    def get_adapter_for_venue_id(self, venue_id: str) -> Optional[MarketAdapter]:
        """
        FIXED V7: Hard safety - if requested adapter doesn't exist, ABORT TRADE not try first available
        Previously dangerous fallback: if requested adapter doesn't exist use first eligible
        Now: if venue=kalshi and Kalshi adapter isn't available, correct result ABORT TRADE not try first venue
        """
        if hasattr(venue_id, 'value'):
            venue_id = venue_id.value
        venue_id = str(venue_id).lower()
        
        adapter = self.adapters.get(venue_id)
        if adapter:
            return adapter
        logger.error(f"Adapter for venue_id {venue_id} not found in {list(self.adapters.keys())} - ABORT TRADE, never fallback")
        return None

    def capability_report(self) -> Dict[str, Any]:
        """
        What each registered venue can actually do.

        The distinction that matters to a user is not eligibility but whether
        code exists behind the venue. Thirteen adapters once advertised market
        discovery and returned invented markets, so adding a venue to the
        registry looked like adding coverage. This separates the two questions:
        which venues are reachable, and which are only known about.

        Driven by AdapterCapability.implementation_status, so it cannot drift
        from what the adapter actually does.
        """
        from .adapter import STATUS_LIVE, STATUS_SCANNER, STATUS_UNIMPLEMENTED

        venues = []
        for venue_id, adapter in sorted(self.adapters.items()):
            caps = adapter.capabilities
            status = getattr(caps, "implementation_status", STATUS_LIVE)
            note = getattr(caps, "implementation_note", "")
            try:
                eligibility = adapter.check_eligibility(self.country_code).value
            except Exception as e:
                eligibility = f"error: {type(e).__name__}"
            venues.append({
                "venue_id": venue_id,
                "venue_type": adapter.venue_type.value,
                "implementation_status": status,
                "implemented": getattr(caps, "is_implemented", status == STATUS_LIVE),
                "returns_real_data": status == STATUS_LIVE,
                "can_trade": status == STATUS_LIVE and caps.supports_trading,
                "eligibility": eligibility,
                "note": note,
                "supports": {
                    "market_discovery": caps.supports_market_discovery,
                    "orderbook": caps.supports_orderbook,
                    "trading": caps.supports_trading,
                    "portfolio": caps.supports_portfolio,
                },
            })

        by_status: Dict[str, int] = {}
        for v in venues:
            by_status[v["implementation_status"]] = by_status.get(v["implementation_status"], 0) + 1

        return {
            "country_code": self.country_code,
            "total_registered": len(venues),
            "by_status": by_status,
            "live": [v["venue_id"] for v in venues if v["returns_real_data"]],
            "tradeable": [v["venue_id"] for v in venues if v["can_trade"]],
            "unimplemented": [v["venue_id"] for v in venues
                              if v["implementation_status"] == STATUS_UNIMPLEMENTED],
            "scanners": [v["venue_id"] for v in venues
                         if v["implementation_status"] == STATUS_SCANNER],
            "venues": venues,
            "how_to_read": ("implements/returns_real_data reflect whether a client exists, "
                            "not whether the venue is legal in your jurisdiction. "
                            "'unimplemented' venues return no markets by design."),
        }

    def rank_opportunities(self, opportunities: List[VenueOpportunity]) -> List[VenueOpportunity]:
        """
        Rank opportunities by common score across all venues.
        FIXED BUG: Previously looked up venue_id only, but update_performance stores venue_id:category
        Now correctly looks up venue_id:category first, then venue_id fallback, then category, then default.
        """
        for opp in opportunities:
            opp.calculate_common_score()
            category = opp.category or "unknown"
            venue_id = opp.venue_id.split("+")[0] if "+" in opp.venue_id else opp.venue_id
            if hasattr(venue_id, 'value'):
                venue_id = venue_id.value
            venue_id = str(venue_id).lower()
            
            key_exact = f"{venue_id}:{category}"
            
            venue_perf = None
            if key_exact in self.venue_performance:
                venue_perf = self.venue_performance[key_exact]
            elif venue_id in self.venue_performance:
                venue_perf = self.venue_performance[venue_id]
            else:
                for k, v in self.venue_performance.items():
                    if k.startswith(f"{venue_id}:"):
                        if venue_perf is None or v.get("total", 0) > venue_perf.get("total", 0):
                            venue_perf = v
                if venue_perf is None:
                    for k, v in self.venue_performance.items():
                        if f":{category}" in k:
                            if venue_perf is None or v.get("total", 0) > venue_perf.get("total", 0):
                                venue_perf = v
            
            if venue_perf is None:
                venue_perf = {}
            
            skill = venue_perf.get("forecast_skill", 0.5)
            brier = venue_perf.get("brier_score", 0.5)
            win_rate = venue_perf.get("win_rate", 0.5)
            
            calibration_multiplier = 1.0
            if brier > 0.3:
                calibration_multiplier = 0.7
            elif brier < 0.2:
                calibration_multiplier = 1.2
            
            win_rate_multiplier = 0.5 + win_rate
            
            opp.score *= (0.5 + skill) * calibration_multiplier * win_rate_multiplier
            
            if venue_perf:
                logger.debug(f"Learning adjustment for {venue_id}:{category} skill={skill:.2f} brier={brier:.3f} win_rate={win_rate:.2f} -> multiplier {(0.5+skill)*calibration_multiplier*win_rate_multiplier:.2f}")

        ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        return ranked

    def get_performance_for_venue_category(self, venue_id: str, category: str) -> Dict:
        key_exact = f"{venue_id}:{category}"
        if key_exact in self.venue_performance:
            return self.venue_performance[key_exact]
        if venue_id in self.venue_performance:
            return self.venue_performance[venue_id]
        best = None
        for k, v in self.venue_performance.items():
            if k.startswith(f"{venue_id}:"):
                if best is None or v.get("total", 0) > best.get("total", 0):
                    best = v
        return best or {"forecast_skill": 0.5, "brier_score": 0.5, "win_rate": 0.5, "total": 0}

    def update_performance(self, venue_id: str, category: str, outcome: Dict):
        key = f"{venue_id}:{category}"
        if key not in self.venue_performance:
            self.venue_performance[key] = {
                "total": 0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "profit": 0.0,
                "brier_sum": 0.0,
                "brier_score": 0.5,
                "forecast_skill": 0.5
            }
        perf = self.venue_performance[key]
        perf["total"] += 1
        if outcome.get("win"):
            perf["wins"] += 1
        perf["win_rate"] = perf["wins"] / perf["total"]
        forecast = outcome.get("forecast", 0.5)
        actual = 1.0 if outcome.get("win") else 0.0
        perf["brier_sum"] += (forecast - actual) ** 2
        perf["brier_score"] = perf["brier_sum"] / perf["total"]
        perf["forecast_skill"] = max(0, 1 - perf["brier_score"] * 2)

    def get_venue_leaderboard(self) -> List[Dict]:
        leaderboard = []
        for key, perf in self.venue_performance.items():
            venue_id, category = key.split(":", 1) if ":" in key else (key, "unknown")
            leaderboard.append({
                "venue": venue_id,
                "category": category,
                "total": perf["total"],
                "win_rate": perf["win_rate"],
                "brier": perf["brier_score"],
                "skill": perf["forecast_skill"],
                "profit": perf["profit"]
            })
        return sorted(leaderboard, key=lambda x: x["skill"], reverse=True)

    def should_concentrate_on(self) -> Dict[str, str]:
        leaderboard = self.get_venue_leaderboard()
        if not leaderboard:
            return {"message": "Not enough data - need paper trading"}
        
        top = leaderboard[:3]
        bottom = [x for x in leaderboard if x["skill"] < 0.55][:3]
        
        return {
            "strong_venues": [f"{x['venue']}:{x['category']} skill={x['skill']:.2f} win={x['win_rate']:.2f}" for x in top],
            "weak_venues": [f"{x['venue']}:{x['category']} skill={x['skill']:.2f}" for x in bottom],
            "recommendation": f"Concentrate on {top[0]['venue']} {top[0]['category']}" if top else "Need more data"
        }
