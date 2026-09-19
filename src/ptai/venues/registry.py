"""
Venue Registry - manages all adapters, learns which venues work
"""
from typing import Dict, List, Optional
from loguru import logger

from .adapter import MarketAdapter, VenueType, EligibilityStatus, VenueOpportunity
from ..markets.base import Market


class VenueRegistry:
    """
    Registry of all venue adapters.
    Learns performance per venue/strategy/category.
    Concentrates research where demonstrated edge is strongest.
    """
    def __init__(self, country_code: str = "UG"):
        self.adapters: Dict[str, MarketAdapter] = {}
        self.country_code = country_code
        self.eligibility_cache: Dict[str, EligibilityStatus] = {}
        # Performance tracking per venue/category
        self.venue_performance: Dict[str, Dict] = {}

    def register(self, adapter: MarketAdapter):
        self.adapters[adapter.venue_id] = adapter
        logger.info(f"Registered venue {adapter.venue_id} type {adapter.venue_type.value}")

    def check_all_eligibility(self) -> Dict[str, EligibilityStatus]:
        """Check eligibility for all venues - critical for geographic compliance"""
        results = {}
        for venue_id, adapter in self.adapters.items():
            status = adapter.check_eligibility(self.country_code)
            self.eligibility_cache[venue_id] = status
            results[venue_id] = status
            if status == EligibilityStatus.RESTRICTED:
                logger.warning(f"Venue {venue_id} RESTRICTED for {self.country_code} - will not trade")
        return results

    def get_eligible_adapters(self) -> List[MarketAdapter]:
        """Only return adapters that are eligible and not restricted"""
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
        """Discover markets from all eligible venues"""
        all_markets = []
        for adapter in self.get_eligible_adapters():
            try:
                markets = await adapter.discover_markets(target_count=target_per_venue)
                all_markets.extend(markets)
                logger.info(f"{adapter.venue_id}: discovered {len(markets)} markets")
            except Exception as e:
                logger.error(f"{adapter.venue_id} discovery failed: {e}")
        return all_markets

    def rank_opportunities(self, opportunities: List[VenueOpportunity]) -> List[VenueOpportunity]:
        """
        Rank opportunities by common score across all venues.
        PTAI learns which venues it is good at and concentrates there.
        FIXED BUG: Previously looked up venue_id only, but update_performance stores venue_id:category
        Now correctly looks up venue_id:category first, then venue_id fallback, then category, then default.
        This makes learning/concentration system actually effective.
        """
        # Calculate common score for each
        for opp in opportunities:
            opp.calculate_common_score()
            # FIXED: Look up performance by venue_id:category first (as stored), then fallbacks
            category = opp.category or "unknown"
            venue_id = opp.venue_id.split("+")[0] if "+" in opp.venue_id else opp.venue_id  # handle composite arb ids
            
            # Try exact key venue:category
            key_exact = f"{venue_id}:{category}"
            # Try venue_id only (legacy)
            # Try category only
            # Try any key containing venue_id
            venue_perf = None
            if key_exact in self.venue_performance:
                venue_perf = self.venue_performance[key_exact]
            elif venue_id in self.venue_performance:
                venue_perf = self.venue_performance[venue_id]
            else:
                # Find any performance for this venue (any category)
                for k, v in self.venue_performance.items():
                    if k.startswith(f"{venue_id}:"):
                        if venue_perf is None or v.get("total", 0) > venue_perf.get("total", 0):
                            venue_perf = v
                # If still none, try category match across venues
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
            
            # Boost opportunities from venues where we have demonstrated skill
            # Skill 0.5 -> 1.0x multiplier, 0.8 -> 1.3x, 0.9 -> 1.4x
            # Also penalize poorly calibrated venues
            calibration_multiplier = 1.0
            if brier > 0.3:  # poorly calibrated
                calibration_multiplier = 0.7
            elif brier < 0.2:  # well calibrated
                calibration_multiplier = 1.2
            
            # Win rate multiplier
            win_rate_multiplier = 0.5 + win_rate  # 0.5 win rate -> 1.0x, 0.7 -> 1.2x
            
            opp.score *= (0.5 + skill) * calibration_multiplier * win_rate_multiplier
            
            # Log for debugging learning effectiveness
            if venue_perf:
                logger.debug(f"Learning adjustment for {venue_id}:{category} skill={skill:.2f} brier={brier:.3f} win_rate={win_rate:.2f} -> score multiplier {(0.5+skill)*calibration_multiplier*win_rate_multiplier:.2f}")

        # Sort by score descending
        ranked = sorted(opportunities, key=lambda x: x.score, reverse=True)
        return ranked

    def get_performance_for_venue_category(self, venue_id: str, category: str) -> Dict:
        """Helper to get performance with correct key handling"""
        key_exact = f"{venue_id}:{category}"
        if key_exact in self.venue_performance:
            return self.venue_performance[key_exact]
        if venue_id in self.venue_performance:
            return self.venue_performance[venue_id]
        # Find best match for venue
        best = None
        for k, v in self.venue_performance.items():
            if k.startswith(f"{venue_id}:"):
                if best is None or v.get("total", 0) > best.get("total", 0):
                    best = v
        return best or {"forecast_skill": 0.5, "brier_score": 0.5, "win_rate": 0.5, "total": 0}

    def update_performance(self, venue_id: str, category: str, outcome: Dict):
        """Update performance tracking after trade resolution"""
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
        # Update Brier
        forecast = outcome.get("forecast", 0.5)
        actual = 1.0 if outcome.get("win") else 0.0
        perf["brier_sum"] += (forecast - actual) ** 2
        perf["brier_score"] = perf["brier_sum"] / perf["total"]
        # Forecast skill = 1 - Brier (higher better), calibrated
        perf["forecast_skill"] = max(0, 1 - perf["brier_score"] * 2)

    def get_venue_leaderboard(self) -> List[Dict]:
        """Get leaderboard of venues by demonstrated skill - for learning"""
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
        """Suggest where to concentrate research based on demonstrated edge"""
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
