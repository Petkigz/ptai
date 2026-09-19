"""
Exposure Manager - portfolio exposure caps
Single market ≤6%, category ≤15%, correlated ≤20%, total ≤40-50%
Because 8 independent 6% positions can be same bet.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Exposure:
    total_exposure_usd: float
    total_exposure_pct: float
    by_market: Dict[str, float] = field(default_factory=dict)
    by_category: Dict[str, float] = field(default_factory=dict)
    by_correlation_group: Dict[str, float] = field(default_factory=dict)
    by_venue: Dict[str, float] = field(default_factory=dict)


class ExposureManager:
    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll
        self.max_single_pct = 0.06  # 6%
        self.max_category_pct = 0.15  # 15%
        self.max_correlated_pct = 0.20  # 20%
        self.max_total_pct = 0.50  # 50%
        self.max_open_positions = 6

        # Current exposures
        self.positions: List[Dict] = []  # List of open positions

    def update_bankroll(self, bankroll: float):
        self.bankroll = bankroll

    def add_position(self, market_id: str, amount_usd: float, category: str = "unknown", correlation_group: str = "", venue: str = "polymarket"):
        self.positions.append({
            "market_id": market_id,
            "amount_usd": amount_usd,
            "category": category,
            "correlation_group": correlation_group or category,
            "venue": venue
        })

    def remove_position(self, market_id: str):
        self.positions = [p for p in self.positions if p["market_id"] != market_id]

    def get_exposure(self) -> Exposure:
        total_usd = sum(p["amount_usd"] for p in self.positions)
        total_pct = total_usd / max(1, self.bankroll)

        by_market = {}
        by_category = {}
        by_corr = {}
        by_venue = {}

        for p in self.positions:
            by_market[p["market_id"]] = by_market.get(p["market_id"], 0) + p["amount_usd"]
            by_category[p["category"]] = by_category.get(p["category"], 0) + p["amount_usd"]
            by_corr[p["correlation_group"]] = by_corr.get(p["correlation_group"], 0) + p["amount_usd"]
            by_venue[p["venue"]] = by_venue.get(p["venue"], 0) + p["amount_usd"]

        return Exposure(
            total_exposure_usd=total_usd,
            total_exposure_pct=total_pct,
            by_market=by_market,
            by_category=by_category,
            by_correlation_group=by_corr,
            by_venue=by_venue
        )

    def can_open(self, market_id: str, amount_usd: float, category: str = "unknown", correlation_group: str = "", venue: str = "polymarket") -> tuple[bool, str]:
        """
        Check if we can open new position respecting all caps.
        Example: Trump, Republican, Senate, X policy may all correlate.
        """
        exposure = self.get_exposure()

        # Check max open positions
        if len(self.positions) >= self.max_open_positions:
            return False, f"Max open positions {self.max_open_positions} reached"

        # Check single market cap
        current_market_exposure = exposure.by_market.get(market_id, 0)
        if (current_market_exposure + amount_usd) / self.bankroll > self.max_single_pct:
            return False, f"Single market cap {self.max_single_pct*100}% exceeded: {(current_market_exposure + amount_usd)/self.bankroll*100:.1f}%"

        # Check category cap
        corr_group = correlation_group or category
        current_category = exposure.by_category.get(category, 0)
        if (current_category + amount_usd) / self.bankroll > self.max_category_pct:
            return False, f"Category {category} cap {self.max_category_pct*100}% exceeded: {(current_category + amount_usd)/self.bankroll*100:.1f}%"

        # Check correlated cap
        current_corr = exposure.by_correlation_group.get(corr_group, 0)
        if (current_corr + amount_usd) / self.bankroll > self.max_correlated_pct:
            return False, f"Correlated group {corr_group} cap {self.max_correlated_pct*100}% exceeded"

        # Check total exposure
        if (exposure.total_exposure_usd + amount_usd) / self.bankroll > self.max_total_pct:
            return False, f"Total exposure cap {self.max_total_pct*100}% exceeded: {(exposure.total_exposure_usd + amount_usd)/self.bankroll*100:.1f}%"

        return True, "OK"

    def get_risk_report(self) -> Dict:
        exposure = self.get_exposure()
        return {
            "bankroll": self.bankroll,
            "total_exposure_usd": exposure.total_exposure_usd,
            "total_exposure_pct": exposure.total_exposure_pct,
            "open_positions": len(self.positions),
            "by_category": exposure.by_category,
            "by_correlation": exposure.by_correlation_group,
            "by_venue": exposure.by_venue,
            "caps": {
                "single": self.max_single_pct,
                "category": self.max_category_pct,
                "correlated": self.max_correlated_pct,
                "total": self.max_total_pct,
                "max_positions": self.max_open_positions
            },
            "utilization": {
                "total": exposure.total_exposure_pct / self.max_total_pct,
                "positions": len(self.positions) / self.max_open_positions
            }
        }
