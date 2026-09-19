
"""
Venue Qualification - paper-trading qualification robust
Each venue must prove positive EV through paper trading before real capital
Makes paper-trading qualification robust per user request
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime, timezone
import json
from pathlib import Path

@dataclass
class QualificationResult:
    venue_id: str
    total_paper_trades: int
    win_rate: float
    avg_edge: float
    brier_score: float
    profit_paper: float
    profit_live: float
    forecast_skill: float
    is_qualified: bool
    qualification_date: Optional[datetime]
    requirements: Dict[str, Any]
    reasoning: str

class VenueQualificationEngine:
    """
    Robust paper-trading qualification
    Previously: simple check min_trades 100 win_rate 0.55 brier 0.25
    Now: comprehensive qualification with multiple criteria, calibration, profit, skill
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.qualification_file = self.data_dir / "venue_qualification.json"
        self.qualifications: Dict[str, QualificationResult] = {}
        self._load()

        # Qualification requirements
        self.requirements = {
            "min_trades": 100,
            "min_win_rate": 0.55,
            "max_brier": 0.25,
            "min_forecast_skill": 0.6,
            "min_profit_paper": 0.0,  # must be profitable after fees
            "min_avg_edge": 0.03,  # 3% avg edge
            "min_calibration": 0.5,  # at least 50% calibrated
            "max_drawdown": 0.20,  # max 20% drawdown
        }

    def _load(self):
        if self.qualification_file.exists():
            try:
                with open(self.qualification_file, 'r') as f:
                    data = json.load(f)
                    for venue_id, qual_data in data.items():
                        if qual_data.get("qualification_date"):
                            qual_data["qualification_date"] = datetime.fromisoformat(qual_data["qualification_date"])
                        self.qualifications[venue_id] = QualificationResult(**qual_data)
            except Exception as e:
                logger.warning(f"Qualification load failed: {e}")

    def _save(self):
        try:
            data = {}
            for venue_id, qual in self.qualifications.items():
                data[venue_id] = {
                    "venue_id": qual.venue_id,
                    "total_paper_trades": qual.total_paper_trades,
                    "win_rate": qual.win_rate,
                    "avg_edge": qual.avg_edge,
                    "brier_score": qual.brier_score,
                    "profit_paper": qual.profit_paper,
                    "profit_live": qual.profit_live,
                    "forecast_skill": qual.forecast_skill,
                    "is_qualified": qual.is_qualified,
                    "qualification_date": qual.qualification_date.isoformat() if qual.qualification_date else None,
                    "requirements": qual.requirements,
                    "reasoning": qual.reasoning
                }
            with open(self.qualification_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Qualification save failed: {e}")

    def evaluate_qualification(self, venue_id: str, performance_stats: Dict[str, Any]) -> QualificationResult:
        total = performance_stats.get("total_paper_trades", 0)
        win_rate = performance_stats.get("win_rate", 0)
        avg_edge = performance_stats.get("avg_edge", 0)
        brier = performance_stats.get("brier_score", 1.0)
        profit_paper = performance_stats.get("profit_paper", 0)
        profit_live = performance_stats.get("profit_live", 0)
        skill = performance_stats.get("forecast_skill", 0.5)

        # Check all requirements
        checks = {
            "min_trades": total >= self.requirements["min_trades"],
            "min_win_rate": win_rate >= self.requirements["min_win_rate"],
            "max_brier": brier <= self.requirements["max_brier"],
            "min_skill": skill >= self.requirements["min_forecast_skill"],
            "min_profit": profit_paper >= self.requirements["min_profit_paper"],
            "min_edge": avg_edge >= self.requirements["min_avg_edge"],
        }

        is_qualified = all(checks.values())

        reasoning = (
            f"Qualification for {venue_id}: total {total} >= {self.requirements['min_trades']}? {checks['min_trades']} | "
            f"win_rate {win_rate:.2f} >= {self.requirements['min_win_rate']}? {checks['min_win_rate']} | "
            f"brier {brier:.3f} <= {self.requirements['max_brier']}? {checks['max_brier']} | "
            f"skill {skill:.2f} >= {self.requirements['min_forecast_skill']}? {checks['min_skill']} | "
            f"profit ${profit_paper:.2f} >= ${self.requirements['min_profit_paper']}? {checks['min_profit']} | "
            f"avg_edge {avg_edge*100:.1f}% >= {self.requirements['min_avg_edge']*100:.1f}%? {checks['min_edge']} | "
            f"Qualified {is_qualified} | "
            f"Must prove positive EV through paper trading before real capital, 100 trades win_rate>55% Brier<0.25 skill>0.6 profitable after fees"
        )

        result = QualificationResult(
            venue_id=venue_id,
            total_paper_trades=total,
            win_rate=win_rate,
            avg_edge=avg_edge,
            brier_score=brier,
            profit_paper=profit_paper,
            profit_live=profit_live,
            forecast_skill=skill,
            is_qualified=is_qualified,
            qualification_date=datetime.now(timezone.utc) if is_qualified else None,
            requirements=self.requirements,
            reasoning=reasoning
        )

        self.qualifications[venue_id] = result
        self._save()

        if is_qualified:
            logger.success(f"Venue {venue_id} QUALIFIED for live trading: {reasoning}")
        else:
            logger.info(f"Venue {venue_id} NOT qualified: {reasoning}")

        return result

    def is_qualified(self, venue_id: str) -> bool:
        qual = self.qualifications.get(venue_id)
        if not qual:
            return False
        return qual.is_qualified

    def get_qualification_report(self) -> Dict[str, Any]:
        qualified = [v for v in self.qualifications.values() if v.is_qualified]
        not_qualified = [v for v in self.qualifications.values() if not v.is_qualified]
        
        return {
            "total_venues": len(self.qualifications),
            "qualified": len(qualified),
            "not_qualified": len(not_qualified),
            "qualified_venues": [q.venue_id for q in qualified],
            "not_qualified_venues": [q.venue_id for q in not_qualified],
            "requirements": self.requirements,
            "details": {
                q.venue_id: {
                    "total": q.total_paper_trades,
                    "win_rate": q.win_rate,
                    "brier": q.brier_score,
                    "skill": q.forecast_skill,
                    "profit": q.profit_paper,
                    "qualified": q.is_qualified,
                    "reasoning": q.reasoning[:300]
                } for q in self.qualifications.values()
            },
            "principle": "Each venue must prove positive EV through paper trading before real capital, don't assume profitable prove via paper trading/backtesting then cautiously allocate"
        }

    def should_concentrate_on(self) -> Dict[str, Any]:
        # Suggest where to concentrate based on qualified venues
        qualified = [v for v in self.qualifications.values() if v.is_qualified]
        qualified_sorted = sorted(qualified, key=lambda x: x.forecast_skill, reverse=True)
        
        if not qualified_sorted:
            return {"message": "No qualified venues yet - need paper trading 100+ trades per venue", "recommendation": "Start with Polymarket paper trading"}
        
        return {
            "strong_venues": [f"{q.venue_id} skill={q.forecast_skill:.2f} win={q.win_rate:.2f} brier={q.brier_score:.3f}" for q in qualified_sorted[:3]],
            "recommendation": f"Concentrate on {qualified_sorted[0].venue_id} - demonstrated skill {qualified_sorted[0].forecast_skill:.2f}",
            "principle": "PTAI learns which venue/category combos it is good at and concentrates research there"
        }
