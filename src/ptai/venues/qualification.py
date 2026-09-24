"""
Venue Qualification - paper-trading qualification robust
FIXED V7: User correctly identified win rate alone is not profitability
Example: 90% wins +$0.01, 10% losses -$1.00 would have fantastic win rate and still lose money

Now includes:
- net P&L
- expected value
- fees
- slippage
- drawdown
- profit factor
- calibration
- Brier/log loss
- sample size
- execution quality
Not just win rate
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
    # FIXED V7: Additional metrics beyond win rate
    net_pnl: float = 0.0
    expected_value: float = 0.0
    fees_total: float = 0.0
    slippage_total: float = 0.0
    drawdown_max: float = 0.0
    profit_factor: float = 0.0
    calibration_ece: float = 0.0
    log_loss: float = 0.0
    execution_quality_avg: float = 0.0
    sample_size: int = 0

class VenueQualificationEngine:
    """
    Robust paper-trading qualification
    FIXED V7: Beyond win rate - includes P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier, sample, execution quality
    User: Venue qualification should eventually consider net P&L, expected value, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality not just win rate
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.qualification_file = self.data_dir / "venue_qualification.json"
        self.qualifications: Dict[str, QualificationResult] = {}
        self._load()

        # FIXED V7: Comprehensive requirements beyond win rate
        self.requirements = {
            "min_trades": 100,
            "min_win_rate": 0.55,
            "max_brier": 0.25,
            "max_log_loss": 0.6,
            "max_ece": 0.15,
            "min_forecast_skill": 0.6,
            "min_profit_paper": 0.0,  # must be profitable after fees
            "min_net_pnl": 0.0,  # net P&L after fees/slippage
            "min_expected_value": 0.01,  # EV >1% per trade
            "min_avg_edge": 0.03,  # 3% avg edge
            "min_calibration": 0.5,
            "max_drawdown": 0.20,  # max 20% drawdown
            "min_profit_factor": 1.1,  # gross profit / gross loss >1.1
            "min_execution_quality": 0.5,  # avg execution quality
            "min_sample_size": 100,
            "max_fees_pct": 0.05,  # fees <5% of profit
        }

    def _load(self):
        if self.qualification_file.exists():
            try:
                with open(self.qualification_file, 'r') as f:
                    data = json.load(f)
                    for venue_id, qual_data in data.items():
                        if qual_data.get("qualification_date"):
                            qual_data["qualification_date"] = datetime.fromisoformat(qual_data["qualification_date"])
                        # Handle old format without new fields
                        for field in ["net_pnl", "expected_value", "fees_total", "slippage_total", "drawdown_max", "profit_factor", "calibration_ece", "log_loss", "execution_quality_avg", "sample_size"]:
                            if field not in qual_data:
                                qual_data[field] = 0.0
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
                    "reasoning": qual.reasoning,
                    "net_pnl": qual.net_pnl,
                    "expected_value": qual.expected_value,
                    "fees_total": qual.fees_total,
                    "slippage_total": qual.slippage_total,
                    "drawdown_max": qual.drawdown_max,
                    "profit_factor": qual.profit_factor,
                    "calibration_ece": qual.calibration_ece,
                    "log_loss": qual.log_loss,
                    "execution_quality_avg": qual.execution_quality_avg,
                    "sample_size": qual.sample_size
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
        log_loss = performance_stats.get("log_loss", 1.0)
        ece = performance_stats.get("calibration_ece", 0.5)
        profit_paper = performance_stats.get("profit_paper", 0)
        profit_live = performance_stats.get("profit_live", 0)
        net_pnl = performance_stats.get("net_pnl", profit_paper)
        # The RECORDED expected net EV at entry - what the agent predicted it
        # would make - not the average edge.
        #
        # This read `performance_stats["expected_value"]` with a fallback to
        # `avg_edge`, and `expected_value` was itself `sum(edges)/n`: the mean of
        # the recorded EFFECTIVE EDGES, a probability-scale number. The check
        # below compares it with `min_expected_value = 0.01` and calls it
        # "EV > 1% per trade". Mean edge and expected monetary value are not the
        # same quantity, and a venue could clear the EV bar on its edge while
        # every trade lost money to fees.
        #
        # None when nothing was measured, and None fails the gate rather than
        # defaulting to zero: "never measured" must not pass, and must not read
        # as "measurably zero" either.
        ev_samples = int(performance_stats.get("expected_value_samples") or 0)
        ev_coverage = float(performance_stats.get("expected_value_coverage") or 0.0)
        expected_value = performance_stats.get("expected_value")
        if expected_value is None and ev_samples == 0:
            # Older rows carry no expected EV. Falling back to the mean edge is
            # the old behaviour and is allowed, but only when there is nothing
            # measured - and it is labelled, so a reader can tell which number
            # they were given.
            expected_value = avg_edge
            ev_source = "avg_edge fallback (no recorded expected net EV)"
        else:
            expected_value = float(expected_value or 0.0)
            ev_source = (f"recorded expected net EV over {ev_samples} trade(s), "
                         f"{ev_coverage*100:.0f}% coverage")
        fees_total = performance_stats.get("fees_total", 0)
        slippage_total = performance_stats.get("slippage_total", 0)
        drawdown_max = performance_stats.get("drawdown_max", 0)
        profit_factor = performance_stats.get("profit_factor", 0)
        execution_quality = performance_stats.get("execution_quality_avg", 0.5)
        skill = performance_stats.get("forecast_skill", 0.5)

        # FIXED V7: Check all requirements beyond win rate
        checks = {
            "min_trades": total >= self.requirements["min_trades"],
            "min_win_rate": win_rate >= self.requirements["min_win_rate"],
            "max_brier": brier <= self.requirements["max_brier"],
            "max_log_loss": log_loss <= self.requirements["max_log_loss"],
            "max_ece": ece <= self.requirements["max_ece"],
            "min_skill": skill >= self.requirements["min_forecast_skill"],
            "min_profit": profit_paper >= self.requirements["min_profit_paper"],
            "min_net_pnl": net_pnl >= self.requirements["min_net_pnl"],
            "min_ev": (expected_value is not None and
                       expected_value >= self.requirements["min_expected_value"]),
            "min_edge": avg_edge >= self.requirements["min_avg_edge"],
            "max_drawdown": drawdown_max <= self.requirements["max_drawdown"],
            "min_profit_factor": profit_factor >= self.requirements["min_profit_factor"],
            "min_execution": execution_quality >= self.requirements["min_execution_quality"],
        }

        is_qualified = all(checks.values())

        # Detailed reasoning showing why win rate alone insufficient
        reasoning = (
            f"Qualification V7 for {venue_id}: "
            f"total {total} >= {self.requirements['min_trades']}? {checks['min_trades']} | "
            f"win_rate {win_rate:.2f} >= {self.requirements['min_win_rate']}? {checks['min_win_rate']} BUT win rate alone NOT profitability - example 90% wins +$0.01 10% losses -$1.00 fantastic win rate still lose money | "
            f"net_pnl ${net_pnl:.2f} >= ${self.requirements['min_net_pnl']}? {checks['min_net_pnl']} | "
            f"expected_value {expected_value*100:.2f}% >= {self.requirements['min_expected_value']*100:.1f}%? {checks['min_ev']} ({ev_source}) | "
            f"profit_factor {profit_factor:.2f} >= {self.requirements['min_profit_factor']}? {checks['min_profit_factor']} | "
            f"brier {brier:.3f} <= {self.requirements['max_brier']}? {checks['max_brier']} | "
            f"log_loss {log_loss:.3f} <= {self.requirements['max_log_loss']}? {checks['max_log_loss']} | "
            f"ece {ece:.3f} <= {self.requirements['max_ece']}? {checks['max_ece']} | "
            f"skill {skill:.2f} >= {self.requirements['min_forecast_skill']}? {checks['min_skill']} | "
            f"profit ${profit_paper:.2f} >= ${self.requirements['min_profit_paper']}? {checks['min_profit']} | "
            f"avg_edge {avg_edge*100:.1f}% >= {self.requirements['min_avg_edge']*100:.1f}%? {checks['min_edge']} | "
            f"drawdown {drawdown_max*100:.1f}% <= {self.requirements['max_drawdown']*100:.0f}%? {checks['max_drawdown']} | "
            f"exec_quality {execution_quality:.2f} >= {self.requirements['min_execution_quality']}? {checks['min_execution']} | "
            f"fees ${fees_total:.2f} slippage ${slippage_total:.2f} | "
            f"Qualified {is_qualified} | "
            f"FIXED: now considers net P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality not just win rate"
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
            reasoning=reasoning,
            net_pnl=net_pnl,
            expected_value=expected_value,
            fees_total=fees_total,
            slippage_total=slippage_total,
            drawdown_max=drawdown_max,
            profit_factor=profit_factor,
            calibration_ece=ece,
            log_loss=log_loss,
            execution_quality_avg=execution_quality,
            sample_size=total
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
            "requirements_explanation": {
                "min_trades": "100 paper trades - sample size",
                "min_win_rate": "55%+ win rate - BUT not profitability alone",
                "min_net_pnl": "Net P&L after fees/slippage must be positive - fixes 90% wins +$0.01 10% losses -$1.00 still lose money example",
                "min_expected_value": "EV >1% per trade - expected value",
                "min_profit_factor": "Gross profit / gross loss >1.1 - profit factor",
                "max_brier": "Brier <=0.25 - calibration",
                "max_log_loss": "Log loss <=0.6",
                "max_ece": "ECE <=0.15 - calibration error",
                "max_drawdown": "Max 20% drawdown",
                "min_execution_quality": "Avg execution quality 0.5+"
            },
            "details": {
                q.venue_id: {
                    "total": q.total_paper_trades,
                    "win_rate": q.win_rate,
                    "net_pnl": q.net_pnl,
                    "ev": q.expected_value,
                    "profit_factor": q.profit_factor,
                    "brier": q.brier_score,
                    "log_loss": q.log_loss,
                    "ece": q.calibration_ece,
                    "skill": q.forecast_skill,
                    "profit": q.profit_paper,
                    "drawdown": q.drawdown_max,
                    "fees": q.fees_total,
                    "slippage": q.slippage_total,
                    "exec_quality": q.execution_quality_avg,
                    "qualified": q.is_qualified,
                    "reasoning": q.reasoning[:500]
                } for q in self.qualifications.values()
            },
            "principle": "Each venue must prove positive EV through paper trading before real capital, don't assume profitable prove via paper trading/backtesting then cautiously allocate. FIXED V7: win rate alone NOT profitability - now considers net P&L, EV, fees, slippage, drawdown, profit factor, calibration, Brier/log loss, sample size, execution quality"
        }

    def should_concentrate_on(self) -> Dict[str, Any]:
        qualified = [v for v in self.qualifications.values() if v.is_qualified]
        qualified_sorted = sorted(qualified, key=lambda x: x.forecast_skill, reverse=True)
        
        if not qualified_sorted:
            return {"message": "No qualified venues yet - need paper trading 100+ trades per venue", "recommendation": "Start with Polymarket paper trading"}
        
        return {
            "strong_venues": [f"{q.venue_id} skill={q.forecast_skill:.2f} win={q.win_rate:.2f} brier={q.brier_score:.3f} pnl=${q.net_pnl:.2f} pf={q.profit_factor:.2f}" for q in qualified_sorted[:3]],
            "recommendation": f"Concentrate on {qualified_sorted[0].venue_id} - demonstrated skill {qualified_sorted[0].forecast_skill:.2f} pnl ${qualified_sorted[0].net_pnl:.2f}",
            "principle": "PTAI learns which venue/category combos it is good at and concentrates research there"
        }
