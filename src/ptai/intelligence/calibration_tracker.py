"""
Calibration Tracking + Ensemble Fair Value
Log every prediction, compute Brier score and reliability diagrams, only trust LLM when calibrated in category

Ensemble: Combine LLM estimate + base rate + external odds + market price, weight by historical calibration

Top 5 to implement first per user - stops LLM from hallucinating edges
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger
import math
import json
from pathlib import Path


@dataclass
class PredictionLog:
    market_id: str
    question: str
    category: str
    timestamp: datetime
    market_price: float
    llm_prob: float
    base_rate: float
    reference_odds: Optional[float]
    ensemble_prob: float
    confidence: float
    uncertainty: float
    outcome: Optional[int] = None  # 1 if YES won, 0 if NO, None if not resolved
    resolved_at: Optional[datetime] = None


@dataclass
class CalibrationResult:
    category: str
    total_predictions: int
    resolved_predictions: int
    brier_score: float
    brier_skill: float  # vs base rate
    calibration_error: float  # ECE expected calibration error
    reliability_bins: List[Dict]  # for reliability diagram
    win_rate_when_said_70: float  # when LLM said 70%, how often it won
    is_calibrated: bool
    should_trust: bool
    weight_for_ensemble: float  # 0-1 weight based on calibration


class CalibrationTracker:
    """
    Tracks calibration of LLM predictions over time
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.log_file = self.data_dir / "calibration_log.jsonl"
        self.predictions: List[PredictionLog] = []
        self._load()

    def _load(self):
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        # Parse datetime
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        if data.get('resolved_at'):
                            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
                        self.predictions.append(PredictionLog(**data))
            except Exception as e:
                logger.warning(f"Calibration load failed: {e}")

    def _save_prediction(self, pred: PredictionLog):
        try:
            with open(self.log_file, 'a') as f:
                data = {
                    "market_id": pred.market_id,
                    "question": pred.question,
                    "category": pred.category,
                    "timestamp": pred.timestamp.isoformat(),
                    "market_price": pred.market_price,
                    "llm_prob": pred.llm_prob,
                    "base_rate": pred.base_rate,
                    "reference_odds": pred.reference_odds,
                    "ensemble_prob": pred.ensemble_prob,
                    "confidence": pred.confidence,
                    "uncertainty": pred.uncertainty,
                    "outcome": pred.outcome,
                    "resolved_at": pred.resolved_at.isoformat() if pred.resolved_at else None
                }
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.warning(f"Calibration save failed: {e}")

    def log_prediction(self, market_id: str, question: str, category: str,
                      market_price: float, llm_prob: float, base_rate: float,
                      reference_odds: Optional[float], ensemble_prob: float,
                      confidence: float, uncertainty: float):
        pred = PredictionLog(
            market_id=market_id,
            question=question,
            category=category,
            timestamp=datetime.now(timezone.utc),
            market_price=market_price,
            llm_prob=llm_prob,
            base_rate=base_rate,
            reference_odds=reference_odds,
            ensemble_prob=ensemble_prob,
            confidence=confidence,
            uncertainty=uncertainty
        )
        self.predictions.append(pred)
        self._save_prediction(pred)
        logger.info(f"Calibration logged: {market_id} LLM {llm_prob:.3f} ensemble {ensemble_prob:.3f} market {market_price:.3f} cat {category}")

    def resolve_prediction(self, market_id: str, outcome: int):
        """Resolve prediction when market resolves: outcome 1 if YES won, 0 if NO"""
        for pred in self.predictions:
            if pred.market_id == market_id and pred.outcome is None:
                pred.outcome = outcome
                pred.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Calibration resolved: {market_id} outcome {outcome}")
                break

    def calculate_brier_score(self, predictions: List[PredictionLog]) -> float:
        """Brier score: mean squared error between predicted prob and outcome"""
        if not predictions:
            return 0.5
        resolved = [p for p in predictions if p.outcome is not None]
        if not resolved:
            return 0.5
        return sum((p.ensemble_prob - p.outcome) ** 2 for p in resolved) / len(resolved)

    def calculate_brier_skill(self, predictions: List[PredictionLog]) -> float:
        """Brier skill vs base rate: 1 - (Brier_model / Brier_baseline)"""
        resolved = [p for p in predictions if p.outcome is not None]
        if len(resolved) < 5:
            return 0.0
        
        brier_model = self.calculate_brier_score(resolved)
        # Baseline Brier: predict 0.5 always => Brier 0.25 average
        # Or predict base rate
        brier_baseline = sum((0.5 - p.outcome) ** 2 for p in resolved) / len(resolved)
        if brier_baseline == 0:
            return 0.0
        return 1 - (brier_model / brier_baseline)

    def calculate_calibration(self, predictions: List[PredictionLog], n_bins: int = 10) -> Tuple[float, List[Dict]]:
        """
        Expected Calibration Error (ECE) and reliability diagram
        Bin predictions by predicted prob, check actual win rate per bin
        """
        resolved = [p for p in predictions if p.outcome is not None]
        if len(resolved) < 10:
            return 0.0, []
        
        bins = [{"count": 0, "sum_prob": 0, "sum_outcome": 0, "bin_low": i/n_bins, "bin_high": (i+1)/n_bins} for i in range(n_bins)]
        
        for pred in resolved:
            bin_idx = min(int(pred.ensemble_prob * n_bins), n_bins-1)
            bins[bin_idx]["count"] += 1
            bins[bin_idx]["sum_prob"] += pred.ensemble_prob
            bins[bin_idx]["sum_outcome"] += pred.outcome
        
        ece = 0.0
        reliability_bins = []
        for b in bins:
            if b["count"] > 0:
                avg_prob = b["sum_prob"] / b["count"]
                avg_outcome = b["sum_outcome"] / b["count"]
                bin_error = abs(avg_prob - avg_outcome) * b["count"] / len(resolved)
                ece += bin_error
                reliability_bins.append({
                    "bin": f"{b['bin_low']:.1f}-{b['bin_high']:.1f}",
                    "count": b["count"],
                    "avg_predicted": avg_prob,
                    "actual_win_rate": avg_outcome,
                    "calibration_error": abs(avg_prob - avg_outcome)
                })
        
        return ece, reliability_bins

    def get_category_calibration(self, category: str) -> CalibrationResult:
        cat_preds = [p for p in self.predictions if p.category == category]
        resolved = [p for p in cat_preds if p.outcome is not None]
        
        brier = self.calculate_brier_score(cat_preds)
        brier_skill = self.calculate_brier_skill(cat_preds)
        ece, reliability_bins = self.calculate_calibration(cat_preds)
        
        # Win rate when said 70%
        said_70 = [p for p in resolved if 0.65 <= p.ensemble_prob <= 0.75]
        win_rate_70 = sum(p.outcome for p in said_70) / len(said_70) if said_70 else 0.0
        
        # Is calibrated? Brier <0.25 and ECE <0.15
        is_calibrated = brier < 0.25 and ece < 0.15 and len(resolved) >= 20
        should_trust = is_calibrated and brier_skill > 0.1
        
        # Weight for ensemble based on calibration
        # Well calibrated => high weight, poorly calibrated => low weight
        if len(resolved) < 10:
            weight = 0.5  # neutral if not enough data
        elif is_calibrated:
            weight = min(0.9, 0.5 + brier_skill * 0.5 + (0.15 - ece) * 2)
        else:
            weight = max(0.1, 0.5 - ece * 2 - (0.25 - brier) * 2)
        
        return CalibrationResult(
            category=category,
            total_predictions=len(cat_preds),
            resolved_predictions=len(resolved),
            brier_score=brier,
            brier_skill=brier_skill,
            calibration_error=ece,
            reliability_bins=reliability_bins,
            win_rate_when_said_70=win_rate_70,
            is_calibrated=is_calibrated,
            should_trust=should_trust,
            weight_for_ensemble=weight
        )

    def get_all_calibration(self) -> Dict[str, CalibrationResult]:
        categories = set(p.category for p in self.predictions)
        results = {}
        for cat in categories:
            results[cat] = self.get_category_calibration(cat)
        return results

    def get_ensemble_weights(self, category: str, has_reference: bool = False) -> Dict[str, float]:
        """
        Get ensemble weights for fair value: LLM, base rate, reference odds, market price
        Weight by historical calibration
        """
        cal = self.get_category_calibration(category)
        
        # Base weights
        if cal.resolved_predictions < 10:
            # Not enough data, equal weights
            weights = {
                "llm": 0.35,
                "base_rate": 0.25,
                "reference": 0.25 if has_reference else 0.0,
                "market": 0.15
            }
        elif cal.is_calibrated:
            # Well calibrated, trust LLM more
            weights = {
                "llm": 0.50 * cal.weight_for_ensemble,
                "base_rate": 0.20,
                "reference": 0.20 if has_reference else 0.0,
                "market": 0.10
            }
        else:
            # Poorly calibrated, trust base rate and market more, LLM less
            weights = {
                "llm": 0.20 * cal.weight_for_ensemble,
                "base_rate": 0.35,
                "reference": 0.25 if has_reference else 0.0,
                "market": 0.20
            }
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights

    def calculate_ensemble_fair_value(self, llm_prob: float, base_rate: float,
                                     reference_odds: Optional[float], market_price: float,
                                     category: str) -> Tuple[float, float, Dict[str, float], str]:
        """
        Calculate ensemble fair value weighted by calibration
        Returns ensemble prob, confidence, weights, reasoning
        """
        has_ref = reference_odds is not None
        weights = self.get_ensemble_weights(category, has_reference=has_ref)
        
        # Weighted ensemble
        ensemble = 0.0
        ensemble += llm_prob * weights.get("llm", 0)
        ensemble += base_rate * weights.get("base_rate", 0)
        if has_ref:
            ensemble += reference_odds * weights.get("reference", 0)
        ensemble += market_price * weights.get("market", 0)
        
        # Confidence based on agreement between sources and calibration
        sources = [llm_prob, base_rate, market_price]
        if has_ref:
            sources.append(reference_odds)
        
        # If sources agree (low variance), high confidence, else low
        mean = sum(sources) / len(sources)
        variance = sum((s - mean) ** 2 for s in sources) / len(sources)
        agreement_confidence = max(0.1, 1.0 - variance * 5)  # high variance => low confidence
        
        cal = self.get_category_calibration(category)
        calibration_confidence = cal.weight_for_ensemble if cal.resolved_predictions >= 10 else 0.5
        
        final_confidence = (agreement_confidence * 0.5 + calibration_confidence * 0.5)
        
        ref_weight = weights.get("reference", 0)
        ref_part = f"reference {reference_odds}×{ref_weight:.2f} + " if has_ref else ""
        reasoning = (
            f"Ensemble fair value: LLM {llm_prob:.3f}×{weights.get('llm',0):.2f} + "
            f"base_rate {base_rate:.3f}×{weights.get('base_rate',0):.2f} + "
            f"{ref_part}"
            f"market {market_price:.3f}×{weights.get('market',0):.2f} = {ensemble:.3f} | "
            f"Category {category} calibration Brier {cal.brier_score:.3f} skill {cal.brier_skill:.3f} "
            f"ECE {cal.calibration_error:.3f} trust {cal.should_trust} weight {cal.weight_for_ensemble:.2f} | "
            f"Agreement conf {agreement_confidence:.2f} calibration conf {calibration_confidence:.2f} final {final_confidence:.2f}"
        )
        
        return ensemble, final_confidence, weights, reasoning

    def get_report(self) -> Dict[str, Any]:
        all_cal = self.get_all_calibration()
        return {
            "total_predictions": len(self.predictions),
            "resolved_predictions": len([p for p in self.predictions if p.outcome is not None]),
            "categories": list(all_cal.keys()),
            "calibration_by_category": {
                cat: {
                    "total": res.total_predictions,
                    "resolved": res.resolved_predictions,
                    "brier": res.brier_score,
                    "brier_skill": res.brier_skill,
                    "ece": res.calibration_error,
                    "win_rate_when_said_70": res.win_rate_when_said_70,
                    "is_calibrated": res.is_calibrated,
                    "should_trust": res.should_trust,
                    "weight": res.weight_for_ensemble,
                    "reliability_bins": res.reliability_bins[:5]  # top 5 bins
                } for cat, res in all_cal.items()
            },
            "importance": "Top 5 to implement first - stops LLM from hallucinating edges, only trust when calibrated",
            "method": "Log every prediction, compute Brier score and reliability diagrams, weight ensemble by calibration"
        }
