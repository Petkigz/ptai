"""
Calibration Engine - tracks whether 70% forecasts actually win 70% of time
Extremely important for long-term profitability.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime, timezone
from loguru import logger


@dataclass
class CalibrationPoint:
    forecast_id: str
    market_id: str
    question: str
    forecast_prob: float
    confidence: float
    market_price: float
    category: str
    timestamp: datetime
    actual_outcome: Optional[float] = None  # 1.0 YES, 0.0 NO, None pending
    resolved_at: Optional[datetime] = None


class CalibrationEngine:
    """
    Maintains calibration database:
    Prediction -> forecast prob -> market resolves -> actual outcome -> calibration DB
    Calculates Brier score, log loss, calibration curve, accuracy by bucket/category/time/source
    """
    def __init__(self, storage=None, memory=None):
        self.storage = storage
        self.memory = memory
        # In-memory cache for quick calibration - would be backed by DB
        self.points: List[CalibrationPoint] = []
        # V10 FIX #10: Learned calibration adjustments from history, not hardcoded
        # Previously: hardcoded politics 70%→64% etc - should be learned from data
        # Now: start empty, learn from actual outcomes
        self.category_adjustments: Dict[str, Dict] = {
            "default": {"adjustment": 0.0, "learned": False, "sample_size": 0}
        }
        # Learned adjustments storage - keyed by category and prob bucket
        self.learned_adjustments: Dict[str, Dict] = {}
        # Keep track of when adjustments were learned
        self.adjustment_history: List[Dict] = []

    def record_forecast(self, market_id: str, question: str, forecast_prob: float, confidence: float, market_price: float, category: str = "default") -> str:
        """Record a forecast before resolution"""
        import uuid
        forecast_id = str(uuid.uuid4())[:8]
        point = CalibrationPoint(
            forecast_id=forecast_id,
            market_id=market_id,
            question=question,
            forecast_prob=forecast_prob,
            confidence=confidence,
            market_price=market_price,
            category=category,
            timestamp=datetime.now(timezone.utc)
        )
        self.points.append(point)
        
        # Also store in DB if available
        if self.storage:
            try:
                self.storage.conn.execute(
                    "INSERT INTO calibration (id, market_id, forecast_prob, confidence, market_price, category, timestamp) VALUES (?,?,?,?,?,?,?)",
                    (forecast_id, market_id, forecast_prob, confidence, market_price, category, point.timestamp.isoformat())
                )
                self.storage.conn.commit()
            except Exception as e:
                logger.warning(f"Calibration DB insert failed: {e}")

        logger.info(f"Calibration recorded {forecast_id}: {question[:50]} forecast={forecast_prob:.3f} market={market_price:.3f}")
        return forecast_id

    def record_resolution(self, forecast_id: str, actual_outcome: float):
        """Record actual outcome when market resolves"""
        for p in self.points:
            if p.forecast_id == forecast_id:
                p.actual_outcome = actual_outcome
                p.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Calibration resolved {forecast_id}: forecast={p.forecast_prob:.3f} actual={actual_outcome}")
                break

    def calculate_brier_score(self, category: str = None) -> float:
        """Brier score: mean squared error of forecasts - lower is better, good <0.2"""
        relevant = [p for p in self.points if p.actual_outcome is not None]
        if category:
            relevant = [p for p in relevant if p.category == category]
        if not relevant:
            return 0.5  # neutral
        
        total = sum((p.forecast_prob - p.actual_outcome) ** 2 for p in relevant)
        return total / len(relevant)

    def calculate_log_loss(self, category: str = None) -> float:
        """Log loss"""
        relevant = [p for p in self.points if p.actual_outcome is not None]
        if category:
            relevant = [p for p in relevant if p.category == category]
        if not relevant:
            return 0.7
        
        total = 0
        for p in relevant:
            prob = max(0.01, min(0.99, p.forecast_prob))
            if p.actual_outcome == 1.0:
                total += -math.log(prob)
            else:
                total += -math.log(1 - prob)
        return total / len(relevant)

    def calibration_curve(self, buckets: int = 10) -> List[Dict]:
        """Calibration curve: for predictions where it says 70%, does 70% actually win?"""
        relevant = [p for p in self.points if p.actual_outcome is not None]
        if not relevant:
            return []
        
        bucket_size = 1.0 / buckets
        curve = []
        for i in range(buckets):
            low = i * bucket_size
            high = (i+1) * bucket_size
            bucket_points = [p for p in relevant if low <= p.forecast_prob < high]
            if bucket_points:
                avg_forecast = sum(p.forecast_prob for p in bucket_points) / len(bucket_points)
                avg_actual = sum(p.actual_outcome for p in bucket_points) / len(bucket_points)
                curve.append({
                    "bucket": f"{low:.1f}-{high:.1f}",
                    "forecast": avg_forecast,
                    "actual": avg_actual,
                    "count": len(bucket_points),
                    "overconfident": avg_forecast > avg_actual + 0.05
                })
        return curve

    def accuracy_by_confidence(self) -> Dict[str, float]:
        """Accuracy by confidence bucket"""
        relevant = [p for p in self.points if p.actual_outcome is not None]
        buckets = {"low": [], "medium": [], "high": []}
        for p in relevant:
            if p.confidence < 0.6:
                buckets["low"].append(p)
            elif p.confidence < 0.8:
                buckets["medium"].append(p)
            else:
                buckets["high"].append(p)
        
        result = {}
        for bucket_name, points in buckets.items():
            if points:
                correct = sum(1 for p in points if (p.forecast_prob >= 0.5 and p.actual_outcome == 1.0) or (p.forecast_prob < 0.5 and p.actual_outcome == 0.0))
                result[bucket_name] = correct / len(points)
            else:
                result[bucket_name] = 0.0
        return result

    def calibrate(self, probability: float, category: str = "default", confidence: float = 0.7) -> float:
        """
        V10 FIX #10: Adjust probability based on LEARNED historical calibration, not hardcoded
        Previously: hardcoded politics 70%→64% etc
        Now: learn from actual outcomes per category and prob bucket
        """
        relevant = [p for p in self.points if p.actual_outcome is not None and p.category == category]
        
        if len(relevant) < 20:
            # Not enough data - return as-is, no hardcoded adjustment
            return probability

        similar = [p for p in relevant if abs(p.forecast_prob - probability) <= 0.05]
        if len(similar) >= 10:
            actual_rate = sum(p.actual_outcome for p in similar) / len(similar)
            adjustment = actual_rate - probability
            
            # V10 FIX #10: Learn and store adjustment
            bucket_key = f"{int(probability*10)*10}-{(int(probability*10)+1)*10}%"
            if category not in self.learned_adjustments:
                self.learned_adjustments[category] = {}
            self.learned_adjustments[category][bucket_key] = {
                "forecast_prob": probability,
                "actual_rate": actual_rate,
                "adjustment": adjustment,
                "sample_size": len(similar),
                "learned_at": datetime.now(timezone.utc).isoformat()
            }
            if category not in self.category_adjustments:
                self.category_adjustments[category] = {}
            self.category_adjustments[category][bucket_key] = actual_rate
            self.category_adjustments[category]["adjustment"] = adjustment
            self.category_adjustments[category]["learned"] = True
            self.category_adjustments[category]["sample_size"] = len(relevant)
            
            weight_historical = min(0.5, len(similar) / 100)
            calibrated = probability * (1 - weight_historical) + actual_rate * weight_historical
            logger.info(f"Calibrated {category} {bucket_key} prob {probability:.3f} -> {calibrated:.3f} based on {len(similar)} similar (actual {actual_rate:.3f} adj {adjustment:+.3f}) - LEARNED V10 FIX #10")
            return max(0.01, min(0.99, calibrated))
        else:
            if len(relevant) >= 50:
                overall_forecast = sum(p.forecast_prob for p in relevant) / len(relevant)
                overall_actual = sum(p.actual_outcome for p in relevant) / len(relevant)
                category_adj = overall_actual - overall_forecast
                calibrated = probability + category_adj * 0.3
                logger.info(f"Calibrated {category} prob {probability:.3f} -> {calibrated:.3f} using category-level adj {category_adj:+.3f} from {len(relevant)} samples - LEARNED V10")
                return max(0.01, min(0.99, calibrated))
        
        return probability

    def learn_adjustments_from_history(self) -> Dict[str, Dict]:
        """
        V10 FIX #10: Explicit learning method - compute adjustments from all history
        """
        categories = set(p.category for p in self.points if p.actual_outcome is not None)
        learned = {}
        
        for category in categories:
            relevant = [p for p in self.points if p.actual_outcome is not None and p.category == category]
            if len(relevant) < 20:
                continue
            
            for bucket_start in [i*0.1 for i in range(10)]:
                bucket_low = bucket_start
                bucket_high = bucket_start + 0.1
                bucket_points = [p for p in relevant if bucket_low <= p.forecast_prob < bucket_high]
                
                if len(bucket_points) >= 5:
                    avg_forecast = sum(p.forecast_prob for p in bucket_points) / len(bucket_points)
                    avg_actual = sum(p.actual_outcome for p in bucket_points) / len(bucket_points)
                    adjustment = avg_actual - avg_forecast
                    
                    bucket_key = f"{int(bucket_start*100)}-{int(bucket_high*100)}%"
                    if category not in learned:
                        learned[category] = {}
                    learned[category][bucket_key] = {
                        "avg_forecast": avg_forecast,
                        "avg_actual": avg_actual,
                        "adjustment": adjustment,
                        "sample_size": len(bucket_points)
                    }
        
        self.learned_adjustments = learned
        logger.info(f"V10 FIX #10 Learned calibration adjustments: {len(learned)} categories, total buckets {sum(len(v) for v in learned.values())} - no hardcoded values")
        return learned

    def get_stats(self) -> Dict:
        """Get calibration stats"""
        total = len(self.points)
        resolved = len([p for p in self.points if p.actual_outcome is not None])
        return {
            "total_forecasts": total,
            "resolved": resolved,
            "pending": total - resolved,
            "brier_score": self.calculate_brier_score(),
            "log_loss": self.calculate_log_loss(),
            "brier_by_category": {cat: self.calculate_brier_score(cat) for cat in set(p.category for p in self.points)},
            "calibration_curve": self.calibration_curve(),
            "accuracy_by_confidence": self.accuracy_by_confidence(),
            "needs_more_data": resolved < 50
        }

    def is_degrading(self) -> bool:
        """Check if model calibration is degrading - trigger kill switch if so"""
        if len([p for p in self.points if p.actual_outcome is not None]) < 50:
            return False
        brier = self.calculate_brier_score()
        # If Brier > 0.3, poorly calibrated
        if brier > 0.3:
            logger.warning(f"Calibration degrading: Brier {brier:.3f} > 0.3")
            return True
        return False
