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
        # Historical calibration adjustments per category
        # e.g. Politics 70% forecasts -> historically 64% => adjustment -6%
        self.category_adjustments: Dict[str, Dict] = {
            "politics": {"70%": 0.64, "adjustment": -0.06},
            "crypto": {"70%": 0.72, "adjustment": 0.02},
            "sports": {"70%": 0.69, "adjustment": -0.01},
            "default": {"adjustment": 0.0}
        }

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
        Adjust probability based on historical calibration.
        If Politics 70% forecasts historically 64%, adjust down.
        """
        # Simple implementation: look up historical adjustment
        # Production: use isotonic regression or Platt scaling learned from data
        
        # If we have enough data for this category, calculate adjustment
        relevant = [p for p in self.points if p.actual_outcome is not None and p.category == category]
        if len(relevant) < 20:
            # Not enough data, use default adjustment or no adjustment
            return probability

        # Find similar forecasts (±5%) and see actual win rate
        similar = [p for p in relevant if abs(p.forecast_prob - probability) <= 0.05]
        if len(similar) >= 10:
            actual_rate = sum(p.actual_outcome for p in similar) / len(similar)
            # Blend forecast with historical actual rate
            # If forecast says 70% but historically similar forecasts win 64%, calibrated = 0.7*0.7 + 0.64*0.3 etc
            # More weight to historical if we have more data
            weight_historical = min(0.5, len(similar) / 100)
            calibrated = probability * (1 - weight_historical) + actual_rate * weight_historical
            logger.info(f"Calibrated {category} prob {probability:.3f} -> {calibrated:.3f} based on {len(similar)} similar (actual {actual_rate:.3f})")
            return max(0.01, min(0.99, calibrated))
        
        return probability

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
