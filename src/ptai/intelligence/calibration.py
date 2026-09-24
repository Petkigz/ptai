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
    # Which venue hosts this market, and which trade it backs. Without these a
    # settlement pass cannot know whom to ask, and an outcome cannot be routed
    # back to the position it settles.
    venue_id: Optional[str] = None
    trade_id: Optional[int] = None


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
        # Reload persisted forecasts. Without this, calibration restarted from
        # zero on every process launch: `points` is in-memory, the engine is
        # constructed per agent, and the cyclic agent builds one per run.
        self.load_from_storage()

    # -- persistence ---------------------------------------------------------

    def load_from_storage(self) -> int:
        """
        Load previously recorded forecasts and resolutions into memory.

        Called from __init__. Returns the number of points loaded.

        Before this existed, every forecast lived in a list on an object that
        was thrown away at the end of the process, so the agent began each run
        with an empty calibration history - it could not accumulate the 50
        resolved forecasts `is_degrading()` needs, no matter how long it ran.
        """
        if not self.storage:
            return 0
        try:
            rows = self.storage.conn.execute(
                "SELECT id, market_id, question, forecast_prob, confidence, "
                "market_price, category, timestamp, actual_outcome, resolved_at, "
                "venue_id, trade_id "
                "FROM calibration ORDER BY timestamp"
            ).fetchall()
        except Exception as e:
            logger.error(
                f"CALIBRATION LOAD FAILED: {type(e).__name__}: {e}. The agent "
                f"would otherwise run on unlearned priors while appearing to "
                f"track calibration.")
            return 0

        # Deduplicate: CalibrationDB also restores points from a JSON file, and
        # loading the same forecast twice would double-count it in the Brier
        # score. Keyed on forecast_id, which is the primary key of both.
        existing = {p.forecast_id for p in self.points}

        loaded = 0
        for row in rows:
            if row["id"] in existing:
                continue
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (TypeError, ValueError):
                timestamp = datetime.now(timezone.utc)
            resolved_at = None
            if row["resolved_at"]:
                try:
                    resolved_at = datetime.fromisoformat(row["resolved_at"])
                except (TypeError, ValueError):
                    resolved_at = None
            self.points.append(CalibrationPoint(
                forecast_id=row["id"],
                market_id=row["market_id"],
                question=row["question"] or "",
                forecast_prob=float(row["forecast_prob"]),
                confidence=float(row["confidence"] or 0.0),
                market_price=float(row["market_price"] or 0.0),
                category=row["category"] or "default",
                timestamp=timestamp,
                actual_outcome=(
                    float(row["actual_outcome"])
                    if row["actual_outcome"] is not None else None
                ),
                resolved_at=resolved_at,
                venue_id=row["venue_id"],
                trade_id=row["trade_id"],
            ))
            loaded += 1

        resolved = sum(1 for p in self.points if p.actual_outcome is not None)
        logger.info(
            f"Calibration history loaded: {loaded} forecasts, {resolved} resolved")
        return loaded

    def pending_forecasts(self, since: Optional[datetime] = None) -> List[CalibrationPoint]:
        """Forecasts with no outcome yet - the settlement work queue."""
        pending = [p for p in self.points if p.actual_outcome is None]
        if since is not None:
            pending = [p for p in pending if p.timestamp >= since]
        return pending

    def has_forecast_for(self, market_id: str) -> bool:
        return any(p.market_id == market_id and p.actual_outcome is None
                   for p in self.points)

    def record_forecast(self, market_id: str, question: str, forecast_prob: float, confidence: float, market_price: float, category: str = "default", venue_id: str = None, trade_id: int = None) -> str:
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
            timestamp=datetime.now(timezone.utc),
            venue_id=venue_id,
            trade_id=trade_id,
        )
        self.points.append(point)
        
        # Persist. A recorded forecast that does not survive the process is not
        # recorded, and the agent would keep trading on unlearned priors while
        # the log said "Calibration recorded". This raises rather than warns:
        # silent loss of learning data is worse than a loud failure, because the
        # agent continues as if it had learned.
        if self.storage:
            try:
                self.storage.conn.execute(
                    "INSERT INTO calibration (id, market_id, question, forecast_prob, confidence, market_price, category, timestamp, venue_id, trade_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (forecast_id, market_id, question, forecast_prob, confidence,
                     market_price, category, point.timestamp.isoformat(),
                     venue_id, trade_id)
                )
                self.storage.conn.commit()
            except Exception as e:
                # Roll the in-memory point back so the two views cannot diverge.
                self.points = [p for p in self.points if p.forecast_id != forecast_id]
                logger.error(
                    f"CALIBRATION FORECAST NOT PERSISTED for {market_id}: "
                    f"{type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Could not persist forecast for {market_id}: {e}"
                ) from e

        logger.info(f"Calibration recorded {forecast_id}: {question[:50]} forecast={forecast_prob:.3f} market={market_price:.3f}")
        return forecast_id

    def record_resolution(self, forecast_id: str, actual_outcome: float) -> bool:
        """
        Record actual outcome when market resolves. Returns True if applied.

        Persists, and reports when there is nothing to update instead of
        silently succeeding - a resolution that lands nowhere is how the
        resolved count stays at zero forever.
        """
        target = None
        for p in self.points:
            if p.forecast_id == forecast_id:
                target = p
                break

        if target is None:
            logger.warning(
                f"Calibration resolution for unknown forecast_id {forecast_id} "
                f"(outcome {actual_outcome}) - nothing to update")
            return False

        if target.actual_outcome is not None:
            logger.info(
                f"Calibration forecast {forecast_id} already resolved as "
                f"{target.actual_outcome}; ignoring {actual_outcome}")
            return False

        target.actual_outcome = actual_outcome
        target.resolved_at = datetime.now(timezone.utc)

        if self.storage:
            try:
                self.storage.conn.execute(
                    "UPDATE calibration SET actual_outcome = ?, resolved_at = ? WHERE id = ?",
                    (actual_outcome, target.resolved_at.isoformat(), forecast_id)
                )
                self.storage.conn.commit()
            except Exception as e:
                target.actual_outcome = None
                target.resolved_at = None
                logger.error(
                    f"CALIBRATION RESOLUTION NOT PERSISTED for {forecast_id}: "
                    f"{type(e).__name__}: {e}")
                raise RuntimeError(
                    f"Could not persist resolution for {forecast_id}: {e}"
                ) from e

        logger.info(
            f"Calibration resolved {forecast_id}: forecast={target.forecast_prob:.3f} "
            f"actual={actual_outcome} market={target.market_id}")
        self._refresh_learned_adjustments()
        return True

    def resolved_count(self) -> int:
        return sum(1 for p in self.points if p.actual_outcome is not None)

    def _refresh_learned_adjustments(self):
        """
        Recompute category adjustments from resolved data.

        `category_adjustments` existed but nothing ever wrote to it, so the
        adjustments the agent applied to fair value were permanently 0.0 while
        appearing to be learned.
        """
        by_category: Dict[str, List[CalibrationPoint]] = {}
        for p in self.points:
            if p.actual_outcome is None:
                continue
            by_category.setdefault(p.category, []).append(p)

        for category, pts in by_category.items():
            n = len(pts)
            mean_forecast = sum(p.forecast_prob for p in pts) / n
            mean_actual = sum(p.actual_outcome for p in pts) / n
            # Positive adjustment = the model is under-confident in this
            # category and its probabilities should be nudged up.
            self.category_adjustments[category] = {
                "adjustment": mean_actual - mean_forecast,
                "learned": True,
                "sample_size": n,
            }

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
