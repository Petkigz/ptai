"""
Model Evaluation - evaluates forecasting models
"""
from typing import Dict, List
from loguru import logger


class ModelEvaluator:
    def __init__(self, calibration_db=None):
        self.calibration_db = calibration_db

    def evaluate(self) -> Dict:
        """Evaluate all models"""
        if not self.calibration_db:
            return {"message": "No calibration DB"}
        
        stats = self.calibration_db.get_stats()
        curve = stats.get("calibration_curve", [])
        
        # Check for overconfidence
        overconfident_buckets = [b for b in curve if b.get("overconfident")]
        
        return {
            "brier_score": stats.get("brier_score", 0.5),
            "log_loss": stats.get("log_loss", 0.7),
            "total_forecasts": stats.get("total_forecasts", 0),
            "resolved": stats.get("resolved", 0),
            "overconfident_buckets": overconfident_buckets,
            "is_overconfident": len(overconfident_buckets) > len(curve) / 2 if curve else False,
            "accuracy_by_confidence": stats.get("accuracy_by_confidence", {}),
            "recommendation": "Model overconfident - reduce confidence" if len(overconfident_buckets) > 2 else "Calibration OK"
        }
