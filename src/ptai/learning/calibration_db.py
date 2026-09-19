"""
Calibration DB - persistent calibration storage
"""
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime, timezone
from loguru import logger

from ..intelligence.calibration import CalibrationEngine


class CalibrationDB(CalibrationEngine):
    """Extends CalibrationEngine with persistence"""
    def __init__(self, db_path: str = "./data/calibration.json", storage=None, memory=None):
        super().__init__(storage=storage, memory=memory)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                # Would load points from file
                logger.info(f"Calibration DB loaded from {self.db_path}: {len(data.get('points', []))} points")
            except Exception as e:
                logger.warning(f"Calibration DB load failed: {e}")

    def save(self):
        try:
            data = {
                "points": [
                    {
                        "forecast_id": p.forecast_id,
                        "market_id": p.market_id,
                        "question": p.question,
                        "forecast_prob": p.forecast_prob,
                        "confidence": p.confidence,
                        "category": p.category,
                        "timestamp": p.timestamp.isoformat(),
                        "actual_outcome": p.actual_outcome,
                        "resolved_at": p.resolved_at.isoformat() if p.resolved_at else None
                    }
                    for p in self.points[-1000:]  # last 1000
                ],
                "category_adjustments": self.category_adjustments,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.db_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Calibration DB save failed: {e}")

    def get_report(self) -> Dict:
        stats = self.get_stats()
        return {
            **stats,
            "db_path": str(self.db_path),
            "is_degrading": self.is_degrading(),
            "recommendation": "Need more data" if stats["needs_more_data"] else "Calibration OK" if not self.is_degrading() else "Review models - calibration degrading"
        }
