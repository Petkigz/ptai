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
        """
        V9 FIX #5: Make calibration genuinely persistent - actually restore points
        Previously only logged count, didn't restore into memory - learning lost after restart
        """
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                points_data = data.get('points', [])
                # Actually restore points into memory
                from ..intelligence.calibration import CalibrationPoint
                restored = 0
                for pd in points_data:
                    try:
                        # Parse timestamp
                        ts_str = pd.get('timestamp')
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        else:
                            ts = datetime.now(timezone.utc)
                        resolved_at = None
                        if pd.get('resolved_at'):
                            resolved_at = datetime.fromisoformat(pd['resolved_at'].replace('Z', '+00:00'))
                        point = CalibrationPoint(
                            forecast_id=pd.get('forecast_id', pd.get('id', 'unknown')),
                            market_id=pd.get('market_id', 'unknown'),
                            question=pd.get('question', ''),
                            forecast_prob=float(pd.get('forecast_prob', 0.5)),
                            confidence=float(pd.get('confidence', 0.5)),
                            market_price=float(pd.get('market_price', pd.get('forecast_prob', 0.5))),
                            category=pd.get('category', 'default'),
                            timestamp=ts,
                            actual_outcome=pd.get('actual_outcome'),
                            resolved_at=resolved_at
                        )
                        self.points.append(point)
                        restored += 1
                    except Exception as e:
                        logger.debug(f"Failed to restore calibration point {pd.get('forecast_id')}: {e}")
                        continue
                # Restore category adjustments
                if 'category_adjustments' in data:
                    self.category_adjustments.update(data['category_adjustments'])
                logger.info(f"Calibration DB loaded from {self.db_path}: {len(points_data)} points in file, {restored} restored to memory - V9 FIX persistent")
            except Exception as e:
                logger.warning(f"Calibration DB load failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

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
