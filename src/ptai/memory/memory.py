"""
Memory - Learning, understanding, intelligence for AI teammates
Premium product: bots remember past work, learn from outcomes, improve calibration
Local-only, SQLite + JSON, no cloud
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
from loguru import logger

@dataclass
class MemoryEntry:
    id: str
    type: str  # task, insight, trade, calibration, research
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: str = None
    importance: float = 0.5  # 0-1
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.id:
            self.id = hashlib.md5(f"{self.type}:{self.content}:{self.created_at}".encode()).hexdigest()[:12]

class Memory:
    """
    Premium Memory for AI Teammates - learns and understands
    - Remembers past tasks, trades, insights
    - Tracks calibration (Brier score)
    - Few-shot learning from resolved markets
    - Semantic search over memories
    """
    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"Memory initialized at {db_path}")
    
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                metadata TEXT,
                importance REAL,
                created_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibrations (
                id TEXT PRIMARY KEY,
                market_id TEXT,
                question TEXT,
                predicted REAL,
                actual REAL,
                brier REAL,
                confidence REAL,
                edge REAL,
                created_at TEXT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at DESC)")
        self.conn.commit()
    
    def remember(self, content: str, type: str = "insight", metadata: Dict = None, importance: float = 0.5) -> str:
        """Remember something for future learning"""
        entry = MemoryEntry(
            id="",
            type=type,
            content=content[:2000],  # limit
            metadata=metadata or {},
            importance=importance
        )
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO memories (id, type, content, metadata, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (entry.id, entry.type, entry.content, json.dumps(entry.metadata), entry.importance, entry.created_at)
            )
            self.conn.commit()
            logger.debug(f"Memory remembered [{entry.type}]: {content[:80]}...")
            return entry.id
        except Exception as e:
            logger.warning(f"Memory remember failed: {e}")
            return ""
    
    def remember_task(self, task):
        """Remember a completed task"""
        try:
            content = f"Task {task.type} by {task.assigned_to}: {task.description} -> {str(task.result)[:500] if task.result else 'no result'} in {task.duration_seconds:.1f}s"
            self.remember(content, type="task", metadata={"task_id": task.id, "teammate": task.assigned_to, "duration": task.duration_seconds}, importance=0.6)
        except Exception as e:
            logger.debug(f"Remember task failed: {e}")
    
    def recall(self, query: str = "", type: str = None, limit: int = 10) -> List[MemoryEntry]:
        """Recall memories - semantic search (simple keyword for now, can add embeddings)"""
        try:
            if type:
                cur = self.conn.execute("SELECT * FROM memories WHERE type=? ORDER BY importance DESC, created_at DESC LIMIT ?", (type, limit))
            elif query:
                # Simple keyword search - for premium, use embeddings
                cur = self.conn.execute("SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC, created_at DESC LIMIT ?", (f"%{query}%", limit))
            else:
                cur = self.conn.execute("SELECT * FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?", (limit,))
            
            results = []
            for row in cur.fetchall():
                results.append(MemoryEntry(
                    id=row["id"],
                    type=row["type"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    importance=row["importance"],
                    created_at=row["created_at"]
                ))
            return results
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return []
    
    def track_calibration(self, market_id: str, question: str, predicted: float, actual: float, confidence: float, edge: float):
        """Track Brier score for calibration learning"""
        try:
            brier = (predicted - actual) ** 2
            id = hashlib.md5(f"{market_id}:{predicted}:{actual}".encode()).hexdigest()[:12]
            self.conn.execute(
                "INSERT OR REPLACE INTO calibrations (id, market_id, question, predicted, actual, brier, confidence, edge, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (id, market_id, question[:500], predicted, actual, brier, confidence, edge, datetime.now(timezone.utc).isoformat())
            )
            self.conn.commit()
            logger.info(f"Calibration tracked {market_id}: pred {predicted:.2f} actual {actual:.0f} brier {brier:.3f}")
            
            # Remember as insight
            self.remember(f"Calibration: {question[:80]} pred {predicted:.2f} actual {actual:.0f} brier {brier:.3f} edge {edge:.1%}", type="calibration", importance=0.8)
        except Exception as e:
            logger.warning(f"Track calibration failed: {e}")
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        try:
            cur = self.conn.execute("SELECT COUNT(*) as count, AVG(brier) as avg_brier, AVG(confidence) as avg_conf FROM calibrations")
            row = cur.fetchone()
            cur2 = self.conn.execute("SELECT * FROM calibrations ORDER BY created_at DESC LIMIT 10")
            recent = [dict(r) for r in cur2.fetchall()]
            return {
                "count": row["count"] or 0,
                "avg_brier": row["avg_brier"],
                "avg_confidence": row["avg_conf"],
                "recent": recent,
                "calibration": "Good if Brier <0.2, bad if >0.3. Lower is better."
            }
        except Exception as e:
            return {"error": str(e), "count": 0}
    
    def get_insights(self, limit: int = 20) -> List[MemoryEntry]:
        return self.recall(type="insight", limit=limit)
    
    def close(self):
        try:
            self.conn.close()
        except:
            pass
