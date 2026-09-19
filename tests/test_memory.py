"""Tests for memory - learning, calibration"""
import tempfile
from pathlib import Path
from src.ptai.memory.memory import Memory

class TestMemory:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            assert memory is not None
            memory.close()
    
    def test_remember(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            memory.remember("Test insight about BTC", type="insight", importance=0.8)
            insights = memory.get_insights(limit=10)
            assert len(insights) >= 1
            memory.close()
    
    def test_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            memory.remember("BTC is going up", type="insight")
            memory.remember("ETH is stable", type="insight")
            results = memory.recall(query="BTC", limit=10)
            assert len(results) >= 1
            memory.close()
    
    def test_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            # Add calibration
            memory.track_calibration(market_id="test1", question="Test?", predicted=0.7, actual=1.0, confidence=0.8, edge=0.1)
            stats = memory.get_calibration_stats()
            assert "count" in stats
            memory.close()
    
    def test_insights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = Memory(db_path=str(Path(tmpdir) / "memory.db"))
            memory.remember("Insight 1", type="insight", importance=0.9)
            memory.remember("Insight 2", type="insight", importance=0.5)
            insights = memory.get_insights(limit=10)
            assert len(insights) == 2
            memory.close()
