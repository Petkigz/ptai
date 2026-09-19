"""Tests for storage - bankroll, trades, scans"""
import tempfile
from pathlib import Path
from src.ptai.storage.db import Storage

class TestStorage:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            assert storage.db_path.exists()
            storage.close()
    
    def test_bankroll(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.set_bankroll(100.0)
            assert storage.get_bankroll() == 100.0
            storage.close()
    
    def test_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.set_state("test_key", "test_value")
            assert storage.get_state("test_key") == "test_value"
            storage.close()
    
    def test_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.log_trade({
                "market_id": "test_market",
                "market_question": "Will BTC go up?",
                "edge": 0.15,
                "position_size_usd": 3.0,
                "status": "executed"
            })
            trades = storage.get_recent_trades(10)
            assert len(trades) == 1
            assert trades[0]["market_id"] == "test_market"
            storage.close()
    
    def test_performance_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.set_bankroll(60.0)
            perf = storage.get_performance_summary()
            assert "bankroll" in perf
            assert "total_trades" in perf
            storage.close()
    
    def test_self_preservation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.set_bankroll(50.0)
            sp = storage.check_self_preservation()
            assert "bankroll" in sp
            assert "should_shutdown" in sp
            storage.close()
    
    def test_market_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(db_path=str(Path(tmpdir) / "test.db"))
            storage.log_scan(
                markets_scanned=500,
                opportunities_found=10,
                avg_edge=0.12,
                execution_time=1.5,
                bankroll=50.0
            )
            # Check via direct query
            cur = storage.conn.execute("SELECT * FROM market_scans")
            scans = cur.fetchall()
            assert len(scans) == 1
            storage.close()
