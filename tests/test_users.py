"""Tests for users - multi-tenant"""
import tempfile
from pathlib import Path
from src.ptai.users.manager import UserManager

class TestUserManager:
    def test_create_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UserManager(db_path=str(Path(tmpdir) / "users.db"))
            user = manager.create_user(email="test@example.com", name="Test User", bankroll=50.0, plan="free")
            assert user.email == "test@example.com"
            assert user.bankroll == 50.0
            manager.close()
    
    def test_list_users(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UserManager(db_path=str(Path(tmpdir) / "users.db"))
            manager.create_user(email="test1@example.com", name="User1")
            manager.create_user(email="test2@example.com", name="User2")
            users = manager.list_users()
            assert len(users) == 2
            manager.close()
    
    def test_get_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UserManager(db_path=str(Path(tmpdir) / "users.db"))
            user = manager.create_user(email="test@example.com", name="Test")
            retrieved = manager.get_user(user.id)
            assert retrieved.id == user.id
            manager.close()
    
    def test_user_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UserManager(db_path=str(Path(tmpdir) / "users.db"))
            user = manager.create_user(email="test@example.com", name="Test")
            paths = manager.get_user_paths(user.id)
            assert "db" in paths
            assert "vault" in paths
            manager.close()
