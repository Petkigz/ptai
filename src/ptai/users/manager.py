"""
User Manager - Multi-user support for product
Each user has own bankroll, wallet, vault, memory, browser profile
Premium product feature
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
class User:
    id: str
    email: str
    name: str
    created_at: str
    bankroll: float = 50.0
    plan: str = "free"  # free, pro, premium
    is_active: bool = True
    last_login: Optional[str] = None

class UserManager:
    """
    Multi-user manager for product - each user isolated
    """
    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"UserManager initialized at {db_path}")
    
    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                name TEXT,
                bankroll REAL,
                plan TEXT,
                is_active INTEGER,
                created_at TEXT,
                last_login TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()
    
    def create_user(self, email: str, name: str = "", bankroll: float = 50.0, plan: str = "free") -> User:
        user_id = hashlib.md5(f"{email}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:12]
        user = User(
            id=user_id,
            email=email,
            name=name or email.split("@")[0],
            created_at=datetime.now(timezone.utc).isoformat(),
            bankroll=bankroll,
            plan=plan,
            is_active=True
        )
        try:
            self.conn.execute(
                "INSERT INTO users (id, email, name, bankroll, plan, is_active, created_at, last_login, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user.id, user.email, user.name, user.bankroll, user.plan, 1 if user.is_active else 0, user.created_at, None, json.dumps({}))
            )
            self.conn.commit()
            logger.success(f"User created {user.email} id {user.id}")
            
            # Create per-user directories
            Path(f"./data/users/{user.id}").mkdir(parents=True, exist_ok=True)
            Path(f"./browser/profiles/{user.id}").mkdir(parents=True, exist_ok=True)
            Path(f"./logs/users/{user.id}").mkdir(parents=True, exist_ok=True)
            
            return user
        except Exception as e:
            logger.error(f"Create user failed: {e}")
            # If exists, return existing
            return self.get_user_by_email(email)
    
    def get_user(self, user_id: str) -> Optional[User]:
        try:
            cur = self.conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            if row:
                return User(
                    id=row["id"],
                    email=row["email"],
                    name=row["name"],
                    bankroll=row["bankroll"],
                    plan=row["plan"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    last_login=row["last_login"]
                )
        except Exception as e:
            logger.warning(f"Get user failed: {e}")
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        try:
            cur = self.conn.execute("SELECT * FROM users WHERE email=?", (email,))
            row = cur.fetchone()
            if row:
                return User(
                    id=row["id"],
                    email=row["email"],
                    name=row["name"],
                    bankroll=row["bankroll"],
                    plan=row["plan"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    last_login=row["last_login"]
                )
        except Exception as e:
            logger.warning(f"Get user by email failed: {e}")
        return None
    
    def list_users(self) -> List[User]:
        try:
            cur = self.conn.execute("SELECT * FROM users ORDER BY created_at DESC")
            users = []
            for row in cur.fetchall():
                users.append(User(
                    id=row["id"],
                    email=row["email"],
                    name=row["name"],
                    bankroll=row["bankroll"],
                    plan=row["plan"],
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    last_login=row["last_login"]
                ))
            return users
        except Exception as e:
            logger.warning(f"List users failed: {e}")
            return []
    
    def get_user_paths(self, user_id: str) -> Dict[str, Path]:
        """Get per-user isolated paths for vault, memory, browser, etc"""
        base = Path(f"./data/users/{user_id}")
        return {
            "base": base,
            "db": base / "ptai.db",
            "vault": base / "vault.json",
            "memory": base / "memory.db",
            "browser_profile": Path(f"./browser/profiles/{user_id}"),
            "logs": Path(f"./logs/users/{user_id}/ptai.log")
        }
    
    def close(self):
        try:
            self.conn.close()
        except:
            pass
