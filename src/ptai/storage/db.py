"""
Local SQLite storage - fully offline, no cloud
Tracks bankroll, trades, market analysis, agent performance
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os

from loguru import logger

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_question TEXT,
    event_slug TEXT,
    outcome TEXT,
    side TEXT,
    market_price REAL,
    fair_value REAL,
    edge REAL,
    kelly_fraction REAL,
    position_size_usd REAL,
    position_size_pct REAL,
    confidence REAL,
    status TEXT DEFAULT 'pending',
    tx_hash TEXT,
    order_id TEXT,
    pnl REAL DEFAULT 0,
    resolved BOOLEAN DEFAULT 0,
    notes TEXT,
    venue_id TEXT
);

CREATE TABLE IF NOT EXISTS market_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    markets_scanned INTEGER,
    opportunities_found INTEGER,
    avg_edge REAL,
    execution_time_seconds REAL,
    bankroll REAL
);

CREATE TABLE IF NOT EXISTS market_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    question TEXT,
    market_price REAL,
    fair_value REAL,
    edge REAL,
    sentiment_score REAL,
    sentiment_summary TEXT,
    reasoning TEXT,
    confidence REAL,
    should_trade BOOLEAN,
    raw_data TEXT
);

CREATE TABLE IF NOT EXISTS bankroll_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    bankroll REAL NOT NULL,
    daily_pnl REAL,
    total_pnl REAL,
    open_positions INTEGER,
    daily_cost REAL
);

CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS research_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT,
    tool TEXT,
    query TEXT,
    result TEXT
);

-- OrderManager.create_order has always inserted into this table, which was
-- never created. Every insert raised "no such table: orders" into a warning, so
-- no order was ever persisted: the platform had no record of what it had
-- submitted, which makes reconciliation, duplicate-order detection and any
-- audit of live trading impossible.
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    token_id TEXT,
    side TEXT,
    max_price REAL,
    max_spend REAL,
    status TEXT,
    created_at TEXT NOT NULL,
    venue_id TEXT,
    updated_at TEXT,
    amount_usd REAL,
    avg_price REAL,
    raw_response TEXT
);

-- The calibration table was referenced by CalibrationEngine.record_forecast
-- but never created, so every forecast insert raised "no such table" into a
-- warning and the forecast lived only in an in-memory list on an object built
-- fresh each cycle. Nothing persisted, nothing could ever be resolved, the
-- resolved count stayed 0, and is_degrading() - guarded by resolved >= 50 -
-- could never fire. The agent was structurally incapable of learning from its
-- own outcomes.
CREATE TABLE IF NOT EXISTS calibration (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    question TEXT,
    forecast_prob REAL NOT NULL,
    confidence REAL,
    market_price REAL,
    category TEXT DEFAULT 'default',
    timestamp TEXT NOT NULL,
    actual_outcome REAL,
    resolved_at TEXT,
    venue_id TEXT,
    trade_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_calibration_unresolved
    ON calibration (actual_outcome, market_id);

-- TradeOutcomeTracker kept its outcomes in a plain in-memory list, so venue,
-- strategy and category performance vanished on every restart: the tracker was
-- rebuilt empty each run and the agent re-estimated allocation from nothing.
-- Learning has to survive a restart to be learning.
CREATE TABLE IF NOT EXISTS trade_outcomes (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    venue_id TEXT,
    strategy TEXT,
    category TEXT,
    forecast_prob REAL,
    market_price REAL,
    edge REAL,
    side TEXT,
    amount_usd REAL,
    actual_outcome REAL,
    pnl REAL DEFAULT 0,
    resolved_at TEXT,
    brier_score REAL,
    was_correct INTEGER,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_outcomes_venue
    ON trade_outcomes (venue_id, strategy, category);
"""

class Storage:
    def __init__(self, db_path: str = "./data/ptai.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"Storage initialized at {self.db_path}")

    # Columns added after the first release. CREATE TABLE IF NOT EXISTS silently
    # does nothing on an existing database, so an older data/ptai.db keeps its
    # original shape and every insert naming a new column fails.
    _MIGRATIONS = (
        ("trades", "venue_id", "TEXT"),
    )

    def _migrate(self):
        for table, column, sqltype in self._MIGRATIONS:
            try:
                cols = {r[1] for r in self.conn.execute(
                    f"PRAGMA table_info({table})").fetchall()}
                if not cols:
                    continue  # table not created yet; schema will include it
                if column not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sqltype}")
                    logger.info(f"Migrated: added {table}.{column}")
            except Exception as e:
                logger.error(f"Migration {table}.{column} failed: {e}")

    def _init_db(self):
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()
        self._migrate()
        # Initialize bankroll if not exists
        if not self.get_state("bankroll"):
            self.set_state("bankroll", "50.0")
        if not self.get_state("initial_bankroll"):
            self.set_state("initial_bankroll", "50.0")
        if not self.get_state("total_trades"):
            self.set_state("total_trades", "0")
        if not self.get_state("unprofitable_days"):
            self.set_state("unprofitable_days", "0")

    def set_state(self, key: str, value: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now)
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM agent_state WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def get_bankroll(self) -> float:
        v = self.get_state("bankroll")
        return float(v) if v else 50.0

    def set_bankroll(self, amount: float):
        self.set_state("bankroll", str(amount))
        # Also log to history
        now = datetime.now(timezone.utc).isoformat()
        initial = float(self.get_state("initial_bankroll") or 50.0)
        total_pnl = amount - initial
        # Get today's pnl
        cur = self.conn.execute(
            "SELECT bankroll FROM bankroll_history ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        prev = row["bankroll"] if row else initial
        daily_pnl = amount - prev if row else 0
        self.conn.execute(
            "INSERT INTO bankroll_history (timestamp, bankroll, daily_pnl, total_pnl, open_positions, daily_cost) VALUES (?, ?, ?, ?, ?, ?)",
            (now, amount, daily_pnl, total_pnl, self.count_open_positions(), 0)
        )
        self.conn.commit()

    def log_trade(self, trade: Dict[str, Any]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute("""
            INSERT INTO trades (timestamp, market_id, market_question, event_slug, outcome, side, market_price, fair_value, edge, kelly_fraction, position_size_usd, position_size_pct, confidence, status, notes, venue_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            trade.get("market_id"),
            trade.get("market_question"),
            trade.get("event_slug"),
            trade.get("outcome"),
            trade.get("side"),
            trade.get("market_price"),
            trade.get("fair_value"),
            trade.get("edge"),
            trade.get("kelly_fraction"),
            trade.get("position_size_usd"),
            trade.get("position_size_pct"),
            trade.get("confidence"),
            trade.get("status", "pending"),
            trade.get("notes", ""),
            trade.get("venue_id"),
        ))
        self.conn.commit()
        # Update total trades
        total = int(self.get_state("total_trades") or 0) + 1
        self.set_state("total_trades", str(total))
        return cur.lastrowid

    def log_market_analysis(self, analysis: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO market_analysis (timestamp, market_id, question, market_price, fair_value, edge, sentiment_score, sentiment_summary, reasoning, confidence, should_trade, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            analysis.get("market_id"),
            analysis.get("question"),
            analysis.get("market_price"),
            analysis.get("fair_value"),
            analysis.get("edge"),
            analysis.get("sentiment_score"),
            analysis.get("sentiment_summary"),
            analysis.get("reasoning"),
            analysis.get("confidence"),
            analysis.get("should_trade", False),
            json.dumps(analysis.get("raw_data", {}))
        ))
        self.conn.commit()

    def log_scan(self, markets_scanned: int, opportunities_found: int, avg_edge: float, execution_time: float, bankroll: float):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO market_scans (timestamp, markets_scanned, opportunities_found, avg_edge, execution_time_seconds, bankroll)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now, markets_scanned, opportunities_found, avg_edge, execution_time, bankroll))
        self.conn.commit()

    def log_research(self, market_id: str, tool: str, query: str, result: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT INTO research_logs (timestamp, market_id, tool, query, result)
            VALUES (?, ?, ?, ?, ?)
        """, (now, market_id, tool, query, result[:5000]))
        self.conn.commit()

    def resolve_trade(self, trade_id: int, outcome: float, pnl: float,
                      notes: str = "") -> bool:
        """
        Close a trade: mark it resolved, bank the P&L, log the new bankroll.

        There was no way to close a trade at all. `trades.resolved` was written
        as 0 and never updated, `get_open_positions()` therefore returned every
        trade ever logged, and `get_performance_summary()` computed win rate
        over `WHERE resolved=1` - an empty set forever, so win_rate was
        permanently 0. The bankroll could only ever go down (when a trade was
        opened) and never up.

        Returns False if the trade does not exist or was already resolved, so a
        double settlement cannot pay out twice.
        """
        cur = self.conn.execute(
            "SELECT id, resolved, position_size_usd FROM trades WHERE id = ?",
            (trade_id,))
        row = cur.fetchone()
        if row is None:
            logger.warning(f"resolve_trade: no trade with id {trade_id}")
            return False
        if row["resolved"]:
            logger.warning(f"resolve_trade: trade {trade_id} already resolved, ignoring")
            return False

        self.conn.execute(
            "UPDATE trades SET resolved = 1, outcome = ?, pnl = ?, status = 'settled', notes = ? WHERE id = ?",
            (outcome, pnl, notes, trade_id))
        self.conn.commit()

        self.set_bankroll(self.get_bankroll() + pnl)
        logger.info(
            f"Trade {trade_id} settled: outcome={outcome} pnl=${pnl:+.2f} "
            f"bankroll=${self.get_bankroll():.2f}")
        return True

    def get_unresolved_trades(self, limit: int = 200) -> List[Dict]:
        """Open trades awaiting settlement - the settlement work queue."""
        cur = self.conn.execute(
            "SELECT * FROM trades WHERE resolved = 0 ORDER BY timestamp LIMIT ?",
            (limit,))
        return [dict(r) for r in cur.fetchall()]

    def count_open_positions(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) as c FROM trades WHERE resolved=0 AND status IN ('executed','pending','open')")
        row = cur.fetchone()
        return row["c"] if row else 0

    def get_open_positions(self) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM trades WHERE resolved=0 ORDER BY timestamp DESC")
        return [dict(r) for r in cur.fetchall()]

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        cur = self.conn.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def get_performance_summary(self) -> Dict:
        cur = self.conn.execute("SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 30")
        history = [dict(r) for r in cur.fetchall()]
        bankroll = self.get_bankroll()
        initial = float(self.get_state("initial_bankroll") or 50.0)
        total_pnl = bankroll - initial
        total_trades = int(self.get_state("total_trades") or 0)
        # Calculate win rate from resolved trades
        cur2 = self.conn.execute("SELECT COUNT(*) as total, SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins, AVG(pnl) as avg_pnl FROM trades WHERE resolved=1")
        row = cur2.fetchone()
        win_rate = (row["wins"] / row["total"] * 100) if row and row["total"] else 0
        return {
            "bankroll": bankroll,
            "initial_bankroll": initial,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / initial * 100) if initial else 0,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": row["avg_pnl"] if row else 0,
            "history": history,
            "open_positions": self.count_open_positions()
        }

    def check_self_preservation(self, daily_cost: float = 5.0, max_unprofitable_days: int = 3) -> Dict:
        """
        Capital PRESERVATION checks - the reasons trading should stop.

        The three shutdown conditions here are risk controls and stay:
          * the bankroll is depleted,
          * the drawdown exceeds the limit,
          * the account has lost money for N consecutive days.

        `daily_cost` is NOT a target. It used to be framed as "earn enough to
        pay for yourself or shut down", which made a $50 account responsible for
        covering a server bill. Whether the profit covers running costs is the
        operator's decision about what to do with the money, not a goal for the
        trading engine, and treating it as one pushed the agent to trade when it
        should have done nothing. It is now reported as an advisory comparison
        and drives no decision.
        """
        summary = self.get_performance_summary()
        bankroll = summary["bankroll"]
        initial = summary["initial_bankroll"]
        total_pnl = summary["total_pnl"]

        # Days since start (approx from bankroll_history)
        cur = self.conn.execute("SELECT COUNT(DISTINCT DATE(timestamp)) as days FROM bankroll_history")
        days = cur.fetchone()["days"] or 1
        if days == 0:
            days = 1

        # Advisory only: how the realised P&L compares with a running-cost
        # budget the operator supplied. Nothing downstream branches on it.
        required_profit = days * daily_cost
        is_profitable_enough = total_pnl >= required_profit

        # Check unprofitable days
        cur2 = self.conn.execute("""
            SELECT DATE(timestamp) as day, SUM(daily_pnl) as pnl FROM bankroll_history
            GROUP BY DATE(timestamp) ORDER BY day DESC LIMIT ?
        """, (max_unprofitable_days,))
        recent_days = cur2.fetchall()
        unprofitable_streak = 0
        for r in recent_days:
            if r["pnl"] is not None and r["pnl"] < 0:
                unprofitable_streak += 1
            else:
                break

        should_shutdown = False
        reason = None
        if bankroll <= 0:
            should_shutdown = True
            reason = "Bankroll depleted"
        elif unprofitable_streak >= max_unprofitable_days and days > 2:
            should_shutdown = True
            reason = f"Unprofitable for {unprofitable_streak} consecutive days, required {required_profit:.2f} but got {total_pnl:.2f}"
        elif summary["total_pnl_pct"] <= -30:  # 30% drawdown
            should_shutdown = True
            reason = f"Max drawdown exceeded: {summary['total_pnl_pct']:.1f}%"

        return {
            "bankroll": bankroll,
            "total_pnl": total_pnl,
            "days_active": days,
            "required_profit": required_profit,
            "is_profitable_enough": is_profitable_enough,
            "unprofitable_streak": unprofitable_streak,
            "should_shutdown": should_shutdown,
            "shutdown_reason": reason,
            "summary": summary
        }

    def close(self):
        self.conn.close()
