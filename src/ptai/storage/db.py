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

# Order states that will never change again. Anything else is still working and
# has capital behind it. Kept here rather than in the execution layer so the
# storage layer can answer "what is still open" without importing execution.
TERMINAL_ORDER_STATUSES = (
    "filled", "cancelled", "canceled", "rejected", "failed", "expired",
    "unmatched", "not_cancelled", "abandoned",
)

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
CREATE TABLE IF NOT EXISTS redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    asset TEXT,
    outcome TEXT,
    size REAL,
    value_usd REAL,
    transaction_id TEXT,
    redeemed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_redemptions_condition
    ON redemptions(condition_id);

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
    recorded_at TEXT NOT NULL,
    -- Realised costs and the mode the trade was executed in. NULL means NOT
    -- MEASURED, which the qualification gate must treat as a failure rather
    -- than as a zero-cost trade.
    fees_usd REAL,
    slippage_bps REAL,
    execution_quality REAL,
    data_mode TEXT
);

CREATE INDEX IF NOT EXISTS idx_trade_outcomes_venue
    ON trade_outcomes (venue_id, strategy, category);
"""

PAPER = "paper"
LIVE = "live"


def _execution_mode(value, status=None) -> str:
    """
    Normalise the execution mode of a trade to 'paper' or 'live'.

    Fail-closed in the direction that matters: anything unrecognised is treated
    as PAPER, because the only other option is to treat an unknown trade as one
    that moved real money - and that is what would corrupt the bankroll.
    """
    text = str(value or "").strip().lower()
    if text in ("paper", "simulated", "sim", "dry_run", "shadow"):
        return PAPER
    if text in ("live", "real", "executed"):
        return LIVE
    # No explicit mode, so fall back to the status the row was written with.
    # The writer sets status='paper' for a simulated execution and 'open' for a
    # real one, so 'open' means live - reading it as paper would be the worst
    # possible mistake in this function: real P&L would never reach the real
    # bankroll, and the account would look frozen.
    status_text = str(status or "").strip().lower()
    if status_text in ("open", "executed", "pending", "settled"):
        return LIVE
    return PAPER


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
        # Resting-order reconciliation. An order that rests in the book, or
        # fills in pieces, has state the trades table cannot express: how much
        # of it is still working, and how much of the requested size has
        # matched. Without these columns the agent could not tell a filled
        # order from one that is still sitting there, and the capital locked
        # behind it stayed invisible.
        ("orders", "token_id", "TEXT"),
        ("orders", "venue_id", "TEXT"),
        ("orders", "original_size", "REAL"),
        ("orders", "size_matched", "REAL"),
        ("orders", "matched_usd", "REAL"),
        ("orders", "limit_price", "REAL"),
        ("orders", "trade_id", "INTEGER"),
        ("orders", "updated_at", "TEXT"),
        ("orders", "last_synced_at", "TEXT"),
        ("orders", "terminal_reason", "TEXT"),
        ("orders", "raw_response", "TEXT"),
        ("orders", "requested_usd", "REAL"),
        # The forecast that JUSTIFIED the order, stored with the order.
        #
        # Without these, an order that rested and filled hours later produced a
        # position attributed to strategy "resting_order_fill" with edge 0 and
        # confidence 0. The trade had a thesis; the fill had amnesia. Learning
        # from the outcome then taught the agent about a strategy that never
        # chose the trade, which is worse than not learning at all.
        # A position row has to remember WHICH strategy chose it, and which
        # order it came from. Neither was stored: log_trade read only the columns
        # in its INSERT, so strategy/category/order_id were silently dropped -
        # which is why a delayed fill could not be attributed to anything.
        # What a trade COST and whether it was real. The qualification gate
        # claims to weigh fees, slippage, execution quality and a paper/live
        # split, and none of the four was recorded anywhere, so those inputs were
        # constant - and a constant that happens to equal the threshold passes it.
        ("trade_outcomes", "fees_usd", "REAL"),
        ("trade_outcomes", "slippage_bps", "REAL"),
        ("trade_outcomes", "execution_quality", "REAL"),
        ("trade_outcomes", "data_mode", "TEXT"),
        ("trades", "strategy", "TEXT"),
        ("trades", "category", "TEXT"),
        ("trades", "order_id", "TEXT"),
        ("orders", "fair_price", "REAL"),
        ("orders", "edge", "REAL"),
        ("orders", "confidence", "REAL"),
        ("orders", "strategy", "TEXT"),
        ("orders", "category", "TEXT"),
        ("orders", "data_mode", "TEXT"),
        # EXECUTION MODE, separate from data mode.
        #
        # `data_mode` describes the MARKET DATA (real prices and books from a
        # live venue) and `execution_mode` describes what actually happened to
        # the money (a simulated fill, or real capital). One paper exploration
        # trade against live market data reads `data_mode='live'` - which is
        # factually correct and tells you nothing about whether money moved.
        # The paper/live split in qualification was computed from `data_mode`,
        # so a simulated trade could be counted as a live outcome.
        ("trades", "execution_mode", "TEXT"),
        # The price actually paid per share, in the units the P&L needs.
        #
        # `market_price` meant the YES price when the opportunity was built and
        # the TOKEN price once a fill price existed, so the same column held two
        # different numbers depending on whether the order had filled - and
        # settlement, which needs the token price, was given whichever it was.
        # Both are now stored, explicitly.
        ("trades", "token_price_at_entry", "REAL"),
        ("trades", "yes_price_at_entry", "REAL"),
        ("trades", "fees_usd", "REAL"),
        ("trade_outcomes", "expected_net_ev", "REAL"),
        ("trade_outcomes", "expected_net_ev_pct", "REAL"),
        ("trade_outcomes", "execution_mode", "TEXT"),
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

    @staticmethod
    def _backfill_execution_mode(conn):
        """
        Label the rows written before `execution_mode` existed.

        The trade's own `notes` carry the execution status that was recorded at
        the time (`exec_status=paper_filled`), and `status` is 'paper' for a
        simulated position that has not settled yet. Anything else is treated as
        live, which is the conservative direction: a mislabelled paper trade
        would corrupt the real bankroll, whereas a mislabelled live trade only
        leaves paper statistics slightly pessimistic.

        Deliberately NOT derived from `data_mode` in the notes: that is the
        market data mode, and using it is the bug this column exists to fix.
        """
        try:
            cur = conn.execute(
                "UPDATE trades SET execution_mode = 'paper' "
                "WHERE execution_mode IS NULL AND "
                "(LOWER(COALESCE(status,'')) = 'paper' "
                " OR LOWER(COALESCE(notes,'')) LIKE '%exec_status=paper%')")
            backfilled = cur.rowcount
            conn.execute(
                "UPDATE trades SET execution_mode = 'live' "
                "WHERE execution_mode IS NULL")
            conn.commit()
            if backfilled:
                logger.info(
                    f"Backfilled execution_mode=paper on {backfilled} "
                    f"pre-existing trade row(s)")
        except Exception as e:
            logger.error(f"execution_mode backfill failed: {e}")

    def _init_db(self):
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()
        self._migrate()
        self._backfill_execution_mode(self.conn)
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

    def get_paper_bankroll(self) -> float:
        """
        The paper account's own bankroll - real capital's shadow, and never it.

        Kept in state beside the real bankroll so a simulation has somewhere to
        accumulate its results without touching the number that is true.
        """
        v = self.get_state("paper_bankroll")
        if v:
            return float(v)
        return float(self.get_state("initial_bankroll") or 50.0)

    def set_paper_bankroll(self, amount: float):
        self.set_state("paper_bankroll", str(amount))
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
            INSERT INTO trades (timestamp, market_id, market_question, event_slug, outcome, side, market_price, fair_value, edge, kelly_fraction, position_size_usd, position_size_pct, confidence, status, notes, venue_id, strategy, category, order_id, execution_mode, token_price_at_entry, yes_price_at_entry, fees_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            trade.get("strategy"),
            trade.get("category"),
            trade.get("order_id"),
            # Named, or silently dropped on the floor. This INSERT has already
            # lost columns that way once (strategy/category/order_id), and a
            # dropped execution_mode would put paper P&L back into the bankroll.
            _execution_mode(trade.get("execution_mode"), trade.get("status")),
            trade.get("token_price_at_entry"),
            trade.get("yes_price_at_entry"),
            trade.get("fees_usd"),
        ))
        self.conn.commit()
        # Update total trades
        total = int(self.get_state("total_trades") or 0) + 1
        self.set_state("total_trades", str(total))
        return cur.lastrowid

    # ------------------------------------------------------------------
    # orders - resting and partially filled
    # ------------------------------------------------------------------

    def upsert_order(self, order: Dict[str, Any]) -> bool:
        """
        Record or update an order by the VENUE's order id.

        Keyed on the venue id rather than a local uuid, because reconciliation
        asks the venue about an order by the id the venue knows. A local
        generated id cannot be looked up anywhere.
        """
        order_id = str(order.get("order_id") or order.get("id") or "")
        if not order_id:
            logger.error("upsert_order refused: no order id, so it can never be reconciled")
            return False
        now = datetime.now(timezone.utc).isoformat()
        existing = self.conn.execute(
            "SELECT id FROM orders WHERE id = ?", (order_id,)).fetchone()
        fields = {
            "market_id": order.get("market_id"),
            "token_id": order.get("token_id"),
            "side": order.get("side"),
            "max_price": order.get("limit_price", order.get("max_price")),
            "max_spend": order.get("requested_usd", order.get("max_spend")),
            "status": order.get("status"),
            "venue_id": order.get("venue_id"),
            "amount_usd": order.get("matched_usd"),
            "avg_price": order.get("limit_price", order.get("avg_price")),
            "raw_response": json.dumps(order.get("raw") or order.get("raw_response") or {},
                                       default=str)[:4000],
            "original_size": order.get("original_size"),
            "size_matched": order.get("size_matched"),
            "matched_usd": order.get("matched_usd"),
            "limit_price": order.get("limit_price", order.get("max_price")),
            "requested_usd": order.get("requested_usd", order.get("max_spend")),
            "trade_id": order.get("trade_id"),
            "last_synced_at": order.get("last_synced_at", now),
            "terminal_reason": order.get("terminal_reason"),
            # The forecast travels WITH the order, so a fill that arrives hours
            # later can still be attributed to the strategy that chose it.
            "fair_price": order.get("fair_price"),
            "edge": order.get("edge"),
            "confidence": order.get("confidence"),
            "strategy": order.get("strategy"),
            "category": order.get("category"),
            "data_mode": order.get("data_mode"),
        }
        try:
            if existing:
                # A partial update must not erase columns it does not mention.
                # Reconciliation re-writes an order with only its state, and
                # assigning the untouched fields from the caller's sparse dict
                # set market_id to NULL and aborted the whole write.
                changed = {k: v for k, v in fields.items() if v is not None}
                changed["updated_at"] = now
                assignments = ", ".join(f"{k} = ?" for k in changed)
                self.conn.execute(
                    f"UPDATE orders SET {assignments} WHERE id = ?",
                    (*changed.values(), order_id))
            else:
                fields["id"] = order_id
                fields["created_at"] = order.get("created_at", now)
                fields["updated_at"] = now
                columns = ", ".join(fields)
                placeholders = ", ".join("?" for _ in fields)
                self.conn.execute(
                    f"INSERT INTO orders ({columns}) VALUES ({placeholders})",
                    tuple(fields.values()))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"upsert_order({order_id}) failed: {type(e).__name__}: {e}")
            return False

    @staticmethod
    def _order_row(row) -> Dict[str, Any]:
        order = dict(row)
        order["order_id"] = order.get("id")
        order["limit_price"] = order.get("limit_price") or order.get("max_price") or 0.0
        order["requested_usd"] = order.get("requested_usd") or order.get("max_spend") or 0.0
        order["size_matched"] = order.get("size_matched") or 0.0
        order["matched_usd"] = order.get("matched_usd") or 0.0
        return order

    def get_open_orders(self, venue_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Orders that are not in a final state, from the local record.

        The authoritative copy is the venue's; this is what the agent believes,
        and reconciliation is the act of checking the two against each other.
        """
        try:
            if venue_id:
                rows = self.conn.execute(
                    "SELECT * FROM orders WHERE status NOT IN "
                    f"({','.join('?' for _ in TERMINAL_ORDER_STATUSES)}) "
                    "AND venue_id = ? ORDER BY created_at DESC",
                    (*TERMINAL_ORDER_STATUSES, venue_id)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM orders WHERE status NOT IN "
                    f"({','.join('?' for _ in TERMINAL_ORDER_STATUSES)}) "
                    "ORDER BY created_at DESC",
                    tuple(TERMINAL_ORDER_STATUSES)).fetchall()
        except Exception as e:
            logger.error(f"get_open_orders failed: {type(e).__name__}: {e}")
            return []
        return [self._order_row(r) for r in rows]

    def get_order_row(self, order_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM orders WHERE id = ?", (str(order_id),)).fetchone()
        return self._order_row(row) if row else None

    def resting_capital_usd(self, venue_id: Optional[str] = None) -> float:
        """
        Cash locked behind orders that have not filled.

        For each working order, the unfilled share of the request. Booked
        positions are NOT included here - their cost is already in the trades
        table - so there is no double count between the two.
        """
        total = 0.0
        for order in self.get_open_orders(venue_id=venue_id):
            requested = float(order.get("requested_usd") or 0.0)
            matched = float(order.get("matched_usd") or 0.0)
            if order.get("status") == "unconfirmed_send":
                # The send failed without a confirmation: the whole request may
                # be committed. Assume the worst until the venue says otherwise.
                total += requested
                continue
            total += max(0.0, requested - matched)
        return round(total, 6)

    def add_to_position(self, trade_id: int, add_usd: float,
                        add_price: float) -> bool:
        """
        Grow an existing position by a later fill of the same order.

        ONE position row per market is required, not one per fill: settlement
        resolves a market's open trade by looking it up by market id, so a
        second row for the same market would never be closed and its P&L would
        never be realised. Later fills therefore increase the size of the row
        that exists, and the entry price becomes the size-weighted average of
        what was actually paid.
        """
        cur = self.conn.execute(
            "SELECT position_size_usd, market_price, resolved FROM trades WHERE id = ?",
            (trade_id,))
        row = cur.fetchone()
        if row is None:
            logger.error(f"add_to_position: no trade {trade_id}")
            return False
        if row["resolved"]:
            logger.error(
                f"add_to_position: trade {trade_id} is already resolved; refusing "
                f"to add a fill to a closed position")
            return False
        if add_usd <= 0:
            return False

        old_size = float(row["position_size_usd"] or 0.0)
        old_price = float(row["market_price"] or 0.0)
        new_size = old_size + float(add_usd)
        if new_size <= 0:
            return False
        new_price = ((old_size * old_price) + (float(add_usd) * float(add_price))) / new_size
        self.conn.execute(
            "UPDATE trades SET position_size_usd = ?, market_price = ? WHERE id = ?",
            (new_size, new_price, trade_id))
        self.conn.commit()
        logger.info(
            f"Position {trade_id} grown by ${add_usd:.4f} at {add_price:.4f} -> "
            f"${new_size:.4f} at {new_price:.4f} entry")
        return True

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
            "SELECT id, resolved, position_size_usd, execution_mode, status "
            "FROM trades WHERE id = ?",
            (trade_id,))
        row = cur.fetchone()
        if row is None:
            logger.warning(f"resolve_trade: no trade with id {trade_id}")
            return False
        if row["resolved"]:
            logger.warning(f"resolve_trade: trade {trade_id} already resolved, ignoring")
            return False

        # The MODE IS READ FROM THE ROW, not taken from the caller.
        #
        # It used to bank every settlement the same way, so a simulated
        # exploration trade that resolved in a live market did this:
        #
        #   paper trade -> paper P&L -> set_bankroll(get_bankroll() + pnl)
        #
        # and the agent's real capital changed because a simulation said so. A
        # caller cannot now mislabel a trade to make it bank, and a paper row
        # banks into the paper account even if someone asks for otherwise.
        mode = _execution_mode(row["execution_mode"], row["status"])

        self.conn.execute(
            "UPDATE trades SET resolved = 1, outcome = ?, pnl = ?, status = 'settled', notes = ? WHERE id = ?",
            (outcome, pnl, notes, trade_id))
        self.conn.commit()

        if mode == PAPER:
            paper_before = self.get_paper_bankroll()
            self.set_paper_bankroll(paper_before + pnl)
            logger.info(
                f"Trade {trade_id} settled (PAPER): outcome={outcome} "
                f"pnl=${pnl:+.2f} paper bankroll="
                f"${self.get_paper_bankroll():.2f} | REAL bankroll unchanged "
                f"at ${self.get_bankroll():.2f}")
        else:
            self.set_bankroll(self.get_bankroll() + pnl)
            logger.info(
                f"Trade {trade_id} settled (LIVE): outcome={outcome} "
                f"pnl=${pnl:+.2f} bankroll=${self.get_bankroll():.2f}")
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
        # Win rate and average P&L over LIVE trades only.
        #
        # Every resolved trade used to be counted here, so simulated wins moved
        # the operator's win rate - and `total_pnl` is bankroll minus initial,
        # which used to include paper P&L banked into the real bankroll. The
        # two account for different things and are now reported separately.
        cur2 = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins, "
            "AVG(pnl) as avg_pnl FROM trades "
            "WHERE resolved=1 AND COALESCE(execution_mode,'live')='live'")
        row = cur2.fetchone()
        win_rate = (row["wins"] / row["total"] * 100) if row and row["total"] else 0

        paper = self.get_paper_performance()
        return {
            "bankroll": bankroll,
            "initial_bankroll": initial,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / initial * 100) if initial else 0,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": row["avg_pnl"] if row else 0,
            "history": history,
            "open_positions": self.count_open_positions(),
            # The simulation's own record, kept apart so it can inform learning
            # without ever being mistaken for real capital.
            "paper": paper,
        }

    def get_paper_performance(self) -> Dict:
        """
        The paper account's results, separate from the real one.

        A paper equity curve is what makes step 6 of the engineering sequence
        possible - proving the loop on real data before risking capital - so it
        has to be kept honestly, and it has to be kept somewhere other than the
        bankroll.
        """
        try:
            row = self.conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins, "
                "SUM(COALESCE(pnl,0)) AS net_pnl, "
                "AVG(CASE WHEN pnl IS NOT NULL THEN pnl END) AS avg_pnl "
                "FROM trades WHERE resolved=1 "
                "AND COALESCE(execution_mode,'live')='paper'").fetchone()
            open_paper = self.conn.execute(
                "SELECT COUNT(*) AS c FROM trades "
                "WHERE resolved=0 AND COALESCE(execution_mode,'live')='paper'"
            ).fetchone()
        except Exception as e:
            logger.error(f"get_paper_performance failed: {e}")
            return {"error": str(e)}

        initial = float(self.get_state("initial_bankroll") or 50.0)
        total = int(row["total"] or 0) if row else 0
        net = float(row["net_pnl"] or 0.0) if row else 0.0
        bankroll = self.get_paper_bankroll()
        return {
            "bankroll": bankroll,
            "initial_bankroll": initial,
            "net_pnl": net,
            "net_pnl_pct": (net / initial * 100) if initial else 0.0,
            "settled_trades": total,
            "open_positions": int(open_paper["c"] or 0) if open_paper else 0,
            "win_rate": ((row["wins"] / total * 100) if total else 0.0) if row else 0.0,
            "avg_pnl": float(row["avg_pnl"] or 0.0) if row else 0.0,
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
