# db.py
"""
SQLite layer for signals and positions.

WAL mode is enabled so the dashboard can read while main.py writes.
All connections use timeout=30 to handle brief lock contention.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Iterator

from config import DB_PATH

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    contract_id      TEXT NOT NULL,
    question         TEXT,
    market_class     TEXT,
    market_p         REAL,
    model_p          REAL,
    ev               REAL,
    recommended_side TEXT,
    kelly_size       REAL,
    executed         INTEGER DEFAULT 0,
    outcome          TEXT,
    pnl              REAL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id  TEXT,
    side         TEXT,
    size_usdc    REAL,
    entry_price  REAL,
    entry_time   TEXT,
    status       TEXT DEFAULT 'open',
    exit_price   REAL,
    exit_time    TEXT,
    pnl          REAL,
    -- Snapshot of the signal at entry time — lets us detect thesis changes later
    entry_market_yes_p REAL,
    entry_model_p      REAL
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp  ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_contract   ON signals(contract_id);
CREATE INDEX IF NOT EXISTS idx_positions_status   ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_contract ON positions(contract_id);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """
    Yield a sqlite3 connection with row_factory set, 30s timeout, autocommit on success.
    Rolls back and closes on exception.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables/indexes if missing and enable WAL journaling."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        # WAL allows readers (dashboard) and writers (main.py) to not block each other
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)

        # Lazy migration: add new positions columns to existing DBs
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(positions)").fetchall()}
        for col, decl in [
            ("entry_market_yes_p", "REAL"),
            ("entry_model_p",      "REAL"),
            ("unrealized_pnl",     "REAL"),
            ("current_price",      "REAL"),
            ("last_priced_at",     "TEXT"),
            ("close_reason",       "TEXT"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {decl}")

        # Lazy migration: add market_class to signals table for existing DBs
        existing_sig_cols = {r["name"] for r in conn.execute("PRAGMA table_info(signals)").fetchall()}
        if "market_class" not in existing_sig_cols:
            conn.execute("ALTER TABLE signals ADD COLUMN market_class TEXT")

        # Reconcile executed flag: mark the latest signal for each contract as
        # executed=1 if an open position already exists for that contract+side.
        # This fixes the case where a position was placed before its signal was
        # inserted (e.g. via a direct API/UI call) so mark_signal_executed never ran.
        conn.execute("""
            UPDATE signals
               SET executed = 1
             WHERE executed = 0
               AND id IN (
                   SELECT s.id
                     FROM signals s
                     JOIN positions p
                       ON p.contract_id = s.contract_id
                      AND p.side        = s.recommended_side
                      AND p.status      = 'open'
                    WHERE s.id = (
                        SELECT id FROM signals s2
                         WHERE s2.contract_id = s.contract_id
                         ORDER BY s2.timestamp DESC
                         LIMIT 1
                    )
               )
        """)
    logger.info(f"Database initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def insert_signal(signal: dict) -> int:
    """
    Insert a signal dict (from edge.py) into the signals table.
    Deduplicates: if the same contract_id was inserted in the last 30 minutes,
    updates the existing row instead of creating a new one.
    Returns the inserted/updated row id.
    """
    from datetime import timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()

    with get_conn() as conn:
        # Check for a recent signal for the same contract
        existing = conn.execute(
            "SELECT id FROM signals WHERE contract_id = ? AND timestamp > ? ORDER BY id DESC LIMIT 1",
            (signal["contract_id"], cutoff)
        ).fetchone()

        if existing:
            # Update the existing row with fresh model/market data
            conn.execute(
                """UPDATE signals SET market_p=?, model_p=?, ev=?, recommended_side=?,
                   kelly_size=?, market_class=?, timestamp=? WHERE id=?""",
                (signal.get("market_p"), signal.get("model_p"), signal.get("ev"),
                 signal.get("recommended_side"), signal.get("kelly_size"),
                 signal.get("market_class"),
                 signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
                 existing["id"])
            )
            return int(existing["id"])

        # Insert fresh row
        cursor = conn.execute(
            """
            INSERT INTO signals (
                timestamp, contract_id, question, market_class, market_p, model_p,
                ev, recommended_side, kelly_size, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
                signal["contract_id"],
                signal.get("question", ""),
                signal.get("market_class"),
                signal.get("market_p"),
                signal.get("model_p"),
                signal.get("ev"),
                signal.get("recommended_side"),
                signal.get("kelly_size"),
                int(signal.get("executed", 0)),
            ),
        )
        row_id = int(cursor.lastrowid)

        # Prune: keep only the 500 most recent signals to prevent unbounded growth.
        # Executed signals and those with outcomes are preserved regardless.
        # Note: SQLite does not allow ORDER BY inside a UNION branch directly —
        # it must be wrapped in a subquery.
        conn.execute("""
            DELETE FROM signals
            WHERE id NOT IN (
                SELECT id FROM signals
                WHERE executed = 1 OR outcome IS NOT NULL
                UNION
                SELECT id FROM (
                    SELECT id FROM signals
                    ORDER BY timestamp DESC
                    LIMIT 500
                )
            )
        """)

        return row_id


def mark_signal_executed(signal_id: int) -> None:
    """Flag a signal as executed=1 after a successful order placement."""
    with get_conn() as conn:
        conn.execute("UPDATE signals SET executed = 1 WHERE id = ?", (signal_id,))


def update_signal_outcome(signal_id: int, outcome: str, pnl: float) -> None:
    """
    Record the resolved outcome for a signal — used by backtest.py and the
    calibration chart in dashboard.py. outcome is 'YES'/'NO' or '1'/'0'.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE signals SET outcome = ?, pnl = ? WHERE id = ?",
            (outcome, pnl, signal_id),
        )


def get_recent_signals(limit: int = 20) -> list[dict]:
    """Most recent N signals, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def insert_position(position: dict) -> int:
    """Open a new position. Returns the row id."""
    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO positions (
                contract_id, side, size_usdc, entry_price, entry_time, status,
                entry_market_yes_p, entry_model_p
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position["contract_id"],
                position["side"],
                position["size_usdc"],
                position["entry_price"],
                position.get("entry_time", datetime.now(timezone.utc).isoformat()),
                position.get("status", "open"),
                position.get("entry_market_yes_p"),
                position.get("entry_model_p"),
            ),
        )
        return int(cursor.lastrowid)


def get_open_positions() -> list[dict]:
    """All positions with status='open'."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'open'"
        ).fetchall()
        return [dict(r) for r in rows]


def update_position_outcome(
    position_id: int,
    exit_price: float,
    pnl: float,
    status: str = "closed",
) -> None:
    """Close a position with realized exit price and P&L."""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE positions
               SET exit_price = ?,
                   exit_time  = ?,
                   pnl        = ?,
                   status     = ?
             WHERE id = ?
            """,
            (exit_price, datetime.now(timezone.utc).isoformat(), pnl, status, position_id),
        )


def get_daily_pnl(target_date: date | None = None) -> float:
    """
    Sum of realized P&L for positions closed on the given UTC date (default: today).
    Used by risk.check_daily_drawdown.
    """
    if target_date is None:
        target_date = date.today()
    day_str = target_date.isoformat()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(pnl), 0.0) AS total
              FROM positions
             WHERE status IN ('closed','closed_manual')
               AND DATE(exit_time) = ?
            """,
            (day_str,),
        ).fetchone()
        return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    # Run the self-test against a throwaway DB so we don't pollute data/signals.db
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DB_PATH"] = tmp.name

    # Re-import so DB_PATH picks up the override
    import importlib
    import config
    importlib.reload(config)
    import db as _db
    importlib.reload(_db)

    _db.init_db()
    print(f"[smoke] DB initialized at {_db.DB_PATH}")

    sid = _db.insert_signal({
        "contract_id":      "test_abc123",
        "question":         "Will it snow > 2in in NYC on April 30?",
        "market_p":         0.45,
        "model_p":          0.62,
        "ev":               0.085,
        "recommended_side": "YES",
        "kelly_size":       12.50,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    })
    print(f"[smoke] inserted signal id={sid}")

    pid = _db.insert_position({
        "contract_id":  "test_abc123",
        "side":         "YES",
        "size_usdc":    12.50,
        "entry_price":  0.45,
    })
    print(f"[smoke] inserted position id={pid}")
    print(f"[smoke] open positions: {len(_db.get_open_positions())}")

    _db.update_position_outcome(pid, exit_price=1.0, pnl=6.875)
    _db.mark_signal_executed(sid)
    print(f"[smoke] daily PnL: ${_db.get_daily_pnl():.2f}")
    print(f"[smoke] open positions after close: {len(_db.get_open_positions())}")
    print(f"[smoke] recent signals returned: {len(_db.get_recent_signals())}")

    os.unlink(tmp.name)
    # Also clean up WAL/SHM sidecars
    for ext in ("-wal", "-shm"):
        p = tmp.name + ext
        if os.path.exists(p):
            os.unlink(p)
    print("[smoke] cleaned up temp db")
