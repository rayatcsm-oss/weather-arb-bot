# fix_and_start.py
"""
Run on startup: fix any stuck positions, then hand off to api.py.
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

db_path = os.getenv("DA_PATH", "data/signals.db")

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='open' AND "
            "current_price IS NOT NULL AND entry_price > 0"
        ).fetchall()
        rows = [dict(r) for r in rows]
        closed = 0
        now = datetime.now(timezone.utc).isoformat()
        for r in rows:
            entry = r['entry_price'] or 0
            cur = r['current_price'] or entry
            size = r['size_usdc'] or 20
            side = r.get('side') or 'YES'
            if entry > 0:
                # Both YES and NO positions store prices in side-specific space
                shares = size / entry
                pnl = round(shares * (cur - entry), 2)
            else:
                pnl = 0.0
            conn.execute(
                "UPDATE positions SET status='closed_manual', exit_price=?, exit_time=?, pnl=?, close_reason='force_closed' WHERE id=?",
                (cur, now, pnl, r['id'])
            )
            closed += 1
            print(f"[startup] Force-closed position #{r['id']} {side} pnl={pnl:+.2f}")
        if closed:
            conn.commit()
            print(f"[startup] Closed {closed} stuck positions")
        else:
            print("[startup] No stuck positions found")
        conn.close()
    except Exception as e:
        print(f"[startup] DB fix failed: {e}")
else:
    print("[startup] No DB found yet, will be created fresh")

# Now start the real API
print("[startup] Starting bot API...")
api_path = os.path.join(os.path.dirname(__file__), "api.py")
os.execv(sys.executable, [sys.executable, api_path])
