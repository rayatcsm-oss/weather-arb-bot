#!/usr/bin/env python3
"""
This script runs automatically when the bot starts.
It closes any stuck/bad positions then launches the API.
"""
import sqlite3, os, sys, glob, subprocess
from datetime import datetime, timezone

# Find DB
candidates = [
    os.path.join(os.path.dirname(__file__), "data/signals.db"),
    os.path.expanduser("~/weather-arb-bot/bot/data/signals.db"),
]
candidates += glob.glob(os.path.expanduser("~/*/bot/data/signals.db"))
candidates += glob.glob(os.path.expanduser("~/Downloads/*/bot/data/signals.db"))

db_path = None
for p in candidates:
    if os.path.exists(p):
        db_path = p
        break

if db_path:
    print(f"[startup] Found DB: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
        now = datetime.now(timezone.utc).isoformat()
        closed = 0
        for r in rows:
            r = dict(r)
            entry = r['entry_price'] or 0
            cur = r['current_price'] or entry
            size = r['size_usdc'] or 20
            side = r.get('side') or 'YES'
            if entry > 0:
                # Both YES and NO positions store prices in side-specific space:
                # - YES: entry/cur are YES prices
                # - NO:  entry/cur are NO prices (resolver stores no_price directly)
                # So the formula is the same for both sides.
                shares = size / entry
                pnl = round(shares * (cur - entry), 2)
            else:
                pnl = 0.0
            conn.execute(
                "UPDATE positions SET status='closed_manual', exit_price=?, exit_time=?, pnl=?, close_reason='force_closed' WHERE id=?",
                (cur, now, pnl, r['id'])
            )
            closed += 1
            print(f"[startup] Force-closed position #{r['id']} {r['side']} pnl=${pnl:+.2f}")
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
