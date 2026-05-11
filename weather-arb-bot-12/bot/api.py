# api.py
"""
FastAPI control panel for the weather arb bot.

Endpoints:
  GET  /api/status        - bot run state, last scan time, summary numbers
  GET  /api/settings      - current config values
  POST /api/settings      - update settings (writes to .env, requires restart)
  GET  /api/positions     - open positions with marked-to-market P&L
  GET  /api/positions/closed - closed positions with realized P&L
  GET  /api/signals       - recent signals (last 100)
  GET  /api/pnl_curve     - cumulative P&L over time for chart
  POST /api/bot/start     - start the bot scheduler thread
  POST /api/bot/stop      - stop the bot scheduler thread
  POST /api/bot/scan_now  - run one trading_run() immediately
  POST /api/resolver/run  - manually run the resolver
  GET  /                  - serve the static HTML UI
"""

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from db import get_conn, get_open_positions, get_recent_signals, init_db, get_daily_pnl
from edge import run_edge_scan
from execution import execute_signal, get_clob_client
from risk import run_all_checks
from resolver import resolver_pass
from sizing import get_bankroll

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
)


# ---------------------------------------------------------------------------
# Bot runtime state
# ---------------------------------------------------------------------------

class BotState:
    """Singleton holding the running scheduler + status."""
    def __init__(self):
        self.scheduler: BackgroundScheduler | None = None
        self.last_scan_at: str | None = None
        self.last_scan_stats: dict = {}
        self.scan_running: bool = False
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.scheduler is not None and self.scheduler.running


state = BotState()


def _trading_pass():
    """One scan + execute + resolver iteration. Called by the scheduler."""
    with state.lock:
        state.scan_running = True
    try:
        bankroll = get_bankroll()
        signals = run_edge_scan(bankroll=bankroll)

        executed, skipped, errors = 0, 0, 0

        # Initialise the CLOB client once. On failure (bad/missing key, network
        # error) log clearly and fall through with client=None so that
        # execute_signal routes to paper mode rather than crashing the whole pass.
        client = None
        try:
            client = get_clob_client()
        except Exception as ce:
            logger.error(
                f"trading_pass: get_clob_client() failed — {ce}. "
                f"Signals will not be executed this pass. "
                f"If PAPER_TRADE=False, check POLYMARKET_PRIVATE_KEY in .env."
            )

        for sig in signals:
            try:
                ok, fails = run_all_checks(sig, bankroll)
                if not ok:
                    skipped += 1
                    logger.info(
                        f"trading_pass: skip {sig.get('contract_id','?')[:12]} "
                        f"[{sig.get('recommended_side')}] — {'; '.join(fails)}"
                    )
                    continue
                res = execute_signal(sig, client=client)
                status = res.get("status", "")

                if status in ("placed", "paper"):
                    executed += 1
                    # Mark the signal as executed so the UI shows it correctly
                    try:
                        from db import mark_signal_executed, get_conn
                        with get_conn() as conn:
                            row = conn.execute(
                                "SELECT id FROM signals WHERE contract_id=? ORDER BY id DESC LIMIT 1",
                                (sig["contract_id"],)
                            ).fetchone()
                            if row:
                                mark_signal_executed(row["id"])
                    except Exception as me:
                        logger.warning(f"Failed to mark signal executed: {me}")

                elif status in ("skipped_duplicate", "skipped_correlated"):
                    skipped += 1
                else:
                    errors += 1

            except Exception as e:
                logger.exception(f"trading_pass: signal failed: {e}")
                errors += 1

        # After execution, refresh marks + close any resolved markets + auto-exits
        rstats = resolver_pass()

        state.last_scan_at = datetime.now(timezone.utc).isoformat()
        state.last_scan_stats = {
            "signals":     len(signals),
            "executed":    executed,
            "skipped":     skipped,
            "errors":      errors,
            "refreshed":   rstats.get("refreshed", 0),
            "auto_exits":  rstats.get("auto_exits", 0),
            "closed":      rstats.get("closed", 0),
        }
        logger.info(f"trading_pass done: {state.last_scan_stats}")
    except Exception as top_e:
        logger.exception(f"trading_pass: unexpected top-level error: {top_e}")
        state.last_scan_at = datetime.now(timezone.utc).isoformat()
        state.last_scan_stats = {"error": str(top_e)}
    finally:
        with state.lock:
            state.scan_running = False


def _resolver_pass():
    """Just refresh marks + close resolved. Cheaper than full trading pass."""
    rstats = resolver_pass()
    logger.info(f"resolver_pass done: {rstats}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Weather Arb Bot Control Panel", version="1.0")
init_db()  # ensure tables exist before any request

# Run resolver immediately on startup so any resolved markets get closed
# before the user even sees the UI
import threading
def _startup_resolver():
    try:
        rstats = resolver_pass()
        if rstats.get("closed", 0) > 0 or rstats.get("auto_exits", 0) > 0:
            logger.info(f"Startup resolver: {rstats}")
    except Exception as e:
        logger.warning(f"Startup resolver failed: {e}")
threading.Thread(target=_startup_resolver, daemon=True).start()


# --------- /api/status ----------------------------------------------------

@app.get("/api/status")
def status():
    daily_pnl = get_daily_pnl()
    bankroll = get_bankroll()
    open_pos = get_open_positions()
    open_exposure = sum(p.get("size_usdc", 0) for p in open_pos)
    open_unrealized = sum((p.get("unrealized_pnl") or 0) for p in open_pos)

    # Compute how long since last scan and flag if unhealthy
    scan_gap_minutes: float | None = None
    scan_gap_warning = False
    if state.last_scan_at:
        try:
            last_dt = datetime.fromisoformat(state.last_scan_at)
            now_dt  = datetime.now(timezone.utc)
            scan_gap_minutes = round((now_dt - last_dt).total_seconds() / 60, 1)
            # Warn if >2 scan intervals (60 min) have passed without a scan
            scan_gap_warning = scan_gap_minutes > 60
            if scan_gap_warning:
                logger.warning(
                    f"Scan gap alert: last scan was {scan_gap_minutes:.0f} min ago "
                    f"(expected every 30 min)"
                )
        except Exception:
            pass

    return {
        "running":           state.running,
        "scan_in_progress":  state.scan_running,
        "last_scan_at":      state.last_scan_at,
        "last_scan_stats":   state.last_scan_stats,
        "scan_gap_minutes":  scan_gap_minutes,
        "scan_gap_warning":  scan_gap_warning,
        "paper_trade":       config.PAPER_TRADE,
        "bankroll":          bankroll,
        "open_positions":    len(open_pos),
        "open_exposure":     round(open_exposure, 2),
        "unrealized_pnl":    round(open_unrealized, 2),
        "daily_pnl":         round(daily_pnl, 2),
        "now":               datetime.now(timezone.utc).isoformat(),
    }


# --------- /api/settings --------------------------------------------------

@app.get("/api/settings")
def get_settings():
    return {
        "PAPER_TRADE":             config.PAPER_TRADE,
        "BANKROLL_USDC":           config.INITIAL_BANKROLL,
        "current_bankroll":        round(get_bankroll(), 2),  # actual = initial + realized P&L
        "KELLY_FRACTION":          config.KELLY_FRACTION,
        "EDGE_THRESHOLD":          config.EDGE_THRESHOLD,
        "MAX_POSITION_PCT":        config.MAX_POSITION_PCT,
        "MAX_TOTAL_EXPOSURE_PCT":  config.MAX_TOTAL_EXPOSURE_PCT,
        "MIN_LIQUIDITY_USD":       config.MIN_LIQUIDITY_USD,
        "MIN_HOURS_TO_EXPIRY":     config.MIN_HOURS_TO_EXPIRY,
        "MAX_SOURCE_DISAGREEMENT": config.MAX_SOURCE_DISAGREEMENT,
        "MAX_DAILY_DRAWDOWN_PCT":  config.MAX_DAILY_DRAWDOWN_PCT,
    }


class SettingsUpdate(BaseModel):
    BANKROLL_USDC:           float | None = None
    KELLY_FRACTION:          float | None = None
    EDGE_THRESHOLD:          float | None = None
    MAX_POSITION_PCT:        float | None = None
    MAX_TOTAL_EXPOSURE_PCT:  float | None = None
    MIN_LIQUIDITY_USD:       float | None = None
    MIN_HOURS_TO_EXPIRY:     float | None = None
    MAX_SOURCE_DISAGREEMENT: float | None = None
    MAX_DAILY_DRAWDOWN_PCT:  float | None = None


@app.post("/api/settings")
def update_settings(s: SettingsUpdate):
    """
    Update settings — writes to .env and reloads config in-memory.
    Some changes (PAPER_TRADE) require a restart to take full effect.
    """
    env_path = Path(__file__).parent.parent / ".env"
    env_lines: list[str] = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()

    def set_kv(key: str, value):
        nonlocal env_lines
        if value is None:
            return
        line = f"{key}={value}"
        for i, ln in enumerate(env_lines):
            if ln.startswith(f"{key}=") or ln.startswith(f"{key} ="):
                env_lines[i] = line
                return
        env_lines.append(line)

    updates = s.model_dump(exclude_unset=True)
    for k, v in updates.items():
        set_kv(k, v)

    env_path.write_text("\n".join(env_lines) + "\n")

    # Hot-reload: update config in memory so the next scan uses new values.
    # Note: changing PAPER_TRADE this way is intentionally NOT supported —
    # that requires a full restart to ensure all guards re-initialize.
    for k, v in updates.items():
        if hasattr(config, k):
            setattr(config, k, v)

    logger.info(f"Settings updated: {updates}")
    return {"updated": updates, "note": "Some changes require restart (PAPER_TRADE)"}


# --------- /api/positions / signals / pnl --------------------------------

@app.get("/api/positions")
def positions_open():
    rows = get_open_positions()
    out = []
    for p in rows:
        with get_conn() as c:
            sig = c.execute(
                "SELECT question, model_p, market_p, ev FROM signals "
                "WHERE contract_id = ? ORDER BY id DESC LIMIT 1",
                (p["contract_id"],),
            ).fetchone()

        side = p.get("side")
        latest_model_p = sig["model_p"] if sig else None
        latest_market_p = sig["market_p"] if sig else None

        # "Thesis changed" warning: has the model flipped its directional view
        # relative to the CURRENT market price?
        #
        # Use current_price (refreshed every resolver pass) rather than the
        # stale signal market_p (which reflects the price at signal-generation
        # time).  For YES positions current_price is the YES price; for NO
        # positions it is the NO price — both set by resolver.update_unrealized_pnl.
        # Convert NO price back to YES price for the comparison.
        thesis_changed = False
        thesis_note = None
        if latest_model_p is not None:
            cur_raw = p.get("current_price")
            if cur_raw is not None:
                # Resolve to YES-equivalent price for comparison
                if side == "NO":
                    current_yes_price = 1.0 - cur_raw
                else:
                    current_yes_price = cur_raw
                current_view = "YES" if latest_model_p >= current_yes_price else "NO"
                if current_view != side:
                    thesis_changed = True
                    thesis_note = (
                        f"Bot now thinks {current_view} "
                        f"(model {latest_model_p:.2f} vs current market {current_yes_price:.2f})"
                    )
            elif latest_market_p is not None:
                # Fallback: no live price yet, use stale signal price
                current_view = "YES" if latest_model_p >= latest_market_p else "NO"
                if current_view != side:
                    thesis_changed = True
                    thesis_note = (
                        f"Bot now thinks {current_view} "
                        f"(model {latest_model_p:.2f} vs signal market {latest_market_p:.2f})"
                    )

        out.append({
            **dict(p),
            "question":         sig["question"] if sig else "",
            # latest_model_p: the model's current probability estimate (from most recent signal)
            # entry_model_p:  the model probability when the position was placed (on position row)
            "model_p":          latest_model_p,
            "signal_market_p":  latest_market_p,
            "ev":               sig["ev"] if sig else None,
            "thesis_changed":   thesis_changed,
            "thesis_note":      thesis_note,
            # model_drift: how much the model has changed since position entry
            # positive = model has become more confident in our direction
            "model_drift": (
                round(latest_model_p - p["entry_model_p"], 3)
                if latest_model_p is not None and p.get("entry_model_p") is not None
                else None
            ),
        })
    return out


@app.get("/api/positions/closed")
def positions_closed(limit: int = 100):
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM positions WHERE status IN ('closed','closed_manual') "
            "ORDER BY exit_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for p in rows:
            sig = c.execute(
                "SELECT question, market_class FROM signals WHERE contract_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (p["contract_id"],),
            ).fetchone()
            out.append({
                **dict(p),
                "question":     sig["question"]     if sig else "",
                "market_class": sig["market_class"] if sig else None,
            })
        return out


@app.get("/api/signals")
def signals(limit: int = 100):
    return get_recent_signals(limit=limit)


@app.get("/api/pnl_curve")
def pnl_curve():
    """Cumulative P&L over time, including unrealized for the current open book."""
    with get_conn() as c:
        rows = c.execute(
            """SELECT exit_time, pnl FROM positions
                WHERE status IN ('closed','closed_manual') AND pnl IS NOT NULL
                ORDER BY exit_time ASC"""
        ).fetchall()
    realized_curve = []
    cum = 0.0
    for r in rows:
        cum += r["pnl"] or 0
        realized_curve.append({"t": r["exit_time"], "cumulative_pnl": round(cum, 2)})

    # Also report current unrealized
    open_pos = get_open_positions()
    unrealized = sum((p.get("unrealized_pnl") or 0) for p in open_pos)
    return {
        "realized_points": realized_curve,
        "current_unrealized": round(unrealized, 2),
        "total_realized": round(cum, 2),
    }


# --------- /api/bot start/stop/scan ---------------------------------------

@app.post("/api/bot/start")
def bot_start():
    if state.running:
        return {"running": True, "note": "already running"}
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(_trading_pass, trigger=IntervalTrigger(minutes=30),
                  id="trading", name="Trading scan", misfire_grace_time=300)
    sched.add_job(_resolver_pass, trigger=IntervalTrigger(minutes=5),
                  id="resolver", name="Resolve open positions", misfire_grace_time=120)
    sched.start()
    state.scheduler = sched
    logger.info("Bot scheduler started — running immediate first scan")
    # Run an immediate scan so the user sees results right away
    t = threading.Thread(target=_trading_pass, daemon=True)
    t.start()
    return {"running": True, "note": "started + first scan launched"}


@app.post("/api/bot/stop")
def bot_stop():
    if not state.running:
        return {"running": False, "note": "already stopped"}
    state.scheduler.shutdown(wait=False)
    state.scheduler = None
    logger.info("Bot scheduler stopped")
    return {"running": False}


@app.post("/api/bot/scan_now")
def bot_scan_now():
    if state.scan_running:
        raise HTTPException(409, "scan already in progress")
    # Run in a background thread but wait up to 90 s for it to finish before
    # responding.  This avoids blocking the uvicorn worker indefinitely while
    # still returning actual results for fast clients; slow clients/proxies
    # that time out at 30 s will get a 202 with a retry hint instead.
    import time
    t = threading.Thread(target=_trading_pass, daemon=True)
    t.start()
    deadline = time.monotonic() + 90
    while t.is_alive() and time.monotonic() < deadline:
        time.sleep(0.5)
    if t.is_alive():
        # Still running — return 202 Accepted so the client can poll /api/status
        return JSONResponse(
            status_code=202,
            content={
                "completed": False,
                "note": "Scan is still running — poll GET /api/status for completion",
            },
        )
    return {
        "completed": True,
        "stats": state.last_scan_stats,
    }


@app.post("/api/resolver/run")
def manual_resolver():
    rstats = resolver_pass()
    return rstats


class CloseRequest(BaseModel):
    reason: str = "manual"


@app.post("/api/positions/{position_id}/close")
def close_position_manually(position_id: int, body: CloseRequest = CloseRequest()):
    """
    Manually close an open position at the CURRENT market price.
    Fetches a fresh price from Polymarket; falls back to last known price
    if the live fetch fails (e.g. market removed or conditionId mismatch).
    """
    from resolver import _fetch_market_status

    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Position {position_id} not found")
    if row["status"] not in ("open",):
        raise HTTPException(400, f"Position {position_id} is already {row['status']}")

    contract_id = row["contract_id"]
    side        = row["side"]
    entry_price = row["entry_price"]
    size_usdc   = row["size_usdc"]

    # Try to get a fresh live price; fall back to last known price gracefully
    exit_price = None
    try:
        status = _fetch_market_status(contract_id)
        if status:
            exit_price = status["yes_price"] if side == "YES" else status["no_price"]
    except Exception as e:
        logger.warning(f"Live price fetch failed for position {position_id}: {e}")

    # Fallback: use last known current_price from DB, then entry price
    if exit_price is None:
        exit_price = row["current_price"] or entry_price
        logger.info(
            f"Using last known price {exit_price:.4f} for position {position_id} "
            f"(live fetch failed — market may be removed or conditionId mismatched)"
        )

    if entry_price and entry_price > 0:
        shares = size_usdc / entry_price
        pnl = round(shares * (exit_price - entry_price), 2)
    else:
        pnl = 0.0

    from db import update_position_outcome
    update_position_outcome(position_id, exit_price=exit_price, pnl=pnl, status="closed_manual")

    try:
        with get_conn() as conn:
            conn.execute(
                """UPDATE positions SET close_reason = ? WHERE id = ?""",
                (body.reason, position_id),
            )
            conn.execute(
                """UPDATE signals SET outcome = ?, pnl = ?
                    WHERE contract_id = ? AND executed = 1 AND outcome IS NULL""",
                ("CLOSED_MANUAL", pnl, contract_id),
            )
    except Exception as e:
        logger.warning(f"Failed to update signal outcome: {e}")

    logger.info(
        f"MANUAL CLOSE #{position_id} {contract_id[:12]} "
        f"side={side} entry={entry_price:.4f} exit={exit_price:.4f} pnl=${pnl:+.2f}"
    )
    return {
        "position_id":  position_id,
        "side":         side,
        "entry_price":  entry_price,
        "exit_price":   exit_price,
        "size_usdc":    size_usdc,
        "pnl":          pnl,
        "status":       "closed_manual",
    }


# --------- Static UI ------------------------------------------------------

UI_DIR = Path(__file__).parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def index():
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({
        "error": "UI not found",
        "expected_path": str(html_path),
    })


@app.post("/api/positions/force_close_all")
def force_close_all_positions():
    """
    Force-close ALL open positions at their last stored price.
    No live Polymarket price fetch needed — uses current_price from DB.
    Use this to clear stuck positions that can't be closed normally.
    """
    from db import update_position_outcome
    from datetime import datetime, timezone

    closed = []
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM positions WHERE status='open'"
        ).fetchall()]

    for row in rows:
        entry = row.get("entry_price") or 0
        cur   = row.get("current_price") or entry
        size  = row.get("size_usdc") or 20
        side  = row.get("side") or "YES"

        if entry <= 0:
            pnl = 0.0
        else:
            # Both YES and NO positions store prices in their own side's price space:
            # - YES: entry_price = YES limit price, current_price = YES market price
            # - NO:  entry_price = NO limit price,  current_price = NO market price
            # (resolver stores no_price directly for NO positions)
            # So the P&L formula is the same for both: shares * (cur - entry)
            shares = size / entry
            pnl = round(shares * (cur - entry), 2)

        update_position_outcome(row["id"], exit_price=cur, pnl=pnl, status="closed_manual")
        with get_conn() as conn:
            conn.execute(
                "UPDATE positions SET close_reason='force_closed' WHERE id=?",
                (row["id"],)
            )
        closed.append({"id": row["id"], "pnl": pnl})
        logger.info(f"Force-closed position #{row['id']} {side} pnl=${pnl:+.2f}")

    return {"force_closed": len(closed), "positions": closed}



# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
