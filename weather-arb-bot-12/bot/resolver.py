# resolver.py
"""
Periodic position resolver.

For every open paper position, fetches the current Polymarket price.
If the market has resolved (one outcome at ~1.0 and the other at ~0.0),
closes the position with realized P&L:

    pnl = (exit_price - entry_price) * shares
    shares = size_usdc / entry_price

    For YES: outcome=1.0 means we win (collect $1 per share)
    For NO:  outcome=1.0 on the YES side means we LOSE (NO settles at $0)
"""

import json
import logging
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from db import get_open_positions, update_position_outcome, get_conn

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_market_status(contract_id: str) -> dict | None:
    """
    Fetch a market's current status. Returns dict with:
        yes_price, no_price, closed (bool), resolved (bool)
    or None on failure.

    Tries conditionId lookup first, then falls back to walking /events
    (handles truncated IDs and old-format conditionIds).
    """
    def _parse_market(market: dict) -> dict | None:
        try:
            prices   = json.loads(market.get("outcomePrices", "[]"))
            outcomes = json.loads(market.get("outcomes", "[]"))
        except Exception:
            return None
        if len(prices) != 2 or len(outcomes) != 2:
            return None
        yes_idx = next((i for i, o in enumerate(outcomes) if str(o).upper() == "YES"), None)
        if yes_idx is None:
            return None
        yes_price = float(prices[yes_idx])
        no_price  = float(prices[1 - yes_idx])
        closed    = bool(market.get("closed", False))
        resolved  = (yes_price > 0.99 and no_price < 0.01) or \
                    (no_price > 0.99 and yes_price < 0.01)
        return {
            "yes_price": yes_price,
            "no_price":  no_price,
            "closed":    closed,
            "resolved":  resolved,
            "winning_side": "YES" if yes_price > 0.5 else "NO",
        }

    # 1) Try by conditionId (exact match)
    try:
        resp = httpx.get(f"{GAMMA_BASE}/markets",
                         params={"condition_ids": contract_id, "limit": 1},
                         timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            market = None
            if isinstance(data, list) and data:
                market = data[0]
            elif isinstance(data, dict) and data.get("markets"):
                market = data["markets"][0]
            if market:
                result = _parse_market(market)
                if result:
                    return result
    except Exception as e:
        logger.debug(f"conditionId lookup failed for {contract_id[:12]}: {e}")

    # 2) Fallback: walk /events to find a matching conditionId prefix
    # This handles truncated IDs stored by old code versions
    try:
        for offset in range(0, 1000, 200):
            resp = httpx.get(f"{GAMMA_BASE}/events",
                             params={"active": "true", "closed": "false",
                                     "limit": 200, "offset": offset},
                             timeout=10)
            if resp.status_code != 200:
                break
            events = resp.json() or []
            for event in events:
                for m in (event.get("markets") or []):
                    cid = m.get("conditionId", "")
                    if cid and (cid == contract_id or
                                cid.startswith(contract_id[:20]) or
                                contract_id.startswith(cid[:20])):
                        result = _parse_market(m)
                        if result:
                            return result
            if len(events) < 200:
                break
    except Exception as e:
        logger.debug(f"Events fallback failed for {contract_id[:12]}: {e}")

    return None


def _calculate_pnl(side: str, entry_price: float, size_usdc: float, won: bool) -> float:
    """
    P&L for a closed binary-outcome bet.

    Each contract pays $1 if you win, $0 if you lose.
    shares = size_usdc / entry_price
    """
    if entry_price <= 0:
        return 0.0
    shares = size_usdc / entry_price
    if won:
        # Win: collect $1 per share, paid $entry_price per share => profit per share = 1 - entry_price
        return round(shares * (1.0 - entry_price), 2)
    else:
        # Loss: collected $0, paid $entry_price per share => loss = -size_usdc
        return round(-size_usdc, 2)


def update_unrealized_pnl() -> int:
    """
    For every open position, refresh current market price and recompute
    unrealized P&L. Stored on the position via a side-channel (we add an
    `unrealized_pnl` column lazily via ALTER if missing).

    Returns count of positions whose price was refreshed.
    """
    # Lazily add the `unrealized_pnl` column if it doesn't exist
    with get_conn() as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
        if "unrealized_pnl" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN unrealized_pnl REAL")
        if "current_price" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN current_price REAL")
        if "last_priced_at" not in cols:
            conn.execute("ALTER TABLE positions ADD COLUMN last_priced_at TEXT")

    open_pos = get_open_positions()
    refreshed = 0

    for p in open_pos:
        contract_id = p.get("contract_id")
        if not contract_id:
            continue
        try:
            status = _fetch_market_status(contract_id)
        except Exception as e:
            logger.warning(f"Failed to refresh price for {contract_id[:12]}: {e}")
            continue
        if not status:
            continue

        side        = p.get("side")
        entry_price = p.get("entry_price") or 0
        size_usdc   = p.get("size_usdc")    or 0

        # Current price for our SIDE
        cur_px = status["yes_price"] if side == "YES" else status["no_price"]

        # Mark-to-market unrealized PnL: shares * (cur_px - entry_price)
        if entry_price > 0:
            shares = size_usdc / entry_price
            unrealized = round(shares * (cur_px - entry_price), 2)
        else:
            unrealized = 0.0

        with get_conn() as conn:
            conn.execute(
                """UPDATE positions
                      SET current_price = ?, unrealized_pnl = ?, last_priced_at = ?
                    WHERE id = ?""",
                (cur_px, unrealized,
                 datetime.now(timezone.utc).isoformat(),
                 p["id"]),
            )
        refreshed += 1
    return refreshed


def resolve_closed_positions() -> int:
    """
    For every open position whose underlying market has resolved, close it
    with realized P&L. Returns the number closed this run.
    """
    closed_count = 0
    for p in get_open_positions():
        contract_id = p.get("contract_id")
        if not contract_id:
            continue
        try:
            status = _fetch_market_status(contract_id)
        except Exception as e:
            logger.warning(f"Failed to check resolution for {contract_id[:12]}: {e}")
            continue
        if not status or not status["resolved"]:
            continue

        side        = p.get("side")
        entry_price = p.get("entry_price") or 0
        size_usdc   = p.get("size_usdc")    or 0
        won         = (side == status["winning_side"])
        exit_price  = 1.0 if won else 0.0
        pnl         = _calculate_pnl(side, entry_price, size_usdc, won)

        update_position_outcome(p["id"], exit_price=exit_price, pnl=pnl, status="closed")
        logger.info(
            f"RESOLVED {contract_id[:12]} | side={side} entry={entry_price:.3f} "
            f"won={won} pnl=${pnl:+.2f}"
        )

        # Also mark the matching signal as having an outcome
        try:
            with get_conn() as conn:
                conn.execute(
                    """UPDATE signals
                          SET outcome = ?, pnl = ?
                        WHERE contract_id = ? AND executed = 1 AND outcome IS NULL""",
                    ("WON" if won else "LOST", pnl, contract_id),
                )
        except Exception as e:
            logger.warning(f"Failed to update signal outcome: {e}")

        closed_count += 1
    return closed_count



# ===========================================================================
# Exit rules — strictly following the guide
# ===========================================================================
#
# THE GUIDE SAYS: Hold every position to resolution. Binary contracts pay
# $1.00 (win) or $0.00 (loss) at the deadline. No stop-loss. No take-profit.
#
# WHY NO STOP-LOSS: If model says 44% probability and the market drops to
# 9%, your edge just went from 26pp to 35pp. Selling here destroys expected
# value. A stop-loss exits when the bet is MOST in your favor mathematically.
#
# WHY NO TAKE-PROFIT: Same logic. If you bought YES at 3 cents and it's now
# at 30 cents, you've gained 900% but if the true probability is 44%, the
# market is STILL underpriced. Selling early surrenders future edge.
#
# THE ONE EXCEPTION — THESIS FLIP:
# If the model's underlying probability estimate changes direction, that's
# a genuine new signal, not market noise. Example: model said 44% for a
# named storm, but then June 1 passes with no storm and the season ends.
# Now model says <5%. That's a real thesis change → exit and stop the bleeding.
# Requires 7pp edge in the new direction to confirm it's a real flip, not noise.


def check_exit_rules() -> int:
    """
    Check for thesis flips only. Returns count of auto-exited positions.

    Guide says hold to resolution. The only auto-exit is a genuine thesis flip:
    when the model now says the opposite side has edge (>= 7pp) vs current price.
    Stop-loss and take-profit are deliberately NOT implemented — they destroy
    expected value by exiting when edge is largest.
    """
    exits = 0
    open_pos = get_open_positions()

    for p in open_pos:
        pid         = p["id"]
        side        = p.get("side")
        entry_price = p.get("entry_price") or 0
        cur_price   = p.get("current_price")
        size_usdc   = p.get("size_usdc") or 0
        unreal      = p.get("unrealized_pnl")
        contract_id = p.get("contract_id")

        if cur_price is None or entry_price <= 0 or size_usdc <= 0:
            continue

        # THESIS FLIP: re-fetch live market price and compare to last model estimate.
        # This runs every 5 min so catches flips without waiting for the 30-min scan.
        status = None
        try:
            status = _fetch_market_status(contract_id)
        except Exception:
            pass

        if not status:
            continue

        current_yes_price = status["yes_price"]

        with get_conn() as conn:
            latest_sig = conn.execute(
                "SELECT model_p FROM signals "
                "WHERE contract_id = ? ORDER BY id DESC LIMIT 1",
                (contract_id,)
            ).fetchone()

        if not latest_sig or latest_sig["model_p"] is None:
            continue

        model_p = latest_sig["model_p"]
        current_view = "YES" if model_p >= current_yes_price else "NO"

        if current_view == side:
            continue  # thesis intact, hold

        # Thesis flipped — but only exit if the new edge is meaningful (>= 7pp)
        flip_edge = abs(model_p - current_yes_price)
        if flip_edge < 0.07:
            continue  # tiny edge, might be noise — hold

        reason = (
            f"thesis_flip: model={model_p:.3f} now favors {current_view} "
            f"vs market={current_yes_price:.3f} (edge={flip_edge:.3f})"
        )
        actual_pnl = round(unreal, 2) if unreal is not None else 0.0

        update_position_outcome(pid, exit_price=cur_price, pnl=actual_pnl, status="closed")
        with get_conn() as conn:
            conn.execute(
                "UPDATE positions SET close_reason = ? WHERE id = ?",
                (reason, pid),
            )
        logger.info(
            f"THESIS-FLIP EXIT #{pid} {contract_id[:12]} | {reason} | "
            f"side={side} entry={entry_price:.4f} exit={cur_price:.4f} pnl=${actual_pnl:+.2f}"
        )
        exits += 1

    return exits


def resolver_pass() -> dict:
    """One full pass: refresh prices → check exit rules → close resolved. Returns counts."""
    refreshed   = update_unrealized_pnl()
    auto_exits  = check_exit_rules()
    resolved    = resolve_closed_positions()
    return {"refreshed": refreshed, "auto_exits": auto_exits, "closed": resolved}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    result = resolver_pass()
    print(f"Resolver pass: {result}")
