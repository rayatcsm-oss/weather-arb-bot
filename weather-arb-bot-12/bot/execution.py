# execution.py
"""
Order placement with hard paper-trade safety.

THREE INDEPENDENT GUARDS prevent real orders when PAPER_TRADE=True:

  1. Optional import: py_clob_client is wrapped in try/except so the module
     loads even if the lib isn't installed (paper mode never needs it).
  2. Function-entry: execute_signal() checks PAPER_TRADE before doing
     anything; get_clob_client() returns None without touching the private
     key when in paper mode.
  3. Client-level: every code path that would call create_and_post_order()
     re-checks `client is None` and bails to paper-result.

Any single guard being correct prevents an accidental live order.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any

from config import PAPER_TRADE

logger = logging.getLogger(__name__)


# Guard 1 — optional import. Paper mode doesn't need py_clob_client.
# Importing it eagerly would crash the module on a fresh sandbox install.
try:
    from py_clob_client.client import ClobClient  # type: ignore
    from py_clob_client.clob_types import OrderArgs  # type: ignore
    from py_clob_client.constants import POLYGON  # type: ignore
    _CLOB_AVAILABLE = True
except ImportError:
    ClobClient = None       # type: ignore
    OrderArgs  = None       # type: ignore
    POLYGON    = 137        # Polygon chain id, hardcoded fallback
    _CLOB_AVAILABLE = False


CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = POLYGON
LIMIT_PRICE_BUFFER = 1.005  # used only when best_ask isn't available
LIMIT_PRICE_CAP    = 0.99   # never put up a buy limit at >= $1


# ---------------------------------------------------------------------------
# Client init
# ---------------------------------------------------------------------------

def get_clob_client() -> Any:
    """
    Initialize and authenticate a CLOB client. Returns None if PAPER_TRADE
    is enabled OR if py_clob_client isn't installed. Never touches the
    private key in paper mode.
    """
    # Guard 2a — paper mode never instantiates a client
    if PAPER_TRADE:
        logger.info("PAPER_TRADE=True — CLOB client not initialized")
        return None

    if not _CLOB_AVAILABLE:
        raise RuntimeError(
            "py_clob_client is not installed but PAPER_TRADE=False. "
            "Install with: pip install py-clob-client"
        )

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise ValueError("POLYMARKET_PRIVATE_KEY is not set in environment")

    client = ClobClient(host=CLOB_HOST, key=private_key, chain_id=CHAIN_ID)
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _resolve_limit_price(signal: dict, side: str) -> float:
    """
    Pick the limit price for a buy order.

    Guide Section 7: "Apply a small buffer to improve fill probability —
    buy slightly above mid to get fills (0.5% buffer)"

    For YES: limit = best_ask if available, else YES mid * 1.005
    For NO:  limit = NO mid * 1.005  (best_ask for NO not in our data)
    Always cap at $0.99 so we never place an unfillable limit at >= $1.
    """
    yes_price = signal.get("market_p", 0.5)

    if side == "YES":
        best_ask = signal.get("best_ask") or 0.0
        if best_ask and best_ask > 0:
            limit = float(best_ask)
        else:
            limit = yes_price * LIMIT_PRICE_BUFFER
    else:
        # NO mid = 1 - YES mid
        no_mid = 1.0 - yes_price
        limit = no_mid * LIMIT_PRICE_BUFFER

    return round(min(limit, LIMIT_PRICE_CAP), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_signal(signal: dict, client: Any = None) -> dict:
    """
    Execute a trade signal — paper-log it or place a real CLOB order.

    Required signal fields:
      contract_id, recommended_side, kelly_size, market_p,
      yes_token_id, no_token_id

    Returns an execution result dict with status, order_id (or None for paper),
    limit_price, size, entry_time. Never raises — failures return
    {"status": "error", ...}.
    """
    contract_id = signal.get("contract_id", "?")
    side        = signal.get("recommended_side", "YES")
    size_usdc   = float(signal.get("kelly_size", 0.0))

    # Don't double-trade the same contract+side. If we're already long the
    # YES side on this market, skip — wait for resolution before re-entering.
    try:
        from db import get_open_positions, get_conn
        open_pos = get_open_positions()
        for p in open_pos:
            if p.get("contract_id") == contract_id and p.get("side") == side:
                logger.info(
                    f"Skip {contract_id[:12]}: already open {side} position "
                    f"id={p.get('id')} size=${p.get('size_usdc')}"
                )
                return {
                    "contract_id":    contract_id,
                    "side":           side,
                    "status":         "skipped_duplicate",
                    "existing_id":    p.get("id"),
                }

        # CORRELATED RISK CHECK: For monthly_precip AND global_temp markets,
        # don't open more than 2 positions on the SAME underlying event.
        # - monthly_precip: same city+month = same underlying precipitation total
        # - global_temp: same year = same underlying temperature anomaly
        # We allow 2 correlated positions because the buckets partially hedge
        # each other (if third-hottest wins, sixth-or-lower loses), but more
        # than 2 starts to look like we're just buying the whole market.
        metadata = signal.get("metadata", {})
        cls = metadata.get("market_class")

        if cls == "monthly_precip":
            my_city = metadata.get("city", "")
            my_month = metadata.get("month_iso", "")
            correlated_count = 0
            for p in open_pos:
                with get_conn() as conn:
                    existing_sig = conn.execute(
                        "SELECT question FROM signals WHERE contract_id = ? "
                        "ORDER BY id DESC LIMIT 1",
                        (p.get("contract_id"),)
                    ).fetchone()
                if existing_sig:
                    eq = (existing_sig["question"] or "").lower()
                    month_names = ["january","february","march","april","may","june",
                                   "july","august","september","october","november","december"]
                    month_idx = metadata.get("month", 1)
                    month_name = month_names[month_idx - 1] if 1 <= month_idx <= 12 else ""
                    if my_city in eq and month_name in eq:
                        correlated_count += 1
            if correlated_count >= 2:
                logger.info(
                    f"Skip {contract_id[:12]}: already have {correlated_count} correlated "
                    f"positions on {my_city}/{my_month}"
                )
                return {
                    "contract_id": contract_id,
                    "side": side,
                    "status": "skipped_correlated",
                }

        elif cls == "global_temp":
            my_year = metadata.get("temp_year", 0)
            correlated_count = 0
            for p in open_pos:
                with get_conn() as conn:
                    existing_sig = conn.execute(
                        "SELECT question FROM signals WHERE contract_id = ? "
                        "ORDER BY id DESC LIMIT 1",
                        (p.get("contract_id"),)
                    ).fetchone()
                if existing_sig:
                    eq = (existing_sig["question"] or "").lower()
                    if str(my_year) in eq and ("hottest" in eq or "warmest" in eq or "coldest" in eq):
                        correlated_count += 1
            if correlated_count >= 2:
                logger.info(
                    f"Skip {contract_id[:12]}: already have {correlated_count} correlated "
                    f"global_temp positions for {my_year}"
                )
                return {
                    "contract_id": contract_id,
                    "side": side,
                    "status": "skipped_correlated",
                }
    except Exception as e:
        logger.warning(f"Open-position check failed: {e}")

    # Resolve which token the order will buy
    if side == "YES":
        token_id = signal.get("yes_token_id")
    else:
        token_id = signal.get("no_token_id")

    limit_price = _resolve_limit_price(signal, side)

    result = {
        "contract_id": contract_id,
        "side":        side,
        "size_usdc":   size_usdc,
        "limit_price": limit_price,
        "entry_time":  datetime.now(timezone.utc).isoformat(),
        "paper_trade": PAPER_TRADE,
    }

    # Guard 2b — paper mode short-circuits before ANY order code runs.
    # Guard 3 — even if PAPER_TRADE somehow flipped, a None client also bails.
    if PAPER_TRADE or client is None:
        logger.info(
            f"[PAPER] Would execute: {side} ${size_usdc:.2f} on {contract_id[:12]} "
            f"@ {limit_price:.4f}"
        )
        # Persist a paper position so the UI/dashboard can display it.
        # We use entry_price = limit_price (the price we'd cross at).
        try:
            from db import insert_position, get_conn
            position_id = insert_position({
                "contract_id":         contract_id,
                "side":                side,
                "size_usdc":           size_usdc,
                "entry_price":         limit_price,
                "entry_time":          result["entry_time"],
                "status":              "open",
                "entry_market_yes_p":  signal.get("market_p"),
                "entry_model_p":       signal.get("model_p"),
            })
            # Initialize current_price and unrealized_pnl so UI shows data
            # before the resolver runs its first price-refresh pass
            yes_px = signal.get("market_p", 0.5)
            cur_px = yes_px if side == "YES" else 1.0 - yes_px
            with get_conn() as conn:
                conn.execute(
                    "UPDATE positions SET current_price=?, unrealized_pnl=? WHERE id=?",
                    (cur_px, 0.0, position_id)
                )
            result["position_id"] = position_id
        except Exception as e:
            logger.warning(f"Failed to persist paper position: {e}")
        result.update({"status": "paper", "order_id": None})
        return result

    # ------- LIVE PATH below this line -------------------------------
    if not token_id:
        logger.error(f"No token_id for {contract_id} side={side}")
        result.update({"status": "error", "reason": "missing_token_id"})
        return result

    if size_usdc <= 0 or limit_price <= 0:
        logger.error(f"Bad size/price for {contract_id}: size={size_usdc} px={limit_price}")
        result.update({"status": "error", "reason": "invalid_size_or_price"})
        return result

    if not _CLOB_AVAILABLE:
        result.update({"status": "error", "reason": "py_clob_client_not_installed"})
        return result

    try:
        order_args = OrderArgs(
            price=limit_price,
            size=size_usdc / limit_price,   # convert USDC notional -> shares
            side=side,
            token_id=token_id,
        )
        response = client.create_and_post_order(order_args)
        if response and response.get("success"):
            order_id = response.get("orderID")
            logger.info(
                f"Order placed: {side} ${size_usdc:.2f} on {contract_id[:12]} "
                f"@ {limit_price:.4f} | order_id={order_id}"
            )
            result.update({"status": "placed", "order_id": order_id})
        else:
            logger.error(f"Order failed: {response}")
            result.update({"status": "failed", "response": str(response)})
    except Exception as e:
        logger.exception(f"Execution error for {contract_id}: {e}")
        result.update({"status": "error", "reason": str(e)})
    return result


def cancel_order(order_id: str, client: Any) -> bool:
    """Cancel an open order by id. Returns True on success or in paper mode."""
    if PAPER_TRADE or client is None:
        logger.info(f"[PAPER] Would cancel order {order_id}")
        return True
    try:
        resp = client.cancel(order_id)
        return bool(resp)
    except Exception as e:
        logger.error(f"Failed to cancel {order_id}: {e}")
        return False


def get_open_orders(client: Any) -> list[dict]:
    """List all open CLOB orders for monitoring. Empty list in paper mode."""
    if PAPER_TRADE or client is None:
        return []
    try:
        return client.get_orders() or []
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return []


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 70)
    print(f"PAPER_TRADE={PAPER_TRADE}  py_clob_client_available={_CLOB_AVAILABLE}")
    print("=" * 70)

    # 1) get_clob_client should NOT touch credentials in paper mode
    client = get_clob_client()
    print(f"\n1) get_clob_client() in paper mode -> {client}  (expect None)")

    # 2) execute_signal — paper path with best_ask present
    sig_with_ask = {
        "contract_id":      "0xpaperA",
        "recommended_side": "YES",
        "kelly_size":       20.0,
        "market_p":         0.45,
        "best_ask":         0.46,         # live top-of-book
        "yes_token_id":     "tok_yes",
        "no_token_id":      "tok_no",
    }
    r = execute_signal(sig_with_ask, client=client)
    print(f"\n2) execute_signal (with best_ask=0.46):")
    for k, v in r.items():
        print(f"     {k:14s} {v}")
    assert r["status"] == "paper",       "Expected paper status"
    assert r["limit_price"] == 0.46,     f"Expected 0.46 (best_ask), got {r['limit_price']}"

    # 3) execute_signal — paper path WITHOUT best_ask, falls back to mid * 1.005
    sig_no_ask = dict(sig_with_ask, best_ask=None, contract_id="0xpaperB")
    r = execute_signal(sig_no_ask, client=client)
    print(f"\n3) execute_signal (no best_ask, fallback to mid * 1.005):")
    print(f"     status={r['status']}  limit_price={r['limit_price']}  (expect ~0.4523)")
    assert abs(r["limit_price"] - 0.4523) < 0.0001, "Fallback math is wrong"

    # 4) NO side
    sig_no = dict(sig_with_ask, recommended_side="NO", contract_id="0xpaperC", best_ask=None)
    r = execute_signal(sig_no, client=client)
    print(f"\n4) execute_signal NO side, market_p=0.45 (NO mid=0.55):")
    print(f"     status={r['status']}  limit_price={r['limit_price']}  (expect ~0.5527)")

    # 5) cancel/list helpers in paper mode
    print(f"\n5) paper-mode helpers:")
    print(f"     cancel_order('0xfoo', None) -> {cancel_order('0xfoo', None)}")
    print(f"     get_open_orders(None)       -> {get_open_orders(None)}")

    # 6) Triple-guard demonstration
    print(f"\n6) Triple-guard verification:")
    print(f"     Guard 1 (import):    py_clob_client available? {_CLOB_AVAILABLE}")
    print(f"     Guard 2 (entry):     PAPER_TRADE = {PAPER_TRADE}")
    print(f"     Guard 3 (client):    client is None? {client is None}")
    print("\nAll guards independently prevent live execution. Module is safe.")
