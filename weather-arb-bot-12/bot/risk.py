# risk.py
"""
Pre-trade risk checks. Every signal must clear ALL of these before
execution.execute_signal() is ever called.

All thresholds come from config.py — nothing hardcoded here. The order of
checks is roughly cheap-first so we bail out before doing more work.
"""

import logging
from datetime import datetime

from config import (
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_LIQUIDITY_USD,
    MIN_HOURS_TO_EXPIRY,
    MAX_SOURCE_DISAGREEMENT,
    MAX_DAILY_DRAWDOWN_PCT,
)
from db import get_open_positions, get_daily_pnl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RiskCheck:
    """A single check result. Truthy if passed, falsy if failed."""

    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return f"RiskCheck(passed={self.passed}, reason={self.reason!r})"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_position_size(size_usdc: float, bankroll: float) -> RiskCheck:
    """No single position may exceed MAX_POSITION_PCT of bankroll (default 2%)."""
    max_size = MAX_POSITION_PCT * bankroll
    if size_usdc > max_size:
        return RiskCheck(
            False,
            f"position ${size_usdc:.2f} exceeds max ${max_size:.2f} "
            f"({MAX_POSITION_PCT*100:.0f}% of ${bankroll:.0f})",
        )
    return RiskCheck(True)


def check_total_exposure(new_position_size: float, bankroll: float) -> RiskCheck:
    """
    Total open exposure across all weather contracts must not exceed
    MAX_TOTAL_EXPOSURE_PCT of bankroll (default 20%).
    """
    open_positions = get_open_positions()
    current_exposure = sum(p.get("size_usdc", 0.0) or 0.0 for p in open_positions)
    new_total = current_exposure + new_position_size
    max_total = MAX_TOTAL_EXPOSURE_PCT * bankroll
    if new_total > max_total:
        return RiskCheck(
            False,
            f"total exposure ${new_total:.2f} would exceed "
            f"${max_total:.2f} ({MAX_TOTAL_EXPOSURE_PCT*100:.0f}% of bankroll)",
        )
    return RiskCheck(True)


def check_time_to_expiry(resolution_date_str: str) -> RiskCheck:
    """
    Don't trade within MIN_HOURS_TO_EXPIRY hours of resolution (default 6h).
    Liquidity dries up and spreads widen as a contract approaches resolution.
    """
    if not resolution_date_str:
        return RiskCheck(False, "no resolution date provided")
    try:
        raw = resolution_date_str.strip()
        # Bare date like "2026-04-30" → treat as end of that day (23:59:59),
        # not midnight, because the market covers the full calendar day and
        # actual resolution happens hours later once data is finalized.
        if "T" not in raw and len(raw) == 10:
            raw = raw + "T23:59:59"
        resolution_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        now = datetime.now(tz=resolution_dt.tzinfo) if resolution_dt.tzinfo else datetime.now()
        hours_remaining = (resolution_dt - now).total_seconds() / 3600
    except (ValueError, TypeError) as e:
        return RiskCheck(False, f"could not parse resolution date {resolution_date_str!r}: {e}")

    if hours_remaining < MIN_HOURS_TO_EXPIRY:
        return RiskCheck(
            False,
            f"only {hours_remaining:.1f}h to expiry (min {MIN_HOURS_TO_EXPIRY:.0f}h)",
        )
    return RiskCheck(True)


def check_liquidity(liquidity_usd: float) -> RiskCheck:
    """Skip contracts with order-book liquidity below MIN_LIQUIDITY_USD (default $500)."""
    if liquidity_usd < MIN_LIQUIDITY_USD:
        return RiskCheck(
            False,
            f"liquidity ${liquidity_usd:.0f} below minimum ${MIN_LIQUIDITY_USD:.0f}",
        )
    return RiskCheck(True)


def check_model_disagreement(disagreement: float) -> RiskCheck:
    """
    Skip if weather sources disagree by more than MAX_SOURCE_DISAGREEMENT
    (default 0.15 = 15 percentage points). High disagreement means the
    forecast is genuinely uncertain — that's not edge, it's noise.
    """
    if disagreement > MAX_SOURCE_DISAGREEMENT:
        return RiskCheck(
            False,
            f"model disagreement {disagreement:.3f} > {MAX_SOURCE_DISAGREEMENT:.3f}",
        )
    return RiskCheck(True)


def check_daily_drawdown(bankroll: float) -> RiskCheck:
    """
    Stop placing new trades if today's realized P&L loss exceeds
    MAX_DAILY_DRAWDOWN_PCT of bankroll (default 10%). Circuit breaker
    against runaway losses from a model going off the rails.
    """
    daily_pnl = get_daily_pnl()
    max_loss = MAX_DAILY_DRAWDOWN_PCT * bankroll
    if daily_pnl < 0 and abs(daily_pnl) > max_loss:
        return RiskCheck(
            False,
            f"daily drawdown ${abs(daily_pnl):.2f} exceeds "
            f"${max_loss:.2f} ({MAX_DAILY_DRAWDOWN_PCT*100:.0f}% of bankroll)",
        )
    return RiskCheck(True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def run_all_checks(signal: dict, bankroll: float) -> tuple[bool, list[str]]:
    """
    Run every risk check on a candidate signal.
    Returns (all_passed, list_of_failure_reasons).
    Failures are logged at INFO; we do NOT raise — caller decides what to do.
    """
    failures: list[str] = []

    checks = [
        ("position_size",      check_position_size(signal.get("kelly_size", 0.0), bankroll)),
        ("total_exposure",     check_total_exposure(signal.get("kelly_size", 0.0), bankroll)),
        ("liquidity",          check_liquidity(signal.get("liquidity_usd", 0.0))),
        ("model_disagreement", check_model_disagreement(signal.get("disagreement", 0.0))),
        ("daily_drawdown",     check_daily_drawdown(bankroll)),
    ]

    # Resolution date may be in metadata.date (ISO date) or contract.resolution_date (ISO datetime)
    resolution = (
        (signal.get("metadata") or {}).get("date")
        or signal.get("resolution_date", "")
    )
    if resolution:
        checks.append(("time_to_expiry", check_time_to_expiry(resolution)))

    cid = signal.get("contract_id", "?")[:12]
    for name, result in checks:
        if not result:
            failures.append(f"{name}: {result.reason}")
            logger.info(f"Risk check FAILED for {cid}: {name} — {result.reason}")

    if not failures:
        logger.debug(f"All risk checks passed for {cid}")
    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Use a throwaway DB so we can simulate open positions / daily P&L
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DB_PATH"] = tmp.name

    import importlib, config, db
    importlib.reload(config); importlib.reload(db)
    db.init_db()

    BANKROLL = 1000.0
    print(f"\n{'=' * 70}")
    print(f"Per-check demo (bankroll ${BANKROLL:.0f})")
    print(f"{'=' * 70}")

    cases = [
        ("position_size      OK ",     check_position_size,    (15.0,  BANKROLL)),
        ("position_size      FAIL",    check_position_size,    (50.0,  BANKROLL)),
        ("total_exposure     OK ",     check_total_exposure,   (15.0,  BANKROLL)),
        ("liquidity          OK ",     check_liquidity,        (1500.0,)),
        ("liquidity          FAIL",    check_liquidity,        (200.0,)),
        ("disagreement       OK ",     check_model_disagreement, (0.05,)),
        ("disagreement       FAIL",    check_model_disagreement, (0.25,)),
        ("daily_drawdown     OK ",     check_daily_drawdown,   (BANKROLL,)),
        ("expiry            OK ",  check_time_to_expiry,
            ((datetime.now() + timedelta(hours=48)).isoformat(),)),
        ("expiry            FAIL", check_time_to_expiry,
            ((datetime.now() + timedelta(hours=2)).isoformat(),)),
    ]
    for label, fn, args in cases:
        r = fn(*args)
        flag = "✓" if r else "✗"
        print(f"  {flag} {label:25s} -> passed={bool(r)}  {('reason='+r.reason) if not r else ''}")

    # Now simulate an open position and re-run total_exposure
    db.insert_position({
        "contract_id":  "0xexisting",
        "side":         "YES",
        "size_usdc":    180.0,   # already $180 of $200 max exposure used
        "entry_price":  0.50,
    })
    print(f"\n  After inserting $180 open position:")
    r = check_total_exposure(15.0, BANKROLL)
    print(f"    new $15 trade   -> passed={bool(r)} reason={r.reason!r}")
    r = check_total_exposure(50.0, BANKROLL)
    print(f"    new $50 trade   -> passed={bool(r)} reason={r.reason!r}")

    # Aggregator demo
    print(f"\n{'=' * 70}")
    print(f"run_all_checks() demo")
    print(f"{'=' * 70}")
    good_signal = {
        "contract_id":     "0xgood",
        "kelly_size":      15.0,
        "liquidity_usd":   2000.0,
        "disagreement":    0.05,
        "resolution_date": (datetime.now() + timedelta(hours=48)).isoformat(),
        "metadata":        {"date": (datetime.now() + timedelta(hours=48)).date().isoformat()},
    }
    passed, fails = run_all_checks(good_signal, BANKROLL)
    print(f"  GOOD signal:  passed={passed}  failures={fails}")

    bad_signal = {
        "contract_id":     "0xbad",
        "kelly_size":      50.0,         # over 2% cap
        "liquidity_usd":   200.0,         # under min liquidity
        "disagreement":    0.25,          # over disagreement cap
        "resolution_date": (datetime.now() + timedelta(hours=2)).isoformat(),  # too soon
    }
    passed, fails = run_all_checks(bad_signal, BANKROLL)
    print(f"  BAD signal:   passed={passed}  failures:")
    for f in fails:
        print(f"    - {f}")

    # cleanup
    for ext in ("", "-wal", "-shm"):
        p = tmp.name + ext
        if os.path.exists(p): os.unlink(p)
