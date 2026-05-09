# tests/test_risk.py
"""Unit tests for risk.py."""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

import pytest
from datetime import datetime, timedelta, timezone

# Use a temp DB so tests don't pollute real data
@pytest.fixture(autouse=True)
def isolated_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DB_PATH"] = tmp.name
    # Reload modules so they pick up the new DB_PATH
    import importlib, config, db
    importlib.reload(config)
    importlib.reload(db)
    db.init_db()
    yield
    for ext in ("", "-wal", "-shm"):
        p = tmp.name + ext
        if os.path.exists(p):
            os.unlink(p)


def test_check_position_size_passes_under_cap():
    from risk import check_position_size
    r = check_position_size(size_usdc=15, bankroll=1000)  # under 2% cap
    assert r.passed


def test_check_position_size_fails_over_cap():
    from risk import check_position_size
    r = check_position_size(size_usdc=50, bankroll=1000)
    assert not r.passed
    assert "exceeds max" in r.reason


def test_check_liquidity_passes_above_minimum():
    from risk import check_liquidity
    assert check_liquidity(2000.0).passed


def test_check_liquidity_fails_below_minimum():
    """Test with a value definitely below MIN_LIQUIDITY_USD (currently $50)."""
    from config import MIN_LIQUIDITY_USD
    r = __import__("risk").check_liquidity(MIN_LIQUIDITY_USD - 1)
    assert not r.passed
    assert "below minimum" in r.reason


def test_check_disagreement_passes_low():
    from risk import check_model_disagreement
    assert check_model_disagreement(0.05).passed


def test_check_disagreement_fails_high():
    from risk import check_model_disagreement
    r = check_model_disagreement(0.25)
    assert not r.passed
    assert "disagreement" in r.reason


def test_check_time_to_expiry_passes_far_future():
    from risk import check_time_to_expiry
    future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    assert check_time_to_expiry(future).passed


def test_check_time_to_expiry_fails_imminent():
    from risk import check_time_to_expiry
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    r = check_time_to_expiry(soon)
    assert not r.passed
    assert "h to expiry" in r.reason


def test_check_total_exposure_with_open_positions():
    """After inserting an open $180 position, a new $50 trade should exceed the 20% cap."""
    from db import insert_position
    from risk import check_total_exposure
    insert_position({
        "contract_id": "0xexisting", "side": "YES",
        "size_usdc": 180.0, "entry_price": 0.5,
    })
    # 180 + 50 = 230 > 20% of $1000 = $200
    r = check_total_exposure(50.0, bankroll=1000)
    assert not r.passed


def test_run_all_checks_pass_for_clean_signal():
    from risk import run_all_checks
    good = {
        "contract_id":   "0xgood",
        "kelly_size":    15.0,
        "liquidity_usd": 5000.0,
        "disagreement":  0.05,
        "metadata":      {"date": (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()},
    }
    passed, failures = run_all_checks(good, bankroll=1000.0)
    assert passed
    assert failures == []


def test_run_all_checks_fails_for_bad_signal():
    from risk import run_all_checks
    bad = {
        "contract_id":   "0xbad",
        "kelly_size":    50.0,        # > 2% cap
        "liquidity_usd": 0.0,         # below any reasonable minimum
        "disagreement":  0.25,        # > 0.15 max
        # Use a date firmly in the past so time-to-expiry fails even with end-of-day
        "metadata":      {"date": "2020-01-01"},
    }
    passed, failures = run_all_checks(bad, bankroll=1000.0)
    assert not passed
    assert len(failures) >= 3   # position_size, liquidity, disagreement, time_to_expiry
