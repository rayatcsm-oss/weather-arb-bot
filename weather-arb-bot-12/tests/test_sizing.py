# tests/test_sizing.py
"""Unit tests for sizing.py — pure math, no I/O, no mocks needed."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

import pytest
from sizing import calculate_kelly_size


class TestKellySize:
    def test_zero_edge_returns_zero(self):
        assert calculate_kelly_size(edge=0.0, odds=1.0, bankroll=1000) == 0.0

    def test_negative_edge_returns_zero(self):
        assert calculate_kelly_size(edge=-0.05, odds=1.0, bankroll=1000) == 0.0

    def test_zero_bankroll_returns_zero(self):
        assert calculate_kelly_size(edge=0.10, odds=1.0, bankroll=0) == 0.0

    def test_zero_odds_returns_zero(self):
        assert calculate_kelly_size(edge=0.10, odds=0.0, bankroll=1000) == 0.0

    def test_capped_at_max_position_pct(self):
        """Strong edge should hit the 2% bankroll cap, not exceed it."""
        size = calculate_kelly_size(edge=0.20, odds=1.0, bankroll=1000)
        # 2% of $1000 = $20
        assert size == pytest.approx(20.0, abs=0.01)

    def test_small_edge_below_cap(self):
        """A weaker edge should size below the cap."""
        size = calculate_kelly_size(edge=0.02, odds=1.0, bankroll=1000)
        assert 0 < size < 20.0

    def test_proportional_to_bankroll(self):
        """Same edge × different bankrolls should scale proportionally (within the cap)."""
        small = calculate_kelly_size(edge=0.20, odds=1.0, bankroll=100)
        big   = calculate_kelly_size(edge=0.20, odds=1.0, bankroll=1000)
        # Both should hit cap (2%): $2 and $20
        assert small == pytest.approx(2.0, abs=0.01)
        assert big   == pytest.approx(20.0, abs=0.01)

    def test_minimum_one_dollar(self):
        """Even tiny positive sizes are bumped to $1 minimum."""
        # Very small bankroll where Kelly fraction × bankroll < 1
        size = calculate_kelly_size(edge=0.01, odds=1.0, bankroll=10)
        assert size >= 1.0
