# tests/test_edge.py
"""Unit tests for edge.py — pure math + mocked pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

import pytest
from unittest.mock import patch
import edge as edge_mod
from edge import calculate_ev, determine_side, _odds_for_side, analyze_contract


class TestEV:
    def test_positive_yes_edge(self):
        # model says 72%, market says 58% -> bet YES
        ev = calculate_ev(0.72, 0.58, "YES")
        assert ev == pytest.approx(0.14, abs=1e-6)

    def test_positive_no_edge(self):
        # model says 30%, market says 58% -> bet NO
        ev = calculate_ev(0.30, 0.58, "NO")
        assert ev == pytest.approx(0.28, abs=1e-6)

    def test_no_edge_returns_zero(self):
        # model agrees with market -> EV = 0
        ev = calculate_ev(0.50, 0.50, "YES")
        assert ev == pytest.approx(0.0, abs=1e-6)

    def test_negative_yes_ev_when_market_above_model(self):
        # market thinks YES is more likely than we do -> negative EV on YES
        ev = calculate_ev(0.40, 0.60, "YES")
        assert ev < 0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            calculate_ev(0.5, 0.5, "MAYBE")


class TestDetermineSide:
    def test_yes_when_model_above_market(self):
        side, edge = determine_side(0.72, 0.58)
        assert side == "YES"
        assert edge == pytest.approx(0.14, abs=1e-6)

    def test_no_when_model_below_market(self):
        side, edge = determine_side(0.30, 0.58)
        assert side == "NO"
        assert edge == pytest.approx(0.28, abs=1e-6)

    def test_ties_default_to_yes(self):
        # When model == market, edge is 0 — side label doesn't matter much
        # but we want deterministic behavior; current impl returns YES.
        side, edge = determine_side(0.50, 0.50)
        assert side == "YES"
        assert edge == 0.0


class TestOdds:
    def test_yes_odds_at_60_cents(self):
        # Buying YES at $0.60: pays $0.40 per dollar risked
        assert _odds_for_side(0.60, "YES") == pytest.approx(2/3, abs=1e-6)

    def test_no_odds_at_60_cents_yes(self):
        # NO is at $0.40: pays $0.60 per dollar risked
        assert _odds_for_side(0.60, "NO") == pytest.approx(1.5, abs=1e-6)

    def test_zero_price_returns_zero(self):
        assert _odds_for_side(0.0, "YES") == 0.0

    def test_one_price_returns_zero(self):
        assert _odds_for_side(1.0, "YES") == 0.0


class TestAnalyzeContract:
    BASE_CONTRACT = {
        "contract_id":     "0xtest",
        "question":        "Will it snow more than 2 inches in NYC on March 20, 2027?",
        "yes_price":       0.45,
        "no_price":        0.55,
        "yes_token_id":    "tok_yes",
        "no_token_id":     "tok_no",
        "liquidity_usd":   5000.0,
        "resolution_date": "2027-03-20T23:59:59Z",
    }

    def test_returns_signal_when_strong_edge(self):
        with patch.object(edge_mod, "get_ensemble_probability", return_value={
            "probability":  0.62,    # 62% vs 45% market = +17pp YES edge
            "sources":      [{"source": "nws", "probability": 0.62, "confidence": 0.85}],
            "disagreement": 0.05,
            "n_sources":    1,
        }):
            sig = analyze_contract(self.BASE_CONTRACT, bankroll=1000.0)
        assert sig is not None
        assert sig["recommended_side"] == "YES"
        assert sig["edge"] == pytest.approx(0.17, abs=1e-3)
        assert sig["model_p"] == 0.62
        assert sig["market_p"] == 0.45
        assert sig["kelly_size"] > 0

    def test_returns_none_when_disagreement_too_high(self):
        with patch.object(edge_mod, "get_ensemble_probability", return_value={
            "probability":  0.62,
            "sources":      [],
            "disagreement": 0.30,    # > MAX_SOURCE_DISAGREEMENT (0.15)
            "n_sources":    3,
        }):
            sig = analyze_contract(self.BASE_CONTRACT, bankroll=1000.0)
        assert sig is None

    def test_returns_none_when_edge_below_threshold(self):
        with patch.object(edge_mod, "get_ensemble_probability", return_value={
            "probability":  0.46,    # only 1pp from market — below 7pp threshold
            "sources":      [],
            "disagreement": 0.05,
            "n_sources":    1,
        }):
            sig = analyze_contract(self.BASE_CONTRACT, bankroll=1000.0)
        assert sig is None

    def test_returns_none_when_no_ensemble_data(self):
        with patch.object(edge_mod, "get_ensemble_probability", return_value=None):
            sig = analyze_contract(self.BASE_CONTRACT, bankroll=1000.0)
        assert sig is None

    def test_returns_none_when_unparseable_question(self):
        bad = dict(self.BASE_CONTRACT, question="Will Bitcoin hit $1M before 2027?")
        # parse_contract_metadata will return None — no city, no weather variable
        sig = analyze_contract(bad, bankroll=1000.0)
        assert sig is None
