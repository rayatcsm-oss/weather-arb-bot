# tests/test_weather.py
"""Unit tests for weather.py — ensemble aggregation logic with mocked sources."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

import pytest
from unittest.mock import patch
import weather as weather_mod
from weather import get_ensemble_probability


def _src(name, p, conf=0.8):
    return {"source": name, "probability": p, "confidence": conf}


class TestEnsemble:

    def test_single_source(self):
        with patch.object(weather_mod, "get_nws_probability",         return_value=_src("nws", 0.6)), \
             patch.object(weather_mod, "get_openmeteo_probability",   return_value=None), \
             patch.object(weather_mod, "get_noaa_probability",        return_value=None), \
             patch.object(weather_mod, "get_tomorrowio_probability",  return_value=None):
            result = get_ensemble_probability(40.7, -74.0, "2027-01-01", "rain")
        assert result is not None
        assert result["probability"] == 0.6
        assert result["n_sources"] == 1
        assert result["disagreement"] == 0.0  # std dev of single value

    def test_all_sources_fail_returns_none(self):
        with patch.object(weather_mod, "get_nws_probability",         return_value=None), \
             patch.object(weather_mod, "get_openmeteo_probability",   return_value=None), \
             patch.object(weather_mod, "get_noaa_probability",        return_value=None), \
             patch.object(weather_mod, "get_tomorrowio_probability",  return_value=None):
            result = get_ensemble_probability(40.7, -74.0, "2027-01-01", "rain")
        assert result is None

    def test_high_disagreement_reported(self):
        with patch.object(weather_mod, "get_nws_probability",         return_value=_src("nws", 0.10)), \
             patch.object(weather_mod, "get_openmeteo_probability",   return_value=_src("openmeteo", 0.90)), \
             patch.object(weather_mod, "get_noaa_probability",        return_value=None), \
             patch.object(weather_mod, "get_tomorrowio_probability",  return_value=None):
            result = get_ensemble_probability(40.7, -74.0, "2027-01-01", "rain")
        assert result is not None
        assert result["disagreement"] > 0.5    # huge spread between 10% and 90%

    def test_brier_weighting_favors_high_weight_source(self):
        """NWS has weight 0.40, NOAA has 0.15. Ensemble should lean toward NWS."""
        # NWS says 0.20, NOAA says 0.80 — ensemble should be much closer to 0.20
        with patch.object(weather_mod, "get_nws_probability",         return_value=_src("nws", 0.20)), \
             patch.object(weather_mod, "get_openmeteo_probability",   return_value=None), \
             patch.object(weather_mod, "get_noaa_probability",        return_value=_src("noaa", 0.80)), \
             patch.object(weather_mod, "get_tomorrowio_probability",  return_value=None):
            result = get_ensemble_probability(40.7, -74.0, "2027-01-01", "rain")
        assert result is not None
        # weighted average:  (0.20 * 0.40 + 0.80 * 0.15) / (0.40 + 0.15) = 0.20 / 0.55 ≈ 0.3636
        # The midpoint would be 0.50 — we should be well below that
        assert result["probability"] < 0.45
        assert result["probability"] > 0.30   # but not artificially low

    def test_probability_clamped_to_unit_interval(self):
        """If a fetcher returns >1 due to a bug, ensemble should still clamp."""
        with patch.object(weather_mod, "get_nws_probability",         return_value=_src("nws", 1.5)), \
             patch.object(weather_mod, "get_openmeteo_probability",   return_value=None), \
             patch.object(weather_mod, "get_noaa_probability",        return_value=None), \
             patch.object(weather_mod, "get_tomorrowio_probability",  return_value=None):
            result = get_ensemble_probability(40.7, -74.0, "2027-01-01", "rain")
        assert result is not None
        assert 0.0 <= result["probability"] <= 1.0


class TestParseHelpers:
    """Verify the polymarket question parser handles common patterns."""

    def test_synthetic_snow_question(self):
        from polymarket import parse_contract_metadata
        meta = parse_contract_metadata({
            "question": "Will it snow more than 2 inches in NYC on March 20, 2026?",
            "resolution_date": "2026-03-20T23:59:59Z",
        })
        assert meta is not None
        assert meta["variable"] == "snow"
        assert meta["threshold"] == 2.0
        assert meta["unit"] == "inches"
        assert meta["date"] == "2026-03-20"
        # NYC area coords (airport ASOS stations are slightly outside city center)
        assert 40.5 < meta["lat"] < 41.0
        assert -74.5 < meta["lon"] < -73.5

    def test_synthetic_temp_question(self):
        from polymarket import parse_contract_metadata
        meta = parse_contract_metadata({
            "question": "Will Chicago's high temperature exceed 85 degrees on July 4, 2026?",
            "resolution_date": "2026-07-04T23:59:59Z",
        })
        assert meta is not None
        assert meta["variable"] == "temp_high"
        assert meta["threshold"] == 85.0
        assert meta["unit"] == "fahrenheit"

    def test_unparseable_returns_none(self):
        from polymarket import parse_contract_metadata
        # No city, no weather variable
        meta = parse_contract_metadata({
            "question": "Will Bitcoin hit $1M before 2030?",
            "resolution_date": "2030-01-01T00:00:00Z",
        })
        assert meta is None
