# tests/test_hurricane.py
"""Unit tests for hurricane_model.py — base rates, ENSO, time decay, classification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

import pytest
from datetime import date
from hurricane_model import (
    BASE_RATES,
    classify_hurricane_market,
    enso_multiplier,
    estimate_hurricane_probability,
    _preseason_time_factor,
    _full_year_time_factor,
)


class TestClassify:
    def test_hurricane_landfall_preseason(self):
        assert classify_hurricane_market(
            "Will a hurricane make landfall in the US by May 31?"
        ) == "HURRICANE_LANDFALL_PRESEASON"

    def test_hurricane_forms_preseason(self):
        assert classify_hurricane_market(
            "Will a hurricane form by May 31?"
        ) == "HURRICANE_FORMS_PRESEASON"

    def test_named_storm_preseason(self):
        assert classify_hurricane_market(
            "Named storm forms before hurricane season?"
        ) == "NAMED_STORM_PRESEASON"

    def test_cat4_landfall(self):
        assert classify_hurricane_market(
            "Will any Category 4 hurricane make landfall in the US in before 2027?"
        ) == "CAT4_LANDFALL_YEAR"

    def test_cat5_landfall(self):
        assert classify_hurricane_market(
            "Will any Category 5 hurricane make landfall in the US in before 2027?"
        ) == "CAT5_LANDFALL_YEAR"

    def test_unrelated_question_returns_none(self):
        assert classify_hurricane_market("Will Bitcoin hit $1M before 2030?") is None
        assert classify_hurricane_market("Will it rain in NYC?") is None

    def test_carolina_hurricanes_hockey_not_classified(self):
        """A common false-positive guard."""
        assert classify_hurricane_market(
            "Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?"
        ) is None


class TestEnsoMultiplier:
    def test_la_nina_increases_activity(self):
        mult, label = enso_multiplier(-0.8)
        assert mult > 1.0
        assert "Niña" in label

    def test_el_nino_decreases_activity(self):
        mult, label = enso_multiplier(+0.8)
        assert mult < 1.0
        assert "Niño" in label

    def test_neutral_returns_one(self):
        mult, label = enso_multiplier(-0.16)
        assert mult == 1.0
        assert "neutral" in label

    def test_none_returns_neutral(self):
        mult, label = enso_multiplier(None)
        assert mult == 1.0
        assert "no data" in label or "neutral" in label


class TestTimeFactors:
    def test_preseason_decays_to_zero_at_deadline(self):
        deadline = date(2026, 5, 31)
        # Day before deadline: factor is small (only ~1 day of May left)
        factor = _preseason_time_factor(deadline, today=date(2026, 5, 30))
        assert factor < 0.10  # small but not zero — May 30-31 still has activity weight
        # On or after deadline: zero
        assert _preseason_time_factor(deadline, today=date(2026, 5, 31)) == 0.0

    def test_preseason_full_at_year_start(self):
        factor = _preseason_time_factor(date(2026, 5, 31), today=date(2026, 1, 1))
        assert factor > 0.95  # all months available, weights sum to ~1.0

    def test_full_year_factor_full_before_season(self):
        # April: full season ahead
        factor = _full_year_time_factor(date(2026, 12, 31), today=date(2026, 4, 28))
        assert factor == 1.0

    def test_full_year_factor_minimal_after_season(self):
        # December: only ~1% of activity after Nov 30
        factor = _full_year_time_factor(date(2026, 12, 31), today=date(2026, 12, 15))
        assert factor < 0.05


class TestEstimateProbability:
    def test_returns_none_for_unknown_type(self):
        assert estimate_hurricane_probability(
            "INVALID_TYPE", date(2026, 5, 31)
        ) is None

    def test_preseason_probability_below_base_rate_late_in_window(self):
        """Late April pre-May-31 question should be well below the annual base rate."""
        result = estimate_hurricane_probability(
            "HURRICANE_LANDFALL_PRESEASON",
            deadline=date(2026, 5, 31),
            today=date(2026, 4, 28),
            oni=-0.16,
        )
        assert result is not None
        # Base rate is 0.02; we're 80% through the window so we should be << 0.02
        assert result["probability"] < BASE_RATES["HURRICANE_LANDFALL_PRESEASON"]
        assert result["probability"] > 0  # but not zero
        assert result["source"] == "hurricane_model"

    def test_full_year_probability_at_base_rate_pre_season(self):
        """Cat-4 question in late April: full season ahead, prob ~= base rate (no ENSO adj)."""
        result = estimate_hurricane_probability(
            "CAT4_LANDFALL_YEAR",
            deadline=date(2026, 12, 31),
            today=date(2026, 4, 28),
            oni=-0.16,
        )
        assert result is not None
        assert result["probability"] == pytest.approx(BASE_RATES["CAT4_LANDFALL_YEAR"], abs=0.01)

    def test_la_nina_boosts_probability(self):
        """Same market, La Niña ONI: probability should be higher."""
        neutral = estimate_hurricane_probability(
            "CAT4_LANDFALL_YEAR", date(2026, 12, 31), today=date(2026, 4, 28), oni=-0.16,
        )
        la_nina = estimate_hurricane_probability(
            "CAT4_LANDFALL_YEAR", date(2026, 12, 31), today=date(2026, 4, 28), oni=-1.0,
        )
        assert la_nina["probability"] > neutral["probability"]

    def test_el_nino_dampens_probability(self):
        neutral = estimate_hurricane_probability(
            "CAT4_LANDFALL_YEAR", date(2026, 12, 31), today=date(2026, 4, 28), oni=-0.16,
        )
        el_nino = estimate_hurricane_probability(
            "CAT4_LANDFALL_YEAR", date(2026, 12, 31), today=date(2026, 4, 28), oni=+1.5,
        )
        assert el_nino["probability"] < neutral["probability"]
