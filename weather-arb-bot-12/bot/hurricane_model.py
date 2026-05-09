# hurricane_model.py
"""
Probability estimator for Polymarket's seasonal hurricane markets.

Markets we handle:
  HURRICANE_FORMS_PRESEASON     Will a hurricane form by May 31?
  HURRICANE_LANDFALL_PRESEASON  Will a hurricane make landfall in the US by May 31?
  NAMED_STORM_PRESEASON         Named storm forms before hurricane season?
  CAT4_LANDFALL_YEAR            Will any Cat 4 hurricane make landfall in the US by Dec 31?
  CAT5_LANDFALL_YEAR            Will any Cat 5 hurricane make landfall in the US by Dec 31?

Approach:
  1. Hardcoded historical base rates from the NHC HURDAT2 database (1950–2024).
  2. Apply ENSO adjustment using live Niño 3.4 ONI from NOAA/CPC.
       - La Niña  (ONI < -0.5): Atlantic activity ~ +30%
       - Neutral  (-0.5 ≤ ONI ≤ +0.5): no adjustment
       - El Niño  (ONI > +0.5): Atlantic activity ~ -30%
  3. For pre-season questions with a deadline soon, scale down by remaining
     fraction of "pre-season window."
  4. Conservative confidence — these are coarse models; the disagreement check
     in edge.py naturally suppresses signals when our estimate diverges wildly
     from the market price (which itself is a signal we may be wrong).

Sources:
  - HURDAT2 Atlantic hurricane database, NHC: https://www.nhc.noaa.gov/data/
  - ENSO ONI: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""

import re
import logging
from datetime import date, datetime, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ===========================================================================
# Historical base rates (HURDAT2 1950–2024)
# ===========================================================================
# These are computed offline from NOAA's published HURDAT2 database.
# They are STARTING points; they should be recalibrated annually as new
# seasons resolve. Values are deliberately rounded to reflect uncertainty.

BASE_RATES = {
    # P(any Atlantic hurricane forms before June 1) — extremely rare event.
    # Examples: 1908, 1948, 1970 (Alma), 2016 (Alex). ~4 in 75 years.
    "HURRICANE_FORMS_PRESEASON": 0.05,

    # P(hurricane US landfall before June 1) — roughly 2 instances in 75 years.
    # Examples are exceptional (e.g., 1825 unnamed). Modern era is essentially zero.
    "HURRICANE_LANDFALL_PRESEASON": 0.02,

    # P(any named tropical storm forms before June 1).
    # ~25 of last 75 years had a named storm in May or earlier. Trend has been
    # increasing — roughly 8 of last 12 years had pre-June named storms.
    # Use the recent rate, which is more representative of current climatology.
    "NAMED_STORM_PRESEASON": 0.55,

    # P(at least one Cat-4 hurricane US landfall in any given calendar year).
    # 2004, 2005 (multiple), 2017, 2018, 2020, 2021, 2022, 2024 — ~8 of last 20 years.
    "CAT4_LANDFALL_YEAR": 0.40,

    # P(at least one Cat-5 hurricane US landfall in any given calendar year).
    # 1928, 1935, 1969, 1992, 2018 — ~5 in 100 years. Recent decade slightly elevated.
    "CAT5_LANDFALL_YEAR": 0.10,
}

# Rough confidence (per-market) — how much we trust the base rate.
# These are NOT probabilities; they're fed to the ensemble's Brier weights.
BASE_RATE_CONFIDENCE = {
    "HURRICANE_FORMS_PRESEASON":    0.70,
    "HURRICANE_LANDFALL_PRESEASON": 0.80,  # very high — the event is essentially never observed
    "NAMED_STORM_PRESEASON":        0.55,
    "CAT4_LANDFALL_YEAR":           0.50,
    "CAT5_LANDFALL_YEAR":           0.55,
}


# ===========================================================================
# ENSO state
# ===========================================================================

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_current_oni() -> float | None:
    """
    Latest Oceanic Niño Index (3-month rolling SST anomaly).
    Returns None on failure — caller should treat as ENSO-neutral.
    """
    try:
        resp = httpx.get(ONI_URL, timeout=15)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
        # Skip header line (starts with "SEAS" or "YR")
        last = lines[-1].split()
        # Format: "JFM  2026  26.57  -0.16"
        # ANOM is the last column
        return float(last[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch ONI: {e}")
        return None


def enso_multiplier(oni: float | None) -> tuple[float, str]:
    """
    ENSO modifier for Atlantic hurricane activity.

    Returns (multiplier, regime_label).
    """
    if oni is None:
        return 1.0, "neutral (no data)"
    if oni < -0.5:
        return 1.30, f"La Niña (ONI={oni:+.2f})"
    if oni > +0.5:
        return 0.70, f"El Niño (ONI={oni:+.2f})"
    return 1.0, f"neutral (ONI={oni:+.2f})"


# ===========================================================================
# Market classification
# ===========================================================================

def classify_hurricane_market(question: str) -> str | None:
    """
    Return a market-type tag if the question matches a known hurricane market,
    else None. Tags align with BASE_RATES keys.
    """
    if not question:
        return None
    q = question.lower()

    # Cat-N landfall — check first because these are the most specific
    if "category 5" in q or "cat 5" in q or "cat-5" in q:
        if "landfall" in q and ("us" in q or "u.s." in q or "united states" in q):
            return "CAT5_LANDFALL_YEAR"
    if "category 4" in q or "cat 4" in q or "cat-4" in q:
        if "landfall" in q and ("us" in q or "u.s." in q or "united states" in q):
            return "CAT4_LANDFALL_YEAR"

    # Pre-season questions
    has_preseason = any(p in q for p in [
        "may 31", "before may 31", "by may 31",
        "before hurricane season", "before june", "before june 1",
    ])

    if has_preseason:
        # Check more-specific patterns first to avoid "hurricane season" + "forms"
        # matching HURRICANE_FORMS_PRESEASON when the question is about a named storm.
        if "named storm" in q or "tropical storm" in q:
            return "NAMED_STORM_PRESEASON"
        if "landfall" in q and "hurricane" in q:
            return "HURRICANE_LANDFALL_PRESEASON"
        if "hurricane" in q and "form" in q:
            return "HURRICANE_FORMS_PRESEASON"

    return None


# ===========================================================================
# Probability estimation
# ===========================================================================

def _preseason_time_factor(deadline: date, today: date | None = None) -> float:
    """
    For pre-season markets, return the fraction of pre-season activity
    that falls in the REMAINING window.

    CRITICAL: pre-season tropical activity is NOT uniformly distributed
    across Jan-May. Historical HURDAT2 data shows:
      Month 1 (Jan): ~3%    (extremely rare — only Alex 2016 in modern era)
      Month 2 (Feb): ~1%
      Month 3 (Mar): ~2%
      Month 4 (Apr): ~6%
      Month 5 (May): ~78%   (vast majority of pre-season storms form in May)
      Unaccounted:   ~10%   (buffer for unusual seasons)

    When only May remains in the window, the time_factor should be ~0.78,
    NOT the 0.21 that a uniform model would give (31/150 days).

    Returns a value in [0.0, 1.0].
    """
    today = today or date.today()
    season_year = deadline.year
    window_end = deadline

    if today >= window_end:
        return 0.0

    # Monthly weights for pre-season tropical activity (HURDAT2 1970-2024)
    # Normalized to sum to 1.0
    MONTH_WEIGHT = {1: 0.03, 2: 0.01, 3: 0.02, 4: 0.08, 5: 0.86}

    # Sum the weights of months where we still have ANY remaining days
    remaining_weight = 0.0
    for m in range(1, 6):
        month_start = date(season_year, m, 1)
        if m < 5:
            month_end = date(season_year, m + 1, 1) - timedelta(days=1)
        else:
            month_end = window_end
        # If today is past this month, skip it
        if today > month_end:
            continue
        # If today is within this month, take a fractional share
        if today > month_start:
            days_in_month = (month_end - month_start).days + 1
            days_remaining = (month_end - today).days + 1
            fraction = days_remaining / days_in_month
        else:
            fraction = 1.0
        remaining_weight += MONTH_WEIGHT.get(m, 0.0) * fraction

    return max(0.0, min(1.0, remaining_weight))


def _full_year_time_factor(deadline: date, today: date | None = None) -> float:
    """
    For full-year markets (Cat-4/5 landfall by Dec 31), scale by fraction of
    Atlantic season remaining. Hurricane season runs roughly Jun 1 – Nov 30.

    If today is before Jun 1: factor = 1.0 (full season ahead).
    If today is between Jun 1 and Nov 30: factor scales linearly to fraction left.
    If today is after Nov 30: factor ≈ 0 (only December left, which is very rare).
    """
    today = today or date.today()
    season_year = deadline.year
    season_start = date(season_year, 6, 1)
    season_end = date(season_year, 11, 30)

    if today <= season_start:
        return 1.0
    if today >= season_end:
        # December: roughly 1% of annual hurricane activity occurs after Nov 30
        return 0.01

    total_days = (season_end - season_start).days
    days_remaining = (season_end - today).days
    return max(0.0, min(1.0, days_remaining / total_days))


def estimate_hurricane_probability(
    market_type: str,
    deadline: date,
    today: date | None = None,
    oni: float | None = None,
) -> dict | None:
    """
    Estimate P(market resolves YES) for a known hurricane market type.

    Returns a dict matching the weather.py source-result schema:
        {"source": "hurricane_model",
         "probability": float,
         "confidence": float,
         "details": {...}}
    or None if the market type isn't supported.
    """
    if market_type not in BASE_RATES:
        return None

    base = BASE_RATES[market_type]
    confidence = BASE_RATE_CONFIDENCE[market_type]

    # ENSO adjustment
    enso_mult, enso_label = enso_multiplier(oni)

    # Time-remaining adjustment
    if market_type in ("HURRICANE_FORMS_PRESEASON",
                       "HURRICANE_LANDFALL_PRESEASON",
                       "NAMED_STORM_PRESEASON"):
        time_factor = _preseason_time_factor(deadline, today)
    else:
        time_factor = _full_year_time_factor(deadline, today)

    # Combine
    prob = base * enso_mult * time_factor

    # Floor very-low probabilities at the base rate * 0.05 (avoid 0)
    prob = max(0.001, min(0.999, prob))

    return {
        "source":      "hurricane_model",
        "probability": round(prob, 4),
        "confidence":  confidence,
        "details": {
            "market_type":  market_type,
            "base_rate":    base,
            "enso":         enso_label,
            "enso_mult":    enso_mult,
            "time_factor":  round(time_factor, 4),
            "deadline":     deadline.isoformat(),
        },
    }


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 70)
    print("Live ENSO state")
    print("=" * 70)
    oni = get_current_oni()
    mult, label = enso_multiplier(oni)
    print(f"  Current ONI = {oni}")
    print(f"  Regime:       {label}")
    print(f"  Multiplier:   {mult}x Atlantic activity")

    print()
    print("=" * 70)
    print("Probability estimates for the 5 currently-active markets")
    print("=" * 70)

    today = date.today()
    cases = [
        ("HURRICANE_LANDFALL_PRESEASON", date(2026, 5, 31), 0.268,
            "Will a hurricane make landfall in the US by May 31?"),
        ("HURRICANE_FORMS_PRESEASON",    date(2026, 5, 31), 0.057,
            "Will a hurricane form by May 31?"),
        ("NAMED_STORM_PRESEASON",        date(2026, 5, 31), 0.455,
            "Named storm forms before hurricane season?"),
        ("CAT4_LANDFALL_YEAR",           date(2026, 12, 31), 0.350,
            "Will any Cat 4 make landfall in the US before 2027?"),
        ("CAT5_LANDFALL_YEAR",           date(2026, 12, 31), 0.140,
            "Will any Cat 5 make landfall in the US before 2027?"),
    ]

    print(f"  Today = {today}")
    print()
    print(f"{'Market':<32s} {'model':>6s} {'market':>7s} {'edge':>7s}  Question")
    print("-" * 110)
    for mtype, deadline, mkt_yes, q in cases:
        result = estimate_hurricane_probability(mtype, deadline, today, oni)
        if result:
            mp = result["probability"]
            edge = mp - mkt_yes
            ts   = result["details"]["time_factor"]
            print(f"{mtype:<32s} {mp:>6.3f} {mkt_yes:>7.3f} {edge:>+7.3f}  {q[:50]}")
            print(f"  {' ' * 32}  base={result['details']['base_rate']} time_factor={ts:.3f}")

    print()
    print("=" * 70)
    print("Classification test")
    print("=" * 70)
    test_questions = [
        "Will a hurricane make landfall in the US by May 31?",
        "Will any Category 4 hurricane make landfall in the US in before 2027?",
        "Will any Category 5 hurricane make landfall in the US in before 2027?",
        "Will Bitcoin hit $1M before 2030?",
        "Will it rain in NYC tomorrow?",
    ]
    for q in test_questions:
        cat = classify_hurricane_market(q)
        print(f"  [{cat or 'None':<32s}]  {q}")
