# global_temp_model.py
"""
Probability estimator for annual global temperature ranking markets.
e.g. "Will 2026 be the hottest year on record?"

Strategy:
  1. Pull Berkeley Earth annual temperature anomalies (free, no auth)
  2. Get current year's month-to-date anomaly from NOAA
  3. Use ENSO state to adjust forecast for remaining months
  4. Estimate P(2026 rank = N) using ensemble of recent trends

Resolution source: most Polymarket temperature ranking markets resolve
against Berkeley Earth or NASA GISS annual global temperature anomaly data.
"""

import re
import logging
import statistics
import time
from datetime import date

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BERKELEY_URL = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt"

# Module-level cache so we only fetch Berkeley data once per process run
# (or at most once per 12 hours — refreshed if stale)
_BERKELEY_CACHE: dict[int, float] = {}
_BERKELEY_CACHE_TIME: float = 0.0
_BERKELEY_CACHE_TTL = 43200  # 12 hours in seconds


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_berkeley_annual_anomalies() -> dict[int, float]:
    """
    Fetch Berkeley Earth annual global temperature anomalies.
    Returns {year: anomaly_celsius} for all available years.
    Cached in-memory for 12 hours to avoid repeated downloads within a scan.
    """
    global _BERKELEY_CACHE, _BERKELEY_CACHE_TIME
    now = time.time()
    if _BERKELEY_CACHE and (now - _BERKELEY_CACHE_TIME) < _BERKELEY_CACHE_TTL:
        return _BERKELEY_CACHE

    resp = httpx.get(BERKELEY_URL, timeout=15)
    resp.raise_for_status()

    anomalies: dict[int, float] = {}
    for line in resp.text.splitlines():
        parts = line.split()
        if not parts or len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            if year < 1850:
                continue
            anomaly = float(parts[1])
            if anomaly != anomaly:  # NaN check
                continue
            anomalies[year] = round(anomaly, 4)
        except (ValueError, IndexError):
            continue

    if anomalies:
        _BERKELEY_CACHE = anomalies
        _BERKELEY_CACHE_TIME = now
        logger.info(f"Berkeley Earth data cached: {len(anomalies)} years, latest={max(anomalies.keys())}")

    return anomalies


def classify_temp_market(question: str) -> dict | None:
    """
    Parse an annual temperature ranking question.
    Returns {"rank": int, "year": int, "direction": "hottest"|"coldest"}
    or None if not parseable.

    Examples:
      "Will 2026 be the hottest year on record?" -> {"rank":1, "year":2026, "direction":"hottest"}
      "Will 2026 be the second-hottest year on record?" -> {"rank":2, ...}
      "Will 2026 rank as the sixth-hottest year on record or lower?" -> {"rank":6, "direction":"sixth_or_lower"}
    """
    if not question:
        return None
    q = question.lower()

    # Must mention a year and some temperature ranking concept
    if "hottest" not in q and "coldest" not in q and "warmest" not in q:
        return None
    if "year on record" not in q and "year" not in q:
        return None

    # Extract year
    year_m = re.search(r"\b(20\d{2})\b", question)
    if not year_m:
        return None
    year = int(year_m.group(1))

    direction = "hottest" if ("hottest" in q or "warmest" in q) else "coldest"

    # Extract rank
    ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
        "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    }
    rank = 1  # default: "hottest" = first hottest
    for word, n in ordinals.items():
        if word in q:
            rank = n
            break

    # Check for "sixth or lower" style
    is_lower_bound = "or lower" in q or "or below" in q

    return {
        "year":       year,
        "rank":       rank,
        "direction":  direction,
        "lower_bound": is_lower_bound,
    }


def estimate_temp_ranking_probability(
    year: int,
    rank: int,
    direction: str = "hottest",
    lower_bound: bool = False,
    oni: float | None = None,
) -> dict | None:
    """
    Estimate P(year ranks exactly Nth hottest) using historical data + ENSO.

    For "sixth or lower" markets, estimate P(rank >= 6).

    Returns a source-result dict compatible with the signal pipeline,
    or None on failure.
    """
    try:
        anomalies = get_berkeley_annual_anomalies()
    except Exception as e:
        logger.warning(f"Berkeley Earth fetch failed: {e}")
        return None

    if not anomalies:
        return None

    # Estimate the current year's likely final anomaly.
    #
    # IMPORTANT: Do NOT use a naive 5-year average — 2023 and 2024 were the two
    # hottest years on record driven by a strong El Niño. A neutral-ENSO year like
    # 2026 is likely to be 0.15-0.20°C cooler than those peak years.
    #
    # Better approach: use the long-term warming trend + ENSO adjustment.
    # 1. Fit a linear trend through the last 10 years (removes single-year noise)
    # 2. Project to the target year
    # 3. Apply ENSO adjustment on top of the trend value
    sorted_years = sorted(anomalies.keys())
    recent_10 = [(y, anomalies[y]) for y in sorted_years[-10:]]

    if len(recent_10) >= 4:
        # Simple linear regression (least squares)
        n = len(recent_10)
        xs = [y for y, _ in recent_10]
        ys = [v for _, v in recent_10]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, ys))
        den = sum((xi - x_mean) ** 2 for xi in xs)
        slope = num / den if den > 0 else 0.0
        intercept = y_mean - slope * x_mean
        trend_estimate = intercept + slope * year
        base_recent_avg = trend_estimate
    else:
        # Fallback if not enough data
        recent = [anomalies[y] for y in sorted_years[-5:]]
        base_recent_avg = statistics.mean(recent)

    # ENSO adjustment: El Niño (+) warms global temps, La Niña (-) cools them
    # Rough effect: ±0.12°C per unit of ONI on annual anomaly (empirically derived)
    if oni is not None:
        enso_adjustment = oni * 0.12
    else:
        enso_adjustment = 0.0

    # For 2026 specifically — ONI is currently near-neutral, trending back toward neutral
    # 2024 was El Niño peak (+1.2°C anomaly), 2025 was La Niña, 2026 is early season
    estimated_anomaly = base_recent_avg + enso_adjustment

    # Create a simple distribution around the estimate using historical variance
    all_anomalies = list(anomalies.values())
    std_dev = statistics.stdev(all_anomalies[-20:])  # use recent volatility

    # Simulate 1000 possible final anomalies for this year
    import random
    random.seed(42)  # reproducible
    simulated = [
        estimated_anomaly + random.gauss(0, std_dev * 0.6)  # tighter than full historical std
        for _ in range(1000)
    ]

    # Get the ranking record (sorted descending = hottest first)
    historical = [(y, v) for y, v in anomalies.items() if y != year]
    historical_sorted = sorted(historical, key=lambda x: -x[1])

    # Count how often this year's simulated anomaly would achieve the target rank
    in_target = 0
    for sim_val in simulated:
        # How does sim_val rank among historical years?
        n_hotter = sum(1 for _, v in historical_sorted if v > sim_val)
        sim_rank = n_hotter + 1  # rank 1 = hottest

        if lower_bound:
            # "sixth or lower" = rank >= 6
            if sim_rank >= rank:
                in_target += 1
        else:
            if sim_rank == rank:
                in_target += 1

    prob = in_target / len(simulated)

    return {
        "source":      "global_temp_model",
        "probability": round(prob, 4),
        "confidence":  0.45,  # moderate — annual temp models have real uncertainty
        "details": {
            "estimated_anomaly": round(estimated_anomaly, 3),
            "base_recent_avg":   round(base_recent_avg, 3),
            "enso_adjustment":   round(enso_adjustment, 3),
            "hottest_on_record": max(all_anomalies),
            "hottest_year":      max(anomalies, key=anomalies.get),
            "target_rank":       rank,
            "lower_bound":       lower_bound,
        },
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    anomalies = get_berkeley_annual_anomalies()
    print(f"Berkeley Earth: {len(anomalies)} years of data")
    sorted_years = sorted(anomalies.items(), key=lambda x: -x[1])
    print("Top 10 hottest:")
    for rank, (yr, anomaly) in enumerate(sorted_years[:10], 1):
        print(f"  #{rank:2d}: {yr} = +{anomaly:.3f}°C")

    print()
    print("Market probability estimates for 2026:")
    markets = [
        (1, "hottest",  False, "Will 2026 be the hottest year on record?",          0.345),
        (2, "hottest",  False, "Will 2026 be the second-hottest year on record?",    0.555),
        (3, "hottest",  False, "Will 2026 be the third-hottest year on record?",     0.028),
        (4, "hottest",  False, "Will 2026 be the fourth-hottest year on record?",    0.044),
        (5, "hottest",  False, "Will 2026 be the fifth-hottest year on record?",     0.005),
        (6, "hottest",  True,  "Will 2026 rank as the sixth-hottest or lower?",      0.038),
    ]
    print(f"\n{'Question':<50s} {'model':>6s} {'market':>7s} {'edge':>7s}")
    print("-" * 75)
    for rank, direction, lower, q, mkt in markets:
        r = estimate_temp_ranking_probability(2026, rank, direction, lower)
        if r:
            edge = r["probability"] - mkt
            print(f"{q:<50s} {r['probability']:>6.3f} {mkt:>7.3f} {edge:>+7.3f}")
        else:
            print(f"{q:<50s} (failed)")
