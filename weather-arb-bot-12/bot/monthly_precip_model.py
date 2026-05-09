# monthly_precip_model.py
"""
Probability estimator for "Will [city] have [X-Y] inches of precipitation
in [month]?" markets on Polymarket.

Strategy
--------
For each market:
  1. Pull OBSERVED daily precipitation from Open-Meteo's archive API
     (free, no auth) for the days of the month already past.
  2. Pull a FORECAST for the remaining days from Open-Meteo's ensemble API
     to get a distribution, not just a point estimate.
  3. Total monthly precipitation = observed + forecast_remaining.
     Observed is fixed; forecast carries uncertainty.
  4. Compute P(total falls in bucket [low_mm, high_mm]) by counting
     ensemble member outcomes that land in the bucket.

This gives the bot a structural edge: late in the month, observed
precipitation dominates and most uncertainty is gone — but retail
traders frequently misprice the remaining tail risk.
"""

import logging
import statistics
import time
from datetime import date, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ARCHIVE_BASE  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_BASE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# In-memory cache to prevent hitting rate limits on Open-Meteo APIs.
# Observed data cache: 6 hours TTL (archive doesn't update intraday)
# Forecast cache: 2 hours TTL (forecasts update more frequently)
_ARCHIVE_CACHE: dict[str, tuple[float, list[float]]] = {}  # key -> (timestamp, data)
_FORECAST_CACHE: dict[str, tuple[float, list[float]]] = {}
_ARCHIVE_TTL = 21600   # 6 hours
_FORECAST_TTL = 7200   # 2 hours

# Daily request counter to stay well under Open-Meteo's free tier limit
# Open-Meteo free tier: 10,000 requests/day. We cap at 200 archive calls/day
# to leave headroom for other API users and future market growth.
_ARCHIVE_DAILY_COUNT: int = 0
_ARCHIVE_DAILY_DATE: str = ""
_ARCHIVE_DAILY_LIMIT: int = 200


def _check_archive_rate_limit() -> bool:
    """Returns True if we're under the daily archive request limit."""
    global _ARCHIVE_DAILY_COUNT, _ARCHIVE_DAILY_DATE
    today = date.today().isoformat()
    if _ARCHIVE_DAILY_DATE != today:
        _ARCHIVE_DAILY_COUNT = 0
        _ARCHIVE_DAILY_DATE = today
    if _ARCHIVE_DAILY_COUNT >= _ARCHIVE_DAILY_LIMIT:
        logger.warning(
            f"Open-Meteo archive: daily limit of {_ARCHIVE_DAILY_LIMIT} requests reached. "
            f"Using cached data only. Resets at midnight."
        )
        return False
    _ARCHIVE_DAILY_COUNT += 1
    return True


def _cache_key(lat: float, lon: float, start: date, end: date) -> str:
    return f"{lat:.4f}_{lon:.4f}_{start}_{end}"


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=2, max=8))
def get_observed_precipitation(
    lat: float, lon: float,
    start: date, end: date,
    timezone: str = "auto",
) -> list[float]:
    """Pull observed daily precipitation totals (mm) from Open-Meteo archive."""
    if start > end:
        return []

    # Check cache first
    key = _cache_key(lat, lon, start, end)
    if key in _ARCHIVE_CACHE:
        ts, cached = _ARCHIVE_CACHE[key]
        if time.time() - ts < _ARCHIVE_TTL:
            return cached

    # Check daily rate limit before making the API call
    if not _check_archive_rate_limit():
        # Return cached data even if stale rather than hitting the rate limit
        if key in _ARCHIVE_CACHE:
            _, cached = _ARCHIVE_CACHE[key]
            return cached
        return []

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      "precipitation_sum",
        "timezone":   timezone,
    }
    resp = httpx.get(ARCHIVE_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    result = [v for v in (daily.get("precipitation_sum") or []) if v is not None]

    # Cache the result
    _ARCHIVE_CACHE[key] = (time.time(), result)
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_forecast_ensemble_total(
    lat: float, lon: float,
    start: date, end: date,
    timezone: str = "auto",
) -> list[float] | None:
    """
    Pull ensemble-member precipitation TOTALS (mm) for the forecast window.
    Returns ~50 totals — the empirical distribution.
    """
    if start > end:
        return [0.0]
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      "precipitation_sum",
        "models":     "ecmwf_ifs025",
        "timezone":   timezone,
    }
    try:
        resp = httpx.get(ENSEMBLE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Ensemble forecast failed ({lat},{lon}) {start}..{end}: {e}")
        return None

    daily = data.get("daily", {})
    member_totals: list[float] = []
    for k, vals in daily.items():
        if k.startswith("precipitation_sum_member") and isinstance(vals, list):
            total = sum(v for v in vals if v is not None)
            member_totals.append(total)
    return member_totals or None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_forecast_deterministic_total(
    lat: float, lon: float,
    start: date, end: date,
    timezone: str = "auto",
) -> float | None:
    """Single-value deterministic forecast (mm sum over window)."""
    if start > end:
        return 0.0
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "daily":      "precipitation_sum",
        "timezone":   timezone,
    }
    resp = httpx.get(FORECAST_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})
    vals = [v for v in (daily.get("precipitation_sum") or []) if v is not None]
    return sum(vals) if vals else None


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def month_window(year: int, month: int) -> tuple[date, date]:
    """First and last calendar day of a month."""
    first = date(year, month, 1)
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return first, last


def split_window(today: date, first: date, last: date) -> tuple[date, date, date, date]:
    """
    Split month into observed (first .. yesterday) and forecast (today .. last).
    Empty windows have start > end.
    """
    obs_end = min(today - timedelta(days=1), last)
    fc_start = max(today, first)
    return first, obs_end, fc_start, last


# ---------------------------------------------------------------------------
# Probability estimation
# ---------------------------------------------------------------------------

def estimate_monthly_precip_probability(
    lat: float,
    lon: float,
    year: int,
    month: int,
    bucket_low_mm: float,
    bucket_high_mm: float | None,    # None = open-ended top
    today: date | None = None,
) -> dict | None:
    """Estimate P(monthly precip in mm falls in [low, high))."""
    today = today or date.today()
    first, last = month_window(year, month)

    # Month hasn't started yet
    if today < first:
        return None

    # Month entirely in past -> archive only, deterministic answer
    if last < today:
        try:
            observed = get_observed_precipitation(lat, lon, first, last)
        except Exception as e:
            logger.warning(f"Archive lookup failed: {e}")
            return None
        total = sum(observed)
        in_bucket = bucket_low_mm <= total < (bucket_high_mm if bucket_high_mm is not None else 1e9)
        return {
            "source":      "monthly_precip_model",
            "probability": 1.0 if in_bucket else 0.0,
            "confidence":  0.95,
            "details": {
                "regime":         "fully_observed",
                "observed_mm":    round(total, 2),
                "n_obs_days":     len(observed),
                "bucket_low_mm":  bucket_low_mm,
                "bucket_high_mm": bucket_high_mm,
            },
        }

    # Month not started yet — skip for now (no climatology integration here)
    if today < first:
        return None

    # Mixed case: observed [first..yesterday], forecast [today..last]
    _, obs_end, fc_start, _ = split_window(today, first, last)

    # Handle very early in the month (obs_end < first means no observed days yet)
    observed_total = 0.0
    n_obs = 0
    coverage = 1.0

    if obs_end >= first:
        try:
            observed = get_observed_precipitation(lat, lon, first, obs_end)
            observed_total = sum(observed)
            n_obs = len(observed)
            expected_obs_days = (obs_end - first).days + 1
            coverage = n_obs / expected_obs_days if expected_obs_days > 0 else 0.0
        except Exception as e:
            logger.warning(f"Archive lookup failed: {e}")
            # Don't return None - fall through with 0 observed, full forecast
            # This handles rate limiting and early-month cases
            coverage = 0.0

    member_totals = get_forecast_ensemble_total(lat, lon, fc_start, last)
    forecast_method = "ensemble"
    if not member_totals:
        det = get_forecast_deterministic_total(lat, lon, fc_start, last)
        if det is None:
            logger.warning(f"All forecast paths failed ({lat},{lon}) {fc_start}..{last}")
            return None
        member_totals = [det]
        forecast_method = "deterministic"

    monthly_totals = [observed_total + m for m in member_totals]
    high = bucket_high_mm if bucket_high_mm is not None else float("inf")
    in_bucket_count = sum(1 for t in monthly_totals if bucket_low_mm <= t < high)
    prob = in_bucket_count / len(monthly_totals)

    # KNIFE-EDGE GUARD: If the median total sits within +/- KNIFE_EDGE_MM of a
    # bucket boundary, our point estimate is fragile. Real-world measurement
    # noise (gauge error, station-vs-grid mismatch, archive late-data updates)
    # can swing the actual settlement by 1-3mm. We pull the probability toward
    # 50/50 in proportion to how close we are to the edge.
    KNIFE_EDGE_MM = 3.0
    p50_total = statistics.median(monthly_totals)
    dist_to_low  = abs(p50_total - bucket_low_mm)
    dist_to_high = abs(p50_total - high) if high != float("inf") else float("inf")
    nearest_edge_mm = min(dist_to_low, dist_to_high)

    knife_edge_active = nearest_edge_mm < KNIFE_EDGE_MM
    if knife_edge_active:
        # Linearly blend toward 0.5 as nearest_edge_mm goes 0 -> KNIFE_EDGE_MM
        blend = 1.0 - (nearest_edge_mm / KNIFE_EDGE_MM)  # 1 at edge, 0 at safe distance
        prob = prob * (1 - blend) + 0.5 * blend
        logger.info(
            f"Knife-edge guard active: p50={p50_total:.2f}mm is "
            f"{nearest_edge_mm:.2f}mm from boundary -> blend {blend:.2f} -> p={prob:.4f}"
        )

    fc_days_remaining = (last - fc_start).days + 1 if fc_start <= last else 0
    days_remaining_factor = 1.0 - (fc_days_remaining / 31)
    method_weight = 0.85 if forecast_method == "ensemble" else 0.55
    confidence = round(method_weight * (0.4 + 0.6 * days_remaining_factor) * max(coverage, 0.5), 3)
    if knife_edge_active:
        # Slash confidence so the disagreement check / risk filters can bail too
        confidence = round(confidence * 0.5, 3)

    sorted_totals = sorted(monthly_totals)
    p10 = sorted_totals[max(0, int(0.1*len(sorted_totals))-1)]
    p50 = statistics.median(sorted_totals)
    p90 = sorted_totals[min(len(sorted_totals)-1, int(0.9*len(sorted_totals)))]

    return {
        "source":      "monthly_precip_model",
        "probability": round(prob, 4),
        "confidence":  confidence,
        "details": {
            "regime":         "mixed",
            "observed_mm":    round(observed_total, 2),
            "n_obs_days":     n_obs,
            "obs_coverage":   round(coverage, 3),
            "fc_method":      forecast_method,
            "fc_days":        fc_days_remaining,
            "fc_n_members":   len(member_totals),
            "fc_p10_mm":      round(p10, 2),
            "fc_p50_mm":      round(p50, 2),
            "fc_p90_mm":      round(p90, 2),
            "bucket_low_mm":  bucket_low_mm,
            "bucket_high_mm": bucket_high_mm,
        },
    }


# Alias: edge.py uses the shorter name
estimate_bucket_probability = estimate_monthly_precip_probability


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 75)
    print("Live: NYC April 2026 precipitation buckets vs market prices")
    print("=" * 75)

    # NYC market YES prices captured from /events earlier
    NYC_BUCKETS_INCHES = [
        (0.0, 2.0,  0.003, "less than 2 inches"),
        (2.0, 3.0,  0.605, "2 to 3 inches"),
        (3.0, 4.0,  0.249, "3 to 4 inches"),
        (4.0, 5.0,  0.053, "4 to 5 inches"),
        (5.0, 6.0,  0.009, "5 to 6 inches"),
        (6.0, None, 0.002, "more than 6 inches"),
    ]
    print(f"\n{'Bucket':<22s} {'model':>6s} {'market':>6s}  {'edge':>7s}  details")
    print("-" * 95)
    for low_in, high_in, market_yes, label in NYC_BUCKETS_INCHES:
        low_mm  = low_in * 25.4
        high_mm = high_in * 25.4 if high_in is not None else None
        result = estimate_monthly_precip_probability(
            lat=40.7128, lon=-74.0060,
            year=2026, month=4,
            bucket_low_mm=low_mm, bucket_high_mm=high_mm,
        )
        if not result:
            print(f"  {label:<20s} (None)")
            continue
        mp, edge = result["probability"], result["probability"] - market_yes
        d = result["details"]
        details = f"obs={d.get('observed_mm','?')}mm  p50={d.get('fc_p50_mm','?')}mm  conf={result['confidence']}"
        print(f"  {label:<20s} {mp:>6.3f} {market_yes:>6.3f}  {edge:>+7.3f}  {details}")

    print()
    print("=" * 75)
    print("Live: Seattle April 2026 precipitation buckets vs market prices")
    print("=" * 75)
    SEATTLE_BUCKETS = [
        (0.0,  2.5, 0.001, "less than 2.5\""),
        (2.5,  3.0, 0.890, "2.5-3\""),
        (3.0,  3.5, 0.110, "3-3.5\""),
        (3.5,  4.0, 0.003, "3.5-4\""),
        (4.0,  4.5, 0.004, "4-4.5\""),
        (4.5,  5.0, 0.007, "4.5-5\""),
        (5.0, None, 0.003, "more than 5\""),
    ]
    print(f"\n{'Bucket':<22s} {'model':>6s} {'market':>6s}  {'edge':>7s}  details")
    print("-" * 95)
    for low_in, high_in, market_yes, label in SEATTLE_BUCKETS:
        low_mm  = low_in * 25.4
        high_mm = high_in * 25.4 if high_in is not None else None
        result = estimate_monthly_precip_probability(
            lat=47.6062, lon=-122.3321,
            year=2026, month=4,
            bucket_low_mm=low_mm, bucket_high_mm=high_mm,
        )
        if not result:
            print(f"  {label:<20s} (None)")
            continue
        mp = result["probability"]
        edge = mp - market_yes
        d = result["details"]
        details = f"obs={d.get('observed_mm','?')}mm  p50={d.get('fc_p50_mm','?')}mm  conf={result['confidence']}"
        print(f"  {label:<20s} {mp:>6.3f} {market_yes:>6.3f}  {edge:>+7.3f}  {details}")
