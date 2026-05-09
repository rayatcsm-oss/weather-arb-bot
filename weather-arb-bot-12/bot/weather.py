# weather.py
"""
Weather probability fetchers for four data sources, plus a Brier-weighted
ensemble aggregator.

Each fetcher returns:   {"source": str, "probability": float, "confidence": float}
                        or None on failure.

The aggregator returns: {"probability": float, "sources": list[dict],
                         "disagreement": float, "n_sources": int}
"""

import os
import re
import logging
import statistics
from datetime import date, datetime, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import BRIER_WEIGHTS

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. NOAA Climate Data Online (CDO) API
# ===========================================================================

NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")

# Hand-picked GHCND station IDs for primary airport ASOS sites. These have
# multi-decade daily records and report nearly every day — far more reliable
# than the citizen-science (CoCoRaHS / GHCND-D) stations that NOAA's
# nearest-station endpoint tends to return for major cities.
# Format: city_lat_lon (rounded) -> GHCND station id
MAJOR_NOAA_STATIONS: dict[tuple[float, float], str] = {
    (40.71, -74.01): "GHCND:USW00094728",  # NYC Central Park
    (41.88, -87.63): "GHCND:USW00094846",  # Chicago O'Hare
    (34.05, -118.24): "GHCND:USW00023174", # Los Angeles LAX
    (25.76, -80.19): "GHCND:USW00012839",  # Miami International
    (47.61, -122.33): "GHCND:USW00024233", # Seattle Sea-Tac
    (42.36, -71.06): "GHCND:USW00014739",  # Boston Logan
    (32.78, -96.80): "GHCND:USW00003927",  # Dallas DFW
    (39.74, -104.99): "GHCND:USW00003017", # Denver DIA
    (33.75, -84.39): "GHCND:USW00013874",  # Atlanta Hartsfield
    (29.76, -95.37): "GHCND:USW00012960",  # Houston Bush
    (33.45, -112.07): "GHCND:USW00023183", # Phoenix Sky Harbor
}


def _lookup_major_station(lat: float, lon: float, tol: float = 0.5) -> str | None:
    """Match (lat, lon) to a major-airport ASOS station within `tol` degrees."""
    for (slat, slon), station in MAJOR_NOAA_STATIONS.items():
        if abs(lat - slat) < tol and abs(lon - slon) < tol:
            return station
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_noaa_historical(station_id: str, date_str: str, datatype: str) -> float | None:
    """
    Fetch a historical observed value from NOAA CDO.
    Returns the observed value (e.g., precipitation in tenths of mm) or None.
    Used for calibrating model accuracy (Brier scoring).
    """
    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": datatype,
        "startdate": date_str,
        "enddate": date_str,
        "units": "standard",
        "limit": 1,
    }
    resp = httpx.get(f"{NOAA_BASE}/data", headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    return results[0]["value"]


def get_noaa_probability(
    lat: float,
    lon: float,
    date_str: str,
    variable: str,
    threshold_f: float | None = None,
    comparison: str = "gte",
) -> dict | None:
    """
    Derive a probability estimate from NOAA climatological normals.

    For 'rain'/'snow': returns historical fraction of years with nonzero
    precip on the same calendar day at the nearest major-airport station.

    For 'temp_high'/'temp_low': uses the empirical CDF of historical values
    at that station for the same calendar day. Without `threshold_f` we
    can't return a meaningful probability for temp questions — returns None.

    Returns: {"source": "noaa", "probability": float, "confidence": 0.5}
    """
    datatype_map = {
        "rain": "PRCP",
        "snow": "SNOW",
        "temp_high": "TMAX",
        "temp_low": "TMIN",
    }
    datatype = datatype_map.get(variable)
    if not datatype:
        return None

    try:
        station_id = _lookup_major_station(lat, lon) or _find_nearest_noaa_station(lat, lon)
    except Exception as e:
        logger.warning(f"NOAA station lookup failed: {e}")
        return None
    if not station_id:
        return None

    target_date = date.fromisoformat(date_str)
    month_day = f"{target_date.month:02d}-{target_date.day:02d}"
    historical_values = _get_historical_values_for_day(station_id, datatype, month_day, years=10)
    if not historical_values:
        return None

    if variable in ("rain", "snow"):
        prob = sum(1 for v in historical_values if v > 0) / len(historical_values)
    else:
        # Temperature path requires a threshold to compute a probability.
        if threshold_f is None:
            return None
        # NOAA returns °F when units=standard
        if comparison == "gte":
            prob = sum(1 for v in historical_values if v >= threshold_f) / len(historical_values)
        else:  # lte
            prob = sum(1 for v in historical_values if v <= threshold_f) / len(historical_values)

    return {"source": "noaa", "probability": round(prob, 4), "confidence": 0.5}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _find_nearest_noaa_station(lat: float, lon: float) -> str | None:
    """Find the nearest GHCND station to a lat/lon coordinate."""
    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GHCND",
        "extent": f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}",
        "limit": 1,
        "sortfield": "name",
    }
    resp = httpx.get(f"{NOAA_BASE}/stations", headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    return results[0]["id"]


def _get_historical_values_for_day(
    station_id: str, datatype: str, month_day: str, years: int = 10
) -> list[float]:
    """Fetch historical values for the same calendar day across multiple years."""
    current_year = date.today().year
    values = []
    for year in range(current_year - years, current_year):
        date_str = f"{year}-{month_day}"
        try:
            val = get_noaa_historical(station_id, date_str, datatype)
            if val is not None:
                values.append(val)
        except Exception as e:
            logger.debug(f"NOAA historical for {date_str} failed: {e}")
            continue
    return values


# ===========================================================================
# 2. Open-Meteo (free, no API key)
# ===========================================================================

OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_openmeteo_probability(
    lat: float,
    lon: float,
    date_str: str,
    variable: str,
    threshold_f: float | None = None,
    comparison: str = "gte",
) -> dict | None:
    """
    Use Open-Meteo ensemble models to derive probability estimates.
    The ensemble API returns ~50 member forecasts; the fraction of members
    on the event side gives a direct probability estimate.

    NOTE: Open-Meteo forecast APIs only cover up to 16 days ahead.
    For dates beyond that, we fall back to NOAA climatology automatically.

    For temperature variables (temp_high/temp_low), pass:
        threshold_f: the threshold IN FAHRENHEIT (Polymarket markets are °F)
        comparison: 'gte' for "exceed/above X" or 'lte' for "below X"
    Open-Meteo returns °C; we convert.

    Returns: {"source": "openmeteo", "probability": float, "confidence": 0.8}
    """
    # Check forecast horizon — Open-Meteo only covers 16 days
    try:
        target = date.fromisoformat(date_str)
        days_ahead = (target - date.today()).days
        if days_ahead > 15:
            logger.debug(
                f"Open-Meteo: {date_str} is {days_ahead} days out — beyond 16-day limit. "
                f"Skipping (NOAA climatology will cover this date)."
            )
            return None
        if days_ahead < 0:
            logger.debug(f"Open-Meteo: {date_str} is in the past — skipping")
            return None
    except (ValueError, TypeError):
        pass

    # Open-Meteo ensemble uses *_sum daily aggregates, NOT bare names like "precipitation".
    # Each ensemble member returns as a separate key, e.g. precipitation_sum_member01..50
    param_map = {
        "rain":      "precipitation_sum",
        "snow":      "snowfall_sum",
        "temp_high": "temperature_2m_max",
        "temp_low":  "temperature_2m_min",
    }
    param = param_map.get(variable)
    if not param:
        return None

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "daily":      param,
        "start_date": date_str,
        "end_date":   date_str,
        "models":     "ecmwf_ifs025",  # ECMWF ensemble — best calibration
    }

    try:
        resp = httpx.get(OPENMETEO_ENSEMBLE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError:
        return _get_openmeteo_deterministic(lat, lon, date_str, variable)

    daily = data.get("daily", {})

    # Collect per-member values: keys are param + "_member01" .. "_member50"
    # The bare `param` key is the ensemble mean; we want all members for distribution.
    member_values = []
    for k, v in daily.items():
        if k.startswith(f"{param}_member") and isinstance(v, list) and v and v[0] is not None:
            member_values.append(v[0])

    if not member_values:
        # Fallback to deterministic if ensemble returned all-None (common for >7d horizons)
        return _get_openmeteo_deterministic(lat, lon, date_str, variable)

    if variable in ("rain", "snow"):
        # 0.1 inches in mm
        threshold_mm = 2.54
        prob = sum(1 for v in member_values if v >= threshold_mm) / len(member_values)
    elif variable in ("temp_high", "temp_low"):
        # Members are in °C; convert threshold from °F if provided.
        if threshold_f is None:
            # Without a threshold there's no meaningful probability
            logger.debug("Open-Meteo temp called without threshold — returning None")
            return None
        threshold_c = (threshold_f - 32.0) * 5.0 / 9.0
        if comparison == "gte":
            prob = sum(1 for v in member_values if v >= threshold_c) / len(member_values)
        else:  # 'lte'
            prob = sum(1 for v in member_values if v <= threshold_c) / len(member_values)
    else:
        return None

    return {"source": "openmeteo", "probability": round(prob, 4), "confidence": 0.8}


def _get_openmeteo_deterministic(lat: float, lon: float, date_str: str, variable: str) -> dict | None:
    """Fallback: deterministic Open-Meteo forecast with precipitation probability."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
        "start_date": date_str,
        "end_date": date_str,
    }
    try:
        resp = httpx.get(OPENMETEO_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        logger.warning(f"Open-Meteo deterministic fallback failed: {e}")
        return None

    daily = data.get("daily", {})

    if variable == "rain":
        pp = daily.get("precipitation_probability_max", [None])[0]
        if pp is None:
            return None
        return {"source": "openmeteo_det", "probability": round(pp / 100, 4), "confidence": 0.65}
    elif variable == "snow":
        snowfall = daily.get("snowfall_sum", [None])[0]
        if snowfall is None:
            return None
        return {"source": "openmeteo_det", "probability": 1.0 if snowfall > 0 else 0.0, "confidence": 0.6}
    return None


# ===========================================================================
# 3. National Weather Service (NWS) API
# ===========================================================================

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {"User-Agent": "WeatherArbBot/1.0 (rayatcsm@gmail.com)"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def get_nws_probability(
    lat: float,
    lon: float,
    date_str: str,
    variable: str,
    threshold_f: float | None = None,
    comparison: str = "gte",
) -> dict | None:
    """
    Use the National Weather Service API to get forecast probability.
    NWS covers approximately 7 days ahead. Beyond that, returns None so
    the ensemble falls back to NOAA climatology which covers all dates.

    For 'rain' / 'snow': returns the maximum probabilityOfPrecipitation across
    matching forecast periods.

    For 'temp_high' / 'temp_low': uses NWS's point temperature forecast plus
    a typical NWS forecast standard error (~3°F at day-1, growing) to compute
    a normal-CDF probability of crossing the threshold. Without `threshold_f`
    we can't return a meaningful probability — returns None.

    Returns: {"source": "nws", "probability": float, "confidence": 0.85}
    """
    # NWS only publishes ~7 days of hourly/daily forecast periods
    try:
        target = date.fromisoformat(date_str)
        days_ahead = (target - date.today()).days
        if days_ahead > 7:
            logger.debug(
                f"NWS: {date_str} is {days_ahead} days out — beyond NWS 7-day window. "
                f"Skipping (NOAA climatology covers this date)."
            )
            return None
        if days_ahead < 0:
            logger.debug(f"NWS: {date_str} is in the past — skipping")
            return None
    except (ValueError, TypeError):
        pass

    try:
        points_url = f"{NWS_BASE}/points/{lat},{lon}"
        resp = httpx.get(points_url, headers=NWS_HEADERS, timeout=10)
        resp.raise_for_status()
        properties = resp.json()["properties"]
        forecast_url = properties["forecast"]
    except (httpx.HTTPError, KeyError) as e:
        logger.warning(f"NWS points lookup failed: {e}")
        return None

    try:
        resp = httpx.get(forecast_url, headers=NWS_HEADERS, timeout=10)
        resp.raise_for_status()
        periods = resp.json()["properties"]["periods"]
    except (httpx.HTTPError, KeyError) as e:
        logger.warning(f"NWS forecast fetch failed: {e}")
        return None

    target_date = date.fromisoformat(date_str)
    matching_periods = []
    for period in periods:
        start_time = datetime.fromisoformat(period["startTime"].replace("Z", "+00:00"))
        if start_time.date() == target_date:
            matching_periods.append(period)

    if not matching_periods:
        return None

    if variable in ("rain", "snow"):
        probs = []
        for period in matching_periods:
            pop = period.get("probabilityOfPrecipitation", {})
            if pop and pop.get("value") is not None:
                probs.append(pop["value"] / 100.0)
            else:
                text = period.get("detailedForecast", "")
                match = re.search(r"(\d+)\s*percent", text, re.IGNORECASE)
                if match:
                    probs.append(int(match.group(1)) / 100.0)

        if not probs:
            return None

        prob = max(probs)

        if variable == "snow":
            snow_mentioned = any(
                "snow" in p.get("shortForecast", "").lower()
                for p in matching_periods
            )
            if not snow_mentioned:
                prob *= 0.3

        return {"source": "nws", "probability": round(prob, 4), "confidence": 0.85}

    elif variable in ("temp_high", "temp_low"):
        if threshold_f is None:
            return None  # need a threshold to compute probability

        # Pick the right period: daytime for highs, nighttime for lows
        if variable == "temp_high":
            target_periods = [p for p in matching_periods if p.get("isDaytime", False)]
        else:
            target_periods = [p for p in matching_periods if not p.get("isDaytime", True)]

        if not target_periods:
            return None
        forecast_temp = target_periods[0].get("temperature")
        unit = (target_periods[0].get("temperatureUnit") or "F").upper()
        if forecast_temp is None:
            return None

        if unit == "C":
            forecast_temp = forecast_temp * 9.0 / 5.0 + 32.0  # to °F

        # NWS day-1 temperature MAE is roughly 3 °F. Beyond day 1, it grows
        # at ~1 °F per day. Use that as our normal sigma to derive P.
        days_ahead = max(0, (target_date - date.today()).days)
        sigma = 3.0 + days_ahead

        # Normal CDF using math.erf
        import math
        z = (threshold_f - forecast_temp) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        if comparison == "gte":
            # P(temp >= threshold) = 1 - CDF
            prob = 1.0 - cdf
        else:
            # P(temp <= threshold) = CDF
            prob = cdf

        return {
            "source":      "nws",
            "probability": round(prob, 4),
            "confidence":  0.85,
            "details": {
                "forecast_temp_f": round(forecast_temp, 1),
                "threshold_f":     threshold_f,
                "comparison":      comparison,
                "sigma":           sigma,
            },
        }

    else:
        return None


# ===========================================================================
# 4. Tomorrow.io (optional)
# ===========================================================================

TOMORROWIO_BASE = "https://api.tomorrow.io/v4/weather/forecast"
TOMORROWIO_KEY = os.getenv("TOMORROWIO_API_KEY")


def get_tomorrowio_probability(lat: float, lon: float, date_str: str, variable: str) -> dict | None:
    """
    Tomorrow.io provides hyperlocal probabilistic forecasts.
    Only called if API key is present in environment.
    """
    if not TOMORROWIO_KEY:
        return None

    params = {
        "location": f"{lat},{lon}",
        "apikey": TOMORROWIO_KEY,
        "fields": "precipitationProbability,precipitationType,temperatureMax,temperatureMin,snowAccumulation",
        "timesteps": "1d",
        "startTime": f"{date_str}T00:00:00Z",
        "endTime": f"{date_str}T23:59:59Z",
        "units": "imperial",
    }

    try:
        resp = httpx.get(TOMORROWIO_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("timelines", {}).get("daily", [])
        if not daily:
            return None
        values = daily[0].get("values", {})

        # Tomorrow.io daily fields use Avg/Max/Min suffixes — there is NO bare
        # "precipitationProbability" field on the daily timestep.
        prob_max = values.get("precipitationProbabilityMax")
        if prob_max is None:
            return None
        prob = prob_max / 100.0

        # Detect snow vs rain via accumulation sums (precipitationType is hourly-only)
        snow_sum = values.get("snowAccumulationSum", 0) or 0
        rain_sum = values.get("rainAccumulationSum", 0) or 0
        is_snow_day = snow_sum > 0 and rain_sum < 0.05

        if variable == "rain":
            # If the precip is forecast as all-snow, "rain" probability is near zero
            if is_snow_day:
                prob *= 0.1
            return {"source": "tomorrowio", "probability": round(prob, 4), "confidence": 0.75}

        elif variable == "snow":
            # Only credit the precip prob if snow accumulation is actually expected
            if snow_sum > 0:
                return {"source": "tomorrowio", "probability": round(prob, 4), "confidence": 0.75}
            return {"source": "tomorrowio", "probability": 0.0, "confidence": 0.75}

    except httpx.HTTPError as e:
        logger.warning(f"Tomorrow.io request failed: {e}")
        return None

    return None


# ===========================================================================
# 5. Ensemble aggregator
# ===========================================================================

def get_ensemble_probability(
    lat: float,
    lon: float,
    date_str: str,
    variable: str,
    threshold_f: float | None = None,
    comparison: str = "gte",
) -> dict | None:
    """
    Fetch from all sources, weight by historical Brier scores, return ensemble P.

    For temperature variables, pass `threshold_f` (in °F) and `comparison`
    ('gte' for "exceed/above" or 'lte' for "below"). Without a threshold,
    temperature sources return None and the ensemble falls back to whatever
    signals do return a probability (or None if nothing usable).

    Returns: {
        "probability":  float,
        "sources":      list of source dicts,
        "disagreement": float,   # std dev across sources
        "n_sources":    int,
    }
    """
    # Tomorrow.io doesn't currently support thresholds — gets plain calls.
    # NOAA, Open-Meteo, and NWS all accept threshold_f/comparison.
    results = []

    for fetcher_name, fetcher in [
        ("tomorrowio", get_tomorrowio_probability),
    ]:
        try:
            result = fetcher(lat, lon, date_str, variable)
            if result and result.get("probability") is not None:
                results.append(result)
        except Exception as e:
            logger.debug(f"{fetcher_name} raised: {e}")

    for fetcher_name, fetcher in [
        ("noaa",       get_noaa_probability),
        ("openmeteo",  get_openmeteo_probability),
        ("nws",        get_nws_probability),
    ]:
        try:
            result = fetcher(
                lat, lon, date_str, variable,
                threshold_f=threshold_f, comparison=comparison,
            )
            if result and result.get("probability") is not None:
                results.append(result)
        except Exception as e:
            logger.debug(f"{fetcher_name} raised: {e}")

    if not results:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = BRIER_WEIGHTS.get(r["source"], 0.2)
        weighted_sum += r["probability"] * w
        total_weight += w

    if total_weight == 0:
        return None

    ensemble_p = weighted_sum / total_weight
    ensemble_p = max(0.0, min(1.0, ensemble_p))

    probs = [r["probability"] for r in results]
    disagreement = statistics.stdev(probs) if len(probs) > 1 else 0.0

    return {
        "probability":  round(ensemble_p, 4),
        "sources":      results,
        "disagreement": round(disagreement, 4),
        "n_sources":    len(results),
    }


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    lat, lon = 41.85, -87.65
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    variable = "rain"

    print(f"--- Smoke test: {variable} at ({lat},{lon}) on {tomorrow} ---\n")

    print("[NWS]")
    nws = get_nws_probability(lat, lon, tomorrow, variable)
    print(json.dumps(nws, indent=2) if nws else "  None")

    print("\n[Open-Meteo]")
    om = get_openmeteo_probability(lat, lon, tomorrow, variable)
    print(json.dumps(om, indent=2) if om else "  None")

    print("\n[Tomorrow.io]")
    ti = get_tomorrowio_probability(lat, lon, tomorrow, variable)
    print(json.dumps(ti, indent=2) if ti else "  None")

    print("\n[NOAA] (climatology — 10y of historical days, slow)")
    noaa = get_noaa_probability(lat, lon, tomorrow, variable)
    print(json.dumps(noaa, indent=2) if noaa else "  None")

    print("\n--- Ensemble ---")
    ens = get_ensemble_probability(lat, lon, tomorrow, variable)
    if ens:
        print(f"probability:  {ens['probability']}")
        print(f"disagreement: {ens['disagreement']}")
        print(f"n_sources:    {ens['n_sources']}")
        print("sources:")
        for s in ens["sources"]:
            print(f"  - {s['source']:15s} p={s['probability']:.4f} confidence={s['confidence']}")
    else:
        print("No ensemble result (all sources failed)")
