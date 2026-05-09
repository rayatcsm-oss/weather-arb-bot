# polymarket.py
"""
Polymarket Gamma API integration.

Three public functions per the guide spec:
  * search_weather_markets(min_liquidity)  -> list[dict]
  * get_contract_price(contract_id)        -> dict | None
  * parse_contract_metadata(contract)      -> dict | None

DEVIATIONS FROM GUIDE (live API shape changed since guide was written):
  - Markets do NOT contain a `tokens` array. Instead, `outcomes`,
    `outcomePrices`, and `clobTokenIds` are returned as JSON-encoded STRINGS
    that must be json.loads()'d and zipped by index.
  - Numeric fields like liquidity/volume come back as STRINGS in `liquidity`
    and `volume`; the float versions live in `liquidityNum` and `volumeNum`.
  - `bestBid` and `bestAsk` are available top-of-book and are more accurate
    for execution than the mid-price; we expose them.
"""

import re
import json
import time
import logging
import calendar
from datetime import datetime, date

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Keywords used to surface candidate weather markets via the Gamma `q=` filter.
# The filter is loose (matches description text too), so callers should still
# verify questions look meteorological before trading them.
WEATHER_KEYWORDS = [
    "rain", "snow", "temperature", "weather", "precipitation",
    "inches", "degrees", "storm", "hurricane", "flood", "frost",
]

# Stricter pattern applied to the QUESTION TEXT (not description) post-search.
# "storm" / "heat" alone match WNBA Seattle Storm, NBA Miami Heat, etc.,
# and "Hurricanes" matches Carolina Hurricanes (NHL). We use unambiguously
# meteorological vocabulary plus an explicit team-name exclusion list.
_QUESTION_WEATHER_PATTERN = re.compile(
    r"\b(rain|rainfall|precipitation|"
    r"snow|snowfall|blizzard|"
    r"hurricane|cyclone|tornado|"
    r"temperature|degrees?|fahrenheit|°f|celsius|"
    r"flood|frost|drought|"
    r"heat\s*wave|cold\s*snap|"
    r"named\s+storm|tropical\s+storm|"
    r"hottest|coldest|warmest|"
    r"climate|sea\s+ice|arctic\s+ice|"
    r"high\s+(?:in|at|of)\s+(?:nyc|new york|chicago|los angeles|miami|seattle|"
    r"boston|dallas|denver|atlanta|houston|phoenix|"
    r"london|seoul|hong kong|tokyo|paris|berlin|madrid|sydney|singapore|mumbai)|"
    r"low\s+(?:in|at|of)\s+(?:nyc|new york|chicago|los angeles|miami|seattle|"
    r"boston|dallas|denver|atlanta|houston|phoenix|"
    r"london|seoul|hong kong|tokyo|paris|berlin|madrid|sydney|singapore|mumbai))\b",
    re.IGNORECASE,
)

# Explicit deny-list — these strings unambiguously mean a non-weather context
# even though they contain meteorological words.
_NON_WEATHER_PHRASES = [
    "carolina hurricane", "miami heat", "seattle storm", "as storm",
    "oklahoma city thunder", "stanley cup", "nba finals", "nhl ",
    "wnba ", "stars fc", "thunder fc", "fc storm",
]


def _question_looks_weather(question: str) -> bool:
    """True iff the question text contains an unambiguous meteorological term."""
    if not question:
        return False
    qlow = question.lower()
    if any(p in qlow for p in _NON_WEATHER_PHRASES):
        return False
    return bool(_QUESTION_WEATHER_PATTERN.search(question))


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gamma_get(path: str, params: dict) -> list | dict:
    """Rate-limited GET to Gamma API (max ~1 req/sec)."""
    time.sleep(1)
    resp = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_market(raw: dict) -> dict | None:
    """
    Convert a raw Gamma API market dict to our internal schema.

    Internal schema:
      contract_id, question, yes_price, no_price, yes_token_id, no_token_id,
      best_bid, best_ask, liquidity_usd, volume_usd, resolution_date,
      resolution_source, description, end_date_iso

    Returns None if the market can't be parsed (missing outcomes, malformed
    JSON in outcomes/prices/tokenIds, no YES/NO pair, etc.).
    """
    try:
        # outcomes/prices/clobTokenIds are JSON-encoded strings — parse and zip
        outcomes_raw = raw.get("outcomes", "[]")
        prices_raw   = raw.get("outcomePrices", "[]")
        tokens_raw   = raw.get("clobTokenIds", "[]")

        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices   = json.loads(prices_raw)   if isinstance(prices_raw, str)   else prices_raw
        tokens   = json.loads(tokens_raw)   if isinstance(tokens_raw, str)   else tokens_raw

        if not (isinstance(outcomes, list) and isinstance(prices, list) and isinstance(tokens, list)):
            return None
        if len(outcomes) != len(prices) or len(outcomes) != len(tokens):
            return None

        # Locate YES and NO indices (Polymarket uses "Yes"/"No" capitalization)
        yes_idx = next((i for i, o in enumerate(outcomes) if str(o).upper() == "YES"), None)
        no_idx  = next((i for i, o in enumerate(outcomes) if str(o).upper() == "NO"),  None)
        if yes_idx is None or no_idx is None:
            return None

        yes_price = float(prices[yes_idx])
        no_price  = float(prices[no_idx])

        # liquidity/volume: prefer the *Num float fields, fall back to parsing strings
        def _num(*keys) -> float:
            for k in keys:
                v = raw.get(k)
                if v is None or v == "":
                    continue
                try:
                    return float(v)
                except (ValueError, TypeError):
                    continue
            return 0.0

        return {
            "contract_id":       raw.get("conditionId") or str(raw.get("id", "")),
            "question":          raw.get("question", ""),
            "yes_price":         yes_price,
            "no_price":          no_price,
            "yes_token_id":      str(tokens[yes_idx]),
            "no_token_id":       str(tokens[no_idx]),
            "best_bid":          _num("bestBid"),
            "best_ask":          _num("bestAsk"),
            "liquidity_usd":     _num("liquidityNum", "liquidityClob", "liquidity"),
            "volume_usd":        _num("volumeNum", "volumeClob", "volume"),
            "resolution_date":   raw.get("endDate") or raw.get("endDateIso", ""),
            "resolution_source": raw.get("resolutionSource", "") or "",
            "description":       raw.get("description", "") or "",
            "end_date_iso":      raw.get("endDateIso", ""),
        }
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        logger.debug(f"Failed to normalize market: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_weather_markets(
    min_liquidity: float = 500.0,
    max_pages: int = 10,
    page_size: int = 500,
) -> list[dict]:
    """
    Find every active weather market on Polymarket.

    IMPORTANT: walks /events (not /markets) because Polymarket organizes
    multi-bucket weather markets (e.g. "Precipitation in NYC in April")
    as one event with N child markets. Those child markets are NOT
    returned by the top-level /markets endpoint at all.

    Returns a list of normalized contract dicts. Filters to a weather-y
    question and liquidity >= min_liquidity.
    """
    raw_markets: list[dict] = []
    seen_event_ids: set[str] = set()

    for page in range(max_pages):
        offset = page * page_size
        try:
            time.sleep(0.5)  # gentle rate limit
            resp = httpx.get(
                f"{GAMMA_BASE}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit":  page_size,
                    "offset": offset,
                },
                timeout=20,
            )
            resp.raise_for_status()
            events = resp.json() or []
        except Exception as e:
            logger.error(f"Failed to fetch events page {page} (offset={offset}): {e}")
            break

        if not events:
            break

        for event in events:
            eid = str(event.get("id", ""))
            if eid in seen_event_ids:
                continue
            seen_event_ids.add(eid)

            # Walk every child market
            for m in event.get("markets", []) or []:
                if _question_looks_weather(m.get("question", "")):
                    raw_markets.append(m)

        if len(events) < page_size:
            break  # last page

    # Deduplicate child markets by id / conditionId
    seen: set[str] = set()
    unique: list[dict] = []
    for m in raw_markets:
        mid = m.get("conditionId") or str(m.get("id", ""))
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(m)

    # Normalize and apply liquidity filter
    result: list[dict] = []
    for m in unique:
        norm = _normalize_market(m)
        if norm and norm["liquidity_usd"] >= min_liquidity:
            result.append(norm)

    logger.info(
        f"Found {len(result)} weather markets >= ${min_liquidity:.0f} liquidity "
        f"(of {len(unique)} unique weather questions across {len(seen_event_ids)} events)"
    )
    return result


def get_contract_price(contract_id: str) -> dict | None:
    """Refresh the prices for one contract by id or conditionId."""
    try:
        data = _gamma_get(f"/markets/{contract_id}", {})
        # /markets/{id} returns a single object (or list with one element)
        if isinstance(data, list):
            data = data[0] if data else None
        if not data:
            return None
        return _normalize_market(data)
    except Exception as e:
        logger.error(f"Failed to get price for {contract_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Question-text parsing
# ---------------------------------------------------------------------------

# Hardcoded city-to-coord lookup (matches the major-airport stations in
# weather.MAJOR_NOAA_STATIONS). In production this should be replaced with a
# proper geocoding API call.
# City -> (lat, lon, station_trust)
# station_trust ∈ {'high', 'medium', 'low'}:
#   high   = our (lat, lon) is at or very near the actual resolution station
#   medium = nearby but the market's station may differ
#   low    = international station with possible large mismatch (skip or lower size)
CITY_COORDS_FULL: dict[str, tuple[float, float, str]] = {
    # US — NOAA stations
    "new york":    (40.7794, -73.9692, "high"),    # Central Park, NYC (NOAA station USW00094728)
    "nyc":         (40.7794, -73.9692, "high"),
    "chicago":     (41.9786, -87.9048, "high"),    # Chicago O'Hare
    "los angeles": (33.9416, -118.4085, "high"),   # LAX
    "miami":       (25.7959, -80.2870, "high"),    # Miami Intl
    "seattle":     (47.4502, -122.3088, "medium"), # Sea-Tac (market says "Seattle City Area" — may differ)
    "boston":      (42.3656, -71.0096, "high"),    # Logan
    "dallas":      (32.8998, -97.0403, "high"),    # DFW
    "denver":      (39.8617, -104.6731, "high"),   # DIA
    "atlanta":     (33.6407, -84.4277, "high"),    # Hartsfield
    "houston":     (29.9844, -95.3414, "high"),    # IAH (Bush)
    "phoenix":     (33.4373, -112.0078, "high"),   # Sky Harbor
    # International — coords set to RESOLUTION STATION, not city center
    "london":      (51.4700, -0.4543, "medium"),   # Heathrow per Met Office
    "seoul":       (37.5714, 126.9658, "low"),     # KMA Seoul observatory; station ID differs
    "hong kong":   (22.3017, 114.1747, "low"),     # HK Observatory headquarters station
    "tokyo":       (35.6895, 139.6917, "low"),     # JMA Tokyo (Otemachi)
    "paris":       (48.8331, 2.3994, "low"),       # Paris Montsouris
    "berlin":      (52.4536, 13.3035, "low"),      # Berlin Tempelhof
    "madrid":      (40.4521, -3.5642, "low"),      # Madrid Barajas
    "sydney":      (-33.8688, 151.2093, "low"),    # Sydney Observatory Hill
    "singapore":   (1.3644, 103.9915, "low"),      # Changi
    "mumbai":      (19.0903, 72.8678, "low"),      # Santacruz
}

# Backwards-compat — anything that just wants (lat, lon)
CITY_COORDS: dict[str, tuple[float, float]] = {
    name: (lat, lon) for name, (lat, lon, _trust) in CITY_COORDS_FULL.items()
}


def _city_trust(city_name: str) -> str:
    """Look up resolution-station match confidence for a parsed city name."""
    entry = CITY_COORDS_FULL.get(city_name)
    return entry[2] if entry else "low"


def parse_contract_metadata(contract: dict) -> dict | None:
    """
    Extract structured parameters from a contract's question.

    Returns one of:
      * Monthly-precipitation bucket (market_class='monthly_precip'):
          city, lat, lon, month_iso ("YYYY-MM"), bucket_low, bucket_high, unit
      * Daily-weather schema (market_class='daily_weather'):
          lat, lon, date, variable, threshold, unit
      * Hurricane schema (market_class='hurricane'):
          hurricane_type, deadline
      * None if no match.
    """
    question = contract.get("question", "").lower()
    if not question:
        return None

    # --- 0) Global temperature ranking markets --------------------------
    # These must be checked first because they don't contain city names
    # and would fall through to the daily-weather parser incorrectly.
    try:
        from global_temp_model import classify_temp_market
        temp_info = classify_temp_market(contract.get("question", ""))
    except ImportError:
        temp_info = None

    if temp_info:
        rd = contract.get("resolution_date", "")
        deadline = None
        if rd:
            try:
                deadline = datetime.fromisoformat(rd.replace("Z", "+00:00")).date().isoformat()
            except (ValueError, TypeError):
                pass
        return {
            "market_class": "global_temp",
            "temp_year":    temp_info["year"],
            "temp_rank":    temp_info["rank"],
            "temp_direction": temp_info["direction"],
            "temp_lower_bound": temp_info.get("lower_bound", False),
            "date":         deadline,
            "lat": None, "lon": None, "variable": None, "threshold": None,
        }

    # --- 1) Monthly-precipitation bucket markets --------------------------
    monthly = _parse_monthly_precip(question, contract.get("resolution_date", ""))
    if monthly:
        return monthly

    # --- 2) Hurricane markets ---------------------------------------------
    try:
        from hurricane_model import classify_hurricane_market
        h_type = classify_hurricane_market(contract.get("question", ""))
    except ImportError:
        h_type = None

    if h_type:
        deadline = None
        rd = contract.get("resolution_date", "")
        if rd:
            try:
                deadline = datetime.fromisoformat(
                    rd.replace("Z", "+00:00")
                ).date().isoformat()
            except (ValueError, TypeError):
                pass
        if not deadline:
            return None
        return {
            "market_class":   "hurricane",
            "hurricane_type": h_type,
            "deadline":       deadline,
            "lat":            None,
            "lon":            None,
            "date":           deadline,
            "variable":       None,
            "threshold":      None,
            "unit":           None,
        }

    # --- 3) Daily-weather (city-specific) ---------------------------------
    lat = lon = None
    for city, coords in CITY_COORDS.items():
        if city in question:
            lat, lon = coords
            break
    if lat is None:
        return None

    variable: str | None = None
    if any(w in question for w in ["snow", "snowfall", "blizzard"]):
        variable = "snow"
    elif any(w in question for w in ["rain", "precipitation", "precip", "flood"]):
        variable = "rain"
    elif any(w in question for w in ["high temperature", "highest temperature", "max temperature",
                                      "high in", "max temp"]):
        variable = "temp_high"
    elif any(w in question for w in ["low temperature", "lowest temperature", "min temperature",
                                      "low in", "min temp"]):
        variable = "temp_low"
    elif "high" in question or "maximum" in question:
        variable = "temp_high"
    elif "low" in question or "minimum" in question:
        variable = "temp_low"
    if variable is None:
        return None

    threshold: float | None = None
    unit: str | None = None
    inch_match = re.search(r"(\d+(?:\.\d+)?)\s*inch(?:es)?", question)
    deg_match  = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|°f?)", question)
    if inch_match:
        threshold = float(inch_match.group(1))
        unit = "inches"
    elif deg_match:
        threshold = float(deg_match.group(1))
        unit = "fahrenheit"

    # Comparison direction for temperature questions
    # Default: 'gte' (Will Chicago's high exceed 85°F? => P(temp ≥ 85))
    comparison = "gte"
    if any(w in question for w in ["below", "less than", "under", "stay under", "fall below"]):
        comparison = "lte"
    elif any(w in question for w in ["exceed", "above", "more than", "over", "reach", "hit", "surpass"]):
        comparison = "gte"

    date_str = _extract_date_from_question(question, contract.get("resolution_date", ""))

    return {
        "market_class": "daily_weather",
        "lat":          lat,
        "lon":          lon,
        "date":         date_str,
        "variable":     variable,
        "threshold":    threshold,
        "unit":         unit,
        "comparison":   comparison,
    }


# ---------------------------------------------------------------------------
# Monthly precipitation bucket parser
# ---------------------------------------------------------------------------

# Maps month name -> (number, days). Used to compute exact resolution window.
_MONTH_INFO = {
    "january": (1, 31), "february": (2, 28), "march": (3, 31), "april": (4, 30),
    "may": (5, 31), "june": (6, 30), "july": (7, 31), "august": (8, 31),
    "september": (9, 30), "october": (10, 31), "november": (11, 30), "december": (12, 31),
}


# Station trust by city. NOAA-resolved markets match Open-Meteo's archive
# closely (both use ASOS-style observations). International markets resolve
# via local met agencies whose station/methodology may differ — so we tag
# them low-trust until we integrate the actual resolution source.
_HIGH_TRUST_CITIES = {
    "new york", "nyc", "chicago", "los angeles", "miami", "seattle", "boston",
    "dallas", "denver", "atlanta", "houston", "phoenix",
}
_MEDIUM_TRUST_CITIES = set()  # currently empty
_LOW_TRUST_CITIES = {"london", "seoul", "hong kong", "tokyo", "paris", "berlin",
                     "madrid", "sydney", "singapore", "mumbai"}


def _city_trust(city: str) -> str:
    if city in _HIGH_TRUST_CITIES:
        return "high"
    if city in _MEDIUM_TRUST_CITIES:
        return "medium"
    return "low"


def _parse_monthly_precip(question: str, resolution_date: str) -> dict | None:
    """
    Match patterns like:
      "Will NYC have between 2 and 3 inches of precipitation in April?"
      "Will Seattle have less than 2.5 inches of precipitation in April?"
      "Will NYC have more than 6 inches of precipitation in April?"
      "Will London have between 20-30mm of precipitation in April?"
      "Will Seoul have 75mm or more of precipitation in April?"
      "Will Hong Kong have less than 130mm of precipitation in April?"

    Returns a metadata dict (with bucket bounds normalized to MM internally,
    but `unit` reflecting how the market was phrased) or None.
    """
    if "precipitation" not in question:
        return None

    # City detection — longest match first so "los angeles" beats just "la"
    city_name = None
    lat = lon = None
    for city in sorted(CITY_COORDS.keys(), key=len, reverse=True):
        if city in question:
            city_name = city
            lat, lon = CITY_COORDS[city]
            break
    if not city_name:
        return None

    # Month detection
    month_num = None
    month_name = None
    for mn, (num, _days) in _MONTH_INFO.items():
        if mn in question:
            month_name = mn
            month_num = num
            break
    if month_num is None:
        return None

    # Year — try resolution_date first, then question, fallback to current year
    year = None
    if resolution_date:
        try:
            year = datetime.fromisoformat(resolution_date.replace("Z","+00:00")).year
        except (ValueError, TypeError):
            pass
    if year is None:
        ym = re.search(r"\b(20\d{2})\b", question)
        year = int(ym.group(1)) if ym else date.today().year

    # Detect unit. mm if "mm" appears, else inches (default).
    is_mm = bool(re.search(r"\d+\s*mm\b", question))
    unit = "mm" if is_mm else "inches"

    # Bucket extraction — supports several phrasings
    low = high = None

    # Pattern 1: "between X and Y inches/mm" or "between X-Y mm"
    m = re.search(r"between\s+([\d.]+)\s*(?:and|[-–])\s*([\d.]+)\s*(?:mm|inch)", question)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
    if low is None:
        # Pattern 2: "less than / under / below X"
        m = re.search(r"(?:less than|under|below)\s+([\d.]+)\s*(?:mm|inch)", question)
        if m:
            low, high = 0.0, float(m.group(1))
    if low is None:
        # Pattern 3: "more than / above / over / at least X" / "X or more"
        m = re.search(r"(?:more than|above|over|at least)\s+([\d.]+)\s*(?:mm|inch)", question)
        if m:
            low, high = float(m.group(1)), float("inf")
    if low is None:
        # Pattern 4: "X mm or more" (the common phrasing for the open-ended top bucket)
        m = re.search(r"([\d.]+)\s*(?:mm|inch(?:es)?)\s+or\s+more", question)
        if m:
            low, high = float(m.group(1)), float("inf")
    if low is None:
        return None

    # Normalize internally to MM so the model can use the same units everywhere
    if unit == "inches":
        low_mm  = low * 25.4
        high_mm = high * 25.4 if high != float("inf") else float("inf")
    else:
        low_mm, high_mm = low, high

    return {
        "market_class":   "monthly_precip",
        "city":           city_name,
        "lat":            lat,
        "lon":            lon,
        "station_trust":  _city_trust(city_name),
        "month":          month_num,
        "year":           year,
        "month_iso":      f"{year}-{month_num:02d}",
        "bucket_low":     low,
        "bucket_high":    high,
        "bucket_low_mm":  round(low_mm, 2),
        "bucket_high_mm": (round(high_mm, 2) if high_mm != float("inf") else None),
        "unit":           unit,
        # Daily-weather fields kept for downstream consistency
        "date":           f"{year}-{month_num:02d}-{_MONTH_INFO[month_name][1]:02d}",
        "variable":       "rain",
        "threshold":      None,
    }


def _extract_date_from_question(question: str, resolution_date: str) -> str | None:
    """Extract an ISO date from question text; fall back to resolution_date."""
    # Try resolution_date first — most reliable
    if resolution_date:
        try:
            dt = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except (ValueError, TypeError):
            pass

    # Look for "Month DD" patterns in the question
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    months.update({m[:3].lower(): i for i, m in enumerate(calendar.month_name) if m})

    for month_str, month_num in months.items():
        if month_str in question:
            day_match = re.search(rf"{month_str}\s+(\d{{1,2}})", question)
            if day_match:
                day = int(day_match.group(1))
                year_match = re.search(r"\b(202\d)\b", question)
                year = int(year_match.group(1)) if year_match else date.today().year
                try:
                    return date(year, month_num, day).isoformat()
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    print("=" * 70)
    print("1) search_weather_markets() — live Gamma API")
    print("=" * 70)
    markets = search_weather_markets(min_liquidity=500.0)
    print(f"\n{len(markets)} markets passed all filters\n")
    for m in markets:
        print(f"  - {m['contract_id'][:18]:18s}  ${m['liquidity_usd']:>9,.0f} liq  "
              f"YES={m['yes_price']:.3f}  | {m['question'][:80]}")

    print()
    print("=" * 70)
    print("2) parse_contract_metadata() — does any market parse?")
    print("=" * 70)
    parsed = [(m, parse_contract_metadata(m)) for m in markets]
    parseable = [(m, p) for m, p in parsed if p]
    print(f"{len(parseable)} of {len(markets)} have parseable city/date/variable")
    for m, p in parseable[:3]:
        print(f"\n  Question: {m['question']}")
        print(f"  Parsed:   {p}")

    print()
    print("=" * 70)
    print("3) Sanity check parser with a synthetic guide-style question")
    print("=" * 70)
    fake = {
        "question": "Will it snow more than 2 inches in NYC on March 20, 2026?",
        "resolution_date": "2026-03-20T23:59:59Z",
    }
    p = parse_contract_metadata(fake)
    print(f"  Synthetic Q: {fake['question']}")
    print(f"  Parsed:      {p}")
