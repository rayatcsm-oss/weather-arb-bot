# edge.py
"""
Signal generation engine — the integration point that ties weather data,
Polymarket prices, EV calculation, and Kelly sizing together.

Pipeline per contract:
  1. parse_contract_metadata()   -> {lat, lon, date, variable, threshold}
  2. get_ensemble_probability()  -> ensemble {probability, disagreement, ...}
  3. calculate_ev() + determine_side()
  4. calculate_kelly_size()
  5. db.insert_signal() if EV passes EDGE_THRESHOLD
"""

import logging
from datetime import datetime, date, timezone

from config import EDGE_THRESHOLD, MIN_LIQUIDITY_USD, MAX_SOURCE_DISAGREEMENT
from weather import get_ensemble_probability
from polymarket import search_weather_markets, parse_contract_metadata
from db import insert_signal
from sizing import calculate_kelly_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def calculate_ev(p_model: float, p_market: float, side: str) -> float:
    """
    Expected value per dollar wagered.

    For YES at price p_market:
        win  with prob p_model     -> profit (1 - p_market)
        lose with prob (1-p_model) -> loss   (p_market)
        EV = p_model * (1 - p_market) - (1 - p_model) * p_market

    For NO at price (1 - p_market):
        win  with prob (1 - p_model) -> profit p_market
        lose with prob p_model       -> loss   (1 - p_market)
        EV = (1 - p_model) * p_market - p_model * (1 - p_market)
    """
    if side == "YES":
        return p_model * (1 - p_market) - (1 - p_model) * p_market
    elif side == "NO":
        return (1 - p_model) * p_market - p_model * (1 - p_market)
    else:
        raise ValueError(f"Invalid side: {side!r}")


def determine_side(p_model: float, p_market: float) -> tuple[str, float]:
    """
    Pick the side to bet and report the absolute edge.

    Returns ('YES', edge) when our model says the event is more likely than
    the market price implies, ('NO', edge) otherwise. edge = |p_model - p_market|.
    """
    delta = p_model - p_market
    if delta >= 0:
        return "YES", abs(delta)
    return "NO", abs(delta)


def _odds_for_side(p_market_yes: float, side: str) -> float:
    """
    Net decimal odds (profit per $1 risked) for the chosen side.

    BUGFIX: the guide's edge.py passes `1.0 / p_market_yes` here, but
    sizing.calculate_kelly_size expects net odds (1-p)/p. Off by 1 in the
    guide. Computing correctly per Kelly's formula.
    """
    if side == "YES":
        # Buying YES at price p; payout 1 - p per dollar risked
        if p_market_yes <= 0 or p_market_yes >= 1:
            return 0.0
        return (1.0 - p_market_yes) / p_market_yes
    else:  # NO
        # Buying NO at price (1 - p); payout p per dollar risked
        p_no = 1.0 - p_market_yes
        if p_no <= 0 or p_no >= 1:
            return 0.0
        return (1.0 - p_no) / p_no


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _analyze_daily_weather(
    contract: dict, metadata: dict, bankroll: float
) -> dict | None:
    """Analyze a daily-city weather contract via the ensemble forecast."""
    contract_id = contract.get("contract_id", "?")

    # Time-to-expiry early-out (full check in risk.py)
    try:
        resolution_date = date.fromisoformat(metadata["date"])
        seconds_to_expiry = (
            datetime.combine(resolution_date, datetime.max.time()) - datetime.now()
        ).total_seconds()
        hours_to_expiry = seconds_to_expiry / 3600
        if hours_to_expiry < 6:
            logger.debug(f"Skip {contract_id[:12]}: expires in {hours_to_expiry:.1f}h")
            return None
    except (ValueError, TypeError):
        pass

    # Ensemble weather probability
    ensemble = get_ensemble_probability(
        lat=metadata["lat"],
        lon=metadata["lon"],
        date_str=metadata["date"],
        variable=metadata.get("variable", "rain"),
    )
    if not ensemble or ensemble.get("probability") is None:
        logger.warning(f"Skip {contract_id[:12]}: no ensemble probability")
        return None

    return _build_signal(
        contract=contract,
        metadata=metadata,
        p_model=ensemble["probability"],
        disagreement=ensemble.get("disagreement", 0.0),
        n_sources=ensemble.get("n_sources", 0),
        sources=ensemble.get("sources", []),
        bankroll=bankroll,
    )


def _analyze_hurricane(
    contract: dict, metadata: dict, bankroll: float, oni: float | None = None
) -> dict | None:
    """Analyze a seasonal hurricane contract via the historical-base-rate model."""
    contract_id = contract.get("contract_id", "?")

    from hurricane_model import estimate_hurricane_probability, get_current_oni

    try:
        deadline = date.fromisoformat(metadata["deadline"])
    except (ValueError, TypeError, KeyError):
        logger.warning(f"Skip {contract_id[:12]}: bad hurricane deadline")
        return None

    hours_to_expiry = (
        datetime.combine(deadline, datetime.max.time()) - datetime.now()
    ).total_seconds() / 3600
    if hours_to_expiry < 6:
        logger.debug(f"Skip {contract_id[:12]}: hurricane market expires in {hours_to_expiry:.1f}h")
        return None

    # Use cached ONI if available, otherwise fetch (single-market analysis path)
    oni_value = oni if oni is not None else get_current_oni()

    result = estimate_hurricane_probability(
        market_type=metadata["hurricane_type"],
        deadline=deadline,
        oni=oni_value,
    )
    if not result:
        logger.warning(f"Skip {contract_id[:12]}: hurricane model returned None")
        return None

    return _build_signal(
        contract=contract,
        metadata=metadata,
        p_model=result["probability"],
        disagreement=0.0,
        n_sources=1,
        sources=[result],
        bankroll=bankroll,
    )


def _build_signal(
    contract: dict,
    metadata: dict,
    p_model: float,
    disagreement: float,
    n_sources: int,
    sources: list,
    bankroll: float,
) -> dict | None:
    """Common signal-construction step for both market classes."""
    contract_id = contract.get("contract_id", "?")
    question = contract.get("question", "")

    # Disagreement gate (only meaningful for the multi-source ensemble)
    if disagreement > MAX_SOURCE_DISAGREEMENT:
        logger.info(
            f"Skip {contract_id[:12]}: disagreement {disagreement:.3f} "
            f"> {MAX_SOURCE_DISAGREEMENT:.3f}"
        )
        return None

    p_market_yes = contract.get("yes_price", 0.5)

    # Clamp p_model to a sane range — values at 0.0 or 1.0 mean the underlying
    # model hit a degenerate case (e.g. fully-observed month already in bucket,
    # or all ensemble members agree). Trading on these extremes produces
    # unreliable Kelly sizes; cap at 5%–95% to reflect irreducible uncertainty.
    P_MODEL_CLAMP_LOW  = 0.05
    P_MODEL_CLAMP_HIGH = 0.95
    if p_model < P_MODEL_CLAMP_LOW or p_model > P_MODEL_CLAMP_HIGH:
        logger.info(
            f"Clamping p_model {p_model:.4f} -> "
            f"[{P_MODEL_CLAMP_LOW},{P_MODEL_CLAMP_HIGH}] for {contract_id[:12]}"
        )
        p_model = max(P_MODEL_CLAMP_LOW, min(P_MODEL_CLAMP_HIGH, p_model))

    side, edge = determine_side(p_model, p_market_yes)
    if edge < EDGE_THRESHOLD:
        logger.debug(
            f"Skip {contract_id[:12]}: edge {edge:.3f} < {EDGE_THRESHOLD:.3f} "
            f"(model={p_model:.3f} market={p_market_yes:.3f})"
        )
        return None

    # GUIDE SECTION 5: The only entry filter is |P_model - P_market| > 0.07
    # The previous 55% confidence rule was WRONG — it blocked valid trades like
    # "Will 2026 be the third-hottest year?" where model=44%, market=2.8% (+41pp edge).
    #
    # The Seattle 3-3.5" loss happened because of a DATA SOURCE problem (wrong
    # station coordinates), not because the model said 42%. We fixed that by adding
    # the resolution-station lookup and the data-mismatch sanity check.
    #
    # The correct protection against bad bets is:
    #   1. Edge > 7pp threshold (guide's rule) — already checked above
    #   2. Data-mismatch sanity check (our addition) — applied per market class
    #   3. Minimum model floor of 10% — prevents bets where model has no real signal
    #      (e.g., model says 2% YES with market at 1% → 1pp of apparent edge is noise)
    MIN_MODEL_FLOOR = 0.10
    if side == "YES" and p_model < MIN_MODEL_FLOOR:
        logger.info(
            f"Skip {contract_id[:12]}: model {p_model:.3f} below {MIN_MODEL_FLOOR:.0%} floor for YES"
        )
        return None
    if side == "NO" and p_model > (1.0 - MIN_MODEL_FLOOR):
        logger.info(
            f"Skip {contract_id[:12]}: model {p_model:.3f} above {1-MIN_MODEL_FLOOR:.0%} for NO"
        )
        return None
    # Also protect NO bets when the model says the event is very likely to happen
    # (model < 15% means we're betting NO that something almost certainly won't happen —
    # that's fine. But if model > 85% and we're betting NO, something is wrong.)
    if side == "NO" and p_model < MIN_MODEL_FLOOR:
        logger.info(
            f"Skip {contract_id[:12]}: model {p_model:.3f} implausibly low (< {MIN_MODEL_FLOOR:.0%}) for NO bet"
        )
        return None

    # EV must be positive
    ev = calculate_ev(p_model, p_market_yes, side)
    if ev <= 0:
        logger.debug(f"Skip {contract_id[:12]}: EV {ev:.4f} <= 0")
        return None

    odds = _odds_for_side(p_market_yes, side)
    kelly_size = calculate_kelly_size(edge=edge, odds=odds, bankroll=bankroll)

    signal = {
        "contract_id":      contract_id,
        "question":         question,
        "market_class":     metadata.get("market_class"),
        "market_p":         round(p_market_yes, 4),
        "model_p":          round(p_model, 4),
        "ev":               round(ev, 4),
        "recommended_side": side,
        "edge":             round(edge, 4),
        "disagreement":     round(disagreement, 4),
        "kelly_size":       round(kelly_size, 2),
        "n_sources":        n_sources,
        "sources":          sources,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "metadata":         metadata,
        "yes_token_id":     contract.get("yes_token_id"),
        "no_token_id":      contract.get("no_token_id"),
        "liquidity_usd":    contract.get("liquidity_usd", 0.0),
        "resolution_date":  contract.get("resolution_date", ""),
    }

    logger.info(
        f"SIGNAL {contract_id[:12]} [{metadata.get('market_class')}] | "
        f"side={side} model={p_model:.3f} market={p_market_yes:.3f} "
        f"edge={edge:.3f} EV={ev:.3f} kelly=${kelly_size:.2f}"
    )
    return signal


def _analyze_monthly_precip(
    contract: dict, metadata: dict, bankroll: float
) -> dict | None:
    """
    Analyze a "Precipitation in [city] in [month]" bucket market via the
    monthly_precip_model (observed + ensemble forecast → bucket probability).

    Adjusts trust based on resolution-station match:
      * high   - trade normally
      * medium - only emit signal if edge >= EDGE_THRESHOLD * 1.5
      * low    - require edge >= EDGE_THRESHOLD * 3.0 OR skip
    """
    contract_id = contract.get("contract_id", "?")

    from monthly_precip_model import estimate_bucket_probability

    # EARLY EXIT: Skip low-trust cities before making any API calls.
    # These markets resolve via local met agencies that differ from Open-Meteo,
    # so even a huge apparent edge is unreliable. Skipping here avoids wasting
    # 15-20 seconds per scan on Open-Meteo archive/forecast requests that we'd
    # discard anyway.
    trust = metadata.get("station_trust", "low")
    if trust == "low":
        logger.info(
            f"Skip {contract_id[:12]} [low-trust]: city={metadata.get('city')} "
            f"— resolution station data not yet integrated"
        )
        return None

    # CRITICAL FIX: Use the resolution station's coordinates, not generic city coords.
    # Polymarket specifies the exact station in the market description (e.g., "Central Park NY").
    # Open-Meteo's gridded data varies by ~0.3" across a metro area, which is enough
    # to change which bucket the total falls in. Using the wrong grid cell = wrong bet.
    lat = metadata["lat"]
    lon = metadata["lon"]
    desc = (contract.get("description") or "").lower()

    # Resolution station lookup — match the description to known NOAA stations
    RESOLUTION_COORDS = {
        "central park":   (40.7789, -73.9692),   # GHCND:USW00094728
        "jfk":            (40.6413, -73.7781),   # GHCND:USW00094789
        "laguardia":      (40.7769, -73.8740),   # GHCND:USW00014732
        "sea-tac":        (47.4502, -122.3088),  # GHCND:USW00024233
        "seattle-tacoma": (47.4502, -122.3088),
        "o'hare":         (41.9786, -87.9048),   # GHCND:USW00094846
        "midway":         (41.7868, -87.7522),   # GHCND:USW00014819
        "miami intl":     (25.7906, -80.3164),   # GHCND:USW00012839
        "heathrow":       (51.4700, -0.4543),
        "hong kong obs":  (22.3017, 114.1747),
    }
    station_name = None
    for keyword, coords in RESOLUTION_COORDS.items():
        if keyword in desc:
            lat, lon = coords
            station_name = keyword
            break

    if station_name:
        logger.info(f"{contract_id[:12]}: using resolution station '{station_name}' coords ({lat},{lon})")
    else:
        logger.debug(f"{contract_id[:12]}: no resolution station found in description, using default city coords")

    try:
        result = estimate_bucket_probability(
            lat=lat,
            lon=lon,
            year=metadata["year"],
            month=metadata["month"],
            bucket_low_mm=metadata["bucket_low_mm"],
            bucket_high_mm=metadata["bucket_high_mm"],
        )
    except Exception as e:
        logger.warning(f"Skip {contract_id[:12]}: monthly_precip_model crashed: {e}")
        return None

    if not result or result.get("probability") is None:
        logger.warning(f"Skip {contract_id[:12]}: monthly_precip returned no probability")
        return None

    # SANITY CHECK: If our model disagrees with a high-volume market by >50pp,
    # it usually means our data source is wrong (wrong station, timezone issue,
    # stale archive data), not that we found a massive edge. Skip with a warning.
    p_market = contract.get("yes_price", 0.5)
    raw_edge = abs(result["probability"] - p_market)
    volume = contract.get("volume_usd", 0)

    if raw_edge > 0.50 and volume > 5000:
        logger.warning(
            f"Skip {contract_id[:12]}: edge {raw_edge:.3f} too large on ${volume:,.0f} volume market. "
            f"Model={result['probability']:.3f} Market={p_market:.3f}. "
            f"Likely data source mismatch — check resolution station."
        )
        return None

    # Station-trust gating (low-trust already bailed out at top of function)
    if trust == "medium":
        if raw_edge < EDGE_THRESHOLD * 1.5:
            return None
        if raw_edge > 0.50:
            logger.warning(f"Skip {contract_id[:12]} [medium-trust]: edge too large to trust")
            return None

    return _build_signal(
        contract=contract,
        metadata=metadata,
        p_model=result["probability"],
        disagreement=0.0,
        n_sources=1,
        sources=[result],
        bankroll=bankroll,
    )


def _analyze_global_temp(
    contract: dict, metadata: dict, bankroll: float, oni_cache: float | None = None
) -> dict | None:
    """Analyze a global annual temperature ranking market."""
    contract_id = contract.get("contract_id", "?")

    from global_temp_model import estimate_temp_ranking_probability

    result = estimate_temp_ranking_probability(
        year=metadata["temp_year"],
        rank=metadata["temp_rank"],
        direction=metadata.get("temp_direction", "hottest"),
        lower_bound=metadata.get("temp_lower_bound", False),
        oni=oni_cache,
    )
    if not result:
        logger.warning(f"Skip {contract_id[:12]}: global_temp model returned None")
        return None

    # Annual temperature models have wider uncertainty than daily forecasts.
    # Require 10pp edge (vs guide's 7pp default) to ensure a real signal.
    p_market = contract.get("yes_price", 0.5)
    raw_edge = abs(result["probability"] - p_market)
    if raw_edge < 0.10:
        logger.debug(f"Skip {contract_id[:12]}: global_temp edge {raw_edge:.3f} < 0.10")
        return None

    return _build_signal(
        contract=contract,
        metadata=metadata,
        p_model=result["probability"],
        disagreement=0.0,
        n_sources=1,
        sources=[result],
        bankroll=bankroll,
    )


def analyze_contract(
    contract: dict,
    bankroll: float = 1000.0,
    oni_cache: float | None = None,
    _metadata: dict | None = None,
) -> dict | None:
    """
    Full pipeline for one Polymarket contract. Dispatches to the correct
    sub-analyzer based on parsed market_class.

    _metadata: optional pre-parsed metadata (from run_edge_scan's batch parse)
               to avoid calling parse_contract_metadata twice per contract.
    """
    contract_id = contract.get("contract_id", "?")
    question = contract.get("question", "")

    metadata = _metadata if _metadata is not None else parse_contract_metadata(contract)
    if not metadata:
        logger.debug(f"Skip {contract_id[:12]}: cannot parse | {question[:80]}")
        return None

    market_class = metadata.get("market_class")
    if market_class == "global_temp":
        return _analyze_global_temp(contract, metadata, bankroll, oni_cache=oni_cache)
    elif market_class == "hurricane":
        return _analyze_hurricane(contract, metadata, bankroll, oni=oni_cache)
    elif market_class == "monthly_precip":
        return _analyze_monthly_precip(contract, metadata, bankroll)
    elif market_class == "daily_weather":
        if not metadata.get("lat") or not metadata.get("date"):
            logger.debug(f"Skip {contract_id[:12]}: missing lat/date in daily metadata")
            return None
        return _analyze_daily_weather(contract, metadata, bankroll)
    elif market_class == "monthly_temp_rank":
        # TODO: build full monthly-temp-ranking model (needs NOAA monthly anomaly data)
        # For now: log that we recognise the market but have no model yet.
        logger.info(
            f"Skip {contract_id[:12]} [monthly_temp_rank]: no model yet | {question[:60]}"
        )
        return None
    elif market_class == "monthly_temp_anomaly":
        # TODO: build monthly global-temperature anomaly model
        logger.info(
            f"Skip {contract_id[:12]} [monthly_temp_anomaly]: no model yet | {question[:60]}"
        )
        return None
    elif market_class == "arctic_sea_ice":
        # TODO: build Arctic sea ice extent model (NSIDC data)
        logger.info(
            f"Skip {contract_id[:12]} [arctic_sea_ice]: no model yet | {question[:60]}"
        )
        return None
    else:
        logger.debug(f"Skip {contract_id[:12]}: unknown market_class {market_class!r}")
        return None


def run_edge_scan(bankroll: float = 1000.0) -> list[dict]:
    """
    Top-level scan: enumerate weather markets, analyze each, persist signals.
    Returns the list of signals generated this run.
    """
    logger.info("Starting edge scan...")
    contracts = search_weather_markets(min_liquidity=MIN_LIQUIDITY_USD)
    logger.info(f"Analyzing {len(contracts)} weather contracts")

    # Pre-parse all metadata once so we don't call parse_contract_metadata twice per contract.
    # Key by contract_id — the canonical ID field set by _normalize_market — so the lookup
    # below uses the same key as the build, avoiding silent cache misses.
    def _contract_key(c: dict) -> str:
        return c.get("contract_id") or c.get("conditionId") or str(c.get("id", ""))

    parsed_metadata: dict[str, dict | None] = {
        _contract_key(c): parse_contract_metadata(c)
        for c in contracts
    }

    # Pre-fetch ENSO ONI once for the whole scan
    oni_cache: float | None = None
    has_hurricane = any(
        m and m.get("market_class") in ("hurricane", "global_temp")
        for m in parsed_metadata.values()
    )
    if has_hurricane:
        try:
            from hurricane_model import get_current_oni
            oni_cache = get_current_oni()
            logger.info(f"ENSO ONI for this scan: {oni_cache}")
        except Exception as e:
            logger.warning(f"ONI fetch failed: {e}")

    signals: list[dict] = []
    for contract in contracts:
        cid = _contract_key(contract)
        pre_meta = parsed_metadata.get(cid)   # always hits — same key function used above
        signal = analyze_contract(contract, bankroll=bankroll, oni_cache=oni_cache, _metadata=pre_meta)
        if signal:
            try:
                insert_signal(signal)
            except Exception as e:
                logger.exception(f"Failed to persist signal {signal['contract_id'][:12]}: {e}")
            signals.append(signal)

    logger.info(
        f"Edge scan complete: {len(signals)} signals from {len(contracts)} contracts"
    )
    return signals


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    # ------- 1) Pure math sanity ----------------------------------------
    print("=" * 70)
    print("1) calculate_ev / determine_side / _odds_for_side")
    print("=" * 70)
    ev_yes = calculate_ev(0.72, 0.58, "YES")
    print(f"  EV(p_model=0.72, p_market=0.58, YES) = {ev_yes:.4f}  (expect ~+0.140)")
    ev_no = calculate_ev(0.30, 0.58, "NO")
    print(f"  EV(p_model=0.30, p_market=0.58, NO)  = {ev_no:.4f}  (expect ~+0.280)")
    side, edge = determine_side(0.72, 0.58)
    print(f"  determine_side(0.72, 0.58) = ({side!r}, {edge:.3f})")
    side, edge = determine_side(0.30, 0.58)
    print(f"  determine_side(0.30, 0.58) = ({side!r}, {edge:.3f})")
    print(f"  odds_for_side(0.60, 'YES') = {_odds_for_side(0.60, 'YES'):.4f}  (expect 0.6667)")
    print(f"  odds_for_side(0.60, 'NO')  = {_odds_for_side(0.60, 'NO'):.4f}  (expect 1.5)")

    # ------- 2 & 3) analyze_contract with monkey-patched ensemble --------
    # Note: this script runs as __main__, so unittest.mock.patch("edge.X")
    # would patch a different module object. Patch via __main__ directly.
    me = sys.modules[__name__]
    real_ensemble_fn = me.get_ensemble_probability

    fake_contract = {
        "contract_id":     "0xtest",
        "question":        "Will it snow more than 2 inches in NYC on March 20, 2027?",
        "yes_price":       0.45,
        "no_price":        0.55,
        "yes_token_id":    "tok_yes",
        "no_token_id":     "tok_no",
        "liquidity_usd":   5000.0,
        "resolution_date": "2027-03-20T23:59:59Z",
    }

    print()
    print("=" * 70)
    print("2) analyze_contract with mocked ensemble (good signal expected)")
    print("=" * 70)
    me.get_ensemble_probability = lambda **_kw: {
        "probability":  0.62,
        "sources":      [{"source": "nws", "probability": 0.62, "confidence": 0.85}],
        "disagreement": 0.05,
        "n_sources":    1,
    }
    sig = analyze_contract(fake_contract, bankroll=1000.0)
    if sig:
        for k in ["recommended_side", "model_p", "market_p", "edge", "ev",
                  "kelly_size", "disagreement", "n_sources"]:
            print(f"  {k:18s} {sig[k]}")
    else:
        print("  No signal returned (something is off)")

    print()
    print("=" * 70)
    print("3) High disagreement should suppress the signal")
    print("=" * 70)
    me.get_ensemble_probability = lambda **_kw: {
        "probability":  0.62,
        "sources":      [],
        "disagreement": 0.30,   # > MAX_SOURCE_DISAGREEMENT (0.15)
        "n_sources":    3,
    }
    sig = analyze_contract(fake_contract, bankroll=1000.0)
    print(f"  signal returned: {sig is not None}  (expect False)")

    # restore for the live scan below
    me.get_ensemble_probability = real_ensemble_fn

    # ------- 4) Live run_edge_scan against real APIs --------------------
    print()
    print("=" * 70)
    print("4) Live run_edge_scan() against real Polymarket + Weather APIs")
    print("=" * 70)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["DB_PATH"] = tmp.name

    import importlib, config, db
    importlib.reload(config); importlib.reload(db)
    db.init_db()

    signals = run_edge_scan(bankroll=1000.0)
    print(f"  signals generated: {len(signals)}")
    for s in signals[:3]:
        print(f"    - {s['contract_id'][:12]} side={s['recommended_side']} "
              f"edge={s['edge']:.3f} EV={s['ev']:.3f} size=${s['kelly_size']:.2f}")

    # cleanup
    for ext in ("", "-wal", "-shm"):
        p = tmp.name + ext
        if os.path.exists(p): os.unlink(p)
