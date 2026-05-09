# backtest.py
"""
Historical backtest for the weather edge model.

Pulls resolved Polymarket weather markets, replays each through our model,
and reports calibration (Brier score) and simulated P&L.

TWO IMPORTANT CAVEATS:

1. Forecast freshness: The weather APIs return CURRENT/FORECAST data, not
   what the forecast looked like at the original trade time. So this
   backtest is ONLY meaningful when the resolved date is in the past AND
   we use a true historical reanalysis source (NOAA archive, ERA5).
   Pure-forecast sources (NWS, Open-Meteo forecast API) will return None
   for past dates and the row gets dropped.

2. Market price freshness: The Gamma /markets endpoint exposes only the
   FINAL closing price (1.0 for winner, 0.0 for loser). For real edge
   measurement we need trade-time prices from the CLOB price-history
   endpoint, which isn't wired up yet. Until that's done, the `pnl`
   number here is unreliable. Brier score on `model_p` vs outcome is the
   only metric in this module that's currently trustworthy.

For real forecast-skill measurement, we'd need:
  * a forecast archive (HRRR archive, ECMWF reforecast, ERA5)
  * the CLOB price-history endpoint for entry-time market prices
Both are TODOs.
"""

import json
import logging
from datetime import date, datetime, timedelta

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from polymarket import _normalize_market, parse_contract_metadata, _question_looks_weather
from weather import get_ensemble_probability
from config import EDGE_THRESHOLD

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Fetching resolved markets
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_resolved_page(offset: int, limit: int = 500) -> list[dict]:
    """One page of closed markets sorted by most recent end date."""
    resp = httpx.get(
        f"{GAMMA_BASE}/markets",
        params={
            "closed":    "true",
            "limit":     limit,
            "offset":    offset,
            "order":     "endDate",
            "ascending": "false",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json() or []


def fetch_resolved_weather_markets(
    days_back: int = 180,
    max_pages: int = 10,
) -> list[dict]:
    """
    Pull resolved weather markets that ended in the last `days_back` days.

    Walks paginated `closed=true` markets, filters question text for weather
    keywords, and stops as soon as we hit markets older than the cutoff.
    """
    cutoff = date.today() - timedelta(days=days_back)
    found: list[dict] = []

    for page in range(max_pages):
        offset = page * 500
        try:
            batch = _fetch_resolved_page(offset, limit=500)
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            break
        if not batch:
            break

        # The page is sorted by endDate desc — once we drop below cutoff, stop
        oldest_in_batch = batch[-1].get("endDate", "")
        for m in batch:
            if not _question_looks_weather(m.get("question", "")):
                continue
            try:
                end_dt = datetime.fromisoformat(
                    m.get("endDate", "").replace("Z", "+00:00")
                )
                if end_dt.date() < cutoff:
                    continue
            except (ValueError, TypeError):
                continue
            found.append(m)

        # Stop if the *oldest* row in this batch is already past the cutoff
        try:
            oldest_dt = datetime.fromisoformat(oldest_in_batch.replace("Z", "+00:00"))
            if oldest_dt.date() < cutoff:
                logger.info(
                    f"Stopping pagination at page {page}: oldest endDate {oldest_dt.date()} < cutoff {cutoff}"
                )
                break
        except (ValueError, TypeError):
            pass

    logger.info(f"Found {len(found)} weather questions resolved in last {days_back}d")
    return found


# ---------------------------------------------------------------------------
# Per-contract replay
# ---------------------------------------------------------------------------

def _resolved_outcome(raw: dict) -> int | None:
    """
    Decode the actual YES/NO winner from a closed market.

    Polymarket stores final prices as decimal strings — the winning side
    rounds to 1.0, loser to 0.0. Returns 1 if YES won, 0 if NO won, None
    if the market didn't have a clean resolution (price strings like
    '0','0' from the very early Polymarket era).
    """
    try:
        outcomes = json.loads(raw.get("outcomes", "[]"))
        prices   = json.loads(raw.get("outcomePrices", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).upper() == "YES"), None)
    if yes_idx is None:
        return None

    try:
        yes_final = float(prices[yes_idx])
        no_final  = float(prices[1 - yes_idx])
    except (ValueError, TypeError):
        return None

    if yes_final > 0.99:
        return 1
    if no_final > 0.99:
        return 0
    return None


def backtest_contract(raw_contract: dict) -> dict | None:
    """
    Replay one resolved contract through the model.

    Returns:
      {
        contract_id, question,
        model_p, market_p, outcome (0/1),
        brier_score, edge, had_signal, recommended_side, pnl,
      }
    or None if this contract isn't usable (not parseable, no outcome data, etc).
    """
    normalized = _normalize_market(raw_contract)
    if not normalized:
        return None

    metadata = parse_contract_metadata(normalized)
    if not metadata or not metadata.get("date") or not metadata.get("lat"):
        return None

    outcome = _resolved_outcome(raw_contract)
    if outcome is None:
        return None

    # Get what our ensemble would have produced for this date.
    # NOTE: this fetches the *current* forecast for that historical date,
    # which is only meaningful if the date is in the past (free archive
    # endpoints will return historical data). For dates >7d in the past
    # most of our forecast sources will return None → no result.
    try:
        ensemble = get_ensemble_probability(
            lat=metadata["lat"],
            lon=metadata["lon"],
            date_str=metadata["date"],
            variable=metadata.get("variable", "rain"),
        )
    except Exception as e:
        logger.debug(f"Ensemble call failed for {normalized['contract_id'][:12]}: {e}")
        return None
    if not ensemble or ensemble.get("probability") is None:
        return None

    p_model  = ensemble["probability"]
    p_market = normalized["yes_price"]   # last/closing market price as proxy

    # Brier score: lower is better (0 = perfect, 0.25 = always-50/50, 1 = always-wrong)
    brier = (p_model - outcome) ** 2

    # Simulated P&L at $10 per signal, threshold-gated like live
    edge = p_model - p_market
    had_signal = abs(edge) >= EDGE_THRESHOLD
    side = "YES" if edge >= 0 else "NO"
    if had_signal:
        if side == "YES":
            pnl = (outcome - p_market) * 10
        else:
            pnl = ((1 - outcome) - (1 - p_market)) * 10
    else:
        pnl = 0.0

    return {
        "contract_id":      normalized["contract_id"],
        "question":         normalized["question"][:100],
        "model_p":          round(p_model, 4),
        "market_p":         round(p_market, 4),
        "outcome":          outcome,
        "brier_score":      round(brier, 4),
        "edge":             round(edge, 4),
        "had_signal":       had_signal,
        "recommended_side": side,
        "pnl":              round(pnl, 4),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def run_backtest(days_back: int = 180, max_pages: int = 10) -> dict:
    """Run the full backtest and return summary stats + per-contract details."""
    raw = fetch_resolved_weather_markets(days_back=days_back, max_pages=max_pages)
    logger.info(f"Backtesting {len(raw)} resolved weather markets")

    results: list[dict] = []
    for r in raw:
        try:
            row = backtest_contract(r)
            if row:
                results.append(row)
        except Exception as e:
            logger.exception(f"backtest_contract crashed: {e}")
            continue

    if not results:
        return {
            "total_contracts":   len(raw),
            "usable_contracts":  0,
            "details":           [],
            "note":              "Zero usable rows — typical when no recent daily weather markets exist on Polymarket.",
        }

    briers = [r["brier_score"] for r in results]
    signals_only = [r for r in results if r["had_signal"]]
    wins = [r for r in signals_only if r["pnl"] > 0]

    return {
        "total_contracts":      len(raw),
        "usable_contracts":     len(results),
        "signals_generated":    len(signals_only),
        "avg_brier_score":      round(sum(briers) / len(briers), 4),
        "win_rate":             round(len(wins) / len(signals_only), 4) if signals_only else None,
        "avg_pnl_per_signal":   round(sum(r["pnl"] for r in signals_only) / len(signals_only), 4) if signals_only else None,
        "total_pnl":            round(sum(r["pnl"] for r in results), 4),
        "details":              results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    p = argparse.ArgumentParser(description="Backtest the weather edge model")
    p.add_argument("--days", type=int, default=180, help="Look-back window (days)")
    p.add_argument("--max-pages", type=int, default=10, help="Max pages of 500 closed markets to scan")
    args = p.parse_args()

    result = run_backtest(days_back=args.days, max_pages=args.max_pages)
    summary = {k: v for k, v in result.items() if k != "details"}

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:24s} {v}")

    if result.get("details"):
        print("\nPer-contract details (first 10):")
        for r in result["details"][:10]:
            print(f"  outcome={r['outcome']} model={r['model_p']:.3f} market={r['market_p']:.3f} "
                  f"brier={r['brier_score']:.4f} signal={r['had_signal']} pnl={r['pnl']:+.2f}")
            print(f"     Q: {r['question'][:90]}")
