# Weather Arbitrage Bot — Claude Code Context

## What This Project Does
This is a Python trading bot that finds and trades pricing inefficiencies in
Polymarket weather prediction market contracts. It fetches probabilistic weather
forecasts from NOAA, NWS, and Open-Meteo, compares them to Polymarket's implied
probabilities, and executes trades when the edge exceeds 7 percentage points.

## Current Development Status — TRADING

The bot now generates real signals against live Polymarket weather markets.

- [x] config.py — env loading
- [x] db.py — SQLite + WAL + schema + helpers
- [x] weather.py — NWS, Open-Meteo, Tomorrow.io, NOAA + Brier-weighted ensemble
- [x] polymarket.py — Pulls weather markets via /events (multi-bucket child markets)
                     parses three market_classes: monthly_precip, hurricane, daily_weather
                     city coords are airport ASOS station coords (matches NOAA resolution)
- [x] sizing.py — fractional Kelly + 2% hard cap
- [x] edge.py — dispatches by market_class to correct model
- [x] risk.py — six checks
- [x] execution.py — paper-mode CLOB execution
- [x] main.py — APScheduler, --dry-run
- [x] dashboard.py — Streamlit UI
- [x] hurricane_model.py — historical base rate × ENSO × time-remaining
- [x] monthly_precip_model.py — observed (Open-Meteo archive) + forecast (ensemble)
                                = projected monthly total → P(in bucket)
- [x] backtest.py — resolved-market replay (caveats: needs trade-time prices + reanalysis)
- [x] tests/ — 64 pytest tests passing

## Live behavior (April 28, 2026)
- Found: 67 weather markets across 5,000 active events (60 monthly_precip, 5 hurricane, others)
- Skipped: International monthly_precip markets (London/Seoul/HK) — `low` station_trust
  until we integrate the actual resolution data sources (KMA/HKO/Met Office)
- Trade-eligible: NYC, Seattle precipitation; all hurricane markets
- Last paper run: 5 signals → 5 paper-executed trades
  * Hurricane landfall by May 31: NO @ 24.5pp edge
  * Named storm pre-season: NO @ 19.4pp edge
  * NYC 3-4" April precip: YES @ ~24pp edge
  * Seattle 2.5-3" April precip: NO @ 33pp edge (already over 2.5" with 3 days left)
  * Seattle 3-3.5" April precip: YES @ 33pp edge (same setup, opposite side)

## Architecture: market_class dispatch
```
search_weather_markets() walks /events
  ↓
parse_contract_metadata() classifies each child market
  ↓
analyze_contract() dispatches:
   monthly_precip → monthly_precip_model (Open-Meteo archive + ensemble)
   hurricane      → hurricane_model     (HURDAT2 base rates + ENSO)
   daily_weather  → weather.get_ensemble_probability (4-source ensemble)
  ↓
_build_signal() computes EV, Kelly, persists, returns
```

## Live API Reality
- Gamma `/markets?q=` is broken: ignores query, returns default-sorted results
- Gamma `/markets` ALSO does not return child markets of multi-market events
- Use `/events?active=true&closed=false` paginated, then walk event.markets
- Multi-bucket weather markets (NYC/Seattle/London/Seoul/HK precipitation) live there
- Outcome shape: outcomes/outcomePrices/clobTokenIds are JSON-encoded strings

## Live API Reality (Apr 2026)
- Gamma `/markets` returns market dicts with `outcomes`/`outcomePrices`/`clobTokenIds` as **JSON-encoded strings**, NOT a `tokens` array. polymarket._normalize_market handles this.
- `liquidityNum` (float) is preferred over `liquidity` (string).
- `q=` parameter matches description text — must post-filter on question text.
- Currently only ~5 weather markets exist on Polymarket, all multi-month seasonal hurricanes. The daily-city contracts the guide describes don't exist. Bot will sit idle until they return.

## Project Structure
```
bot/
├── main.py        # APScheduler orchestrator (not yet built)
├── weather.py     # Weather probability fetchers (NOAA, NWS, Open-Meteo)
├── polymarket.py  # Polymarket Gamma + CLOB API integration
├── edge.py        # Signal generation and EV calculation
├── sizing.py      # Kelly criterion position sizing
├── execution.py   # Order placement (paper/live toggle via PAPER_TRADE)
├── risk.py        # Risk rule enforcement
├── db.py          # SQLite: signals and positions tables
├── config.py      # All constants loaded from .env
└── dashboard.py   # Streamlit UI
```

## Core Conventions — ALWAYS Follow These

### Code Style
- Python 3.11+
- Type hints on all function signatures
- Return None on API failure, never raise to caller from API functions
- Every function that calls an external API must have: timeout=10, tenacity retry (3 attempts, exponential backoff)
- All API keys from os.getenv(), never hardcoded

### Key Data Structures

Signal dict (from edge.py):
```python
{
    "contract_id": str,
    "question": str,
    "market_p": float,        # Polymarket implied probability
    "model_p": float,         # Our ensemble weather probability
    "ev": float,              # Expected value per dollar
    "recommended_side": "YES" | "NO",
    "edge": float,            # |model_p - market_p|
    "disagreement": float,    # Std dev across weather sources
    "kelly_size": float,      # Recommended USDC position size
    "timestamp": str,         # ISO format UTC
}
```

Ensemble result (from weather.py):
```python
{
    "probability": float,
    "sources": list[dict],
    "disagreement": float,
    "n_sources": int,
}
```

### Critical Rules
1. PAPER_TRADE=True is the default. Never execute real orders unless PAPER_TRADE=False.
2. Every risk check must pass before execution (see risk.py).
3. Probability values must be clamped to [0.0, 1.0] everywhere.
4. All thresholds (edge, liquidity, Kelly fraction) come from config.py, never hardcoded.
5. SQLite uses WAL mode. Always use timeout=30 in sqlite3.connect().

## External APIs Used
| Service          | Base URL                                      | Auth                  | Rate Limit  |
|------------------|-----------------------------------------------|-----------------------|-------------|
| NOAA CDO         | https://www.ncdc.noaa.gov/cdo-web/api/v2      | Token header          | 1000/day    |
| NWS              | https://api.weather.gov                       | None (User-Agent req.)| ~1/sec      |
| Open-Meteo       | https://api.open-meteo.com/v1/forecast        | None                  | 10000/day   |
| Polymarket Gamma | https://gamma-api.polymarket.com              | None                  | ~1/sec      |
| Polymarket CLOB  | https://clob.polymarket.com                   | Private key           | 10/sec      |

## What NOT to Change Without Asking
- The function signatures of get_ensemble_probability(), analyze_contract(), run_all_checks()
- The SQLite schema (changing it requires a migration)
- The PAPER_TRADE guard in execution.py

## How to Run
```bash
cd ~/weather-arb-bot/bot
cp ../.env.example ../.env       # Fill in your keys first
python main.py                   # Full bot with scheduler
python main.py --dry-run         # Single run then exit
streamlit run dashboard.py       # Monitoring UI
pytest ../tests/                 # Run all tests
```
