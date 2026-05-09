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
