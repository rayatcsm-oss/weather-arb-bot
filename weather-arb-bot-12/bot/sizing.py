# sizing.py
"""
Fractional Kelly position sizing for binary prediction-market bets.

Why 1/4 Kelly (default) instead of full Kelly:
  * Full Kelly maximizes log-bankroll growth ONLY if your probability estimate
    is exact. Any model error in p shifts you above the optimum, which is
    asymmetric: overbetting can wipe you out, underbetting only slows growth.
  * Our model probability `p_model` is an ensemble of weather forecasts —
    inherently noisy, with a Brier score we haven't yet calibrated to live
    Polymarket outcomes. Treating that as exact is reckless.
  * 1/4 Kelly preserves ~75% of the long-run growth rate of full Kelly while
    cutting variance dramatically and surviving bad streaks. Once we have
    3+ months of paper-trade data with a documented Brier score, consider
    moving to 1/2 Kelly.
"""

import os
import logging

from config import KELLY_FRACTION, MAX_POSITION_PCT, INITIAL_BANKROLL

logger = logging.getLogger(__name__)


def calculate_kelly_size(edge: float, odds: float, bankroll: float) -> float:
    """
    Recommended position size in USDC under fractional Kelly.

    Parameters
    ----------
    edge : float
        |p_model - p_market| — the probability edge we believe we have.
        Must be > 0; this function returns 0 for non-positive edge.
    odds : float
        DECIMAL NET odds for the side we're betting (profit per $1 risked).
        For YES priced at $0.60: odds = (1 - 0.60) / 0.60 = 0.667.
        For YES at $0.50: odds = 1.0 (even money).
    bankroll : float
        Total USDC available.

    Returns
    -------
    float
        Position size in USDC, capped at MAX_POSITION_PCT * bankroll.
        Minimum $1 (the smallest meaningful trade); 0 if no edge.

    Formula
    -------
    Full Kelly:        f = (b·p − q) / b      where b=odds, p=win_prob, q=1-p
    Fractional Kelly:  f_frac = f · KELLY_FRACTION   (default 0.25)
    Sized:             max(1, min(f_frac · bankroll, MAX_POSITION_PCT · bankroll))
    """
    if edge <= 0 or odds <= 0 or bankroll <= 0:
        return 0.0

    # Recover p_model from edge + market odds
    # For YES side: p_market = 1 / (1 + odds), so p_model = p_market + edge
    p_market = 1.0 / (1.0 + odds)
    p_model = p_market + edge
    p_model = min(max(p_model, 0.0), 1.0)  # clamp to [0, 1]

    b = odds
    q = 1.0 - p_model

    if b == 0:
        return 0.0

    full_kelly_fraction = (b * p_model - q) / b

    if full_kelly_fraction <= 0:
        logger.debug(f"Kelly fraction non-positive ({full_kelly_fraction:.4f}) — no bet")
        return 0.0

    # Apply fractional Kelly (env var wins over config default at runtime)
    fraction = float(os.getenv("KELLY_FRACTION", KELLY_FRACTION))
    fractional_kelly = full_kelly_fraction * fraction

    # Raw size, then cap at the hard position-pct ceiling
    raw_size = fractional_kelly * bankroll
    max_size = MAX_POSITION_PCT * bankroll
    position_size = min(raw_size, max_size)

    logger.debug(
        f"Kelly: full={full_kelly_fraction:.4f} fractional={fractional_kelly:.4f} "
        f"raw=${raw_size:.2f} capped=${position_size:.2f}"
    )

    return max(1.0, round(position_size, 2))


def get_bankroll() -> float:
    """
    Current available bankroll in USDC.

    Starts from BANKROLL_USDC (env var / config) and adjusts for realized P&L.
    This ensures Kelly sizing uses the current actual bankroll, not just the
    starting value — a critical requirement for proper Kelly math.

    In live mode this should also query the on-chain USDC balance to reconcile.
    """
    initial = float(os.getenv("BANKROLL_USDC", INITIAL_BANKROLL))

    # Adjust for realized P&L so Kelly sizes down after losses and up after wins
    try:
        from db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) AS total FROM positions "
                "WHERE status IN ('closed', 'closed_manual')"
            ).fetchone()
        realized_pnl = float(row["total"]) if row else 0.0
    except Exception:
        realized_pnl = 0.0

    # Floor at 10% of initial to avoid Kelly sizing going to zero after bad streaks
    current = initial + realized_pnl
    floor = initial * 0.10
    return max(floor, current)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    examples = [
        # (label,                     edge, odds, bankroll)
        ("YES @ $0.60, edge=10pp",    0.10, 0.667, 1000.0),
        ("YES @ $0.70, edge=15pp",    0.15, 0.429, 1000.0),
        ("YES @ $0.50, edge=5pp",     0.05, 1.000, 1000.0),
        ("YES @ $0.40, edge=8pp",     0.08, 1.500, 1000.0),
        ("Tiny edge below threshold", 0.02, 1.000, 1000.0),
        ("Negative edge → no bet",   -0.05, 1.000, 1000.0),
        ("Big edge but small bankroll", 0.20, 0.667,  100.0),
    ]

    print(f"\n{'='*70}")
    print(f"Kelly sizing examples (KELLY_FRACTION={KELLY_FRACTION}, MAX_POSITION_PCT={MAX_POSITION_PCT})")
    print(f"{'='*70}\n")
    print(f"{'Label':<32s} {'edge':>6s} {'odds':>7s} {'bankroll':>10s} {'->':>4s} {'size':>10s}")
    print("-" * 75)
    for label, edge, odds, bk in examples:
        size = calculate_kelly_size(edge, odds, bk)
        print(f"{label:<32s} {edge:>+6.3f} {odds:>7.3f} ${bk:>8.0f} {'->':>4s} ${size:>8.2f}")

    print(f"\nget_bankroll() returned: ${get_bankroll():.2f}")
