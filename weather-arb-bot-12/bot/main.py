# main.py
"""
Bot orchestrator.

Two scheduled jobs:
  * trading_run    — every 30 minutes: edge scan → risk check → execute
  * discovery_run  — every 4 hours: enumerate active weather markets (heartbeat)

Both run once at startup before the scheduler takes over.

CLI flags:
  --dry-run   Run one trading_run, then exit. Useful for cron deployments
              and end-to-end smoke testing without the scheduler.
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import LOG_LEVEL, PAPER_TRADE, DB_PATH
from db import init_db
from edge import run_edge_scan
from risk import run_all_checks
from execution import get_clob_client, execute_signal, _CLOB_AVAILABLE
from sizing import get_bankroll
from polymarket import search_weather_markets


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"


def configure_logging() -> logging.Logger:
    """Configure root logger to write to stdout and a rotating bot.log."""
    os.makedirs("logs", exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any handlers added by libraries before us
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(LOG_FORMAT)

    # Stream handler -> stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Rotating file handler -> logs/bot.log (10 MB x 5)
    fh = RotatingFileHandler("logs/bot.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Quiet noisy upstream loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    return logging.getLogger("main")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

def discovery_run() -> None:
    """Heartbeat: confirm we can reach the Gamma API and count weather markets."""
    logger.info("=== DISCOVERY RUN START ===")
    try:
        markets = search_weather_markets()
        logger.info(f"Discovery: found {len(markets)} active weather markets above liquidity floor")
    except Exception as e:
        logger.exception(f"Discovery run failed: {e}")
    logger.info("=== DISCOVERY RUN END ===")


def trading_run() -> dict:
    """
    One full pass:
      1. Get bankroll
      2. Generate signals via edge.run_edge_scan
      3. For each signal: risk check → execute or skip
      4. Log a summary

    Returns a stats dict so --dry-run callers (and tests) can introspect.
    """
    logger.info("=== TRADING RUN START ===")
    bankroll = get_bankroll()
    logger.info(f"Bankroll: ${bankroll:.2f} | Paper trade: {PAPER_TRADE}")

    stats = {"signals": 0, "executed": 0, "skipped_risk": 0, "execution_errors": 0}

    # Generate
    try:
        signals = run_edge_scan(bankroll=bankroll)
    except Exception as e:
        logger.exception(f"Edge scan failed: {e}")
        logger.info("=== TRADING RUN END ===")
        return stats

    stats["signals"] = len(signals)
    if not signals:
        logger.info("No signals this run")
        logger.info("=== TRADING RUN END ===")
        return stats

    # CLOB client (None in paper mode)
    client = get_clob_client()

    # Filter + execute
    for signal in signals:
        cid = signal.get("contract_id", "?")[:12]
        passed, failures = run_all_checks(signal, bankroll)
        if not passed:
            stats["skipped_risk"] += 1
            logger.info(f"Skip {cid} — risk failures: {'; '.join(failures)}")
            continue

        result = execute_signal(signal, client=client)
        status = result.get("status")
        if status in ("placed", "paper"):
            stats["executed"] += 1
            logger.info(
                f"Executed {cid}: {signal['recommended_side']} ${signal['kelly_size']:.2f} "
                f"@ {result.get('limit_price'):.4f} | status={status}"
            )
        else:
            stats["execution_errors"] += 1
            logger.error(f"Execution failed for {cid}: {result}")

    logger.info(
        f"Trading run done: {stats['executed']} executed, "
        f"{stats['skipped_risk']} skipped by risk, {stats['execution_errors']} errors"
    )
    logger.info("=== TRADING RUN END ===")
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket weather arbitrage bot")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one trading_run() and exit; do not start the scheduler.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Weather arbitrage bot starting up")
    logger.info(f"Paper trade mode:        {PAPER_TRADE}")
    logger.info(f"py_clob_client installed: {_CLOB_AVAILABLE}")
    logger.info(f"Database:                {DB_PATH}")

    # Hard fail-safe: live mode requires py_clob_client
    if not PAPER_TRADE and not _CLOB_AVAILABLE:
        logger.error(
            "PAPER_TRADE=False but py_clob_client is not installed. "
            "Refusing to start. Either pip install py-clob-client or set PAPER_TRADE=True."
        )
        return 2

    init_db()

    if args.dry_run:
        logger.info("--dry-run: single trading_run then exit")
        stats = trading_run()
        logger.info(f"Dry-run stats: {stats}")
        return 0

    # Run once immediately to warm the system
    discovery_run()
    trading_run()

    # Schedule recurring jobs
    scheduler = BlockingScheduler()
    scheduler.add_job(
        trading_run,
        trigger=IntervalTrigger(minutes=30),
        id="trading_run",
        name="Trading scan",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        discovery_run,
        trigger=IntervalTrigger(hours=4),
        id="discovery_run",
        name="Market discovery",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started. Next trading_run in 30 minutes.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl-C")
        return 0
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
