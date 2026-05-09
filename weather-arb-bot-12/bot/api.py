# api.py
"""
FastAPI control panel for the weather arb bot.

Endpoints:
  GET  /api/status        - bot run state, last scan time, summary numbers
  GET  /api/settings      - current config values
  POST /api/settings      - update settings (writes to .env, requires restart)
  GET  /api/positions     - open positions with marked-to-market P&L
  GET  /api/positions/closed - closed positions with realized P&L
  GET  /api/signals       - recent signals (last 100)
  GET  /api/pnl_curve     - cumulative P&L over time for chart
  POST /api/bot/start     - start the bot scheduler thread
  POST /api/bot/stop      - stop the bot scheduler thread
  POST /api/bot/scan_now  - run one trading_run() immediately
  POST /api/resolver/run  - manually run the resolver
  GET  /                  - serve the static HTML UI
"""

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from db import get_conn, get_open_positions, get_recent_signals, init_db, get_daily_pnl
from edge import run_edge_scan
from execution import execute_signal, get_clob_client
from risk import run_all_checks
from resolver import resolver_pass
from sizing import get_bankroll

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
)
