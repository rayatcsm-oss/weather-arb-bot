# Weather Arb Bot

## How to start

**Mac:** Double-click `START_BOT.command`
**Windows:** Double-click `START_BOT.bat`

The first time takes ~30 seconds to install dependencies. After that it starts in ~3 seconds.

Your browser will open to http://localhost:8000 automatically.

## What you see

- Click **⚡ Scan now** to find trades (takes ~30 seconds)
- Click **▶ Start bot** to run automatically every 30 minutes
- Click **Close** on any position row to manually exit a trade
- Click **↻ Refresh prices** to update mark-to-market P&L

## Your API keys are already configured

The `.env` file has your NOAA and Tomorrow.io keys pre-filled.
Paper trading is ON by default — no real money is at risk.

## If the launcher doesn't work

Open Terminal (Mac) or Command Prompt (Windows) and run:
```
cd path/to/weather-arb-bot
pip install -r requirements.txt
cd bot
python3 api.py
```
Then open http://localhost:8000
