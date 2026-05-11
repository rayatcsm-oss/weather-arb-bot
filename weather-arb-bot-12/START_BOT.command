#!/bin/bash
# Weather Arb Bot - Mac Launcher
# Double-click this file in Finder to start the bot.

BOT_DIR="$(dirname "$0")/bot"
cd "$BOT_DIR"

echo "========================================"
echo "  Weather Arb Bot Starting..."
echo "========================================"
echo ""

# Kill any existing instance on port 8000 so we never get EADDRINUSE
EXISTING_PID=$(lsof -ti tcp:8000 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
    echo "Stopping existing bot process (PID=$EXISTING_PID)..."
    kill "$EXISTING_PID" 2>/dev/null
    sleep 2
fi

# Install/upgrade dependencies quietly
pip3 install -q -r ../requirements.txt --break-system-packages 2>/dev/null || true

# Run fix_and_start.py which closes any stuck positions then starts the API
python3 fix_and_start.py &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# Wait for API to be ready (up to 30 seconds)
echo "Waiting for API to start..."
for i in $(seq 1 30); do
    sleep 1
    if curl -s http://localhost:8000/api/status > /dev/null 2>&1; then
        echo "Bot is ready!"
        break
    fi
done

# Open browser
open http://localhost:8000

echo ""
echo "Bot running at http://localhost:8000"
echo "Press Ctrl+C to stop"
wait $BOT_PID
