#!/bin/bash
# Weather Arb Bot - Mac Launcher
cd "$(dirname "$0")/bot"

echo "========================================"
echo "  Weather Arb Bot Starting..."
echo "========================================"
echo ""

# Install/upgrade dependencies quietly
pip3 install -q -r ../requirements.txt --break-system-packages 2>/dev/null || true

# Run fix_and_start.py which closes any stuck positions then starts the API
python3 fix_and_start.py &
BOT_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in $(seq 1 20); do
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
