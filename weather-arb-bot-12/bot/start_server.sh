#!/bin/bash
# Starts the Weather Arb Bot API server and keeps it running.
# Usage: ./start_server.sh [--restart-on-crash]
#
# Logs go to logs/server.log (rotates at 10 MB, keeps 3).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

mkdir -p "$LOG_DIR"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[start_server] Stopping existing process PID=$OLD_PID"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Rotate log if > 10 MB
if [ -f "$LOG_FILE" ] && [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt $((10*1024*1024)) ]; then
    mv "$LOG_FILE" "${LOG_FILE}.1"
fi

cd "$SCRIPT_DIR"
echo "[start_server] Starting at $(date -u)" >> "$LOG_FILE"

if [ "${1:-}" = "--restart-on-crash" ]; then
    # Watchdog loop: restart automatically if the process exits
    while true; do
        python3 -u api.py >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "[start_server] Started PID=$(cat $PID_FILE) with watchdog" >> "$LOG_FILE"
        wait "$(cat $PID_FILE)" || true
        echo "[start_server] Process exited at $(date -u) — restarting in 5s" >> "$LOG_FILE"
        sleep 5
    done
else
    python3 -u api.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "[start_server] Started PID=$(cat $PID_FILE)" >> "$LOG_FILE"
    echo "Weather Arb Bot started (PID=$(cat $PID_FILE)). Logs: $LOG_FILE"
fi
