#!/bin/bash
# deploy/setup.sh — Run once on a fresh Ubuntu 22.04 VPS
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install Python 3.11 + git
apt-get install -y python3.11 python3.11-venv python3-pip git

# Install Node.js 18 (needed for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Create non-root bot user
if ! id botuser &>/dev/null; then
  useradd -m -s /bin/bash botuser
  usermod -aG sudo botuser
fi

mkdir -p /home/botuser/weather-arb-bot
chown botuser:botuser /home/botuser/weather-arb-bot

cat <<EOF

VPS setup complete. Next steps:
  1. Copy your project files to /home/botuser/weather-arb-bot/
  2. Copy your .env file (NEVER commit this)
  3. cd /home/botuser/weather-arb-bot
  4. python3.11 -m venv venv
  5. source venv/bin/activate && pip install -r requirements.txt
  6. sudo cp deploy/weather_bot.service /etc/systemd/system/
  7. sudo systemctl enable weather_bot && sudo systemctl start weather_bot
  8. Tail logs:  sudo journalctl -u weather_bot -f

EOF
