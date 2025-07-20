#!/bin/bash

####################################################################
# serve_and_tunnel.sh — Anthony Morin
# Compatible local/Docker
####################################################################

command -v curl >/dev/null 2>&1 || { echo "❌ curl not found. Install it."; exit 1; }
command -v ngrok >/dev/null 2>&1 || { echo "❌ ngrok not found. Install it."; exit 1; }

# === Load environment variables from .env ===
set -a
if [[ -f .env ]]; then
  source .env
elif [[ -f /app/.env ]]; then
  source /app/.env
else
  echo "❌ No .env file found in ./ or /app/"
  exit 1
fi
set +a

# === Validate Ngrok token ===
if [[ -z "$NGROK_AUTH_TOKEN" ]]; then
  echo "❌ Missing NGROK_AUTH_TOKEN. Please set it in your .env file."
  exit 1
fi

# === Free port 8000 if already in use ===
if lsof -i :8000 &>/dev/null; then
  echo "⚠️ Port 8000 is already in use. Killing existing process..."
  kill -9 $(lsof -ti :8000)
  sleep 1
fi

# === Start FastAPI backend ===
echo "🚀 Starting FastAPI backend on port 8000..."
uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload &

# === Give the server time to boot ===
sleep 2

# === Start Ngrok tunnel ===
echo "🌐 Démmarage du tunnel Ngrok sur le port 8000..."
ngrok authtoken "$NGROK_AUTH_TOKEN"
ngrok http 8000 > /dev/null &

# Wait and fetch URL
NGROK_API="http://localhost:4040/api/tunnels"
RETRIES=10
PUBLIC_URL=""

for i in $(seq 1 $RETRIES); do
  PUBLIC_URL=$(curl -s $NGROK_API | grep -o 'https://[0-9a-zA-Z.-]*\.ngrok-free\.app' | head -n 1)
  if [[ -n "$PUBLIC_URL" ]]; then
    break
  fi
  echo "⏳ Waiting for Ngrok tunnel to initialize ($i/$RETRIES)..."
  sleep 10
done

if [[ -z "$PUBLIC_URL" ]]; then
  echo "❌ Failed to retrieve Ngrok public URL."
else
  echo "✅ Public URL: $PUBLIC_URL"
fi

wait
