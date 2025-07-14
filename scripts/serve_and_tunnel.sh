#!/bin/bash

####################################################################
# serve_and_tunnel.sh
#
# Description:
# Launches the FastAPI backend server (on port 8000) and creates
# a public tunnel using Ngrok for external access.
#
# - Loads environment variables from .env
# - Automatically kills any process already using port 8000
# - Starts the backend server in background
# - Creates the Ngrok tunnel on port 8000
#
# Author: Anthony Morin
# Created: 2025-07-14
####################################################################

# === Load environment variables ===
set -a
source .env
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
echo "🌐 Starting Ngrok tunnel on port 8000..."
ngrok config add-authtoken "$NGROK_AUTH_TOKEN"
ngrok http 8000