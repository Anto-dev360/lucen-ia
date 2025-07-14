#!/bin/bash

FRONTEND_DIR="lucenai/frontend"
PORT=8080

echo "📦 Serving DYOR Frontend from ./${FRONTEND_DIR} on port ${PORT}..."

# Check if port is already in use
if lsof -i :$PORT &>/dev/null; then
  echo "❌ Port $PORT is already in use. Kill the process or choose another port."
  exit 1
fi

# Start HTTP server in background
python3 -m http.server $PORT --directory $FRONTEND_DIR &
SERVER_PID=$!
echo "🌐 Local frontend served at http://localhost:$PORT (PID: $SERVER_PID)"

# Start ngrok tunnel
if ! command -v ngrok &>/dev/null; then
  echo "❌ ngrok is not installed. Please install ngrok first."
  kill $SERVER_PID
  exit 1
fi

echo "🚀 Starting ngrok tunnel..."
ngrok http $PORT