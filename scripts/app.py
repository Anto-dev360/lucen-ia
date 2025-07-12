"""
app.py

FastAPI application to serve a sentiment analysis model (DistilBERT).

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import sys
import os
import uvicorn
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
from dotenv import load_dotenv

# Load environment variables (for NGROK_AUTH_TOKEN)
load_dotenv()

# Ensure project modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lucenai.api.predict import predict_sentiment
from lucenai.config import settings

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        result = predict_sentiment(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get Ngrok auth token from environment
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    if not NGROK_AUTH_TOKEN:
        raise RuntimeError("Missing NGROK_AUTH_TOKEN environment variable.")

    # Set ngrok auth token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Run uvicorn in a background thread
    def run_uvicorn():
        uvicorn.run("scripts.app:app", host="0.0.0.0", port=8000)

    threading.Thread(target=run_uvicorn, daemon=True).start()

    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"🔗 Public URL (copy in your browser): {public_url}")

    while True:
        time.sleep(10)
