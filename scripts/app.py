"""
app.py

FastAPI server for LucenAI: exposes a sentiment prediction API
and serves a static frontend interface.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

import json
import logging
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Reduce TensorFlow log level to warn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Local imports (after sys.path modification if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lucenai.api.predict import aggregate_sentiment, predict_sentiment
from lucenai.api.schemas import TextInput
from lucenai.config.settings import API_METADATA

# === Initialize FastAPI ===
app = FastAPI(
    title=API_METADATA.title,
    description=API_METADATA.description,
    version=API_METADATA.version
)

# === Enable CORS for frontend access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Mount static frontend assets ===
frontend_path = os.path.join(os.path.dirname(__file__), "..", "lucenai", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serves the site's favicon if present.
    """
    favicon_path = os.path.join(frontend_path, "favicon.ico")
    if not os.path.exists(favicon_path):
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(favicon_path)


@app.get("/", response_class=FileResponse)
async def serve_index():
    """
    Serves the main frontend HTML page.
    """
    return os.path.join(frontend_path, "index.html")


@app.post("/predict")
async def predict(input: TextInput):
    """
    Predicts sentiment for a single tweet or free-form text.

    Args:
        input (TextInput): Request body containing 'text' string.

    Returns:
        dict: Prediction result with 'label' and 'score' fields.
    """
    return predict_sentiment(input.text)


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyzes sentiment from a batch of tweets stored in a JSON file.

    The JSON file must contain a list of tweet-like dictionaries with a 'text' field.
    Example: [{"text": "This is bullish"}, {"text": "Not looking good for BTC"}]

    Returns:
        dict: Dictionary with keys 'positive', 'negative', and 'total'.
    """
    try:
        content = await file.read()
        tweets = json.loads(content)

        if not isinstance(tweets, list):
            raise HTTPException(status_code=400,
             detail="Uploaded file must contain a list of tweet objects.")

        texts = [t.get("text", "") for t in tweets if isinstance(t, dict) and "text" in t]

        if not texts:
            raise HTTPException(status_code=422,
             detail="No valid 'text' fields found in uploaded file.")

        # return aggregate_sentiment(texts)
        # +Debug:
                # Liste des prédictions individuelles
        detailed = [predict_sentiment(text) for text in texts]
        for i, pred in enumerate(detailed):
            pred["text"] = texts[i]  # ajoute le texte pour affichage/debug

        aggregate = aggregate_sentiment(texts)

        return {
            **aggregate,
            "details": detailed  # 👈 inclus ici
        }
        # -Debug


    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze file: {str(e)}")
