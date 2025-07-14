"""
serve_api.py

FastAPI backend to serve sentiment analysis predictions via HTTP POST.
Also serves static frontend assets (HTML/JS) for local testing or full-stack integration.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lucenai.api.predict import predict_sentiment

# === Initialize FastAPI application ===
app = FastAPI(
    title="LucenAI Sentiment API",
    description="API for analyzing sentiment of tweets using fine-tuned DistilBERT.",
    version="1.0.0"
)

# === Mount static frontend directory (for serving index.html, app.js, etc.) ===
frontend_path = os.path.join(os.path.dirname(__file__), "..", "lucenai", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """
    Serve the main frontend HTML page.

    Returns:
        FileResponse: index.html file served as root route.
    """
    return os.path.join(frontend_path, "index.html")


# === API Endpoint for prediction ===
class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict(input: TextInput) -> dict:
    """
    Predict sentiment for a given text input.

    Args:
        input (TextInput): Object containing a single tweet or text.

    Returns:
        dict: Prediction result with sentiment label or score.
    """
    return predict_sentiment(input.text)


def main() -> None:
    """
    Entry point when run as standalone script.
    Useful for local testing (e.g. `python serve_api.py`).
    """
    import uvicorn
    uvicorn.run("scripts.serve_api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()