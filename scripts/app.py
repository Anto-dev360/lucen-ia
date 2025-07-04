"""
app.py

FastAPI application to serve a sentiment analysis model (DistilBERT).

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lucenai.api.predict import predict_sentiment
from lucenai.config import settings

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

class PredictionRequest(BaseModel):
    """
    Input schema for prediction request.

    Attributes:
        text (str): The input text to analyze.
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Output schema for prediction response.

    Attributes:
        label (str): Sentiment label ('positive' or 'negative').
        score (float): Confidence score of the prediction.
    """
    label: str
    score: float

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: API status confirmation.
    """
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict the sentiment of a given text.

    Args:
        request (PredictionRequest): Text input.

    Returns:
        PredictionResponse: Predicted label and score.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    return predict_sentiment(request.text)

# Optional entry point for local development
if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)