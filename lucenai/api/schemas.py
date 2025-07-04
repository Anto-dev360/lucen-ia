"""
schemas.py

Pydantic schemas for input and output formats used in the API.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """
    Request schema for sentiment prediction.

    Attributes:
        text (str): The text input to analyze.
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Response schema for sentiment prediction.

    Attributes:
        label (str): Predicted label ('positive' or 'negative').
        score (float): Confidence score for the prediction.
    """
    label: str
    score: float