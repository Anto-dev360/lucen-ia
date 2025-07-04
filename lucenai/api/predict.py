"""
predict.py

Load and run sentiment predictions using the fine-tuned DistilBERT model.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
from src.config import TOKENIZER_PATH, MODEL_WEIGHTS_PATH

# Load model and tokenizer at startup
model = tf.keras.models.load_model(MODEL_WEIGHTS_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def predict_sentiment(text: str) -> dict:
    """
    Predict the sentiment of a given text using the loaded DistilBERT model.

    Args:
        text (str): Input text.

    Returns:
        dict: Dictionary containing the predicted label and confidence score.
    """
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=256
    )
    logits = model(inputs)[0].numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    label = "positive" if np.argmax(probs) == 1 else "negative"
    score = float(np.max(probs))
    return {"label": label, "score": score}