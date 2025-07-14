"""
predict.py

Load and run sentiment predictions using the fine-tuned DistilBERT model.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import logging
from typing import Dict

import numpy as np
import tensorflow as tf

from lucenai.config.settings import MODEL_PATHS, TRAINING_PARAMS, LABELS
from lucenai.training.model import load_best_model_and_tokenizer

# === Configure logger ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer_safely():
    """
    Loads the fine-tuned DistilBERT model and tokenizer from configured paths.

    Returns:
        Tuple[tf.keras.Model, transformers.PreTrainedTokenizer]:
            - The fine-tuned Keras model.
            - The associated tokenizer.

    Raises:
        RuntimeError: If loading the model or tokenizer fails.
    """
    try:
        logger.info(f"📥 Loading model from: {MODEL_PATHS.best_weights}")
        logger.info(f"📥 Loading tokenizer from: {MODEL_PATHS.best_tokenizer}")

        model, tokenizer = load_best_model_and_tokenizer(
            model_path=MODEL_PATHS.best_weights,
            tokenizer_path=MODEL_PATHS.best_tokenizer
        )

        logger.info("✅ Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"❌ Failed to load model/tokenizer: {e}")
        raise RuntimeError(f"Could not load model/tokenizer: {e}")


# === Load model and tokenizer once at module level ===
model, tokenizer = load_model_and_tokenizer_safely()


def predict_sentiment(text: str) -> Dict[str, float]:
    """
    Predict the sentiment of a given input text using the fine-tuned DistilBERT model.

    This function performs the following steps:
        1. Validates the input text.
        2. Tokenizes the input using a pretrained tokenizer.
        3. Feeds the tokenized input into a DistilBERT-based model.
        4. Applies a sigmoid to obtain a probability score.
        5. Maps the score to a sentiment label.

    Sentiment labels:
        - "positive": Indicates a generally favorable sentiment.
        - "negative": Indicates a generally unfavorable sentiment.
        - "invalid": Returned when input is empty or only whitespace.

    Args:
        text (str): The raw input text to classify.

    Returns:
        Dict[str, float]: A dictionary with:
            - "label": The predicted sentiment label ("positive", "negative", or "invalid").
            - "score": The model's confidence score for the predicted label, rounded to 4 decimal places.

    Raises:
        RuntimeError: If the prediction process fails due to model or tokenizer errors.
    """
    try:
        if not text.strip():
            logger.warning("⚠️ Empty input received for prediction.")
            return {"label": LABELS.INVALID, "score": 0.0}

        inputs = tokenizer(
            text,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=TRAINING_PARAMS.max_len
        )

        logits = model(inputs, training=False)
        score = float(logits.numpy()[0][0])  # Single output due to sigmoid
        label = LABELS.POSITIVE if score >= 0.5 else LABELS.NEGATIVE

        logger.debug(f"📝 Input: {text}")
        logger.debug(f"📈 Sigmoid score: {score:.4f}")
        logger.debug(f"🏷️ Predicted label: {label}")

        return {"label": label, "score": round(score, 4)}

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")
