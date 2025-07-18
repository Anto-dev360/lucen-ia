"""
predict.py

Load and run sentiment predictions using the fine-tuned DistilBERT model.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import logging
from typing import TypedDict, Dict, List

from lucenai.api.utils import load_inference_model
from lucenai.training.preprocess import clean_text
from lucenai.config.settings import LABELS, TRAINING_PARAMS, API_METADATA

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type representing the output of the sentiment prediction
class PredictionResult(TypedDict):
    label: str
    score: float

# Load model and tokenizer once at module level
model, tokenizer = load_inference_model()

def predict_sentiment(text: str) -> PredictionResult:
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
        PredictionResult: A dictionary with:
            - "label": The predicted sentiment ("positive", "negative", or "invalid")
            - "score": Confidence score (float between 0 and 1, rounded to 4 decimals)

    Raises:
        RuntimeError: If the prediction process fails due to model or tokenizer errors.
    """
    try:
        if not text.strip():
            logger.warning("⚠️ Empty input received for prediction.")
            return {"label": LABELS.invalid, "score": 0.0}

        # Apply same text cleaning as during training phase
        cleaned_text = clean_text(text)

        inputs = tokenizer(
            cleaned_text,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=TRAINING_PARAMS.max_len
        )

        logits = model(inputs, training=False)
        score = float(logits.numpy()[0][0])  # Single output due to sigmoid
        label = LABELS.positive if score >= API_METADATA.threshold else LABELS.negative

        logger.debug(f"📝 Input: {text} (cleaned: {cleaned_text})")
        logger.debug(f"📈 Sigmoid score: {score:.4f}")
        logger.debug(f"🏷️ Predicted label: {label}")

        return {"label": label, "score": round(score, 4)}

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")


def aggregate_sentiment(texts: List[str]) -> Dict[str, float]:
    """
    Aggregates sentiment predictions over a batch of tweet texts.

    Each text is analyzed using the `predict_sentiment` function.
    Results are classified as 'positive' or 'negative' and averaged.

    Args:
        texts (List[str]): List of raw tweet texts.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "positive": Proportion of positive tweets (0 to 1)
            - "negative": Proportion of negative tweets (0 to 1)
            - "total": Total number of tweets analyzed
    """
    results = [predict_sentiment(text) for text in texts]
    total = len(results)

    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "total": 0}

    positive = sum(1 for r in results if r.get("label") == "positive") / total
    negative = sum(1 for r in results if r.get("label") == "negative") / total

    return {
        "positive": round(positive, 4),
        "negative": round(negative, 4),
        "total": total
    }
