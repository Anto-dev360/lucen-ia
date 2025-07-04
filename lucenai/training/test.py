"""
test.py

Inference utilities for predicting sentiment on BTC-related tweets using a trained DistilBERT model.
Includes batch prediction from a CSV file for post-training evaluation.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import pandas as pd
import tensorflow as tf
from config.settings import DATA_PATHS, MODEL_PATHS, TRAINING_PARAMS
from transformers import DistilBertTokenizerFast

from lucenai.training.tokenizer import encode_single_text


def load_model_and_tokenizer(model_path=None, tokenizer_name=None):
    """
    Loads the trained TensorFlow model and corresponding tokenizer.

    Args:
        model_path (str): Path to the saved Keras model.
        tokenizer_name (str): Name of the pretrained tokenizer.

    Returns:
        tuple: (model, tokenizer)
    """
    model_path = model_path or MODEL_PATHS.base
    tokenizer_name = tokenizer_name or TRAINING_PARAMS.model_name

    model = tf.keras.models.load_model(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    return model, tokenizer


def predict_on_csv():
    """
    Loads a CSV file containing BTC-related tweets, runs sentiment prediction on each,
    and prints the result for a small sample.

    The path to the CSV file is defined in `config/settings.py` (TEST_PREDICTION_CSV).
    The CSV must contain a 'text' column.

    This function is meant to be called post-training to verify model behavior.
    """
    print(f"📄 Loading test data from {DATA_PATHS.sample_prediction_csv}")
    df = pd.read_csv(DATA_PATHS.sample_prediction_csv)

    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    model, tokenizer = load_model_and_tokenizer()

    print("🔍 Running predictions on sample tweets...\n")
    sample = df["text"].dropna().head(5)

    for text in sample:
        encoded = encode_single_text(text, tokenizer)
        prediction = model.predict(encoded, verbose=0)[0][0]
        label = "Positive" if prediction > 0.5 else "Negative"
        print(f"📝 Tweet: {text[:80]}...\n➡️ Prediction: {label} ({prediction:.2f})\n")
