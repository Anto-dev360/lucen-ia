"""
train.py

Train a DistilBERT model for binary sentiment classification using TensorFlow and Keras.
Includes data loading, tokenization, model compilation, callbacks, and training loop.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import os
# Reduce TensorFlow log level to warn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
# import lucenai as a Python package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lucenai.training.preprocess import load_and_preprocess_dataset
from lucenai.training.tokenizer import get_tokenizer_and_dataset
# ruff: noqa
# from lucenai.training.model import train_distilbert_model
# ruff: noqa
# from lucenai.training.test import predict_on_csv
from lucenai.training.utils import configure_environment_for_nlp
from lucenai.config.settings import TRAINING_PARAMS


def main():
    """
    Runs the full training pipeline for a DistilBERT model on a binary sentiment classification task.

    Steps:
    1. Load and preprocess the training and validation data
    2. Tokenize texts using a HuggingFace tokenizer
    3. Build and compile the model using TensorFlow/Keras
    4. Train the model with configured callbacks
    5. Save the trained model to disk

    Uses parameters defined in `config/settings.py`.
    """
    # Configure display, GPU and seed
    configure_environment_for_nlp()

    # Clean and split the dataset
    raw_train_texts, raw_train_labels, raw_val_texts, raw_val_labels, raw_test_texts, raw_test_labels = load_and_preprocess_dataset(True)
    print("🔍 Sample of training data (first 10 rows):\n")
    for i in range(10):
        print(f"{i+1:>2}. 📄 Text: {raw_train_texts[i]}")
        print(f"    🏷️ Label: {raw_train_labels[i]}\n")

    # Tokenize dataset
    tokenizer, train_dataset, val_dataset = get_tokenizer_and_dataset(
        raw_train_texts, raw_train_labels,
        raw_val_texts, raw_val_labels
    )

    # Build, fine-tune, compile, fit and save model
    # train_distilbert_model(train_dataset, val_dataset, tokenizer)

    # Test model
    # predict_on_csv()

if __name__ == "__main__":
    main()