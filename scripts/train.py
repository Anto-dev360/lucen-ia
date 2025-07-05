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
import sys
import argparse
from pathlib import Path

# Reduce TensorFlow log level to warn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import lucenai as a Python package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lucenai.training.preprocess import load_and_preprocess_dataset
from lucenai.training.tokenizer import get_tokenizer_and_dataset
from lucenai.training.model import train_distilbert_model, load_best_model_and_tokenizer
from lucenai.training.test import evaluate_model_on_test_set
from lucenai.training.utils import configure_environment_for_nlp
from lucenai.config.settings import MODEL_PATHS

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including --force flag.
    """
    parser = argparse.ArgumentParser(description="Train DistilBERT for sentiment classification.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force retraining even if a model already exists."
    )
    return parser.parse_args()


def main() -> None:
    """
    Executes the full training and evaluation pipeline for binary sentiment classification using DistilBERT.

    Workflow:
    1. Configure environment (GPU, logging, seeds).
    2. If no model exists or --force is set:
        - Load and preprocess training/validation/test splits.
        - Tokenize and prepare tf.data.Dataset objects.
        - Fine-tune and save the model and tokenizer.
    3. Always:
        - Load the best saved model and tokenizer.
        - Load test data (if not already done).
        - Run final evaluation on the test set.

    Model configuration and paths are defined in `config/settings.py`.
    """
    args = parse_args()

    # Configure display, GPU and seed
    configure_environment_for_nlp()

    # Model already exists and --force is not used
    if MODEL_PATHS.best_model.exists() and not args.force:
        print(f"⚠️ Found existing model at: {MODEL_PATHS.best_model}")
        print("⏩ Skipping training. Use --force to retrain.\n")

        # Only load test set
        (
            _, _,
            _, _,
            raw_test_texts, raw_test_labels
        ) = load_and_preprocess_dataset(return_test=True)

    # Training required
    else:
        (
            raw_train_texts, raw_train_labels,
            raw_val_texts, raw_val_labels,
            raw_test_texts, raw_test_labels
        ) = load_and_preprocess_dataset(return_test=True)

        print("🔍 Sample of training data (first 10 rows):\n")
        for i in range(10):
            print(f"{i+1:>2}. 📄 Text: {raw_train_texts[i]}")
            print(f"    🏷️ Label: {raw_train_labels[i]}\n")

        # Tokenize dataset
        tokenizer, train_dataset, val_dataset = get_tokenizer_and_dataset(
            raw_train_texts, raw_train_labels,
            raw_val_texts, raw_val_labels
        )

        # Build, fine-tune, compile, fit and save model.
        train_distilbert_model(train_dataset, val_dataset, tokenizer)

    # Load model and tokenizer
    best_model, tokenizer = load_best_model_and_tokenizer(
        model_path=MODEL_PATHS.best_model,
        tokenizer_path=MODEL_PATHS.tokenizer
    )

    # Final test evaluation
    evaluate_model_on_test_set(best_model, tokenizer, raw_test_texts, raw_test_labels)


if __name__ == "__main__":
    main()