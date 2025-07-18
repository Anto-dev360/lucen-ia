"""
utils.py

Helper functions for the LucenAI FastAPI interface.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

import logging

from lucenai.config.settings import MODEL_PATHS
from lucenai.training.model import load_best_model_and_tokenizer

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_inference_model():
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