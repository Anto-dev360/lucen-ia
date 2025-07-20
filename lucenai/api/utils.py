"""
utils.py

Helper functions for the LucenAI FastAPI interface.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from lucenai.config.settings import MODEL_PATHS
from lucenai.training.model import load_best_model_and_tokenizer


class ProbabilityCalibrator:
    """
    Calibrates model probability outputs using isotonic regression.

    This class is typically used post-training to improve the reliability
    of predicted probabilities by aligning them with true likelihoods
    observed on a validation set.
    """

    def __init__(self):
        """
        Initializes the calibrator object. The internal model is set to None
        until `fit` is called with validation data.
        """
        self.calibrator: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fits the isotonic regression model on predicted probabilities and true labels.

        Args:
            probs (np.ndarray): Raw predicted probabilities (shape: [n_samples]).
            true_labels (np.ndarray): Ground-truth binary labels (shape: [n_samples]).
        """
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probs, true_labels)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrates new probabilities using the fitted model.

        Args:
            probs (np.ndarray): Raw predicted probabilities to calibrate.

        Returns:
            np.ndarray: Calibrated probabilities.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        return self.calibrator.transform(probs)


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


def load_and_fit_calibrator(calibration_path: Path = MODEL_PATHS.base / "calibration_data.csv") -> ProbabilityCalibrator:
    """
    Loads validation predictions from a CSV file and fits a probability calibrator.

    This function checks if calibration data exists at the specified path.
    If it does, it fits a ProbabilityCalibrator using saved probabilities and labels.
    If not, it logs a warning and returns an untrained calibrator that uses raw probabilities.

    Args:
        calibration_path (Path): Path to the saved CSV file containing 'prob' and 'label' columns.

    Returns:
        ProbabilityCalibrator: A fitted or default calibrator.
    """
    calibrator = ProbabilityCalibrator()

    if calibration_path.exists():
        val_data = pd.read_csv(calibration_path)
        probs = val_data["prob"].to_numpy()
        labels = val_data["label"].to_numpy()
        calibrator.fit(probs, labels)
        logger.info("✅ Calibrator fitted from validation predictions.")
    else:
        logger.warning("⚠️ Calibration file not found. Using raw probabilities.")

    return calibrator
