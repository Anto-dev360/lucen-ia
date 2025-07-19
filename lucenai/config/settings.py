"""
settings.py

Centralized configuration for the lucen_ai project.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

# === GENERAL ===

PROJECT_NAME = "lucen_ai"
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# === DATA PATHS ===

@dataclass
class DataPaths:
    data_dir: Path = BASE_DIR / "data"
    raw_dataset: Path = data_dir / "BTC_Tweets_Sentiments.csv"
    train: Path = data_dir / "train.csv"
    val: Path = data_dir / "val.csv"
    test: Path = data_dir / "test.csv"
    sample_prediction_csv: Path = data_dir / "test" / "sample_btc_tweets.csv"

DATA_PATHS = DataPaths()


# === MODEL PATHS ===

@dataclass
class ModelPaths:
    base: Path = BASE_DIR / "lucenai" / "models" / "distilbert_sentiment"

    # Best fine-tuned model
    best_root: Path = base
    best_weights: Path = base / "checkpoint" / "best_model"
    best_tokenizer: Path = base / "tokenizer"

    # Student (distilled)
    student_root: Path = base / "student_model"
    student_weights: Path = student_root / "checkpoint" / "weights"
    student_tokenizer: Path = student_root / "tokenizer"

    def ensure_dirs(self):
        """Create all required directory trees."""
        self.base.mkdir(parents=True, exist_ok=True)
        self.student_root.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = ModelPaths()


# === LOGGING PATHS ===

@dataclass
class LoggingPaths:
    tensorboard_log_dir: Path = MODEL_PATHS.base / "logs" / "training"
    csv_log_file: Path = tensorboard_log_dir / "training_log.csv"

    def ensure_dirs(self):
        """Create all required directory trees."""
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    def clean_logs(self):
        """Remove old logs (CSV and TensorBoard) before training."""
        if self.tensorboard_log_dir.exists():
            print("🧹 Cleaning up previous logs...")
            shutil.rmtree(self.tensorboard_log_dir)
        self.ensure_dirs()

LOGGING_PATHS = LoggingPaths()


# === TRAINING PARAMETERS ===

@dataclass(frozen=True)
class TrainingParams:
    model_name: str = "distilbert-base-uncased"
    batch_size: int = 32
    epochs: int = 10
    max_len: int = 64
    learning_rate: float = 2e-5
    seed: int = 42
    dropout_rate: float = 0.4

TRAINING_PARAMS = TrainingParams()


# === DISTILLATION PARAMETERS ===

@dataclass(frozen=True)
class DistillationParams:
    temperature: float = 1.0    # Temperature for soft target smoothing.
    alpha: float = 0.3          # Weight between distillation and ground truth loss.
    learning_rate: float = 1e-4 # Optimizer learning rate.
    vocab_size: int = 30522     # Vocabulary size (e.g., from tokenizer).
    embedding_dim: int = 64     # Dimension of token embeddings.
    dropout_rate: float = 0.3   # Dropout rate.
    epochs: int = 5

DISTILATION_PARAMS = DistillationParams()


# === CALLBACK SETTINGS ===

@dataclass(frozen=True)
class CallbackConfig:
    early_stopping_patience: int = 2
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6

CALLBACK_CONFIG = CallbackConfig()


# === SENTIMENT LABELS ===

@dataclass(frozen=True)
class SentimentLabels:
    positive: str = "positive"
    negative: str = "negative"
    invalid: str = "invalid"

LABELS = SentimentLabels()


# === API METADATA ===

@dataclass(frozen=True)
class APIMetadata:
    title: str = "API d'analyse de sentiment"
    version: str = "1.0"
    threshold: float = 0.27
    description: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "description",
            f"Prédit le sentiment d’un tweet à l’aide de DistilBERT.\n"
            f"Seuil de classification utilisé lors de l'inférence : {self.threshold}"
        )

API_METADATA = APIMetadata()