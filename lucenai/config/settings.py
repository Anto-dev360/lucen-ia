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

# Project base configuration
PROJECT_NAME = "lucen_ai"
BASE_DIR = Path(__file__).resolve().parent.parent.parent    # Root directory of the project


@dataclass
class DataPaths:
    """
    Paths to all relevant datasets used in the project, including raw and split files.
    """
    data_dir: Path = BASE_DIR / "data"
    raw_dataset: Path = data_dir / "BTC_Tweets_Sentiments.csv"
    train: Path = data_dir / "train.csv"
    val: Path = data_dir / "val.csv"
    test: Path = data_dir / "test.csv"
    sample_prediction_csv: Path = data_dir / "test" / "sample_btc_tweets.csv"

DATA_PATHS = DataPaths()


@dataclass
class ModelPaths:
    """
    Directory structure for storing trained models:
    - Teacher model (fine-tuned)
    - Student model (distilled)
    """
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


@dataclass
class LoggingPaths:
    """
    Paths for logging training progress:
    - TensorBoard logs
    - CSV training logs
    """
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


@dataclass(frozen=True)
class TrainingParams:
    """
    Core training hyperparameters for the teacher model, including optimizer settings,
    dropout, and sequence length.
    """
    model_name: str = "distilbert-base-uncased"
    batch_size: int = 32
    epochs: int = 10
    max_len: int = 64
    learning_rate: float = 2e-5
    seed: int = 42
    dropout_rate: float = 0.4

TRAINING_PARAMS = TrainingParams()


@dataclass(frozen=True)
class DistillationParams:
    """
    Specific parameters used for knowledge distillation between teacher and student models.
    """
    temperature: float = 1.0
    alpha: float = 0.3
    learning_rate: float = 1e-4
    vocab_size: int = 30522
    embedding_dim: int = 64
    dropout_rate: float = 0.3
    epochs: int = 5

DISTILLATION_PARAMS = DistillationParams()


@dataclass(frozen=True)
class CallbackConfig:
    """
    Settings for model training callbacks such as early stopping and learning rate reduction.
    """
    early_stopping_patience: int = 2
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6

CALLBACK_CONFIG = CallbackConfig()


@dataclass(frozen=True)
class SentimentLabels:
    """
    Canonical labels used throughout the sentiment analysis pipeline.
    """
    positive: str = "positive"
    negative: str = "negative"
    invalid: str = "invalid"

LABELS = SentimentLabels()


@dataclass(frozen=True)
class APIMetadata:
    """
    Metadata configuration for the sentiment analysis API (FastAPI).
    Includes title, version, description in french and custom threshold
    """
    title: str = "API d'analyse de sentiment"
    version: str = "1.0"
    threshold: float = 0.55
    description: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "description",
            f"Prédit le sentiment d’un tweet à l’aide de DistilBERT.\n"
            f"Seuil de classification utilisé lors de l'inférence : {self.threshold}"
        )

API_METADATA = APIMetadata()