"""
settings.py

Centralized configuration for the lucen_ai project.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path

# === 📁 GENERAL ===

PROJECT_NAME = "lucen_ai"
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# === 📊 DATA PATHS ===

@dataclass
class DataPaths:
    data_dir: Path = BASE_DIR / "data"
    raw_dataset: Path = data_dir / "BTC_Tweets_Sentiments.csv"
    train: Path = data_dir / "train.csv"
    val: Path = data_dir / "val.csv"
    test: Path = data_dir / "test.csv"
    sample_prediction_csv: Path = data_dir / "test" / "sample_btc_tweets.csv"

DATA_PATHS = DataPaths()


# === 🧠 MODEL PATHS ===

@dataclass
class ModelPaths:
    base: Path = BASE_DIR / "lucenai" / "models" / "distilbert_sentiment"
    best_model: Path = base / "checkpoint" / "best_model" 
    tokenizer: Path = base / "tokenizer"

    def ensure_dirs(self):
        self.base.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = ModelPaths()


# === ⚙️ TRAINING PARAMETERS ===

@dataclass
class TrainingParams:
    model_name: str = "distilbert-base-uncased"
    batch_size: int = 32
    epochs: int = 10
    max_len: int = 64
    learning_rate: float = 2e-5
    seed: int = 42
    dropout_rate: float = 0.4

TRAINING_PARAMS = TrainingParams()


# === ⚙️ DISTILLATION PARAMETERS ===

@dataclass
class DistillationParams:
    temperature: float = 2.0    # Temperature for soft target smoothing.
    alpha: float = 0.7          # Weight between distillation and ground truth loss.
    learning_rate: float = 1e-4 # Optimizer learning rate.
    vocab_size: int = 30522     # Vocabulary size (e.g., from tokenizer).
    embedding_dim: int = 64     # Dimension of token embeddings.
    dropout_rate: float = 0.3   # Dropout rate.
    epochs: int = 5

DISTILATION_PARAMS = DistillationParams()


# === 🧪 CALLBACK SETTINGS ===

@dataclass
class CallbackConfig:
    early_stopping_patience: int = 2
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6

CALLBACK_CONFIG = CallbackConfig()


# === 📝 LOGGING ===

CSV_LOG_FILE = BASE_DIR / "training_log.csv"
TENSORBOARD_LOG_DIR = BASE_DIR / "logs"


# === 🌐 API METADATA ===

API_TITLE = "Sentiment Analysis API"
API_VERSION = "1.0"
API_DESCRIPTION = "Predict sentiment from a tweet using DistilBERT"