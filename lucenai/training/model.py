
"""
model.py

Model building, compilation, training, and saving utilities for DistilBERT sentiment classification.
Encapsulates the full training lifecycle into modular functions for readability and reuse.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

from pathlib import Path
import tensorflow as tf
from transformers import TFDistilBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.data import Dataset
from lucenai.config.settings import TRAINING_PARAMS, MODEL_PATHS


def build_model(transformer_model_name: str) -> Model:
    """
    Builds a DistilBERT-based classification model using Keras Functional API.

    Args:
        transformer_model_name (str): Name of the HuggingFace DistilBERT model.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    bert_encoder = TFDistilBertModel.from_pretrained(transformer_model_name)

    input_ids = Input(shape=(TRAINING_PARAMS.max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(TRAINING_PARAMS.max_len,), dtype=tf.int32, name="attention_mask")

    bert_output = bert_encoder(input_ids, attention_mask=attention_mask)[0]
    cls_token = bert_output[:, 0, :]  # [CLS] token

    x = Dropout(0.3)(cls_token)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    return model


def compile_model(model: Model) -> Model:
    """
    Compiles the Keras model with binary crossentropy loss and accuracy metrics.

    Args:
        model (tf.keras.Model): The uncompiled model.

    Returns:
        tf.keras.Model: The compiled model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_PARAMS.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def launch_model_training(
    model: Model,
    train_dataset: Dataset,
    val_dataset: Dataset
) -> Model:
    """
    Launches the model training process using the configured datasets and callbacks.

    Args:
        model (Model): The compiled TensorFlow Keras model to be trained.
        train_dataset (tf.data.Dataset): Dataset used for training.
        val_dataset (tf.data.Dataset): Dataset used for validation during training.

    Returns:
        Model: The trained Keras model.
    """
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TRAINING_PARAMS.epochs,
        callbacks=get_callbacks()
    )
    return model

def get_callbacks():
    pass # TODO


def train_distilbert_model(train_dataset, val_dataset, tokenizer) -> None:
    """
    Full training pipeline wrapper to build, compile, train and save the model.

    Args:
        train_dataset (tf.data.Dataset): Tokenized training dataset.
        val_dataset (tf.data.Dataset): Tokenized validation dataset.
        tokenizer (DistilBertTokenizerFast): Tokenizer used for the model.
    """
    print("🧠 Building the model...")
    model = build_model(transformer_model_name=TRAINING_PARAMS.model_name)

    print("⚙️ Compiling the model...")
    model = compile_model(model)

    print("🚀 Starting training...")
    model = launch_model_training(model, train_dataset, val_dataset)

    print("💾 Saving model...")
    save_model(model, MODEL_PATHS.base)

    print("✅ Training complete.")


def save_model(model: Model, path: Path):
    """
    Saves a trained Keras model to the specified path.

    Args:
        model (Model): A compiled and trained TensorFlow/Keras model.
        path (Path): Destination directory or file to save the model.
    """
    MODEL_PATHS.ensure_dirs()
    model.save(str(path))
    print(f"📦 Model saved to {path}")