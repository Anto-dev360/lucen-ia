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
from typing import List, Tuple

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from transformers import PreTrainedTokenizerFast, TFAutoModel

try:
    from tensorflow_addons.metrics import F1Score
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False

try:
    from tensorflow_addons.metrics import F1Score
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False

from lucenai.config.settings import CALLBACK_CONFIG, MODEL_PATHS, TRAINING_PARAMS


def create_sentiment_model(distilbert_model, dropout_rate=TRAINING_PARAMS.dropout_rate):
    """
    Builds a sentiment classification model based on DistilBERT.

    Architecture:
    1. Pretrained DistilBERT base
    2. Extract [CLS] token representation
    3. Dropout for regularization
    4. Dense projection layer (optional but helpful)
    5. Dense layer for binary classification

    Args:
        distilbert_model: Pretrained DistilBERT model
        dropout_rate: Dropout rate (default: from TRAINING_PARAMS)

    Returns:
        model: Full Keras model
        base_model: The DistilBERT model (for optional fine-tuning access)
    """
    print("🏗️ Building full architecture...")

    # Input layers
    input_ids = tf.keras.layers.Input(
        shape=(TRAINING_PARAMS.max_len,),
        dtype=tf.int32,
        name='input_ids'
    )
    attention_mask = tf.keras.layers.Input(
        shape=(TRAINING_PARAMS.max_len,),
        dtype=tf.int32,
        name='attention_mask'
    )
    print("   ✅ Input layers created")

    # DistilBERT
    distilbert_output = distilbert_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    sequence_output = distilbert_output.last_hidden_state
    print("   ✅ DistilBERT integrated")

    # Pooling ([CLS] token)
    cls_token = sequence_output[:, 0, :]
    print("   ✅ [CLS] token extracted")

    # Dropout layer
    dropout_output = tf.keras.layers.Dropout(dropout_rate, name='dropout')(cls_token)
    print(f"   ✅ Dropout applied (rate={dropout_rate})")

    # Dense projection layer (adds non-linearity and helps learning)
    dense_projection = tf.keras.layers.Dense(
        256, activation='relu', name='projection'
    )(dropout_output)
    print("   ✅ Dense projection layer added (256 units, ReLU)")

    # Classification layer
    predictions = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        name='classifier'
    )(dense_projection)
    print("   ✅ Classification layer added")

    # Final model
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=predictions,
        name='DistilBERT_Sentiment_Classifier'
    )
    print("\n✅ Model successfully built!")

    return model, distilbert_model


def build_model(transformer_model_name: str) -> tf.keras.Model:
    """
    Builds and compiles a DistilBERT-based binary classification model
    using the Keras Functional API.

    Args:
        transformer_model_name (str): HuggingFace model identifier.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    print("🧠 Loading pretrained DistilBERT model...")
    distilbert_model = TFAutoModel.from_pretrained(
        transformer_model_name,
        return_dict=True
    )

    model, base_model = create_sentiment_model(distilbert_model)

    total_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
    print("\n📊 Model Summary:")
    print(f"   🔢 Total trainable parameters: {total_params:,}")
    print("   🎯 Task: Binary sentiment classification")
    print(f"   📏 Input length: {TRAINING_PARAMS.max_len} tokens")
    print(f"   🎲 Dropout rate: {TRAINING_PARAMS.dropout_rate}")

    # Affichage du résumé du modèle
    print("📋 Model detailed architecture")
    print("═" * 80)
    model.summary()
    print("═" * 80)

    return model


def define_hyperparameters(use_extended_metrics: bool = False) -> Tuple[
    tf.keras.optimizers.Optimizer,
    tf.keras.losses.Loss,
    List[tf.keras.metrics.Metric]
]:
    """
    Defines the optimizer, loss function, and evaluation metrics for training.

    Args:
        use_extended_metrics (bool): Whether to include AUC and F1Score
        (requires tensorflow-addons).

    Returns:
        Tuple: (optimizer, loss_function, metrics list)
    """
    print("⚙️ Configuring training hyperparameters...\n")

    # Optimizer for fine-tuning
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=TRAINING_PARAMS.learning_rate,
        epsilon=1e-8,
        clipnorm=1.0
    )

    # Binary classification loss
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Standard metrics
    metrics: List[tf.keras.metrics.Metric] = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    if use_extended_metrics:
        print("➕ Extended metrics enabled:")
        metrics.append(tf.keras.metrics.AUC(name='auc'))

        if TFA_AVAILABLE:
            metrics.append(F1Score(num_classes=1, threshold=0.5, average='micro', name='f1_score'))
        else:
            print("⚠️ tensorflow-addons is not installed; skipping F1Score.")

    # Display all metrics being tracked
    print("📊 Tracking metrics:")
    for metric in metrics:
        print(f"   • {metric.name}")

    return optimizer, loss_fn, metrics


def compile_model(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    metrics: List[tf.keras.metrics.Metric]
) -> tf.keras.Model:
    """
    Compiles a Keras model for binary sentiment classification.

    Args:
        model (tf.keras.Model): The uncompiled Keras model.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer (e.g., Adam).
        loss_fn (tf.keras.losses.Loss): Loss function (e.g., BinaryCrossentropy).
        metrics (List[tf.keras.metrics.Metric]): List of metrics to track during training.

    Returns:
        tf.keras.Model: The compiled model, ready for training.
    """
    print("🔧 Compiling model...\n")

    # Validate metrics type
    assert all(isinstance(m, tf.keras.metrics.Metric) for m in metrics), \
        "All metrics must be instances of tf.keras.metrics.Metric"

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )

    print("✅ Model compiled successfully!")

    # Print final config
    print("\n📋 Final configuration:")
    print("   🎯 Task: Binary sentiment classification")
    print("   🧠 Architecture: DistilBERT + classification head")
    print(f"   📊 Trainable parameters: {model.count_params():,}")
    print(f"   ⚡ Optimizer: {optimizer.__class__.__name__}")
    print(f"   📉 Loss function: {loss_fn.__class__.__name__}")
    print(f"   📈 Metrics: {[m.name for m in metrics]}")

    print("\n🚀 Ready to train!")
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


def get_callbacks() -> list:
    """
    Returns a list of Keras callbacks used during training:
    - EarlyStopping: Stops training when validation loss stops improving
    - ReduceLROnPlateau: Reduces LR if validation loss plateaus
    - ModelCheckpoint: Saves best model weights during training
    """
    print("📦 Initializing training callbacks...")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CALLBACK_CONFIG.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=CALLBACK_CONFIG.lr_reduce_factor,
            patience=CALLBACK_CONFIG.lr_reduce_patience,
            min_lr=CALLBACK_CONFIG.min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATHS.base / "checkpoint" / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    return callbacks


def save_model_and_tokenizer(model: Model, tokenizer, save_path: Path = MODEL_PATHS.base) -> None:
    """
    Saves the trained model and tokenizer to disk.

    Args:
        model (tf.keras.Model): Trained Keras model.
        tokenizer: Hugging Face tokenizer used during training.
        save_path (Path): Directory to save model and tokenizer (default: MODEL_PATHS.base)
    """
    print(f"💾 Saving model to: {save_path}")
    MODEL_PATHS.ensure_dirs()
    
    # Save model
    model.save(save_path)
    print("✅ Model saved")

    # Save tokenizer
    tokenizer.save_pretrained(MODEL_PATHS.tokenizer)
    print(f"✅ Tokenizer saved to: {MODEL_PATHS.tokenizer}")


def train_distilbert_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast
) -> None:
    """
    Full training pipeline wrapper to build, compile, train and save the model.

    Args:
        train_dataset (tf.data.Dataset): Tokenized training dataset.
        val_dataset (tf.data.Dataset): Tokenized validation dataset.
        tokenizer (DistilBertTokenizerFast): Tokenizer used for the model.
    """
    # Build the model.
    model = build_model(transformer_model_name=TRAINING_PARAMS.model_name)

    # Configure training.
    optimizer, loss_fn, metrics = define_hyperparameters(use_extended_metrics=True)

    # Compile the model.
    model = compile_model(model, optimizer, loss_fn, metrics)

    # Start training.
    model = launch_model_training(model, train_dataset, val_dataset)

    # Save model and tokenizer for prediction.
    save_model_and_tokenizer(model, tokenizer)
