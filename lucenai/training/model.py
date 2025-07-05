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
import json
from typing import List, Tuple, Optional
import time

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
from tensorflow.keras.models import Model, load_model
from transformers import PreTrainedTokenizerFast, TFAutoModel, DistilBertTokenizerFast, TFDistilBertModel

try:
    from tensorflow_addons.metrics import F1Score
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False

from lucenai.config.settings import CALLBACK_CONFIG, MODEL_PATHS, TRAINING_PARAMS


def create_sentiment_model(
    distilbert_model: TFDistilBertModel,
    dropout_rate: float = TRAINING_PARAMS.dropout_rate
) -> Tuple[tf.keras.Model, TFDistilBertModel]:
    """
    Builds a Keras sentiment classification model using a pretrained DistilBERT backbone.

    Architecture:
    1. Input layers for token IDs and attention masks
    2. DistilBERT base
    3. [CLS] token extraction
    4. Dropout + Dense + Sigmoid for binary classification

    Args:
        distilbert_model (TFDistilBertModel): A pretrained DistilBERT model.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        Tuple:
            - model (tf.keras.Model): Fully assembled Keras model.
            - distilbert_model (TFDistilBertModel): The base transformer model.
    """
    print("🏗️ Building full architecture...")

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

    distilbert_output = distilbert_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    sequence_output = distilbert_output.last_hidden_state
    print("   ✅ DistilBERT integrated")

    cls_token = sequence_output[:, 0, :]
    print("   ✅ [CLS] token extracted")

    dropout_output = tf.keras.layers.Dropout(dropout_rate, name='dropout')(cls_token)
    print(f"   ✅ Dropout applied (rate={dropout_rate})")

    dense_projection = tf.keras.layers.Dense(
        256, activation='relu', name='projection'
    )(dropout_output)
    print("   ✅ Dense projection layer added")

    predictions = tf.keras.layers.Dense(
        1, activation='sigmoid', name='classifier'
    )(dense_projection)
    print("   ✅ Classification layer added")

    model = tf.keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=predictions,
        name="DistilBERT_Sentiment_Classifier"
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
    print(f"   📈 Metrics: {[m.name for m in metrics]}\n")

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
            filepath=str(MODEL_PATHS.base / "checkpoint" / "best_model"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    return callbacks


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
    print("🚀 Start model training")
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TRAINING_PARAMS.epochs,
        callbacks=get_callbacks()
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished! ({training_time/60:.1f} minutes)")

    return model, history


def training_summary(history: History, export_path: Optional[str] = None) -> None:
    """
    Displays a summary of training performance including final accuracy/loss,
    overfitting analysis, best epoch evaluation, and optionally exports metrics.

    Args:
        history (History): Keras History object returned by model.fit().
        export_path (Optional[str]): Path to export summary as JSON (optional).
    """
    print("📊 Training Summary\n")

    history_data = history.history
    required_keys = {'accuracy', 'val_accuracy', 'loss', 'val_loss'}
    
    if not required_keys.issubset(history_data.keys()):
        raise ValueError(f"❌ Missing required keys in history: {required_keys - set(history_data.keys())}")

    # Final metrics
    final_train_accuracy = history_data['accuracy'][-1]
    final_val_accuracy = history_data['val_accuracy'][-1]
    final_train_loss = history_data['loss'][-1]
    final_val_loss = history_data['val_loss'][-1]

    print(f"🎯 Final performance:")
    print(f"   📈 Training accuracy: {final_train_accuracy:.4f} ({final_train_accuracy * 100:.2f}%)")
    print(f"   ✅ Validation accuracy: {final_val_accuracy:.4f} ({final_val_accuracy * 100:.2f}%)")
    print(f"   📉 Training loss: {final_train_loss:.4f}")
    print(f"   📊 Validation loss: {final_val_loss:.4f}")

    # Overfitting analysis
    overfitting_gap = final_train_accuracy - final_val_accuracy
    print(f"\n🔍 Overfitting analysis:")
    if overfitting_gap < 0.05:
        print(f"   ✅ Well balanced (gap: {overfitting_gap:.4f})")
    elif overfitting_gap < 0.10:
        print(f"   ⚠️ Slight overfitting (gap: {overfitting_gap:.4f})")
    else:
        print(f"   🚨 Overfitting detected (gap: {overfitting_gap:.4f})")

    # Best epoch by validation accuracy
    best_epoch = history_data['val_accuracy'].index(max(history_data['val_accuracy'])) + 1
    best_val_acc = max(history_data['val_accuracy'])

    print(f"\n🏆 Best epoch:")
    print(f"   📊 Epoch: {best_epoch}/{len(history_data['val_accuracy'])}")
    print(f"   🎯 Validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

    # Performance interpretation
    if best_val_acc > 0.90:
        print("   🌟 Excellent performance!")
    elif best_val_acc > 0.85:
        print("   👍 Very good performance!")
    elif best_val_acc > 0.80:
        print("   ✅ Good performance")
    else:
        print("   ⚠️ Needs improvement")

    # Additional metrics
    extra_metrics = {k: v[-1] for k, v in history_data.items() if k not in required_keys}
    if extra_metrics:
        print("\n📈 Additional metrics:")
        for name, value in extra_metrics.items():
            print(f"   • {name}: {value:.4f}")

    # Optional export
    if export_path:
        export_data = {
            "final": {
                "train_accuracy": final_train_accuracy,
                "val_accuracy": final_val_accuracy,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "overfitting_gap": overfitting_gap,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc
            },
            "additional_metrics": extra_metrics
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n📁 Metrics exported to: {export_path}")


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
    model, history = launch_model_training(model, train_dataset, val_dataset)

    # Print training statistics
    training_summary(history)

    # Save model and tokenizer for prediction.
    save_model_and_tokenizer(model, tokenizer)


def load_best_model_and_tokenizer(model_path: Path, tokenizer_path: Path) -> Tuple[Model, DistilBertTokenizerFast]:
    """
    Loads a fine-tuned Keras model (SavedModel format) and its associated tokenizer.

    Args:
        model_path (Path): Path to the saved TensorFlow model directory (e.g., containing saved_model.pb).
        tokenizer_path (Path): Path to the tokenizer directory (containing vocab.txt, tokenizer_config.json, etc.)

    Returns:
        Tuple[Model, DistilBertTokenizerFast]: The loaded model and tokenizer, ready for inference or evaluation.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"❌ Tokenizer path not found: {tokenizer_path}")

    print(f"📥 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully.")

    print(f"📥 Loading tokenizer from: {tokenizer_path}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    print("✅ Tokenizer loaded successfully.")

    return model, tokenizer
