"""
evaluation.py

Evaluation utilities for a fine-tuned DistilBERT model on a binary sentiment
classification task.
This script runs model evaluation on the test split from the original dataset
and saves metrics and visualizations.

Includes:
- Classification report
- ROC AUC, F1, precision, recall, accuracy
- Confusion matrix and ROC curve visualizations
- JSON export of performance metrics

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import PreTrainedTokenizerBase

from lucenai.config.settings import TRAINING_PARAMS


def evaluate_model_on_test_set(
    model: tf.keras.Model,
    tokenizer: PreTrainedTokenizerBase,
    test_texts: List[str],
    test_labels: List[int],
    max_len: int = TRAINING_PARAMS.max_len,
    output_dir: Path = None
) -> None:
    """
    Evaluates a trained model on the test set and saves metrics and plots to disk.

    Args:
        model (tf.keras.Model): Trained model.
        tokenizer (PreTrainedTokenizer): Tokenizer used to preprocess text.
        test_texts (List[str]): Input texts.
        test_labels (List[int]): Corresponding binary labels.
        max_len (int): Maximum sequence length (default: 64).
    """
    print("🧪 Evaluating model on test set...\n")

    # Tokenize
    encodings = tokenizer(
        test_texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )

    input_ids = tf.cast(encodings["input_ids"], dtype=tf.int32)
    attention_mask = tf.cast(encodings["attention_mask"], dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        },
        tf.convert_to_tensor(test_labels, dtype=tf.int32)
    )).batch(32)

    # Predict
    y_probs = model.predict(dataset).squeeze()
    y_preds = (y_probs > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(test_labels, y_preds)
    precision = precision_score(test_labels, y_preds)
    recall = recall_score(test_labels, y_preds)
    f1 = f1_score(test_labels, y_preds)
    auc = roc_auc_score(test_labels, y_probs)
    report = classification_report(test_labels, y_preds, digits=4, output_dict=True)

    print("📊 Test Performance:")
    print(f"   ✅ Accuracy : {accuracy:.4f}")
    print(f"   🎯 Precision: {precision:.4f}")
    print(f"   🔁 Recall   : {recall:.4f}")
    print(f"   🧠 F1 Score : {f1:.4f}")
    print(f"   🧬 ROC AUC  : {auc:.4f}")

    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    with open(output_dir / "test_report.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "classification_report": report
        }, f, indent=4)

    print(f"\n💾 Report saved: {output_dir / 'test_report.json'}")

    # Plot confusion matrix
    cm = confusion_matrix(test_labels, y_preds)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    print(f"🖼️ Confusion matrix saved: {output_dir / 'confusion_matrix.png'}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    print(f"🖼️ ROC curve saved: {output_dir / 'roc_curve.png'}")
