""" 
distillation.py

Knowledge Distillation wrapper to compress large teacher models (e.g., fine-tuned DistilBERT)
into smaller student networks. Implements custom Keras training logic combining ground truth
and soft-label supervision.

Author: Anthony Morin
Created: 2025-07-07
Project: lucen_ai
License: MIT
""" 

from pathlib import Path
from typing import List
import time

import tensorflow as tf
from tensorflow.keras.losses import KLDivergence, BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from transformers import PreTrainedTokenizerBase

from lucenai.config.settings import TRAINING_PARAMS, DISTILATION_PARAMS
from lucenai.training.evaluation import evaluate_model_on_test_set
from lucenai.training.model import save_model_and_tokenizer

class DistillationModel(tf.keras.Model):
    """ 
    Custom Keras model implementing knowledge distillation.
    Trains a smaller student model using both the true labels and the teacher's soft predictions.

    Args:
        student (tf.keras.Model): The smaller model to be trained.
        teacher (tf.keras.Model): The pre-trained teacher model.
    """
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = DISTILATION_PARAMS.temperature
        self.alpha = DISTILATION_PARAMS.alpha

        # Losses and metrics
        self.student_loss_fn = BinaryCrossentropy(from_logits=False)
        self.distillation_loss_fn = KLDivergence()
        self.train_acc = BinaryAccuracy()
        self.train_precision = Precision()
        self.train_recall = Recall()

    def compile(self, optimizer):
        """Compile model with a single optimizer."""
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        """Custom training step with combined student and distillation loss."""
        x, y_true = data

        # Forward pass for teacher (no gradient)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            # Ground truth loss
            student_loss = self.student_loss_fn(y_true, student_preds)

            # Distillation loss (soft targets)
            student_soft = tf.nn.sigmoid(student_preds / self.temperature)
            teacher_soft = tf.nn.sigmoid(teacher_preds / self.temperature)
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)

            # Total loss (weighted sum)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        # Backpropagation
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Metrics update
        self.train_acc.update_state(y_true, student_preds)
        self.train_precision.update_state(y_true, student_preds)
        self.train_recall.update_state(y_true, student_preds)

        return {
            "loss": total_loss,
            "accuracy": self.train_acc.result(),
            "precision": self.train_precision.result(),
            "recall": self.train_recall.result()
        }

    def test_step(self, data):
        """Custom evaluation logic."""
        x, y_true = data
        student_preds = self.student(x, training=False)
        loss = self.student_loss_fn(y_true, student_preds)

        self.train_acc.update_state(y_true, student_preds)
        self.train_precision.update_state(y_true, student_preds)
        self.train_recall.update_state(y_true, student_preds)

        return {
            "loss": loss,
            "accuracy": self.train_acc.result(),
            "precision": self.train_precision.result(),
            "recall": self.train_recall.result()
        }

    def reset_metrics(self):
        """
        Resets all training metrics.
        Useful between epochs to avoid metric accumulation.
        """
        self.train_acc.reset_states()
        self.train_precision.reset_states()
        self.train_recall.reset_states()

    @property
    def metrics(self):
        """
        Returns the list of metrics for tracking during training.
        Required by Keras to manage metric resets between epochs.
        """
        return [self.train_acc, self.train_precision, self.train_recall]

    def build(self, input_shape):
        """
        Builds the underlying student model so that summary() can be used.
        """
        self.student.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass through the student model.
        """
        return self.student(inputs)


def create_student_model(max_len=TRAINING_PARAMS.max_len):
    """
    Builds a lightweight student model using an embedding + average pooling architecture.

    This avoids using heavy transformers while preserving the ability to generalize from token sequences.

    Args:
        max_len (int): Maximum sequence length.

    Returns:
        tf.keras.Model: Lightweight student model.
    """
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")  # Ignored by student, mandatory for teacher


    # Embedding layer instead of transformer encoder
    x = tf.keras.layers.Embedding(input_dim=DISTILATION_PARAMS.vocab_size, output_dim=DISTILATION_PARAMS.embedding_dim, input_length=max_len)(input_ids)

    # Mean pooling over time dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(DISTILATION_PARAMS.dropout_rate)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(DISTILATION_PARAMS.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs={"input_ids": input_ids}, outputs=outputs, name="StudentModel")
    return model


def train_evaluate_student_model(
    teacher_model,
    train_dataset,
    val_dataset,
    tokenizer: PreTrainedTokenizerBase,
    test_texts: List[str],
    test_labels: List[int],
    export_dir: Path,
    ):
    """
    Trains a lightweight student model using knowledge distillation from a pre-trained teacher.

    Combines soft predictions from the teacher and ground-truth labels using a DistillationModel wrapper.

    Perform evaluation after training.

    Args:
        teacher_model (tf.keras.Model): The trained teacher model (e.g., DistilBERT fine-tuned).
        train_dataset (tf.data.Dataset): Training dataset (tokenized).
        val_dataset (tf.data.Dataset): Validation dataset (tokenized).
        test_texts (List[str]): Input texts.
        test_labels (List[int]): Corresponding binary labels.
        export_dir (Path): Path to export the student model and artifacts.
        tokenizer (PreTrainedTokenizerFast): Tokenizer used for training.

    Returns:
        tf.keras.Model: Trained student model.
    """
    print("🎓 Creating student model...")
    student = create_student_model()

    print("🧠 Wrapping in DistillationModel...")
    distill_model = DistillationModel(student=student, teacher=teacher_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=DISTILATION_PARAMS.learning_rate)
    distill_model.compile(optimizer=optimizer)

    print("🚀 Starting student training with distillation...")
    # Build the model by calling it once to initialize weights (required before .fit() or .save())
    for x_batch, _ in train_dataset.take(1):
        _ = distill_model(x_batch)
    start_time = time.time()
    distill_model.fit(train_dataset, validation_data=val_dataset, epochs=DISTILATION_PARAMS.epochs)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Student model training completed ({training_time/60:.1f} minutes).")

    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

        print("🧪 Evaluating model on test set...")
        evaluate_model_on_test_set(distill_model.student, tokenizer, test_texts, test_labels, output_dir=export_dir)

        save_model_and_tokenizer(distill_model.student, tokenizer, export_dir)

    return distill_model.student
