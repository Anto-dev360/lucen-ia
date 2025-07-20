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

import time
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from transformers import DistilBertTokenizerFast, PreTrainedTokenizerBase

from lucenai.config.settings import DISTILLATION_PARAMS, MODEL_PATHS, TRAINING_PARAMS
from lucenai.training.evaluation import evaluate_model_on_test_set
from lucenai.training.model import save_model_and_tokenizer


class DistillationModel(tf.keras.Model):
    """
    Custom Keras model implementing knowledge distillation.
    Trains a smaller student model using both the true labels and soft predictions from a teacher.

    Args:
        student (tf.keras.Model): The student model to train.
        teacher (tf.keras.Model): The pretrained teacher model (frozen).
        temperature (float): Temperature parameter for softening the logits.
        alpha (float): Weight to balance student loss and distillation loss.
    """
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = DISTILLATION_PARAMS.temperature,
        alpha: float = DISTILLATION_PARAMS.alpha,
    ):
        super().__init__()
        self.student_model = student
        self.teacher_model = teacher
        self.temperature = temperature
        self.alpha = alpha

        # Loss functions
        self.student_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Training metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        print(f"🧪 Using distillation params: α={self.alpha}, T={self.temperature}")

    def compile(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """
        Compile model with a given optimizer.
        """
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Performs one training step for the student model using knowledge distillation.

        This step combines:
        - A supervised loss between student predictions and true labels (binary crossentropy),
        - A distillation loss (KL divergence) between student and teacher softened outputs.

        The final loss is a weighted sum:
            total_loss = alpha * supervised_loss + (1 - alpha) * distillation_loss

        Returns:
            dict[str, tf.Tensor]: Dictionary of tracked metrics including total loss,
            student loss, distillation loss, and accuracy.
        """
        x, y = data

        # Forward pass of the teacher (frozen)
        teacher_logits = self.teacher_model(x, training=False)
        teacher_probs_raw = tf.nn.sigmoid(teacher_logits)

        # Convert to 2-class distribution: [p_neg, p_pos]
        teacher_probs = tf.concat([1.0 - teacher_probs_raw, teacher_probs_raw], axis=1)

        with tf.GradientTape() as tape:
            # Forward pass of the student (trainable)
            student_logits = self.student_model(x, training=True)
            student_probs_raw = tf.nn.sigmoid(student_logits)
            student_soft_probs = tf.concat([1.0 - student_probs_raw, student_probs_raw], axis=1)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_logits)

            kl = tf.keras.losses.KLDivergence()
            distillation_loss = kl(
                tf.nn.softmax(teacher_probs / self.temperature),
                tf.nn.softmax(student_soft_probs / self.temperature)
            ) * (self.temperature ** 2)

            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Backpropagation
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy_tracker.update_state(y, student_probs_raw)

        return {
            "loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Custom evaluation step for student model.
        """
        x, y = data
        student_logits = self.student_model(x, training=False)
        student_probs = tf.nn.sigmoid(student_logits)
        student_loss = self.student_loss_fn(y, student_logits)

        self.accuracy_tracker.update_state(y, student_probs)

        return {
            "loss": student_loss,
            "accuracy": self.accuracy_tracker.result(),
        }

    def reset_metrics(self):
        """
        Resets all tracked metrics at the end of each epoch.
        """
        for metric in self.metrics:
            metric.reset_states()

    @property
    def metrics(self):
        """
        Metrics tracked during training and evaluation.
        Required for proper reset at each epoch.
        """
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.accuracy_tracker,
        ]

    def build(self, input_shape: tuple[int, ...]) -> None:
        """
        Build the model with a given input shape (e.g., for summary()).
        """
        self.student_model.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the student model.
        """
        return self.student_model(inputs)


def create_student_model(vocab_size: int = DISTILLATION_PARAMS.vocab_size,
                         embedding_dim: int = DISTILLATION_PARAMS.embedding_dim,
                         dropout_rate: float = DISTILLATION_PARAMS.dropout_rate,
                         max_len: int = TRAINING_PARAMS.max_len) -> tf.keras.Model:
    """
    Builds a lightweight student model using an embedding layer followed by average pooling
    and a small classification head. This architecture is used for knowledge distillation
    to approximate the behavior of a larger teacher model while keeping inference efficient.
    """
    
    # Define input layers for token IDs and attention mask
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Embedding layer: maps token IDs to dense vectors
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="embedding"
    )(input_ids)

    # Global average pooling to reduce sequence dimension by averaging embeddings
    x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)

    # First dense layer with ReLU activation
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    # First dropout for regularization
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    # Second dense layer with ReLU activation
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    # Second dropout for regularization
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)

    # Output layer with sigmoid activation for binary classification
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

    # Define the model with named inputs and output
    model = tf.keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=outputs,
        name="StudentModel"
    )

    # Display model summary for verification
    print("📋 Distilled model detailed architecture")
    print("═" * 80)
    model.summary()
    print("═" * 80)

    return model


def train_evaluate_student_model(
    teacher_model,
    train_dataset,
    val_dataset,
    tokenizer: PreTrainedTokenizerBase,
    test_texts: List[str],
    test_labels: List[int]
    ):
    """
    Trains a lightweight student model using knowledge distillation
    from a pre-trained teacher.

    Combines soft predictions from the teacher and ground-truth labels
    using a DistillationModel wrapper.

    Perform evaluation after training.

    Args:
        teacher_model (tf.keras.Model): The trained teacher model
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

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=DISTILLATION_PARAMS.learning_rate
    )
    distill_model.compile(optimizer=optimizer)

    # Set training callback
    log_predictions = LambdaCallback (
        on_epoch_end=lambda epoch, logs: print(
            f"[Epoch {epoch+1}] 🔍 Sample student preds (sigmoid): "
            f"{distill_model.student_model.predict(val_dataset.take(1), verbose=0).flatten()[:10]}"
        )
    )       
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        log_predictions
    ]

    print("🚀 Starting student training with distillation...")
    # Build the model by calling it once to initialize weights (required before .fit() or .save())
    for x_batch, _ in train_dataset.take(1):
        _ = distill_model(x_batch)
    start_time = time.time()
    distill_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=DISTILLATION_PARAMS.epochs,
        callbacks = callbacks
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Student model training completed ({training_time/60:.1f} minutes).")

    MODEL_PATHS.ensure_dirs()

    evaluate_model_on_test_set(
        distill_model.student_model,
        tokenizer,
        test_texts,
        test_labels,
        output_dir=MODEL_PATHS.student_root
    )

    save_model_and_tokenizer(
        distill_model.student_model,
        tokenizer,
        save_path=MODEL_PATHS.student_root
    )

    return distill_model.student_model


def load_student_model_and_tokenizer(
    model_path: Path,
    tokenizer_path: Path
) -> Tuple[tf.keras.Model, DistilBertTokenizerFast]:
    """
    Charge un modèle DistilBERT allégé (student) et son tokenizer associé.

    Cette fonction :
    - Vérifie l'existence des chemins
    - Recharge le tokenizer HuggingFace
    - Reconstruit l'architecture du modèle student
    - Charge les poids entraînés (TensorFlow checkpoint)

    Args:
        model_path (Path): Préfixe du chemin vers les poids (excluant .index/.data).
        tokenizer_path (Path): Dossier du tokenizer HuggingFace.

    Returns:
        Tuple contenant le modèle Keras et le tokenizer HuggingFace.
    
    Raises:
        FileNotFoundError: si les chemins n'existent pas.
    """
    # Ajustement : on attend le préfixe de poids
    weight_index = model_path.with_suffix(".index")
    weight_data = model_path.with_name(model_path.name + ".data-00000-of-00001")

    if not weight_index.exists() or not weight_data.exists():
        raise FileNotFoundError(f"❌ Weights not found at: {model_path} (index or data file missing)")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"❌ Tokenizer path not found: {tokenizer_path}")

    print(f"📥 Loading tokenizer from: {tokenizer_path}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    print("✅ Tokenizer loaded successfully.")

    print("🧠 Rebuilding student model architecture...")
    model = create_student_model()

    print(f"📦 Loading model weights from: {model_path}")
    status = model.load_weights(str(model_path))
    status.expect_partial()
    print("✅ Weights loaded successfully.")

    return model, tokenizer