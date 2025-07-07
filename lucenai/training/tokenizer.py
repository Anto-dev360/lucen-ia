"""
tokenizer.py

Tokenization utilities for preparing text data for DistilBERT input.
Includes tokenizer loading, encoding, and conversion to TensorFlow Datasets.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

from typing import List, Tuple, Dict

import tensorflow as tf
from transformers import DistilBertTokenizerFast, BatchEncoding

from lucenai.config.settings import TRAINING_PARAMS


def get_tokenizer_and_dataset(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int]
) -> Tuple[DistilBertTokenizerFast, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the DistilBERT tokenizer and prepares tokenized TensorFlow
    datasets for training and validation.

    Args:
        train_texts (List[str]): List of training text samples.
        train_labels (List[int]): Corresponding labels for the training samples.
        val_texts (List[str]): List of validation text samples.
        val_labels (List[int]): Corresponding labels for the validation samples.

    Returns:
        Tuple:
            - tokenizer: HuggingFace DistilBERT tokenizer
            - train_dataset: tf.data.Dataset for training
            - val_dataset: tf.data.Dataset for validation
    """
    print("📦 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(TRAINING_PARAMS.model_name)

    def tokenize(texts: List[str], labels: List[int]) -> tf.data.Dataset:
        # Tokenize and convert to NumPy arrays for TF Dataset compatibility
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=TRAINING_PARAMS.max_len,
            return_tensors="np"
        )

        dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"]
            },
            labels
        ))

        return dataset.shuffle(1000).batch(TRAINING_PARAMS.batch_size).prefetch(tf.data.AUTOTUNE)

    print("📄 Tokenizing training and validation sets...")
    train_dataset = tokenize(train_texts, train_labels)
    val_dataset = tokenize(val_texts, val_labels)

    return tokenizer, train_dataset, val_dataset


def encode_single_text(text: str, tokenizer: DistilBertTokenizerFast) -> Dict[str, tf.Tensor]:
    """
    Tokenizes and encodes a single input tweet for DistilBERT sentiment classification.

    Args:
        text (str): The tweet or sentence to analyze.
        tokenizer (DistilBertTokenizerFast): Loaded HuggingFace tokenizer instance.

    Returns:
        Dict[str, tf.Tensor]: Dictionary containing 'input_ids' and 'attention_mask'
                              formatted for TensorFlow model input.
    """
    encoded: BatchEncoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=TRAINING_PARAMS.max_len,
        return_tensors='tf'
    )

    return {
        "input_ids": tf.convert_to_tensor(encoded["input_ids"]),
        "attention_mask": tf.convert_to_tensor(encoded["attention_mask"])
    }
