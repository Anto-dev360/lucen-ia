"""
tokenizer.py

Tokenization utilities for preparing text data for DistilBERT input.
Includes tokenizer loading, encoding, and conversion to TensorFlow Datasets.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

from transformers import DistilBertTokenizerFast
import tensorflow as tf
from lucenai.config.settings import TRAINING_PARAMS


def get_tokenizer_and_dataset(train_texts, train_labels, val_texts, val_labels):
    """
    Loads the DistilBERT tokenizer and prepares tokenized TensorFlow datasets for training and validation.

    Args:
        train_texts (list): List of training text samples.
        train_labels (list): Corresponding labels for the training samples.
        val_texts (list): List of validation text samples.
        val_labels (list): Corresponding labels for the validation samples.

    Returns:
        tuple: (tokenizer, train_dataset, val_dataset)
            - tokenizer: HuggingFace DistilBERT tokenizer
            - train_dataset: tf.data.Dataset for training
            - val_dataset: tf.data.Dataset for validation
    """
    print("📦 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(TRAINING_PARAMS.model_name)

    def tokenize(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=TRAINING_PARAMS.max_len,
            return_tensors='tf'
        )
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))
        return dataset.shuffle(1000).batch(TRAINING_PARAMS.batch_size).prefetch(tf.data.AUTOTUNE)

    print("📄 Tokenizing training and validation sets...")
    train_dataset = tokenize(train_texts, train_labels)
    val_dataset = tokenize(val_texts, val_labels)

    return tokenizer, train_dataset, val_dataset

def encode_single_text(text, tokenizer):
    """
    Tokenizes and encodes a single input tweet for sentiment prediction with DistilBERT.

    Args:
        text (str): The tweet or sentence to analyze.
        tokenizer (DistilBertTokenizerFast): Loaded tokenizer instance.
        max_len (int): Maximum token length for padding/truncation.

    Returns:
        dict: Dictionary of encoded tensors (input_ids, attention_mask)
              formatted for TensorFlow model input.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=TRAINING_PARAMS.max_len,
        return_tensors='tf'
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
