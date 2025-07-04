"""
preprocess.py

Preprocessing utilities for fine-tuning a Transformer model on BTC-related
tweet sentiment classification. Includes dataset loading, text cleaning,
class balancing, and train/val/test splitting.

Author: Anthony Morin
Created: 2025-07-01
Project: lucen_ai
License: MIT
"""

import re
import pandas as pd
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from lucenai.config.settings import TRAINING_PARAMS, DATA_PATHS

VALID_LABELS = {"positive": 1, "negative": 0}

def load_and_preprocess_dataset(return_test: bool = False):
    """
    Loads and preprocesses the raw BTC tweet dataset.

    Args:
        return_test (bool): If True, also return the test set.

    Returns:
        tuple: (train_texts, train_labels, val_texts, val_labels [, test_texts, test_labels])
    """
    print("📥 Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATHS.raw_dataset)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Dataset file not found at path: {DATA_PATHS.raw_dataset}")
    except pd.errors.ParserError:
        raise ValueError(f"❌ Failed to parse CSV file: {DATA_PATHS.raw_dataset}")
    except Exception as e:
        raise RuntimeError(f"❌ Unexpected error while loading dataset: {e}")

    df = clean_and_encode_labels(df)
    df = remove_duplicates(df)
    df = remove_empty_texts(df)
    df = format_dataframe(df)
    df = balance_classes(df)

    split_result = split_dataset(
        df, val_size=0.2, test_size=0.1 if return_test else 0.0
    )

    train_df, val_df = split_result[:2]
    train_texts = train_df["clean_text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["clean_text"].tolist()
    val_labels = val_df["label"].tolist()

    if return_test:
        test_df = split_result[2]
        test_texts = test_df["clean_text"].tolist()
        test_labels = test_df["label"].tolist()
        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    return train_texts, train_labels, val_texts, val_labels


def clean_and_encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans sentiment labels and tweet texts, maps to binary classes.

    Args:
        df (pd.DataFrame): Raw input DataFrame

    Returns:
        pd.DataFrame: Cleaned and encoded DataFrame
    """
    df['Sentiment'] = df['Sentiment'].str.replace(r"[\[\]']", "", regex=True).str.strip()
    df = df[df['Sentiment'].isin(VALID_LABELS.keys())].reset_index(drop=True)
    df['label'] = df['Sentiment'].map(VALID_LABELS)
    df["clean_text"] = df["Tweet"].apply(clean_text)
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove URLs, mentions, hashtags, special characters and retweet prefix.

    Args:
        text (str): Input tweet.

    Returns:
        str: Cleaned lowercase text.
    """
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)               # Remove non-alphanumerics
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)  # Remove "rt" as standalone word
    text = text.replace("amp", "")                           # Remove HTML artifact
    text = re.sub(r"\s+", " ", text)                         # Normalize whitespace
    return text.lower().strip()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate tweets based on cleaned text.

    Args:
        df (pd.DataFrame): Input DataFrame with 'clean_text' column.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate entries.")
    return df


def remove_empty_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where 'clean_text' is empty or NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with 'clean_text' column.

    Returns:
        pd.DataFrame: Cleaned DataFrame with empty texts removed.
    """
    before = len(df)
    df = df[df["clean_text"].notna() & (df["clean_text"] != "")]
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"🧹 Removed {before - after} empty entries.")
    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the dataset by undersampling the majority class.

    Args:
        df (pd.DataFrame): Input DataFrame with 'label' column

    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    min_count = df['label'].value_counts().min()
    df_balanced = pd.concat([
        df[df['label'] == 0].sample(min_count, random_state=TRAINING_PARAMS.seed),
        df[df['label'] == 1].sample(min_count, random_state=TRAINING_PARAMS.seed)
    ]).sample(frac=1, random_state=TRAINING_PARAMS.seed).reset_index(drop=True)
    return df_balanced


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the DataFrame: drops unnecessary columns, renames columns, resets index, and reorders columns.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    # Drop 'id' column if it exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Rename 'Tweet' to 'tweet' for consistency
    if 'Tweet' in df.columns:
        df.rename(columns={'Tweet': 'tweet'}, inplace=True)

    # Reset index and add new 'id' column starting from 1
    df = df.reset_index(drop=True)
    df.insert(0, 'id', range(1, len(df) + 1))

    # Reorder columns
    return df[['id', 'tweet', 'clean_text', 'label']]


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.0
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """
    Splits the dataset into train, validation, and optionally test sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        val_size (float): Proportion of data to allocate for validation.
        test_size (float): Proportion of data to allocate for testing.

    Returns:
        tuple: (train_df, val_df) or (train_df, val_df, test_df)
    """
    if test_size > 0.0:
        temp_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=TRAINING_PARAMS.seed
        )
        train_df, val_df = train_test_split(
            temp_df,
            test_size=val_size / (1 - test_size),
            stratify=temp_df['label'],
            random_state=TRAINING_PARAMS.seed
        )
        return train_df, val_df, test_df
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            stratify=df["label"],
            random_state=TRAINING_PARAMS.seed
        )
        return train_df, val_df
