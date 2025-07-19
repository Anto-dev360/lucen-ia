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

import random
import re
from typing import Tuple, Union

import nltk
import pandas as pd
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

from lucenai.config.settings import DATA_PATHS, TRAINING_PARAMS

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
        df = pd.read_csv(DATA_PATHS.raw_dataset, encoding="utf-8")
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
    df = augment_dataset(df, fraction=0.2)
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

    return train_texts, train_labels, val_texts, val_labels, None, None


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
    Clean tweet text by removing noise and preserving key sentiment indicators.

    Steps:
    - Remove URLs, mentions, hashtags, and HTML artifacts.
    - Convert percentage values (e.g., -4.5%) to text form (e.g., -4.5 percent), preserving signs.
    - Remove standalone "rt" and unwanted special characters.
    - Keep emojis and numeric signs (+/-) only when relevant.
    - Normalize whitespace and convert to lowercase.

    Args:
        text (str): Raw tweet text.

    Returns:
        str: Cleaned, lowercase text with emojis and numeric sentiment preserved.
    """
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)         # Remove URLs, mentions, hashtags
    text = re.sub(r"([-+]?\d+(?:\.\d+)?)%", r"\1 percent", text)    # Convert % values, keep sign
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)         # Remove standalone "rt"
    text = text.replace("amp", "")                                  # Remove HTML artifact "amp"

    # Remove special chars except letters, numbers, emojis, +/-
    text = re.sub(r"[^\w\s\.\-\+"
                  "\U0001F600-\U0001F64F"                           # Emojis
                  "\U0001F300-\U0001F5FF"                           # Symbols & pictographs
                  "\U0001F680-\U0001F6FF"                           # Transport & map
                  "\U0001F1E0-\U0001F1FF"                           # Flags
                  "]", "", text)

    text = re.sub(r"(?<!\d)[+-](?!\d)", "", text)                   # Remove + / - not in numbers
    text = re.sub(r"\s+", " ", text)                                # Normalize whitespace
    return text.lower().strip()                                     # Lowercase and trim


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
    Formats the DataFrame: drops unnecessary columns, renames columns,
    resets index, and reorders columns.

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
        print("✅ Class distribution after split:")
        print(f"Train:{train_df['label'].value_counts(normalize=True)}")
        print(f"Val:{val_df['label'].value_counts(normalize=True)}")
        print(f"Test:{test_df['label'].value_counts(normalize=True)}")
        return train_df, val_df, test_df
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            stratify=df["label"],
            random_state=TRAINING_PARAMS.seed
        )
        print("✅ Class distribution after split:")
        print(f"Train:{train_df['label'].value_counts(normalize=True)}")
        print(f"Val:{val_df['label'].value_counts(normalize=True)}")
        return train_df, val_df


nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word: str) -> list:
    """
    Returns a list of synonyms for a given word using WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


def augment_text(text: str, num_changes: int = 2) -> str:
    """
    Randomly replaces words with synonyms to augment the input text.

    Args:
        text (str): Original cleaned text.
        num_changes (int): Number of words to attempt replacing.

    Returns:
        str: Augmented text.
    """
    words = text.split()
    if len(words) < 3:
        return text  # Avoid changing very short texts

    indices = list(range(len(words)))
    random.shuffle(indices)

    changes = 0
    for idx in indices:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            words[idx] = random.choice(synonyms)
            changes += 1
        if changes >= num_changes:
            break

    return " ".join(words)


def augment_dataset(df: pd.DataFrame, fraction: float = 0.2) -> pd.DataFrame:
    """
    Augments the dataset by generating paraphrased versions of a subset of rows.

    Args:
        df (pd.DataFrame): Input DataFrame after cleaning but before balancing.
        fraction (float): Fraction of samples to augment (e.g., 0.2 = 20%).

    Returns:
        pd.DataFrame: Augmented DataFrame with additional rows.
    """
    n_samples = int(len(df) * fraction)
    sampled_df = df.sample(n=n_samples, random_state=TRAINING_PARAMS.seed)

    augmented_rows = []
    for _, row in sampled_df.iterrows():
        augmented_text = augment_text(row["clean_text"])
        augmented_rows.append({
            "tweet": row["tweet"],
            "clean_text": augmented_text,
            "label": row["label"]
        })

    augmented_df = pd.DataFrame(augmented_rows)
    print(f"🧪 Data augmentation: added {len(augmented_df)} synthetic samples.")
    return pd.concat([df, augmented_df], ignore_index=True)
