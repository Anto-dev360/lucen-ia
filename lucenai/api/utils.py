"""
utils.py

Helper functions for the LucenAI FastAPI interface.

Includes:
- Sentiment aggregation logic for batch tweet analysis.

Author: Anthony Morin
Created: 2025-07-14
Project: lucen_ai
License: MIT
"""

from typing import List, Dict
from lucenai.api.predict import predict_sentiment


def aggregate_sentiment(texts: List[str]) -> Dict[str, float]:
    """
    Aggregates sentiment predictions over a batch of tweet texts.

    Each text is analyzed using the `predict_sentiment` function.
    Results are classified as 'positive' or 'negative' and averaged.

    Args:
        texts (List[str]): List of raw tweet texts.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "positive": Proportion of positive tweets (0 to 1)
            - "negative": Proportion of negative tweets (0 to 1)
            - "total": Total number of tweets analyzed
    """
    results = [predict_sentiment(text) for text in texts]
    total = len(results)

    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "total": 0}

    positive = sum(1 for r in results if r.get("label") == "positive") / total
    negative = sum(1 for r in results if r.get("label") == "negative") / total

    return {
        "positive": round(positive, 4),
        "negative": round(negative, 4),
        "total": total
    }