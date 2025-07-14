"""
schemas.py

Pydantic request/response models used by LucenAI's API.

Author: Anthony Morin
Created: 2025-07-14
"""

from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    """
    Request schema for a single tweet or text input.
    """
    text: str


class TweetItem(BaseModel):
    """
    Represents a single tweet in a batch analysis.
    """
    text: str


class TweetBatch(BaseModel):
    """
    Request schema for a batch of tweets.
    """
    tweets: List[TweetItem]