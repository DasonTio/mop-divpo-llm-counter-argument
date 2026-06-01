"""Automated diversity and semantic-similarity metrics for evaluation."""
from __future__ import annotations

from .diversity import distinct_n, self_bleu
from .semantic import (
    cross_group_mean_cosine,
    embed,
    mean_pairwise_cosine,
    pairwise_cosine_matrix,
)

__all__ = [
    "distinct_n",
    "self_bleu",
    "embed",
    "pairwise_cosine_matrix",
    "mean_pairwise_cosine",
    "cross_group_mean_cosine",
]
