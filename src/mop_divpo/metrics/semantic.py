"""SBERT semantic-similarity metrics.

Embedding (heavy, needs sentence-transformers + torch) is isolated in `embed`.
The aggregation functions take plain numpy arrays so they are unit-testable
without loading a model.

Convention: SBERT pairwise cosine ↓ (lower = more semantically diverse).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str = DEFAULT_SBERT_MODEL) -> np.ndarray:
    """Encode texts into an (N, D) float array. Model is cached across calls."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_model(model_name)
    return np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return matrix / norms


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full (N, N) cosine-similarity matrix."""
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    normed = _l2_normalize(embeddings)
    return normed @ normed.T


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine over unique off-diagonal pairs within one set.

    This is the headline within-method diversity number (lower = more diverse).
    Returns 0.0 for fewer than two embeddings.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    matrix = pairwise_cosine_matrix(embeddings)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(matrix[iu]))


def cross_group_mean_cosine(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """Mean cosine between every a-vector and every b-vector (no within-group pairs).

    Used for the inter-persona distinctness matrix off-diagonals.
    Returns 0.0 if either group is empty.
    """
    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return 0.0
    a = _l2_normalize(embeddings_a)
    b = _l2_normalize(embeddings_b)
    return float(np.mean(a @ b.T))
