"""Quality and rarity scoring for DivPO pair selection."""
from __future__ import annotations

import re

import numpy as np

_REPETITION_RE = re.compile(r"(\b\w+\b)( \1){3,}", re.IGNORECASE)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def score_quality(
    prompt: str,
    response: str,
    embedder,
    min_words: int = 10,
    max_words: int = 400,
    include_relevance: bool = True,
    _prompt_emb: "np.ndarray | None" = None,
    _response_emb: "np.ndarray | None" = None,
) -> float:
    """Heuristic quality score in [0, 1].

    v1 (include_relevance=True):  relevance (50%) + coherence (30%) + length (20%).
    v2 (include_relevance=False): coherence (60%) + length (40%).

    Relevance = prompt-response cosine. For counter-argument generation this is
    counterproductive: a strong counter-argument diverges semantically from the
    prompt, so high relevance rewards paraphrase, not opposition. v2 drops it.
    """
    words = response.split()
    n = len(words)
    if n < min_words or n > max_words:
        return 0.0

    repetitions = len(_REPETITION_RE.findall(response))
    coherence = max(0.0, 1.0 - 0.25 * repetitions)

    ideal = 100
    length_score = min(n, ideal) / ideal if n <= ideal else ideal / n

    if not include_relevance:
        return 0.6 * coherence + 0.4 * length_score

    try:
        if _prompt_emb is not None and _response_emb is not None:
            relevance = max(0.0, _cosine(_prompt_emb, _response_emb))
        else:
            embs = embedder.encode([prompt, response], convert_to_numpy=True)
            relevance = max(0.0, _cosine(embs[0], embs[1]))
    except Exception:
        relevance = 0.5

    return 0.5 * relevance + 0.3 * coherence + 0.2 * length_score


def score_rarity(
    response: str,
    others: list[str],
    embedder,
    _response_emb: "np.ndarray | None" = None,
    _other_embs: "list[np.ndarray] | None" = None,
) -> float:
    """Rarity = 1 - mean cosine similarity to peer candidates.

    Falls back to 1.0 if no peers, 0.5 on embedding failure.
    Pass _response_emb/_other_embs to skip the encode call (batch path).
    """
    if not others:
        return 1.0
    try:
        if _response_emb is not None and _other_embs is not None:
            sims = [_cosine(_response_emb, e) for e in _other_embs]
        else:
            embs = embedder.encode([response] + others, convert_to_numpy=True)
            target = embs[0]
            sims = [_cosine(target, embs[i + 1]) for i in range(len(others))]
        return max(0.0, 1.0 - float(np.mean(sims)))
    except Exception:
        return 0.5
