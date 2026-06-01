"""Lexical diversity metrics: Distinct-n and Self-BLEU.

Both are computed over a *set* of outputs (e.g. the 4 outputs produced for one
prompt, or all outputs from one method). No external NLP dependency: Self-BLEU
is implemented directly (modified n-gram precision + brevity penalty + NIST
smoothing) so the metrics run identically on Colab, Kaggle, and locally.

Convention:
- Distinct-1 / Distinct-2 ↑  (higher = more diverse)
- Self-BLEU ↓                (lower  = more diverse; each output is scored
  against the others as references and the scores are averaged)
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Punctuation dropped."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across all texts.

    Returns 0.0 when no n-grams exist (all texts shorter than n).
    """
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = _ngrams(tokenize(text), n)
        total += len(grams)
        unique.update(grams)
    if total == 0:
        return 0.0
    return len(unique) / total


def _modified_precision(
    candidate: list[str], references: list[list[str]], n: int
) -> tuple[int, int]:
    """Clipped n-gram match count and candidate n-gram total (BLEU numerator/denominator)."""
    cand_ngrams = Counter(_ngrams(candidate, n))
    if not cand_ngrams:
        return 0, 0
    max_ref = Counter()
    for ref in references:
        ref_ngrams = Counter(_ngrams(ref, n))
        for gram, count in ref_ngrams.items():
            if count > max_ref[gram]:
                max_ref[gram] = count
    clipped = sum(min(count, max_ref[gram]) for gram, count in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped, total


def _sentence_bleu(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """BLEU of one candidate against reference token lists, NIST geometric smoothing."""
    if not candidate or not references:
        return 0.0

    log_precisions: list[float] = []
    smooth = 1.0
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(candidate, references, n)
        if total == 0:
            # candidate too short for this order; treat as fully smoothed
            smooth *= 2
            p = 1.0 / (smooth * total) if total else 1.0 / (smooth * max(1, len(candidate)))
        elif clipped == 0:
            smooth *= 2
            p = 1.0 / (smooth * total)
        else:
            p = clipped / total
        log_precisions.append(math.log(p))

    geo_mean = math.exp(sum(log_precisions) / max_n)

    cand_len = len(candidate)
    ref_len = min((len(r) for r in references), key=lambda rl: (abs(rl - cand_len), rl))
    if cand_len == 0:
        brevity = 0.0
    elif cand_len > ref_len:
        brevity = 1.0
    else:
        brevity = math.exp(1 - ref_len / cand_len)

    return brevity * geo_mean


def self_bleu(texts: list[str], max_n: int = 4) -> float:
    """Mean Self-BLEU over a set. Each text scored against the others as references.

    Returns 0.0 for fewer than two texts (no peers to compare against).
    """
    if len(texts) < 2:
        return 0.0
    tokenized = [tokenize(t) for t in texts]
    scores: list[float] = []
    for i, candidate in enumerate(tokenized):
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]
        scores.append(_sentence_bleu(candidate, references, max_n=max_n))
    return sum(scores) / len(scores)
