"""DivPO preference pair construction from scored candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .scoring import score_quality, score_rarity


@dataclass
class ScoredCandidate:
    text: str
    quality: float
    rarity: float
    combined: float


@dataclass
class DivPORecord:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


def select_pair(
    prompt: str,
    candidates: list[str],
    embedder,
    min_quality: float = 0.35,
    quality_weight: float = 0.4,
    rarity_weight: float = 0.6,
    persona: str = "",
    candidates_per_prompt: int = 4,
    _prompt_emb=None,
    _cand_embs=None,
) -> Optional[DivPORecord]:
    """Select a (chosen, rejected) pair using DivPO scoring.

    chosen   = highest combined score among quality-eligible candidates
    rejected = lowest combined score among quality-eligible candidates
    Returns None if fewer than 2 eligible candidates.
    Pass _prompt_emb/_cand_embs (pre-computed numpy arrays) to skip encode calls.
    """
    if len(candidates) < 2:
        return None

    scored: list[ScoredCandidate] = []
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        cand_emb = _cand_embs[i] if _cand_embs is not None else None
        other_embs = [_cand_embs[j] for j in range(len(candidates)) if j != i] if _cand_embs is not None else None
        q = score_quality(prompt, cand, embedder, _prompt_emb=_prompt_emb, _response_emb=cand_emb)
        r = score_rarity(cand, others, embedder, _response_emb=cand_emb, _other_embs=other_embs)
        combined = quality_weight * q + rarity_weight * r
        scored.append(ScoredCandidate(text=cand, quality=q, rarity=r, combined=combined))

    eligible = [s for s in scored if s.quality >= min_quality]
    if len(eligible) < 2:
        return None

    chosen = max(eligible, key=lambda s: s.combined)
    rejected = min(eligible, key=lambda s: s.combined)

    if chosen.text == rejected.text:
        return None

    return DivPORecord(
        prompt=prompt,
        chosen=chosen.text,
        rejected=rejected.text,
        metadata={
            "persona": persona,
            "chosen_quality": round(chosen.quality, 4),
            "chosen_rarity": round(chosen.rarity, 4),
            "rejected_quality": round(rejected.quality, 4),
            "rejected_rarity": round(rejected.rarity, 4),
            "quality_weight": quality_weight,
            "rarity_weight": rarity_weight,
            "min_quality": min_quality,
            "candidates_per_prompt": candidates_per_prompt,
        },
    )


def select_pairs_batch(
    prompts: list[str],
    candidates_batch: list[list[str]],
    embedder,
    min_quality: float = 0.35,
    quality_weight: float = 0.4,
    rarity_weight: float = 0.6,
    persona: str = "",
    candidates_per_prompt: int = 4,
) -> list[Optional[DivPORecord]]:
    """Batch version of select_pair — one embedder.encode call for all prompts.

    Encodes all prompts + candidates together, then calls select_pair with
    pre-computed embeddings. 10-20x faster than calling select_pair per prompt.
    """
    import numpy as np

    # Build flat text list: [prompt_0, cand_0_0, ..., cand_0_N, prompt_1, ...]
    all_texts: list[str] = []
    offsets: list[int] = []
    for prompt, candidates in zip(prompts, candidates_batch):
        offsets.append(len(all_texts))
        all_texts.append(prompt)
        all_texts.extend(candidates)

    all_embs: np.ndarray = embedder.encode(all_texts, convert_to_numpy=True, batch_size=64)

    results: list[Optional[DivPORecord]] = []
    for prompt, candidates, offset in zip(prompts, candidates_batch, offsets):
        prompt_emb = all_embs[offset]
        cand_embs = list(all_embs[offset + 1: offset + 1 + len(candidates)])
        results.append(select_pair(
            prompt=prompt,
            candidates=candidates,
            embedder=embedder,
            min_quality=min_quality,
            quality_weight=quality_weight,
            rarity_weight=rarity_weight,
            persona=persona,
            candidates_per_prompt=candidates_per_prompt,
            _prompt_emb=prompt_emb,
            _cand_embs=cand_embs,
        ))
    return results
