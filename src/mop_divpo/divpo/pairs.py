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
) -> Optional[DivPORecord]:
    """Select a (chosen, rejected) pair using DivPO scoring.

    chosen   = highest combined score among quality-eligible candidates
    rejected = lowest combined score among quality-eligible candidates
    Returns None if fewer than 2 eligible candidates.
    """
    if len(candidates) < 2:
        return None

    scored: list[ScoredCandidate] = []
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        q = score_quality(prompt, cand, embedder)
        r = score_rarity(cand, others, embedder)
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
