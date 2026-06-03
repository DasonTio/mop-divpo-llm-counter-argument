"""DivPO preference pair construction from scored candidates.

Two pair-selection strategies:
  select_pairs_batch               — v1: rarity against within-persona siblings.
  select_pairs_cross_persona_batch — v2: rarity against the full cross-persona pool.

v2 aligns the training signal with the cross-persona SBERT-cosine metric used in
evaluation. v1 optimised within-batch diversity which does not transfer to the
cross-persona setting and empirically reduced cross-persona distinctness.
"""
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
    min_rarity_margin: float = 0.0,
    quality_weight: float = 0.4,
    rarity_weight: float = 0.6,
    persona: str = "",
    candidates_per_prompt: int = 4,
    quality_override: Optional[list[float]] = None,
    _prompt_emb=None,
    _cand_embs=None,
) -> Optional[DivPORecord]:
    """Select a (chosen, rejected) pair using DivPO scoring.

    chosen   = highest combined score among quality-eligible candidates
    rejected = lowest combined score among quality-eligible candidates
    Returns None if fewer than 2 eligible candidates.
    Pass _prompt_emb/_cand_embs (pre-computed numpy arrays) to skip encode calls.

    quality_override: per-candidate quality in [0, 1] from a reward model. When
    provided it replaces the heuristic `score_quality` — this is the ArmoRM
    anchor path that restores DivPO's "rare AND good" guarantee.
    """
    if len(candidates) < 2:
        return None
    if quality_override is not None and len(quality_override) != len(candidates):
        raise ValueError(
            f"quality_override length {len(quality_override)} != "
            f"candidates length {len(candidates)}"
        )

    scored: list[ScoredCandidate] = []
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        cand_emb = _cand_embs[i] if _cand_embs is not None else None
        other_embs = [_cand_embs[j] for j in range(len(candidates)) if j != i] if _cand_embs is not None else None
        if quality_override is not None:
            q = quality_override[i]
        else:
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
    if chosen.rarity - rejected.rarity < min_rarity_margin:
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
            "min_rarity_margin": min_rarity_margin,
            "candidates_per_prompt": candidates_per_prompt,
            "quality_source": "reward_model" if quality_override is not None else "heuristic",
        },
    )


def select_pairs_batch(
    prompts: list[str],
    candidates_batch: list[list[str]],
    embedder,
    min_quality: float = 0.35,
    min_rarity_margin: float = 0.0,
    quality_weight: float = 0.4,
    rarity_weight: float = 0.6,
    persona: str = "",
    candidates_per_prompt: int = 4,
    quality_scorer=None,
) -> list[Optional[DivPORecord]]:
    """Batch version of select_pair — one embedder.encode call for all prompts.

    Encodes all prompts + candidates together, then calls select_pair with
    pre-computed embeddings. 10-20x faster than calling select_pair per prompt.

    quality_scorer: optional reward-model scorer (see divpo.reward_quality). When
    provided, quality is scored per-persona pool by the reward model instead of
    the heuristic; rarity stays within-persona (this is the v1 path).
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
        quality_override = (
            quality_scorer.score_batch(prompt, candidates) if quality_scorer is not None else None
        )
        results.append(select_pair(
            prompt=prompt,
            candidates=candidates,
            embedder=embedder,
            min_quality=min_quality,
            min_rarity_margin=min_rarity_margin,
            quality_weight=quality_weight,
            rarity_weight=rarity_weight,
            persona=persona,
            candidates_per_prompt=candidates_per_prompt,
            quality_override=quality_override,
            _prompt_emb=prompt_emb,
            _cand_embs=cand_embs,
        ))
    return results


def select_pairs_cross_persona_batch(
    prompts: list[str],
    candidates_per_persona: dict[str, list[list[str]]],
    embedder,
    min_quality: float = 0.40,
    min_rarity_margin: float = 0.0,
    quality_weight: float = 0.5,
    rarity_weight: float = 0.5,
    candidates_per_prompt: int = 4,
    quality_scorer=None,
) -> dict[str, list[Optional[DivPORecord]]]:
    """Cross-persona DivPO pair selection (v2).

    For each prompt, all candidates from ALL personas are embedded together.
    Rarity for each candidate is computed against the full cross-persona pool
    (not just same-persona siblings). This directly aligns the DivPO training
    signal with the cross-persona SBERT-cosine metric used at evaluation time.

    Quality uses include_relevance=False: prompt-response cosine penalises
    good counter-arguments (which must semantically oppose the prompt). When
    quality_scorer is provided, the heuristic is replaced by the reward model
    scored over the FULL cross-persona pool (same scope as rarity).

    Args:
        prompts: shared prompt list (must be the same order for all personas).
        candidates_per_persona: {persona_id: [[cands per prompt_0], [cands per prompt_1], ...]}.
        embedder: SentenceTransformer instance.
        quality_scorer: optional reward-model scorer (see divpo.reward_quality);
            None falls back to the heuristic floor.

    Returns:
        {persona_id: [Optional[DivPORecord] per prompt]}.
    """
    import numpy as np

    persona_ids = list(candidates_per_persona.keys())
    results: dict[str, list[Optional[DivPORecord]]] = {p: [] for p in persona_ids}

    for prompt_idx, prompt in enumerate(prompts):
        # Flatten all candidates from all personas for this prompt.
        all_cands: list[str] = []
        persona_slices: dict[str, tuple[int, int]] = {}
        for persona in persona_ids:
            per_prompt = candidates_per_persona[persona]
            cands = per_prompt[prompt_idx] if prompt_idx < len(per_prompt) else []
            start = len(all_cands)
            all_cands.extend(cands)
            persona_slices[persona] = (start, len(all_cands))

        if not all_cands:
            for persona in persona_ids:
                results[persona].append(None)
            continue

        # One encode call: [prompt] + all candidates from all personas.
        all_texts = [prompt] + all_cands
        all_embs: np.ndarray = embedder.encode(all_texts, convert_to_numpy=True, batch_size=64)
        prompt_emb = all_embs[0]
        cand_embs = list(all_embs[1:])  # aligned with all_cands

        # Reward-model quality over the FULL cross-persona pool (one call per
        # prompt); indexed by global candidate position below.
        pool_quality = (
            quality_scorer.score_batch(prompt, all_cands) if quality_scorer is not None else None
        )

        for persona in persona_ids:
            start, end = persona_slices[persona]
            persona_cands = all_cands[start:end]
            persona_embs = cand_embs[start:end]

            if len(persona_cands) < 2:
                results[persona].append(None)
                continue

            scored: list[ScoredCandidate] = []
            for local_i, (cand, cand_emb) in enumerate(zip(persona_cands, persona_embs)):
                global_i = start + local_i
                if pool_quality is not None:
                    q = pool_quality[global_i]
                else:
                    # Quality: no relevance term (see module docstring).
                    q = score_quality(
                        prompt, cand, embedder,
                        include_relevance=False,
                        _prompt_emb=prompt_emb,
                        _response_emb=cand_emb,
                    )
                # Rarity: against ALL candidates from ALL personas (cross-persona).
                other_embs = [e for j, e in enumerate(cand_embs) if j != global_i]
                r = score_rarity(
                    cand, [], embedder,
                    _response_emb=cand_emb,
                    _other_embs=other_embs,
                )
                scored.append(ScoredCandidate(
                    text=cand, quality=q, rarity=r,
                    combined=quality_weight * q + rarity_weight * r,
                ))

            eligible = [s for s in scored if s.quality >= min_quality]
            if len(eligible) < 2:
                results[persona].append(None)
                continue

            chosen = max(eligible, key=lambda s: s.combined)
            rejected = min(eligible, key=lambda s: s.combined)

            if chosen.text == rejected.text:
                results[persona].append(None)
                continue
            if chosen.rarity - rejected.rarity < min_rarity_margin:
                results[persona].append(None)
                continue

            results[persona].append(DivPORecord(
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
                    "min_rarity_margin": min_rarity_margin,
                    "candidates_per_prompt": candidates_per_prompt,
                    "cross_persona": True,
                    "quality_source": "reward_model" if pool_quality is not None else "heuristic",
                },
            ))

    return results
