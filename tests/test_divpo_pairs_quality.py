"""Reward-model quality floor for DivPO pair selection.

Verifies the ArmoRM anchor path: injected reward-model quality drives chosen/
rejected selection and the quality floor, replacing the heuristic. Uses a fake
scorer + fake embedder so the suite runs without the 8B reward model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.divpo.pairs import select_pair, select_pairs_cross_persona_batch
from mop_divpo.divpo.reward_quality import normalize_scores


class FakeEmbedder:
    """Deterministic embeddings — distinct unit-ish vectors per text."""

    def encode(self, texts, convert_to_numpy=True, batch_size=64):
        return np.array([[i + 1.0, float(i % 2), 0.5] for i in range(len(texts))])


class FakeScorer:
    """Returns preset quality per prompt and records the pools it scored."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def score_batch(self, prompt, responses):
        self.calls.append(list(responses))
        return list(self._scores[: len(responses)])


# ── normalize_scores ────────────────────────────────────────────────────────

def test_normalize_basic():
    assert normalize_scores([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalize_degenerate_pools():
    assert normalize_scores([]) == []
    assert normalize_scores([5.0]) == [1.0]
    assert normalize_scores([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]


# ── select_pair: quality_override drives selection + floor ────────────────────

def test_quality_override_selects_and_filters():
    cands = ["A", "B", "C"]
    cand_embs = [np.array([1.0, 0.0, 0.0]),
                 np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 0.0, 1.0])]
    rec = select_pair(
        prompt="claim",
        candidates=cands,
        embedder=None,                       # unused: override + precomputed embs
        min_quality=0.5,
        min_rarity_margin=-1.0,              # disable rarity gate to isolate quality
        quality_weight=1.0, rarity_weight=0.0,  # isolate quality
        quality_override=[0.9, 0.1, 0.8],    # B is below the 0.5 floor
        _prompt_emb=np.array([0.5, 0.5, 0.5]),
        _cand_embs=cand_embs,
    )
    assert rec is not None
    assert rec.chosen == "A"                 # highest quality
    assert rec.rejected == "C"               # lowest among eligible (B filtered out)
    assert rec.rejected != "B"
    assert rec.metadata["quality_source"] == "reward_model"


def test_quality_override_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_pair(
            prompt="claim",
            candidates=["A", "B"],
            embedder=None,
            quality_override=[0.9],          # wrong length
            _prompt_emb=np.zeros(3),
            _cand_embs=[np.zeros(3), np.ones(3)],
        )


def test_heuristic_path_marks_metadata():
    """Without override, quality_source records the heuristic floor.

    Three candidates (each >= 10 words so the length floor passes) with
    non-orthogonal embeddings so rarity differs and chosen != rejected.
    """
    long = "this is a sufficiently long counter argument with at least ten words here"
    rec = select_pair(
        prompt="claim",
        candidates=[long + " one", long + " two", long + " three"],
        embedder=FakeEmbedder(),
        min_quality=0.0,
        min_rarity_margin=-1.0,
        _prompt_emb=np.array([0.0, 0.0, 0.0]),               # relevance neutral
        _cand_embs=[np.array([1.0, 0.0, 0.0]),
                    np.array([1.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0])],
    )
    assert rec is not None
    assert rec.metadata["quality_source"] == "heuristic"


# ── cross-persona (v2): scorer applied over full pool, indexed per persona ────

def test_cross_persona_uses_reward_model_over_full_pool():
    prompts = ["claim"]
    candidates_per_persona = {
        "x": [["x0", "x1"]],
        "y": [["y0", "y1"]],
    }
    # Full pool order is x0, x1, y0, y1 -> 4 scores, all above the 0.5 floor.
    scorer = FakeScorer([0.9, 0.6, 0.85, 0.55])

    results = select_pairs_cross_persona_batch(
        prompts=prompts,
        candidates_per_persona=candidates_per_persona,
        embedder=FakeEmbedder(),
        min_quality=0.5,
        min_rarity_margin=-1.0,                   # disable rarity gate to isolate quality
        quality_weight=1.0, rarity_weight=0.0,   # isolate quality
        quality_scorer=scorer,
    )

    # Scored once, over the FULL cross-persona pool of 4 candidates.
    assert len(scorer.calls) == 1
    assert scorer.calls[0] == ["x0", "x1", "y0", "y1"]

    rx = results["x"][0]
    ry = results["y"][0]
    assert rx.chosen == "x0" and rx.rejected == "x1"
    assert ry.chosen == "y0" and ry.rejected == "y1"
    assert rx.metadata["quality_source"] == "reward_model"
    assert ry.metadata["cross_persona"] is True
