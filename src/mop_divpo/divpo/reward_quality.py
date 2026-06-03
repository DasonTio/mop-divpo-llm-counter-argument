"""Reward-model quality scoring for DivPO pair selection.

The heuristic quality floor in `scoring.py` (prompt-response cosine + length +
repetition regex) cannot tell a vacuous-but-fluent counter-argument from a
substantive one. The original DivPO (Lanchantin et al., 2025) avoids this by
using a trained reward model (ArmoRM) as the quality threshold; the entire
"rare AND good" guarantee rests on that anchor. This module restores it.

The scorer is dependency-injectable: `pairs.py` accepts any object exposing
`score_batch(prompt, responses) -> list[float]`, so tests run with a fake and
the real 8B model is only ever loaded on GPU (Colab/A100), never imported here
at module load time.

Usage (GPU):
    scorer = RewardModelScorer()                 # lazy-loads on first score
    qualities = scorer.score_batch(prompt, cands)  # normalized to [0, 1]
"""
from __future__ import annotations

from typing import Protocol


DEFAULT_REWARD_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"


class QualityScorer(Protocol):
    """Anything that maps a prompt + a pool of responses to [0, 1] quality."""

    def score_batch(self, prompt: str, responses: list[str]) -> list[float]:
        ...


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize raw rewards to [0, 1] within a candidate pool.

    Makes the absolute `min_quality` floor behave as a DivPO-style *relative*
    threshold (a fraction of the pool's reward range), which is faithful to
    DivPO's rho band rather than an arbitrary absolute cutoff.

    Edge case: a degenerate pool (all scores equal, or <2 scores) returns all
    1.0 so nothing is spuriously filtered.
    """
    if len(scores) < 2:
        return [1.0] * len(scores)
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class RewardModelScorer:
    """ArmoRM-based quality scorer. Lazy-loads the model on first use.

    ArmoRM returns a single gated multi-objective reward per (prompt, response).
    Raw rewards are min-max normalized within each candidate pool by default so
    the `min_quality` floor in `pairs.py` keeps [0, 1] semantics.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_REWARD_MODEL,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        max_length: int = 2048,
        normalize: bool = True,
        token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self.normalize = normalize
        self.token = token
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, token=self.token
        )

        kwargs: dict = dict(
            trust_remote_code=True,
            token=self.token,
        )
        if self.load_in_4bit:
            # 4-bit via bitsandbytes: ~5GB VRAM vs ~16GB for bf16.
            # Required when VRAM < 16GB (e.g. RTX 5060 Ti 15.9GB).
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
            kwargs["torch_dtype"] = torch_dtype

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **kwargs
        )
        self._model.eval()

    def score_raw(self, prompt: str, response: str) -> float:
        """Raw ArmoRM reward for one (prompt, response) pair."""
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model(input_ids)
        # ArmoRM exposes the gated scalar reward as `output.score`.
        return float(output.score.float().item())

    def score_batch(self, prompt: str, responses: list[str]) -> list[float]:
        """Score a pool of responses for one prompt.

        ArmoRM's custom head scores one pair at a time, so this loops; the cost
        is dominated by the 8B forward pass either way. Returns normalized [0, 1]
        scores unless `normalize=False`.
        """
        if not responses:
            return []
        raw = [self.score_raw(prompt, r) for r in responses]
        return normalize_scores(raw) if self.normalize else raw
