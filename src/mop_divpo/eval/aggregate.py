"""Aggregation + statistics for the baseline table (research-plan.md §3.5-3.6).

Pure functions: metric observations come in, the table and bootstrap statistics
come out. Embedding is injected as `embed_fn` so this module is unit-testable
without SBERT. Each prompt is treated as one observation (§3.5).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

import numpy as np

from mop_divpo.metrics.diversity import distinct_n, self_bleu
from mop_divpo.metrics.semantic import mean_pairwise_cosine

# (display name, lower_is_better) for each metric column in the table.
AUTOMATED_METRICS = {
    "self_bleu": ("Self-BLEU", True),
    "distinct_1": ("Distinct-1", False),
    "distinct_2": ("Distinct-2", False),
    "sbert_cosine": ("SBERT-cos", True),
    "corpus_novelty": ("Novelty(corpus)", False),
}
JUDGE_METRICS = {
    "quality": ("Quality(L)", False),
    "novelty": ("Novelty(L)", False),
    "utility": ("Utility(L)", False),
    "persona_fidelity": ("Fidelity(L)", False),
}

EmbedFn = Callable[[list[str]], "np.ndarray"]


def per_prompt_automated_metrics(
    outputs_by_method_prompt: dict[tuple[str, str], list[str]],
    embed_fn: EmbedFn,
    corpus_novelty: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute automated metrics per (method, prompt), grouped into per-method lists.

    Returns obs[method][metric] = [value per prompt].
    """
    obs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, prompt), outputs in outputs_by_method_prompt.items():
        if len(outputs) < 2:
            continue
        obs[method]["self_bleu"].append(self_bleu(outputs))
        obs[method]["distinct_1"].append(distinct_n(outputs, 1))
        obs[method]["distinct_2"].append(distinct_n(outputs, 2))
        obs[method]["sbert_cosine"].append(mean_pairwise_cosine(embed_fn(outputs)))
        if corpus_novelty and (method, prompt) in corpus_novelty:
            obs[method]["corpus_novelty"].append(corpus_novelty[(method, prompt)])
    return {m: dict(v) for m, v in obs.items()}


def _judge_scalar(scored: dict, metric: str) -> float | None:
    """Reduce one rubric's JSON to a single 1-5 score (quality = mean of three)."""
    block = scored.get(metric)
    if block is None:
        return None
    if metric == "quality":
        vals = [block.get(k) for k in ("relevance", "coherence", "substance")]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else None
    key = {"novelty": "novelty", "utility": "prewriting_utility",
           "persona_fidelity": "persona_fidelity"}[metric]
    val = block.get(key)
    return float(val) if isinstance(val, (int, float)) else None


def per_prompt_judge_metrics(
    scored_records: list[dict],
) -> dict[str, dict[str, list[float]]]:
    """Average judge scores per (method, prompt), grouped into per-method lists."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in scored_records:
        by_group[(record["method"], record["prompt"])].append(record)

    obs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, _prompt), records in by_group.items():
        for metric in JUDGE_METRICS:
            vals = [s for s in (_judge_scalar(r, metric) for r in records) if s is not None]
            if vals:
                obs[method][metric].append(float(np.mean(vals)))
    return {m: dict(v) for m, v in obs.items()}


def summarize(observations: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, dict]]:
    """mean/std/n per method per metric."""
    out: dict[str, dict[str, dict]] = {}
    for method, metrics in observations.items():
        out[method] = {}
        for metric, values in metrics.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            out[method][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(arr.size),
            }
    return out


def bootstrap_paired_pvalue(
    a: list[float], b: list[float], *, lower_is_better: bool,
    n_resamples: int = 1000, seed: int = 0,
) -> dict:
    """Paired bootstrap test that method-a beats method-b on a metric.

    a, b are paired per-prompt observations (same prompt order). Returns the
    observed mean difference (signed so positive = a is better), a one-sided
    p-value (fraction of resamples where a does NOT beat b), and effect size
    (Cohen's d on the paired differences).
    """
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "effect_size": 0.0, "n": 0}
    a_arr, b_arr = np.asarray(a[:n], float), np.asarray(b[:n], float)
    # signed improvement of a over b
    diff = (b_arr - a_arr) if lower_is_better else (a_arr - b_arr)
    observed = float(diff.mean())

    rng = random.Random(seed)
    wins = 0
    for _ in range(n_resamples):
        sample = [diff[rng.randrange(n)] for _ in range(n)]
        if float(np.mean(sample)) > 0:
            wins += 1
    p_value = 1.0 - wins / n_resamples  # one-sided: P(a fails to beat b)

    std = diff.std(ddof=1) if n > 1 else 0.0
    effect = observed / std if std > 1e-9 else 0.0
    return {"mean_diff": observed, "p_value": p_value, "effect_size": effect, "n": n}


def all_metric_specs() -> dict[str, tuple[str, bool]]:
    return {**AUTOMATED_METRICS, **JUDGE_METRICS}


def build_markdown_table(summary: dict, method_order: list[str]) -> str:
    specs = all_metric_specs()
    present = [m for m in specs if any(m in summary.get(meth, {}) for meth in method_order)]
    header_cells = []
    for m in present:
        name, lower = specs[m]
        header_cells.append(f"{name}{'↓' if lower else '↑'}")
    lines = ["| Method | " + " | ".join(header_cells) + " |",
             "|" + "---|" * (len(present) + 1)]
    for meth in method_order:
        cells = [meth]
        for m in present:
            stat = summary.get(meth, {}).get(m)
            cells.append(f"{stat['mean']:.3f}" if stat else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_csv(summary: dict, method_order: list[str]) -> str:
    specs = all_metric_specs()
    present = [m for m in specs if any(m in summary.get(meth, {}) for meth in method_order)]
    rows = ["method," + ",".join(present)]
    for meth in method_order:
        cells = [meth]
        for m in present:
            stat = summary.get(meth, {}).get(m)
            cells.append(f"{stat['mean']:.4f}" if stat else "")
        rows.append(",".join(cells))
    return "\n".join(rows)
