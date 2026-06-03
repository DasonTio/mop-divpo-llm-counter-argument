"""Persona-level diversity diagnostics.

These metrics separate the question "are outputs diverse overall?" from
"does persona conditioning create separable response modes?".
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np

from mop_divpo.metrics.semantic import pairwise_cosine_matrix

PERSONA_ORDER = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]


def _persona_sort_key(persona: str) -> tuple[int, str]:
    try:
        return (PERSONA_ORDER.index(persona), persona)
    except ValueError:
        return (len(PERSONA_ORDER), persona)


def _stat(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    return 1.0 - pairwise_cosine_matrix(embeddings)


def _mean_pairwise_distance(embeddings: np.ndarray) -> float | None:
    n = embeddings.shape[0]
    if n < 2:
        return None
    matrix = _distance_matrix(embeddings)
    upper = np.triu_indices(n, k=1)
    return float(np.mean(matrix[upper]))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    return float(1.0 - np.dot(a / a_norm, b / b_norm))


def _centroid(embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray(embeddings[indices], dtype=float).mean(axis=0)


def _silhouette_by_label(embeddings: np.ndarray, labels: list[str]) -> dict:
    if len(labels) < 3 or len(set(labels)) < 2:
        return {"summary": _stat([]), "per_label": {}}

    matrix = _distance_matrix(embeddings)
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    scores: list[float] = []
    per_label_scores: dict[str, list[float]] = defaultdict(list)
    for i, label in enumerate(labels):
        same = [j for j in label_to_indices[label] if j != i]
        if not same:
            continue
        a = float(np.mean(matrix[i, same]))
        other_means = [
            float(np.mean(matrix[i, indices]))
            for other, indices in label_to_indices.items()
            if other != label and indices
        ]
        if not other_means:
            continue
        b = min(other_means)
        denom = max(a, b)
        if denom <= 1e-9:
            continue
        score = (b - a) / denom
        scores.append(score)
        per_label_scores[label].append(score)

    return {
        "summary": _stat(scores),
        "per_label": {
            label: _stat(values)
            for label, values in sorted(per_label_scores.items(), key=lambda kv: _persona_sort_key(kv[0]))
        },
    }


def build_persona_diversity_report(
    records: list[dict],
    embeddings: np.ndarray,
    *,
    run_label: str,
    base_method: str = "base",
) -> dict:
    """Build inter/intra-persona diagnostics from generated outputs.

    The current generation files contain one output per (prompt, persona) for
    persona methods. Therefore true same-prompt intra-persona diversity is only
    available for methods with repeated samples for the same persona and prompt.
    """
    if len(records) != embeddings.shape[0]:
        raise ValueError(
            f"records/embeddings length mismatch: {len(records)} vs {embeddings.shape[0]}"
        )

    by_method_prompt_persona: dict[tuple[str, str, str | None], list[int]] = defaultdict(list)
    by_method_prompt: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_method_persona: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    methods: list[str] = []
    seen_methods: set[str] = set()

    for i, record in enumerate(records):
        method = record["method"]
        prompt = record["prompt"]
        persona = record.get("persona")
        if method not in seen_methods:
            seen_methods.add(method)
            methods.append(method)
        by_method_prompt_persona[(method, prompt, persona)].append(i)
        by_method_prompt[(method, prompt)].append(i)
        by_method_persona[(method, persona)].append(i)

    reports: dict[str, dict] = {}

    base_same_prompt_by_prompt: dict[str, float] = {}
    for (method, prompt, persona), indices in by_method_prompt_persona.items():
        if method != base_method or len(indices) < 2:
            continue
        distance = _mean_pairwise_distance(embeddings[indices])
        if distance is not None:
            base_same_prompt_by_prompt[prompt] = distance

    for method in methods:
        method_report: dict = {}

        # Inter-persona diversity: same prompt, different persona centroids.
        inter_values: list[float] = []
        inter_by_prompt: dict[str, float] = {}
        inter_pair_values: dict[str, list[float]] = defaultdict(list)
        for (m, prompt), _indices in by_method_prompt.items():
            if m != method:
                continue
            persona_groups = {
                persona: idxs
                for (mm, pp, persona), idxs in by_method_prompt_persona.items()
                if mm == method and pp == prompt and persona is not None
            }
            if len(persona_groups) < 2:
                continue
            personas = sorted(persona_groups, key=_persona_sort_key)
            centroids = {p: _centroid(embeddings, persona_groups[p]) for p in personas}
            distances = []
            for a, b in combinations(personas, 2):
                distance = _cosine_distance(centroids[a], centroids[b])
                distances.append(distance)
                inter_pair_values[f"{a}__{b}"].append(distance)
            prompt_mean = float(np.mean(distances))
            inter_values.append(prompt_mean)
            inter_by_prompt[prompt] = prompt_mean

        method_report["prompt_controlled_inter_persona"] = {
            "summary": _stat(inter_values),
            "per_prompt": inter_by_prompt,
            "per_pair": {
                pair: _stat(values)
                for pair, values in sorted(inter_pair_values.items())
            },
            "definition": "Mean SBERT cosine distance between persona outputs for the same prompt.",
        }

        # Same-prompt intra-persona diversity: repeated samples for same persona and prompt.
        intra_values: list[float] = []
        intra_by_persona: dict[str, list[float]] = defaultdict(list)
        intra_by_prompt_persona: dict[str, float] = {}
        for (m, prompt, persona), indices in by_method_prompt_persona.items():
            if m != method or len(indices) < 2:
                continue
            distance = _mean_pairwise_distance(embeddings[indices])
            if distance is None:
                continue
            persona_key = "none" if persona is None else persona
            key = f"{prompt}::{persona_key}"
            intra_values.append(distance)
            intra_by_persona[persona_key].append(distance)
            intra_by_prompt_persona[key] = distance

        method_report["same_prompt_intra_persona"] = {
            "summary": _stat(intra_values),
            "per_persona": {
                persona: _stat(values)
                for persona, values in sorted(intra_by_persona.items(), key=lambda kv: _persona_sort_key(kv[0]))
            },
            "per_prompt_persona": intra_by_prompt_persona,
            "available": bool(intra_values),
            "definition": "Mean SBERT cosine distance among repeated samples for the same prompt and persona.",
        }

        # Current-data proxy: each persona's dispersion across prompts.
        cross_prompt_values: list[float] = []
        cross_prompt_by_persona: dict[str, float] = {}
        for (m, persona), indices in by_method_persona.items():
            if m != method or persona is None or len(indices) < 2:
                continue
            distance = _mean_pairwise_distance(embeddings[indices])
            if distance is None:
                continue
            cross_prompt_values.append(distance)
            cross_prompt_by_persona[persona] = distance

        method_report["cross_prompt_persona_dispersion"] = {
            "summary": _stat(cross_prompt_values),
            "per_persona": {
                persona: cross_prompt_by_persona[persona]
                for persona in sorted(cross_prompt_by_persona, key=_persona_sort_key)
            },
            "caveat": "Prompt-confounded: outputs differ by prompt as well as persona.",
            "definition": "Mean SBERT cosine distance among one persona's outputs across prompts.",
        }

        # Compare persona separation to base random-sample variation for the same prompts.
        matched_prompts = [
            prompt for prompt in inter_by_prompt
            if prompt in base_same_prompt_by_prompt and base_same_prompt_by_prompt[prompt] > 1e-9
        ]
        paired_ratios = [
            inter_by_prompt[prompt] / base_same_prompt_by_prompt[prompt]
            for prompt in matched_prompts
        ]
        diffs = [
            inter_by_prompt[prompt] - base_same_prompt_by_prompt[prompt]
            for prompt in matched_prompts
        ]
        inter_matched = [inter_by_prompt[prompt] for prompt in matched_prompts]
        base_matched = [base_same_prompt_by_prompt[prompt] for prompt in matched_prompts]
        ratio_of_means = None
        if base_matched and float(np.mean(base_matched)) > 1e-9:
            ratio_of_means = float(np.mean(inter_matched) / np.mean(base_matched))
        method_report["persona_over_base_sampling"] = {
            "ratio": {
                "mean": ratio_of_means,
                "std": 0.0 if ratio_of_means is not None else None,
                "n": len(matched_prompts),
            },
            "paired_ratio": _stat(paired_ratios),
            "difference": _stat(diffs),
            "n_matched_prompts": len(matched_prompts),
            "definition": (
                "Ratio of mean inter-persona distance to mean base same-prompt random-sample distance "
                "over matched prompts. paired_ratio stores the mean of per-prompt ratios."
            ),
        }

        # Non-prompt-controlled cluster probe. Useful, but should not be the headline.
        persona_indices: list[int] = []
        labels: list[str] = []
        for (m, persona), indices in by_method_persona.items():
            if m == method and persona is not None:
                persona_indices.extend(indices)
                labels.extend([persona] * len(indices))
        method_report["persona_silhouette"] = _silhouette_by_label(
            embeddings[persona_indices], labels
        ) if persona_indices else {"summary": _stat([]), "per_label": {}}
        method_report["persona_silhouette"]["definition"] = (
            "Cosine-distance silhouette by persona label across all prompts; positive means persona clustering."
        )

        reports[method] = method_report

    return {
        "run_label": run_label,
        "n_records": len(records),
        "methods": methods,
        "metrics": reports,
        "global_caveat": (
            "True same-prompt intra-persona diversity requires multiple generations for each "
            "(prompt, persona). Existing persona-method files contain one generation per "
            "(prompt, persona), so intra-persona same-prompt metrics are unavailable there."
        ),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def build_persona_diversity_markdown(report: dict) -> str:
    lines = [
        f"# Persona Diversity Evaluation: {report['run_label']}",
        "",
        "Distances are `1 - SBERT cosine`, so higher means more semantically diverse.",
        "",
        f"**Caveat:** {report['global_caveat']}",
        "",
        "## Prompt-Controlled Inter-Persona Diversity",
        "",
        "| Method | Mean Distance↑ | Std | Prompts | Ratio vs Base Sampling↑ | Δ vs Base Sampling↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        block = report["metrics"][method]
        inter = block["prompt_controlled_inter_persona"]["summary"]
        ratio = block["persona_over_base_sampling"]["ratio"]
        diff = block["persona_over_base_sampling"]["difference"]
        lines.append(
            "| "
            + " | ".join([
                method,
                _fmt(inter["mean"]),
                _fmt(inter["std"]),
                str(inter["n"]),
                _fmt(ratio["mean"]),
                _fmt(diff["mean"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Same-Prompt Intra-Persona Diversity",
        "",
        "| Method | Mean Distance↑ | Std | Groups | Available? |",
        "|---|---:|---:|---:|---|",
    ])
    for method in report["methods"]:
        intra = report["metrics"][method]["same_prompt_intra_persona"]
        summary = intra["summary"]
        lines.append(
            "| "
            + " | ".join([
                method,
                _fmt(summary["mean"]),
                _fmt(summary["std"]),
                str(summary["n"]),
                "yes" if intra["available"] else "no",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Cross-Prompt Persona Dispersion",
        "",
        "This is a proxy for persona breadth, but it is prompt-confounded.",
        "",
        "| Method | Mean Distance↑ | Std | Personas |",
        "|---|---:|---:|---:|",
    ])
    for method in report["methods"]:
        summary = report["metrics"][method]["cross_prompt_persona_dispersion"]["summary"]
        lines.append(
            "| "
            + " | ".join([method, _fmt(summary["mean"]), _fmt(summary["std"]), str(summary["n"])])
            + " |"
        )

    lines.extend([
        "",
        "## Persona Silhouette",
        "",
        "This is not prompt-controlled. Positive values mean outputs cluster by persona; negative values suggest prompt/topic dominates persona clustering.",
        "",
        "| Method | Silhouette↑ | Std | Outputs |",
        "|---|---:|---:|---:|",
    ])
    for method in report["methods"]:
        summary = report["metrics"][method]["persona_silhouette"]["summary"]
        lines.append(
            "| "
            + " | ".join([method, _fmt(summary["mean"]), _fmt(summary["std"]), str(summary["n"])])
            + " |"
        )

    return "\n".join(lines) + "\n"


def build_persona_diversity_csv(report: dict) -> str:
    rows = [
        "run,method,inter_mean,inter_std,inter_n,base_ratio_mean,base_diff_mean,"
        "same_prompt_intra_mean,same_prompt_intra_n,cross_prompt_dispersion_mean,"
        "silhouette_mean,silhouette_n"
    ]
    for method in report["methods"]:
        block = report["metrics"][method]
        inter = block["prompt_controlled_inter_persona"]["summary"]
        ratio = block["persona_over_base_sampling"]["ratio"]
        diff = block["persona_over_base_sampling"]["difference"]
        intra = block["same_prompt_intra_persona"]["summary"]
        cross = block["cross_prompt_persona_dispersion"]["summary"]
        sil = block["persona_silhouette"]["summary"]
        rows.append(",".join([
            report["run_label"],
            method,
            "" if inter["mean"] is None else f"{inter['mean']:.6f}",
            "" if inter["std"] is None else f"{inter['std']:.6f}",
            str(inter["n"]),
            "" if ratio["mean"] is None else f"{ratio['mean']:.6f}",
            "" if diff["mean"] is None else f"{diff['mean']:.6f}",
            "" if intra["mean"] is None else f"{intra['mean']:.6f}",
            str(intra["n"]),
            "" if cross["mean"] is None else f"{cross['mean']:.6f}",
            "" if sil["mean"] is None else f"{sil['mean']:.6f}",
            str(sil["n"]),
        ]))
    return "\n".join(rows) + "\n"


def build_comparison_markdown(reports: list[dict]) -> str:
    lines = [
        "# Persona Diversity Comparison",
        "",
        "Distances are `1 - SBERT cosine`, so higher means more semantically diverse.",
        "",
        "| Run | Method | Inter-Persona↑ | Ratio vs Base Sampling↑ | Same-Prompt Intra↑ | Cross-Prompt Dispersion↑ | Silhouette↑ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        for method in report["methods"]:
            block = report["metrics"][method]
            inter = block["prompt_controlled_inter_persona"]["summary"]
            ratio = block["persona_over_base_sampling"]["ratio"]
            intra = block["same_prompt_intra_persona"]["summary"]
            cross = block["cross_prompt_persona_dispersion"]["summary"]
            sil = block["persona_silhouette"]["summary"]
            lines.append(
                "| "
                + " | ".join([
                    report["run_label"],
                    method,
                    _fmt(inter["mean"]),
                    _fmt(ratio["mean"]),
                    _fmt(intra["mean"]),
                    _fmt(cross["mean"]),
                    _fmt(sil["mean"]),
                ])
                + " |"
            )
    return "\n".join(lines) + "\n"


def build_comparison_csv(reports: list[dict]) -> str:
    rows = [
        "run,method,inter_mean,base_ratio_mean,same_prompt_intra_mean,"
        "cross_prompt_dispersion_mean,silhouette_mean"
    ]
    for report in reports:
        for method in report["methods"]:
            block = report["metrics"][method]
            values = [
                block["prompt_controlled_inter_persona"]["summary"]["mean"],
                block["persona_over_base_sampling"]["ratio"]["mean"],
                block["same_prompt_intra_persona"]["summary"]["mean"],
                block["cross_prompt_persona_dispersion"]["summary"]["mean"],
                block["persona_silhouette"]["summary"]["mean"],
            ]
            rows.append(",".join([
                report["run_label"],
                method,
                *("" if value is None else f"{value:.6f}" for value in values),
            ]))
    return "\n".join(rows) + "\n"
