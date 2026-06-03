#!/usr/bin/env python
"""Per-persona diversity and judge-quality breakdowns."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.metrics.diversity import distinct_n, self_bleu
from mop_divpo.metrics.semantic import embed, mean_pairwise_cosine

PERSONA_ORDER = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]
QUALITY_FIELDS = ["quality", "novelty", "utility", "persona_fidelity"]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH for --run.")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    return label.strip(), Path(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_scalar(record: dict, metric: str) -> float | None:
    block = record.get(metric)
    if not isinstance(block, dict):
        return None
    if metric == "quality":
        values = [block.get(k) for k in ("relevance", "coherence", "substance")]
        values = [float(v) for v in values if isinstance(v, (int, float))]
        return float(np.mean(values)) if values else None
    key = {
        "novelty": "novelty",
        "utility": "prewriting_utility",
        "persona_fidelity": "persona_fidelity",
    }[metric]
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def stat(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def sort_key(row: dict) -> tuple:
    persona = row["persona"]
    try:
        persona_idx = PERSONA_ORDER.index(persona)
    except ValueError:
        persona_idx = len(PERSONA_ORDER)
    return (row["run"], row["method"], persona_idx, persona)


def build_quality_lookup(judge_path: Path | None) -> dict[str, dict[str, dict]]:
    if not judge_path or not judge_path.exists():
        return {}
    lookup: dict[str, dict[str, dict]] = {}
    for record in read_jsonl(judge_path):
        output_id = record.get("output_id")
        if not output_id:
            continue
        lookup[output_id] = {
            metric: judge_scalar(record, metric)
            for metric in QUALITY_FIELDS
        }
    return lookup


def summarize_run(
    *,
    label: str,
    generation_path: Path,
    judge_path: Path | None,
    sbert_model: str,
) -> list[dict]:
    records = read_jsonl(generation_path)
    texts = [record["output"] for record in records]
    embeddings = embed(texts, model_name=sbert_model)
    quality_lookup = build_quality_lookup(judge_path)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        persona = record.get("persona")
        if persona is None:
            continue
        groups[(record["method"], persona)].append(i)

    rows: list[dict] = []
    for (method, persona), indices in sorted(groups.items()):
        outputs = [records[i]["output"] for i in indices]
        embs = embeddings[indices]
        row = {
            "run": label,
            "method": method,
            "persona": persona,
            "n_outputs": len(outputs),
            "self_bleu": self_bleu(outputs),
            "distinct_1": distinct_n(outputs, 1),
            "distinct_2": distinct_n(outputs, 2),
            "sbert_cosine": mean_pairwise_cosine(embs),
            "sbert_distance": 1.0 - mean_pairwise_cosine(embs),
        }

        for metric in QUALITY_FIELDS:
            values = [
                quality_lookup.get(records[i]["output_id"], {}).get(metric)
                for i in indices
            ]
            values = [value for value in values if value is not None]
            metric_stat = stat(values)
            row[metric] = metric_stat["mean"]
            row[f"{metric}_n"] = metric_stat["n"]
        rows.append(row)
    return rows


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "method",
        "persona",
        "n_outputs",
        "self_bleu",
        "distinct_1",
        "distinct_2",
        "sbert_cosine",
        "sbert_distance",
        "quality",
        "novelty",
        "utility",
        "persona_fidelity",
        "quality_n",
        "novelty_n",
        "utility_n",
        "persona_fidelity_n",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Per-Persona Diversity and Quality",
        "",
        "Diversity is computed across one persona's 30 prompt-conditioned outputs. "
        "This is useful as a persona-breadth diagnostic, but it is prompt-confounded.",
        "",
        "Quality columns are present only where LLM-judge scores exist.",
        "",
        "| Run | Method | Persona | n | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ | Quality↑ | Novelty↑ | Utility↑ | Fidelity↑ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["run"],
                row["method"],
                row["persona"],
                str(row["n_outputs"]),
                fmt(row["self_bleu"]),
                fmt(row["distinct_1"]),
                fmt(row["distinct_2"]),
                fmt(row["sbert_cosine"]),
                fmt(row["quality"]),
                fmt(row["novelty"]),
                fmt(row["utility"]),
                fmt(row["persona_fidelity"]),
            ])
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-persona diversity and quality metrics.")
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        type=parse_run,
        help="Optional judge scores, formatted as LABEL=PATH_TO_LLM_JUDGE_JSONL.",
    )
    parser.add_argument("--outdir", default="outputs/persona_breakdown")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    judge_paths = dict(args.judge)

    all_rows: list[dict] = []
    for label, generation_path in args.run:
        print(f"[{label}] evaluating {generation_path} ...", flush=True)
        all_rows.extend(
            summarize_run(
                label=label,
                generation_path=generation_path,
                judge_path=judge_paths.get(label),
                sbert_model=args.sbert_model,
            )
        )

    all_rows = sorted(all_rows, key=sort_key)
    (outdir / "persona_breakdown.json").write_text(
        json.dumps({"rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(outdir / "persona_breakdown.csv", all_rows)
    write_markdown(outdir / "persona_breakdown.md", all_rows)
    print(f"Wrote persona_breakdown.{{json,csv,md}} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
