#!/usr/bin/env python
"""Evaluate inter-persona and intra-persona diversity from generations.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.eval.persona_diversity import (
    build_comparison_csv,
    build_comparison_markdown,
    build_persona_diversity_csv,
    build_persona_diversity_markdown,
    build_persona_diversity_report,
)
from mop_divpo.metrics.semantic import embed


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH for --run.")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute persona-level diversity diagnostics from saved generations."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        help="Run to evaluate, formatted as LABEL=PATH_TO_GENERATIONS_JSONL.",
    )
    parser.add_argument("--outdir", default="outputs/persona_diversity")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for label, gen_path in args.run:
        records = read_jsonl(gen_path)
        texts = [record["output"] for record in records]
        print(f"[{label}] embedding {len(texts)} outputs from {gen_path} ...", flush=True)
        embeddings = embed(texts, model_name=args.sbert_model)
        report = build_persona_diversity_report(records, embeddings, run_label=label)
        reports.append(report)

        stem = f"{label}_persona_diversity"
        (outdir / f"{stem}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (outdir / f"{stem}.md").write_text(
            build_persona_diversity_markdown(report),
            encoding="utf-8",
        )
        (outdir / f"{stem}.csv").write_text(
            build_persona_diversity_csv(report),
            encoding="utf-8",
        )
        print(f"[{label}] wrote {stem}.{{json,md,csv}}", flush=True)

    if len(reports) > 1:
        (outdir / "comparison.md").write_text(
            build_comparison_markdown(reports),
            encoding="utf-8",
        )
        (outdir / "comparison.csv").write_text(
            build_comparison_csv(reports),
            encoding="utf-8",
        )
        (outdir / "comparison.json").write_text(
            json.dumps({"runs": reports}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote comparison.{{json,md,csv}} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
