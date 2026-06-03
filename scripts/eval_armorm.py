#!/usr/bin/env python
"""Score existing evaluation outputs with ArmoRM reward model.

Adds a reward-model-grounded quality metric to the evaluation table, replacing
dependence on LLM-as-judge quality scores with a trained, human-preference-aligned
signal. This is the ground truth anchor DivPO was designed to use.

Reads:  outputs/{eval_dir}/generations.jsonl
Writes: outputs/{eval_dir}/armorm_scores.jsonl
        outputs/{eval_dir}/armorm_table.md

Usage (RTX 5060 Ti, 15.9GB VRAM — use 4-bit):
    python scripts/eval_armorm.py --dir outputs/evaluation_a100 --load-in-4bit

Usage (A100 80GB):
    python scripts/eval_armorm.py --dir outputs/evaluation_a100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/evaluation_a100")
    p.add_argument("--model", default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization via bitsandbytes. Required for <16GB VRAM.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Scoring batch size. ArmoRM scores one pair at a time internally.")
    p.add_argument("--token", default=None)
    args = p.parse_args()

    d = Path(args.dir)
    gens = _read_jsonl(d / "generations.jsonl")
    print(f"Loaded {len(gens)} outputs from {d}/generations.jsonl", flush=True)

    from mop_divpo.divpo.reward_quality import RewardModelScorer

    scorer = RewardModelScorer(
        model_id=args.model,
        load_in_4bit=args.load_in_4bit,
        normalize=False,  # keep raw rewards for absolute comparison
        token=args.token,
    )
    print(f"Loading ArmoRM ({'4-bit' if args.load_in_4bit else 'bf16'})...", flush=True)
    scorer._ensure_loaded()
    print("ArmoRM loaded.", flush=True)

    scored: list[dict] = []
    for i, rec in enumerate(gens):
        raw = scorer.score_raw(rec["prompt"], rec["output"])
        scored.append({**rec, "armorm_score": raw})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(gens)}] scored", flush=True)

    _write_jsonl(d / "armorm_scores.jsonl", scored)
    print(f"Wrote {len(scored)} scored records → {d}/armorm_scores.jsonl", flush=True)

    # Per-method mean ArmoRM score
    by_method: dict[str, list[float]] = defaultdict(list)
    for rec in scored:
        by_method[rec["method"]].append(rec["armorm_score"])

    METHOD_ORDER = ["base", "prompt_only", "single_lora", "mop_sft", "mop_divpo", "mop_divpo_v2"]
    lines = ["# ArmoRM Quality Scores (reward-model ground truth)\n",
             "| Method | ArmoRM Mean↑ | ArmoRM Std | N |",
             "|---|---|---|---|"]
    results = {}
    for m in METHOD_ORDER:
        scores = by_method.get(m, [])
        if not scores:
            continue
        import numpy as np
        arr = np.array(scores)
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        results[m] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(scores)}
        lines.append(f"| {m} | {mean:.4f} | {std:.4f} | {len(scores)} |")

    md = "\n".join(lines) + "\n"
    (d / "armorm_table.md").write_text(md, encoding="utf-8")
    (d / "armorm_table.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"Wrote armorm_table.{{md,json}} → {d}", flush=True)


if __name__ == "__main__":
    main()
