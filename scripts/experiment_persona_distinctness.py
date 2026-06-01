#!/usr/bin/env python
"""Phase 1 — persona distinctness validation (docs/research-plan.md §1).

Generates N outputs per (prompt, persona) on persona-neutral prompts, embeds
them with SBERT, and builds the 4x4 inter-persona cosine matrix. The
off-diagonals decide which personas survive into the paper (§1.5 decision rule).

Run on Colab/Kaggle/vast.ai T4 (~30 min for 400 outputs):
    pip install transformers peft accelerate sentence-transformers
    export HF_TOKEN=hf_xxx
    python scripts/experiment_persona_distinctness.py --stage divpo --n 5

Output: outputs/experiments/persona_distinctness.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]

DECISION_RULE = [
    (0.5, "distinct", "keep both"),
    (0.7, "borderline", "keep both, report as limitation"),
    (float("inf"), "mode-collapsed", "drop one or merge"),
]


def classify(value: float) -> tuple[str, str]:
    for threshold, label, action in DECISION_RULE:
        if value < threshold:
            return label, action
    return DECISION_RULE[-1][1], DECISION_RULE[-1][2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 persona distinctness experiment.")
    p.add_argument("--stage", default="divpo", choices=["sft", "divpo"],
                   help="Adapter stage to test (the deployed method is divpo).")
    p.add_argument("--n", type=int, default=5, help="Outputs per (prompt, persona).")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--personas", nargs="+", default=PERSONA_IDS, choices=PERSONA_IDS)
    p.add_argument("--limit-prompts", type=int, default=None,
                   help="Use only the first K prompts (smoke test).")
    p.add_argument("--output", default="outputs/experiments/persona_distinctness.json")
    p.add_argument("--token", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or None

    from mop_divpo.eval.prompts import PERSONA_DISTINCTNESS_PROMPTS
    from mop_divpo.inference.generate import MoPGenerator
    from mop_divpo.metrics.semantic import cross_group_mean_cosine, embed

    prompts = PERSONA_DISTINCTNESS_PROMPTS
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    gen = MoPGenerator(adapter_stage=args.stage, token=token, use_persona_prompt=True)

    # outputs_by_persona[persona] = list of generated strings (across all prompts)
    outputs_by_persona: dict[str, list[str]] = {p: [] for p in args.personas}
    records: list[dict] = []
    try:
        for prompt_id, prompt in enumerate(prompts):
            for persona in args.personas:
                texts = gen.generate(
                    prompt,
                    persona=persona,
                    n=args.n,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                for idx, text in enumerate(texts):
                    outputs_by_persona[persona].append(text)
                    records.append(
                        {"prompt_id": prompt_id, "prompt": prompt,
                         "persona": persona, "output_index": idx, "text": text}
                    )
                print(f"[{prompt_id + 1}/{len(prompts)}] {persona}: {args.n} outputs",
                      flush=True)
    finally:
        gen.unload()

    print("\nEmbedding outputs with SBERT ...", flush=True)
    embeddings = {p: embed(outputs_by_persona[p]) for p in args.personas}

    # 4x4 matrix: diagonal = 1.0, off-diagonal = mean cross-persona cosine.
    matrix: dict[str, dict[str, float]] = {}
    for a in args.personas:
        matrix[a] = {}
        for b in args.personas:
            if a == b:
                matrix[a][b] = 1.0
            else:
                matrix[a][b] = round(cross_group_mean_cosine(embeddings[a], embeddings[b]), 4)

    pair_findings = []
    for a, b in itertools.combinations(args.personas, 2):
        value = matrix[a][b]
        label, action = classify(value)
        pair_findings.append({"pair": [a, b], "cosine": value, "label": label, "action": action})

    distinct_pairs = sum(1 for f in pair_findings if f["cosine"] < 0.5)
    collapsed_pairs = [f for f in pair_findings if f["cosine"] > 0.7]
    verdict = (
        "PASS — MoP architectural claim validated"
        if distinct_pairs >= 1 and not collapsed_pairs
        else "REVIEW — see decision rule"
    )

    result = {
        "stage": args.stage,
        "n_per_pair": args.n,
        "n_prompts": len(prompts),
        "total_outputs": len(records),
        "personas": args.personas,
        "matrix": matrix,
        "pair_findings": sorted(pair_findings, key=lambda f: f["cosine"], reverse=True),
        "distinct_pairs_below_0.5": distinct_pairs,
        "collapsed_pairs_above_0.7": [f["pair"] for f in collapsed_pairs],
        "verdict": verdict,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": result, "records": records}, ensure_ascii=False,
                              indent=2), encoding="utf-8")

    print("\n=== Inter-persona SBERT cosine (off-diagonal = lower is more distinct) ===")
    header = "".join(f"{p[:10]:>12}" for p in args.personas)
    print(f"{'':>14}{header}")
    for a in args.personas:
        row = "".join(f"{matrix[a][b]:>12.3f}" for b in args.personas)
        print(f"{a[:12]:>14}{row}")
    print("\nPair findings (most similar first):")
    for f in result["pair_findings"]:
        print(f"  {f['pair'][0]:>22} <-> {f['pair'][1]:<22} {f['cosine']:.3f}  "
              f"[{f['label']}: {f['action']}]")
    print(f"\nVerdict: {verdict}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
