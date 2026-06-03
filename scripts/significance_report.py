#!/usr/bin/env python
"""Paired-bootstrap significance on EXISTING evaluation results.

The baseline table reports means only; reviewers ask whether the differences
are real. This recomputes per-prompt observations from a finished run
(generations.jsonl + llm_judge_scores.jsonl), then runs the paired bootstrap
(research-plan.md §3.5) for one reference method against every other method,
across all automated + judge metrics. CPU only — no API, no GPU, no retraining.

Usage:
    python scripts/significance_report.py --dir outputs/evaluation_a100 \
        --reference mop_divpo_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.eval.aggregate import (
    all_metric_specs,
    bootstrap_paired_pvalue,
    per_prompt_automated_metrics,
    per_prompt_judge_metrics,
    summarize,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="outputs/evaluation_a100",
                   help="Eval dir with generations.jsonl + llm_judge_scores.jsonl.")
    p.add_argument("--reference", default="mop_divpo_v2",
                   help="Method tested against all others (the paper's hero).")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = Path(args.dir)
    gens = _read_jsonl(d / "generations.jsonl")
    scored = _read_jsonl(d / "llm_judge_scores.jsonl")

    # Per-prompt observations (each prompt = one paired observation, §3.5).
    from mop_divpo.metrics.semantic import embed

    outputs_by_group: dict[tuple[str, str], list[str]] = {}
    for r in gens:
        outputs_by_group.setdefault((r["method"], r["prompt"]), []).append(r["output"])

    print(f"Embedding {len(gens)} outputs for SBERT metric (CPU)...", flush=True)
    auto_obs = per_prompt_automated_metrics(outputs_by_group, embed)
    judge_obs = per_prompt_judge_metrics(scored)

    methods = sorted({r["method"] for r in gens})
    if args.reference not in methods:
        p.error(f"reference {args.reference!r} not in methods {methods}")

    merged: dict[str, dict[str, list[float]]] = {
        m: {**auto_obs.get(m, {}), **judge_obs.get(m, {})} for m in methods
    }
    summary = summarize(merged)
    specs = all_metric_specs()

    # reference vs each other method, per metric
    ref = args.reference
    others = [m for m in methods if m != ref]
    sig: dict[str, dict[str, dict]] = {}
    for method in others:
        sig[method] = {}
        for metric, (_name, lower) in specs.items():
            a = merged[ref].get(metric)
            b = merged[method].get(metric)
            if a and b:
                sig[method][metric] = bootstrap_paired_pvalue(
                    a, b, lower_is_better=lower,
                    n_resamples=args.n_resamples, seed=args.seed,
                )

    present = [m for m in specs if any(m in merged[meth] for meth in methods)]

    # ── markdown ──────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# Significance: `{ref}` vs baselines (paired bootstrap, "
                 f"{args.n_resamples} resamples, n=30 prompts)\n")
    lines.append("Cell = signed mean diff (reference − baseline, oriented so **+ "
                 "means the reference is better**); `*` = p<0.05, `**` = p<0.01.\n")

    header = "| Comparison | " + " | ".join(
        f"{specs[m][0]}{'↓' if specs[m][1] else '↑'}" for m in present) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(present) + 1))

    for method in others:
        cells = [f"{ref} vs {method}"]
        for m in present:
            s = sig[method].get(m)
            if not s:
                cells.append("—")
                continue
            star = "**" if s["p_value"] < 0.01 else ("*" if s["p_value"] < 0.05 else "")
            cells.append(f"{s['mean_diff']:+.3f}{star} (p={s['p_value']:.3f})")
        lines.append("| " + " | ".join(cells) + " |")

    # verdict
    lines.append("\n## Verdict\n")
    for method in others:
        wins = [specs[m][0] for m in present
                if sig[method].get(m) and sig[method][m]["mean_diff"] > 0
                and sig[method][m]["p_value"] < 0.05]
        losses = [specs[m][0] for m in present
                  if sig[method].get(m) and sig[method][m]["mean_diff"] < 0
                  and sig[method][m]["p_value"] < 0.05]
        lines.append(f"- **vs {method}**: significantly better on {len(wins)} "
                     f"({', '.join(wins) or 'none'}); significantly worse on "
                     f"{len(losses)} ({', '.join(losses) or 'none'}).")

    md = "\n".join(lines) + "\n"
    out_md = d / f"significance_{ref}.md"
    out_json = d / f"significance_{ref}.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(
        {"reference": ref, "n_resamples": args.n_resamples,
         "summary": summary, "significance": sig}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(md)
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
