#!/usr/bin/env python
"""Phase 3 — the baseline evaluation table (research-plan.md §3).

Five methods x seven metrics + paired-bootstrap significance. This is the table
the paper lives or dies on (§3.6).

Pipeline (each stage is resumable so compute and API cost are never repeated):
    1. generate   600 outputs (5 methods x 30 prompts x 4)   -> generations.jsonl
    2. automated  Self-BLEU / Distinct-1,2 / SBERT / novelty -> CPU
    3. judge      LLM-as-judge 4 rubrics (cached)            -> llm_judge_scores.jsonl
    4. aggregate  mean+/-std, bootstrap vs MoP+DivPO, table  -> baseline_table.{json,csv,md}

Run generation on a GPU box (Colab/Kaggle/vast.ai), then judge+aggregate anywhere:
    export HF_TOKEN=hf_xxx ANTHROPIC_API_KEY=sk-...
    python scripts/run_baseline_evaluation.py --judge anthropic

    # split across machines:
    python scripts/run_baseline_evaluation.py --only-generate            # GPU box
    python scripts/run_baseline_evaluation.py --skip-generation --judge anthropic  # laptop
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]

# stage / persona prompt / per-persona expansion for each baseline (§3.1).
METHODS = {
    "base":        {"stage": "base",   "use_persona_prompt": False, "per_persona": False},
    "prompt_only": {"stage": "base",   "use_persona_prompt": True,  "per_persona": True},
    "single_lora": {"stage": "single", "use_persona_prompt": True,  "per_persona": True},
    "mop_sft":     {"stage": "sft",    "use_persona_prompt": True,  "per_persona": True},
    "mop_divpo":   {"stage": "divpo",  "use_persona_prompt": True,  "per_persona": True},
    "mop_divpo_v2": {"stage": "divpo_v2", "use_persona_prompt": True, "per_persona": True},
}
HEADLINE = "mop_divpo"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def generate_all(methods: list[str], prompts: list[str], token: str | None,
                 *, n_base: int = 4, temperature: float = 0.9,
                 max_new_tokens: int = 256) -> list[dict]:
    from mop_divpo.inference.generate import MoPGenerator

    records: list[dict] = []
    for method in methods:
        cfg = METHODS[method]
        gen = MoPGenerator(
            adapter_stage=cfg["stage"], token=token,
            use_persona_prompt=cfg["use_persona_prompt"],
        )
        try:
            for prompt in prompts:
                slug = _slug(prompt)
                if cfg["per_persona"]:
                    for i, persona in enumerate(PERSONA_IDS):
                        text = gen.generate(
                            prompt, persona=persona, n=1, temperature=temperature,
                            max_new_tokens=max_new_tokens, as_counter_argument=True,
                        )[0]
                        records.append({
                            "output_id": f"{method}__{slug}__{i}", "method": method,
                            "prompt": prompt, "persona": persona, "output": text,
                        })
                else:
                    texts = gen.generate(
                        prompt, persona=None, n=n_base, temperature=temperature,
                        max_new_tokens=max_new_tokens, as_counter_argument=True,
                    )
                    for i, text in enumerate(texts):
                        records.append({
                            "output_id": f"{method}__{slug}__{i}", "method": method,
                            "prompt": prompt, "persona": None, "output": text,
                        })
                print(f"  [{method}] {prompt[:50]}", flush=True)
        finally:
            gen.unload()
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_records_to_prompts(records: list[dict], prompts: list[str]) -> list[dict]:
    """Keep records whose prompt is in the selected evaluation prompt list."""
    selected = set(prompts)
    return [record for record in records if record["prompt"] in selected]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 baseline evaluation table.")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--judge", choices=["anthropic", "openai", "none"], default="none")
    p.add_argument("--judge-model", default=None,
                   help="Primary judge model. Default: gpt-4o-mini (openai) or claude-sonnet-4 (anthropic).")
    p.add_argument("--calibration-judge", choices=["anthropic", "openai", "none"], default="none",
                   help="Stronger judge for inter-judge agreement on a random sample. "
                        "Use 'openai' for gpt-4o when --judge uses gpt-4o-mini.")
    p.add_argument("--calibration-model", default=None,
                   help="Calibration judge model. Default: gpt-4o (openai) or claude-opus (anthropic).")
    p.add_argument("--calibration-fraction", type=float, default=0.1,
                   help="Fraction of outputs to re-score with calibration judge (default: 0.1 = 10%%).")
    p.add_argument("--only-generate", action="store_true", help="Generate then stop.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Reuse existing generations.jsonl.")
    p.add_argument("--outdir", default="outputs/evaluation")
    p.add_argument("--token", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or None
    outdir = Path(args.outdir)
    gen_path = outdir / "generations.jsonl"

    from mop_divpo.eval.prompts import EVALUATION_PROMPTS

    prompts = EVALUATION_PROMPTS
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    # --- 1. Generation ---
    if args.skip_generation:
        records = read_jsonl(gen_path)
        records = filter_records_to_prompts(records, prompts)
        print(f"Loaded {len(records)} generations from {gen_path}")
    else:
        print(f"Generating outputs for {args.methods} on {len(prompts)} prompts ...")
        records = generate_all(args.methods, prompts, token)
        write_jsonl(gen_path, records)
        print(f"Wrote {len(records)} generations -> {gen_path}")
    if args.only_generate:
        return

    # --- 2. Automated metrics ---
    from mop_divpo.eval.aggregate import (
        bootstrap_paired_pvalue, build_csv, build_markdown_table,
        per_prompt_automated_metrics, per_prompt_judge_metrics, summarize,
        all_metric_specs,
    )
    from mop_divpo.metrics.semantic import embed

    outputs_by_group: dict[tuple[str, str], list[str]] = {}
    for r in records:
        outputs_by_group.setdefault((r["method"], r["prompt"]), []).append(r["output"])
    auto_obs = per_prompt_automated_metrics(outputs_by_group, embed)

    # --- 3. LLM judge ---
    judge_obs: dict = {}
    if args.judge != "none":
        from mop_divpo.eval.llm_judge import (
            JudgeCache, make_anthropic_caller, make_openai_caller,
            score_all_outputs, compute_interjudge_agreement,
        )

        # Primary judge: gpt-4o-mini by default (cheap, strong vs Qwen 0.5B).
        if args.judge == "anthropic":
            call_fn = make_anthropic_caller(args.judge_model or "claude-sonnet-4-20250514")
        else:
            call_fn = make_openai_caller(args.judge_model or "gpt-4o-mini")

        cache = JudgeCache(outdir / "judge_cache.json")
        print(f"Running LLM judge ({args.judge} / {args.judge_model or 'gpt-4o-mini'}) "
              f"over {len(records)} outputs ...")
        scored = score_all_outputs(
            records, call_fn=call_fn, cache=cache,
            output_path=outdir / "llm_judge_scores.jsonl",
        )
        judge_obs = per_prompt_judge_metrics(scored)

        # --- 3b. Inter-judge calibration (Spearman ρ) ---
        if args.calibration_judge != "none":
            if args.calibration_judge == "anthropic":
                calib_fn = make_anthropic_caller(args.calibration_model or "claude-opus-4-20250514")
            else:
                calib_fn = make_openai_caller(args.calibration_model or "gpt-4o")
            calib_cache = JudgeCache(outdir / "calibration_cache.json")
            n_sample = max(10, int(len(scored) * args.calibration_fraction))
            print(f"\nInter-judge calibration: {args.calibration_judge} / "
                  f"{args.calibration_model or 'gpt-4o'} on {n_sample} outputs "
                  f"({args.calibration_fraction:.0%} sample) ...")
            compute_interjudge_agreement(
                scored,
                call_fn_calibration=calib_fn,
                fraction=args.calibration_fraction,
                cache=calib_cache,
                output_path=outdir / "inter_judge_agreement.json",
            )

    # --- 4. Aggregate + table ---
    merged_obs: dict[str, dict[str, list[float]]] = {}
    for method in args.methods:
        merged_obs[method] = {**auto_obs.get(method, {}), **judge_obs.get(method, {})}
    summary = summarize(merged_obs)

    specs = all_metric_specs()
    significance: dict[str, dict[str, dict]] = {}
    if HEADLINE in merged_obs:
        for method in args.methods:
            if method == HEADLINE:
                continue
            significance[method] = {}
            for metric, (_name, lower) in specs.items():
                a = merged_obs[HEADLINE].get(metric)
                b = merged_obs[method].get(metric)
                if a and b:
                    significance[method][metric] = bootstrap_paired_pvalue(
                        a, b, lower_is_better=lower
                    )

    table_md = build_markdown_table(summary, args.methods)
    table_csv = build_csv(summary, args.methods)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "baseline_table.md").write_text(
        f"# Baseline Evaluation\n\n{table_md}\n\n"
        f"(↓ lower is better, ↑ higher is better; L = LLM-judge. "
        f"Significance vs {HEADLINE} in baseline_table.json.)\n",
        encoding="utf-8",
    )
    (outdir / "baseline_table.csv").write_text(table_csv, encoding="utf-8")
    (outdir / "baseline_table.json").write_text(
        json.dumps({"summary": summary, "significance": significance,
                    "n_prompts": len(prompts), "methods": args.methods},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + table_md)
    print(f"\nWrote baseline_table.{{md,csv,json}} -> {outdir}")
    if significance:
        print(f"\nSignificance (MoP+DivPO beats baseline, p<0.05):")
        for method, metrics in significance.items():
            wins = [m for m, s in metrics.items() if s["mean_diff"] > 0 and s["p_value"] < 0.05]
            print(f"  vs {method}: {len(wins)} metrics — {', '.join(wins) or 'none'}")


if __name__ == "__main__":
    main()
