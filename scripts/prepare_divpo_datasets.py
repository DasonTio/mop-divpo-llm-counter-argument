#!/usr/bin/env python
"""Prepare DivPO preference datasets for each MoP persona.

Requires trained SFT adapters. Adapters can be local OR pulled from HF Hub.

Usage:
    # Pull adapters from HF Hub (after train_sft.py ran on Colab)
    python scripts/prepare_divpo_datasets.py --all --from-hub --token hf_xxx

    # Use local adapters
    python scripts/prepare_divpo_datasets.py --all --adapter-dir outputs/adapters/sft
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.data.writers import write_json, write_jsonl
from mop_divpo.divpo.pairs import select_pair

PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]


def _require_adapters() -> None:
    try:
        import peft  # noqa: F401
    except ImportError:
        print("ERROR: peft not installed. Run: pip install peft transformers accelerate torch")
        sys.exit(1)


def _load_prompt_pool(persona: str, sft_dir: Path, max_prompts: int) -> list[str]:
    sft_path = sft_dir / f"{persona}.jsonl"
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT data not found: {sft_path}\n"
            "Run prepare_sft_datasets.py first."
        )
    prompts: list[str] = []
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            try:
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                user_text = next((m["content"] for m in msgs if m["role"] == "user"), None)
                if user_text:
                    prompts.append(user_text)
            except json.JSONDecodeError:
                continue
    return prompts


def _resolve_adapter(
    persona: str,
    adapter_dir: Path,
    from_hub: bool,
    token: str,
) -> str:
    """Return local path or HF Hub repo+subfolder string for the SFT adapter."""
    if from_hub:
        from mop_divpo.hub import MODEL_REPO
        # huggingface_hub snapshot_download pulls the subfolder locally
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            allow_patterns=f"sft/{persona}/*",
            token=token or None,
        )
        path = Path(local) / "sft" / persona
        if not path.exists():
            raise FileNotFoundError(
                f"sft/{persona} not found in {MODEL_REPO}. "
                "Run train_sft.py on Colab and push first."
            )
        return str(path)
    else:
        adapter_path = adapter_dir / persona
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"SFT adapter not found: {adapter_path}\n"
                "Run scripts/train_sft.py or use --from-hub."
            )
        return str(adapter_path)


def _generate_candidates(
    prompt: str,
    adapter_path: str,
    base_model: str,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    candidates: list[str] = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        candidates.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return candidates


def _write_divpo_examples(persona: str, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"\n\n# DivPO Examples — {persona}\n\n"]
    for i, rec in enumerate(records[:3], 1):
        meta = rec.get("metadata", {})
        lines.append(f"## Pair {i}\n\n")
        lines.append(f"**Prompt:**\n```\n{rec['prompt']}\n```\n\n")
        lines.append(f"**Chosen:**\n```\n{rec['chosen']}\n```\n\n")
        lines.append(f"**Rejected:**\n```\n{rec['rejected']}\n```\n\n")
        lines.append(
            f"*chosen_quality={meta.get('chosen_quality')}, "
            f"chosen_rarity={meta.get('chosen_rarity')}, "
            f"rejected_quality={meta.get('rejected_quality')}, "
            f"rejected_rarity={meta.get('rejected_rarity')}*\n\n"
        )
    mode = "a" if path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write("".join(lines))


def run_persona(persona: str, args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer  # type: ignore

    sft_dir = Path(args.prompt_pool)
    output_dir = Path(args.output_dir)
    checks_dir = Path("outputs/data_checks")
    token = getattr(args, "token", "") or ""

    adapter_path = _resolve_adapter(
        persona,
        adapter_dir=Path(args.adapter_dir),
        from_hub=args.from_hub,
        token=token,
    )
    prompts = _load_prompt_pool(persona, sft_dir, max_prompts=args.limit)
    print(f"  {len(prompts)} prompts loaded")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    records: list[dict] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"  [{i}/{len(prompts)}] generating...", flush=True)

        try:
            candidates = _generate_candidates(
                prompt,
                adapter_path,
                args.base_model,
                args.candidate_count,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
            )
        except Exception as exc:
            skipped += 1
            skip_reasons["generation_error"] = skip_reasons.get("generation_error", 0) + 1
            print(f"  WARNING generation failed: {exc}")
            continue

        pair = select_pair(
            prompt=prompt,
            candidates=candidates,
            embedder=embedder,
            min_quality=args.min_quality,
            quality_weight=args.quality_weight,
            rarity_weight=args.rarity_weight,
            persona=persona,
            candidates_per_prompt=args.candidate_count,
        )
        if pair is None:
            skipped += 1
            skip_reasons["no_eligible_pair"] = skip_reasons.get("no_eligible_pair", 0) + 1
            continue

        records.append(pair.to_dict())

    output_path = output_dir / f"{persona}.jsonl"
    write_jsonl(records, output_path)
    print(f"  Wrote {len(records)} DivPO pairs → {output_path}")

    avg_cq = round(sum(r["metadata"]["chosen_quality"] for r in records) / max(len(records), 1), 4)
    avg_cr = round(sum(r["metadata"]["chosen_rarity"] for r in records) / max(len(records), 1), 4)
    summary = {
        "persona": persona,
        "total_pairs": len(records),
        "total_skipped": skipped,
        "skip_reasons": skip_reasons,
        "avg_chosen_quality": avg_cq,
        "avg_chosen_rarity": avg_cr,
    }
    write_json(summary, checks_dir / f"divpo_summary_{persona}.json")

    if records:
        _write_divpo_examples(persona, records, checks_dir / "divpo_examples.md")


def main() -> None:
    _require_adapters()

    parser = argparse.ArgumentParser(description="Prepare DivPO preference datasets.")
    parser.add_argument("--persona", choices=PERSONA_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--adapter-dir", default="outputs/adapters/sft")
    parser.add_argument("--prompt-pool", default="data/processed/sft")
    parser.add_argument("--output-dir", default="data/processed/divpo")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--min-quality", type=float, default=0.35)
    parser.add_argument("--quality-weight", type=float, default=0.4)
    parser.add_argument("--rarity-weight", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--limit", type=int, default=1000, help="Max prompts per persona.")
    parser.add_argument("--from-hub", action="store_true",
                        help="Pull SFT adapters from HF Hub (DasonTio/mop-divpo-coauthor).")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var).")
    args = parser.parse_args()
    if args.from_hub and not (args.token or __import__("os").environ.get("HF_TOKEN")):
        parser.error("--from-hub requires --token or HF_TOKEN env var.")

    if not args.persona and not args.all:
        parser.error("Provide --persona <name> or --all.")

    personas = PERSONA_IDS if args.all else [args.persona]
    for p in personas:
        print(f"\n=== {p} ===")
        run_persona(p, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
