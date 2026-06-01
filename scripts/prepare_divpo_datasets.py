#!/usr/bin/env python
"""Prepare DivPO preference datasets for each MoP persona.

Requires trained SFT adapters. Adapters can be local OR pulled from HF Hub.

Usage:
    # Pull adapters from HF Hub (after train_sft.py ran on Colab)
    python scripts/prepare_divpo_datasets.py --all --from-hub

    # Use local adapters
    python scripts/prepare_divpo_datasets.py --all --adapter-dir outputs/adapters/sft

    # Push to HF Hub immediately after generation
    python scripts/prepare_divpo_datasets.py --all --from-hub --push
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.data.writers import write_json, write_jsonl
from mop_divpo.divpo.pairs import select_pair, select_pairs_batch

PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]


def _require_adapters() -> None:
    from mop_divpo.training_env import check_torchao_compatibility

    check_torchao_compatibility()

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
    if from_hub:
        from mop_divpo.hub import MODEL_REPO
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


def _load_model(adapter_path: str, base_model: str, token: str):
    """Load tokenizer + SFT adapter once per persona. Returns (model, tokenizer)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation with padding

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        token=token or None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def _generate_candidates(
    prompt: str,
    system_prompt: str,
    model,
    tokenizer,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """Generate n candidate responses for a prompt. Model loaded externally."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=n,
        )
    return [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True) for i in range(n)]


def _generate_candidates_batch(
    prompts: list[str],
    system_prompt: str,
    model,
    tokenizer,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """Generate n candidates for each prompt in a batch.

    Tokenizes B unique prompts once, then uses num_return_sequences=n.
    Output shape: [B*n]. Ordering: [p0_s0..p0_{n-1}, p1_s0..p1_{n-1}, ...].
    Requires tokenizer.padding_side == "left" (set in _load_model).
    """
    import torch

    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=n,
        )

    # out: [B*n, prompt_len + max_new_tokens]
    responses = [
        tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True)
        for i in range(len(prompts) * n)
    ]
    return [responses[i * n: (i + 1) * n] for i in range(len(prompts))]


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


def run_persona(persona: str, args: argparse.Namespace, token: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer  # type: ignore
    from personas import PERSONAS

    sft_dir = Path(args.prompt_pool)
    output_dir = Path(args.output_dir)
    checks_dir = Path("outputs/data_checks")

    adapter_path = _resolve_adapter(
        persona,
        adapter_dir=Path(args.adapter_dir),
        from_hub=args.from_hub,
        token=token,
    )
    prompts = _load_prompt_pool(persona, sft_dir, max_prompts=args.limit)
    print(f"  {len(prompts)} prompts loaded")

    system_prompt = PERSONAS[persona]["system_prompt"]

    import torch
    torch.backends.cudnn.benchmark = True  # auto-tune kernels for consistent batch shapes

    print("  Loading model...", flush=True)
    model, tokenizer = _load_model(adapter_path, args.base_model, token)
    print("  Model loaded.", flush=True)

    # Place embedder on GPU 1 (frees GPU 0 entirely for the LLM) or GPU 0 if single-GPU
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        embedder_device = "cuda:1"
    elif n_gpu == 1:
        embedder_device = "cuda:0"
    else:
        embedder_device = "cpu"
    print(f"  Embedder device: {embedder_device}", flush=True)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    records: list[dict] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}
    batch_size = args.gen_batch_size

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        if batch_start % max(batch_size, 1) == 0:
            print(f"  [{batch_start}/{len(prompts)}] generating batch of {len(batch_prompts)}...", flush=True)

        # Batched generation — one model.generate() for all prompts × n candidates
        try:
            all_candidates = _generate_candidates_batch(
                batch_prompts,
                system_prompt,
                model,
                tokenizer,
                args.candidate_count,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
            )
        except Exception as exc:
            print(f"  WARNING batch generation failed, falling back to serial: {exc}", flush=True)
            all_candidates = []
            for prompt in batch_prompts:
                try:
                    cands = _generate_candidates(
                        prompt, system_prompt, model, tokenizer,
                        args.candidate_count, args.temperature, args.top_p, args.max_new_tokens,
                    )
                    all_candidates.append(cands)
                except Exception as inner_exc:
                    skipped += 1
                    skip_reasons["generation_error"] = skip_reasons.get("generation_error", 0) + 1
                    print(f"  WARNING serial generation failed: {inner_exc}")
                    all_candidates.append([])

        # Batch scoring — one embedder.encode() call for all prompts + candidates
        valid_prompts = [p for p, c in zip(batch_prompts, all_candidates) if c]
        valid_candidates = [c for c in all_candidates if c]

        if not valid_prompts:
            continue

        pairs = select_pairs_batch(
            prompts=valid_prompts,
            candidates_batch=valid_candidates,
            embedder=embedder,
            min_quality=args.min_quality,
            min_rarity_margin=args.min_rarity_margin,
            quality_weight=args.quality_weight,
            rarity_weight=args.rarity_weight,
            persona=persona,
            candidates_per_prompt=args.candidate_count,
        )

        for pair in pairs:
            if pair is None:
                skipped += 1
                skip_reasons["no_eligible_pair"] = skip_reasons.get("no_eligible_pair", 0) + 1
                continue
            records.append(pair.to_dict())

    # Free VRAM before next persona
    del model
    torch.cuda.empty_cache()

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

    if args.push and records:
        from mop_divpo.hub import push_divpo_file
        print(f"  Pushing {persona} DivPO data to HF Hub...", flush=True)
        url = push_divpo_file(persona, output_path, token)
        print(f"  Pushed → {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DivPO preference datasets.")
    parser.add_argument("--persona", choices=PERSONA_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--adapter-dir", default="outputs/adapters/sft")
    parser.add_argument("--prompt-pool", default="data/processed/sft")
    parser.add_argument("--output-dir", default="data/processed/divpo")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--min-quality", type=float, default=0.35)
    parser.add_argument("--min-rarity-margin", type=float, default=0.0)
    parser.add_argument("--quality-weight", type=float, default=0.4)
    parser.add_argument("--rarity-weight", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--limit", type=int, default=500, help="Max prompts per persona.")
    parser.add_argument("--gen-batch-size", type=int, default=16,
                        help="Prompts per generation batch. Higher = better GPU utilization. "
                             "T4: 16–32, A100: 64–128.")
    parser.add_argument("--from-hub", action="store_true",
                        help="Pull SFT adapters from HF Hub (DasonTio/mop-divpo-coauthor).")
    parser.add_argument("--push", action="store_true",
                        help="Push each persona's DivPO data to HF Hub after generation.")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "") or args.token or ""
    if args.from_hub and not token:
        parser.error("--from-hub requires --token or HF_TOKEN env var.")
    if args.push and not token:
        parser.error("--push requires --token or HF_TOKEN env var.")

    if not args.persona and not args.all:
        parser.error("Provide --persona <name> or --all.")

    _require_adapters()

    personas = PERSONA_IDS if args.all else [args.persona]
    for p in personas:
        print(f"\n=== {p} ===")
        run_persona(p, args, token)

    print("\nDone.")


if __name__ == "__main__":
    main()
