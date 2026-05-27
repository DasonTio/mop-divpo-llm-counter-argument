#!/usr/bin/env python
"""Pre-evaluation validation for trained DivPO adapters.

This is a qualitative gate before running full evaluation. It verifies that
adapters load, generation uses the persona chat format, and outputs satisfy a
minimal counter-argument sanity check.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REPO = "DasonTio/mop-divpo-coauthor"
PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]
DEFAULT_CLAIMS = [
    "Remote work is strictly better for productivity.",
    "AI tutors will improve education for every student.",
    "Cities should prioritize car infrastructure to reduce congestion.",
]

COUNTER_SIGNALS = {
    "but",
    "however",
    "although",
    "yet",
    "ignores",
    "assumes",
    "assumption",
    "overlooks",
    "undermines",
    "not",
    "isn't",
    "cannot",
    "cost",
    "trade-off",
    "constraint",
}


def configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Already found a `peft_config` attribute in the model\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are trying to modify a model with PEFT for a second time\.",
        category=UserWarning,
    )


def build_messages(persona: str, claim: str, personas: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": personas[persona]["system_prompt"]},
        {
            "role": "user",
            "content": f"Generate a counter-argument to this claim:\n\n{claim}",
        },
    ]


def score_generation(claim: str, text: str) -> dict[str, Any]:
    words = text.split()
    lowered = text.lower()
    has_counter_signal = any(signal in lowered for signal in COUNTER_SIGNALS)
    is_long_enough = len(words) >= 10
    is_not_empty = bool(text.strip())
    is_not_truncated = text.rstrip().endswith((".", "!", "?", '"', "'"))

    issues: list[str] = []
    if not is_not_empty:
        issues.append("empty output")
    if not is_long_enough:
        issues.append("too short")
    if not has_counter_signal:
        issues.append("missing counter-argument signal")
    if not is_not_truncated:
        issues.append("possibly truncated")

    return {
        "claim": claim,
        "word_count": len(words),
        "has_counter_signal": has_counter_signal,
        "is_long_enough": is_long_enough,
        "is_not_truncated": is_not_truncated,
        "passes_gate": not issues,
        "issues": issues,
    }


def resolve_adapter(
    persona: str,
    *,
    local_adapters: dict[str, Path],
    hub_files: set[str],
    model_repo: str,
) -> dict[str, Any]:
    local_path = local_adapters.get(persona)
    if local_path is not None:
        return {"source": str(local_path), "kwargs": {}}

    subfolder = f"divpo/{persona}"
    if f"{subfolder}/adapter_config.json" in hub_files:
        return {"source": model_repo, "kwargs": {"subfolder": subfolder}}

    raise FileNotFoundError(
        f"No DivPO adapter found for {persona}. Expected local adapter or "
        f"{model_repo}/{subfolder}/adapter_config.json"
    )


def build_sft_adapter_kwargs(persona: str, model_repo: str, token: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"subfolder": f"sft/{persona}"}
    if token:
        kwargs["token"] = token
    return {"source": model_repo, "kwargs": kwargs}


def existing_local_adapters(local_root: Path) -> dict[str, Path]:
    return {
        persona: local_root / persona
        for persona in PERSONA_IDS
        if (local_root / persona / "adapter_config.json").exists()
    }


def load_hub_files(model_repo: str, token: str | None) -> set[str]:
    from huggingface_hub import list_repo_files

    try:
        return set(list_repo_files(model_repo, repo_type="model", token=token))
    except Exception as exc:
        print(f"Warning: could not list Hub files for {model_repo}: {exc}", file=sys.stderr)
        return set()


def generate_for_persona(
    persona: str,
    claims: list[str],
    args: argparse.Namespace,
    personas: dict[str, dict[str, str]],
    adapter: dict[str, Any],
    token: str | None,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        token=token,
    )
    adapter_kwargs = dict(adapter["kwargs"])
    if adapter["source"] == args.model_repo and token:
        adapter_kwargs["token"] = token
    sft_adapter = build_sft_adapter_kwargs(persona, args.model_repo, token)
    model = PeftModel.from_pretrained(base, sft_adapter["source"], **sft_adapter["kwargs"])
    model = PeftModel.from_pretrained(model, adapter["source"], **adapter_kwargs)
    model.eval()

    records: list[dict[str, Any]] = []
    for claim in claims:
        messages = build_messages(persona, claim, personas)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        gate = score_generation(claim, generated)
        records.append(
            {
                "persona": persona,
                "claim": claim,
                "adapter_source": adapter["source"],
                "adapter_kwargs": adapter["kwargs"],
                "output": generated,
                "gate": gate,
            }
        )

    del model
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DivPO inference before evaluation.")
    parser.add_argument("--persona", choices=PERSONA_IDS, action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue even if some requested adapters are unavailable.",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--local-root", default="outputs/adapters/divpo")
    parser.add_argument("--output", default="outputs/evaluation/pre_eval_validation.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--claim", action="append", help="Claim to test. Can be repeated.")
    parser.add_argument("--token", default=None)
    return parser


def main() -> None:
    configure_warning_filters()
    args = build_parser().parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or None

    from personas import PERSONAS

    requested_personas = PERSONA_IDS if args.all or not args.persona else args.persona
    claims = args.claim or DEFAULT_CLAIMS
    local_adapters = existing_local_adapters(Path(args.local_root))
    hub_files = load_hub_files(args.model_repo, token)

    adapters: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    missing_reasons: list[str] = []
    for persona in requested_personas:
        try:
            adapters[persona] = resolve_adapter(
                persona,
                local_adapters=local_adapters,
                hub_files=hub_files,
                model_repo=args.model_repo,
            )
        except FileNotFoundError as exc:
            missing.append(persona)
            missing_reasons.append(str(exc))

    if not adapters:
        raise RuntimeError("No requested DivPO adapters are available:\n" + "\n".join(missing_reasons))

    if missing:
        available = ", ".join(sorted(adapters))
        missing_list = ", ".join(sorted(missing))
        details = "\n".join(f"  - {item}" for item in missing_reasons)
        message = (
            "Missing DivPO adapters for: "
            f"{missing_list}\n"
            f"Available adapters: {available}\n"
            "Either upload/sync the missing adapters, or re-run with --allow-missing "
            "or --persona to select a subset.\n"
            "Details:\n"
            f"{details}"
        )
        if not args.allow_missing:
            raise RuntimeError(message)
        print("Skipping unavailable adapters:")
        print(message)

    all_records: list[dict[str, Any]] = []
    for persona, adapter in adapters.items():
        print(f"\nValidating {persona} from {adapter['source']} ...", flush=True)
        all_records.extend(generate_for_persona(persona, claims, args, PERSONAS, adapter, token))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    passed = sum(1 for record in all_records if record["gate"]["passes_gate"])
    total = len(all_records)
    print(f"\nPre-evaluation validation: {passed}/{total} generations passed heuristic gate")
    print(f"Personas validated: {', '.join(sorted(adapters))}")
    print(f"Wrote: {output_path}")
    for record in all_records:
        status = "PASS" if record["gate"]["passes_gate"] else "FAIL"
        issues = ", ".join(record["gate"]["issues"]) or "none"
        print(f"[{status}] {record['persona']} | {record['claim']} | issues: {issues}")


if __name__ == "__main__":
    main()
