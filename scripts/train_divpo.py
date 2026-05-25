#!/usr/bin/env python
"""DivPO (DPO-based) training — one adapter per persona on top of SFT.

Pulls SFT adapter + DivPO dataset from HF Hub, trains with TRL DPOTrainer,
pushes DivPO adapter back to HF Hub.

Requires SFT training to have been run first (scripts/train_sft.py).
Also requires DivPO preference data (scripts/prepare_divpo_datasets.py +
scripts/push_to_hub.py --divpo).

Colab setup (run these first):
    !pip install transformers peft trl accelerate bitsandbytes datasets huggingface_hub
    import os; os.environ["HF_TOKEN"] = "hf_xxx"

Usage:
    python scripts/train_divpo.py --persona contrarian
    python scripts/train_divpo.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train_persona(persona: str, args: argparse.Namespace, token: str) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from mop_divpo.hub import DIVPO_DATA_REPO, MODEL_REPO, push_adapter

    print(f"\n{'='*60}")
    print(f"  Training DivPO adapter: {persona}")
    print(f"{'='*60}\n")

    # --- Data ---
    print("Loading DivPO dataset from HF Hub...", flush=True)
    ds = load_dataset(
        DIVPO_DATA_REPO,
        data_files={"train": f"{persona}.jsonl"},
        split="train",
        token=token,
    )
    print(f"  {len(ds)} preference pairs")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model: load SFT adapter from HF ---
    sft_adapter_id = f"{MODEL_REPO}"
    sft_subfolder = f"sft/{persona}"
    print(f"Loading SFT adapter from {sft_adapter_id}/{sft_subfolder} ...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    model = PeftModel.from_pretrained(
        base,
        sft_adapter_id,
        subfolder=sft_subfolder,
        token=token,
        is_trainable=True,
    )
    model.config.use_cache = False

    # Reference model (frozen SFT — DPO needs it for KL constraint)
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    ref_model = PeftModel.from_pretrained(
        ref_base,
        sft_adapter_id,
        subfolder=sft_subfolder,
        token=token,
        is_trainable=False,
    )

    # Add new LoRA on top for DivPO stage
    from peft import get_peft_model
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Training ---
    output_dir = f"outputs/adapters/divpo/{persona}"
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    print("Training...", flush=True)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    if not args.no_push:
        print(f"Pushing to {MODEL_REPO}/divpo/{persona} ...", flush=True)
        url = push_adapter(output_dir, "divpo", persona, token)
        print(f"  Pushed → {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-persona DivPO adapters.")
    parser.add_argument("--persona", choices=PERSONA_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    if not args.persona and not args.all:
        parser.error("Specify --persona <name> or --all.")

    token = os.environ.get("HF_TOKEN", "") or args.token or ""
    if not token:
        parser.error("HF token required. Set HF_TOKEN env var or pass --token.")

    personas = PERSONA_IDS if args.all else [args.persona]
    for p in personas:
        train_persona(p, args, token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
