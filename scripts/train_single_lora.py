#!/usr/bin/env python
"""Single-LoRA baseline trainer (research-plan.md §3.1, method 3).

Trains ONE LoRA adapter on all four persona SFT datasets merged, to serve as the
"no mixture-of-personas" control in the baseline table. Pushed to
MODEL_REPO/single/all so inference can load it via adapter_stage="single".

Same base model and LoRA config as scripts/train_sft.py. The only difference is
the training set: the union of all persona datasets instead of one per adapter.

Colab/Kaggle/vast.ai setup:
    pip install transformers peft trl accelerate datasets huggingface_hub
    export HF_TOKEN=hf_xxx
    python scripts/train_single_lora.py

Usage:
    python scripts/train_single_lora.py                 # train + push to single/all
    python scripts/train_single_lora.py --no-push       # local only
"""
from __future__ import annotations

import argparse
import inspect
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the merged single-LoRA baseline.")
    p.add_argument("--name", default="all", help="Subfolder name under single/.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--token", default=None)
    p.add_argument("--no-push", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN", "") or args.token or ""
    if not token:
        build_parser().error("HF token required. Set HF_TOKEN env var or pass --token.")

    from mop_divpo.training_env import check_torchao_compatibility

    check_torchao_compatibility()

    import torch
    from datasets import concatenate_datasets
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from mop_divpo.hub import MODEL_REPO, load_sft_dataset, push_adapter

    print(f"\n{'='*60}\n  Training Single-LoRA baseline (merged personas)\n{'='*60}\n")

    print("Loading + merging all persona SFT datasets from HF Hub...", flush=True)
    parts = []
    for persona in PERSONA_IDS:
        ds = load_sft_dataset(persona, token=token)
        print(f"  {persona}: {len(ds)} examples")
        parts.append(ds)
    merged = concatenate_datasets(parts).shuffle(seed=42)
    print(f"  merged: {len(merged)} examples")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", token=token
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

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

    def format_messages(example):
        msgs = example["messages"]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in msgs)
        return {"text": text}

    merged = merged.map(format_messages, remove_columns=merged.column_names)

    output_dir = f"outputs/adapters/single/{args.name}"
    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "fp16": True,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "max_seq_length": args.max_seq_len,
        "dataset_text_field": "text",
        "packing": False,
    }
    sft_params = inspect.signature(SFTConfig).parameters
    if "max_seq_length" not in sft_params and "max_length" in sft_params:
        sft_kwargs["max_length"] = sft_kwargs.pop("max_seq_length")
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {"model": model, "args": sft_config, "train_dataset": merged}
    tokenizer_arg = "processing_class"
    if tokenizer_arg not in inspect.signature(SFTTrainer.__init__).parameters:
        tokenizer_arg = "tokenizer"
    trainer_kwargs[tokenizer_arg] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    print("Training...", flush=True)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    if not args.no_push:
        print(f"Pushing to {MODEL_REPO}/single/{args.name} ...", flush=True)
        url = push_adapter(output_dir, "single", args.name, token)
        print(f"  Pushed → {url}")

    print("\nDone.")


if __name__ == "__main__":
    main()
