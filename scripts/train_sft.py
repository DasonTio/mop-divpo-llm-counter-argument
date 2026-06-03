#!/usr/bin/env python
"""SFT training — one LoRA adapter per persona.

Designed to run on Google Colab (T4/A100). Pulls data from HF Hub,
trains with TRL SFTTrainer, pushes adapter back to HF Hub.

Colab setup (run these first):
    !pip install transformers peft trl accelerate bitsandbytes datasets huggingface_hub
    !pip uninstall -y torchao
    import os; os.environ["HF_TOKEN"] = "hf_xxx"

Usage:
    python scripts/train_sft.py --persona contrarian
    python scripts/train_sft.py --all
    python scripts/train_sft.py --persona contrarian --epochs 3 --lr 2e-4 --batch-size 8
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

# LoRA config per CLAUDE.md: r=16, alpha=32
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train_persona(persona: str, args: argparse.Namespace, token: str) -> None:
    from mop_divpo.training_env import check_torchao_compatibility

    check_torchao_compatibility()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer

    from mop_divpo.hub import MODEL_REPO, load_sft_dataset, push_adapter

    print(f"\n{'='*60}")
    print(f"  Training SFT adapter: {persona}")
    print(f"{'='*60}\n")

    # --- Data ---
    if args.dataset_dir:
        import json as _json
        from datasets import Dataset as _Dataset

        dataset_path = Path(args.dataset_dir) / f"{persona}.jsonl"
        print(f"Loading dataset from local file {dataset_path}...", flush=True)
        _records = [_json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
        ds = _Dataset.from_list(_records)
    else:
        print("Loading dataset from HF Hub...", flush=True)
        ds = load_sft_dataset(persona, token=token)
    print(f"  {len(ds)} training examples")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
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

    # --- Format dataset to text ---
    def format_messages(example):
        msgs = example["messages"]
        # Apply chat template if available, else join with delimiters
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in msgs)
        return {"text": text}

    ds = ds.map(format_messages, remove_columns=ds.column_names)

    # --- Training ---
    output_dir = f"outputs/adapters/{args.output_stage}/{persona}"
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
        "optim": "adamw_torch_fused",
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
    }
    sft_params = inspect.signature(SFTConfig).parameters
    if "max_seq_length" not in sft_params and "max_length" in sft_params:
        sft_kwargs["max_length"] = sft_kwargs.pop("max_seq_length")
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": ds,
    }
    tokenizer_arg = "processing_class"
    if tokenizer_arg not in inspect.signature(SFTTrainer.__init__).parameters:
        tokenizer_arg = "tokenizer"
    trainer_kwargs[tokenizer_arg] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("Training...", flush=True)
    trainer.train()

    # Save locally
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    # Push to HF Hub
    if not args.no_push:
        print(f"Pushing to {MODEL_REPO}/{args.output_stage}/{persona} ...", flush=True)
        url = push_adapter(output_dir, args.output_stage, persona, token)
        print(f"  Pushed → {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-persona SFT LoRA adapters.")
    parser.add_argument("--persona", choices=PERSONA_IDS)
    parser.add_argument("--all", action="store_true", help="Train all personas sequentially.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dataset-dir", default=None,
                        help="Optional local directory containing {persona}.jsonl SFT files.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-stage", default="sft",
                        help="Hub/output stage name, e.g. sft or sft_1p5b.")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var).")
    parser.add_argument("--no-push", action="store_true", help="Skip HF Hub push.")
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
