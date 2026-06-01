#!/usr/bin/env python
"""DivPO (DPO-based) training — one adapter per persona on top of SFT.

Pulls SFT adapter + DivPO dataset from HF Hub, trains with TRL DPOTrainer,
pushes DivPO adapter back to HF Hub.

Requires SFT training to have been run first (scripts/train_sft.py).
Also requires DivPO preference data (scripts/prepare_divpo_datasets.py +
scripts/push_to_hub.py --divpo).

Colab setup (run these first):
    !pip install transformers peft trl accelerate bitsandbytes datasets huggingface_hub
    !pip uninstall -y torchao
    import os; os.environ["HF_TOKEN"] = "hf_xxx"

Usage:
    python scripts/train_divpo.py --persona contrarian
    python scripts/train_divpo.py --all
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import warnings
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


def split_supported_kwargs(callable_obj: object, kwargs: dict) -> tuple[dict, dict]:
    """Split kwargs by whether callable_obj's signature accepts them."""
    params = inspect.signature(callable_obj).parameters
    accepts_any = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_any:
        return dict(kwargs), {}
    supported = {k: v for k, v in kwargs.items() if k in params}
    unsupported = {k: v for k, v in kwargs.items() if k not in params}
    return supported, unsupported


def build_dpo_config(DPOConfig: object, config_kwargs: dict):
    """Create a DPOConfig while preserving kwargs needed by older DPOTrainer APIs."""
    supported, trainer_candidates = split_supported_kwargs(DPOConfig, config_kwargs)
    return DPOConfig(**supported), trainer_candidates


def add_supported_trainer_kwargs(
    trainer_kwargs: dict,
    DPOTrainer: object,
    trainer_candidates: dict,
) -> None:
    """Forward DPO kwargs to DPOTrainer when the installed TRL expects them there."""
    supported, unsupported = split_supported_kwargs(DPOTrainer.__init__, trainer_candidates)
    trainer_kwargs.update(supported)
    optional_unsupported = {"group_by_length", "max_prompt_length"}
    unsupported = {
        k: v for k, v in unsupported.items() if k not in optional_unsupported
    }
    if unsupported:
        ignored = ", ".join(sorted(unsupported))
        warnings.warn(
            f"Installed TRL accepts these DivPO settings in neither DPOConfig nor "
            f"DPOTrainer: {ignored}. Training will continue with TRL defaults.",
            RuntimeWarning,
        )


def train_persona(persona: str, args: argparse.Namespace, token: str) -> None:
    from mop_divpo.training_env import check_torchao_compatibility

    check_torchao_compatibility()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from mop_divpo.hub import DIVPO_DATA_REPO, MODEL_REPO, push_adapter

    print(f"\n{'='*60}")
    print(f"  Training DivPO adapter: {persona}")
    print(f"{'='*60}\n")

    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    print(f"  GPUs available: {n_gpu}", flush=True)

    torch.backends.cudnn.benchmark = True

    # Under accelerate DDP (WORLD_SIZE > 1), do NOT set device_map — accelerate owns placement.
    # Single-process: put both models on cuda:0 (both fit at ~1GB each, no cross-device TRL issues).
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    device_kwargs: dict = {} if use_ddp else {"device_map": {"": 0}}

    # --- Data ---
    if args.dataset_dir:
        dataset_path = Path(args.dataset_dir) / f"{persona}.jsonl"
        print(f"Loading DivPO dataset from local file {dataset_path}...", flush=True)
        # Use Dataset.from_list to avoid load_dataset contacting HF Hub for the
        # "json" builder version — that network check hangs on restricted networks.
        import json as _json
        from datasets import Dataset as _Dataset
        _records = [_json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
        ds = _Dataset.from_list(_records)
    else:
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
        dtype=torch.float16,
        attn_implementation="sdpa",
        **device_kwargs,
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
    model.enable_input_require_grads()

    # Reference model (frozen SFT — DPO needs it for KL constraint)
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        attn_implementation="sdpa",
        **device_kwargs,
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
    output_dir = f"outputs/adapters/{args.output_stage}/{persona}"
    dpo_config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1,
        "weight_decay": 0.01,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "beta": 0.1,
        "max_prompt_length": 256,
        "max_length": args.max_length,
        # --- performance ---
        "optim": "adamw_torch_fused",       # fused kernel: ~10% faster than default AdamW
        "group_by_length": True,            # batch similar-length seqs -> less padding waste
        "gradient_checkpointing": True,     # trade compute for memory -> larger batch fits
        "dataloader_num_workers": 0,        # 0 = main thread; >0 deadlocks DDP on Kaggle
        "remove_unused_columns": False,     # DPO needs all columns
    }
    dpo_config, trainer_config_kwargs = build_dpo_config(DPOConfig, dpo_config_kwargs)

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_config,
        "train_dataset": ds,
    }
    add_supported_trainer_kwargs(trainer_kwargs, DPOTrainer, trainer_config_kwargs)
    tokenizer_arg = "processing_class"
    if tokenizer_arg not in inspect.signature(DPOTrainer.__init__).parameters:
        tokenizer_arg = "tokenizer"
    trainer_kwargs[tokenizer_arg] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    print("Training...", flush=True)
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Adapter saved to {output_dir}")

    if not args.no_push:
        print(f"Pushing to {MODEL_REPO}/{args.output_stage}/{persona} ...", flush=True)
        url = push_adapter(output_dir, args.output_stage, persona, token)
        print(f"  Pushed → {url}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train per-persona DivPO adapters.")
    parser.add_argument("--persona", choices=PERSONA_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--dataset-dir", default=None,
                        help="Local directory containing {persona}.jsonl DivPO files.")
    parser.add_argument("--output-stage", default="divpo",
                        help="Hub/output stage name, e.g. divpo or divpo_v2.")
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-push", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
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
