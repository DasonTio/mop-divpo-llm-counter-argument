#!/usr/bin/env python
"""Push processed datasets and/or trained adapters to HuggingFace Hub.

Usage:
    # Push SFT datasets (after prepare_sft_datasets.py)
    python scripts/push_to_hub.py --sft --token hf_xxx
    python scripts/push_to_hub.py --sft --persona contrarian --token hf_xxx

    # Push DivPO datasets (after prepare_divpo_datasets.py)
    python scripts/push_to_hub.py --divpo --token hf_xxx

    # Push trained adapters
    python scripts/push_to_hub.py --adapters sft --token hf_xxx
    python scripts/push_to_hub.py --adapters divpo --token hf_xxx

Token can also be set via env var: export HF_TOKEN=hf_xxx
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.hub import (
    PERSONA_IDS,
    SFT_DATA_REPO,
    DIVPO_DATA_REPO,
    MODEL_REPO,
    get_token,
    push_adapter,
    push_divpo_file,
    push_sft_file,
)


def push_sft(args: argparse.Namespace, token: str) -> None:
    sft_dir = Path(args.sft_dir)
    personas = [args.persona] if args.persona else PERSONA_IDS
    print(f"Pushing SFT data → {SFT_DATA_REPO}")
    for p in personas:
        path = sft_dir / f"{p}.jsonl"
        if not path.exists():
            print(f"  SKIP {p}: {path} not found — run prepare_sft_datasets.py first")
            continue
        lines = sum(1 for _ in open(path))
        url = push_sft_file(p, path, token)
        print(f"  {p}: {lines} records → {url}")


def push_divpo(args: argparse.Namespace, token: str) -> None:
    divpo_dir = Path(args.divpo_dir)
    personas = [args.persona] if args.persona else PERSONA_IDS
    print(f"Pushing DivPO data → {DIVPO_DATA_REPO}")
    for p in personas:
        path = divpo_dir / f"{p}.jsonl"
        if not path.exists():
            print(f"  SKIP {p}: {path} not found — run prepare_divpo_datasets.py first")
            continue
        lines = sum(1 for _ in open(path))
        url = push_divpo_file(p, path, token)
        print(f"  {p}: {lines} records → {url}")


def push_adapters(stage: str, args: argparse.Namespace, token: str) -> None:
    adapter_base = Path(args.adapter_dir) / stage
    personas = [args.persona] if args.persona else PERSONA_IDS
    print(f"Pushing {stage} adapters → {MODEL_REPO}/{stage}/")
    for p in personas:
        adapter_path = adapter_base / p
        if not adapter_path.exists():
            print(f"  SKIP {p}: {adapter_path} not found")
            continue
        url = push_adapter(adapter_path, stage, p, token)
        print(f"  {p} → {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push data/adapters to HuggingFace Hub.")
    parser.add_argument("--sft", action="store_true", help="Push SFT JSONL datasets.")
    parser.add_argument("--divpo", action="store_true", help="Push DivPO JSONL datasets.")
    parser.add_argument("--adapters", choices=["sft", "divpo"], help="Push trained adapters.")
    parser.add_argument("--persona", choices=PERSONA_IDS, help="Single persona (default: all).")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var).")
    parser.add_argument("--sft-dir", default="data/processed/sft")
    parser.add_argument("--divpo-dir", default="data/processed/divpo")
    parser.add_argument("--adapter-dir", default="outputs/adapters")
    args = parser.parse_args()

    if not any([args.sft, args.divpo, args.adapters]):
        parser.error("Specify at least one of: --sft, --divpo, --adapters sft|divpo")

    token = get_token(args.token)

    if args.sft:
        push_sft(args, token)
    if args.divpo:
        push_divpo(args, token)
    if args.adapters:
        push_adapters(args.adapters, args, token)

    print("\nDone.")


if __name__ == "__main__":
    main()
