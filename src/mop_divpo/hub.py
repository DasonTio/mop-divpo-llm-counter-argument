"""HuggingFace Hub integration — push/pull datasets and adapters."""
from __future__ import annotations

import os
from pathlib import Path

HF_TOKEN_ENV = "HF_TOKEN"
SFT_DATA_REPO = "DasonTio/mop-divpo-sft-data"
DIVPO_DATA_REPO = "DasonTio/mop-divpo-divpo-data"
MODEL_REPO = "DasonTio/mop-divpo-coauthor"
PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]


def get_token(token: str | None = None) -> str:
    t = token or os.environ.get(HF_TOKEN_ENV, "")
    if not t:
        raise ValueError(
            f"HF token not found. Set env var {HF_TOKEN_ENV} or pass --token."
        )
    return t


def login(token: str | None = None) -> None:
    from huggingface_hub import login as hf_login
    hf_login(token=get_token(token))


def ensure_dataset_repo(repo_id: str, token: str) -> None:
    from huggingface_hub import create_repo, repo_exists
    if not repo_exists(repo_id, repo_type="dataset", token=token):
        create_repo(repo_id, repo_type="dataset", token=token, private=False)
        print(f"  Created dataset repo: {repo_id}")


def push_sft_file(persona: str, jsonl_path: str | Path, token: str) -> str:
    """Upload one persona JSONL to the SFT dataset repo. Returns the URL."""
    from huggingface_hub import upload_file

    ensure_dataset_repo(SFT_DATA_REPO, token)
    url = upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=f"{persona}.jsonl",
        repo_id=SFT_DATA_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Add SFT data for {persona}",
    )
    return str(url)


def push_divpo_file(persona: str, jsonl_path: str | Path, token: str) -> str:
    """Upload one persona JSONL to the DivPO dataset repo."""
    from huggingface_hub import upload_file

    ensure_dataset_repo(DIVPO_DATA_REPO, token)
    url = upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=f"{persona}.jsonl",
        repo_id=DIVPO_DATA_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Add DivPO data for {persona}",
    )
    return str(url)


def push_adapter(adapter_dir: str | Path, stage: str, persona: str, token: str) -> str:
    """Upload a trained adapter folder to MODEL_REPO under {stage}/{persona}/."""
    from huggingface_hub import upload_folder

    url = upload_folder(
        folder_path=str(adapter_dir),
        repo_id=MODEL_REPO,
        path_in_repo=f"{stage}/{persona}",
        repo_type="model",
        token=token,
        commit_message=f"Add {stage} adapter for {persona}",
    )
    return str(url)


def load_sft_dataset(persona: str, token: str | None = None):
    """Load persona SFT data from HF hub as a HuggingFace Dataset."""
    from datasets import load_dataset

    kwargs = {}
    if token:
        kwargs["token"] = token
    return load_dataset(
        SFT_DATA_REPO,
        data_files={"train": f"{persona}.jsonl"},
        split="train",
        **kwargs,
    )


def load_divpo_dataset(persona: str, token: str | None = None):
    """Load persona DivPO data from HF hub."""
    from datasets import load_dataset

    kwargs = {}
    if token:
        kwargs["token"] = token
    return load_dataset(
        DIVPO_DATA_REPO,
        data_files={"train": f"{persona}.jsonl"},
        split="train",
        **kwargs,
    )
