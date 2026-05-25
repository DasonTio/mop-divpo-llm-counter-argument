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
    from huggingface_hub import upload_folder, repo_exists, create_repo

    # Ensure the target model repo exists. If it doesn't, try to create it
    # using the provided token. Note: create_repo will create under the
    # account associated with the token; if MODEL_REPO points to another
    # username you may need to create it manually on the Hub or update
    # MODEL_REPO to a repo you control.
    try:
        if not repo_exists(MODEL_REPO, repo_type="model", token=token):
            create_repo(repo_id=MODEL_REPO, repo_type="model", token=token, private=False)
            print(f"  Created model repo: {MODEL_REPO}")
    except Exception as e:
        raise RuntimeError(
            f"Model repo {MODEL_REPO} not found and could not be created automatically. "
            "Either create it manually on Hugging Face, or set MODEL_REPO to an existing repo "
            "you control. Also ensure your HF token has model write/create permissions."
        ) from e

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
    """Load persona SFT data from HF Hub as a chat-style Dataset.

    The public dataset repo has historically contained both stale parquet
    exports and current JSONL files. Downloading the JSONL directly avoids
    Hugging Face dataset-card feature casting against old flat schemas.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    from mop_divpo.data.sft_records import load_sft_jsonl_records

    path = hf_hub_download(
        repo_id=SFT_DATA_REPO,
        filename=f"{persona}.jsonl",
        repo_type="dataset",
        token=token or None,
    )
    return Dataset.from_list(load_sft_jsonl_records(path, persona=persona))


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
