"""Utilities for loading SFT JSONL records without relying on HF dataset cards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .persona_prompts import PERSONA_SYSTEM_PROMPTS


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("SFT record field 'messages' must be a list")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object with role/content")
        role = _as_text(message.get("role"))
        content = _as_text(message.get("content"))
        if not role or not content:
            raise ValueError("Each message must contain non-empty role and content")
        normalized.append({"role": role, "content": content})

    roles = [message["role"] for message in normalized]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("SFT messages must include at least one user and assistant turn")
    return normalized


def _metadata(record: dict[str, Any], persona: str | None) -> dict[str, str]:
    raw = record.get("metadata")
    if isinstance(raw, dict):
        metadata = {str(k): _as_text(v) for k, v in raw.items() if _as_text(v)}
    else:
        metadata = {}

    resolved_persona = persona or _as_text(record.get("persona")) or metadata.get("persona")
    source = _as_text(record.get("source")) or metadata.get("source", "")
    source_id = (
        _as_text(record.get("source_id"))
        or _as_text(record.get("id"))
        or metadata.get("source_id", "")
    )

    if resolved_persona:
        metadata["persona"] = resolved_persona
    if source:
        metadata["source"] = source
    if source_id:
        metadata["source_id"] = source_id
    return metadata


def normalize_sft_record(record: dict[str, Any], persona: str | None = None) -> dict:
    """Return a canonical chat-style SFT record.

    Current datasets use ``messages``. Legacy Hub parquet/JSONL rows used flat
    ``prompt`` and ``response`` columns. Supporting both keeps Colab training
    independent from stale Hugging Face dataset-card schemas and caches.
    """
    if "messages" in record:
        messages = _normalize_messages(record["messages"])
    else:
        resolved_persona = persona or _as_text(record.get("persona"))
        if resolved_persona not in PERSONA_SYSTEM_PROMPTS:
            raise ValueError("Legacy flat SFT records require a known persona")

        prompt = _as_text(record.get("prompt"))
        response = _as_text(record.get("response"))
        if not prompt or not response:
            raise ValueError("SFT record must contain messages or prompt/response")
        messages = [
            {"role": "system", "content": PERSONA_SYSTEM_PROMPTS[resolved_persona]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    return {"messages": messages, "metadata": _metadata(record, persona)}


def load_sft_jsonl_records(path: str | Path, persona: str | None = None) -> list[dict]:
    """Load and normalize a local SFT JSONL file."""
    records: list[dict] = []
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"SFT row in {path} line {line_number} must be an object")
            try:
                records.append(normalize_sft_record(raw, persona=persona))
            except ValueError as exc:
                raise ValueError(f"Invalid SFT row in {path} line {line_number}: {exc}") from exc

    if not records:
        raise ValueError(f"No SFT records found in {path}")
    return records
