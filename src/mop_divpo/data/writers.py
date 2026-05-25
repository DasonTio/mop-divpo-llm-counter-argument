import json
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_examples_md(persona: str, records: list[dict], path: str | Path, n: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"\n\n# SFT Examples — {persona}\n\n"]
    for i, rec in enumerate(records[:n], 1):
        msgs = rec.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        lines.append(f"## Example {i}\n\n")
        lines.append(f"**System:**\n```\n{system}\n```\n\n")
        lines.append(f"**User:**\n```\n{user}\n```\n\n")
        lines.append(f"**Assistant:**\n```\n{assistant}\n```\n\n")
    mode = "a" if path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write("".join(lines))
