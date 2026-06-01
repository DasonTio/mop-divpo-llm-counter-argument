"""LLM-as-judge pipeline (docs/research-plan.md §2.5).

Runs the four rubrics over generated outputs, parses structured JSON, and
caches every call so re-running the evaluation is free after the first pass.

The API client is injected as `call_fn(prompt: str) -> str`. Tests pass a fake;
production passes `make_anthropic_caller(...)` or `make_openai_caller(...)`.
Both production callers use temperature=0 for determinism (a precondition for
the cache to be sound).

Persona descriptions come from `personas.PERSONAS[p]["description"]` (the plan
text references `PERSONA_ROUTING_DESCRIPTIONS`, which does not exist in this
codebase; `description` is the equivalent one-line cognitive-style summary).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mop_divpo.eval.llm_judge_rubrics import RUBRICS, build_rubric_prompt

CallFn = Callable[[str], str]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object out of a model response.

    Tolerates markdown code fences and leading/trailing prose. Raises
    ValueError if no parseable object is found.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from judge response: {text!r}") from exc
    raise ValueError(f"No JSON object found in judge response: {text!r}")


def score_keys_for(rubric_name: str) -> list[str]:
    """The 1-5 integer keys a rubric contributes to the aggregate table."""
    return list(RUBRICS[rubric_name]["score_keys"])


def _cache_key(rubric_name: str, rendered_prompt: str) -> str:
    digest = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
    return f"{rubric_name}:{digest}"


class JudgeCache:
    """File-backed JSON cache keyed by (rubric_name, rendered prompt hash).

    The rendered prompt already folds in the topic, output text, persona
    description, and peers, so hashing it satisfies the plan's
    (rubric_name, prompt, output_hash) key requirement in one shot.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._store: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self._store = json.load(f)

    def get(self, rubric_name: str, rendered_prompt: str) -> dict[str, Any] | None:
        return self._store.get(_cache_key(rubric_name, rendered_prompt))

    def set(self, rubric_name: str, rendered_prompt: str, value: dict[str, Any]) -> None:
        self._store[_cache_key(rubric_name, rendered_prompt)] = value

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._store)


def judge_output(
    rubric_name: str,
    *,
    output: str,
    call_fn: CallFn,
    prompt: str | None = None,
    persona_description: str | None = None,
    peers: list[str] | None = None,
    cache: JudgeCache | None = None,
) -> dict[str, Any]:
    """Run one rubric on one output. Returns the parsed JSON dict.

    Uses the cache when supplied; only calls `call_fn` on a miss.
    """
    rendered = build_rubric_prompt(
        rubric_name,
        output=output,
        prompt=prompt,
        persona_description=persona_description,
        peers=peers,
    )
    if cache is not None:
        hit = cache.get(rubric_name, rendered)
        if hit is not None:
            return hit
    parsed = extract_json(call_fn(rendered))
    if cache is not None:
        cache.set(rubric_name, rendered, parsed)
    return parsed


def _persona_description(persona: str | None) -> str | None:
    if persona is None:
        return None
    from personas import PERSONAS

    return PERSONAS[persona]["description"]


def _peers_for(record: dict[str, Any], grouped: dict[tuple, list[dict]], limit: int = 3) -> list[str]:
    """Other outputs sharing this record's (method, prompt), capped at `limit`."""
    group = grouped[(record["method"], record["prompt"])]
    peers = [r["output"] for r in group if r["output_id"] != record["output_id"]]
    return peers[:limit]


def score_all_outputs(
    outputs: list[dict[str, Any]],
    *,
    call_fn: CallFn,
    cache: JudgeCache | None = None,
    output_path: str | Path | None = None,
    rubrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all rubrics on all outputs. Returns scored records.

    Each input record must have: output_id, method, prompt, persona, output.
    `persona_fidelity` is skipped (set to None) for records with persona=None
    (the Base method has no target cognitive style).
    Writes JSONL to `output_path` if given, and persists the cache.
    """
    rubrics = rubrics or list(RUBRICS)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in outputs:
        grouped[(record["method"], record["prompt"])].append(record)

    scored: list[dict[str, Any]] = []
    for record in outputs:
        persona = record.get("persona")
        result: dict[str, Any] = {
            "output_id": record["output_id"],
            "method": record["method"],
            "prompt": record["prompt"],
            "persona": persona,
            "output": record["output"],
        }
        for rubric_name in rubrics:
            if rubric_name == "persona_fidelity" and persona is None:
                result[rubric_name] = None
                continue
            result[rubric_name] = judge_output(
                rubric_name,
                output=record["output"],
                call_fn=call_fn,
                prompt=record.get("prompt"),
                persona_description=_persona_description(persona),
                peers=_peers_for(record, grouped),
                cache=cache,
            )
        scored.append(result)

    if cache is not None:
        cache.save()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in scored:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return scored


def make_anthropic_caller(model: str = "claude-sonnet-4-20250514", max_tokens: int = 512) -> CallFn:
    """Build a temperature=0 Anthropic judge caller."""
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system="You are a precise evaluator. Return only valid JSON, no prose.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    return call


def make_openai_caller(model: str = "gpt-4o", max_tokens: int = 512) -> CallFn:
    """Build a temperature=0 OpenAI judge caller."""
    from openai import OpenAI

    client = OpenAI()

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Return only valid JSON, no prose."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return call
