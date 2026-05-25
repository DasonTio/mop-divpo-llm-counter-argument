"""SFT dataset extractors — one function per persona."""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Optional

# StackExchange prompts embed the question between these markers
_SE_Q_RE = re.compile(r"Question:\n(.*?)\n\nNow provide", re.DOTALL)

from .filters import (
    FilterStats,
    count_words,
    has_deleted_marker,
    is_empty,
    is_formula_heavy,
    is_mostly_code,
    is_mostly_links,
    is_too_long,
    is_too_short,
)
from .persona_prompts import PERSONA_SYSTEM_PROMPTS, build_user_prompt
from .schema import SFTRecord
from .writers import append_examples_md, write_json, write_jsonl


def _record(persona: str, input_kwargs: dict, response: str, source: str, source_id: str) -> dict:
    return SFTRecord(
        messages=[
            {"role": "system", "content": PERSONA_SYSTEM_PROMPTS[persona]},
            {"role": "user", "content": build_user_prompt(persona, **input_kwargs)},
            {"role": "assistant", "content": response},
        ],
        metadata={"persona": persona, "source": source, "source_id": source_id},
    ).to_dict()


def _save(
    records: list[dict],
    stats: FilterStats,
    persona: str,
    output_path: str | Path,
    stats_path: Optional[str | Path],
    examples_path: Optional[str | Path],
) -> None:
    write_jsonl(records, output_path)
    print(f"  Wrote {len(records)} records → {output_path}")

    if stats_path:
        summary = _summary(records, stats, persona)
        write_json(summary, stats_path)
        print(f"  Stats → {stats_path}")

    if examples_path and records:
        append_examples_md(persona, records, examples_path)
        print(f"  Examples → {examples_path}")


def _summary(records: list[dict], stats: FilterStats, persona: str) -> dict:
    prompt_words, resp_words = [], []
    for rec in records:
        msgs = rec.get("messages", [])
        u = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt_words.append(count_words(u))
        resp_words.append(count_words(a))
    return {
        "persona": persona,
        "total_records": len(records),
        "filter_stats": stats.to_dict(),
        "avg_prompt_words": round(statistics.mean(prompt_words), 1) if prompt_words else 0,
        "avg_response_words": round(statistics.mean(resp_words), 1) if resp_words else 0,
    }


# ---------------------------------------------------------------------------
# Contrarian — CGA-CMV via ConvoKit
# ---------------------------------------------------------------------------

def prepare_contrarian(
    output_path: str | Path,
    limit: int = 5000,
    stats_path: Optional[str | Path] = None,
    examples_path: Optional[str | Path] = None,
) -> tuple[list[dict], FilterStats]:
    from convokit import Corpus, download  # type: ignore

    print("Loading CGA-CMV corpus...", flush=True)
    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))

    records: list[dict] = []
    stats = FilterStats()
    MIN_CLAIM, MAX_CLAIM = 20, 300
    MIN_RESP, MAX_RESP = 30, 250

    for convo in corpus.iter_conversations():
        if len(records) >= limit:
            break

        root = None
        for utt in convo.iter_utterances():
            if utt.reply_to is None:
                root = utt
                break
        if root is None:
            continue

        claim = (root.text or "").strip()

        # Claim-level validation (one check per convo, not recorded in stats)
        if (
            is_empty(claim)
            or has_deleted_marker(claim)
            or is_too_short(claim, MIN_CLAIM)
            or is_too_long(claim, MAX_CLAIM)
        ):
            continue

        for utt in convo.iter_utterances():
            if len(records) >= limit:
                break
            if utt.reply_to != root.id:
                continue

            response = (utt.text or "").strip()

            if is_empty(response):
                stats.record("empty_response")
                continue
            if has_deleted_marker(response):
                stats.record("deleted_marker")
                continue
            if is_too_short(response, MIN_RESP):
                stats.record("response_too_short")
                continue
            if is_too_long(response, MAX_RESP):
                stats.record("response_too_long")
                continue
            if is_mostly_links(response):
                stats.record("mostly_links")
                continue

            records.append(_record("contrarian", {"claim": claim}, response,
                                   "conversations-gone-awry-cmv-corpus", utt.id))
            stats.record(None)

    _save(records, stats, "contrarian", output_path, stats_path, examples_path)
    return records, stats


# ---------------------------------------------------------------------------
# Systems Thinker — PrimeIntellect/stackexchange-question-answering
# ---------------------------------------------------------------------------

def prepare_systems_thinker(
    output_path: str | Path,
    limit: int = 5000,
    stats_path: Optional[str | Path] = None,
    examples_path: Optional[str | Path] = None,
) -> tuple[list[dict], FilterStats]:
    from datasets import load_dataset  # type: ignore

    print("Loading StackExchange dataset...", flush=True)
    ds = load_dataset(
        "PrimeIntellect/stackexchange-question-answering",
        split="train",
        streaming=True,
    )

    records: list[dict] = []
    stats = FilterStats()
    MIN_Q, MAX_Q = 20, 500
    MIN_A, MAX_A = 40, 300

    for row in ds:
        if len(records) >= limit:
            break

        # Prompt is formatted as "...Question:\n{text}\n\nNow provide..."
        raw_prompt = row.get("prompt") or ""
        m = _SE_Q_RE.search(raw_prompt)
        question = m.group(1).strip() if m else raw_prompt.strip()
        answer = (row.get("gold_standard_solution") or "").strip()
        source_id = str(row.get("problem_id") or row.get("id") or "")

        if is_empty(question):
            stats.record("empty_input")
            continue
        if is_empty(answer):
            stats.record("empty_response")
            continue
        if is_too_short(question, MIN_Q):
            stats.record("question_too_short")
            continue
        if is_too_long(question, MAX_Q):
            stats.record("question_too_long")
            continue
        if is_too_short(answer, MIN_A):
            stats.record("answer_too_short")
            continue
        if is_too_long(answer, MAX_A):
            stats.record("answer_too_long")
            continue
        if is_mostly_code(answer):
            stats.record("mostly_code")
            continue
        if is_mostly_links(answer):
            stats.record("mostly_links")
            continue

        records.append(_record("systems_thinker", {"question": question}, answer,
                               "PrimeIntellect/stackexchange-question-answering", source_id))
        stats.record(None)

    _save(records, stats, "systems_thinker", output_path, stats_path, examples_path)
    return records, stats


# ---------------------------------------------------------------------------
# Cross-Domain Analogist — gfissore/arxiv-abstracts-2021
# ---------------------------------------------------------------------------

def prepare_cross_domain_analogist(
    output_path: str | Path,
    limit: int = 5000,
    stats_path: Optional[str | Path] = None,
    examples_path: Optional[str | Path] = None,
) -> tuple[list[dict], FilterStats]:
    from datasets import load_dataset  # type: ignore

    print("Loading ArXiv abstracts dataset...", flush=True)
    ds = load_dataset(
        "gfissore/arxiv-abstracts-2021",
        split="train",
        streaming=True,
    )

    records: list[dict] = []
    stats = FilterStats()
    MIN_TITLE, MAX_TITLE = 4, 40
    MIN_ABS, MAX_ABS = 80, 250

    for row in ds:
        if len(records) >= limit:
            break

        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        source_id = str(row.get("id") or "")

        if is_empty(title):
            stats.record("empty_input")
            continue
        if is_empty(abstract):
            stats.record("empty_response")
            continue
        if is_too_short(title, MIN_TITLE):
            stats.record("title_too_short")
            continue
        if is_too_long(title, MAX_TITLE):
            stats.record("title_too_long")
            continue
        if is_too_short(abstract, MIN_ABS):
            stats.record("abstract_too_short")
            continue
        if is_too_long(abstract, MAX_ABS):
            stats.record("abstract_too_long")
            continue
        if is_formula_heavy(abstract):
            stats.record("formula_heavy")
            continue

        records.append(_record("cross_domain_analogist", {"title": title}, abstract,
                               "gfissore/arxiv-abstracts-2021", source_id))
        stats.record(None)

    _save(records, stats, "cross_domain_analogist", output_path, stats_path, examples_path)
    return records, stats


# ---------------------------------------------------------------------------
# Minimalist — ibm-research/argument_quality_ranking_30k
# ---------------------------------------------------------------------------

def prepare_minimalist(
    output_path: str | Path,
    limit: int = 5000,
    stats_path: Optional[str | Path] = None,
    examples_path: Optional[str | Path] = None,
) -> tuple[list[dict], FilterStats]:
    from datasets import load_dataset, get_dataset_config_names  # type: ignore

    print("Loading IBM Argument Quality dataset...", flush=True)
    # Config 'argument_quality_ranking' has: topic, argument, WA (quality score 0-1)
    ds = load_dataset(
        "ibm-research/argument_quality_ranking_30k",
        name="argument_quality_ranking",
        split="train",
    )
    topic_field, arg_field, quality_field = "topic", "argument", "WA"

    records: list[dict] = []
    stats = FilterStats()
    MIN_TOPIC, MAX_TOPIC = 3, 40
    MIN_ARG, MAX_ARG = 8, 60

    for row in ds:
        if len(records) >= limit:
            break

        topic = (row.get(topic_field) or "").strip()
        argument = (row.get(arg_field) or "").strip()
        source_id = str(row.get("id") or row.get("uid") or "")

        if is_empty(topic):
            stats.record("empty_input")
            continue
        if is_empty(argument):
            stats.record("empty_response")
            continue
        if is_too_short(topic, MIN_TOPIC):
            stats.record("topic_too_short")
            continue
        if is_too_long(topic, MAX_TOPIC):
            stats.record("topic_too_long")
            continue
        if is_too_short(argument, MIN_ARG):
            stats.record("argument_too_short")
            continue
        if is_too_long(argument, MAX_ARG):
            stats.record("argument_too_long")
            continue

        # WA score: mean=0.794, keep top ~75% by requiring >= 0.65
        if quality_field is not None:
            q = row.get(quality_field)
            if isinstance(q, (int, float)) and q < 0.65:
                stats.record("low_quality")
                continue

        records.append(_record("minimalist", {"topic": topic}, argument,
                               "ibm-research/argument_quality_ranking_30k", source_id))
        stats.record(None)

    _save(records, stats, "minimalist", output_path, stats_path, examples_path)
    return records, stats
