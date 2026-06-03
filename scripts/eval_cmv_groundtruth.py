#!/usr/bin/env python
"""CMV delta ground truth evaluation.

Uses human-validated delta-winning arguments from Reddit r/ChangeMyView as
a reference corpus. A delta (Δ) is awarded by the OP when a reply actually
changed their mind — the closest thing to a ground truth for counter-argument
quality and persuasiveness.

Pipeline:
  1. Load winning-args-corpus (ConvoKit CMV) — delta-winning replies only.
  2. For each evaluation prompt, retrieve the top-k most topically similar
     delta-winning arguments via SBERT cosine (no GPT, no API).
  3. Compute BERTScore (F1) of each model output against its top-k references.
  4. Report per-method mean BERTScore as a human-grounded quality metric.

Reads:  outputs/{eval_dir}/generations.jsonl
Writes: outputs/{eval_dir}/cmv_groundtruth_scores.jsonl
        outputs/{eval_dir}/cmv_groundtruth_table.md

Usage:
    python scripts/eval_cmv_groundtruth.py --dir outputs/evaluation_a100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_delta_replies(min_words: int = 20, max_words: int = 300) -> list[str]:
    """Load delta-winning replies from ConvoKit CMV corpus.

    Delta replies are replies that received a Δ from the OP — empirically
    the best human-validated counter-arguments in the dataset.
    """
    print("Loading ConvoKit CMV corpus (downloads on first run)...", flush=True)
    import convokit
    corpus = convokit.Corpus(filename=convokit.download("winning-args-corpus"))

    delta_replies: list[str] = []
    for utt in corpus.iter_utterances():
        # Delta marker in the reply text or meta
        text = utt.text or ""
        meta = utt.meta or {}
        has_delta = (
            "Δ" in text or "∆" in text or "delta" in text.lower()
            or meta.get("score", 0) > 0
            or any("delta" in str(v).lower() for v in meta.values())
        )
        if not has_delta:
            continue
        words = text.split()
        if min_words <= len(words) <= max_words:
            delta_replies.append(text.strip())

    # Fallback: if delta detection yields too few, take highly-scored replies
    if len(delta_replies) < 100:
        print(f"  Only {len(delta_replies)} delta replies found, "
              "expanding to all replies with score > 5...", flush=True)
        for utt in corpus.iter_utterances():
            text = utt.text or ""
            score = (utt.meta or {}).get("score", 0)
            words = text.split()
            if score > 5 and min_words <= len(words) <= max_words:
                delta_replies.append(text.strip())

    print(f"  Loaded {len(delta_replies)} reference delta replies.", flush=True)
    return delta_replies


def retrieve_references(
    prompts: list[str],
    delta_replies: list[str],
    embedder,
    top_k: int = 3,
) -> dict[str, list[str]]:
    """For each prompt, find the top-k most topically similar delta replies."""
    import numpy as np

    print(f"Embedding {len(delta_replies)} delta replies...", flush=True)
    ref_embs = embedder.encode(delta_replies, convert_to_numpy=True,
                               batch_size=64, show_progress_bar=True)

    print(f"Embedding {len(prompts)} evaluation prompts...", flush=True)
    prompt_embs = embedder.encode(prompts, convert_to_numpy=True, batch_size=32)

    # Normalize for cosine
    ref_norm = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-9)
    prompt_norm = prompt_embs / (np.linalg.norm(prompt_embs, axis=1, keepdims=True) + 1e-9)

    prompt_to_refs: dict[str, list[str]] = {}
    for prompt, pv in zip(prompts, prompt_norm):
        sims = ref_norm @ pv
        top_idx = sims.argsort()[::-1][:top_k]
        prompt_to_refs[prompt] = [delta_replies[i] for i in top_idx]

    return prompt_to_refs


def compute_bertscore(
    outputs: list[str],
    references: list[list[str]],
    lang: str = "en",
) -> list[float]:
    """BERTScore F1 of each output against its reference list (max over refs)."""
    from bert_score import score as bs_score

    # Flatten: one ref per output (best match)
    # BERTScore accepts list[str] refs — we pick the single best ref per output
    # by running once per reference and taking the max F1.
    import numpy as np

    n = len(outputs)
    max_ref_count = max(len(r) for r in references)
    f1_matrix = np.zeros((n, max_ref_count))

    for ref_idx in range(max_ref_count):
        ref_texts = [
            (refs[ref_idx] if ref_idx < len(refs) else refs[0])
            for refs in references
        ]
        _, _, F1 = bs_score(outputs, ref_texts, lang=lang, verbose=False)
        f1_matrix[:, ref_idx] = F1.numpy()

    return f1_matrix.max(axis=1).tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/evaluation_a100")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of reference delta replies to retrieve per prompt.")
    p.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    d = Path(args.dir)
    gens = _read_jsonl(d / "generations.jsonl")
    print(f"Loaded {len(gens)} outputs.", flush=True)

    # Load delta corpus
    delta_replies = load_delta_replies()

    # SBERT retrieval
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.sbert_model)
    prompts = list({r["prompt"] for r in gens})
    prompt_to_refs = retrieve_references(prompts, delta_replies, embedder, top_k=args.top_k)

    # Install bert_score if missing
    try:
        import bert_score  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score", "-q"])

    # Score outputs
    print(f"Computing BERTScore for {len(gens)} outputs...", flush=True)
    outputs = [r["output"] for r in gens]
    references = [prompt_to_refs[r["prompt"]] for r in gens]
    f1_scores = compute_bertscore(outputs, references)

    scored = [{**rec, "cmv_bertscore_f1": round(f1, 4), "cmv_refs": refs}
              for rec, f1, refs in zip(gens, f1_scores,
                                       [prompt_to_refs[r["prompt"]] for r in gens])]
    _write_jsonl(d / "cmv_groundtruth_scores.jsonl", scored)
    print(f"Wrote → {d}/cmv_groundtruth_scores.jsonl", flush=True)

    # Per-method summary
    import numpy as np
    by_method: dict[str, list[float]] = defaultdict(list)
    for rec in scored:
        by_method[rec["method"]].append(rec["cmv_bertscore_f1"])

    METHOD_ORDER = ["base", "prompt_only", "single_lora", "mop_sft", "mop_divpo", "mop_divpo_v2"]
    lines = ["# CMV Delta Ground Truth — BERTScore F1\n",
             "Higher = model outputs are more similar to human delta-winning arguments.\n",
             "| Method | BERTScore F1↑ | Std | N |",
             "|---|---|---|---|"]
    results = {}
    for m in METHOD_ORDER:
        scores = by_method.get(m, [])
        if not scores:
            continue
        arr = np.array(scores)
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        results[m] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(scores)}
        lines.append(f"| {m} | {mean:.4f} | {std:.4f} | {len(scores)} |")

    md = "\n".join(lines) + "\n"
    (d / "cmv_groundtruth_table.md").write_text(md, encoding="utf-8")
    (d / "cmv_groundtruth_table.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"Wrote cmv_groundtruth_table.{{md,json}} → {d}", flush=True)


if __name__ == "__main__":
    main()
