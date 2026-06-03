# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Research project: diverse counter-argument generation using **Mixture of Personas (MoP)** + **DivPO** training on a Qwen2.5-0.5B-Instruct base. Goal is a journal paper. See `docs/research-plan.md` for the full execution plan (Phases 1–4).

Four cognitive personas: `contrarian`, `systems_thinker`, `cross_domain_analogist`, `minimalist` — defined in `src/personas.py`. Each trains a separate LoRA adapter.

## Environment

Python 3.12, virtualenv at `.venv/`. Activate: `source .venv/bin/activate`.

The installable package is `src/mop_divpo/`. Scripts run with `PYTHONPATH=src`. The
full pipeline (data → SFT → DivPO → evaluation) is implemented and has been run; final
results and report are in `outputs/` and `paper/`. See `README.md` for the public overview.

## Project Structure

```
src/
  personas.py              # PERSONAS dict: description + system_prompt per persona
  dataset/prepare.py       # Dataset prep scaffold (imports PERSONAS)
  exploration/analysis.py  # ConvoKit corpus exploration (CGA-CMV)
  mop_divpo/               # Main package (planned — partially exists on HuggingFace)
    data/                  # acquire.py, sources.py, normalizers.py, schema.py
    training/              # train_sft.py
    divpo/                 # pairs.py
    inference/             # generate.py (entry point for all inference)
    routing/               # gate.py (persona ranking)
    retrieval/             # corpus.py (SBERT index)
    metrics/               # diversity.py, semantic.py
    eval/                  # evaluate.py, llm_judge.py (planned)
scripts/                   # experiment scripts (planned)
outputs/                   # generated artefacts (adapters, eval results)
docs/
  research-plan.md         # Executable plan — read this first
  lessons/01-08.md         # Background lessons explaining architecture
```

## Architecture

**Training flow:** Raw datasets → normalizers (→ `SFTRecord` schema) → per-persona SFT JSONL → SFT training (4× LoRA, r=16 α=32 on Qwen2.5-0.5B) → DivPO candidate generation → pair construction (`chosen`=rare+good, `rejected`=common) → `trl dpo` (off-the-shelf) → 4 DivPO adapters pushed to HuggingFace (`DasonTio/mop-divpo-coauthor`).

**Inference flow:** Prompt → persona gate (`routing/gate.py`, top-k=2) → parallel generation per persona adapter → output parser → diverse idea set.

**Key invariant:** DivPO's contribution is entirely in `divpo/pairs.py` (pair selection). The DPO gradient update is vanilla `trl dpo` — no custom trainer.

**Corpus retrieval** (`retrieval/corpus.py`) serves dual purpose: injects top-3 similar training examples into the inference system prompt (anti-plagiarism) AND provides corpus-novelty metric scores.

## Trained Adapters

Adapters on HuggingFace Hub: `DasonTio/mop-divpo-coauthor` (SFT stage) and DivPO stage. Before any experiment, verify they load:

```bash
python -c "from mop_divpo.inference.generate import generate; print(generate(prompt='test', personas=['contrarian'], adapter_prefix='DasonTio/mop-divpo-coauthor', adapter_stage='sft'))"
```

If this fails, fix adapter loading before any other work.

## Data

- CGA-CMV corpus (`conversations-gone-awry-cmv-corpus`) via ConvoKit — downloads to `~/.convokit/saved-corpora/`
- Training data paths: `data/processed/sft/{persona}.jsonl`, `data/processed/divpo/{persona}.jsonl`

## Current Research Phase

Per `docs/research-plan.md` (plan date 2026-05-25):

| Phase | Goal | Status |
|---|---|---|
| 1 | Persona distinctness validation (SBERT 4×4 matrix) | DONE — Table 4.1 |
| 2 | LLM-as-judge framework (`src/mop_divpo/eval/llm_judge.py`) | DONE — Tables 4.2–4.3 |
| 3 | Baseline evaluation table (6 methods × metrics) + ArmoRM/BERTScore/significance | DONE — Tables 4.4–4.6 |
| 4 | Paper scoping + drafting | DONE — `paper/MoP-DivPO-Counter-Argument-Report.pdf` |

All phases complete; results in `outputs/` (see `outputs/README.md`). Scaling validation
on Qwen2.5-1.5B/3B confirms findings are not a 0.5B artifact.

## Evaluation

Five baselines: Base (no adapter), Prompt-only, Single LoRA, MoP SFT, MoP+DivPO.

Seven metrics: Self-BLEU↓, Distinct-1↑, Distinct-2↑, SBERT pairwise cosine↓, corpus novelty↑, LLM-judge quality↑, LLM-judge novelty/utility/fidelity↑.

Judge model: `claude-sonnet-4` or `gpt-4o`. Must be stronger than Qwen 0.5B. Cache all judge calls at `temperature=0` keyed by `(rubric_name, prompt, output_hash)`.

LLM-judge rubrics: quality (relevance/coherence/substance), persona fidelity, novelty against peers, pre-writing utility. All return structured JSON.
