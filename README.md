# Mixture of Personas + Diverse Preference Optimization for Counter-Argument Generation

Diverse counter-argument generation by combining **Mixture of Personas (MoP)** with
**Diverse Preference Optimization (DivPO)** on a small base model (Qwen2.5-0.5B-Instruct).

The goal is *pre-writing ideation*: an assistant that surfaces genuinely different
angles on a claim instead of converging on the single safest rebuttal. Aligned LLMs
tend toward *generative monoculture* — fluent but homogeneous output. This project
attacks that on two axes at once:

- **Inter-persona diversity** — four cognitive-persona LoRA adapters, each trained to
  argue in a distinct style.
- **Intra-persona diversity** — DivPO preference training that rewards rare-but-good
  candidates rather than only the highest-reward one.

> **Honest framing.** This is a controlled empirical study of the diversity–quality
> trade-off at small scale — not a claim that the method dominates every metric.
> See [Results](#results) for what actually holds.

## The four personas

| Persona | Cognitive role | SFT dataset |
|---|---|---|
| `contrarian` | challenges hidden assumptions, minority dissent | CGA-CMV (Reddit ChangeMyView) |
| `systems_thinker` | cause–effect, feedback loops, leverage points | StackExchange Q&A |
| `cross_domain_analogist` | cross-field analogy / knowledge transfer | arXiv abstracts |
| `minimalist` | constraint-driven, strips to the core | IBM Argument Quality |

Each persona is one independent LoRA adapter (r=16, α=32) over a frozen base.

## Method

```
Raw datasets → normalize → per-persona SFT JSONL
  → SFT (4× LoRA, base frozen)
  → DivPO candidate generation (4 per prompt, T=0.9)
  → pair construction:  chosen = rare + good,  rejected = common
  → trl DPO (vanilla gradient)  → 4 DivPO adapters
```

Two DivPO variants are compared:
- **DivPO v1** — rarity scored *within* a persona's own candidate pool (ablation).
- **DivPO v2** — rarity scored *across all four personas* + a relevance/length quality
  floor suited to counter-arguments (the proposed change).

The DivPO contribution lives entirely in pair selection (`src/mop_divpo/divpo/pairs.py`);
the gradient update is stock `trl dpo`.

See `paper/` for the full report and both pipeline figures (Gambar 1 training,
Gambar 2 inference).

## Results

Primary evaluation: Qwen2.5-0.5B, 6 methods, 30 CGA-CMV prompts, 720 generations.

| Method | Self-BLEU↓ | Distinct-2↑ | Quality (judge)↑ | Utility↑ | ArmoRM↑ |
|---|---|---|---|---|---|
| Base | 0.086 | 0.894 | 3.417 | 3.058 | 0.0946 |
| Prompt-only | 0.062 | 0.923 | **3.975** | 3.542 | **0.1101** |
| Single LoRA | 0.040 | 0.945 | 2.589 | 2.308 | 0.0813 |
| MoP-SFT | **0.026** | 0.943 | 2.633 | 2.308 | 0.0782 |
| MoP+DivPO (v1) | 0.056 | 0.920 | 3.931 | 3.542 | 0.1054 |
| **MoP+DivPO v2** | 0.051 | 0.931 | 3.956 | **3.550** | 0.1079 |

What the numbers actually say:

1. **Personas are distinct.** All six inter-persona SBERT cosines sit at ~0.19–0.20,
   far below the 0.5 separation threshold — the MoP assumption holds (Table 4.1).
2. **Raw adapter training trades quality for diversity.** Single-LoRA and MoP-SFT win
   lexical diversity but collapse on quality/utility — useless for counter-arguments.
3. **DivPO recovers usability.** MoP+DivPO(v2) restores quality/utility back to the
   prompt-only level while keeping a measurable diversity edge.
4. **v2 > v1, but the margin is small.** Cross-persona rarity beats within-persona on
   Distinct-1 (p=0.008) and Distinct-2 (p=0.014) at equal quality — statistically
   real, practically modest.
5. **Convergent validity.** LLM-judge, ArmoRM (human-preference reward model), and
   BERTScore vs CMV delta-winning arguments agree on the ranking.

Note that **prompt-only is a co-leader** on quality and ArmoRM (the difference vs v2 is
not significant, p≈0.6–0.87). The trained pipeline's value is learned persona
separation + a measurable diversity signal, not a quality win over prompting.

Full tables and their file locations: [`outputs/README.md`](outputs/README.md).

## Repository structure

```
paper/            Final report PDF, pipeline figures, source papers (references/)
src/mop_divpo/    Installable package
  data/           dataset acquisition + SFT record building
  divpo/          pair selection (the DivPO contribution) + reward/quality scoring
  inference/      generate.py — all 6 methods in one code path
  metrics/        diversity (Self-BLEU, Distinct-n) + semantic (SBERT)
  eval/           LLM-judge, persona diversity, aggregation + paired bootstrap
scripts/          training + evaluation entry points
tests/            unit tests
data/processed/   per-persona SFT + DivPO JSONL
outputs/          evaluation results (see outputs/README.md)
docs/             research-plan.md + background lessons (01–08)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Trained adapters live on HuggingFace Hub (`DasonTio/mop-divpo-coauthor`), not in git.
Verify they load:

```bash
PYTHONPATH=src python -c "from mop_divpo.inference.generate import generate; \
print(generate(prompt='test', personas=['contrarian'], \
adapter_prefix='DasonTio/mop-divpo-coauthor', adapter_stage='sft'))"
```

## Reproduce

```bash
# 1. data prep (local)
PYTHONPATH=src python scripts/prepare_sft_datasets.py
PYTHONPATH=src python scripts/prepare_divpo_datasets.py

# 2. training (GPU) — per persona, both stages
PYTHONPATH=src python scripts/train_sft.py
PYTHONPATH=src python scripts/train_divpo.py

# 3. evaluation
PYTHONPATH=src python scripts/experiment_persona_distinctness.py   # Table 4.1
PYTHONPATH=src python scripts/run_baseline_evaluation.py           # Tables 4.2–4.3
PYTHONPATH=src python scripts/eval_armorm.py                       # Table 4.5
PYTHONPATH=src python scripts/eval_cmv_groundtruth.py              # Table 4.6
PYTHONPATH=src python scripts/significance_report.py               # Table 4.4
```

`scripts/vastai_eval_pipeline.sh` runs the full evaluation end-to-end on a rented GPU.

```bash
pytest    # unit tests
```

## Author

Dason Tiovino · Program Studi Informatika, Universitas Pradita · 2026.
Course project for LLM & Generative AI.
