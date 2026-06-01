# Evaluation Runbook

How to run Phases 1–3 of `docs/research-plan.md` on the available compute
(Kaggle T4×2, Google Colab T4, vast.ai). All scripts read `HF_TOKEN` from the
environment and pull adapters from `DasonTio/mop-divpo-coauthor`.

Common setup on any GPU box:

```bash
pip install transformers peft accelerate sentence-transformers datasets huggingface_hub
export HF_TOKEN=hf_xxx
git clone <repo> && cd mop_divpo_llm-counter-argument
export PYTHONPATH=src
```

---

## Prerequisite — Single-LoRA baseline adapter

Phase 3 method 3 needs one adapter trained on all four persona SFT sets merged.
Run once (Colab/Kaggle/vast.ai T4, ~20–40 min); pushes to `single/all`:

```bash
python scripts/train_single_lora.py            # trains + pushes to MODEL_REPO/single/all
```

If skipped, run `run_baseline_evaluation.py --methods base prompt_only mop_sft mop_divpo`.

---

## Phase 1 — Persona distinctness (GATES everything)

GPU box. ~30 min for 400 outputs. Decides which personas survive (§1.5).

```bash
python scripts/experiment_persona_distinctness.py --stage divpo --n 5
# smoke test first:
python scripts/experiment_persona_distinctness.py --stage divpo --n 2 --limit-prompts 3
```

Output: `outputs/experiments/persona_distinctness.json` (4×4 matrix + verdict).
Read the verdict before proceeding. If a pair > 0.7, apply §4.3 reduction.

---

## Phase 2 + 3 — Baseline table

Generation is GPU; judging is API-only and can run on a laptop. Stages are
resumable, so split them across machines.

**On the GPU box — generate 600 outputs (~1–2 h):**

```bash
python scripts/run_baseline_evaluation.py --only-generate
# writes outputs/evaluation/generations.jsonl
```

Copy `outputs/evaluation/generations.jsonl` to wherever you run the judge.

**On any machine with an API key — automated metrics + judge + table:**

```bash
pip install anthropic        # or: pip install openai
export ANTHROPIC_API_KEY=sk-...
python scripts/run_baseline_evaluation.py --skip-generation --judge anthropic
```

Outputs in `outputs/evaluation/`:
- `baseline_table.md` / `.csv` / `.json` — the headline table (§3.6)
- `llm_judge_scores.jsonl` — per-output rubric scores
- `judge_cache.json` — cached calls; re-runs are free

Automated-only (no API cost) for a quick look:

```bash
python scripts/run_baseline_evaluation.py --skip-generation --judge none
```

Smoke test the whole wiring on 2 prompts:

```bash
python scripts/run_baseline_evaluation.py --limit-prompts 2 --judge none
```

---

## Resource assignment

| Stage | Resource | Wall time |
|---|---|---|
| Single-LoRA train | Kaggle / vast.ai T4 | 20–40 min |
| Phase 1 generation | Colab / Kaggle T4 | ~30 min |
| Phase 3 generation (600) | Kaggle T4×2 or vast.ai | 1–2 h |
| Automated metrics | any CPU | <10 min |
| LLM judge (2,400 cached calls) | API (laptop) | 1–2 h |
| Aggregation + stats | any CPU | <1 min |

## Success criteria (research-plan.md §9)

1. ≥2 persona pairs with cosine < 0.5 in Phase 1.
2. MoP+DivPO beats Base, Prompt-only, Single-LoRA on ≥3 of {Self-BLEU, SBERT, judge-novelty, judge-utility}.
3. MoP+DivPO does not lose to MoP-SFT on judge-quality by > 0.3.
4. p < 0.05 on ≥1 metric in criterion 2 (reported in `baseline_table.json`).
