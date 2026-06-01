# AGENT HANDOFF

> Read this first before touching anything. Single source of truth for what is
> done, what is pending, and exactly how to proceed on the next machine.

---

## State as of 2026-06-01

### Training — COMPLETE
All 8 adapters confirmed on HuggingFace Hub (`DasonTio/mop-divpo-coauthor`):
- `sft/contrarian`, `sft/systems_thinker`, `sft/cross_domain_analogist`, `sft/minimalist`
- `divpo/contrarian`, `divpo/systems_thinker`, `divpo/cross_domain_analogist`, `divpo/minimalist`

### Evaluation Pipeline — CODE BUILT, NOT YET RUN

Branch `feat/eval-pipeline` (commit `150b82a`). Main branch is behind.
**Pull this branch on every compute box.**

Built this session:

| File | Purpose |
|---|---|
| `src/mop_divpo/inference/generate.py` | All 5 baselines in one code path |
| `src/mop_divpo/metrics/diversity.py` | Self-BLEU, Distinct-1/2 |
| `src/mop_divpo/metrics/semantic.py` | SBERT pairwise + cross-group cosine |
| `src/mop_divpo/eval/llm_judge.py` | 4-rubric judge, file-backed cache |
| `src/mop_divpo/eval/llm_judge_rubrics.py` | Rubric templates |
| `src/mop_divpo/eval/prompts.py` | 20 distinctness + 30 eval prompts |
| `src/mop_divpo/eval/aggregate.py` | mean/std, paired bootstrap, table builder |
| `scripts/experiment_persona_distinctness.py` | Phase 1 — 4x4 inter-persona SBERT matrix |
| `scripts/train_single_lora.py` | Single-LoRA baseline (method 3, NOT YET TRAINED) |
| `scripts/run_baseline_evaluation.py` | Phase 3 — 5 methods x 7 metrics table |
| `docs/EVAL_RUNBOOK.md` | Full run commands per resource |

67 unit tests pass. No GPU or API runs executed yet.

### Missing outputs (must be generated)
1. `single/all` adapter on Hub — needs `train_single_lora.py` run.
2. `outputs/experiments/persona_distinctness.json` — Phase 1 not run.
3. `outputs/evaluation/baseline_table.*` — Phase 3 not run.

---

## Critical Path

```
[A] Train single-LoRA  →  [B] Phase 1 distinctness  →  [C] Generate 600 outputs  →  [D] Judge + table
   (vast.ai / Kaggle)       (Kaggle / Colab T4)          (Kaggle T4x2 / vast.ai)      (laptop + API key)
```

**B gates C.** If any persona pair cosine > 0.7, drop that persona before Phase 3.
Decision rule: `docs/research-plan.md §1.5`.

---

## Setup (run on every box before any script)

```bash
pip install transformers peft trl accelerate datasets huggingface_hub sentence-transformers
pip uninstall -y torchao          # prevents Kaggle T4 crash
export HF_TOKEN=hf_xxx            # Kaggle Secrets / Colab userdata / env var
git clone https://github.com/<your-repo>.git
cd mop_divpo_llm-counter-argument
git checkout feat/eval-pipeline   # the branch with all eval code
export PYTHONPATH=src
```

---

## Run Order

### A — Train single-LoRA baseline (vast.ai A100 or Kaggle T4×2, ~30 min)

```bash
python scripts/train_single_lora.py
# pushes to DasonTio/mop-divpo-coauthor/single/all
```

Skip A and run Phase 3 with `--methods base prompt_only mop_sft mop_divpo` if short on time.

### B — Phase 1: persona distinctness (Colab T4 / Kaggle, ~30 min)

```bash
# smoke test (2 min, verify setup works):
python scripts/experiment_persona_distinctness.py --stage divpo --n 2 --limit-prompts 3

# full run (400 outputs, 20 prompts x 4 personas x 5):
python scripts/experiment_persona_distinctness.py --stage divpo --n 5
```

Read `outputs/experiments/persona_distinctness.json`. Apply persona reduction from
`docs/research-plan.md §4.3` if any off-diagonal > 0.7 before continuing to C.

### C — Phase 3 generation (Kaggle T4×2 or vast.ai, 1-2 h)

```bash
python scripts/run_baseline_evaluation.py --only-generate
# writes: outputs/evaluation/generations.jsonl
# download this file to the judge machine
```

### D — Phase 3: judge + table (laptop / Colab CPU, 1-2 h API-limited)

```bash
pip install anthropic          # or: pip install openai
export ANTHROPIC_API_KEY=sk-ant-xxx    # or OPENAI_API_KEY
# copy generations.jsonl into outputs/evaluation/ first
python scripts/run_baseline_evaluation.py --skip-generation --judge anthropic
```

Outputs: `outputs/evaluation/baseline_table.{md,csv,json}` — the headline paper table.

Quick automated-only run (no API, no cost):
```bash
python scripts/run_baseline_evaluation.py --skip-generation --judge none
```

---

## After results land

1. Check 4 success criteria: `docs/research-plan.md §9`.
2. Update `docs/research-report.md` with actual numbers.
3. Update `README.md` §Status and §Evaluation Metrics.
4. Start paper draft per `docs/research-plan.md §4.1`.

---

## Key constants (do NOT change without checking training scripts)

```python
BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REPO  = "DasonTio/mop-divpo-coauthor"
PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]
LORA_R = 16  |  LORA_ALPHA = 32  |  LORA_DROPOUT = 0.05
```

---

## Gotchas

- `PERSONA_ROUTING_DESCRIPTIONS` does NOT exist in `personas.py` — code uses `PERSONAS[p]["description"]`.
- `single/all` adapter not on Hub yet — run A first, or skip method 3 explicitly.
- On Kaggle DDP: `dataloader_num_workers` MUST be `0` (> 0 deadlocks multi-GPU).
- Judge calls cached in `outputs/evaluation/judge_cache.json` — re-runs are free.
- `--only-generate` and `--skip-generation` flags let you split GPU vs API work across machines.
