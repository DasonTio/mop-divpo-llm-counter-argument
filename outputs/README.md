# Results map

Every artifact here maps to a table or claim in the report
(`paper/MoP-DivPO-Counter-Argument-Report.pdf`). Trained LoRA adapters are **not**
checked in — they live on HuggingFace Hub (`DasonTio/mop-divpo-coauthor`).

## Primary run — Qwen2.5-0.5B (the paper's headline numbers)

`evaluation_a100/`

| File | Paper reference |
|---|---|
| `baseline_table.{md,json,csv}` | **Table 4.3** — 6 methods × diversity + judge metrics |
| `significance_mop_divpo_v2.{md,json}` | **Table 4.4** — paired bootstrap (n=30) vs each baseline |
| `inter_judge_agreement.json` | **Table 4.2** — GPT-4o-mini vs GPT-4o Spearman ρ |
| `armorm_table.{md,json}` | **Table 4.5** — ArmoRM reward-model quality |
| `cmv_groundtruth_table.{md,json}` | **Table 4.6** — BERTScore F1 vs CMV delta-winning args |
| `generations.jsonl` | raw 720 generations (6 methods × 30 prompts × 4) |
| `llm_judge_scores.jsonl`, `armorm_scores.jsonl` | per-output scores behind the tables |

## Persona distinctness — Section 4.1

| File | Paper reference |
|---|---|
| `persona_distinctness.json` | **Table 4.1** — 4×4 inter-persona SBERT cosine (20 neutral prompts) |
| `persona_diversity/` | per-model persona diversity breakdowns (0.5B + 1.5B) |
| `persona_breakdown/` | per-persona metric breakdown |

## Scaling validation — Section 4.5 (generalization)

| Dir | Content |
|---|---|
| `evaluation_qwen25_1p5b_scale/` | re-run on Qwen2.5-1.5B-Instruct |
| `evaluation_qwen25_3b_scale/` | re-run on Qwen2.5-3B-Instruct |

## Data checks

`data_checks/` — SFT dataset summaries + sampled examples (sanity, Section 3.3).

## Not tracked (regenerable / artifacts)

`scaling_run/` process logs, `judge_cache.json`, `calibration_cache.json`, and
`outputs/adapters/` are git-ignored. Caches rebuild from the eval scripts; the
judge calls are deterministic (`temperature=0`).
