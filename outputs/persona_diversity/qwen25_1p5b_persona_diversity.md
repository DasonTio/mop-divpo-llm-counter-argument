# Persona Diversity Evaluation: qwen25_1p5b

Distances are `1 - SBERT cosine`, so higher means more semantically diverse.

**Caveat:** True same-prompt intra-persona diversity requires multiple generations for each (prompt, persona). Existing persona-method files contain one generation per (prompt, persona), so intra-persona same-prompt metrics are unavailable there.

## Prompt-Controlled Inter-Persona Diversity

| Method | Mean Distance↑ | Std | Prompts | Ratio vs Base Sampling↑ | Δ vs Base Sampling↑ |
|---|---:|---:|---:|---:|---:|
| base | — | — | 0 | — | — |
| prompt_only | 0.290 | 0.082 | 30 | 1.271 | 0.062 |
| mop_sft | 0.489 | 0.086 | 30 | 2.145 | 0.261 |
| mop_divpo_v2 | 0.293 | 0.061 | 30 | 1.288 | 0.066 |

## Same-Prompt Intra-Persona Diversity

| Method | Mean Distance↑ | Std | Groups | Available? |
|---|---:|---:|---:|---|
| base | 0.228 | 0.061 | 30 | yes |
| prompt_only | — | — | 0 | no |
| mop_sft | — | — | 0 | no |
| mop_divpo_v2 | — | — | 0 | no |

## Cross-Prompt Persona Dispersion

This is a proxy for persona breadth, but it is prompt-confounded.

| Method | Mean Distance↑ | Std | Personas |
|---|---:|---:|---:|
| base | — | — | 0 |
| prompt_only | 0.820 | 0.022 | 4 |
| mop_sft | 0.875 | 0.006 | 4 |
| mop_divpo_v2 | 0.818 | 0.023 | 4 |

## Persona Silhouette

This is not prompt-controlled. Positive values mean outputs cluster by persona; negative values suggest prompt/topic dominates persona clustering.

| Method | Silhouette↑ | Std | Outputs |
|---|---:|---:|---:|
| base | — | — | 0 |
| prompt_only | -0.030 | 0.019 | 120 |
| mop_sft | -0.008 | 0.021 | 120 |
| mop_divpo_v2 | -0.030 | 0.016 | 120 |
