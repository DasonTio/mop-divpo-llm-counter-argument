# Persona Diversity Evaluation: qwen25_0p5b

Distances are `1 - SBERT cosine`, so higher means more semantically diverse.

**Caveat:** True same-prompt intra-persona diversity requires multiple generations for each (prompt, persona). Existing persona-method files contain one generation per (prompt, persona), so intra-persona same-prompt metrics are unavailable there.

## Prompt-Controlled Inter-Persona Diversity

| Method | Mean Distance↑ | Std | Prompts | Ratio vs Base Sampling↑ | Δ vs Base Sampling↑ |
|---|---:|---:|---:|---:|---:|
| base | — | — | 0 | — | — |
| prompt_only | 0.292 | 0.094 | 30 | 0.899 | -0.033 |
| single_lora | 0.519 | 0.129 | 30 | 1.597 | 0.194 |
| mop_sft | 0.512 | 0.122 | 30 | 1.576 | 0.187 |
| mop_divpo | 0.295 | 0.089 | 30 | 0.909 | -0.030 |
| mop_divpo_v2 | 0.310 | 0.098 | 30 | 0.954 | -0.015 |

## Same-Prompt Intra-Persona Diversity

| Method | Mean Distance↑ | Std | Groups | Available? |
|---|---:|---:|---:|---|
| base | 0.325 | 0.151 | 30 | yes |
| prompt_only | — | — | 0 | no |
| single_lora | — | — | 0 | no |
| mop_sft | — | — | 0 | no |
| mop_divpo | — | — | 0 | no |
| mop_divpo_v2 | — | — | 0 | no |

## Cross-Prompt Persona Dispersion

This is a proxy for persona breadth, but it is prompt-confounded.

| Method | Mean Distance↑ | Std | Personas |
|---|---:|---:|---:|
| base | — | — | 0 |
| prompt_only | 0.838 | 0.005 | 4 |
| single_lora | 0.871 | 0.004 | 4 |
| mop_sft | 0.872 | 0.023 | 4 |
| mop_divpo | 0.831 | 0.014 | 4 |
| mop_divpo_v2 | 0.832 | 0.010 | 4 |

## Persona Silhouette

This is not prompt-controlled. Positive values mean outputs cluster by persona; negative values suggest prompt/topic dominates persona clustering.

| Method | Silhouette↑ | Std | Outputs |
|---|---:|---:|---:|
| base | — | — | 0 |
| prompt_only | -0.030 | 0.015 | 120 |
| single_lora | -0.023 | 0.018 | 120 |
| mop_sft | -0.011 | 0.027 | 120 |
| mop_divpo | -0.025 | 0.019 | 120 |
| mop_divpo_v2 | -0.025 | 0.019 | 120 |
