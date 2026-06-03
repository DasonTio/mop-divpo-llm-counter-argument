# Persona Diversity Comparison

Distances are `1 - SBERT cosine`, so higher means more semantically diverse.

| Run | Method | Inter-Persona↑ | Ratio vs Base Sampling↑ | Same-Prompt Intra↑ | Cross-Prompt Dispersion↑ | Silhouette↑ |
|---|---|---:|---:|---:|---:|---:|
| qwen25_0p5b | base | — | — | 0.325 | — | — |
| qwen25_0p5b | prompt_only | 0.292 | 0.899 | — | 0.838 | -0.030 |
| qwen25_0p5b | single_lora | 0.519 | 1.597 | — | 0.871 | -0.023 |
| qwen25_0p5b | mop_sft | 0.512 | 1.576 | — | 0.872 | -0.011 |
| qwen25_0p5b | mop_divpo | 0.295 | 0.909 | — | 0.831 | -0.025 |
| qwen25_0p5b | mop_divpo_v2 | 0.310 | 0.954 | — | 0.832 | -0.025 |
| qwen25_1p5b | base | — | — | 0.228 | — | — |
| qwen25_1p5b | prompt_only | 0.290 | 1.271 | — | 0.820 | -0.030 |
| qwen25_1p5b | mop_sft | 0.489 | 2.145 | — | 0.875 | -0.008 |
| qwen25_1p5b | mop_divpo_v2 | 0.293 | 1.288 | — | 0.818 | -0.030 |
