# Baseline Evaluation

| Method | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ | Quality(L)↑ | Novelty(L)↑ | Utility(L)↑ | Fidelity(L)↑ |
|---|---|---|---|---|---|---|---|---|
| base | 0.086 | 0.481 | 0.894 | 0.675 | 3.417 | 3.192 | 3.058 | — |
| prompt_only | 0.062 | 0.576 | 0.923 | 0.708 | 3.975 | 3.458 | 3.542 | 3.042 |
| single_lora | 0.040 | 0.597 | 0.945 | 0.481 | 2.589 | 4.650 | 2.308 | 1.975 |
| mop_sft | 0.026 | 0.559 | 0.943 | 0.488 | 2.633 | 4.700 | 2.308 | 2.192 |
| mop_divpo | 0.056 | 0.556 | 0.920 | 0.705 | 3.931 | 3.642 | 3.542 | 3.025 |
| mop_divpo_v2 | 0.051 | 0.580 | 0.931 | 0.690 | 3.956 | 3.650 | 3.550 | 3.058 |

(↓ lower is better, ↑ higher is better; L = LLM-judge. Significance vs mop_divpo in baseline_table.json.)
