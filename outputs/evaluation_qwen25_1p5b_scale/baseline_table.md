# Baseline Evaluation

| Method | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ |
|---|---|---|---|---|
| base | 0.099 | 0.486 | 0.889 | 0.772 |
| prompt_only | 0.041 | 0.575 | 0.935 | 0.710 |
| mop_sft | 0.022 | 0.577 | 0.953 | 0.511 |
| mop_divpo_v2 | 0.048 | 0.561 | 0.930 | 0.707 |

(↓ lower is better, ↑ higher is better; L = LLM-judge. Significance vs mop_divpo_v2 in baseline_table.json.)
