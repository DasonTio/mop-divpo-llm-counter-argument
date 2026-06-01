# Baseline Evaluation

| Method | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ |
|---|---|---|---|---|
| base | 0.079 | 0.486 | 0.895 | 0.661 |
| prompt_only | 0.053 | 0.574 | 0.926 | 0.709 |
| mop_sft | 0.024 | 0.562 | 0.944 | 0.485 |
| mop_divpo | 0.053 | 0.566 | 0.928 | 0.701 |

(↓ lower is better, ↑ higher is better; L = LLM-judge. Significance vs mop_divpo in baseline_table.json.)
