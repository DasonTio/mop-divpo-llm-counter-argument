# CMV Delta Ground Truth — BERTScore F1 vs Human Delta-Winning Arguments

Reference: 3000 sampled delta-winning arguments (success=1) from ConvoKit winning-args-corpus (Tan et al. 2016)

| Method | BERTScore F1↑ | Std | N |
|---|---|---|---|
| base | 0.8175 | 0.0090 | 120 |
| prompt_only | 0.8234 | 0.0094 | 120 |
| single_lora | 0.8280 | 0.0094 | 120 |
| mop_sft | 0.8204 | 0.0112 | 120 |
| mop_divpo | 0.8214 | 0.0093 | 120 |
| mop_divpo_v2 | 0.8209 | 0.0094 | 120 |
