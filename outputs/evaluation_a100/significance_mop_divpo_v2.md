# Significance: `mop_divpo_v2` vs baselines (paired bootstrap, 10000 resamples, n=30 prompts)

Cell = signed mean diff (reference − baseline, oriented so **+ means the reference is better**); `*` = p<0.05, `**` = p<0.01.

| Comparison | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ | Quality(L)↑ | Novelty(L)↑ | Utility(L)↑ | Fidelity(L)↑ |
|---|---|---|---|---|---|---|---|---|
| mop_divpo_v2 vs base | +0.036** (p=0.000) | +0.099** (p=0.000) | +0.037** (p=0.000) | -0.015 (p=0.675) | +0.539** (p=0.000) | +0.458** (p=0.000) | +0.492** (p=0.000) | — |
| mop_divpo_v2 vs mop_divpo | +0.005 (p=0.211) | +0.024** (p=0.008) | +0.011* (p=0.014) | +0.014 (p=0.262) | +0.025 (p=0.376) | +0.008 (p=0.493) | +0.008 (p=0.495) | +0.033 (p=0.401) |
| mop_divpo_v2 vs mop_sft | -0.024 (p=1.000) | +0.022** (p=0.004) | -0.012 (p=0.998) | -0.202 (p=1.000) | +1.322** (p=0.000) | -1.050 (p=1.000) | +1.242** (p=0.000) | +0.867** (p=0.000) |
| mop_divpo_v2 vs prompt_only | +0.011* (p=0.020) | +0.004 (p=0.282) | +0.008* (p=0.023) | +0.018 (p=0.234) | -0.019 (p=0.609) | +0.192 (p=0.090) | +0.008 (p=0.491) | +0.017 (p=0.457) |
| mop_divpo_v2 vs single_lora | -0.011 (p=0.972) | -0.016 (p=0.920) | -0.014 (p=0.998) | -0.209 (p=1.000) | +1.367** (p=0.000) | -1.000 (p=1.000) | +1.242** (p=0.000) | +1.083** (p=0.000) |

## Verdict

- **vs base**: significantly better on 6 (Self-BLEU, Distinct-1, Distinct-2, Quality(L), Novelty(L), Utility(L)); significantly worse on 0 (none).
- **vs mop_divpo**: significantly better on 2 (Distinct-1, Distinct-2); significantly worse on 0 (none).
- **vs mop_sft**: significantly better on 4 (Distinct-1, Quality(L), Utility(L), Fidelity(L)); significantly worse on 0 (none).
- **vs prompt_only**: significantly better on 2 (Self-BLEU, Distinct-2); significantly worse on 0 (none).
- **vs single_lora**: significantly better on 3 (Quality(L), Utility(L), Fidelity(L)); significantly worse on 0 (none).
