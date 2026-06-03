# Per-Persona Diversity and Quality

Diversity is computed across one persona's 30 prompt-conditioned outputs. This is useful as a persona-breadth diagnostic, but it is prompt-confounded.

Quality columns are present only where LLM-judge scores exist.

| Run | Method | Persona | n | Self-BLEU↓ | Distinct-1↑ | Distinct-2↑ | SBERT-cos↓ | Quality↑ | Novelty↑ | Utility↑ | Fidelity↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen25_0p5b | mop_divpo | contrarian | 30 | 0.094 | 0.292 | 0.837 | 0.181 | 3.778 | 3.500 | 3.500 | 3.033 |
| qwen25_0p5b | mop_divpo | systems_thinker | 30 | 0.100 | 0.329 | 0.847 | 0.176 | 4.156 | 3.800 | 3.633 | 3.533 |
| qwen25_0p5b | mop_divpo | cross_domain_analogist | 30 | 0.070 | 0.331 | 0.861 | 0.150 | 3.911 | 3.800 | 3.633 | 2.233 |
| qwen25_0p5b | mop_divpo | minimalist | 30 | 0.079 | 0.448 | 0.906 | 0.170 | 3.878 | 3.467 | 3.400 | 3.300 |
| qwen25_0p5b | mop_divpo_v2 | contrarian | 30 | 0.096 | 0.305 | 0.847 | 0.180 | 3.956 | 3.500 | 3.567 | 2.900 |
| qwen25_0p5b | mop_divpo_v2 | systems_thinker | 30 | 0.078 | 0.351 | 0.864 | 0.161 | 3.811 | 3.700 | 3.400 | 3.400 |
| qwen25_0p5b | mop_divpo_v2 | cross_domain_analogist | 30 | 0.077 | 0.358 | 0.879 | 0.158 | 4.022 | 3.967 | 3.600 | 2.733 |
| qwen25_0p5b | mop_divpo_v2 | minimalist | 30 | 0.068 | 0.459 | 0.919 | 0.172 | 4.033 | 3.433 | 3.633 | 3.200 |
| qwen25_0p5b | mop_sft | contrarian | 30 | 0.115 | 0.286 | 0.814 | 0.143 | 2.367 | 4.467 | 1.900 | 2.633 |
| qwen25_0p5b | mop_sft | systems_thinker | 30 | 0.097 | 0.259 | 0.797 | 0.100 | 2.567 | 4.733 | 2.367 | 2.267 |
| qwen25_0p5b | mop_sft | cross_domain_analogist | 30 | 0.078 | 0.324 | 0.854 | 0.119 | 2.678 | 4.867 | 2.433 | 1.333 |
| qwen25_0p5b | mop_sft | minimalist | 30 | 0.078 | 0.468 | 0.918 | 0.150 | 2.922 | 4.733 | 2.533 | 2.533 |
| qwen25_0p5b | prompt_only | contrarian | 30 | 0.078 | 0.334 | 0.874 | 0.162 | 4.144 | 3.200 | 3.633 | 3.033 |
| qwen25_0p5b | prompt_only | systems_thinker | 30 | 0.099 | 0.323 | 0.841 | 0.166 | 4.056 | 3.467 | 3.567 | 3.333 |
| qwen25_0p5b | prompt_only | cross_domain_analogist | 30 | 0.079 | 0.354 | 0.876 | 0.163 | 3.956 | 3.800 | 3.667 | 2.600 |
| qwen25_0p5b | prompt_only | minimalist | 30 | 0.066 | 0.472 | 0.920 | 0.155 | 3.744 | 3.367 | 3.300 | 3.200 |
| qwen25_0p5b | single_lora | contrarian | 30 | 0.115 | 0.277 | 0.793 | 0.134 | 2.444 | 4.533 | 2.167 | 2.267 |
| qwen25_0p5b | single_lora | systems_thinker | 30 | 0.094 | 0.340 | 0.848 | 0.125 | 2.722 | 4.733 | 2.467 | 1.667 |
| qwen25_0p5b | single_lora | cross_domain_analogist | 30 | 0.091 | 0.333 | 0.863 | 0.126 | 2.567 | 4.733 | 2.233 | 1.567 |
| qwen25_0p5b | single_lora | minimalist | 30 | 0.072 | 0.396 | 0.897 | 0.131 | 2.622 | 4.600 | 2.367 | 2.400 |
| qwen25_1p5b | mop_divpo_v2 | contrarian | 30 | 0.090 | 0.318 | 0.861 | 0.195 | — | — | — | — |
| qwen25_1p5b | mop_divpo_v2 | systems_thinker | 30 | 0.086 | 0.316 | 0.855 | 0.197 | — | — | — | — |
| qwen25_1p5b | mop_divpo_v2 | cross_domain_analogist | 30 | 0.080 | 0.339 | 0.870 | 0.186 | — | — | — | — |
| qwen25_1p5b | mop_divpo_v2 | minimalist | 30 | 0.064 | 0.454 | 0.911 | 0.149 | — | — | — | — |
| qwen25_1p5b | mop_sft | contrarian | 30 | 0.087 | 0.307 | 0.839 | 0.127 | — | — | — | — |
| qwen25_1p5b | mop_sft | systems_thinker | 30 | 0.073 | 0.275 | 0.838 | 0.132 | — | — | — | — |
| qwen25_1p5b | mop_sft | cross_domain_analogist | 30 | 0.060 | 0.317 | 0.867 | 0.119 | — | — | — | — |
| qwen25_1p5b | mop_sft | minimalist | 30 | 0.083 | 0.516 | 0.932 | 0.124 | — | — | — | — |
| qwen25_1p5b | prompt_only | contrarian | 30 | 0.077 | 0.323 | 0.866 | 0.196 | — | — | — | — |
| qwen25_1p5b | prompt_only | systems_thinker | 30 | 0.084 | 0.317 | 0.859 | 0.199 | — | — | — | — |
| qwen25_1p5b | prompt_only | cross_domain_analogist | 30 | 0.062 | 0.366 | 0.895 | 0.168 | — | — | — | — |
| qwen25_1p5b | prompt_only | minimalist | 30 | 0.052 | 0.488 | 0.934 | 0.155 | — | — | — | — |
