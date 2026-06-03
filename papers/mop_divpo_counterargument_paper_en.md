# Mixture of Cognitive Personas with Diverse Preference Optimization for Diverse Counter-Argument Generation

<p class="paper-meta">Dason Tiovino<br>2026</p>

<div class="abstract">

**Abstract.** Aligned large language models are useful writing assistants, but their suggestions often converge toward the same safe and generic distribution. This work studies that problem in the concrete setting of counter-argument generation, where a useful pre-writing assistant should expose writers to alternative viewpoints rather than merely polish the most probable one. The proposed system combines a frozen Qwen2.5-0.5B-Instruct base model with four LoRA persona adapters: contrarian, systems thinker, cross-domain analogist, and minimalist. Each persona is first trained through supervised fine-tuning on a dataset selected for its cognitive style, then optimized through Diverse Preference Optimization (DivPO). The final evaluation compares six methods: base model, prompt-only persona conditioning, single LoRA, MoP-SFT, MoP+DivPO, and MoP+DivPO v2. The results support a more nuanced conclusion than the original proposal: the persona adapters are strongly distinct, with all off-diagonal SBERT cosine scores around 0.19-0.20, but raw adapter training can maximize diversity by sacrificing quality and utility. DivPO v2 gives the best balanced result, preserving prompt-only quality and utility while slightly improving novelty and lexical diversity. The paper therefore frames the contribution as an empirical study of the diversity-quality trade-off in small LLM counter-argument generation, rather than as an unconditional claim that DivPO wins every metric.

</div>

## 1. Introduction

Large language models (LLMs) are increasingly used as pre-writing assistants. A writer can ask for possible angles, objections, counter-arguments, outlines, or alternative framings before committing to a full draft. This is a high-leverage stage of writing because early ideation constrains the later argument: if the first set of ideas is generic, the final text may remain generic even after careful drafting and editing.

The central problem is not that current LLMs fail to produce fluent text. The problem is that they often produce similar ideas across users, prompts, and sampling attempts. Recent work describes this as a reduction in collective diversity: model assistance can improve individual fluency or perceived creativity while causing different users to converge toward similar outputs [1, 2]. Wu et al. define the broader phenomenon as generative monoculture, where a model's output distribution narrows relative to the diversity available in the underlying task space [3]. This is especially concerning for pre-writing, where the user needs divergent exploration rather than early convergence.

The original project proposal framed the system as a pre-writing ideation tool that combines Mixture of Persona (MoP) and Diverse Preference Optimization (DivPO). That framing remains a useful motivation, but the completed experiment has become more precise. The evaluated task is not a broad human writing study. It is counter-argument generation on CGA-CMV-derived and related prompts. This task is appropriate because counter-arguments are naturally measurable: an output can be judged for relevance, coherence, substance, novelty, usefulness, and persona fidelity. It also directly serves pre-writing because counter-arguments reveal hidden assumptions and alternative positions before the writer drafts.

This paper therefore asks:

1. Can separate cognitive persona adapters produce empirically distinct output distributions?
2. Does DivPO improve the diversity-quality balance relative to base, prompt-only, single-adapter, and MoP-SFT baselines?
3. What should be claimed honestly when diversity metrics and judge quality metrics disagree?

The answer is mixed but informative. The persona distinctness result is strong: all six persona pairs are below the 0.5 collapse threshold, and no pair approaches the 0.7 collapse threshold. The baseline comparison is more nuanced. Single LoRA and MoP-SFT produce the strongest semantic diversity and LLM-judge novelty, but their quality, utility, and persona fidelity are substantially lower. Prompt-only conditioning is strong on quality and utility. MoP+DivPO v2 gives the best balanced version of the proposed method: it keeps quality and utility near prompt-only while improving novelty and lexical diversity modestly. The project contribution is therefore not "DivPO dominates all baselines." The more defensible contribution is: cognitive personas are separable at small scale, and DivPO v2 helps recover usability while maintaining a useful diversity signal.

## 2. Related Work

### 2.1 LLM writing assistance and diversity loss

Doshi and Hauser show that generative AI can improve individual creative output while reducing the collective diversity of produced content [1]. Padmakumar and He similarly study whether writing with language models reduces content diversity, showing that LLM-assisted writing can make outputs more similar across writers [2]. These results motivate treating diversity as a central evaluation objective rather than a secondary aesthetic preference.

Generative monoculture gives a broader machine learning interpretation of the same concern [3]. If post-training alignment makes models concentrate probability mass around safe and preferred responses, repeated sampling may not be enough to recover the full diversity of plausible ideas. This matters for writing because the task is often not to find the single highest-rated continuation, but to expose a writer to multiple plausible framings.

### 2.2 Preference optimization and diversity

Direct Preference Optimization (DPO) and related post-training methods optimize a model toward preferred responses. These methods improve instruction following and preference alignment, but they can also sharpen the model distribution. Kirk et al. show that RLHF-style post-training affects both generalization and output diversity, making the diversity-quality trade-off an empirical question rather than a theoretical assumption [4].

Diverse Preference Optimization (DivPO) changes the construction of preference pairs: instead of always selecting the highest-quality candidate as preferred, it selects a rare but still acceptable candidate [5]. This makes DivPO attractive for ideation because novelty is not useful if it is incoherent, but quality is not enough if every output expresses the same framing.

### 2.3 LoRA, QLoRA, and mixture-style adaptation

LoRA freezes the base model and inserts trainable low-rank matrices, enabling parameter-efficient adaptation [6]. QLoRA extends efficient fine-tuning by combining low-bit quantization with LoRA-style adaptation [7]. These techniques make it feasible to train multiple small adapters even when full model fine-tuning is impractical.

Mixture-style adaptation, including MixLoRA, uses multiple LoRA-based experts and routing mechanisms to specialize behavior [8]. This project adopts the spirit of that architecture but uses cognitive personas rather than task-domain experts. The point is not only computational efficiency; it is behavioral separation. If a single prompt-conditioned model averages all styles into one voice, separate adapters may preserve sharper modes of reasoning.

## 3. Method

### 3.1 Model and persona architecture

The base model is `Qwen/Qwen2.5-0.5B-Instruct`. The system keeps the base model frozen and trains four LoRA adapters, each representing one cognitive style:

| Persona | Intended cognitive move | Training source |
|---|---|---|
| Contrarian | Challenge hidden assumptions and minority positions | Conversations Gone Awry - ChangeMyView |
| Systems thinker | Map causes, constraints, feedback loops, and second-order effects | StackExchange question-answering |
| Cross-domain analogist | Transfer mechanisms from one domain to another | arXiv abstracts |
| Minimalist | Remove secondary assumptions and expose the core disagreement | IBM argument quality ranking data |

This design follows a cognitive-fingerprint principle: each adapter should be trained on data whose natural discourse already resembles the target reasoning move. The contrarian persona uses CMV-style disagreement. The systems thinker uses explanatory Q&A, where answers often trace constraints and causal dependencies. The analogist uses research abstracts, where mechanisms and abstractions can be transferred across domains. The minimalist uses concise argument-quality data, emphasizing compact critique.

### 3.2 Training stages

The training pipeline has two stages. First, each persona receives supervised fine-tuning (SFT) on its persona-specific dataset. Second, each SFT adapter is optimized with DivPO. The original DivPO variant computes rarity against same-persona candidate siblings. DivPO v2 changes pair construction by computing rarity against the full cross-persona candidate pool for a shared prompt set. This matters because the evaluation measures diversity across persona outputs; therefore, the v2 training signal is more aligned with the final cross-persona evaluation.

The quality proxy also changes in v2. The first DivPO version included prompt-response embedding relevance. For counter-argument generation this can be misleading: a strong counter-argument may semantically oppose or reframe the claim rather than paraphrase it. DivPO v2 therefore drops the relevance term and uses a coherence-length proxy for the quality floor before selecting rare candidates.

### 3.3 Compared methods

The final baseline table compares six methods:

| Method | Description |
|---|---|
| Base | Qwen2.5-0.5B-Instruct without adapter or persona prompt |
| Prompt-only | Base model with persona instructions in the system prompt |
| Single LoRA | One LoRA adapter trained on all persona data merged |
| MoP-SFT | Four separate SFT persona adapters |
| MoP+DivPO | Four persona adapters after same-persona DivPO |
| MoP+DivPO v2 | Four persona adapters after cross-persona DivPO v2 |

For persona methods, each prompt produces one output per persona. For the base method, each prompt produces four stochastic outputs without persona context.

## 4. Experimental Setup

### 4.1 Persona distinctness validation

Before the baseline comparison, the experiment tests whether the four persona adapters produce distinct output distributions. The protocol uses 20 persona-neutral prompts. For each prompt and each persona, the model generates five outputs at temperature 0.9, giving 400 outputs in total. Outputs are embedded with `all-MiniLM-L6-v2`, and the mean cross-persona SBERT cosine is computed for each persona pair.

The decision rule is:

| Off-diagonal SBERT cosine | Interpretation | Action |
|---|---|---|
| Below 0.5 | Distinct personas | Keep both |
| 0.5 to 0.7 | Borderline distinct | Keep but report limitation |
| Above 0.7 | Mode-collapsed pair | Drop or merge one persona |

### 4.2 Baseline evaluation

The baseline evaluation uses 30 prompts and six methods. Each method generates 120 outputs, producing 720 outputs total. Automated diversity metrics are computed per prompt and aggregated across prompts. The main automated metrics are Self-BLEU, Distinct-1, Distinct-2, and pairwise SBERT cosine.

The project also uses LLM-as-judge scoring. Each output is scored on quality, novelty, pre-writing utility, and persona fidelity. Quality averages relevance, coherence, and substance. Novelty compares a target output against peer outputs for the same prompt. Utility asks whether the output would help a writer at the pre-writing stage. Persona fidelity asks whether an output expresses the requested cognitive style. The primary judge in the current result artifacts is OpenAI `gpt-4o-mini`, with `gpt-4o` used for calibration on a sample.

### 4.3 Inter-judge calibration

The calibration results show that the judge metrics should be interpreted carefully:

| Metric | Spearman rho | p-value | Interpretation |
|---|---:|---:|---|
| Quality | 0.6972 | 0.0000 | Borderline acceptable |
| Persona fidelity | 0.6200 | 0.0003 | Borderline acceptable |
| Novelty | 0.2659 | 0.0678 | Low agreement |
| Utility | 0.5546 | 0.0000 | Low-to-borderline agreement |

This means the paper should not overstate fine-grained novelty differences from the LLM judge. Quality and fidelity are more stable than novelty. Novelty remains useful as a directional indicator, but it should be triangulated with automated diversity metrics and qualitative inspection.

## 5. Results

### 5.1 Persona distinctness

The distinctness test strongly supports keeping all four personas:

| Persona pair | SBERT cosine |
|---|---:|
| Contrarian - Systems thinker | 0.2046 |
| Contrarian - Cross-domain analogist | 0.2030 |
| Systems thinker - Cross-domain analogist | 0.1973 |
| Contrarian - Minimalist | 0.1922 |
| Systems thinker - Minimalist | 0.1906 |
| Cross-domain analogist - Minimalist | 0.1904 |

All pairs are far below the 0.5 threshold. No pair is close to the 0.7 collapse threshold. This is the strongest architectural evidence in the project: separate persona adapters do not merely produce surface-level prompt variations; they produce measurably different output distributions under shared prompts.

### 5.2 Baseline table

| Method | Self-BLEU lower | Dist-1 higher | Dist-2 higher | SBERT lower | Quality higher | Novelty higher | Utility higher | Fidelity higher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.086 | 0.481 | 0.894 | 0.675 | 3.417 | 3.192 | 3.058 | - |
| Prompt-only | 0.062 | 0.576 | 0.923 | 0.708 | 3.975 | 3.458 | 3.542 | 3.042 |
| Single LoRA | 0.040 | 0.597 | 0.945 | 0.481 | 2.589 | 4.650 | 2.308 | 1.975 |
| MoP-SFT | 0.026 | 0.559 | 0.943 | 0.488 | 2.633 | 4.700 | 2.308 | 2.192 |
| MoP+DivPO | 0.056 | 0.556 | 0.920 | 0.705 | 3.931 | 3.642 | 3.542 | 3.025 |
| MoP+DivPO v2 | 0.051 | 0.580 | 0.931 | 0.690 | 3.956 | 3.650 | 3.550 | 3.058 |

The table shows three regimes.

First, the base model is weaker than persona-conditioned methods on most judge metrics and lexical diversity metrics. MoP+DivPO significantly improves over base on Self-BLEU, Distinct-1, Distinct-2, quality, novelty, and utility in the stored significance table, although not on SBERT cosine.

Second, Single LoRA and MoP-SFT achieve the strongest raw diversity. Their Self-BLEU and SBERT scores are much lower, and judge novelty is much higher. However, they pay for that diversity with low quality and utility. Their outputs are less useful to a writer and show weaker persona fidelity. This is not a small penalty; MoP-SFT has quality 2.633 and utility 2.308, while prompt-only has quality 3.975 and utility 3.542.

Third, MoP+DivPO and MoP+DivPO v2 occupy the balanced region. They do not beat the SFT variants on raw semantic diversity or judged novelty, but they recover quality and utility. The v2 variant is the best final version because it improves over the first DivPO variant on Distinct-1, Distinct-2, SBERT cosine, quality, novelty, utility, and fidelity. Against prompt-only, v2 is not a statistically decisive improvement, but it is numerically competitive while preserving the trained persona architecture.

### 5.3 Main empirical findings

**Finding 1: Persona separation works.** The distinctness matrix validates the MoP architectural premise. All four adapters remain distinct under the same prompts, so the system can legitimately claim inter-persona diversity.

**Finding 2: Diversity can be purchased too cheaply.** Single LoRA and MoP-SFT look excellent on diversity metrics but poor on quality and utility. This suggests that diversity alone is not enough for pre-writing assistance. A system that generates unusual but weak arguments is not useful.

**Finding 3: DivPO v2 is a balance mechanism, not a universal winner.** DivPO v2 does not dominate every metric. Its value is that it moves the model away from the low-quality adapter regime while keeping some diversity gain and preserving persona fidelity.

**Finding 4: Prompt-only remains a strong baseline.** This is important and should not be hidden. Prompt-only conditioning nearly matches or exceeds MoP+DivPO v2 on quality and utility. The trained system must therefore justify itself through distinctness, controllability, reproducibility, and future scalability, not just through one aggregate score.

## 6. Discussion

The most important lesson is that the project should not present diversity as a single scalar objective. There are at least three different forms of diversity in the results. Lexical diversity is measured by Distinct-1/2 and Self-BLEU. Semantic diversity is measured by SBERT cosine. Cognitive-style diversity is measured by persona distinctness and persona fidelity. A method can improve one while damaging another.

The SFT variants appear to increase semantic spread dramatically, but they also degrade quality. One interpretation is that the small base model and limited adapter capacity can move into more unusual regions of the output space, but not always with enough control to remain useful. This is consistent with the broader post-training trade-off: increasing diversity without a strong quality constraint can produce outputs that are different for the wrong reason.

DivPO v2 is conceptually better aligned with the task than the first DivPO implementation. Counter-argument generation does not reward semantic similarity to the prompt in the same way that summarization or answer generation might. A counter-argument should remain relevant, but it may explicitly reject or reframe the premise. Dropping prompt-response cosine from the quality proxy was therefore a justified methodological correction.

The current evidence supports a paper framed around the diversity-quality frontier. The x-axis is diversity, the y-axis is usability. Base sits in a low-to-middle region. Prompt-only is high usability but limited trained specialization. SFT adapters move toward high diversity but low usability. DivPO v2 moves back toward the high-usability region while retaining a measurable diversity signal. This is a coherent story and a more credible one than claiming the proposed method is best on every metric.

## 7. Limitations

The first limitation is model scale. Qwen2.5-0.5B-Instruct is useful for rapid experimentation, but it is small. Some weaknesses of SFT and DivPO may reflect limited base-model capacity rather than limitations of the architecture itself.

The second limitation is judge reliability. The calibration sample shows acceptable agreement for quality and persona fidelity but weak agreement for novelty. Novelty claims should therefore be treated as directional and supported with automated metrics and examples.

The third limitation is evaluation scope. The project evaluates counter-argument generation, not a full human writing workflow. Pre-writing remains the motivation, but the current paper should not claim measured improvements in final human writing quality.

The fourth limitation is dataset alignment. The persona datasets are chosen for cognitive fingerprints, but they are not all from the same domain. This helps style separation but may introduce domain artifacts. DivPO v2 partly addresses this by using a shared prompt pool, but future work should build larger and cleaner counter-argument datasets for all personas.

## 8. Conclusion

This project began as a proposal for a pre-writing ideation system combining Mixture of Persona and Diverse Preference Optimization. The completed results support a sharper and more defensible paper: a study of diverse counter-argument generation using cognitive persona adapters and DivPO training.

The strongest result is persona distinctness. Four separate LoRA adapters produce clearly separated output distributions. The second result is the diversity-quality trade-off. SFT-style adapter methods create high diversity but low quality and utility. The third result is the balanced role of DivPO v2. It does not win every metric, but it gives the most coherent final version of the proposed architecture: persona-conditioned, empirically distinct, quality-preserving, and modestly more diverse than prompt-only on several metrics.

The paper should emphasize this balanced conclusion. A credible negative or mixed result is stronger than an exaggerated positive one. The defensible contribution is not that MoP+DivPO solves LLM monoculture completely, but that it exposes how persona specialization and diversity-aware preference optimization interact at small scale, and it identifies DivPO v2 as a promising direction for future larger-model and human-writer evaluation.

## References

1. A. R. Doshi and O. P. Hauser, "Generative AI enhances individual creativity but reduces the collective diversity of novel content," *Science Advances*, 2024. https://www.science.org/doi/10.1126/sciadv.adn5290
2. V. Padmakumar and H. He, "Does Writing with Language Models Reduce Content Diversity?," ICLR, 2024. https://arxiv.org/abs/2309.05196
3. F. Wu, E. Black, and V. Chandrasekaran, "Generative Monoculture in Large Language Models," ICLR, 2025. https://arxiv.org/abs/2407.02209
4. R. Kirk et al., "Understanding the Effects of RLHF on LLM Generalisation and Diversity," ICLR, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/5a68d05006d5b05dd9463dd9c0219db0-Abstract-Conference.html
5. J. Lanchantin et al., "Diverse Preference Optimization," arXiv, 2025. https://arxiv.org/abs/2501.18101
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022. https://openreview.net/forum?id=nZeVKeeFYf9
7. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS, 2023. https://arxiv.org/abs/2305.14314
8. D. Li et al., "MixLoRA: Enhancing Large Language Model Fine-Tuning with LoRA-based Mixture of Experts," arXiv, 2024. https://arxiv.org/abs/2404.15159
9. J. Zhang et al., "Conversations Gone Awry: Detecting Early Signs of Conversational Failure," ACL, 2018. https://arxiv.org/abs/1805.05345
10. S. Gretz et al., "A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis," AAAI, 2020. https://arxiv.org/abs/1911.11408
11. Qwen Team, "Qwen2.5-0.5B-Instruct," Hugging Face model card, 2024. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
