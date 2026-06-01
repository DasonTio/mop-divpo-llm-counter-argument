# Research Plan — Journal Re-Scope and Execution

> **Purpose.** Translate the strategic re-scope (lessons 01-08 background, see also `docs/lessons/`) into an executable research plan. Every section below is something you actually do, in the order you do it. By the end of Phase 3 you have the numbers needed for a journal submission.

---

## 0. The decision being executed

Existing positioning: *"A pre-writing ideation system combining MoP and DivPO."*

New positioning: *"Diverse counter-argument generation via Mixture of Cognitive Personas and DivPO training, evaluated on CGA-CMV-derived prompts."*

Key moves:
- **Pre-writing** stays as motivation; drops from evaluation claims.
- **Counter-argument generation** becomes the concrete evaluation domain (countable, judgeable, defensible).
- **All four personas** stay in code; the experiment decides which survive into the paper's headline.
- **LLM-as-judge** replaces the human-writer study.
- **Co-author extension** (IdeaCard, sessions, Gradio) demoted to "deployment demonstration" appendix.

This plan does not throw any code away. It re-prioritises what becomes the paper's headline claim.

---

## 1. Phase 1 — Persona Distinctness Validation

**Goal.** Empirically test whether the four trained adapters produce measurably distinct output distributions. Decide which personas survive into the paper.

### 1.1 The test

For each pair `(persona_i, persona_j)`, generate outputs on the same prompts and measure semantic similarity. If a pair is too similar, the personas have collapsed into the same mode and one should be dropped or merged.

### 1.2 Test prompt set

A set of **persona-neutral** prompts — questions that do not lean toward any one cognitive style. The goal is to give each persona a fair chance to express its style on a shared input.

Suggested initial set (you can expand to 30-50 for stronger signal):

```python
PERSONA_DISTINCTNESS_PROMPTS = [
    "What is the future of remote work?",
    "How should cities adapt to climate change?",
    "What's wrong with current education systems?",
    "Why do startups fail in the first three years?",
    "How can governments handle aging populations?",
    "What makes a city worth living in?",
    "Why are mental health rates worsening globally?",
    "What's the right way to evaluate AI safety?",
    "How should universities change in the next decade?",
    "Why is public transit underfunded?",
    "What does meaningful work look like in 2035?",
    "Why is housing unaffordable in major cities?",
    "How should social media platforms be regulated?",
    "Why do diets fail for most people?",
    "What is the role of religion in modern society?",
    "How do we fix scientific peer review?",
    "Why is creative writing declining in schools?",
    "What's wrong with how we measure economic growth?",
    "How should we think about generational wealth?",
    "Why is loneliness rising in connected societies?",
]
```

Properties:
- All open-ended.
- None use persona-trigger keywords (no "challenge," "system," "feedback," "minimal," "analogy" — would bias the gate).
- Topic-diverse so a persona that only specialises in one topic gets exposed.

### 1.3 Protocol

```
For each prompt in PERSONA_DISTINCTNESS_PROMPTS:        (20 prompts)
    For each persona in PERSONA_IDS:                     (4 personas)
        Generate N=5 outputs with temperature=0.9
        Store as (prompt_id, persona, output_index, text)
```

Total outputs: `20 × 4 × 5 = 400`.

### 1.4 Metric — pairwise inter-persona SBERT cosine

For each prompt, compute the mean cosine similarity between outputs from `persona_i` and outputs from `persona_j` (cross-persona, not within-persona). Average across all prompts. Result: a 4×4 symmetric matrix.

```
                contrarian  analogist  systems  minimalist
contrarian          1.00        ?         ?         ?
analogist             ?       1.00        ?         ?
systems               ?         ?       1.00        ?
minimalist            ?         ?         ?       1.00
```

Off-diagonal values are what we care about.

### 1.5 Decision rule

| Off-diagonal value | Interpretation | Action |
|---|---|---|
| `< 0.5` | Genuinely distinct personas | Keep both |
| `0.5 – 0.7` | Borderline distinct | Keep both, report as a limitation |
| `> 0.7` | Mode-collapsed pair | Drop one, or merge into one persona |

If all six off-diagonal pairs are below 0.5, the MoP architectural claim is empirically validated. If some are above 0.7, you have a re-scope decision: which one to drop.

### 1.6 Expected risk

Based on Lesson 06's honest assessment, the highest-risk pair is **minimalist ↔ contrarian** (both can produce "less is more" rhetoric) and **analogist ↔ systems_thinker** (both can produce structural explanations). Test these specifically.

### 1.7 Where the code goes

New script: `scripts/experiment_persona_distinctness.py`.

Skeleton:
```python
from mop_divpo.inference.generate import generate
from mop_divpo.metrics.semantic import cosine_similarity_matrix
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# 1. Generate N outputs per (prompt, persona)
# 2. Embed all outputs with all-MiniLM-L6-v2
# 3. For each prompt and persona pair, compute mean cross-pair cosine
# 4. Aggregate into 4x4 matrix
# 5. Print matrix + write JSON to outputs/experiments/persona_distinctness.json
```

### 1.8 Time budget

- Generation: ~30 min on Colab T4 (400 outputs at ~5 sec each).
- Embedding + matrix: <1 min.
- Total: under an hour of compute.

---

## 2. Phase 2 — LLM-as-Judge Framework

**Goal.** Replace the human-writer study with a structured LLM judgment pipeline that produces quality, diversity, and persona-fidelity scores per output.

### 2.1 Choice of judge model

| Option | Cost per 1K calls | Notes |
|---|---|---|
| `gpt-4o` | ~$5-10 | Strong all-rounder, well-documented for judge use |
| `claude-sonnet-4` | ~$5-15 | Strong on nuanced rubrics, supports JSON mode |
| `gpt-4o-mini` | ~$0.50 | 10× cheaper, weaker on subtle judgments |

Recommendation: `claude-sonnet-4` or `gpt-4o`. Reviewers expect a strong judge. Mini models invite "but you used a weak judge" critique.

**Critical principle:** the judge must be *stronger* than the model being evaluated. Your base is Qwen 0.5B. Both options above are dramatically stronger. Good.

### 2.2 The four rubric prompts

Each output gets judged on four axes. All prompts return structured JSON for cheap parsing.

#### Rubric 1 — Quality (replaces the proxy)

```
You are an expert evaluator scoring the quality of a counter-argument generated for a writer who is brainstorming.

Original topic: {prompt}
Generated counter-argument: {output}

Score on three sub-criteria, each 1-5 (1 = poor, 5 = excellent):

1. RELEVANCE — does the output address the topic?
2. COHERENCE — is the argument internally consistent and well-formed?
3. SUBSTANCE — does the output offer a real point, not platitudes?

Return JSON only:
{"relevance": int, "coherence": int, "substance": int, "rationale": "one sentence"}
```

#### Rubric 2 — Persona fidelity

```
You are an expert evaluator checking whether a generated response demonstrates a specific cognitive style.

Cognitive style: {persona_description}
Generated response: {output}

Score 1-5 on how strongly the response exhibits the cognitive style:
1 = does not exhibit the style at all
3 = somewhat exhibits the style
5 = clearly and strongly exhibits the style

Return JSON only:
{"persona_fidelity": int, "evidence": "quote one phrase from the output that demonstrates the style, or 'none'"}
```

Use the persona's `PERSONA_ROUTING_DESCRIPTIONS` string from `personas.py` as `{persona_description}`.

#### Rubric 3 — Novelty against peers

```
You are an expert evaluator measuring how novel a counter-argument is compared to other counter-arguments on the same topic.

Topic: {prompt}
Target argument: {target_output}
Other arguments on the same topic:
1. {peer_1}
2. {peer_2}
3. {peer_3}

Score the target argument 1-5 on novelty relative to the others:
1 = essentially repeats one of the others
3 = different surface words but similar conceptual point
5 = makes a substantively different argument

Return JSON only:
{"novelty": int, "most_similar_peer_index": int_or_null, "rationale": "one sentence"}
```

#### Rubric 4 — Pre-writing utility (the bridge to the original framing)

```
You are an expert writing coach evaluating whether a brainstormed counter-argument would be useful to a writer at the pre-writing stage.

Writer's topic: {prompt}
Generated counter-argument: {output}

Score 1-5 on pre-writing usefulness:
1 = the writer would discard this immediately
3 = the writer might use this as a starting point
5 = the writer would clearly benefit from following this direction

Return JSON only:
{"prewriting_utility": int, "rationale": "one sentence"}
```

This is the rubric that closes the gap with the original pre-writing claim — without requiring actual writers.

### 2.3 Per-output scoring shape

After running all four rubrics, every output produces a structured record:

```json
{
  "output_id": "base__remote_work__0",
  "method": "base",
  "prompt": "What is the future of remote work?",
  "persona": null,
  "output": "...",
  "quality": {"relevance": 4, "coherence": 4, "substance": 3, "rationale": "..."},
  "persona_fidelity": {"persona_fidelity": 2, "evidence": "..."},
  "novelty": {"novelty": 3, "most_similar_peer_index": 1, "rationale": "..."},
  "utility": {"prewriting_utility": 3, "rationale": "..."}
}
```

### 2.4 Cost estimate

| Variable | Count |
|---|---|
| Methods (baselines) | 5 |
| Prompts | 30 (recommended; can start with 20) |
| Outputs per (method, prompt) | 4 |
| Rubrics per output | 4 |
| Total judge calls | `5 × 30 × 4 × 4 = 2,400` |

At ~$0.005 per Sonnet/GPT-4 call (with prompt caching for the rubric template), this runs **$10-25 total**. Affordable.

### 2.5 Where the code goes

New module: `src/mop_divpo/eval/llm_judge.py`.

Skeleton:
```python
import anthropic  # or openai
import json
from typing import Literal

RUBRICS = {
    "quality": "...",           # the four prompts above
    "persona_fidelity": "...",
    "novelty": "...",
    "utility": "...",
}

def judge_output(output, prompt, persona, peers, rubric_name):
    """Run one rubric on one output. Returns parsed JSON dict."""
    # Build the prompt, call the API, parse JSON, return dict.

def score_all_outputs(outputs):
    """Run all rubrics on all outputs. Returns a list of scored records."""
    # Iterate, accumulate, write to outputs/evaluation/llm_judge_scores.jsonl
```

Important: **cache the judge calls.** API calls are deterministic with `temperature=0`. Use a simple JSON-file cache keyed by `(rubric_name, prompt, output_hash)`. Re-running the eval should be free after the first pass.

### 2.6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Judge bias toward verbose outputs | Add length to the rubric: instruct judge to ignore length. Report length stats separately. |
| Judge bias toward its own outputs | The judge (Claude/GPT) does not produce the test outputs (Qwen does). Low risk. |
| Judge inconsistency across runs | Use `temperature=0`. Re-run a 10% sample to estimate consistency. Report inter-judge agreement. |
| Single-judge dependency | Optionally run a second judge (the other API) on a 10% sample. Report correlation. |

---

## 3. Phase 3 — The Baseline Evaluation Table

**Goal.** Produce the table that anchors the paper. Five methods × seven metrics × statistical comparison.

### 3.1 The five methods (already defined in `lessons/07-evaluation-strategy.md`)

1. **Base** — Qwen 0.5B Instruct, no adapter, no persona instruction.
2. **Prompt-only** — Qwen + persona instruction in system prompt (no fine-tuning).
3. **Single LoRA** — one adapter trained on all four persona datasets merged.
4. **MoP SFT** — four separate adapters, SFT only, no DivPO.
5. **MoP + DivPO** — the full method.

For methods 2-5, the same prompt is run through all four personas (or the routed top-k = 2). For method 1, the prompt is run with no persona context.

### 3.2 The seven metrics

| Metric | Source |
|---|---|
| Self-BLEU ↓ | `metrics/diversity.py` (existing) |
| Distinct-1 ↑ | `metrics/diversity.py` (existing) |
| Distinct-2 ↑ | `metrics/diversity.py` (existing) |
| SBERT pairwise cosine ↓ | `metrics/semantic.py` (existing) |
| Corpus novelty (min, mean) ↑ | `eval/evaluate.py` (existing) |
| LLM-judge quality ↑ | `eval/llm_judge.py` (new — Phase 2) |
| LLM-judge novelty ↑ | `eval/llm_judge.py` (new — Phase 2) |
| LLM-judge utility ↑ | `eval/llm_judge.py` (new — Phase 2) |
| LLM-judge persona fidelity ↑ | `eval/llm_judge.py` (new — Phase 2) |

(Quality proxy from the old `eval/evaluate.py` becomes a supporting number, not a headline.)

### 3.3 The protocol

```
For each method in 1-5:
    For each prompt in EVALUATION_PROMPTS (30):
        Generate 4 outputs with temperature=0.9
        (For methods 2-5: 1 output per persona = 4 outputs total per prompt)
        (For method 1: 4 outputs from base model)
    
    Compute all 7 automated metrics on the 30*4 = 120 outputs.
    Run LLM-as-judge on all 120 outputs (Phase 2 pipeline).
    Aggregate scores per method.

Write final table to outputs/evaluation/baseline_table.{json,csv,md}.
```

### 3.4 Test prompt set

Use the 20 distinctness prompts from §1.2 plus 10 additional CGA-CMV-style prompts (sampled from real Reddit r/ChangeMyView titles that are *not* in the training set).

The 10 additional prompts can be hand-curated from CMV — write them as if they were Reddit thread titles to keep the contrarian persona's expected behaviour aligned.

### 3.5 Statistical reporting

For each metric and each pair of methods, compute:
- Mean ± std across the 30 prompts (treat each prompt as one observation).
- Paired bootstrap test (1000 resamples) for `MoP+DivPO` vs each weaker baseline.
- Report `p`-values and effect sizes.

Reviewers will ask. Have it ready.

### 3.6 The table you produce

```
Method        | Self-BLEU↓ | SBERT↓ | Dist-1↑ | NoveltyL↑ | QualityL↑ | UtilityL↑ | FidelityL↑
Base          |    0.xx    |  0.xx  |  0.xx   |    x.x    |    x.x    |    x.x    |    x.x
Prompt-only   |    0.xx    |  0.xx  |  0.xx   |    x.x    |    x.x    |    x.x    |    x.x
Single LoRA   |    0.xx    |  0.xx  |  0.xx   |    x.x    |    x.x    |    x.x    |    x.x
MoP SFT       |    0.xx    |  0.xx  |  0.xx   |    x.x    |    x.x    |    x.x    |    x.x
MoP + DivPO   |    0.xx    |  0.xx  |  0.xx   |    x.x    |    x.x    |    x.x    |    x.x
            (L = LLM-judge score; ↓ = lower is better; ↑ = higher is better)
```

**The paper lives or dies on this table.** If MoP+DivPO beats all four baselines on at least 4 of 7 metrics with statistical significance, you have a publishable result.

### 3.7 Time and compute budget

| Step | Compute | Wall time |
|---|---|---|
| Generate 120 × 5 = 600 outputs | Colab T4 | 1-2 hours |
| Automated metrics | CPU | <10 min |
| LLM judge (2,400 calls cached) | API | 1-2 hours (rate-limited) |
| Statistical analysis | CPU | <5 min |
| **Total** | | **half a day** |

This is the missing half-day that gates the paper.

---

## 4. Phase 4 — Paper Scoping Decisions

After Phases 1-3 produce numbers, lock the scope.

### 4.1 What goes in the paper

| Section | Content |
|---|---|
| 1 — Introduction | Motivate via generative monoculture in pre-writing. Cite Wu 2025. |
| 2 — Related Work | DivPO (Lanchantin 2025), MixLoRA (Li 2024), LoRA (Hu 2022), creativity in LLMs. |
| 3 — Method | MoP architecture + DivPO objective + persona descriptions + datasets. |
| 4 — Experimental Setup | Base model, hyperparameters, evaluation prompts, judge model, rubrics. |
| 5 — Results | The baseline table from §3.6 + the distinctness matrix from §1.4. |
| 6 — Analysis | Why MoP+DivPO wins where it does. Qualitative examples. |
| 7 — Limitations | LLM-judge limits, single base model, small prototype scale. |
| 8 — Future Work | Human-writer evaluation, larger base model, real reward model. |

### 4.2 What stays out of the paper (kept in code, mentioned briefly or in appendix)

- IdeaCard schema and co-author multi-turn — one paragraph in §8 as "system extension."
- Gradio demo / web UI — single sentence + URL in §1 or footer.
- WritingBrief structure — appendix only.
- PersonaSession — appendix only.
- All routing gate variants except the one used for the headline results.

### 4.3 Persona reduction decision (depends on §1.4 results)

| If §1.4 shows... | Then... |
|---|---|
| All four pairs < 0.5 | Keep all four. Report MoP with all four as headline. |
| One pair > 0.7 | Drop the weaker persona. Report MoP with three. Mention the dropped persona in §6 (Analysis) as an honest negative. |
| Two or more pairs > 0.7 | Drop down to two personas (likely contrarian + one other). Re-frame the paper title around "two cognitive personas." |

Decide *after* the data, not before.

---

## 5. Code Deliverables Checklist

Concrete files to create or update, in order.

| # | File | Action | Owner | Status |
|---|---|---|---|---|
| 1 | `scripts/experiment_persona_distinctness.py` | Create | New | DONE |
| 2 | `src/mop_divpo/eval/llm_judge.py` | Create | New | DONE |
| 3 | `src/mop_divpo/eval/llm_judge_rubrics.py` | Create — store rubric prompts as constants | New | DONE |
| 4 | `scripts/run_baseline_evaluation.py` | Create — wires generation + metrics + judge into the table | New | DONE |
| 5 | `tests/test_llm_judge.py` | Create — unit-test rubric parsing and cache behaviour | New | DONE |
| 6 | `outputs/experiments/persona_distinctness.json` | Generated by #1 | Output | TODO (run on GPU) |
| 7 | `outputs/evaluation/baseline_table.{json,csv,md}` | Generated by #4 | Output | TODO (run on GPU) |
| 8 | `docs/research-report.md` | Update to reflect re-scope after results land | Edit | TODO |
| 9 | `README.md` | Update §Status and §Evaluation Metrics after results land | Edit | TODO |

Supporting modules built alongside the checklist above (prerequisites the plan
assumed existed): `src/mop_divpo/inference/generate.py` (all 5 methods, one code
path), `src/mop_divpo/metrics/{semantic,diversity}.py`, `src/mop_divpo/eval/{prompts,aggregate}.py`,
`scripts/train_single_lora.py` (the Single-LoRA baseline, method 3). Run order and
per-resource commands: `docs/EVAL_RUNBOOK.md`.

---

## 6. Sequencing and Critical Path

```
Day 1: Phase 1 — persona distinctness
       └─→ DECISION: which personas survive
                │
                ▼
Day 2: Phase 2 — LLM-as-judge framework
       ├─ Implement rubrics (half day)
       └─ Validate on 5 outputs by hand (half day)
                │
                ▼
Day 3-4: Phase 3 — baseline evaluation
       ├─ Generate 600 outputs (1-2 hours compute)
       ├─ Run all metrics (10 min)
       ├─ Run LLM-judge (1-2 hours API)
       └─ Build the table
                │
                ▼
Day 5+: Paper drafting (separate effort)
```

Critical path: **Phase 1 gates everything else.** If the personas collapse, the whole paper's scope changes. Do it first. Do not start Phase 2 or 3 until you have the distinctness matrix.

---

## 7. Risks That Could Sink the Plan

| Risk | Likelihood | Mitigation |
|---|---|---|
| Persona distinctness all > 0.5 (no personas are sharp) | Medium | Re-train with more aggressive SFT (more steps, higher LR), or admit MoP is too weak at this scale and pivot paper to DivPO-only |
| MoP+DivPO does not beat MoP-SFT on any metric | Medium | DivPO weights are mis-tuned. Re-run with `rarity_weight=0.8`. If still no effect, paper becomes "negative result on DivPO at small scale." Still publishable. |
| LLM-judge produces noisy/inconsistent scores | Low | Run inter-judge agreement on a 10% sample. If correlation < 0.5, switch judge model or refine rubrics. |
| Adapters are not actually trained | Low | The repo shows `coauthor_adapters/` and `divpo_adapters/` directories. Verify they load successfully before Phase 3 starts. |
| Compute exhaustion | Low | Total compute is half a day on T4. Well within Colab free tier. |

Negative results are publishable if honestly framed. "We tried MoP+DivPO on a 0.5B base and found no significant diversity gain over MoP-SFT — suggesting DivPO requires larger models to manifest its benefit" is a real contribution.

---

## 8. What This Plan Does Not Do

To stay honest:

- **No new training.** The plan assumes existing trained adapters work. If they do not, training is a prerequisite outside this plan's scope.
- **No human study.** LLM-judge is the substitute. Reviewers may ask for human eval as future work.
- **No reward model training.** The quality measure is the LLM-judge quality rubric, not a learned reward model.
- **No web/product evaluation.** The Gradio demo and co-author session are demoted to appendix mentions.
- **No new datasets.** The 30 evaluation prompts come from the existing distinctness set + hand-curated CMV-style prompts.

These are all reasonable scope cuts for a course-project-to-journal upgrade.

---

## 9. Success Criteria

You have a publishable paper if and only if:

1. The distinctness matrix shows at least 2 personas with pairwise cosine < 0.5.
2. The baseline table shows MoP+DivPO beating Base, Prompt-only, and Single LoRA on at least 3 of: Self-BLEU, SBERT pairwise, LLM-judge novelty, LLM-judge utility.
3. MoP+DivPO does not lose to MoP-SFT on quality (LLM-judge quality) by more than 0.3 points on the 1-5 scale.
4. Statistical significance: p < 0.05 on at least one of the metrics in criterion 2.

Hit all four → submit to a workshop or journal. Miss any → understand which one and pivot the paper's claim accordingly.

---

## 10. The First Concrete Action

Do **only** this first, before writing any other code:

1. Verify trained adapters load: `python -c "from mop_divpo.inference.generate import generate; print(generate(prompt='test', personas=['contrarian'], adapter_prefix='DasonTio/mop-divpo-coauthor', adapter_stage='sft'))"`
2. If that works → start `scripts/experiment_persona_distinctness.py`.
3. If that fails → fix the adapter loading first. No other work is meaningful until inference works on real adapters.

That single command tells you whether you have a working starting point.

---

*Plan owner: DasonTiovino · Plan generated: 2026-05-25 · Re-scope target: counter-argument generation via MoP+DivPO with LLM-as-judge evaluation*
