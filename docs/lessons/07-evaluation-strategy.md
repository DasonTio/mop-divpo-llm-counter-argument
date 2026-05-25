# Lesson 07 — Evaluation Strategy

> **Goal of this lesson.** A diversity-focused system that cannot prove it produces diverse outputs is just an unverified hope. This lesson walks through the evaluation strategy: the central tension (diversity vs. quality), the five metrics chosen, the baselines the project compares against, and the honest limits of what can be measured at the prototype stage. By the end you should be able to read a metrics table from this project and know whether the numbers actually support a diversity claim.

---

## 1. Why evaluating creativity is harder than evaluating accuracy

Most ML evaluation tasks are easy in a specific sense: they have a *ground truth*. Image classification has a right label. Translation has a reference translation. Question answering has a correct answer. You compare model output to the truth and report a number.

Creativity has no ground truth. There is no "correct" set of ideas for an essay on AI in education. Any evaluation has to operate without a reference — which means we have to measure *properties* of the output set instead of correctness of individual outputs.

This forces a methodological shift. Instead of asking "is this output right?" the project asks **two simultaneous questions**:

1. **Are the outputs different from each other?** (diversity)
2. **Are the outputs still relevant to the prompt?** (quality)

A diverse-but-irrelevant system is just a random text generator. A relevant-but-monotonous system is the failure mode this project exists to fix. The interesting region is the **upper-right corner** of the diversity-quality plane:

```
        ▲ Quality
        │
 great  │      ★ here is where we want to land
        │
        │
        │
        │ ░ standard RLHF model
        │
        │ ▒ random text generator
        │
        └───────────────────────────▶ Diversity
        low                       high
```

Every metric the project uses is either a *diversity* metric or a *quality* metric. The job of evaluation is to show that the project's outputs move up and to the right at the same time.

---

## 2. The five metrics, mapped to the two axes

| Metric | Axis | Better is | Catches |
|---|---|---|---|
| **Distinct-1** | Diversity (lexical) | Higher | Low vocabulary variety |
| **Distinct-2** | Diversity (lexical) | Higher | Low phrase variety |
| **Self-BLEU** | Diversity (token-level) | Lower | Token-level output similarity |
| **SBERT pairwise cosine** | Diversity (semantic) | Lower | Semantic redundancy across outputs |
| **Corpus novelty** | Diversity (vs. training data) | Higher | Training example memorisation |
| **Quality proxy (prompt-idea cosine)** | Quality | Higher | Off-topic outputs |

Notice the spread. Three different levels of diversity (lexical, token-level, semantic), one anti-memorisation check, one quality floor. None of these on its own is sufficient — they cover different failure modes and must be reported *together*.

The rest of this lesson takes each one in turn.

---

## 3. Distinct-1 and Distinct-2 — the standard lexical diversity baseline

### 3.1 What they measure

The fraction of *unique* n-grams across all outputs.

```
Distinct-n = |unique n-grams in pooled outputs| / |total n-grams in pooled outputs|
```

For Distinct-1, n-grams are individual words. For Distinct-2, they are pairs of consecutive words (bigrams). The range is 0 (every n-gram appears many times — total repetition) to 1 (every n-gram appears exactly once — total variety).

### 3.2 The code

From `src/mop_divpo/metrics/diversity.py`:

```python
def distinct_n(outputs: list[str], n: int) -> float:
    ngrams: list[tuple[str, ...]] = []
    for output in outputs:
        tokens = _tokens(output)
        ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)
```

Read it like a checklist: tokenise, slide an n-gram window across each output, pool all n-grams together, return `|unique| / |total|`. Eight lines, no dependencies.

### 3.3 Why this is reported but not relied on

Distinct-n is the *oldest* established diversity metric in text generation (it has been used since the early 2010s). The project includes it because every comparable paper uses it and reviewers expect it.

But Distinct-n has a famous weakness: **it confuses lexical variety with conceptual variety.** Two outputs that say the same thing using different vocabulary will score high on Distinct-n while being semantically identical. This is why the project always reports Distinct-n alongside SBERT cosine.

> **Rule of thumb.** A high Distinct-n score with a high SBERT cosine score means the model is *paraphrasing itself* — superficially diverse but conceptually monotonous. A high Distinct-n with a *low* SBERT cosine is the real thing.

---

## 4. Self-BLEU — the canonical diversity benchmark

### 4.1 What it measures

For each output, compute how much of its token content overlaps with *all the other outputs* treated as references. Average across all outputs.

```
Self-BLEU = mean over outputs of: (overlap tokens with others) / (total tokens in output)
```

Range: 0 (no shared tokens between any two outputs) to 1 (all outputs identical).

### 4.2 The code

```python
def self_bleu(outputs: list[str]) -> float:
    if len(outputs) < 2:
        return 0.0
    scores = []
    for index, output in enumerate(outputs):
        refs = outputs[:index] + outputs[index + 1 :]
        scores.append(_overlap_score(output, refs))
    return sum(scores) / len(scores)
```

`_overlap_score` counts how many of a candidate's tokens appear *anywhere* in the references. This is a simplified Self-BLEU — the formal version uses n-gram overlaps and BLEU smoothing — but it captures the same idea with far fewer dependencies. For a prototype it is the right call.

### 4.3 Why it is a flagship metric

Self-BLEU was introduced *specifically for diversity evaluation* in text generation (Zhu et al., 2018, "Texygen"). Lower is better. **It is the standard benchmark metric for monoculture.**

If a model produces ten almost-identical outputs, Self-BLEU will be near 1.0. If it produces ten genuinely different outputs, Self-BLEU drops to 0.2-0.4 depending on shared topical vocabulary.

There is a direct connection to training. **DivPO's rarity score in `divpo/scoring.py` is derived from the same token-overlap principle as Self-BLEU.** This is not a coincidence — DivPO is, in a sense, training the model to score well on Self-BLEU. The evaluation metric reflects the training objective.

That tight coupling is a strength (the training signal aligns with the eval signal) and a weakness (the model could in principle game the metric). The project mitigates this by also reporting SBERT cosine (semantic, not token-based) and corpus novelty (training-data-relative).

---

## 5. SBERT pairwise cosine — the semantic check

### 5.1 What it measures

Embed every output with a Sentence-BERT model (`all-MiniLM-L6-v2`). Compute pairwise cosine similarity. Take the mean of the upper triangle of the similarity matrix.

```
SBERT pairwise mean = mean of cosine_sim(embed(o_i), embed(o_j)) for all i < j
```

Range in practice: 0 (semantically unrelated outputs) to 1 (identical meaning).

### 5.2 The code

From `src/mop_divpo/metrics/semantic.py`:

```python
def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / safe
    return normalized @ normalized.T


def average_pairwise_cosine(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    matrix = cosine_similarity_matrix(embeddings)
    upper = matrix[np.triu_indices(len(embeddings), k=1)]
    return float(upper.mean()) if len(upper) else 0.0
```

The full machinery in eight lines. The `np.triu_indices(..., k=1)` call grabs the upper-triangle pairs of the similarity matrix, excluding the diagonal (which is always 1.0 — every embedding has perfect similarity with itself).

### 5.3 Why this is the most important diversity metric for this project

Lexical metrics (Distinct-n, Self-BLEU) can be gamed by paraphrasing. Semantic metrics (SBERT cosine) cannot — two outputs that mean the same thing have nearly identical embeddings regardless of word choice.

For an ideation system, semantic diversity is *the* metric. The user does not care whether the model used different verbs. The user cares whether the *ideas* are different. SBERT cosine is the closest cheap proxy for that.

**Why `all-MiniLM-L6-v2` specifically.** It is small (80 MB), fast (CPU-friendly), and trained on a diverse contrastive objective that makes its embeddings useful for general-purpose semantic similarity. It is also already loaded as part of the `CorpusRetriever` (Lesson 05), so the dependency is free.

---

## 6. Corpus novelty — the anti-memorisation guard

### 6.1 What it measures

For each generated output, compute its *maximum* cosine similarity to any document in the training corpus. The novelty score is `1 - max_similarity`.

```
corpus_novelty(output) = 1 - max(cosine_sim(output, doc) for doc in corpus)
```

Two aggregates are reported:

- **Mean novelty** — average across all outputs.
- **Minimum novelty** — the *worst* output's novelty.

### 6.2 Why this is critical for an ideation system

The other diversity metrics check whether outputs differ *from each other*. They cannot detect a much worse failure: the model regurgitating training examples almost verbatim.

If a Reddit reply about "growth assumptions" happens to be in the contrarian training set, and the model produces it almost word-for-word when asked about growth — every other metric will be happy. Self-BLEU is fine (only one output exists in the eval batch). SBERT pairwise is fine (it is the only one). But the output is *useless* because it is plagiarised.

Corpus novelty catches exactly this. If any output has near-1 similarity to a training example, novelty drops to near 0.

The minimum novelty is the headline number. A model with `corpus_novelty_min = 0.05` has at least one output that is essentially copied from training data — even if the average is healthy. **Always report both mean and min.**

### 6.3 The connection to runtime PRIOR ART injection

Lesson 05 mentioned that the `CorpusRetriever` injects the top-3 most similar training examples into the system prompt at inference and instructs the model not to repeat them. This is a *direct intervention* designed to keep corpus novelty high.

The metric and the intervention are two sides of the same idea. The retriever knows what is in the training data; we use that knowledge both to nudge the model away from it at runtime *and* to check that the nudge worked at evaluation time.

---

## 7. Quality proxy — the relevance floor

### 7.1 What it measures

Cosine similarity between the embedding of the user's prompt and the embedding of each generated idea. Averaged across outputs.

```
quality_proxy = mean over outputs of: cosine_sim(embed(prompt), embed(output))
```

Range: 0 (output is unrelated to the prompt) to 1 (output is highly aligned with the prompt).

### 7.2 Why it is a "proxy" and not a true quality score

This is *not* a quality metric in the sense of "is this idea any good?" That requires either a trained reward model or human judgements, neither of which the prototype has.

What it *is* is a **relevance floor**: a check that the output has not drifted off-topic. A model could achieve perfect Self-BLEU and zero SBERT cosine by generating random text — but the prompt-idea cosine would also be zero. The quality proxy is the bulwark against "diversity by gibberish."

The README is honest about this:

> *This is a proxy, not a reward model. … A proper evaluation would use human judgments or a trained reward model. For the prototype stage, the proxy is sufficient to detect complete topic drift.*

Treat the quality proxy as a *necessary but not sufficient* condition. A high quality proxy does not prove the ideas are good — it only proves they are on-topic. A *low* quality proxy is conclusive evidence that something is wrong.

---

## 8. The baselines — what the project compares itself against

A diversity claim is meaningless without comparison. The project's baselines are:

1. **Base model (no fine-tuning).** Vanilla `Qwen/Qwen2.5-0.5B-Instruct`. Establishes the starting-point diversity.
2. **Prompt-based diversity.** Same base model, with verbalised persona instructions in the system prompt. Tests whether the gains come from fine-tuning or whether prompting alone is enough.
3. **Single LoRA (no persona split).** One LoRA adapter trained on all four datasets merged. Tests the architectural claim of Lesson 03: does specialisation require isolation?
4. **MoP SFT (no DivPO).** Four persona adapters, SFT only, no DivPO phase. Isolates DivPO's contribution.
5. **MoP + DivPO (full method).** The complete system.

This baseline ladder is the project's claim of scientific rigor. Each step adds exactly one ingredient and lets you read off that ingredient's effect:

| Comparison | Tests |
|---|---|
| Baseline 1 vs Baseline 2 | Does prompt engineering alone improve diversity? |
| Baseline 2 vs Baseline 3 | Does any fine-tuning help, even without persona separation? |
| Baseline 3 vs Baseline 4 | Does *separating* the personas (MoP) help? |
| Baseline 4 vs Baseline 5 | Does DivPO add intra-persona diversity on top of MoP? |

A well-presented evaluation should show all five rows. Without the ladder, you cannot tell whether the gains come from MoP, DivPO, or both — and "both" is the only answer that justifies the full system.

### 8.1 The ablation gates

Lesson 03 mentioned that `routing/gate.py` includes a `RandomPersonaGate`. This is the routing equivalent of the baseline ladder. Comparing the embedding gate against the random gate tells you whether the routing is doing real work or whether any routing-shaped function would suffice.

This is honest practice. Most projects do not include a "random" baseline because it is embarrassing if random performs nearly as well. Including it as a first-class implementation says: *we want to know if we are fooling ourselves.*

---

## 9. The evaluation runner

The runner that ties all of this together is `src/mop_divpo/eval/evaluate.py`. Its core function:

```python
def evaluate_outputs(outputs, retriever):
    """Run the full evaluation suite on a list of model outputs.
    Returns a single dict of: distinct_1, distinct_2, self_bleu,
    sbert_pairwise_mean, corpus_novelty_mean, corpus_novelty_min,
    quality_proxy."""
```

The function is intentionally pure: outputs go in, a flat dict comes out. This makes it trivial to:

- Run the same evaluator on baseline outputs and treatment outputs without code changes.
- Loop the evaluator over multiple seeds and aggregate.
- Compare two metric dicts side-by-side in a notebook.

The eval JSONL files land under `outputs/evaluation/`. The project deliberately preserves *raw model outputs* in addition to the metric numbers, because qualitative inspection (just reading the outputs) often catches problems that metrics miss.

---

## 10. The diversity-quality plane in practice

If you have run the full pipeline, you should have a metrics file per baseline. The right way to read it is to plot, not to stare at numbers.

Plot SBERT pairwise mean (x-axis, **inverted** so higher = more diverse) against quality proxy (y-axis). Each baseline becomes a point. The desired motion is **up and to the right**:

```
        ▲ Quality proxy
        │
        │
  high  │              ● MoP + DivPO  ← we want this in the corner
        │
        │         ● MoP SFT (no DivPO)
        │
        │   ● Single LoRA
        │
        │ ● Prompt-based
        │
        │ ● Base model
        │
        └────────────────────────────────▶  (1 - SBERT pairwise) = semantic diversity
        low                              high
```

A successful evaluation looks like the points marching from the lower-left (base model — low diversity, decent quality) toward the upper-right (MoP + DivPO — high diversity, preserved quality). If MoP + DivPO ends up *below* MoP SFT on quality, the rarity weight is set too high — DivPO is buying diversity at too much cost. If MoP + DivPO is to the *left* of MoP SFT on diversity, DivPO is doing nothing — the rarity weight is set too low.

This is the project's central knob and the evaluation's central question.

---

## 11. Honest limits of this evaluation

To stay rigorous, name what this evaluation cannot tell you:

- **No human evaluation.** All metrics are automated. There is no human-judgement signal. For a research prototype this is acceptable; for any production claim it would be required.
- **No reward model.** Quality is a proxy. The quality proxy detects topic drift but not idea quality. A model that produces relevant-but-shallow ideas will score the same as one producing relevant-and-insightful ideas.
- **Small evaluation set.** Prototype evaluations use modest sample sizes. Statistical significance requires either larger sample sizes or repeated-seed runs.
- **Static prompts.** The eval prompts are fixed. A truly robust evaluation would include adversarial prompts designed to trip the persona gate or to elicit boilerplate.

These are the natural next steps. The current evaluation is enough to *demonstrate* the methodology works directionally. It is not enough to *prove* it would survive in production. The README's `Status` section is honest about this.

---

## 12. What to take into Lesson 08

You should leave this lesson with three things:

1. **Two axes.** Diversity and quality. Every metric belongs to one of them. The interesting result is moving up and to the right on both at once.
2. **No single metric.** Distinct-n is gameable. Self-BLEU is the canonical diversity benchmark but token-based. SBERT cosine is the semantic check. Corpus novelty is the anti-plagiarism guard. Quality proxy is the relevance floor. Report all of them.
3. **Baselines are the claim.** Without the baseline ladder (base → prompt → single LoRA → MoP → MoP+DivPO), you cannot attribute gains to specific design choices. The ladder is what turns the project from "it seems to work" into a falsifiable scientific claim.

Lesson 08 closes the series by zooming in on the **co-author extension** — the product-level layer that wraps the persona-LoRA-DivPO core into a multi-turn writing-partner experience, complete with structured IdeaCards. It is where the research becomes a usable tool.

---

*Previous: [Lesson 06 — Datasets as Cognitive Fingerprints](06-datasets-as-cognitive-fingerprints.md) · Next: [Lesson 08 — The Co-Author Extension](08-coauthor-extension.md)*
