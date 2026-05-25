# Lesson 04 — DivPO Explained

> **Goal of this lesson.** Mixture of Persona (Lesson 03) gives us *inter-persona* diversity — different cognitive styles produce different answers. This lesson tackles the second axis: *intra-persona* diversity. Why does asking the same persona the same question ten times still produce ten similar answers? The fix is **Diverse Preference Optimization (DivPO)**, the second half of this project's name. By the end you should understand DPO well enough to see exactly what DivPO changes, and why that one small change matters.

---

## 1. The problem MoP cannot fix

Imagine MoP is fully trained. You route a prompt to the `contrarian` persona and sample ten times with `temperature=0.9`. You get:

```
1. "The hidden assumption is that progress requires growth..."
2. "We need to challenge the idea that progress means growth..."
3. "Most people assume growth equals progress, but..."
4. "The standard view conflates progress with growth..."
5. "An unexamined assumption: that progress is growth..."
...
```

These are ten lexically different sentences. Semantically, they are the same answer. The persona is sharp — every output sounds contrarian. But within the persona, **the model has collapsed onto a single mode of contrarianism**.

This is intra-persona mode collapse. MoP cannot fix it, because MoP's job is to make personas different from *each other*, not to make a persona's outputs different from *itself*.

We need a separate intervention. That intervention is DivPO.

---

## 2. A quick refresher on DPO

DivPO is built on top of DPO, so we need DPO first. If you already know DPO well, skim this section.

**DPO (Direct Preference Optimization)** is the modern, simpler successor to RLHF. The setup:

1. For each training prompt, you have two model responses: a "preferred" one (`chosen`) and a "dispreferred" one (`rejected`).
2. The training objective increases the probability the model assigns to the `chosen` response *relative to* the `rejected` response.
3. There is no separate reward model. The preference pairs encode the reward signal directly.

Mathematically, the DPO loss looks roughly like:

```
L_DPO(chosen, rejected) ∝ -log σ( β · [log π(chosen) - log π(rejected)] - reference_term )
```

Where `π` is the model being trained and `β` is a temperature. You do not need to memorise the formula. The thing to internalise is the shape of the objective:

> **DPO teaches the model: "produce more of what looks like `chosen`, less of what looks like `rejected`."**

DPO is excellent at improving quality, helpfulness, and alignment. But it has the same flaw as RLHF: **nothing in the objective rewards diversity.** Whichever response was labelled `chosen` gets reinforced, and over many training pairs the model converges on the single most-rewarded *mode* of response.

---

## 3. What DivPO changes — in one sentence

> **DivPO** (Lanchantin et al., 2025, [arXiv:2501.18101](https://arxiv.org/abs/2501.18101)) is DPO with a different rule for *picking which response to label as `chosen`*: instead of "the highest-quality response," DivPO picks **"the rarest response that is still above the quality threshold."**

That is the whole idea. The DPO loss does not change. The training loop does not change. The model architecture does not change. **Only the pair-construction step changes.**

This is what makes DivPO a great research idea: a one-line conceptual change with measurable downstream effects.

---

## 4. Why "rare but good" instead of "best"

To see why this works, picture what happens during pair construction.

Suppose for a given prompt the model generates 4 candidate responses:

| # | Response | Quality | Lexical rarity |
|---|---|---|---|
| A | The hidden assumption is that growth equals progress. (common framing) | 0.85 | 0.10 |
| B | Most people assume growth = progress, but evidence shows… (common framing, slightly different phrasing) | 0.80 | 0.15 |
| C | What if the entire growth-progress link is a category error inherited from 19th-century economics? (unusual angle) | 0.70 | 0.85 |
| D | gibberish text low quality output here | 0.10 | 0.95 |

**Standard DPO** would pick `A` as `chosen` (highest quality) and `D` as `rejected` (lowest quality). The model would learn to produce more of the "growth equals progress" framing — which is *already* the most likely thing it would have produced anyway. No new behaviour is learned. Mode collapse continues.

**DivPO** filters out `D` for being below the quality threshold, then asks: among the *remaining* candidates, which is the rarest? Answer: `C`. So `C` becomes `chosen`, and `A` (the most common framing) becomes `rejected`. The model learns to **prefer the unusual-but-still-good response over the typical-but-good one.**

Over thousands of such training pairs, the model's distribution stops collapsing onto the average response and starts spreading across the high-quality region of idea-space.

---

## 5. The trade-off DivPO is making

Notice the trade. DivPO is willing to give up a small amount of quality (0.85 → 0.70) in exchange for a large gain in rarity (0.10 → 0.85). The configurable weights make this trade explicit. From `configs/prototype.yaml`:

```yaml
divpo:
  candidates_per_prompt: 4
  min_quality: 0.35
  rarity_weight: 0.6
  quality_weight: 0.4
```

Read this as a policy: "I require quality at least 0.35. Among everything above that bar, I weight rarity slightly more than quality (0.6 vs 0.4) when picking the winner."

Tuning these knobs is the project's main lever. Crank `rarity_weight` up and the model produces wilder, less reliable outputs. Crank `quality_weight` up and you reproduce standard DPO. The sweet spot — empirically — is somewhere around `0.6 / 0.4`, leaning into rarity.

The `min_quality` floor is the safety net. Without it, DivPO would happily promote gibberish (response `D` in the table) because gibberish is by definition rare. The floor says: "don't reward rarity unless the response is still on-topic."

---

## 6. Looking at the actual code

The project's DivPO implementation lives in two short files. Both fit on a single screen. Here they are, with annotations.

### 6.1 Scoring — `src/mop_divpo/divpo/scoring.py`

```python
from mop_divpo.metrics.diversity import distinct_n


def lexical_rarity_scores(responses: list[str]) -> list[float]:
    scores: list[float] = []
    for index, response in enumerate(responses):
        others = responses[:index] + responses[index + 1 :]
        overlap_penalty = 0.0
        response_tokens = set(response.lower().split())
        if response_tokens and others:
            other_tokens = set(" ".join(others).lower().split())
            overlap_penalty = len(response_tokens & other_tokens) / len(response_tokens)
        scores.append(max(0.0, distinct_n([response], 1) - overlap_penalty))
    max_score = max(scores) if scores else 0.0
    return [score / max_score if max_score else 0.0 for score in scores]


def combined_divpo_score(
    quality: float,
    rarity: float,
    quality_weight: float,
    rarity_weight: float,
) -> float:
    return quality * quality_weight + rarity * rarity_weight
```

Two functions. Both deliberately simple.

**`lexical_rarity_scores`** — for each response in a group of candidates:
1. Look at all the *other* responses.
2. Count how many of this response's word tokens also appear in the others (`overlap_penalty`).
3. Score this response by its own lexical variety (`distinct_n([response], 1)`) minus the overlap penalty.
4. Normalise so the most-rare response gets a score of 1.0.

This is a fast, dependency-free proxy for rarity. A response gets a high rarity score when it uses words that the other candidates do not use. A more sophisticated version would use SBERT cosine distance instead of token overlap; the prototype keeps it lexical for speed and interpretability.

**`combined_divpo_score`** — the weighted sum that decides the winner. This is the operational form of the trade-off discussed in section 5.

### 6.2 Pair construction — `src/mop_divpo/divpo/pairs.py`

```python
def select_divpo_pair(
    candidates: list[CandidateRecord],
    min_quality: float,
    quality_weight: float,
    rarity_weight: float,
) -> DivPOPairRecord:
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required to build a DivPO pair")
    prompts = {candidate.prompt for candidate in candidates}
    personas = {candidate.persona for candidate in candidates}
    if len(prompts) != 1 or len(personas) != 1:
        raise ValueError("Candidates must share the same prompt and persona")

    eligible = [candidate for candidate in candidates if candidate.quality_score >= min_quality]
    if not eligible:
        raise ValueError("No candidates meet the minimum quality threshold")

    chosen = max(
        eligible,
        key=lambda candidate: combined_divpo_score(
            candidate.quality_score,
            candidate.rarity_score,
            quality_weight,
            rarity_weight,
        ),
    )
    rejected = min(
        [candidate for candidate in candidates if candidate.id != chosen.id],
        key=lambda candidate: (
            candidate.quality_score >= min_quality,
            combined_divpo_score(
                candidate.quality_score,
                candidate.rarity_score,
                quality_weight,
                rarity_weight,
            ),
        ),
    )
    return DivPOPairRecord.from_candidates(
        id=f"divpo-{chosen.persona}-{chosen.id}-{rejected.id}",
        chosen=chosen,
        rejected=rejected,
    )
```

This is the entire DivPO algorithm in one function. Walk through it line by line:

1. **Sanity checks.** All candidates must come from the same prompt and the same persona. Pair construction makes no sense otherwise.
2. **Quality filter.** Drop anything below `min_quality`. This is the safety floor from section 5.
3. **`chosen` selection.** Among eligible candidates, pick the one with the highest combined (quality + rarity) score. This is the line where DivPO diverges from DPO.
4. **`rejected` selection.** Among the *remaining* candidates, pick the one with the *lowest* combined score. The tuple `(quality_score >= min_quality, combined_score)` sorts ineligible candidates first (they make the better rejection targets), then within each group sorts by combined score ascending.
5. **Return** a `DivPOPairRecord` formatted exactly the way standard DPO trainers expect: `prompt`, `chosen`, `rejected`.

The clean part: the *training* code does not know about DivPO at all. It just sees standard DPO pairs and runs standard DPO. All the DivPO-ness is concentrated in this one pair-selection function, which means you can swap pair-selection strategies (random pairs, lowest-quality pairs, etc.) as ablations without touching the trainer.

---

## 7. Where the candidates come from

The DivPO loop needs *N* candidate responses per prompt to choose from. Where do those come from?

The recipe is:

1. Start with an SFT-trained persona adapter (the output of MoP Stage 1).
2. For each prompt in your training data, run inference *N* times with stochastic decoding (`temperature ≈ 0.9`, `top_p ≈ 0.95`).
3. Score each candidate's quality and rarity.
4. Build the DivPO pair.
5. Feed all the pairs into a standard DPO trainer (e.g. `trl dpo`).

The README has the concrete command for step 5:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_name_or_path outputs/adapters/sft/contrarian \
  --dataset_name data/processed/divpo/contrarian.jsonl \
  --output_dir outputs/adapters/divpo/contrarian \
  --use_peft true \
  --max_steps 100
```

Notice that this runs **on top of** the SFT adapter, not on top of the raw base model. DivPO is the second of two training phases:

```
Phase 1: SFT on persona data  → outputs/adapters/sft/{persona}
                                                ↓
Phase 2: DivPO with the SFT   → outputs/adapters/divpo/{persona}
        adapter as starting   
        point
```

SFT teaches the persona's style. DivPO teaches the persona to spread its outputs across diverse modes of that style.

---

## 8. Why DivPO is training-time only

A subtle but important architectural point: **DivPO touches nothing at inference time.** Once training is complete, you are left with a regular LoRA adapter file (`outputs/adapters/divpo/contrarian/`) that loads with standard `peft` calls.

This is why the README says:

> *DivPO is **training-time only**. Inference uses standard MixLoRA routing.*

The implication: the entire diversity gain shows up *inside the model weights*. The runtime decoding strategy can stay simple. There is no special sampling code, no diversity-aware beam search, no extra inference cost. Whatever the model now generates with `temperature=0.7` already reflects DivPO's broader distribution.

This is the elegant part. DivPO buys you behavioural diversity without complicating the deployment story.

---

## 9. The quality proxy — an honest disclosure

The DivPO algorithm needs a quality score for each candidate. In a research paper with a real reward model, this would be a trained quality classifier. The prototype does not have that.

Instead, the project uses a **proxy**: normalised `distinct_n` on the candidate itself, optionally combined with prompt-response semantic similarity. This is acknowledged directly in `src/mop_divpo/divpo/scoring.py` (the call to `distinct_n([response], 1)`).

Why this is okay for a prototype:

- It is fast, deterministic, and dependency-free.
- It correlates with quality *enough* to keep the floor working as a safety net.
- It can be swapped out for a real reward model without changing the rest of the pipeline.

Why this is *not* okay for production:

- A quality proxy that uses lexical features can be gamed by responses that are merely wordy.
- A real evaluation requires a real reward model, ideally trained on human preference data.

The README is explicit about this as future work. Read the project as a **methodology demonstration with a placeholder quality signal**, not as a claim that the prototype outputs are high-quality on an absolute scale.

---

## 10. What DivPO is *not*

To stay rigorous:

- **DivPO is not "creative writing training."** It does not teach the model new vocabulary or new genres. It only changes which response gets reinforced from a set the model could already produce.
- **DivPO is not a sampling method.** Top-`p`, temperature, nucleus sampling — all unchanged at inference time.
- **DivPO is not a guarantee of useful diversity.** It optimises for *lexical* rarity in the prototype. Two responses can be lexically rare relative to each other while being conceptually identical. This is why evaluation (Lesson 07) uses *semantic* metrics like SBERT cosine, not just lexical ones.

---

## 11. The pairing with MoP

Now that DivPO is in hand, the two halves of the project fit together cleanly:

| Technique | Stage | Diversity axis it fixes |
|---|---|---|
| **MoP** | Architecture (multiple adapters) + Routing (top-`k`) | Inter-persona — different styles produce different answers |
| **DivPO** | Training-time pair selection | Intra-persona — same style produces varied answers |

Either one alone fixes half the problem. Together they target the full failure space of Lesson 01.

Notice they are **mechanically independent**. You could run DivPO with no MoP (one adapter, diverse outputs within one style). You could run MoP with no DivPO (four adapters, each with collapsed intra-persona diversity). The interesting question for evaluation is: do they *add* (independent gains) or *multiply* (synergistic gains)? Lesson 07 lays out how the project tries to answer this with ablations.

---

## 12. What to take into Lesson 05

You should leave this lesson with three things:

1. **DivPO is DPO with one rule changed.** Pair-selection: instead of "highest quality," pick "rarest above quality floor." Everything else is unchanged.
2. **The trade-off is explicit.** `rarity_weight` and `quality_weight` in the config let you slide between standard DPO behaviour and pure novelty-chasing. The floor (`min_quality`) is the safety net.
3. **All the magic is in pair construction.** `src/mop_divpo/divpo/pairs.py` is the entire algorithm. The trainer is off-the-shelf. This is what makes DivPO modular and easy to reason about.

Lesson 05 zooms out and shows how MoP + DivPO + the data layer + the inference layer wire together into a single end-to-end system, both at training time and at inference time.

---

*Previous: [Lesson 03 — Mixture of Persona](03-mixture-of-persona.md) · Next: [Lesson 05 — End-to-End Architecture](05-architecture-end-to-end.md)*
