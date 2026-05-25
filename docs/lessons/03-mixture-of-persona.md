# Lesson 03 — Mixture of Persona (MoP)

> **Goal of this lesson.** You now know *why* the project exists (Lesson 01) and *where* it intervenes (Lesson 02). This lesson explains the first of the two core techniques: **Mixture of Persona.** By the end you will understand what "persona" means here, why it is implemented as four separate LoRA adapters instead of one cleverly-prompted model, and how the router decides which persona to use.

---

## 1. The naive solution — and why it fails

Imagine you wanted a "diverse brainstormer" without any new training. The most obvious thing to try is **prompt engineering**:

```text
"Give me five ideas about X.
Idea 1 should be a contrarian take.
Idea 2 should be a cross-domain analogy.
Idea 3 should focus on systems and feedback loops.
Idea 4 should be minimalist.
Idea 5 should synthesise the others."
```

This works a little. It does not work nearly enough. Three failure modes show up almost immediately:

**1. The model averages the personas.** When you ask one model to play four roles in one response, the roles bleed into each other. The "contrarian" answer is a softer version of contrarian; the "minimalist" answer is wordy.

**2. The model's prior dominates.** RLHF training pushes the model toward a single voice. Telling it to "be contrarian" in a system prompt is fighting against thousands of training examples that taught it to be agreeable.

**3. The contrarian gets neutered first.** Aligned models are specifically trained *not* to take positions, *not* to challenge assumptions, and *not* to argue minority views. The persona that the project most needs is the one the base model most actively resists.

If you have ever tried persona-prompting at scale, you have felt this. Prompt-based diversity is a ceiling at maybe 20% of what's possible. The project needs to go further, which means going into the *weights*.

---

## 2. The MoP idea in one sentence

> Instead of training one model to do four things, train four small adapters — one per thinking style — on top of the *same frozen base model*, and route each user prompt to one or more of them.

That is it. The rest of this lesson unpacks the words in that sentence.

---

## 3. Why "small adapters" — a fast LoRA refresher

LoRA (Hu et al., ICLR 2022) is the trick that makes the whole project economically possible. Here is the intuition.

A large language model is mostly a stack of giant matrices. A single attention layer has a query projection matrix `W_q` that might be `2048 × 2048`. Fine-tuning the full model means updating all 4 million numbers in just that one matrix — and there are dozens of such matrices across the network.

LoRA's observation is empirical and surprising: **the *change* to those matrices during fine-tuning has very low rank.** That is, even though `W_q` is `2048 × 2048`, the *difference* `ΔW_q` between the pretrained matrix and the fine-tuned matrix can be approximated as the product of two much smaller matrices:

```
ΔW_q ≈ A · B
where A is 2048 × r
      B is r × 2048
      r is tiny, typically 8 or 16
```

So instead of storing 4 million numbers per matrix, you store `2 × 2048 × 16 = 65,536` numbers. The "adapter" is just these `A` and `B` matrices. The base model stays frozen.

For our 0.5B-parameter base model with `r=16`, each persona adapter is **about 2 MB of weights**. Four personas = 8 MB on top of a 1 GB base. The router can swap adapters in and out essentially for free.

This is the economic foundation of MoP. Without LoRA, "train four specialised models" would be a multi-GPU multi-day affair. With LoRA, it is a Colab notebook.

---

## 4. What "Mixture" means

The "Mixture" in Mixture of Persona is borrowed from **Mixture of Experts (MoE)** architectures — a long-standing idea in deep learning where a model has multiple specialised sub-networks ("experts") and a router decides which expert handles each input.

In a classic MoE language model (Switch Transformer, Mixtral, etc.), routing happens *inside* the model, token-by-token, in a custom kernel. That is complex and requires specialised infrastructure.

MoP is a deliberately simpler version of the same idea:

| | Classic MoE | MoP (this project) |
|---|---|---|
| Granularity | Per-token | Per-prompt |
| Router | Learned, inside the model | External, between model calls |
| Expert weights | Separate full networks | LoRA adapters on a shared base |
| Infrastructure | Custom kernel | Standard HuggingFace + PEFT |
| Diversity unit | Specialisation per token | Specialisation per cognitive style |

The MoP approach trades fine-grained per-token routing for **coarse-grained per-prompt routing.** That is the right trade for this task, because:

1. Cognitive style is a *prompt-level* property, not a token-level one. The whole answer should be contrarian, or the whole answer should be analogical — you do not want to flip personas mid-sentence.
2. Coarse routing means the prototype can run on a standard GPU with no custom code.
3. The user-facing experience is clearer. Each idea comes from a single, nameable persona.

(MixLoRA — Li et al., 2024 — is an inspiration here and points toward a future direction where routing could become finer-grained. The current prototype intentionally stays at the simpler MoP design.)

---

## 5. The four personas, and why exactly these four

A natural question: why four? Why these specific four? Could you have eight? Could you have one?

The personas are not arbitrary. Each one is **a counter-pattern to a specific failure mode of mainstream LLM ideation.** The exact list of four came from inspecting the modes that real LLMs collapse onto, then naming the cognitive styles that would escape each one.

Here is the full mapping. The system prompts below are taken directly from `src/mop_divpo/personas.py`.

### 5.1 `contrarian` — escape the "everyone agrees" failure

> *Surfaces hidden assumptions, argues minority positions, reframes common viewpoints.*

```
ROLE
You are a Contrarian Analyst and writing partner.

METHOD
1. Identify the assumption embedded in what the writer said.
2. Propose its negation or a minority-held alternative.
3. Suggest one concrete direction the writer could take with that inverted assumption.
4. Identify what data or precedent already exists that supports the minority position.
```

This persona fights **mode collapse onto consensus**. RLHF-aligned models hate disagreement; the contrarian adapter is trained to surface it.

### 5.2 `cross_domain_analogist` — escape the "stay in your lane" failure

> *Finds structural parallels between distant fields. Borrows mechanisms from biology, physics, logistics, architecture, game design.*

```
ROLE
You are a Cross-Domain Analogist and writing partner.

METHOD
1. Identify the abstract structural problem class underlying what the writer is exploring.
2. Name a domain that has solved this structural class using a different surface-level mechanism.
3. Describe the specific mechanism in that domain that handles the problem class.
4. Translate that mechanism into the writer's domain and suggest a concrete direction.
```

This persona fights **mode collapse onto domain-local clichés**. By construction, it forces the model to reach across fields.

### 5.3 `systems_thinker` — escape the "isolated point" failure

> *Maps causal structures, feedback loops, leverage points.*

```
ROLE
You are a Systems Thinker and writing partner.

METHOD
1. Identify the primary causal chain in what the writer is exploring.
2. Name at least one reinforcing and one balancing feedback loop in the system.
3. Find a leverage point where a small intervention changes the loop structure itself.
4. Suggest a concrete idea that acts at that leverage point.
```

This persona fights **the tendency to treat topics as isolated facts** rather than as nodes in a system.

### 5.4 `minimalist` — escape the "more is better" failure

> *Constraint-driven creativity. Removal as a generative force.*

```
ROLE
You are a Minimalist Designer and writing partner.

METHOD
1. List the components or assumptions typically present in solutions to this topic.
2. Suggest removing all but one — the most load-bearing element.
3. Explore what becomes possible when the removed elements are gone.
4. Suggest a concrete idea that uses absence as a generative force.
```

This persona fights **the bloat tendency** of aligned models, which by default want to add features, examples, and caveats to every answer.

### 5.5 Why four and not more

Three considerations capped the count at four:

1. **Cognitive coverage.** These four cover the four canonical *moves* in human ideation: negate, transfer, connect, distil. Adding a fifth would likely overlap an existing one.
2. **Training cost.** Each persona needs its own dataset, its own SFT run, and its own DivPO pass. Four is what fits in a single Colab session.
3. **User attention.** Showing the writer ten persona outputs is overwhelming. Four feels like a panel of advisors; ten feels like noise.

This is a deliberate design choice. The code is structured so that adding a fifth persona is a clean operation (add to `PERSONA_IDS`, add description, add dataset). The choice of four is a tuning decision, not a hard limit.

---

## 6. Why separate adapters — the weight-level argument

Lesson 03 keeps insisting "four separate adapters, not one model." The structural reason deserves explicit treatment.

Suppose you trained a single LoRA adapter on a merged dataset containing all four persona styles. What happens during training?

For any given training example, the loss function pushes the shared adapter weights toward producing *that example's* style. Over many examples, the gradient signals from different personas point in different directions. The optimiser ends up at a compromise — a set of weights that is "kind of" contrarian, "kind of" analogical, "kind of" all four. **The styles wash out.**

With separate adapters, each adapter only ever sees its own persona's data. Its gradient signal is unambiguous. It can develop weight patterns that are genuinely different from the other adapters. The styles stay sharp.

This is the central empirical claim of MoP, and it is the reason the architecture matters. *Specialisation requires isolation.* You cannot get four sharp personas out of one shared parameter pool.

```
SHARED ADAPTER                       SEPARATE ADAPTERS
─────────────                        ─────────────────
                                          ┌──> [A_contrarian]
   training      gradients clash          │
   data    ──> [ A_shared ] <── pull      │
   (all 4)        ◀━━━━━┓     in 4         ├──> [A_analogist]
                        ┃   directions     │
                  compromise              ├──> [A_systems]
                  weights                  │
                                          └──> [A_minimalist]
                                              (each one trained on
                                               only its own slice)
```

---

## 7. The router — how a prompt finds its persona

Training the adapters is half the job. At inference time, when a user types a prompt, *something* has to decide which adapters to activate. That something is the **persona gate**.

The code lives in `src/mop_divpo/routing/gate.py` and ships with three gate implementations. Walking through them shows the evolution of the design.

### 7.1 The keyword gate — the prototype baseline

```python
class KeywordPersonaGate:
    KEYWORDS: dict[PersonaId, tuple[str, ...]] = {
        "contrarian": ("challenge", "assumption", "opposite", "dissent", "minority"),
        "cross_domain_analogist": ("analogy", "borrow", "domain", "biology", "physics", "architecture"),
        "systems_thinker": ("system", "causal", "feedback", "loop", "long-term", "side effect"),
        "minimalist": ("constraint", "minimal", "remove", "simple", "essential", "limit"),
    }
```

Each persona has a small list of trigger keywords. A prompt is scored by counting how many of its words match each persona's keyword list. The top-`k` highest-scoring personas are selected.

**Why start here.** It is deterministic, debuggable, and requires zero training data. For a research prototype, "I can read the source and predict the routing" is enormously valuable.

**Why it is not enough.** It only fires on literal keyword matches. A prompt like *"How might cities be redesigned for resilience?"* is obviously a systems-thinking question — but it contains none of the keywords. The keyword gate would score it 0 for everything and fall back to alphabetical order.

### 7.2 The embedding gate — the production-style upgrade

```python
class EmbeddingPersonaGate:
    """Ranks personas by cosine similarity between the prompt and each persona's
    routing description, using an injected text encoder."""
```

Each persona has a short *routing description* (`PERSONA_ROUTING_DESCRIPTIONS` in `personas.py`). Both the user prompt and each persona description are encoded with a sentence-transformer (`all-MiniLM-L6-v2` by default). The router ranks personas by cosine similarity in embedding space.

This captures semantic intent, not just literal keywords. *"How might cities be redesigned for resilience?"* embeds near the systems-thinker description even with zero keyword overlap.

### 7.3 The random gate — the ablation control

```python
class RandomPersonaGate:
    """Ablation gate — assigns scores uniformly at random.

    Same interface as `EmbeddingPersonaGate` so callers can swap it in.
    Used to test whether the SBERT-based routing is doing real work, or
    whether any routing (even random) would suffice."""
```

The presence of this gate is itself a teaching moment. **You cannot claim that your router is doing useful work unless you can show it beats random routing.** Including the random gate as a first-class implementation is honest scientific practice — it bakes the null hypothesis into the codebase.

### 7.4 Hysteresis — a small but important detail

```python
def apply_hysteresis(ranked, current_persona, margin=0.05):
    """Keep the currently active persona on top unless another beats it by `margin`.

    Prevents the gate from thrashing between personas on short, ambiguous turns."""
```

In multi-turn conversation, the routing scores change every turn. Without intervention, the active persona can flip turn-to-turn, which is jarring for the user. Hysteresis says: "if the new top persona only beats the current one by a hair, stick with the current one."

This is borrowed from control theory (think thermostats). It is a small detail that exists only because the project actually got tested with users and noticed the flapping problem. Real engineering, not just paper engineering.

---

## 8. Top-`k` routing — why we usually pick more than one persona

A naive router would always pick the single best persona for each prompt. The MoP design picks the **top-`k`** (default `k=2`), meaning multiple personas often respond to the same prompt in parallel.

Why?

1. **Diversity is the *point*.** Showing one answer defeats the purpose. The writer needs multiple distinct angles to choose between.
2. **The router is uncertain.** For most real prompts, two or three personas are nearly equally relevant. Picking only the top one throws away information.
3. **Inter-persona diversity is one of the two target axes** (Lesson 01). Top-`k` routing is what *operationalises* inter-persona diversity at inference time.

The config exposes this directly:

```yaml
routing:
  top_k: 2
```

You can crank it to `4` for maximum coverage, or drop to `1` if you only want a single targeted answer.

---

## 9. The user can also override the router

The router is a default, not a constraint. Every gate implementation accepts a `force_personas` argument:

```python
def rank(self, prompt, top_k, force_personas: list[str] | None = None):
    if force_personas:
        personas = [validate_persona(p) for p in force_personas]
        return [
            PersonaRoute(persona=p, score=1.0, description=PERSONA_DESCRIPTIONS[p])
            for p in personas[:top_k]
        ]
```

A writer who *knows* they want a contrarian and a minimalist take can just say so:

```bash
mop-divpo ideate --topic "..." --persona contrarian --persona minimalist
```

The router silently steps aside. This matters because **the writer is the expert on their own ideation needs.** A good system supports user override without resistance.

---

## 10. What MoP does *not* fix

To stay rigorous: MoP increases **inter-persona diversity**. That is its job. It does *not* fix intra-persona diversity.

If you ask the contrarian persona the same question ten times after MoP training, you will still get ten variations of the same contrarian answer. The mode within the persona has collapsed.

This is exactly the failure DivPO is designed to fix. Lesson 04 picks up there.

---

## 11. What to take into Lesson 04

You should leave this lesson with three things:

1. **MoP is four LoRA adapters on a shared frozen base.** Each adapter specialises in one cognitive style. Total adapter cost: ~8 MB. The base is the same.
2. **Specialisation requires isolation.** A single adapter trained on all four styles produces a compromise. Four adapters trained on one style each stay sharp.
3. **The router is a separable concern.** Keyword, embedding, random, and user-forced routing all share an interface. This makes the routing strategy a research variable you can ablate cleanly.

MoP buys us inter-persona diversity. It does nothing for intra-persona diversity. For that, we need DivPO — the second half of the project's name and the topic of the next lesson.

---

*Previous: [Lesson 02 — Why Pre-Writing](02-why-pre-writing.md) · Next: [Lesson 04 — DivPO Explained](04-divpo-explained.md)*
