# Lesson 08 — The Co-Author Extension

> **Goal of this lesson.** The previous seven lessons covered the research core: monoculture, pre-writing, MoP, DivPO, architecture, datasets, evaluation. This final lesson covers the *product layer* that wraps it all into a usable tool: the **co-author extension**. You will see why a one-shot ideation API was not enough, what the IdeaCard schema is and why every field exists, and the design principle that runs through it all — *writer agency*.

---

## 1. The pivot — from one-shot to writing partner

The original research design was a single-call API:

```python
generate(prompt="What is the future of remote work?", personas=["contrarian"])
# → "The hidden assumption is that remote work is here to stay..."
```

Useful for benchmarks. Awkward as a product. When the project was tested with real users, three problems surfaced immediately.

### 1.1 Problem 1 — single-turn is too thin for ideation

Pre-writing is *iterative* (Lesson 02). A writer hears an angle, thinks about it, has a follow-up question, refines the direction. A one-shot API forces the writer to start over for every refinement, losing context each time. The system was useful for *generating* ideas and useless for *developing* them.

### 1.2 Problem 2 — free-form prose buries the action

A free-form contrarian response might be a 200-word paragraph. The writer has to extract from that paragraph: *what's the actual idea, what's the supporting reasoning, what should I do next?* The cognitive load was being pushed back onto the user — the opposite of what a writing partner should do.

### 1.3 Problem 3 — the system was trying to *write* instead of *help write*

Long prose outputs invited the user to copy-paste the model's words into their draft. That collapses the system into a content generator, which is exactly what Lesson 02 said it was *not* trying to be.

The co-author extension is the response to all three problems. It changes:

| Component | Before | After |
|---|---|---|
| Output format | Free-form text | Structured `IdeaCard` |
| Interaction | Single turn | Multi-turn `PersonaSession` |
| Training data | Raw `(prompt, response)` pairs | Curated `(WritingBrief, IdeaCard)` pairs |
| System prompt | Generic persona description | Conversational persona with explicit 4-step METHOD |
| Prior art injection | Every call | First turn only; later turns use accumulated history |

---

## 2. The IdeaCard — what an answer looks like now

The headline change is the output format. Instead of prose, every co-author response is a JSON object with five fields. From `src/mop_divpo/coauthor/schema.py`:

```python
class IdeaCard(BaseModel):
    persona: PersonaId
    operation: PersonaOperation
    diagnosis: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
```

Each field has a single specific job. They are not arbitrary categories — each one is a force-function on the model's behaviour.

### 2.1 `operation` — name the cognitive move

```python
PersonaOperation = Literal[
    "challenge_thesis",
    "transfer_analogy",
    "map_system",
    "distill_core",
    "synthesize_angles",
]
```

The model must label what kind of move it is making. This is not for the user's benefit — it is a **forcing function on the model**. By requiring an explicit operation label, the schema forbids the model from producing a wishy-washy answer that does several things half-heartedly. Pick one cognitive move and commit.

Notice the mapping back to personas:
- `contrarian` → `challenge_thesis`
- `cross_domain_analogist` → `transfer_analogy`
- `systems_thinker` → `map_system`
- `minimalist` → `distill_core`
- All personas → `synthesize_angles` (when combining)

The operation is the persona's *act*, named.

### 2.2 `diagnosis` — what is wrong or missing

> *What is wrong or missing in the writer's current framing?*

Critical detail: the model is forbidden from going straight to an answer. It first has to **name the problem**. This is a deliberate echo of how good writing teachers work: they do not give you a sentence, they tell you what your current draft is failing to do.

For example, given the brief *"AI in education, goal: find a sharper thesis"*, a diagnosis might be: *"Your framing assumes AI is a tool for delivery efficiency. The sharpest theses challenge the underlying definition of what education is for."*

### 2.3 `idea` — the concrete direction

> *The concrete direction the writer should take.*

The actual proposal. One direction, not a list. Brevity is the test — if the model produces a paragraph here it is failing the schema's intent.

### 2.4 `rationale` — why this direction beats the obvious one

> *Why this direction is better than the obvious one.*

The model has to *defend* the idea. Specifically, it has to explain why this direction is better than the *expected* take. This forces the model to reason about its own diversity claim: *here is the obvious thing, here is what I am suggesting instead, here is why mine is better.*

This field is the closest thing to a meta-cognitive check the schema has. It is also the field where the cognitive personality of each persona shows up most distinctively.

### 2.5 `next_step` — the immediate action

> *One immediate concrete action the writer can take right now.*

The kicker. The model must end with a single specific thing the writer can do *in the next 10 minutes*. Not "consider exploring…" but "write down three counter-examples to your central claim." Not "think about the system" but "list the five most influential actors in this system and draw arrows for who affects whom."

This is the field that turns the system from an *adviser* into a *collaborator*. The user always leaves with an action, never with a vague "interesting question to ponder."

### 2.6 Why these five fields and not more

The fields were chosen to enforce a specific cognitive arc:

```
diagnosis (problem) → idea (proposal) → rationale (defence) → next_step (action)
                  ^
                  operation (the move's name)
```

This is the arc a good editor walks a writer through. Adding more fields would dilute it. Removing any of them collapses the structure — without `diagnosis` the model jumps to answers; without `rationale` it asserts without arguing; without `next_step` it advises without enabling.

Five fields. Each one load-bearing.

---

## 3. The WritingBrief — what the user provides

The other half of the contract is what comes in. The user no longer types a free-form prompt; they fill out a structured **`WritingBrief`**:

```python
class WritingBrief(BaseModel):
    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    audience: str = "general readers"
    constraints: list[str] = Field(default_factory=list)
    notes: str = ""
```

The four meaningful fields:

| Field | Why it exists |
|---|---|
| `topic` | What the piece is about. Required — without it the model has nothing to think about. |
| `goal` | What stage the writer is at. "Find a thesis" vs "tighten this argument" vs "find a counter-example" produce wildly different ideal responses. |
| `audience` | "Industry executives" vs "PhD students" vs "general readers" changes which angles are useful. |
| `constraints` | Hard limits — "max 800 words," "no jargon," "must include a code example." Lets the user prune obviously infeasible directions before the model wastes effort. |

This is a small but important interface design point. **By making the brief structured, the system can hold the writer accountable for naming their own goal.** A writer who cannot articulate `goal` is not ready for ideation help; they need to think more first. The brief is also a tiny act of pre-writing in itself.

The brief gets converted into a single prompt block via `WritingBrief.to_prompt()`:

```
Topic: AI in education
Writer goal: Find a sharper thesis
Audience: general readers
Constraints: max 800 words; no jargon
```

That formatted block is what the user message ends up containing in the chat template.

---

## 4. The PersonaSession — multi-turn done right

The session layer is where the second problem (single-turn thinness) gets fixed. `src/mop_divpo/inference/session.py` introduces a `PersonaSession` that holds:

- One **persona** (fixed for the session — the user does not switch midway).
- A growing list of **messages** (`system`, `user`, `assistant`, repeated).
- A **first-turn marker** that controls PRIOR ART injection.

The lifecycle:

```
turn 0: build_messages(persona_system + PRIOR ART block + brief)
        → model produces IdeaCard
        → append both user_message and assistant_response to history

turn 1+: build_messages(persona_system + history + new_user_turn)
        → no PRIOR ART injection (history carries the context)
        → model produces IdeaCard refined for the follow-up
        → append both to history
```

Two details worth noting.

### 4.1 PRIOR ART is injected only once

In one-shot inference, PRIOR ART (top-3 similar training examples) is injected on every call. In sessions, it is injected only on turn 0. Why?

Three reasons:
1. **It is expensive.** Top-3 retrieval and embedding adds latency to every turn.
2. **It is redundant.** Once injected, the model has "seen" the prior art for the rest of the session via the history.
3. **It distorts later turns.** If the writer's follow-up question is about a different angle, re-injecting prior art biased toward the original prompt would push the model in the wrong direction.

This is one of those tiny correctness decisions that only shows up after the system gets real-world use. It is in the code (`session.py`) precisely because someone noticed the issue.

### 4.2 The persona is locked per session

A `PersonaSession` is tied to *one* persona. If the user wants advice from a different persona, they start a new session. This avoids the cognitive whiplash of getting a contrarian response in one turn and a systems-thinker response in the next within the same conversation.

Routing happens at session creation time, not at each turn. If the user wants top-`k` personas on a brief, the system creates `k` parallel sessions, one per persona. The user can then drill into whichever response they found most interesting.

---

## 5. Training the co-author behaviour

The persona adapters from Lesson 03 know how to *think* in their style. They do not, by default, know how to produce IdeaCard JSON. A second SFT phase teaches them that.

The training data lives in `src/mop_divpo/coauthor/data.py`. It is built in two layers:

### 5.1 Seed records — `build_seed_coauthor_records()`

A small set of hand-curated `(WritingBrief, IdeaCard)` pairs — four per persona, expanded with eight task variants each (`thesis`, `angle`, `outline`, `evidence`, `audience`, `revision`, `comparison`, `next_move`).

These are the **gold standard examples**. They show the model:
- Exactly what an IdeaCard looks like for each persona.
- How to phrase each field in the persona's voice.
- How to vary the `operation` across task types.

About 32 high-quality records per persona. Tiny. But high quality.

### 5.2 External records — `collect_external_coauthor_records()`

Records pulled from Writing Prompts, IBM Argument Quality, and CGA-CMV and *programmatically converted* into `(WritingBrief, IdeaCard)` pairs for all four personas.

This is bulk data. The seeds set the ceiling on quality; the externals provide volume. Together you get a few hundred records per persona, enough for the model to generalise without overfitting to the seeds.

### 5.3 The training row format

Every co-author record is converted to a HuggingFace-style messages list via `CoAuthorRecord.to_messages()`:

```python
def to_messages(self) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": describe_persona(self.card.persona)},
        {"role": "user", "content": self.brief.to_prompt()},
        {"role": "assistant", "content": self.card.to_text()},
    ]
```

The system prompt is the persona's full `ROLE + METHOD` text. The user message is the formatted brief. The assistant message is the IdeaCard serialised as JSON (`IdeaCard.to_text()` returns `json.dumps(...)`).

This is the *only* part of the project where the model sees the output schema during training. The IdeaCard format is learned end-to-end from these examples, not enforced by a separate parser at inference time.

---

## 6. The deeper principle — writer agency

If you step back, every co-author design decision points at one principle:

> **The model does not write. The model helps the human write.**

This shows up in many places:

- **IdeaCards are not drafts.** They are diagnoses, ideas, rationales, and actions — *inputs* to the writer's process, not outputs of it.
- **`next_step` is concrete and small.** A 10-minute action keeps the writer in motion, not waiting for the AI to "do more."
- **`diagnosis` makes the model name the problem before proposing solutions.** This mirrors how good human collaborators work.
- **WritingBrief forces the user to articulate `goal`.** The model never sees a vague prompt; the user has had to think before asking.
- **PRIOR ART injection tells the model what to *avoid*.** The user's prior context is protected from collapse.
- **PersonaSession locks one persona per session.** No surprise cognitive style swaps.

This is not just product polish. It is a *response* to the way generic LLM products tend to drift — toward becoming content generators that the user copy-pastes from. The co-author extension is a deliberate refusal of that trajectory.

Whether it succeeds is an empirical question. But the principle is at least stated clearly, both in the schemas and in the prompts. That clarity is the most important design contribution of the co-author layer.

---

## 7. The `ideate` command — the user-facing surface

What this all looks like to a user:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli ideate \
  --topic "AI in education" \
  --goal "Find a sharper thesis" \
  --persona contrarian \
  --persona minimalist
```

Output: two IdeaCards (one contrarian, one minimalist), printed in a structured format. The user reads them, picks the one whose `diagnosis` resonates, follows the `next_step`, and either keeps writing or comes back for another round.

For the multi-turn version:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli chat
```

Opens an interactive REPL. The user provides the brief, picks a persona, and then converses. The first turn injects PRIOR ART; subsequent turns build on history.

For programmatic use:

```python
from mop_divpo.coauthor.schema import WritingBrief
from mop_divpo.coauthor.ideation import build_idea_cards

brief = WritingBrief(
    topic="AI in education",
    goal="Find a sharper thesis",
    audience="education policy researchers",
    constraints=["max 800 words"],
)
cards = build_idea_cards(brief, personas=["contrarian", "minimalist"])
```

The `build_idea_cards` function is rule-templated for the prototype (no model call required) and useful as a deterministic seed-generator. The real model-backed version goes through `inference.generate.generate(...)` with `adapter_prefix` pointing at the co-author adapters.

---

## 8. Where the co-author extension stops, and why

Things the co-author layer is intentionally *not* doing:

- **No drafting.** It does not produce paragraphs of the actual essay. By design.
- **No editing.** It does not rewrite the user's prose. By design.
- **No "tone" personas.** The personas are cognitive, not stylistic.
- **No memory across sessions.** Each session starts fresh. (Memory would be useful but adds privacy and design complexity that the prototype skips.)
- **No multi-persona within one session.** Switching personas mid-conversation is jarring; users wanting multiple takes start parallel sessions.

These boundaries are what keep the system *sharp*. A tool that does fewer things well beats a tool that does everything passably, especially when the goal is to support a creative cognitive task that benefits from clarity.

---

## 9. Bringing it all together — what the full pipeline now does

After eight lessons, here is the project end-to-end in one paragraph:

> A frozen, small base LLM (`Qwen/Qwen2.5-0.5B-Instruct`) is augmented with four cognitive-style LoRA adapters — `contrarian`, `cross_domain_analogist`, `systems_thinker`, `minimalist` — each trained on a corpus whose natural writing already exhibits its target thinking style. Each adapter goes through a second training phase using DivPO, which selects rare-but-quality candidate responses as preferred over common ones, broadening the adapter's output distribution. At inference, a routing gate (keyword, embedding, or user-forced) picks the top-`k` personas for a user's `WritingBrief`. A corpus retriever injects "prior art" the model should avoid repeating. Each selected persona generates a structured `IdeaCard` with `operation`, `diagnosis`, `idea`, `rationale`, and `next_step` fields. Multi-turn conversations are supported via `PersonaSession`. Evaluation tracks Self-BLEU, Distinct-n, SBERT pairwise cosine, corpus novelty, and a prompt-output quality proxy against five baselines (base, prompt-only, single LoRA, MoP-SFT, MoP+DivPO) to verify that the system moves outputs into the high-diversity / high-relevance corner of the evaluation plane — without sacrificing writer agency, because the model only ever proposes diagnoses, ideas, and next steps; it never writes the piece for the user.

If you can read that paragraph and trace every clause back to a file or design decision, you understand the project.

---

## 10. What to take away from the whole series

A short index of the eight lessons:

| Lesson | Core takeaway |
|---|---|
| 01 | Generative monoculture is a real, structural problem caused by RLHF's averaging objective. |
| 02 | Pre-writing is divergent; LLMs are convergent. That mismatch is the wedge this project chose. |
| 03 | MoP = four LoRA adapters on a shared base. Specialisation requires isolation. |
| 04 | DivPO = DPO with rare-but-good as the chosen rule. One-line conceptual change, big behavioural effect. |
| 05 | Two flows — training builds artefacts, inference loads them. Modular design lets you ablate any piece. |
| 06 | Datasets are cognitive fingerprints. Match the writer's thinking style, not just the topic. |
| 07 | Evaluation lives on the diversity-quality plane. No single metric — report all five against five baselines. |
| 08 | Co-author extension wraps the core in IdeaCards and sessions. Writer agency is the governing principle. |

If you came in cold and made it through all eight, you now understand: **what the problem is, why it's worth fixing, how the fix works mechanically, how it's measured, and how it's wrapped into a tool a writer can actually use.**

That is the entire project. Everything in `src/mop_divpo/` is one of these eight ideas in code form.

---

*Previous: [Lesson 07 — Evaluation Strategy](07-evaluation-strategy.md) · [Back to Lesson 01](01-the-creativity-problem.md)*
