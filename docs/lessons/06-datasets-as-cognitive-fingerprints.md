# Lesson 06 — Datasets as Cognitive Fingerprints

> **Goal of this lesson.** A LoRA adapter is only as sharp as the data it was trained on. This lesson explains *why* each persona is paired with the dataset it is paired with, and why those pairings are not interchangeable. By the end you should be able to look at any cognitive style and reason about what kind of corpus would teach it.

---

## 1. The core principle: data carries cognition

Most discussions of fine-tuning data focus on quality, scale, and licensing. Those matter. But for this project a fourth property dominates the rest: **the cognitive shape of the text.**

A dataset is not just a bag of sentences. It is a record of *how the people who wrote it were thinking when they wrote it*. The grammatical patterns, the rhetorical moves, the typical sentence rhythms — all of these are downstream of the writer's cognitive task at the moment of writing.

Train a model on Reddit r/ChangeMyView and you do not just get text — you get the rhetorical pattern of "your claim assumes X, but X is not inevitable because Y." Train it on academic abstracts and you get "the system exhibits behaviour B because of mechanism M." These are not just topics. These are **thinking styles fossilised in text.**

This is the conceptual move that justifies the dataset-persona pairings in this project. Each persona is paired with a dataset whose natural text already exhibits the cognitive moves we want the persona to learn.

---

## 2. Pairing #1 — Contrarian ↔ CGA-CMV

### 2.1 What the dataset is

**CGA-CMV** stands for *Conversations Gone Awry — Change My View*. Source: Cornell ConvoKit, available as the HuggingFace mirror `mc-ai/conversations-gone-awry-cmv`.

The dataset is a curated corpus of Reddit r/ChangeMyView threads. r/ChangeMyView has a specific structure: a user posts a thesis they hold (the "View") and explicitly *invites* other users to argue against it. The thread is full of replies that attempt to surface flaws in the original claim.

### 2.2 Why this is the right cognitive fingerprint for the contrarian persona

Look at what every reply in a CMV thread is doing:

> "Your claim assumes that X is necessary. But X is not necessary — here is a counter-example from Y."

> "You are framing this as A vs B, but there is a third option C that you have not considered."

> "What you are calling a fact is actually a contested interpretation. The minority position is that…"

These are not stylistic flourishes. They are the *grammatical fossils* of contrarian thinking. Every CMV reply, by the rules of the subreddit, must:

1. **Identify an assumption.** Otherwise there is nothing to challenge.
2. **Argue the negation.** A reply that agrees does not earn a "delta."
3. **Cite supporting precedent.** Pure contradiction without evidence is downvoted.

That maps **one-for-one** onto the contrarian persona's METHOD from `personas.py`:

```
METHOD — when responding, you:
1. Identify the assumption embedded in what the writer said.
2. Propose its negation or a minority-held alternative.
3. Suggest one concrete direction the writer could take with that inverted assumption.
4. Identify what data or precedent already exists that supports the minority position.
```

This is not a coincidence. The METHOD was derived from observing what CMV-style writing looks like.

### 2.3 How it gets normalized

`normalize_reddit_record()` in `data/normalizers.py` extracts the thread title as the assumption to challenge, and the first reply turns as the model response. The prompt becomes:

> `"Challenge this assumption for pre-writing ideation: {title}"`

And the response is the body text of the challenge. The persona is wrapped into the system prompt at training time using `apply_chat_template`.

### 2.4 What would happen if you swapped this dataset

If you trained the contrarian persona on, say, Stack Exchange Q&A instead, the model would learn to *answer* questions politely rather than *challenge* them. The cognitive fingerprint of Stack Exchange is "be helpful and complete," which is the opposite of what the contrarian persona needs. The persona would still be called "contrarian" — but it would behave like a tepid explainer.

This is the whole point of the lesson. **Persona names are labels. Datasets are weights.**

---

## 3. Pairing #2 and #3 — ArXiv Abstracts ↔ {Cross-Domain Analogist, Systems Thinker}

### 3.1 What the dataset is

`gfissore/arxiv-abstracts-2021` — a large collection of paper titles + abstracts from ArXiv, spanning physics, biology, computer science, mathematics, economics, and engineering.

### 3.2 Why one dataset can serve two personas

ArXiv abstracts have two distinguishing properties:

1. **Domain breadth.** A single dataset covers dozens of scientific fields. This is exactly what the cross-domain analogist needs — exposure to many mechanisms from many domains in a single corpus.
2. **Causal language density.** Academic abstracts are unusually rich in causal connectives: *therefore*, *as a result*, *we show that*, *we demonstrate*, *implies*, *requires*. This is the linguistic substrate of systems thinking.

The same abstract supports both cognitive styles because the *style* is in how you read the abstract, not in the abstract itself.

### 3.3 How `normalize_arxiv_record()` exploits this

The normalizer creates *two separate SFT records from each abstract* — one for each persona — with different prompts:

- **Cross-domain analogist**:
  > `"Create a cross-domain analogy from this research idea: {title}"`
- **Systems thinker**:
  > `"Explain the causal system and feedback loops behind this research idea: {title}"`

The response for both is the paper's abstract. Same input data, two different cognitive lenses applied via the prompt.

This is a subtle but important data engineering move. **One raw source produces two persona-shaped training records.** The cognitive specialisation lives in the *prompt*, not in the response. The model learns to interpret "create a cross-domain analogy from…" as a command to map structures, and "explain the causal system behind…" as a command to identify feedback loops.

### 3.4 Why ArXiv specifically and not, say, news articles

A natural alternative would be popular-science news (e.g. *Quanta Magazine*). News writing has the same surface features as academic abstracts: causal language, multi-domain coverage.

The project picks ArXiv because:

- **Compression density.** A 250-word ArXiv abstract typically contains an entire causal argument. News articles dilute the argument with framing and quotes.
- **Domain breadth at scale.** ArXiv has 1.5M+ abstracts across all sciences. News datasets bias toward whatever is currently in the news cycle.
- **Permissive licensing.** ArXiv abstracts are openly redistributable.
- **Structural uniformity.** Every abstract follows roughly the same shape: motivation → method → result. This consistency makes them easier to learn from.

The lesson generalises: when you are picking a corpus for cognitive fingerprinting, ask not just *"what topic does this cover?"* but *"what cognitive move does the writer make in every sentence?"*

---

## 4. Pairing #4 — Stack Exchange Q&A ↔ Systems Thinker

### 4.1 What the dataset is

`PrimeIntellect/stackexchange-question-answering` — a collection of Stack Exchange questions paired with accepted or top-voted answers. Stack Exchange spans dozens of technical and applied domains: software, hardware, physics, philosophy, biology, etc.

### 4.2 Why this also fits the systems thinker

Stack Exchange answers have a very specific cognitive shape: they explain *why* something happens. The most upvoted answer to "why does my Postgres query slow down after a million rows?" will trace the causal chain through index trees, page faults, and disk seek patterns. The most upvoted answer to "why is the sky blue?" will trace through wavelength scattering, atmospheric composition, and human visual perception.

This is **systems thinking applied to concrete, observable phenomena.** And critically, the answers are written for **non-expert audiences**, which trains the persona to express systemic reasoning in accessible language — not jargon-dense paper-style writing.

So the systems thinker gets two complementary datasets:

| Source | Teaches |
|---|---|
| ArXiv abstracts | High-density causal reasoning at scale |
| Stack Exchange | Accessible explanation of causes to non-experts |

ArXiv is the *grammar* of systems thinking; Stack Exchange is its *register*. Together they cover both the "what to say" and the "how to say it."

### 4.3 How it gets normalized

`normalize_stackexchange_record()` wraps the question with the systems-thinker prompt:

> `"Analyze this question through causes, constraints, and feedback loops: {question}"`

The accepted answer becomes the response. This explicitly reframes the question as a system — the model is asked not to *answer* the question, but to *map the system the question lives in*. That subtle shift is the whole training signal.

---

## 5. Pairing #5 — Project Gutenberg ↔ Minimalist

### 5.1 What the dataset is

`zkeown/gutenberg-corpus` (paragraphs configuration) — public-domain literary texts from Project Gutenberg, organised at the paragraph level. Heavy on 19th and early 20th century English-language literature.

### 5.2 Why literary prose teaches minimalism

This is the least obvious pairing in the project, so it deserves the longest explanation.

The minimalist persona's METHOD says:

```
1. List the components or assumptions typically present in solutions to this topic.
2. Suggest removing all but one — the most load-bearing element.
3. Explore what becomes possible when the removed elements are gone.
4. Suggest a concrete idea that uses absence as a generative force.
```

This is hard to teach. The cognitive move — "do more with less" — is structurally absent from most internet text. Web writing is *bloat-incentivised*: longer answers rank better, longer answers score higher on completeness reward signals. A minimalist persona trained on web text would learn that "minimalist" means *fewer words while still hitting every possible point*, which is the opposite of the real thing.

Literary prose is the antidote. Centuries of editorial pressure have selected for sentences that carry maximum weight per word. Hemingway's "Hills Like White Elephants" is six pages of dialogue and a few descriptions of a train station, yet the entire piece is about an abortion that is never mentioned by name. Chekhov's stories withhold their main events. Thoreau's *Walden* makes arguments by removing distractions, not by piling on evidence.

These texts *demonstrate at the sentence level* what the minimalist persona is supposed to do at the idea level: **let absence carry weight.**

### 5.3 How `normalize_gutenberg_record()` handles it

The normalizer uses a fixed prompt:

> `"Create a constraint-driven minimalist writing idea from this passage."`

Paragraphs shorter than 80 characters are filtered out — they are usually titles, chapter markers, or fragments without enough content to learn from. The text itself becomes the response.

The training signal is: "given this minimalist-feeling passage, learn to produce text with the same cognitive density." There is no explicit instruction to "be minimalist." The corpus *is* the instruction.

### 5.4 Honest acknowledgement of the difficulty

This pairing is the most fragile of the four. Three risks:

1. **Era mismatch.** 19th century literary register can leak into the persona's voice. Output may sound antique.
2. **Domain mismatch.** Literary prose teaches *about* love, mortality, weather — not about AI in education or remote work. The persona has to learn the cognitive *style* without inheriting the topic distribution.
3. **Implicit goals.** Literary minimalism is aimed at emotional impact, not at ideation. The persona has to translate the principle to a different domain.

Mitigations come from the persona's system prompt (which keeps the model goal-aligned) and from the SFT examples that explicitly demonstrate the cognitive move on ideation topics. But this is a known soft spot. A future iteration might supplement Gutenberg with a smaller, curated set of minimalist non-fiction essays.

---

## 6. The co-author extension datasets

In addition to the four persona datasets, the **co-author SFT phase** (Lesson 08) brings in three more sources. These exist to teach a different thing — not cognitive style, but **conversational behaviour as a writing partner**.

| Dataset | HuggingFace ID | What it teaches |
|---|---|---|
| **Stanford CoAuthor** | `stanford-coauthor` | Interaction patterns from real human-AI collaborative writing sessions |
| **Writing Prompts** | `llm-aes/writing-prompts` | Creative and argumentative *brief* topics — used as `WritingBrief` seeds |
| **IBM Argument Quality** | `ibm-research/argument_quality_ranking_30k` | Argumentative topic + quality signal — used as both task seeds and a rough quality reference |

The role here is different. These datasets do not feed the *persona-shaped* training. They feed the *co-author-shaped* training, which is layered on top. The result is that each persona learns *both* its cognitive style (from the per-persona dataset) *and* the conversational structure of acting as a writing partner (from these three).

We will dissect the co-author training in Lesson 08.

---

## 7. The "raw source vs SFT target" distinction

A subtle but important point from the README:

> *The raw proposal datasets were not deleted. The important change is that they should no longer be used directly as assistant responses for fine-tuning. Use them as source material, then convert or curate them into co-author examples.*

What this means: the original pipeline used the raw datasets *directly* as `(prompt, response)` SFT pairs. The current pipeline treats them as **raw cognitive material** that needs further curation into the structured IdeaCard format.

The reasoning:

- A raw Reddit reply is contrarian, but it is not a *useful* response to a writer's brief. It is too long, too thread-specific, too informal.
- A raw ArXiv abstract is causally dense, but it is not a *useful* answer when a writer asks "give me a systems-thinking angle on AI in education."

So the data has two layers:

1. **Layer 1 — Raw cognitive corpus** (CGA-CMV, ArXiv, etc.): teaches the *cognitive style* during persona SFT.
2. **Layer 2 — Curated co-author records** (`coauthor/data.py`): teaches the *structured output format* during co-author SFT.

The model has to learn both. Skip Layer 1 and the persona has no cognitive depth. Skip Layer 2 and the persona produces unstructured prose that breaks the IdeaCard contract.

---

## 8. What this looks like in the registry

`src/mop_divpo/data/sources.py` encodes all of the above as a registry of `DatasetSource` records. Each record stores `name`, `hf_dataset_id`, `acquisition_method`, `target_personas`, and `url`. Listing the registry from the CLI:

```bash
mop-divpo list-sources
```

prints the full table — useful when you forget which dataset feeds which persona.

The fact that this is a single-source-of-truth registry, rather than scattered hardcoded references, is what makes it easy to:

- Swap a dataset for a higher-quality mirror.
- Add a new persona without touching the training code.
- Audit "where does the training signal for X come from?" in one place.

This is small-team-engineering done well. The architecture from Lesson 05 only works because the data layer is this disciplined.

---

## 9. The cognitive-fingerprint principle, generalised

If you take only one thing from this lesson, take this rule:

> **When choosing a dataset for a fine-tuned model, ask: "what is the cognitive move that the writer makes in every sentence?" Match that move to the behaviour you want.**

This rule works far beyond this project. It explains:

- Why code-instruction-tuned models perform better when trained on code with *explanations* than on code alone.
- Why "be more concise" fine-tunes work better with literary minimalism corpora than with tweet datasets (tweets are short but cognitively shallow).
- Why training a "polite refusal" persona works better on customer-support transcripts than on Reddit comments labelled as polite.

Most fine-tuning failures are not data-scale failures. They are **cognitive-mismatch** failures. The dataset's natural thinking style does not match the persona's intended thinking style. The model dutifully learns the wrong thing.

This project avoids that trap by being explicit about the pairing.

---

## 10. What to take into Lesson 07

You should leave this lesson with three things:

1. **Datasets are cognitive fingerprints, not just text.** The grammatical and rhetorical patterns of a corpus train the model in those patterns. Pick datasets whose writers were doing the kind of thinking you want the model to do.
2. **Same data can feed two personas via different prompts.** ArXiv feeds both the cross-domain analogist and the systems thinker, separated by the prompt the normalizer attaches.
3. **Raw datasets vs co-author records is a real distinction.** Persona-style data teaches *how to think*. Co-author records teach *how to be a writing partner*. The two-layer separation is intentional.

Lesson 07 picks up the next question: now that you have all this carefully constructed training data and an architecture with two diversity mechanisms, **how do you actually measure whether it works?** Diversity is famously hard to evaluate — the metrics will need their own lesson.

---

*Previous: [Lesson 05 — End-to-End Architecture](05-architecture-end-to-end.md) · Next: [Lesson 07 — Evaluation Strategy](07-evaluation-strategy.md)*
