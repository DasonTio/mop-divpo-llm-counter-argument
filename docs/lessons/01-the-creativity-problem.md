# Lesson 01 — The Creativity Problem

> **Goal of this lesson.** Before you touch a single line of code, you should understand the *real-world failure* this project is trying to fix. By the end you will be able to answer two questions: (1) why is creativity in LLMs a genuine problem, not a marketing problem, and (2) why is "diversity of output" the right thing to optimise, instead of "quality of output."

---

## 1. The thing everyone notices but rarely names

If you have used ChatGPT, Claude, or Gemini to brainstorm an essay topic, a startup idea, a research angle, a product name — you have probably noticed something uncomfortable.

Ask the same model the same prompt ten times. You get back ten answers that *sound* different but actually say the same thing.

Try a concrete example:

> *"Give me five fresh ideas for an essay about remote work."*

You will almost always receive some combination of:

1. The future of hybrid offices
2. Work-life balance in a remote setting
3. Mental health challenges of isolation
4. Productivity and asynchronous collaboration
5. The impact on cities and commercial real estate

Run the prompt again. The headings change. The bullets reshuffle. **The underlying ideas do not.**

This is not a personal observation. It has a name in the research literature.

---

## 2. Generative monoculture

**Definition (Wu et al., ICLR 2025, "Generative Monoculture in Large Language Models"):**

> *Generative monoculture* is the tendency of an aligned language model to produce a narrow, repetitive distribution of outputs even when the input space is wide and the model is sampled stochastically.

Three words to internalise:

- **Aligned** — this happens *because* of RLHF / DPO alignment, not in spite of it.
- **Narrow** — the model's *output* distribution collapses to a small mode, even when its *internal* distribution still has variety.
- **Stochastic** — the problem persists even with `temperature=1.0` and resampling. It is not a sampling bug.

This is the disease. The rest of this project is the proposed treatment.

---

## 3. Why it happens — the short version

You don't need to read the full paper to understand the mechanism. Here is the chain in plain language.

```
Pretraining          → model has a huge, messy distribution of possible outputs
       │
       ▼
RLHF / DPO alignment → human raters pick "best" answers
       │
       ▼
Training objective   → "make the high-rated answers more likely"
       │
       ▼
Side effect          → the average safe, helpful, complete answer wins every time
       │
       ▼
Result               → mode collapse onto the response that scores best on average
```

Notice the trap. The objective is `argmax(quality)`. There is no term anywhere that says `also produce different answers across samples`. The model does exactly what it was trained to do — and it kills creativity as a side effect.

This is the same shape as classroom multiple-choice questions: if the only thing that matters is the right answer, students stop offering unusual interpretations.

---

## 4. Why this hurts real work

Generative monoculture is not just aesthetically annoying. It causes measurable harm in three contexts:

### 4.1 Education and research

A student asked to brainstorm five thesis directions for "AI in healthcare" using an LLM will receive five variations of the same five mainstream topics that every other student also received. The model is, in effect, a copying machine. The student loses the *only* part of the assignment that was supposed to be theirs.

### 4.2 Pre-product ideation

A founder using an LLM to find a startup angle in fintech will get back the same answer everyone else gets back: "AI-powered budgeting app," "fraud detection for small banks," "embedded BNPL." Useful for a quick gut-check. Useless for finding an underserved angle.

### 4.3 Creative writing

Writers using an LLM in the *pre-draft* phase — when they are still deciding what the piece is even about — receive suggestions that all converge on the same mainstream framing. The LLM is not extending the writer's range; it is narrowing it.

The pattern is consistent: **the more the user wants surprise, the worse the model performs.**

---

## 5. Why "just generate more" doesn't fix it

A natural first thought: "Just sample more outputs and pick the diverse ones."

This does not work. Three reasons:

1. **The distribution itself is narrow.** If you sample 100 outputs from a collapsed distribution, you get 100 variations of the same mode. Volume does not create variety.
2. **Temperature increases noise, not ideas.** Cranking `temperature=1.5` makes outputs more chaotic at the *word* level, but the underlying concepts remain the same. You get the same idea phrased more strangely.
3. **Prompting tricks fade fast.** "Be creative," "give me unusual answers," "think outside the box" all show small effects in benchmarks and almost no effect on real ideation. The model has learned that "creative" answers are still expected to be safe and complete.

If we want genuine diversity, **we have to change the training objective**, not just the sampling strategy. This is the central thesis of the entire project.

---

## 6. What "diversity" actually means here

Before going further, be precise about the word *diversity*. The project distinguishes three layers, and they are not the same thing:

| Layer | What it means | Example of failure |
|---|---|---|
| **Lexical** | Different vocabulary and surface words | All outputs use the same five buzzwords |
| **Semantic** | Different meanings and conceptual positions | Two outputs say the same thing using different words |
| **Structural / cognitive** | Different *kinds of thinking* used to arrive at the answer | All five answers use the same reasoning style (e.g. all are causal analyses) |

Most prior work on LLM diversity only tackles the first layer (lexical). DivPO (the paper we will study in Lesson 04) goes after the second. **Mixture of Persona (this project's contribution) goes after the third.**

The hardest, most important, and most under-studied layer is the third one — and it is where this project tries to push the state of the art.

---

## 7. Two flavours of diversity, and why we need both

There is one more distinction that is going to come up in every later lesson, so memorise it now.

- **Inter-persona diversity** — across *different* thinking styles. A contrarian answer should be structurally unlike a systems-thinker answer.
- **Intra-persona diversity** — within *the same* thinking style. Two contrarian answers to the same prompt should still differ from each other.

If we only had inter-persona diversity, we would get four predictable answers — one per persona — and within each persona the same mode collapse as before. If we only had intra-persona diversity, we would get four flavours of the same kind of thinking. The project needs **both**.

The two techniques map cleanly:

| Technique | Layer it fixes |
|---|---|
| **MoP (Mixture of Persona)** | Inter-persona diversity (Lesson 03) |
| **DivPO (Diverse Preference Optimization)** | Intra-persona diversity (Lesson 04) |

This is why the project combines them. Either alone would only fix half the problem.

---

## 8. Why the problem is worth solving now

It would be possible to wait. As base models grow larger and better, some of this might dissolve on its own. Three reasons not to wait:

1. **Larger models make the problem worse, not better.** Larger models are subjected to more rounds of alignment, on more diverse rater pools, with more aggressive convergence pressure. Empirically, GPT-4 shows *more* monoculture than GPT-3.5 on ideation benchmarks, not less.
2. **The cost is silent.** Mode collapse does not throw an error. The user never sees the ideas they did not receive. There is no measurable signal that anything is wrong unless you specifically test for it.
3. **It compounds across users.** Millions of writers, researchers, and founders are now starting from the same five suggestions. Society's idea-space is narrowing in a way that is hard to detect and harder to reverse.

This is a real problem with a real cost, and it is getting worse, not better. That justifies the project's existence.

---

## 9. What this project is *not* trying to do

To set expectations honestly:

- **Not** a new base model. The base model (`Qwen/Qwen2.5-0.5B-Instruct`) is taken as given.
- **Not** a replacement for RLHF. RLHF made the model usable. This project sits on top of it.
- **Not** a general-purpose model. The system is tuned for one specific cognitive task: **pre-writing ideation.** Lesson 02 explains exactly what that means and why it was chosen.
- **Not** an aesthetic / style transfer system. The personas are *cognitive* (how the answer is reasoned about) not stylistic (how the answer is phrased).

Keep these scope boundaries in mind. Many failure modes you might worry about are out of scope — and that is on purpose.

---

## 10. What to take into Lesson 02

You should leave this lesson with three things:

1. **The disease has a name.** *Generative monoculture.* You can find it in Wu et al., ICLR 2025.
2. **The cause is structural.** RLHF optimises average quality. There is no term in the objective that rewards diversity. Sampling tricks cannot fix what training has removed.
3. **The fix needs two layers.** Inter-persona diversity (MoP) and intra-persona diversity (DivPO) target two different failure modes. The project combines them on purpose.

Lesson 02 picks up the next question: out of all the places where LLMs touch the writing process — brainstorming, drafting, editing, publishing — **why did this project pick pre-writing as the wedge?** That choice is not obvious, and it determines the shape of everything that comes after.

---

*Next: [Lesson 02 — Why Pre-Writing](02-why-pre-writing.md)*
