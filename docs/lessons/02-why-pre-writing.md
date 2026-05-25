# Lesson 02 — Why Pre-Writing

> **Goal of this lesson.** Lesson 01 argued that LLM monoculture is a real problem. This lesson answers the second hard question: *out of all the stages of writing — brainstorming, drafting, editing, publishing — why does this project intervene at the pre-writing stage, and not somewhere else?* The choice is deliberate, and it constrains every decision that follows.

---

## 1. The writing process is not a single moment

When non-writers think about "writing," they tend to picture one activity: someone sitting at a keyboard, producing words. In reality, writing research has known for decades that the act splits into several distinct cognitive stages, each with its own goals and failure modes.

The canonical model (Flower & Hayes, 1981; refined many times since) breaks the process into roughly three phases:

```
┌──────────────┐    ┌──────────┐    ┌──────────┐
│  Pre-writing │──▶ │ Drafting │──▶ │  Editing │
│   ideation   │    │ producing│    │ refining │
│   planning   │    │   prose  │    │ polishing│
└──────────────┘    └──────────┘    └──────────┘
   "what to       "how to put    "make it
    say at all"    it in words"   readable"
```

Each phase asks a different question:

| Phase | Central question | Output |
|---|---|---|
| **Pre-writing** | *What is this piece even about?* | A direction, a thesis, a list of angles |
| **Drafting** | *How do I express that?* | A first draft |
| **Editing** | *How can I make this clearer?* | A polished version |

If you blur these together, you cannot reason cleanly about where AI helps and where it hurts. The same AI behaviour that is excellent for editing can be catastrophic for pre-writing — and vice versa.

---

## 2. Where current LLMs are *good*

Be fair to the technology before criticising it. Current LLMs are genuinely strong at the **second and third phases**:

- **Drafting.** Given a clear thesis and a few bullet points, a modern LLM produces serviceable prose in seconds. Grammar, transitions, and tone are handled well.
- **Editing.** Given a draft, an LLM can shorten it, change register, catch repeated words, and reorder sentences. This is high-value, low-creativity work, and it is exactly where convergence to the "safe average answer" is actually helpful.

This is the half of the writing market most products target. ChatGPT, Grammarly, Notion AI, Jasper — almost all of them sit downstream of "you already know what you want to say."

---

## 3. Where current LLMs *fail*

The first phase — pre-writing — is where everything breaks.

Pre-writing is the stage where a human is asking questions like:

- "What angle on this topic hasn't been done to death?"
- "What is the underlying argument I actually want to make?"
- "What's the *opposite* of the obvious take?"
- "What would make this essay surprising instead of generic?"

This is exactly the cognitive task where mode collapse (Lesson 01) does the most damage. The user explicitly wants the *non-obvious* output, and the model is structurally incapable of providing it, because the alignment objective has trained it to converge on the obvious.

Worse, the failure is invisible. When an LLM produces a bad draft you can see it. When it gives you a generic brainstorm, **you just don't notice the better ideas that were never suggested.** The cost is silent, and it scales across every user.

This is the wedge. This is the problem worth picking.

---

## 4. A short cognitive-science detour: divergent vs. convergent thinking

To talk precisely about why pre-writing is special, borrow two terms from creativity research (Guilford, 1956; later expanded by countless cognitive psychologists):

- **Divergent thinking** — generating many possible answers from a single starting point. The classic test: "Name as many uses for a brick as you can in 60 seconds."
- **Convergent thinking** — narrowing many possibilities down to the single best answer. The classic test: standardised math problems.

Modern LLMs, by design, are convergent thinking machines. The training objective `argmax P(token | context)` is a convergent operation. RLHF reinforces this by training the model to pick the single highest-rated response.

Pre-writing is a **divergent** task. The whole point is to expand the space of possible angles before narrowing.

This is a fundamental mismatch. Asking a convergent system to do divergent work is like using a hammer to unscrew a bolt. Some clever wrist motion will get you part of the way, but the tool was built for the opposite job.

> **Takeaway.** Pre-writing is the one stage of writing where the LLM's core training objective fights the user's actual cognitive task. That is *exactly* why this project targets it.

---

## 5. Why not just "improve drafting" or "improve editing"?

A reasonable counter-question: "If LLMs are already good at drafting and editing, why not just make them slightly better at those?"

Three reasons.

**5.1 Diminishing returns.** Drafting and editing quality is already at the level where most users cannot reliably tell a good model from a great one. Marginal improvements there produce marginal user value.

**5.2 Wrong end of the pipeline.** A great editor cannot save a bad idea. If pre-writing produces a generic, predictable angle, every downstream improvement just polishes that generic angle. The leverage point is upstream.

**5.3 Open research problem.** Drafting and editing are crowded markets with diminishing differentiation. Pre-writing has almost no good tools. The research opportunity is here, not there.

This is a case of the *leverage point* idea from systems thinking (which, not coincidentally, is one of the four personas in Lesson 03). Small intervention upstream, large effect downstream.

---

## 6. What pre-writing concretely looks like in this project

To make this real, here is what the prototype actually does. From the README:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli ideate \
  --topic "AI in education" \
  --goal "Find a sharper thesis" \
  --persona contrarian \
  --persona minimalist
```

Notice what this command *does not* do:

- It does not produce a draft essay.
- It does not produce a paragraph of body text.
- It does not even produce a single answer.

What it *does* produce is a small set of structured **IdeaCards**, each from a different cognitive angle:

```json
{
  "operation": "challenge_thesis",
  "diagnosis": "what is wrong or missing in the writer's current framing",
  "idea": "the concrete direction the writer should take",
  "rationale": "why this direction is better than the obvious one",
  "next_step": "one immediate concrete action the writer can take right now"
}
```

(See `src/mop_divpo/coauthor/schema.py` for the full definition. We will analyse this schema in detail in Lesson 08.)

The product never tries to write *for* the user. Its only job is to **widen the space of possible angles** the user could pursue. The writing itself stays with the human.

This is a small but important design choice. We will come back to it in Lesson 08.

---

## 7. Designing for the *worst* moment of the writing process

Another way to frame the pre-writing choice is to ask: when does a writer most need help, and what kind of help?

Writers describe the pre-writing moment in remarkably consistent terms:

- *"I have a vague topic but no angle yet."*
- *"Everything I think of sounds like something I've already read."*
- *"I don't know what would make this interesting."*

Notice the pattern. The writer is not stuck on *how* to say things — they are stuck on *what* to say at all. The help they need is not better prose; it is **better starting points.**

A tool that gives them five new generic starting points is no help. A tool that gives them five *structurally different* angles — one contrarian, one analogical, one systems-based, one minimalist — is help that scales.

This is the user-facing motivation behind the four personas. We are not picking persona styles for variety's sake. We are picking them because **each one breaks the writer out of a different stuck pattern.**

| When the writer is stuck because... | The persona that unsticks them |
|---|---|
| "Everyone agrees on this premise, so my piece has nowhere to go" | `contrarian` |
| "I can't find a fresh framing — everything's been said" | `cross_domain_analogist` |
| "I see the topic but can't see how the pieces connect" | `systems_thinker` |
| "I have too much to say and no clear core" | `minimalist` |

Each persona is a *specific intervention on a specific failure mode* of human ideation. That is why there are exactly four, and that is why they are these four.

---

## 8. The "real problem" test

Earlier the lesson promised to answer: *is this really solving a real problem?* Apply a simple three-question test.

**Question 1: Can the problem be observed without specialised tools?**

Yes. Anyone who has used an LLM to brainstorm can reproduce the failure in under a minute. You do not need access to model weights or a research lab to see mode collapse on ideation prompts.

**Question 2: Does the problem affect a meaningful population?**

Yes. Every student, researcher, founder, writer, marketer, and product manager who uses an LLM for early-stage thinking is affected. The population is large and growing.

**Question 3: Is the cost of the problem larger than the cost of the fix?**

Yes. The fix is incremental — LoRA adapters on top of an existing base model, no new model training from scratch, no human preference labelling needed. The cost of the problem — narrower idea spaces across millions of users — is much larger than the engineering investment required.

All three answers are "yes." The problem is real, the population is real, and the fix is feasible. That is the bar for a research project worth doing.

---

## 9. Honest limits of the wedge choice

To stay rigorous, name what this choice gives up:

- **It is not a general writing assistant.** A user who wants help drafting or editing should use a different tool. This system has no advantage there and would probably be worse.
- **It assumes the user is a writer.** Non-writers using the tool to "generate content" will be confused — the IdeaCards are designed to assist a human writer, not to replace one.
- **It assumes the user actually wants surprise.** Some users want the *safe* answer. For them, monoculture is a feature, not a bug. The system is wrong for them.

The system is sharp on purpose. A sharp tool that solves one problem cleanly is more valuable than a dull tool that half-solves three. Lesson 08 will show how this sharpness shows up directly in the data schemas and the system prompts.

---

## 10. What to take into Lesson 03

You should leave this lesson with three things:

1. **Writing has stages.** Pre-writing, drafting, editing. LLMs are great at the second two and structurally bad at the first.
2. **Pre-writing is a divergent task.** LLMs are convergent machines. That mismatch is the entire reason this project exists, and the entire reason every design decision focuses on diversity over quality.
3. **The four personas are not arbitrary.** Each one targets a specific human failure mode of ideation. Lesson 03 will show how they are constructed inside the model.

Lesson 03 picks up the next question: **what does it mean, mechanically, to give a single base model four different "personas"?** That is where Mixture of Persona comes in — and we get to see how LoRA adapters make this surprisingly cheap.

---

*Previous: [Lesson 01 — The Creativity Problem](01-the-creativity-problem.md) · Next: [Lesson 03 — Mixture of Persona](03-mixture-of-persona.md)*
