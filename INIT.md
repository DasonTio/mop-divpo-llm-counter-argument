# Claude Code Agent Implementation Guide: MoP + DivPO Counter-Argument Research

## Purpose of This File

This markdown file is intended to be used as the main instruction/context file for a Claude Code Agent working inside the repository.

The goal is to help the agent implement the research execution plan for a project on:

> **Diverse counter-argument generation via Mixture of Cognitive Personas and DivPO training.**

The current priority is **dataset preparation**, especially:

1. Supervised Fine-Tuning (SFT) dataset preparation.
2. DivPO / DPO preference dataset preparation.

Do **not** start with UI, product features, Gradio, multi-turn chat, paper writing, or deployment.

The implementation should produce clean, inspectable datasets and modular scripts that can later be used for training and evaluation.

---

# 1. Research Context

## 1.1 Core Problem

Aligned LLMs often produce outputs that look different on the surface but collapse toward the same underlying ideas. This project treats that failure as **generative monoculture**.

The research goal is not simply to make outputs more random. The goal is:

> Increase useful diversity while preserving quality.

The relevant diversity layers are:

1. **Lexical diversity** — different words.
2. **Semantic diversity** — different meanings or arguments.
3. **Cognitive / structural diversity** — different ways of thinking about the same input.

This project focuses especially on cognitive diversity.

---

## 1.2 Current Journal-Facing Scope

The project was originally motivated by pre-writing ideation, but the current research scope is narrower and more defensible:

> **Diverse counter-argument generation using Mixture of Cognitive Personas and DivPO.**

Pre-writing remains the motivation, but the measurable task is counter-argument generation.

Why counter-argument generation?

- It is concrete.
- It is judgeable.
- It naturally benefits from diversity.
- It can be evaluated with quality, novelty, and persona-fidelity rubrics.

---

# 2. Key Concepts the Agent Must Understand

## 2.1 SFT, LoRA, MoP, and DivPO Are Different Things

Use this mental model:

```text
SFT   = learn the persona behavior.
LoRA  = the small trainable adapter mechanism.
MoP   = organize multiple persona adapters and route inputs to them.
DivPO = continue training from SFT adapters to prefer rare-but-good outputs.
```

More specifically:

```text
Base Model
  + empty LoRA adapter
  + persona-specific SFT data
  -> SFT training
  -> SFT persona adapter

SFT persona adapter
  + generated candidate responses
  + rare-but-good preference pair selection
  -> DPO training
  -> DivPO-tuned persona adapter
```

Do not confuse these concepts:

- SFT is not LoRA.
- LoRA is not MoP.
- DivPO does not replace SFT.
- MoP is not only a classifier; it is the whole architecture of multiple persona experts plus a router.

---

## 2.2 LoRA Adapter Model

The base model is frozen. LoRA adds small trainable low-rank matrices.

Conceptually:

```text
Original model weight: W       frozen
LoRA update:           ΔW = A x B trainable
Effective weight:      W + ΔW
```

Each persona should have its own LoRA adapter.

Example:

```text
outputs/adapters/sft/contrarian/
outputs/adapters/sft/systems_thinker/
outputs/adapters/sft/cross_domain_analogist/
outputs/adapters/sft/minimalist/
```

After DivPO:

```text
outputs/adapters/divpo/contrarian/
outputs/adapters/divpo/systems_thinker/
outputs/adapters/divpo/cross_domain_analogist/
outputs/adapters/divpo/minimalist/
```

---

## 2.3 Mixture of Persona

MoP means:

```text
One frozen base model
+ multiple persona-specific LoRA adapters
+ a routing/gating mechanism
```

At inference time:

```text
Input
  -> Persona Gate / Router
  -> ranked personas
  -> select top-k persona adapters
  -> generate one response per selected adapter
  -> return diverse response set
```

MoP does **not** necessarily mean all adapters are blended into the model at once. In the current design, adapters are used in a plug-and-play manner:

```text
base model + contrarian adapter -> contrarian response
base model + systems adapter    -> systems response
```

The router can be:

- keyword-based,
- embedding-similarity-based,
- KNN,
- logistic regression,
- LLM-based,
- random for ablation,
- or user-forced.

For now, keep routing simple. Do not implement complex routing until the data pipeline and adapters work.

---

## 2.4 DivPO

DivPO is DPO with a different pair-selection rule.

Standard DPO says:

```text
prompt + chosen response + rejected response
```

and trains the model to assign higher probability to `chosen` than to `rejected`.

DivPO changes how `chosen` is selected:

```text
chosen = rarest response that is still above a quality threshold
rejected = common, weak, or less diverse response
```

DivPO does not change inference-time architecture. It is a training-time preference optimization stage.

Important:

```text
SFT teaches the persona.
DivPO diversifies the persona.
MoP organizes multiple personas.
```

---

# 3. Current Implementation Priority

## Current Focus

Implement dataset preparation for:

1. SFT datasets.
2. DivPO preference datasets.

Do **not** start training until dataset preparation scripts are implemented, runnable, and produce inspectable JSONL files.

---

# 4. Repository Deliverables

The implementation should produce or update these kinds of files:

```text
src/mop_divpo/data/prepare_sft.py
src/mop_divpo/data/prepare_divpo.py
src/mop_divpo/data/persona_prompts.py
src/mop_divpo/data/filters.py
src/mop_divpo/divpo/scoring.py
src/mop_divpo/divpo/pairs.py
scripts/prepare_sft_datasets.py
scripts/prepare_divpo_datasets.py
```

If similar files already exist, extend them instead of duplicating functionality.

Expected dataset outputs:

```text
data/processed/sft/contrarian.jsonl
data/processed/sft/systems_thinker.jsonl
data/processed/sft/cross_domain_analogist.jsonl
data/processed/sft/minimalist.jsonl

data/processed/divpo/contrarian.jsonl
data/processed/divpo/systems_thinker.jsonl
data/processed/divpo/cross_domain_analogist.jsonl
data/processed/divpo/minimalist.jsonl
```

---

# 5. SFT Dataset Preparation

## 5.1 Purpose of SFT Data

SFT data teaches each persona how to behave.

SFT is imitation learning:

```text
input -> desired assistant response
```

For this project, each persona should learn a distinct cognitive behavior.

---

## 5.2 Canonical SFT Record Format

Each SFT record should be chat-format JSONL.

Preferred structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<persona system prompt>"
    },
    {
      "role": "user",
      "content": "<persona-specific instruction using the input>"
    },
    {
      "role": "assistant",
      "content": "<target response>"
    }
  ],
  "metadata": {
    "persona": "<persona_id>",
    "source": "<dataset_source>",
    "source_id": "<optional source id>"
  }
}
```

If existing training code expects only `messages`, keep metadata optional or add a config flag.

---

# 6. Persona-Specific SFT Requirements

## 6.1 Contrarian SFT

### Persona ID

```text
contrarian
```

### Objective

Train the contrarian persona to generate useful counter-arguments.

### Ideal Behavior

```text
claim -> hidden assumption -> challenge -> alternative framing
```

### Data Source

Use:

```text
CGA-CMV / ChangeMyView via ConvoKit
```

Recommended package:

```bash
pip install convokit
```

Example loading code:

```python
from convokit import Corpus, download

corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
```

### Transformation

Approximate mapping:

```text
parent utterance = claim / statement being responded to
child utterance  = counter-argument / reply
```

### User Prompt Format

```text
Generate a counter-argument to this claim:

{claim}
```

### System Prompt

```text
You are a Contrarian Analyst.
Your task is to generate useful counter-arguments.

Method:
1. Identify the hidden assumption in the claim.
2. Challenge that assumption.
3. Offer an alternative framing.
4. Stay coherent, relevant, and respectful.
```

### Filtering Rules

Remove examples where:

- claim or response is empty,
- claim is too short,
- response is too short,
- response is extremely long,
- response contains `[deleted]` or `[removed]`,
- response is mostly links,
- response is insulting, toxic, or moderator-like,
- response does not actually challenge or complicate the parent claim.

Suggested rough filters:

```text
claim:    20-300 words
response: 30-250 words
```

### Output File

```text
data/processed/sft/contrarian.jsonl
```

---

## 6.2 Systems Thinker SFT

### Persona ID

```text
systems_thinker
```

### Objective

Train the systems thinker persona to produce causal-system critiques.

### Ideal Behavior

```text
claim -> causal chain -> feedback loop -> second-order effect -> leverage point
```

### Data Source

Use:

```text
PrimeIntellect/stackexchange-question-answering
```

via Hugging Face `datasets`.

Recommended package:

```bash
pip install datasets
```

Example loading code:

```python
from datasets import load_dataset

ds = load_dataset(
    "PrimeIntellect/stackexchange-question-answering",
    split="train",
    streaming=True,
)
```

### Important Note

Stack Exchange is not naturally counter-argument data. It is explanation data.

So for the first version, train the systems thinker to produce **causal critiques** or **systems explanations**, not pure rebuttals.

This is acceptable because the systems thinker persona is supposed to reason through:

- causes,
- constraints,
- mechanisms,
- feedback loops,
- second-order effects.

### Transformation

Approximate mapping:

```text
Stack Exchange question = issue / system to analyze
accepted or gold answer = causal/system explanation
```

Expected field names may include:

```text
prompt
gold_standard_solution
metadata
problem_id
```

The script should inspect fields safely and not crash if a field is absent. Prefer robust `.get(...)` access.

### User Prompt Format

```text
Analyze this issue through causes, constraints, feedback loops, and second-order effects:

{question}
```

### System Prompt

```text
You are a Systems Thinker.
Your task is to analyze issues through causes, constraints, feedback loops, and second-order effects.

Method:
1. Identify the primary causal chain.
2. Identify relevant constraints or incentives.
3. Explain second-order effects.
4. Identify a leverage point or structural weakness.
```

### Filtering Rules

Remove examples where:

- question or answer is empty,
- answer is too short,
- answer is too long,
- answer is mostly code,
- answer is mostly links,
- answer is too domain-specific to be useful,
- answer does not explain a mechanism or causal relationship.

Suggested rough filters:

```text
question: 20-500 words
answer:   40-300 words
```

### Output File

```text
data/processed/sft/systems_thinker.jsonl
```

---

## 6.3 Cross-Domain Analogist SFT

### Persona ID

```text
cross_domain_analogist
```

### Objective

Train the analogist persona to produce analogy-based critiques by transferring mechanisms from one domain to another.

### Ideal Behavior

```text
claim -> abstract structure -> distant domain -> borrowed mechanism -> translated critique
```

### Data Source

Use:

```text
gfissore/arxiv-abstracts-2021
```

via Hugging Face `datasets`.

Example loading code:

```python
from datasets import load_dataset

ds = load_dataset(
    "gfissore/arxiv-abstracts-2021",
    split="train",
    streaming=True,
)
```

### Important Note

ArXiv abstracts are not directly analogy-based counter-arguments.

They should be treated as **mechanism source material**, not perfect final target behavior.

For the first implementation, create SFT records that teach the model to extract transferable mechanisms from research ideas.

### Transformation

Approximate mapping:

```text
paper title    = research idea
paper abstract = mechanism-rich explanation
```

Expected fields may include:

```text
id
title
abstract
categories
```

### User Prompt Format

```text
Extract the transferable mechanism from this research idea and explain how it could inspire an analogy:

{title}
```

### System Prompt

```text
You are a Cross-Domain Analogist.
Your task is to identify mechanisms in one domain and translate them into useful analogies for another domain.

Method:
1. Identify the abstract structure of the idea.
2. Name the mechanism being used.
3. Explain why that mechanism transfers across domains.
4. Suggest how it could inspire a critique or reframing elsewhere.
```

### Filtering Rules

Remove examples where:

- title or abstract is empty,
- abstract is too short,
- abstract is too long,
- abstract is too formula-heavy,
- abstract is too dataset-specific without a transferable mechanism.

Suggested rough filters:

```text
title:    4-40 words
abstract: 80-250 words
```

### Future Improvement Note

A stronger version should transform raw abstracts into explicit analogy-shaped responses. For now, keep this implementation simple and inspect outputs manually.

### Output File

```text
data/processed/sft/cross_domain_analogist.jsonl
```

---

## 6.4 Minimalist SFT

### Persona ID

```text
minimalist
```

### Objective

Train the minimalist persona to produce stripped-down core disagreements.

### Ideal Behavior

```text
claim -> remove assumptions -> expose core tension -> concise reframing
```

### Preferred First Data Source

Use:

```text
ibm-research/argument_quality_ranking_30k
```

via Hugging Face `datasets`.

Example loading code:

```python
from datasets import load_dataset

ds = load_dataset(
    "ibm-research/argument_quality_ranking_30k",
    split="train",
)
```

If the dataset requires a config name, inspect available configs first:

```python
from datasets import get_dataset_config_names

print(get_dataset_config_names("ibm-research/argument_quality_ranking_30k"))
```

### Important Note

Do **not** use Project Gutenberg for the first minimalist implementation.

Gutenberg may teach literary compression, but it risks:

- antiquated style leakage,
- domain mismatch,
- literary rather than argumentative behavior.

For the first version, use concise, high-quality arguments from IBM Argument Quality.

### Transformation

Approximate mapping:

```text
topic    = claim / debate topic
argument = concise argument
```

Expected fields may include:

```text
topic
argument
WA
MACE-P
stance
```

The script should inspect actual fields safely.

### User Prompt Format

```text
Generate a concise minimalist counter-argument to this claim:

{topic}
```

### System Prompt

```text
You are a Minimalist Designer.
Your task is to produce concise counter-arguments by removing unnecessary assumptions and exposing the core disagreement.

Method:
1. Remove secondary assumptions.
2. Identify the smallest core disagreement.
3. State the critique in a compact form.
4. Avoid unnecessary explanation.
```

### Filtering Rules

Remove examples where:

- topic or argument is empty,
- argument is too short to be meaningful,
- argument is too long,
- argument is low quality if quality labels are available,
- argument is vague or slogan-like.

Suggested rough filters:

```text
topic:    3-40 words
argument: 8-60 words
```

If a quality field exists, prefer high-quality arguments only.

### Output File

```text
data/processed/sft/minimalist.jsonl
```

---

# 7. SFT Dataset Validation Requirements

After preparing SFT datasets, print and/or save a summary with:

```text
number of records per persona
first 3 examples per persona
average prompt length
average response length
number of filtered records
```

Suggested output path:

```text
outputs/data_checks/sft_summary.json
outputs/data_checks/sft_examples.md
```

Manual inspection is required. The data should not be treated as good just because the script runs.

For each persona, inspect examples and ask:

```text
Does the user prompt match the persona objective?
Does the assistant response actually demonstrate the target cognitive behavior?
Is the response coherent and respectful?
Would we want the model to imitate this?
```

---

# 8. DivPO Dataset Preparation

## 8.1 Purpose of DivPO Data

The DivPO dataset is not downloaded directly.

It is generated from SFT-trained persona adapters.

DivPO teaches the model:

```text
Among multiple acceptable answers, prefer the rare-but-good one.
```

It does not teach the persona from scratch. SFT already did that.

---

## 8.2 DivPO Generation Process

For each persona:

```text
1. Take a claim/prompt from the SFT or evaluation prompt pool.
2. Load the SFT-trained persona adapter.
3. Generate N candidate responses, e.g. 4 for debugging or 8 for final data.
4. Score each candidate for quality.
5. Score each candidate for rarity.
6. Pick chosen = rare but high quality.
7. Pick rejected = common, weak, or less diverse.
8. Save as DPO-style preference pair.
```

---

## 8.3 Canonical DivPO Record Format

Use this as the internal format:

```json
{
  "prompt": "Generate a counter-argument to this claim:\n\n{claim}",
  "chosen": "{rare_but_good_response}",
  "rejected": "{common_or_weaker_response}",
  "metadata": {
    "persona": "contrarian",
    "chosen_quality": 0.82,
    "chosen_rarity": 0.76,
    "rejected_quality": 0.61,
    "rejected_rarity": 0.22,
    "quality_weight": 0.4,
    "rarity_weight": 0.6,
    "min_quality": 0.35,
    "candidates_per_prompt": 4
  }
}
```

If the DPO trainer expects a different format, add a conversion function. Keep this internal canonical format for auditability.

---

## 8.4 Candidate Generation Settings

Suggested settings:

```text
temperature: 0.8-1.0
top_p: 0.9-0.95
max_new_tokens: 200-300
```

Suggested candidate count:

```text
N = 4 for debugging
N = 8 for stronger final DivPO data
```

---

## 8.5 Quality Scoring

For the first version, use a simple quality proxy.

A candidate should score higher if it is:

- relevant to the prompt,
- coherent,
- non-empty,
- not too short,
- not too long,
- not obviously degenerate,
- not mostly repeated tokens,
- not mostly links/code unless the persona requires it.

Suggested components:

```text
quality = relevance_score + coherence_heuristic + length_penalty
```

Possible implementation:

- relevance: embedding similarity between prompt and response,
- coherence: heuristic penalty for repetition or malformed text,
- length penalty: penalize extremely short or extremely long responses.

Keep scoring modular so it can later be replaced by:

- LLM-as-judge,
- reward model,
- stronger semantic evaluator.

---

## 8.6 Rarity Scoring

A candidate should score higher if it is less similar to the other candidates generated for the same prompt/persona.

Preferred first implementation:

```text
rarity = 1 - average semantic similarity to other candidates
```

Use sentence embeddings if available.

Fallback implementation:

```text
rarity = lexical distinctness - token overlap penalty
```

---

## 8.7 Pair Selection Rule

For each candidate group:

```text
eligible = candidates with quality >= min_quality
chosen   = candidate with highest combined score among eligible
rejected = candidate with lowest combined score or lowest quality
```

Combined score:

```text
combined_score = quality_weight * quality + rarity_weight * rarity
```

Suggested defaults:

```text
min_quality = 0.35
quality_weight = 0.4
rarity_weight = 0.6
```

Rationale:

- Quality floor prevents rewarding gibberish.
- Rarity weight slightly above quality weight encourages diversity.
- The method should prefer rare-but-good, not rare-at-all-costs.

---

## 8.8 DivPO Output Files

```text
data/processed/divpo/contrarian.jsonl
data/processed/divpo/systems_thinker.jsonl
data/processed/divpo/cross_domain_analogist.jsonl
data/processed/divpo/minimalist.jsonl
```

---

## 8.9 DivPO Validation Requirements

After preparing DivPO datasets, print and/or save:

```text
number of DivPO pairs per persona
average chosen quality
average chosen rarity
average rejected quality
average rejected rarity
first 3 preference pairs
number of skipped prompt groups
reason counts for skipped groups
```

Suggested output path:

```text
outputs/data_checks/divpo_summary.json
outputs/data_checks/divpo_examples.md
```

---

# 9. Dependencies

Use these packages where appropriate:

```bash
pip install datasets convokit tqdm pandas numpy sentence-transformers
```

For model generation and training stages later:

```bash
pip install transformers peft trl accelerate torch
```

Do not force install packages inside scripts unless the repository already does that. Prefer documenting missing dependencies clearly.

---

# 10. Suggested Script Interfaces

## 10.1 SFT Preparation Script

Create or update:

```text
scripts/prepare_sft_datasets.py
```

Suggested CLI:

```bash
python scripts/prepare_sft_datasets.py --persona contrarian --limit 5000
python scripts/prepare_sft_datasets.py --persona systems_thinker --limit 5000
python scripts/prepare_sft_datasets.py --persona cross_domain_analogist --limit 5000
python scripts/prepare_sft_datasets.py --persona minimalist --limit 5000
python scripts/prepare_sft_datasets.py --all --limit 5000
```

Options:

```text
--persona
--all
--limit
--output-dir data/processed/sft
--write-examples outputs/data_checks/sft_examples.md
--streaming true/false where supported
```

---

## 10.2 DivPO Preparation Script

Create or update:

```text
scripts/prepare_divpo_datasets.py
```

Suggested CLI:

```bash
python scripts/prepare_divpo_datasets.py --persona contrarian --candidate-count 4
python scripts/prepare_divpo_datasets.py --all --candidate-count 4
```

Options:

```text
--persona
--all
--candidate-count
--adapter-dir outputs/adapters/sft
--prompt-pool data/processed/sft
--output-dir data/processed/divpo
--min-quality 0.35
--quality-weight 0.4
--rarity-weight 0.6
--temperature 0.9
--top-p 0.95
--max-new-tokens 250
```

Important: this script requires trained SFT adapters. If adapters are missing, fail with a clear error message and explain that SFT training must be run first.

---

# 11. Important Implementation Rules

## 11.1 Keep the Pipeline Modular

Do not put all logic into one script.

Separate:

```text
data loading
data filtering
persona prompts
record formatting
DivPO scoring
DivPO pair selection
CLI orchestration
```

## 11.2 Preserve Raw and Processed Data Distinction

Raw data should be treated as source material.

Processed data should be persona-shaped and training-ready.

Do not overwrite raw data.

## 11.3 Do Not Assume Dataset Fields Blindly

For Hugging Face datasets, inspect actual row keys before transforming.

Use robust access:

```python
row.get("field_name")
```

and fail gracefully if required fields are unavailable.

## 11.4 Log Filtering Statistics

For each dataset, count how many examples were removed by each filter:

```text
empty_input
empty_response
too_short
too_long
contains_deleted_marker
mostly_links
mostly_code
low_quality
missing_field
other
```

## 11.5 Make Outputs Inspectable

Always save example previews in markdown.

The human researcher must be able to inspect the first few examples before training.

---

# 12. What Not to Implement Yet

Do not implement the following until dataset preparation is complete:

```text
- Gradio UI
- web app
- multi-turn chat
- IdeaCard product layer
- PersonaSession
- deployment
- full paper writing
- complicated router training
- MixLoRA kernel
- human study
```

The current priority is clean, inspectable data preparation.

---

# 13. Later Evaluation Context

The final research must compare these baselines:

```text
1. Base model
2. Prompt-only
3. Single LoRA
4. MoP-SFT
5. MoP + DivPO
```

This matters because the method only becomes scientifically meaningful if the evaluation can answer:

```text
Does prompting alone help?
Does fine-tuning help?
Does separating personas help?
Does DivPO add diversity beyond SFT?
```

The core metrics should include:

```text
Self-BLEU lower is better
Distinct-1 higher is better
Distinct-2 higher is better
SBERT pairwise cosine lower is better
Corpus novelty higher is better
LLM-judge quality higher is better
LLM-judge novelty higher is better
LLM-judge utility higher is better
LLM-judge persona fidelity higher is better
```

But do not implement full evaluation yet unless explicitly asked. Dataset preparation comes first.

---

# 14. Success Criteria for This Agent Task

This task is complete when the repository can produce structurally valid and manually inspectable files:

```text
data/processed/sft/contrarian.jsonl
data/processed/sft/systems_thinker.jsonl
data/processed/sft/cross_domain_analogist.jsonl
data/processed/sft/minimalist.jsonl
```

And when the code for later DivPO dataset creation exists and can produce, after SFT adapters are trained:

```text
data/processed/divpo/contrarian.jsonl
data/processed/divpo/systems_thinker.jsonl
data/processed/divpo/cross_domain_analogist.jsonl
data/processed/divpo/minimalist.jsonl
```

Each output file must have:

- valid JSONL structure,
- correct persona identity,
- correct prompt/response format,
- reasonable filtering,
- summary statistics,
- preview examples.

---

# 15. Immediate First Step for the Agent

Start with SFT dataset preparation only.

Recommended first implementation order:

```text
1. Implement persona prompts.
2. Implement generic JSONL writer and preview writer.
3. Implement filtering utilities.
4. Implement Contrarian / CGA-CMV extraction.
5. Generate data/processed/sft/contrarian.jsonl.
6. Print and save summary stats.
7. Only then implement the other personas.
```

Do not attempt all four personas at once.

The first milestone is:

```text
A clean, inspectable contrarian SFT dataset from CGA-CMV.
```

Once that works, extend to:

```text
systems_thinker -> Stack Exchange
cross_domain_analogist -> ArXiv abstracts
minimalist -> IBM Argument Quality
```

---

# 16. Final Guiding Principle

The project should be coded like a research system, not a demo.

Every file should answer a research question:

```text
prepare_sft.py
What data teaches each persona how to behave?

prepare_divpo.py
Which rare-but-good responses should be preferred?

scoring.py
How do we estimate quality and rarity?

pairs.py
How do we construct valid DPO preference pairs?

filters.py
How do we prevent noisy data from corrupting the adapters?
```

Do not optimize for speed of implementation at the cost of scientific clarity.

Clean data first. Training second. Evaluation third.
