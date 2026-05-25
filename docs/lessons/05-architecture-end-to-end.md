# Lesson 05 — End-to-End Architecture

> **Goal of this lesson.** Lessons 01-04 introduced the ideas one at a time. This lesson is the *map*. It walks through how the data flows through the system from raw datasets all the way to a user-visible idea, both at training time and at inference time. By the end you should be able to point at any file in `src/mop_divpo/` and say what stage it belongs to and why.

---

## 1. Two flows, not one

A common mistake when learning ML systems is to think of "the model" as one thing. Real ML systems have **at least two flows**:

- **Training time** — runs once (or rarely), produces artefacts (adapters, configs, indexes).
- **Inference time** — runs on every user request, consumes the artefacts produced at training time.

These flows touch a lot of the same code, but they have very different shapes. Always know which flow you are looking at. This lesson treats them separately.

The high-level diagram (taken from `docs/research-report.md`, slightly simplified):

```
TRAINING TIME
─────────────────────────────────────────────────────────────────────────────
Raw Datasets → Normalizers → SFT JSONL files per persona
                                      │
                                      ▼
                          Base Model (Qwen2.5-0.5B, frozen)
                          + LoRA config (r=16, α=32)
                                      │
                              SFT Training × 4 personas
                                      │
                          ┌───────────┴───────────┐
                          │   DivPO Pair Building │
                          │   (rarity + quality)  │
                          └───────────┬───────────┘
                                      │
                              4 LoRA Adapters saved /
                              pushed to HuggingFace Hub

INFERENCE TIME
─────────────────────────────────────────────────────────────────────────────
User Prompt ──→ Persona Gate ──→ top-k persona IDs
                                            │
                       ┌────────────────────┤ (optional)
                       ▼                    │
              CorpusRetriever          Adapter load
              top-3 prior art          per persona
              → PRIOR ART block             │
                       └────────────────────▼
                              Parallel generation per persona
                                            │
                              Output Parser (idea / reasoning / unexpected_angle)
                                            │
                              Diverse idea set returned to user
```

The rest of this lesson walks through each box.

---

## 2. Training time — step by step

### 2.1 Step 1: dataset acquisition (`data/acquire.py`, `data/sources.py`)

The pipeline begins with `mop-divpo collect-source <name>`. This streams data from HuggingFace (no full download) into `data/raw/`.

`src/mop_divpo/data/sources.py` holds the registry of all known sources:

```python
# Conceptual structure — actual record shape may differ slightly.
DatasetSource(
    name="cga_cmv",
    hf_dataset_id="mc-ai/conversations-gone-awry-cmv",
    method="huggingface",
    target_personas=["contrarian"],
    url="https://convokit.cornell.edu/documentation/awry_cmv.html",
)
```

The registry is the *single source of truth* for "where does data for persona X come from?" Adding a new source means appending one record. This is a deliberate **data-as-configuration** pattern — datasets are not hardcoded into the training scripts.

### 2.2 Step 2: normalization (`data/normalizers.py`)

Raw HuggingFace datasets all have different shapes. Reddit threads have nested `comments[]`. ArXiv abstracts have `title` + `abstract` + `categories`. Stack Exchange has `question` + `accepted_answer`.

Normalizers reshape every source into one shared schema (`schema.py::SFTRecord`):

```python
{
    "id": "string",
    "persona": "contrarian",
    "source": "cga_cmv",
    "prompt": "Challenge this assumption for pre-writing ideation: …",
    "response": "…the body of the rebuttal…",
    "metadata": {…}
}
```

Why standardise so aggressively? Because downstream training, evaluation, and DivPO pair-building have to be *source-agnostic*. They cannot care whether a record came from Reddit or Project Gutenberg.

The normalizers also encode the cognitive intent of each dataset. For ArXiv, the same abstract can be normalised twice with different prompts:

- Cross-domain: `"Create a cross-domain analogy from this research idea: {title}"`
- Systems: `"Explain the causal system and feedback loops behind this research idea: {title}"`

Same data, different persona-shaped prompts. This is how a single dataset can feed two personas — and a hint at why dataset choice matters so much, which is Lesson 06.

### 2.3 Step 3: per-persona SFT (`training/train_sft.py`)

For each persona, the SFT script:

1. Loads the base model (`Qwen/Qwen2.5-0.5B-Instruct`) frozen.
2. Wraps it with a LoRA adapter (`r=16, α=32`, target modules `q_proj`, `v_proj`).
3. Loads the persona's JSONL file from `data/processed/sft/{persona}.jsonl`.
4. Formats each row using the model's chat template:
   ```
   <|im_start|>system
   {persona ROLE + METHOD}
   <|im_end|>
   <|im_start|>user
   {prompt}
   <|im_end|>
   <|im_start|>assistant
   {response}
   <|im_end|>
   ```
5. Runs `SFTTrainer` for `max_steps`.
6. Saves the adapter to `outputs/adapters/sft/{persona}/`.

Repeat four times. End state: four sharp persona adapters, none of which has learned the others' styles. (This is the architectural argument from Lesson 03 §6.)

### 2.4 Step 4: DivPO candidate generation

For each prompt in the training set, the SFT adapter generates `candidates_per_prompt` candidate responses with stochastic decoding (`candidates_per_prompt: 4` in `configs/prototype.yaml`).

Each candidate becomes a `CandidateRecord` (`schema.py`) with quality and rarity scores attached. See Lesson 04 for the scoring details.

### 2.5 Step 5: DivPO pair construction (`divpo/pairs.py`)

For each prompt's candidate group, `select_divpo_pair(...)` picks the `chosen` and `rejected` responses using the rare-but-good rule from Lesson 04. The output is a stream of `DivPOPairRecord` objects, written to `data/processed/divpo/{persona}.jsonl` in the format expected by `trl dpo`.

### 2.6 Step 6: DPO training on DivPO pairs

The training command is **off-the-shelf `trl dpo`** — no custom trainer code. This is by design: DivPO's contribution lives entirely in step 5. The actual gradient update is standard DPO.

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_name_or_path outputs/adapters/sft/contrarian \
  --dataset_name data/processed/divpo/contrarian.jsonl \
  --output_dir outputs/adapters/divpo/contrarian \
  --use_peft true \
  --max_steps 100
```

Output: a second, improved adapter per persona, stored under `outputs/adapters/divpo/{persona}/`.

### 2.7 Step 7: corpus index for retrieval (`retrieval/corpus.py`)

One more artefact gets built before inference is ready. The `CorpusRetriever` embeds every prompt in `data/processed/sft/*.jsonl` with `all-MiniLM-L6-v2` and stores the embeddings as a search index. At inference, this lets the system find the training examples most similar to the user's prompt — used both for "prior art" injection and for the corpus-novelty metric (Lesson 07).

This is a small but important detail. **Retrieval is not just for the metric.** It is also a runtime input: the top-3 most similar training examples get injected into the system prompt at inference, with an instruction *not to repeat them*. This is a direct anti-plagiarism mechanism that keeps corpus-novelty scores high.

### 2.8 Training-time artefact summary

After a complete training run you should see roughly this on disk:

```
data/processed/sft/{persona}.jsonl       # per-persona training records
data/processed/divpo/{persona}.jsonl     # DivPO chosen/rejected pairs
outputs/adapters/sft/{persona}/          # SFT-only adapters
outputs/adapters/divpo/{persona}/        # DivPO-tuned adapters
(retrieval index — held in memory at startup, rebuilt as needed)
```

These are everything inference needs. No code, no checkpoints, no training infrastructure required at runtime.

---

## 3. Inference time — step by step

Inference is much shorter than training, but it is where the user actually meets the system. Pay close attention to the order of operations.

### 3.1 Step 1: prompt arrives

The user submits a prompt, optionally with `--persona` overrides. The relevant CLI entry points are:

- `mop-divpo infer` — one-shot generation across personas.
- `mop-divpo ideate` — structured IdeaCard output (the co-author workflow; see Lesson 08).
- `mop-divpo chat` — multi-turn session.

All three paths route through `src/mop_divpo/inference/generate.py`.

### 3.2 Step 2: route to personas (`routing/gate.py`)

The persona gate ranks all four personas for the prompt:

```python
gate = build_default_gate()
ranked = gate.rank(prompt, top_k=2, force_personas=user_personas)
```

`build_default_gate()` tries `EmbeddingPersonaGate` first (using SBERT), and falls back to `KeywordPersonaGate` if SBERT is unavailable. (Lesson 03 §7 explains both.) If the user passed `--persona`, those are used directly with full scores of 1.0.

### 3.3 Step 3 (optional): retrieve prior art

If a `CorpusRetriever` is available, the system embeds the user's prompt and pulls the top-3 most similar examples from the training corpus. These get formatted into a "PRIOR ART" block that gets injected into the system prompt:

```text
PRIOR ART — examples already in training data; produce something different:
1. {training prompt} → {training response excerpt}
2. ...
3. ...

Generate a response that does NOT echo any of the above.
```

This is one of the project's small-but-meaningful tricks: rather than hoping the model produces novel output, we **tell it explicitly what to avoid**. The corpus retriever doubles as both a runtime nudge and an evaluation metric source.

### 3.4 Step 4: per-persona generation

For each of the top-`k` personas:

1. Load the persona's adapter (cached after first load — `_load_with_adapter(adapter_id)` in `generate.py`).
2. Build the message list:
   - `system`: the persona's `ROLE + METHOD` from `personas.py`, plus the PRIOR ART block.
   - `user`: the user prompt.
3. Apply the model's chat template and tokenise.
4. Sample from the model.

This is sequential in the prototype — one persona at a time. The MoP design *could* run them in parallel on multiple GPUs; the prototype keeps it sequential because (a) it fits Colab's single GPU and (b) the latency cost is small at the prototype's scale.

### 3.5 Step 5: output parsing (`inference/output_parser.py`)

The model returns raw text. The parser tries to extract a structured idea from it. Two output shapes are supported:

- **One-shot output**: `idea`, `reasoning`, `unexpected_angle` fields parsed via regex/heuristics.
- **Co-author output**: `IdeaCard` JSON with `operation`, `diagnosis`, `idea`, `rationale`, `next_step`. (See Lesson 08.)

The parser returns a structured object with a `parse_success` flag. When parsing fails (the model produced free-form prose without the expected structure), the system falls back to returning the raw text — never crashing.

### 3.6 Step 6: return

The full inference result is a list of `(persona, response)` pairs — one per activated persona. The CLI prints them; the Python API returns them as a list of dicts.

---

## 4. The co-author flow — a third architectural mode

There is a third flow on top of the one-shot inference path: **multi-turn co-author sessions**. The dedicated lesson is 08, but it appears on the architecture map because it changes a few things at inference time.

```
WritingBrief (topic, goal, audience, constraints)
      │
      ▼
PersonaSession (one persona, message history)
      │ turn 1: inject PRIOR ART into system prompt
      │ turn 2+: use full message history (no re-injection)
      ▼
generate_turn() → structured IdeaCard response
      │
      ▼
User continues conversation → session accumulates context
```

Key differences from one-shot:

- The user fills out a structured `WritingBrief` (topic + goal + audience + optional constraints + notes), not just a free-form prompt.
- The system maintains a `PersonaSession` (`inference/session.py`) holding the conversation history.
- PRIOR ART injection happens only on turn 0; later turns rely on the accumulated history for context.
- The expected output is always a structured `IdeaCard`, not free-form prose.

We will dissect the schemas and the agency design in Lesson 08.

---

## 5. The config file — what is tunable

`configs/prototype.yaml` exposes the dials. Worth memorising what each one moves:

```yaml
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

divpo:
  candidates_per_prompt: 4      # how many candidates per DivPO group
  min_quality: 0.35             # safety floor (Lesson 04 §5)
  rarity_weight: 0.6            # weight on rarity vs quality
  quality_weight: 0.4

routing:
  top_k: 2                       # how many personas respond per prompt
```

Read this YAML as a *policy file*. It encodes the project's main research bets:

- "Use a small Qwen so the loop fits in Colab."
- "Generate 4 DivPO candidates, demand at least 0.35 quality, lean slightly into rarity."
- "Route to 2 personas by default — enough for variety, few enough to read."

Changing these numbers is the cheapest way to ablate the system.

---

## 6. Why the system stays modular

A reasonable critique of complex ML systems is "everything is fused together." This project deliberately stays modular. The proof is the layout:

| Layer | Folder | Independence claim |
|---|---|---|
| Data sources | `data/sources.py` | Add a source by appending a record |
| Normalization | `data/normalizers.py` | One function per source |
| Persona definitions | `personas.py` | Single source of truth |
| Routing | `routing/gate.py` | Three swappable implementations (keyword / embedding / random) |
| DivPO scoring | `divpo/scoring.py` | Two pure functions, no model dependency |
| DivPO pair building | `divpo/pairs.py` | One function, no model dependency |
| Training | `training/train_sft.py` | Uses HuggingFace SFTTrainer directly |
| DPO training | `trl dpo` CLI | Off-the-shelf, unmodified |
| Inference | `inference/generate.py` | Pure function — `generate(prompt, personas, ...)` |
| Output parsing | `inference/output_parser.py` | Returns structured result with `parse_success` flag |
| Sessions | `inference/session.py` | Optional layer on top of one-shot |
| Metrics | `metrics/diversity.py`, `metrics/semantic.py` | Independent of model |

Each box can be replaced independently. For example, swapping `KeywordPersonaGate` for `EmbeddingPersonaGate` requires zero changes anywhere else. Replacing the proxy quality score with a learned reward model only touches `divpo/scoring.py`. This is what makes the project a usable *research platform*, not just an experiment.

---

## 7. What is *not* in the architecture (and why)

To stay honest:

- **No reward model.** Quality is currently a proxy. Adding a real reward model is the most important next step.
- **No real MixLoRA kernel.** The prototype approximates MixLoRA by loading adapters sequentially. A real MoE kernel would unlock per-token routing, but is out of scope.
- **No human preference labelling.** Preference pairs come from algorithmic rare-vs-common selection. Real human labels would strengthen DivPO further.
- **No deployment infrastructure.** There is a small `server/` directory and a `web/` directory for demos, but the project is a research codebase, not a production service.

Knowing what is intentionally missing is as important as knowing what is present.

---

## 8. What to take into Lesson 06

You should leave this lesson with three things:

1. **Two flows.** Training builds adapters + corpus index. Inference loads them and routes prompts. Keep these mentally separate.
2. **The data path determines the persona's quality.** Normalizers carry the cognitive intent of each dataset by shaping its prompt. The persona's character lives as much in the data as in the weights.
3. **The system is modular by design.** Every stage has a clear input/output contract, so you can ablate any single component to test what it contributes.

Lesson 06 zooms in on the upstream end: *why does each persona get the dataset it gets?* The dataset-persona alignment is one of the most under-appreciated design decisions in the project, and it determines how sharply each persona ends up specialised.

---

*Previous: [Lesson 04 — DivPO Explained](04-divpo-explained.md) · Next: [Lesson 06 — Datasets as Cognitive Fingerprints](06-datasets-as-cognitive-fingerprints.md)*
