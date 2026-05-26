# Colab Training Guide

## Objective

Train 4 persona-specific LoRA adapters (SFT + DivPO) on Qwen2.5-0.5B-Instruct.
Each adapter learns a distinct cognitive style for counter-argument generation.

## Pipeline

```
[DONE]   prepare_sft_datasets.py → data on HF Hub
[Colab]  train_sft.py            → SFT adapters pushed to HF Hub
[Colab]  prepare_divpo.py        → DivPO preference pairs generated on GPU, pushed to HF Hub
[Colab]  train_divpo.py          → DivPO adapters pushed to HF Hub
```

---

## Resources

| Resource | URL |
|---|---|
| GitHub | https://github.com/DasonTio/mop-divpo-llm-counter-argument |
| HF Token | Store as a private Colab Secret named `HF_TOKEN` |
| SFT datasets | https://huggingface.co/datasets/DasonTio/mop-divpo-sft-data |
| DivPO datasets | https://huggingface.co/datasets/DasonTio/mop-divpo-divpo-data |
| Adapters | https://huggingface.co/DasonTio/mop-divpo-coauthor |
| Base model | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |

---

## Status

- [x] SFT datasets prepared and pushed to `DasonTio/mop-divpo-sft-data`
  - contrarian: 3,662 records (CGA-CMV)
  - systems_thinker: 5,000 records (StackExchange)
  - cross_domain_analogist: 5,000 records (ArXiv abstracts)
  - minimalist: 5,000 records (IBM Argument Quality)
- [x] SFT adapters trained and pushed to `DasonTio/mop-divpo-coauthor/sft/{persona}/`
- [ ] DivPO datasets prepared
- [ ] DivPO adapters trained

---

## Colab Setup

**Runtime:** Use GPU (Runtime → Change runtime type → T4 GPU or better)

---

### Cell 1 — Install dependencies

```python
!pip install -q transformers peft trl accelerate bitsandbytes datasets huggingface_hub sentence-transformers
!pip uninstall -y -q torchao
```

Colab may preinstall an old `torchao` build. This project does not use torchao,
but PEFT will still detect incompatible versions during LoRA injection.
After uninstalling it, restart the runtime once before training if you already
imported `peft` or started a failed training run.

---

### Cell 2 — Clone repo and set credentials

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!git clone https://github.com/DasonTio/mop-divpo-llm-counter-argument.git
%cd mop-divpo-llm-counter-argument
```

---

### Cell 3 — Verify GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None — switch runtime to GPU")
```

---

## Phase 1 — SFT Training

Train 4 persona LoRA adapters on the prepared SFT datasets.
Each adapter learns from ~3,600–5,000 domain-specific examples.

### Cell 4 — Train all SFT adapters (~1–2h per persona on T4)

```python
# Trains contrarian → systems_thinker → cross_domain_analogist → minimalist sequentially
# Each adapter auto-pushed to DasonTio/mop-divpo-coauthor/sft/{persona}/
!python scripts/train_sft.py --all
```

Or train one at a time:

```python
# Run these cells one by one if you want to checkpoint between personas
!python scripts/train_sft.py --persona contrarian
!python scripts/train_sft.py --persona systems_thinker
!python scripts/train_sft.py --persona cross_domain_analogist
!python scripts/train_sft.py --persona minimalist
```

Adapters saved locally to `outputs/adapters/sft/{persona}/` and pushed to HF Hub.

---

## Phase 2 — DivPO Dataset Generation

Generate preference pairs from the trained SFT adapters.
Each prompt gets N candidate responses; rare-but-good is `chosen`, common-or-weak is `rejected`.

### Cell 5 — Download SFT JSONL for prompt pool

```python
import os
from huggingface_hub import hf_hub_download

os.makedirs("data/processed/sft", exist_ok=True)
for persona in ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]:
    hf_hub_download(
        repo_id="DasonTio/mop-divpo-sft-data",
        filename=f"{persona}.jsonl",
        repo_type="dataset",
        local_dir="data/processed/sft",
        token=os.environ["HF_TOKEN"],
    )
    print(f"Downloaded {persona}.jsonl")
```

### Cell 6 — Generate DivPO preference pairs and push (~30–60 min on T4)

`--push` uploads each persona's pairs to HF Hub immediately after generation.
Model is loaded once per persona and freed from VRAM before the next one.

```python
# All 4 personas sequentially — model reloaded + VRAM freed between each
!python scripts/prepare_divpo_datasets.py --all --from-hub --candidate-count 4 --push
```

Or per persona (run one cell at a time to checkpoint):

```python
!python scripts/prepare_divpo_datasets.py --persona contrarian --from-hub --candidate-count 4 --push
```
```python
!python scripts/prepare_divpo_datasets.py --persona systems_thinker --from-hub --candidate-count 4 --push
```
```python
!python scripts/prepare_divpo_datasets.py --persona cross_domain_analogist --from-hub --candidate-count 4 --push
```
```python
!python scripts/prepare_divpo_datasets.py --persona minimalist --from-hub --candidate-count 4 --push
```

Data saved locally to `data/processed/divpo/{persona}.jsonl` and uploaded to
`DasonTio/mop-divpo-divpo-data/{persona}.jsonl`.

> **If you ran without `--push`**, upload manually:
> ```python
> !python scripts/push_to_hub.py --divpo
> ```

---

## Phase 3 — DivPO Training

Fine-tune from SFT adapters using DPO on the preference pairs.
Teaches the model to prefer rare-but-good responses over common ones.

### Cell 8 — Train all DivPO adapters (~1h per persona on T4)

```python
# Pulls SFT adapters + DivPO datasets from HF Hub
# Trains DPO on preference pairs
# Pushes to DasonTio/mop-divpo-coauthor/divpo/{persona}/
!python scripts/train_divpo.py --all
```

Or per persona:

```python
!python scripts/train_divpo.py --persona contrarian
!python scripts/train_divpo.py --persona systems_thinker
!python scripts/train_divpo.py --persona cross_domain_analogist
!python scripts/train_divpo.py --persona minimalist
```

---

## Verify — Load a trained adapter

After training completes, verify an adapter loads and generates:

```python
import torch
import sys
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")

# Load SFT contrarian adapter
model = PeftModel.from_pretrained(base, "DasonTio/mop-divpo-coauthor", subfolder="sft/contrarian")
model.eval()

prompt = "Generate a counter-argument to this claim:\n\nRemote work is strictly better for productivity."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, temperature=0.9, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## HF Hub layout

| Resource | Repo | Path |
|---|---|---|
| SFT datasets | `DasonTio/mop-divpo-sft-data` | `{persona}.jsonl` |
| DivPO datasets | `DasonTio/mop-divpo-divpo-data` | `{persona}.jsonl` |
| SFT adapters | `DasonTio/mop-divpo-coauthor` | `sft/{persona}/` |
| DivPO adapters | `DasonTio/mop-divpo-coauthor` | `divpo/{persona}/` |

Personas: `contrarian` · `systems_thinker` · `cross_domain_analogist` · `minimalist`
