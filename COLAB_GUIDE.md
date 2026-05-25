# Colab Training Guide

## Full pipeline order

```
[Local]  prepare data  →  [Local]  push data to HF  →  [Colab]  train SFT
→  [Local]  prepare DivPO  →  [Local]  push DivPO to HF  →  [Colab]  train DivPO
```

---

## Step 1 — Prepare SFT datasets (local)

```bash
source .venv/bin/activate
python scripts/prepare_sft_datasets.py --all --limit 5000
```

Outputs: `data/processed/sft/{persona}.jsonl`

## Step 2 — Push SFT data to HF Hub (local)

```bash
python scripts/push_to_hub.py --sft --token $HF_TOKEN
```

Uploads to: `DasonTio/mop-divpo-sft-data`

## Step 3 — Train SFT adapters (Google Colab)

```python
# Cell 1: Install
!pip install transformers peft trl accelerate bitsandbytes datasets huggingface_hub

# Cell 2: Set token + clone
import os
os.environ["HF_TOKEN"] = "$HF_TOKEN"
!git clone https://github.com/YOUR_USERNAME/mop_divpo_llm-counter-argument.git
%cd mop_divpo_llm-counter-argument

# Cell 3: Train all personas (sequentially, ~1-2h per persona on T4)
!python scripts/train_sft.py --all --token $HF_TOKEN
# Or train one at a time:
# !python scripts/train_sft.py --persona contrarian --token $HF_TOKEN
```

Each adapter is automatically pushed to: `DasonTio/mop-divpo-coauthor/sft/{persona}/`

## Step 4 — Prepare DivPO datasets (local, after SFT training)

```bash
# Pulls SFT adapters from HF Hub, generates candidates, builds preference pairs
python scripts/prepare_divpo_datasets.py --all --from-hub --candidate-count 4 \
    --token $HF_TOKEN

# NOTE: this runs model inference locally — needs GPU or is slow on CPU
```

Outputs: `data/processed/divpo/{persona}.jsonl`

## Step 5 — Push DivPO data to HF Hub (local)

```bash
python scripts/push_to_hub.py --divpo --token $HF_TOKEN
```

Uploads to: `DasonTio/mop-divpo-divpo-data`

## Step 6 — Train DivPO adapters (Google Colab)

```python
# Cell: Train all DivPO adapters
!python scripts/train_divpo.py --all --token $HF_TOKEN
```

Each adapter pushed to: `DasonTio/mop-divpo-coauthor/divpo/{persona}/`

---

## HuggingFace Hub layout

| Resource | Repo | Path |
|---|---|---|
| SFT datasets | `DasonTio/mop-divpo-sft-data` | `{persona}.jsonl` |
| DivPO datasets | `DasonTio/mop-divpo-divpo-data` | `{persona}.jsonl` |
| SFT adapters | `DasonTio/mop-divpo-coauthor` | `sft/{persona}/` |
| DivPO adapters | `DasonTio/mop-divpo-coauthor` | `divpo/{persona}/` |

---

## Loading trained adapters (inference)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(
    base,
    "DasonTio/mop-divpo-coauthor",
    subfolder="divpo/contrarian",   # or sft/contrarian, divpo/systems_thinker, etc.
)
```
