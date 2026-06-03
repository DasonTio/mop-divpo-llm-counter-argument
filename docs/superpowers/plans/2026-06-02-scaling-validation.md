# Scaling Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce paper-ready scaling evidence for Qwen2.5 by running full 1.5B MoP+DivPO v2 and 3B base/prompt-only baselines.

**Architecture:** Keep the existing 0.5B results untouched. Add script parameters for base model and isolated adapter stage names, then run larger-model artifacts into separate output directories. Do not full-train 3B DivPO on RTX 5060 Ti 16GB because the current fp16 DPO implementation loads policy and reference models.

**Tech Stack:** Python, Hugging Face Transformers/PEFT/TRL, Qwen2.5, Vast.ai RTX 5060 Ti, OpenAI judge cache.

---

### Task 1: Make Training Stage Names Configurable

**Files:**
- Modify: `scripts/train_sft.py`
- Modify: `scripts/train_divpo.py`

- [ ] Add `--base-model` and `--output-stage` to `train_sft.py`.
- [ ] Use `args.base_model` for tokenizer/model loading.
- [ ] Push SFT adapters to `MODEL_REPO/{output_stage}/{persona}` instead of always `sft/{persona}`.
- [ ] Add `--base-model` and `--sft-stage` to `train_divpo.py`.
- [ ] Load SFT adapters from `{sft_stage}/{persona}` and push DivPO adapters to `{output_stage}/{persona}`.
- [ ] Verify parser help for both scripts exits 0.

### Task 2: Make Evaluation Accept Scaling Methods

**Files:**
- Modify: `scripts/run_baseline_evaluation.py`

- [ ] Add `--base-model` for generation.
- [ ] Add `--method-stage METHOD=STAGE` overrides, allowing `mop_sft=sft_1p5b` and `mop_divpo_v2=divpo_v2_1p5b`.
- [ ] Add `--headline` so significance can target `mop_divpo_v2` for scaling runs.
- [ ] Verify a smoke parser run with `--judge none --limit-prompts 1 --skip-generation` fails only if no generations file exists.

### Task 3: Sync and Prepare Vast Instance

**Files:**
- Remote: `/workspace/mop_divpo_llm-counter-argument`

- [ ] Copy the current repo to Vast.
- [ ] Copy `.env` without printing secrets.
- [ ] Install Python dependencies if missing.
- [ ] Set `HF_HOME`, `TRANSFORMERS_CACHE`, and `HF_DATASETS_CACHE` under `/workspace/.cache`.
- [ ] Verify `nvidia-smi`, `torch.cuda.is_available()`, and repository imports.

### Task 4: Run Scaling Jobs

**Remote commands:**
- `python scripts/run_baseline_evaluation.py --base-model Qwen/Qwen2.5-3B-Instruct --methods base prompt_only --outdir outputs/evaluation_qwen25_3b_scale --only-generate`
- `python scripts/train_sft.py --all --base-model Qwen/Qwen2.5-1.5B-Instruct --output-stage sft_1p5b --epochs 2 --batch-size 4 --grad-accum 8 --max-seq-len 512`
- `python scripts/prepare_divpo_datasets.py --cross-persona --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-dir outputs/adapters/sft_1p5b --quality-weight 0.5 --rarity-weight 0.5 --min-quality 0.40 --candidate-count 4 --limit 300 --gen-batch-size 4 --output-dir data/processed/divpo_v2_1p5b`
- `python scripts/train_divpo.py --all --base-model Qwen/Qwen2.5-1.5B-Instruct --sft-stage sft_1p5b --output-stage divpo_v2_1p5b --dataset-dir data/processed/divpo_v2_1p5b --epochs 1 --batch-size 1 --grad-accum 16 --max-length 384`
- `python scripts/run_baseline_evaluation.py --base-model Qwen/Qwen2.5-1.5B-Instruct --methods base prompt_only mop_sft mop_divpo_v2 --method-stage mop_sft=sft_1p5b --method-stage mop_divpo_v2=divpo_v2_1p5b --headline mop_divpo_v2 --outdir outputs/evaluation_qwen25_1p5b_scale --only-generate`

### Task 5: Judge, Download, and Stop Billing

**Files:**
- Download: `outputs/evaluation_qwen25_3b_scale/`
- Download: `outputs/evaluation_qwen25_1p5b_scale/`
- Download: `data/processed/divpo_v2_1p5b/`
- Download: `outputs/adapters/sft_1p5b/`
- Download: `outputs/adapters/divpo_v2_1p5b/`

- [ ] Run judge/aggregation on generated outputs if API keys are available.
- [ ] Copy all scaling outputs back to local `outputs/`.
- [ ] Try `vastai destroy instance 39141905` if CLI/API key is configured.
- [ ] If Vast destroy is unavailable, run `poweroff` and report that manual destroy is still required.
