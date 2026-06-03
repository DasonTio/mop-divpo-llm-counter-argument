#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/mop_divpo_llm-counter-argument}"
PY="${PY:-/venv/main/bin/python}"
INSTANCE_ID="${INSTANCE_ID:-39141905}"
LOG_DIR="$ROOT/outputs/scaling_run"
STATE_DIR="$LOG_DIR/state"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$ROOT/outputs/adapters"
cd "$ROOT"

set -a
source .env
set +a

export PYTHONPATH=src
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.matplotlib}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PERSONAS=(contrarian systems_thinker cross_domain_analogist minimalist)

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/driver.log"
}

run_step() {
  local name="$1"
  shift
  local marker="$STATE_DIR/${name}.done"
  if [[ -f "$marker" ]]; then
    log "SKIP $name: marker exists"
    return 0
  fi
  log "START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  touch "$marker"
  log "DONE $name"
}

run_shell_step() {
  local name="$1"
  shift
  local marker="$STATE_DIR/${name}.done"
  if [[ -f "$marker" ]]; then
    log "SKIP $name: marker exists"
    return 0
  fi
  log "START $name"
  bash -lc "$*" 2>&1 | tee "$LOG_DIR/${name}.log"
  touch "$marker"
  log "DONE $name"
}

log "Scaling validation started on instance $INSTANCE_ID"
nvidia-smi | tee "$LOG_DIR/nvidia_smi_start.txt"
df -h /workspace | tee "$LOG_DIR/df_start.txt"

run_step "remote_unit_tests" \
  "$PY" -m unittest tests.test_inference_and_aggregate tests.test_train_divpo_compat

run_step "qwen25_3b_generate_automated" \
  "$PY" scripts/run_baseline_evaluation.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --methods base prompt_only \
    --outdir outputs/evaluation_qwen25_3b_scale \
    --judge none

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  run_step "qwen25_3b_judge" \
    "$PY" scripts/run_baseline_evaluation.py \
      --base-model Qwen/Qwen2.5-3B-Instruct \
      --methods base prompt_only \
      --outdir outputs/evaluation_qwen25_3b_scale \
      --skip-generation \
      --judge openai
else
  log "SKIP qwen25_3b_judge: OPENAI_API_KEY not set"
fi

for persona in "${PERSONAS[@]}"; do
  if [[ -f "outputs/adapters/sft_1p5b/$persona/adapter_config.json" ]]; then
    log "SKIP sft_1p5b_$persona: adapter exists"
    continue
  fi
  if ! run_step "sft_1p5b_$persona" \
    "$PY" scripts/train_sft.py \
      --persona "$persona" \
      --dataset-dir data/processed/sft \
      --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --output-stage sft_1p5b \
      --epochs 1 \
      --batch-size 2 \
      --grad-accum 16 \
      --max-seq-len 512 \
      --no-push; then
    log "Fallback SFT for $persona with batch=1, seq=384"
    rm -rf "outputs/adapters/sft_1p5b/$persona"
    run_step "sft_1p5b_${persona}_fallback" \
      "$PY" scripts/train_sft.py \
        --persona "$persona" \
        --dataset-dir data/processed/sft \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-stage sft_1p5b \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 32 \
        --max-seq-len 384 \
        --no-push
  fi
done

if ! run_step "prepare_divpo_v2_1p5b" \
  "$PY" scripts/prepare_divpo_datasets.py \
    --cross-persona \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir outputs/adapters/sft_1p5b \
    --quality-weight 0.5 \
    --rarity-weight 0.5 \
    --min-quality 0.40 \
    --candidate-count 4 \
    --limit 300 \
    --gen-batch-size 4 \
    --output-dir data/processed/divpo_v2_1p5b; then
  log "Fallback DivPO v2 data generation with limit=250, gen_batch=2"
  rm -rf data/processed/divpo_v2_1p5b
  run_step "prepare_divpo_v2_1p5b_fallback" \
    "$PY" scripts/prepare_divpo_datasets.py \
      --cross-persona \
      --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter-dir outputs/adapters/sft_1p5b \
      --quality-weight 0.5 \
      --rarity-weight 0.5 \
      --min-quality 0.40 \
      --candidate-count 4 \
      --limit 250 \
      --gen-batch-size 2 \
      --output-dir data/processed/divpo_v2_1p5b
fi

for persona in "${PERSONAS[@]}"; do
  if [[ -f "outputs/adapters/divpo_v2_1p5b/$persona/adapter_config.json" ]]; then
    log "SKIP divpo_v2_1p5b_$persona: adapter exists"
    continue
  fi
  run_step "divpo_v2_1p5b_$persona" \
    "$PY" scripts/train_divpo.py \
      --persona "$persona" \
      --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --sft-stage sft_1p5b \
      --sft-adapter-dir outputs/adapters \
      --output-stage divpo_v2_1p5b \
      --dataset-dir data/processed/divpo_v2_1p5b \
      --epochs 1 \
      --batch-size 1 \
      --grad-accum 16 \
      --max-length 384 \
      --no-push
done

run_step "qwen25_1p5b_generate_automated" \
  "$PY" scripts/run_baseline_evaluation.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-prefix outputs/adapters \
    --methods base prompt_only mop_sft mop_divpo_v2 \
    --method-stage mop_sft=sft_1p5b \
    --method-stage mop_divpo_v2=divpo_v2_1p5b \
    --method-sft-stage mop_divpo_v2=sft_1p5b \
    --headline mop_divpo_v2 \
    --outdir outputs/evaluation_qwen25_1p5b_scale \
    --judge none

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  run_step "qwen25_1p5b_judge" \
    "$PY" scripts/run_baseline_evaluation.py \
      --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter-prefix outputs/adapters \
      --methods base prompt_only mop_sft mop_divpo_v2 \
      --method-stage mop_sft=sft_1p5b \
      --method-stage mop_divpo_v2=divpo_v2_1p5b \
      --method-sft-stage mop_divpo_v2=sft_1p5b \
      --headline mop_divpo_v2 \
      --outdir outputs/evaluation_qwen25_1p5b_scale \
      --skip-generation \
      --judge openai
else
  log "SKIP qwen25_1p5b_judge: OPENAI_API_KEY not set"
fi

df -h /workspace | tee "$LOG_DIR/df_end.txt"
nvidia-smi | tee "$LOG_DIR/nvidia_smi_end.txt"
touch "$STATE_DIR/ALL_DONE"
log "Scaling validation completed. Artifacts ready for download."
