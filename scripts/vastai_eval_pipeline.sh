#!/usr/bin/env bash
# Master pipeline: install deps → ArmoRM eval → CMV ground truth → significance → destroy instance.
# Run on the vast.ai instance after rsyncing the repo.
#
# Usage:
#   bash scripts/vastai_eval_pipeline.sh <EVAL_DIR> <VASTAI_INSTANCE_ID> <VASTAI_API_KEY>
#
# Example:
#   bash scripts/vastai_eval_pipeline.sh outputs/evaluation_a100 39296848 <YOUR_KEY>

set -euo pipefail

EVAL_DIR="${1:-outputs/evaluation_a100}"
INSTANCE_ID="${2:-}"
VASTAI_API_KEY="${3:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$(dirname "$0")/.."
export PYTHONPATH=src

# ── 1. Install dependencies ────────────────────────────────────────────────
log "Installing dependencies..."
pip install -q bitsandbytes bert-score convokit sentence-transformers \
    transformers accelerate torch numpy scipy 2>&1 | tail -5
log "Dependencies installed."

# ── 2. ArmoRM quality scoring on existing 720 outputs ─────────────────────
log "=== Phase 1: ArmoRM quality scoring ==="
python scripts/eval_armorm.py \
    --dir "$EVAL_DIR" \
    --load-in-4bit \
    --model "RLHFlow/ArmoRM-Llama3-8B-v0.1"
log "ArmoRM scoring complete."

# ── 3. CMV delta ground truth ──────────────────────────────────────────────
log "=== Phase 2: CMV delta BERTScore ground truth ==="
python scripts/eval_cmv_groundtruth.py \
    --dir "$EVAL_DIR" \
    --top-k 3
log "CMV ground truth complete."

# ── 4. Significance report with all metrics ────────────────────────────────
log "=== Phase 3: Significance report ==="
python scripts/significance_report.py \
    --dir "$EVAL_DIR" \
    --reference mop_divpo_v2 \
    --n-resamples 10000
log "Significance report complete."

# ── 5. Summary ────────────────────────────────────────────────────────────
log "=== All phases complete. Results in $EVAL_DIR ==="
echo ""
echo "New files:"
ls -lh "$EVAL_DIR"/armorm_*.{md,json} "$EVAL_DIR"/cmv_groundtruth_*.{md,json} \
        "$EVAL_DIR"/significance_mop_divpo_v2.* 2>/dev/null || true
echo ""
cat "$EVAL_DIR/armorm_table.md" 2>/dev/null || true
echo ""
cat "$EVAL_DIR/cmv_groundtruth_table.md" 2>/dev/null || true
echo ""
cat "$EVAL_DIR/significance_mop_divpo_v2.md" 2>/dev/null || true

# ── 6. Auto-destroy instance ───────────────────────────────────────────────
if [[ -n "$INSTANCE_ID" && -n "$VASTAI_API_KEY" ]]; then
    log "Destroying vast.ai instance $INSTANCE_ID to stop billing..."
    curl -s -X DELETE \
        "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
        -H "Authorization: Bearer ${VASTAI_API_KEY}" \
        -H "Content-Type: application/json" \
        --data '{}'
    log "Instance $INSTANCE_ID destroyed. Billing stopped."
else
    log "WARNING: No INSTANCE_ID or VASTAI_API_KEY — instance NOT destroyed."
    log "Destroy manually: vastai destroy instance $INSTANCE_ID"
fi
