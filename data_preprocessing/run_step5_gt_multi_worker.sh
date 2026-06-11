#!/bin/bash
# Generate G2G index with multiple workers in parallel (GT mode)
#
# Equivalent base command:
# python data_preprocessing/step5_generate_stage2_index.py \
#   --config data_preprocessing/configs/stage2_index_generation_gt.yaml \
#   --split train --mode gt --no-precompute-covis
#
# Usage:
#   bash data_preprocessing/run_step5_gt_multi_worker.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/configs/stage2_index_generation_gt.yaml"

# ==================== User configuration ====================
SPLIT="train"
GPU_ID=0
NUM_WORKERS=100
NUM_DATA_WORKERS=4
EXTRA_ARGS="--skip-existing --no-precompute-covis"
# ==================================================

LOG_DIR="${SCRIPT_DIR}/logs_step5_gt_${SPLIT}_w${NUM_WORKERS}"
mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "  Step 5 GT: ${SPLIT} (${NUM_WORKERS} workers)"
echo "============================================================"
echo "  Config: $CONFIG"
echo "  GPU: $GPU_ID"
echo "  Log dir: $LOG_DIR"
echo "  Extra args: $EXTRA_ARGS"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    LOG_FILE="${LOG_DIR}/worker_${i}.log"
    echo "  Starting Worker $i -> $LOG_FILE"

    python "$SCRIPT_DIR/step5_generate_stage2_index.py" \
        --config "$CONFIG" \
        --split "$SPLIT" \
        --mode gt \
        --gpu "$GPU_ID" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$i" \
        --num-data-workers "$NUM_DATA_WORKERS" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "  All workers started, PID: ${PIDS[*]}"
echo "  View logs: tail -f $LOG_DIR/worker_*.log"
echo "  Waiting..."
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "  [FAIL] Worker $i exit code: $CODE"
        FAIL=$((FAIL + 1))
    else
        echo "  [DONE] Worker $i"
    fi
done

if [ $FAIL -ne 0 ]; then
    echo ""
    echo "=== $FAIL worker(s) failed, check logs in $LOG_DIR ==="
    exit 1
fi

echo ""
echo "=== GT split ${SPLIT} completed successfully ==="
