#!/bin/bash
# Multi-GPU parallel precomputation of DINOv2 features (train set)
# Each GPU acts as a worker, sharding scenes by modulo for parallel processing

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STEP3_ROOT="/path/to/data/HM3D/DATA_GEN/step3_render_224_224_uint16_train_2_5ins"
OUTPUT_ROOT="/path/to/data/HM3D/DATA_GEN/step6_dinov2_features_224_train_2_5ins"
MODEL_PATH="/path/to/map-anything-model/"
LOG_DIR="${SCRIPT_DIR}/logs_step6_train"

mkdir -p "$LOG_DIR"

# ==================== User configuration ====================
GPUS=(0 1 2 3)
BATCH_SIZE=32               # Number of images fed into DINOv2 per batch
MAX_TRAJS=25                # Maximum number of trajectories per scene
EXTRA_ARGS="--skip-existing"  # Enable resume/skip-existing
# ============================================================

NUM_WORKERS=${#GPUS[@]}

echo "=== Step 6: Multi-GPU parallel precomputation of DINOv2 features (train) ==="
echo "  Step3 root: $STEP3_ROOT"
echo "  Output root: $OUTPUT_ROOT"
echo "  Model path: $MODEL_PATH"
echo "  GPU: ${GPUS[*]}"
echo "  Workers: $NUM_WORKERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Max trajs/scene: $MAX_TRAJS"
echo "  Log dir: $LOG_DIR"
echo "  Extra args: $EXTRA_ARGS"
echo ""

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    LOG_FILE="${LOG_DIR}/worker_${i}_gpu${GPU_ID}.log"

    echo "Starting Worker $i (GPU $GPU_ID) -> $LOG_FILE"

    python "$SCRIPT_DIR/step6_precompute_dinov2_features.py" \
        --step3-root "$STEP3_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --model-path "$MODEL_PATH" \
        --gpu "$GPU_ID" \
        --batch-size "$BATCH_SIZE" \
        --max-trajs "$MAX_TRAJS" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$i" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All workers started, PID: ${PIDS[*]}"
echo "View logs: tail -f $LOG_DIR/worker_*.log"
echo "Waiting for all workers to finish..."
echo ""

# Wait for all processes to finish and report exit status
FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    GPU_ID=${GPUS[$i]}
    if [ $CODE -ne 0 ]; then
        echo "[FAIL] Worker $i (GPU $GPU_ID) exit code: $CODE"
        FAIL=$((FAIL + 1))
    else
        echo "[DONE] Worker $i (GPU $GPU_ID) finished"
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== All workers completed successfully ==="
else
    echo "=== $FAIL worker(s) failed, check logs in $LOG_DIR ==="
    exit 1
fi
