#!/bin/bash
# VideoMAE v2 feature extraction for low-VRAM environment (GPU 0, <=16GB target)

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-codes/VideoMAEv2/configs/extract_vit_b_gpu0_16gb.yaml}"
OUT_DIR="${OUT_DIR:-data/features/30s_mae_b_16gb_gpu0}"
MAX_LIMIT="${MAX_LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG" >&2
  exit 1
fi

# Keep execution pinned to GPU 0 explicitly.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --config "$CONFIG"
  --out-dir "$OUT_DIR"
  --gpu-id 0
  --batch-size "$BATCH_SIZE"
  --decode-mode batch
  --max-batch-size-batch-decode "$BATCH_SIZE"
)

if [[ -n "$MAX_LIMIT" ]]; then
  ARGS+=(--limit "$MAX_LIMIT")
fi

if (( OVERWRITE == 1 )); then
  ARGS+=(--overwrite)
fi

echo "======================================================================="
echo "VideoMAE v2 extraction (GPU 0, low VRAM profile)"
echo "======================================================================="
echo "Config      : $CONFIG"
echo "Output Dir  : $OUT_DIR"
echo "Batch Size  : $BATCH_SIZE"
echo "GPU         : 0 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
if [[ -n "$MAX_LIMIT" ]]; then
  echo "Max Limit   : $MAX_LIMIT"
fi
echo "======================================================================="

uv run --no-sync python -m VideoMAEv2.extract_features "${ARGS[@]}"
