#!/bin/bash
# Run ActionFormer training with action_type labels.
#
# Usage:
#   bash scripts/run_tal_training.sh
#   DEVICES="0 1 2 3" TAG=exp01 MAX_EPOCHS=30 bash scripts/run_tal_training.sh

set -euo pipefail

cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/work/mt/.uv-cache}"
export HF_HOME="${HF_HOME:-/lustre/work/mt/.cache/huggingface}"
export ANNOT_ROOT_DIR="${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}"
export VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks}"

CONFIG="${CONFIG:-codes/ActionFormer/configs/tal_motion_vit_b.yaml}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-outputs/tal_motion_experiments}"
TAG="${TAG:-action_type_v1}"
DEVICES="${DEVICES:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
RESUME="${RESUME:-}"
PRINT_FREQ="${PRINT_FREQ:-20}"

ARGS=(
  --config "$CONFIG"
  --output-folder "$OUTPUT_FOLDER"
  --tag "$TAG"
  --max-epochs "$MAX_EPOCHS"
  --print-freq "$PRINT_FREQ"
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

read -r -a DEVICE_ARR <<< "$DEVICES"
ARGS+=(--devices "${DEVICE_ARR[@]}")

echo "[train] config=$CONFIG"
echo "[train] output=$OUTPUT_FOLDER tag=$TAG devices=$DEVICES epochs=$MAX_EPOCHS"
NUM_DEVICES="${#DEVICE_ARR[@]}"
if (( NUM_DEVICES > 1 )); then
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${DEVICE_ARR[*]}")"
  export CUDA_VISIBLE_DEVICES
  echo "[train] launching DDP with nproc_per_node=$NUM_DEVICES, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  uv run python -m torch.distributed.run --standalone --nproc_per_node "$NUM_DEVICES" scripts/train_tal.py "${ARGS[@]}"
else
  uv run python scripts/train_tal.py "${ARGS[@]}"
fi
