#! /bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

DEVICES="${DEVICES:-1 3}"
read -r -a DEVICE_ARR <<< "$DEVICES"

ARGS=(
  --config codes/ActionFormer/configs/25k.yaml
  --output-folder outputs/ActionFormer_train/25k_10epochs_16batches
  --devices "${DEVICE_ARR[@]}"
  --tag 25k_10epochs_16batches
  --unit epoch
  --max-epochs 10
  --save-every 5
  --val-every 1
  --batch-size 16
  --print-freq 10
  --val-batch-size 1
  --split-list-dir /lustre/work/mt/okamura/tal-annotation-labeling/data/splits/25k
  --ddp-timeout-hours 0
)

NUM_DEVICES="${#DEVICE_ARR[@]}"
if (( NUM_DEVICES > 1 )); then
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${DEVICE_ARR[*]}")"
  export CUDA_VISIBLE_DEVICES
  uv run python -m torch.distributed.run --standalone --nproc_per_node "$NUM_DEVICES" scripts/train_tal.py "${ARGS[@]}"
else
  uv run python scripts/train_tal.py "${ARGS[@]}"
fi
