#!/usr/bin/env bash
set -euo pipefail

# Two-GPU sharded launcher for VideoMAEv2 feature extraction.
# Usage:
#   bash scripts/run_extract_features_2gpu.sh \
#     --annot-dir "$ANNOT_ROOT_DIR" \
#     --video-dir "$VIDEO_DATA_DIR" \
#     --out-dir data/features/30s_mae_b_16_2 \
#     --ckpt-path /lustre/work/mt/okamura/tal-annotation-labeling/models/vit_b/vit_b_k710_dl_from_giant.pth

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

GPU0=0
GPU1=1
PID0=""
PID1=""

ARGS=("$@")

COMMON_ARGS=(
  --model-name vit_base_patch16_224
  --target-fps 10
  --window-size 16
  --stride 2
  --batch-size 48
  --decode-mode auto
  --auto-batch-threshold-frames 320
  --auto-batch-size
  --max-batch-size-batch-decode 32
  --decode-threads 4
  --progress-log-interval-videos 20
  --cleanup-interval-videos 200
)

cleanup_children() {
  echo "[signal] stopping child extract processes..."
  if [[ -n "${PID0}" ]] && kill -0 "${PID0}" 2>/dev/null; then
    kill "${PID0}" 2>/dev/null || true
  fi
  if [[ -n "${PID1}" ]] && kill -0 "${PID1}" 2>/dev/null; then
    kill "${PID1}" 2>/dev/null || true
  fi
  wait || true
}

trap cleanup_children INT TERM

echo "[launch] shard 0 on GPU ${GPU0}"
uv run --no-sync python codes/VideoMAEv2/extract_features.py \
  "${COMMON_ARGS[@]}" \
  --gpu-id "${GPU0}" \
  --num-shards 2 \
  --shard-id 0 \
  "${ARGS[@]}" &
PID0=$!

echo "[launch] shard 1 on GPU ${GPU1}"
uv run --no-sync python codes/VideoMAEv2/extract_features.py \
  "${COMMON_ARGS[@]}" \
  --gpu-id "${GPU1}" \
  --num-shards 2 \
  --shard-id 1 \
  "${ARGS[@]}" &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[done] both shards completed"
