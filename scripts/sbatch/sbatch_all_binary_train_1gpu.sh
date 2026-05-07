#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J tal_all_bin_1gpu
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

# ActionFormer 二値（class-agnostic）学習 — 1 GPU。
# 前提: data/features/ALL に .npy（＋任意の sidecar .json）があり、
#       data/splits/ALL に train.txt / val.txt（train_val_test_split.py で生成）。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

uv sync

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SPLIT_DIR="${SPLIT_DIR:-${REPO}/data/splits/ALL}"

if [[ ! -f "${SPLIT_DIR}/train.txt" ]] || [[ ! -f "${SPLIT_DIR}/val.txt" ]]; then
  echo "[error] missing ${SPLIT_DIR}/train.txt or val.txt — run train_val_test_split.py first." >&2
  exit 1
fi

uv run python scripts/train_tal.py \
  --config codes/ActionFormer/configs/all_binary.yaml \
  --output-folder outputs/ActionFormer_train/all_binary_1gpu \
  --devices 0 \
  --tag all_binary_1gpu \
  --split-list-dir "${SPLIT_DIR}" \
  --batch-size 8 \
  --val-batch-size 1 \
  --max-workers 8 \
  --val-loader-workers 2 \
  --ddp-timeout-hours 0
