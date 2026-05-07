#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J tal_infer_test_bin
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

# --- Required environment variables ---
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

echo "===== [checkpoint inference test] start ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo

echo "===== [uv sync] ====="
uv sync
echo

CHECKPOINT_PATH="/lustre/work/mt/okamura/tal-annotation-labeling/outputs/ActionFormer_train/all_binary_1gpu/all_binary_all_binary_1gpu/step_00097278.pth.tar"
CONFIG_PATH="codes/ActionFormer/configs/tal_motion_vit_b.yaml"
FEAT_DIR="data/features/ALL"
OUTPUT_DIR="outputs/infer_test/all_binary_step_00097278_seed42"
NUM_VIDEOS=10
SEED=42

echo "===== [run inference test] ====="
echo "checkpoint: ${CHECKPOINT_PATH}"
echo "config: ${CONFIG_PATH}"
echo "feat_dir: ${FEAT_DIR}"
echo "output: ${OUTPUT_DIR}"
echo "num_videos: ${NUM_VIDEOS}, seed: ${SEED}"
echo

uv run python scripts/run_checkpoint_infer_test.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-videos "${NUM_VIDEOS}" \
  --seed "${SEED}" \
  --feat-dir "${FEAT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device 0

echo
echo "===== [checkpoint inference test] done ====="
date
