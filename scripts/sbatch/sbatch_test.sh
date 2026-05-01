#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J sbatch_env_test
#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH -c 16
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

echo "===== [sbatch test] start ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$(pwd)"
echo "USER=${USER:-}"
echo

echo "===== [paths] ====="
echo "which uv: $(command -v uv || true)"
echo "which python: $(command -v python || true)"
echo "which nvidia-smi: $(command -v nvidia-smi || true)"
echo

echo "===== [env vars] ====="
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-}"
echo "HF_HOME=${HF_HOME:-}"
echo "ANNOT_ROOT_DIR=${ANNOT_ROOT_DIR:-}"
echo "VIDEO_DATA_DIR=${VIDEO_DATA_DIR:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo

echo "===== [filesystem checks] ====="
if [[ -f pyproject.toml ]]; then
  echo "pyproject.toml: found"
else
  echo "pyproject.toml: NOT found"
fi
if [[ -d "$ANNOT_ROOT_DIR" ]]; then
  echo "ANNOT_ROOT_DIR: found"
else
  echo "ANNOT_ROOT_DIR: NOT found"
fi
if [[ -d "$VIDEO_DATA_DIR" ]]; then
  echo "VIDEO_DATA_DIR: found"
else
  echo "VIDEO_DATA_DIR: NOT found"
fi
echo

echo "===== [gpu checks] ====="
nvidia-smi || true
echo

echo "===== [uv sync] ====="
uv sync
echo

echo "===== [python/torch checks] ====="
uv run python - <<'PY'
import os
import torch

print("python ok")
print("torch.__version__ =", torch.__version__)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
PY
echo

echo "===== [sbatch test] done ====="
date
