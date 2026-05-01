#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J tal_25k_100e_32b
#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH -c 16
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- 環境変数の設定 ---
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

# --- 仮想環境の構築・同期 ---
uv sync

DEVICE_ARR=()
if [[ -n "${DEVICES:-}" ]]; then
  # Optional manual override, e.g. DEVICES="2 3".
  # Map selected physical GPUs to local indices (0..N-1) for train_tal.py.
  read -r -a SELECTED_DEV_ARR <<< "$DEVICES"
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SELECTED_DEV_ARR[*]}")"
  export CUDA_VISIBLE_DEVICES
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a SELECTED_DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  GPU_COUNT="${SLURM_GPUS_ON_NODE//[^0-9]/}"
  SELECTED_DEV_ARR=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    SELECTED_DEV_ARR+=("$i")
  done
else
  SELECTED_DEV_ARR=("0")
fi

NUM_SELECTED_DEVICES="${#SELECTED_DEV_ARR[@]}"
for ((i = 0; i < NUM_SELECTED_DEVICES; i++)); do
  DEVICE_ARR+=("$i")
done

if (( NUM_SELECTED_DEVICES == 0 )); then
  DEVICE_ARR=("0")
fi

ARGS=(
  --config codes/ActionFormer/configs/25k.yaml
  --output-folder outputs/ActionFormer_train/25k_100epochs_32batches
  --devices "${DEVICE_ARR[@]}"
  --tag 25k_100epochs_32batches
  --unit epoch
  --max-epochs 100
  --save-every 5
  --val-every 1
  --batch-size 32
  --print-freq 10
  --val-batch-size 1
  --split-list-dir /lustre/work/mt/okamura/tal-annotation-labeling/data/splits/25k
  --ddp-timeout-hours 0
)

NUM_DEVICES="${#DEVICE_ARR[@]}"
if (( NUM_DEVICES > 1 )); then
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SELECTED_DEV_ARR[*]}")"
  export CUDA_VISIBLE_DEVICES
  uv run python -m torch.distributed.run --standalone --nproc_per_node "$NUM_DEVICES" scripts/train_tal.py "${ARGS[@]}"
else
  uv run python scripts/train_tal.py "${ARGS[@]}"
fi
