#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J tal_pipeline_vlm
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

# VideoMAE -> ActionFormer -> Qwen VL エンドツーエンドパイプライン
# 事前に batch_logs を作成しておくこと（docs/slurm.md 参照）。
#
# 使い方:
#   1) 下の「編集ブロック」を書き換える、または
#   2) 投入前に export して上書き:
#        export VIDEO_PATH=/raid/.../clip.mp4
#        export VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
#        sbatch scripts/sbatch/sbatch_pipeline_tal_vlm_1gpu.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

# --- Required environment variables (docs/slurm.md) ---
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks
export HF_TOKEN="${HF_TOKEN:-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}"
export VLM_MODEL=google/gemma-4-E4B-it

# ========== 編集ブロック（または export で上書き）==========
# 入力動画（必須）
VIDEO_PATH="${VIDEO_PATH:-}"
# HuggingFace モデル ID またはローカルパス（必須）
VLM_MODEL="google/gemma-4-E4B-it"
# ActionFormer チェックポイント
AF_CKPT="${AF_CKPT:-models/ActionFormer/checkpoint.pth.tar}"
AF_CONFIG="${AF_CONFIG:-codes/ActionFormer/configs/tal_motion_vit_b.yaml}"
MAE_CONFIG="${MAE_CONFIG:-codes/VideoMAEv2/configs/extract_vit_b.yaml}"
SCORE_THRESH="${SCORE_THRESH:-0.3}"
NUM_VLM_FRAMES="${NUM_VLM_FRAMES:-8}"
# 出力先（未設定時はジョブ ID 付き）
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pipeline_tal_vlm/job_${SLURM_JOB_ID:-manual}}"
# GPU 番号（割り当ては 1 枚想定。CUDA は 0 から見える）
DEVICE_MAE="${DEVICE_MAE:-0}"
DEVICE_AF="${DEVICE_AF:-0}"
DEVICE_VLM="${DEVICE_VLM:-0}"
PREP_WORKERS="${PREP_WORKERS:-8}"
# VLM バッチサイズ（0以下で「全区間を一度に推論」）
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-4}"
# ========================================================

echo "===== [pipeline_tal_vlm] start ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "ERROR: VIDEO_PATH が空です。スクリプト先頭の編集ブロックを設定するか、" >&2
  echo "  export VIDEO_PATH=/path/to/video.mp4" >&2
  echo "  してから sbatch してください。" >&2
  exit 1
fi

if [[ "${VLM_MODEL,,}" == *"gemma"* ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: Gemma は gated repo のため HF_TOKEN が必要です。" >&2
  echo "  export HF_TOKEN=hf_xxx を設定してから sbatch してください。" >&2
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "ERROR: 動画が見つかりません: ${VIDEO_PATH}" >&2
  exit 1
fi

if [[ ! -f "${AF_CKPT}" ]]; then
  echo "ERROR: ActionFormer チェックポイントが見つかりません: ${AF_CKPT}" >&2
  exit 1
fi

echo "===== [uv sync] ====="
uv sync
echo

echo "===== [run pipeline_tal_vlm] ====="
echo "video: ${VIDEO_PATH}"
echo "vlm_model: ${VLM_MODEL}"
echo "af_ckpt: ${AF_CKPT}"
echo "output_dir: ${OUTPUT_DIR}"
echo "score_thresh: ${SCORE_THRESH}, num_vlm_frames: ${NUM_VLM_FRAMES}"
echo "prep_workers: ${PREP_WORKERS}, vlm_batch_size: ${VLM_BATCH_SIZE}"
echo

mkdir -p "${OUTPUT_DIR}"

uv run python scripts/pipeline_tal_vlm.py \
  --video "${VIDEO_PATH}" \
  --af-config "${AF_CONFIG}" \
  --af-ckpt "${AF_CKPT}" \
  --mae-config "${MAE_CONFIG}" \
  --vlm-model "${VLM_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --score-thresh "${SCORE_THRESH}" \
  --num-vlm-frames "${NUM_VLM_FRAMES}" \
  --vlm-batch-size "${VLM_BATCH_SIZE}" \
  --device-mae "${DEVICE_MAE}" \
  --device-af "${DEVICE_AF}" \
  --device-vlm "${DEVICE_VLM}" \
  --prep-workers "${PREP_WORKERS}"

echo
echo "===== [pipeline_tal_vlm] done ====="
date
