#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J videomae_feat_2gpu
#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH -c 16
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

# VideoMAE V2 特徴量抽出（2× A100 80GB 想定: shard 0/1 を GPU ごとに並列実行）。
#
# 使い方（リポジトリルートで）:
#   mkdir -p batch_logs
#   sbatch scripts/sbatch/sbatch_videomae_extract_features_2gpu.sh
#
# 任意の環境変数:
#   FEAT_OUT_DIR   出力先 (default: data/features/25k)
#   ANNOT_LIST     省略時は ANNOT_ROOT_DIR 直下の全 *.json（約6万件）を対象。
#                  サブセットにしたいときだけパスを指定（1行1エントリ、basename のみ使用）。
#   EXTRACT_CONFIG extract 用 YAML (default: codes/VideoMAEv2/configs/extract_vit_b.yaml)
#   SHARDS         シャード数 (=GPU 並列数, default: 2)
#   EXTRACT_*  既定は scripts/run_extract_features_2gpu.sh と同じ（direct 経路: num_workers=0）。
#              Slurm の CPU 数は docs/slurm.md / sbatch_test.sh に合わせ -c 16。

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

# --- 環境変数の設定 ---
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

FEAT_OUT_DIR="${FEAT_OUT_DIR:-data/features/25k}"
EXTRACT_CONFIG="${EXTRACT_CONFIG:-codes/VideoMAEv2/configs/extract_vit_b.yaml}"
SHARDS="${SHARDS:-2}"
# run_extract_features_2gpu.sh と同一（num_workers=0 → extract_features 内の direct batching）
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-48}"
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-0}"
EXTRACT_DECODE_THREADS="${EXTRACT_DECODE_THREADS:-4}"
EXTRACT_MAX_BATCH_BATCH_DECODE="${EXTRACT_MAX_BATCH_BATCH_DECODE:-32}"

echo "===== [videomae extract] start ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "FEAT_OUT_DIR=${FEAT_OUT_DIR}"
echo "EXTRACT_CONFIG=${EXTRACT_CONFIG}"
echo "SHARDS=${SHARDS}"
echo "EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE} EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS}"
echo "EXTRACT_DECODE_THREADS=${EXTRACT_DECODE_THREADS} EXTRACT_MAX_BATCH_BATCH_DECODE=${EXTRACT_MAX_BATCH_BATCH_DECODE}"
echo

nvidia-smi || true
echo

# --- 仮想環境の構築・同期 ---
uv sync

mkdir -p "${FEAT_OUT_DIR}"

ANNOT_LIST="${ANNOT_LIST:-}"
if [[ -n "${ANNOT_LIST}" ]]; then
  echo "[info] ANNOT_LIST=${ANNOT_LIST} (subset mode)"
else
  echo "[info] ANNOT_LIST unset -> all annotation JSON under ${ANNOT_ROOT_DIR}"
  echo "[info] existing ${FEAT_OUT_DIR}/*.npy+.json are skipped by extract_features prefilter"
fi

if [[ ! -f "${EXTRACT_CONFIG}" ]]; then
  echo "[error] config not found: ${EXTRACT_CONFIG}" >&2
  exit 1
fi

# 2 プロセスで同じログディレクトリに追記するため、shard ごとにログファイル名が分かれる extract_features 側の挙動を利用

COMMON_EXTRACT_ARGS=(
  --config "${EXTRACT_CONFIG}"
  --annot-dir "${ANNOT_ROOT_DIR}"
  --video-dir "${VIDEO_DATA_DIR}"
  --out-dir "${FEAT_OUT_DIR}"
  --model-name vit_base_patch16_224
  --target-fps 10
  --window-size 16
  --stride 2
  --batch-size "${EXTRACT_BATCH_SIZE}"
  --num-workers "${EXTRACT_NUM_WORKERS}"
  --decode-mode auto
  --auto-batch-threshold-frames 320
  --auto-batch-size
  --max-batch-size-batch-decode "${EXTRACT_MAX_BATCH_BATCH_DECODE}"
  --decode-threads "${EXTRACT_DECODE_THREADS}"
  --progress-log-interval-videos 20
  --cleanup-interval-videos 200
)
if [[ -n "${ANNOT_LIST}" ]]; then
  COMMON_EXTRACT_ARGS+=(--annot-list "${ANNOT_LIST}")
fi

PIDS=()
cleanup_children() {
  echo "[signal] stopping child extract processes..."
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}

trap cleanup_children INT TERM

if ! [[ "${SHARDS}" =~ ^[0-9]+$ ]] || (( SHARDS < 1 )); then
  echo "[error] SHARDS must be a positive integer (got ${SHARDS})" >&2
  exit 1
fi

for ((sid = 0; sid < SHARDS; sid++)); do
  echo "[launch] shard ${sid}/${SHARDS} on cuda:${sid}"
  uv run --no-sync python codes/VideoMAEv2/extract_features.py \
    "${COMMON_EXTRACT_ARGS[@]}" \
    --gpu-id "${sid}" \
    --num-shards "${SHARDS}" \
    --shard-id "${sid}" &
  PIDS+=("$!")
done

EC_ALL=0
set +e
for pid in "${PIDS[@]}"; do
  wait "${pid}"
  ec=$?
  if (( ec != 0 )); then
    echo "[error] extract process pid=${pid} exited with ${ec}" >&2
    EC_ALL="${ec}"
  fi
done
set -e

if (( EC_ALL != 0 )); then
  exit "${EC_ALL}"
fi

echo "===== [videomae extract] done ====="
date
