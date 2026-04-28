#!/usr/bin/env bash
# Usage:
#   bash codes/VideoMAEv2/scripts/extract.sh <vit_b|vit_g> [ckpt_path] [limit]
#
# Defaults:
#   vit_b ckpt: models/vit_b/vit_b_k710_dl_from_giant.pth
#   vit_g ckpt: models/vit_g/vit_g_hybrid_pt_1200e_k710_ft.pth
#
# GPU selection / parallelism
# ---------------------------
#   GPU=0                    single GPU (default)
#   GPU=0,1,2,3              4 parallel processes, one per GPU, video list sharded
#   GPU=2,5                  2 processes pinned to physical GPUs 2 and 5
#   GPU=cpu                  CPU-only (smoke testing)
#
# Each GPU process picks a strided shard of the (sorted) annotation list:
# process k of K runs `--shard-id k --num-shards K`. Outputs go to a single
# out_dir; per-shard summaries are written to index.shardKKofNN.json.
#
# I/O:
#   ANNOT_ROOT_DIR / VIDEO_DATA_DIR (env vars; see CLAUDE.md) override hard-coded paths.
#   OUT_DIR overrides the default output directory.
#
# Examples:
#   bash codes/VideoMAEv2/scripts/extract.sh vit_b
#   GPU=0,1,2,3 bash codes/VideoMAEv2/scripts/extract.sh vit_b
#   GPU=0 bash codes/VideoMAEv2/scripts/extract.sh vit_b "" 4              # limit to 4 videos
#   GPU=0,1 bash codes/VideoMAEv2/scripts/extract.sh vit_g /weights/g.pth
set -euo pipefail

variant="${1:-vit_b}"
ckpt="${2:-}"
limit="${3:-}"

case "${variant}" in
  vit_b)
    cfg="codes/VideoMAEv2/configs/extract_vit_b.yaml"
    default_out="data/features/30s_mae_b_16_2"
    default_ckpt="models/vit_b/vit_b_k710_dl_from_giant.pth"
    ;;
  vit_g)
    cfg="codes/VideoMAEv2/configs/extract_vit_g.yaml"
    default_out="data/features/30s_mae_g_16_2"
    default_ckpt="models/vit_g/vit_g_hybrid_pt_1200e_k710_ft.pth"
    ;;
  *)
    echo "unknown variant: ${variant}" >&2
    exit 1
    ;;
esac
out_dir="${OUT_DIR:-${default_out}}"

if [[ -z "${ckpt}" ]]; then
  ckpt="${default_ckpt}"
fi
if [[ ! -f "${ckpt}" ]]; then
  echo "checkpoint not found: ${ckpt}" >&2
  exit 1
fi

annot_dir="${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}"
video_dir="${VIDEO_DATA_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks}"

mkdir -p "${out_dir}"
mkdir -p "${out_dir}/logs"

extra_args=()
if [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
fi
if [[ -n "${LIST:-}" ]]; then
  extra_args+=(--annot-list "${LIST}")
fi

# ---------- GPU / world size ----------
GPU_LIST="${GPU:-0}"

if [[ "${GPU_LIST}" == "cpu" ]]; then
  device="cpu"
  IFS=',' read -r -a GPU_ARR <<< "0"   # placeholder, single process
  WORLD_SIZE=1
else
  device="cuda"
  IFS=',' read -r -a GPU_ARR <<< "${GPU_LIST}"
  WORLD_SIZE="${#GPU_ARR[@]}"
  command -v nvidia-smi >/dev/null && \
    nvidia-smi --query-gpu=index,name,memory.free,memory.used,utilization.gpu \
               --format=csv || true
fi
echo "[extract] GPU=${GPU_LIST} world_size=${WORLD_SIZE} device=${device}" >&2

run_one() {
  # $1 = physical GPU id (or "" for CPU), $2 = shard_id, $3 = num_shards
  local gpu_id="$1" shard_id="$2" num_shards="$3"
  local log="${out_dir}/logs/shard${shard_id}of${num_shards}.log"
  local extra_dev_args=()
  if [[ "${device}" == "cpu" ]]; then
    extra_dev_args+=(--device cpu --dtype float32)
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}" \
  uv run python -m VideoMAEv2.extract_features \
    --config "${cfg}" \
    --annot-dir "${annot_dir}" \
    --video-dir "${video_dir}" \
    --out-dir "${out_dir}" \
    --ckpt-path "${ckpt}" \
    --shard-id "${shard_id}" \
    --num-shards "${num_shards}" \
    "${extra_dev_args[@]}" \
    "${extra_args[@]}" >"${log}" 2>&1 &
  echo $!
}

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
  # Foreground, no logfile redirection (stdout flows to the user).
  if [[ "${device}" == "cpu" ]]; then
    PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run python -m VideoMAEv2.extract_features \
      --config "${cfg}" \
      --annot-dir "${annot_dir}" \
      --video-dir "${video_dir}" \
      --out-dir "${out_dir}" \
      --ckpt-path "${ckpt}" \
      --device cpu --dtype float32 \
      "${extra_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" \
    PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run python -m VideoMAEv2.extract_features \
      --config "${cfg}" \
      --annot-dir "${annot_dir}" \
      --video-dir "${video_dir}" \
      --out-dir "${out_dir}" \
      --ckpt-path "${ckpt}" \
      "${extra_args[@]}"
  fi
  echo "[extract] done. out_dir=${out_dir}"
  exit 0
fi

# Multi-GPU: launch one shard per GPU in parallel.
pids=()
trap 'echo "[extract] terminating shards..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM
echo "[extract] launching ${WORLD_SIZE} shards. logs: ${out_dir}/logs/" >&2
for ((i = 0; i < WORLD_SIZE; i++)); do
  gpu_id="${GPU_ARR[$i]}"
  pid="$(run_one "${gpu_id}" "${i}" "${WORLD_SIZE}")"
  pids+=("${pid}")
  echo "[extract] shard ${i}/${WORLD_SIZE} pid=${pid} gpu=${gpu_id} log=${out_dir}/logs/shard${i}of${WORLD_SIZE}.log" >&2
done

# Wait on each, capturing the first non-zero exit code without aborting waits.
fail=0
for ((i = 0; i < WORLD_SIZE; i++)); do
  if ! wait "${pids[$i]}"; then
    echo "[extract] shard ${i} failed (see ${out_dir}/logs/shard${i}of${WORLD_SIZE}.log)" >&2
    fail=1
  else
    echo "[extract] shard ${i} ok" >&2
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[extract] one or more shards failed; out_dir=${out_dir}" >&2
  exit 1
fi
echo "[extract] all shards done. out_dir=${out_dir}"
