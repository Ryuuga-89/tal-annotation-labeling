#!/bin/bash
# VideoMAE v2 特徴抽出 — SLURM 並列投下スクリプト
#
# 目的: GPU ノード上で複数シャードを sbatch で並列投下
#
# 実行方法:
#   # 方法1: デフォルト (4 GPU × 4 shards = 4 ジョブ)
#   bash scripts/submit_extract_sharded.sh
#
#   # 方法2: GPU数指定
#   NUM_SHARDS=8 bash scripts/submit_extract_sharded.sh
#
#   # 方法3: カスタムGPU指定と共に
#   GPU="0,1,2,3,4,5,6,7" NUM_SHARDS=8 bash scripts/submit_extract_sharded.sh
#
#   # 方法4: SLURM設定カスタマイズ
#   SLURM_TIME="24:00:00" SLURM_MEM="40G" bash scripts/submit_extract_sharded.sh
#
# 環境変数:
#   NUM_SHARDS       : 投下ジョブ数 (= 並列GPU数, default: 4)
#   GPU_PER_JOB      : 各ジョブに割り当てるGPU数 (default: 1)
#   SLURM_TIME       : ジョブタイムリミット (default: "12:00:00")
#   SLURM_MEM        : メモリ容量 (default: "30G")
#   SLURM_PARTITION  : パーティション指定 (default: "gpu")
#   OUT_DIR          : 出力先 (default: data/features/30s_mae_b_16_2)
#   CONFIG           : 設定ファイル (default: codes/VideoMAEv2/configs/extract_vit_b.yaml)
#   BATCH_SIZE       : バッチサイズ (default: 16)
#   NUM_WORKERS      : 並列ワーカー (default: 1)
#   OVERWRITE        : 上書きフラグ (指定で有効化)
#   MAX_LIMIT        : デバッグ用 limit (指定で有効化)
#   LOG_DIR          : sbatch ログ出力先 (default: logs/extract_logs)

set -euo pipefail

# ============================================================================
# 設定
# ============================================================================

NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_PER_JOB="${GPU_PER_JOB:-1}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_MEM="${SLURM_MEM:-30G}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

OUT_DIR="${OUT_DIR:-data/features/30s_mae_b_16_2}"
CONFIG="${CONFIG:-codes/VideoMAEv2/configs/extract_vit_b.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-logs/extract_logs}"

OVERWRITE="${OVERWRITE:-}"
MAX_LIMIT="${MAX_LIMIT:-}"

# 環境変数（実行時必須）
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/work/mt/.uv-cache}"
export HF_HOME="${HF_HOME:-/lustre/work/mt/.cache/huggingface}"
export ANNOT_ROOT_DIR="${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}"
export VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks}"

# ============================================================================
# バリデーション & 準備
# ============================================================================

cd "$(dirname "$0")/.."  # プロジェクトルート

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ============================================================================
# ログ・ヘッダー
# ============================================================================

echo "======================================================================="
echo "VideoMAE v2 特徴抽出 — SLURM並列投下"
echo "======================================================================="
echo "Num Shards (= Jobs): $NUM_SHARDS"
echo "GPU per Job: $GPU_PER_JOB"
echo ""
echo "SLURM Settings:"
echo "  Partition: $SLURM_PARTITION"
echo "  Time Limit: $SLURM_TIME"
echo "  Memory: $SLURM_MEM"
echo ""
echo "Extract Settings:"
echo "  Config: $CONFIG"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"
echo "  Output: $OUT_DIR"
echo ""
echo "Log Directory: $LOG_DIR"
if [[ -n "$MAX_LIMIT" ]]; then
  echo "Max Limit (debug): $MAX_LIMIT"
fi
if [[ -n "$OVERWRITE" ]]; then
  echo "Overwrite: enabled"
fi
echo "======================================================================="
echo ""

# ============================================================================
# 各シャード用ジョブを投下
# ============================================================================

SUBMITTED_JOBS=()

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  JOB_NAME="extract_shard${SHARD_ID}"
  LOG_PREFIX="$LOG_DIR/${JOB_NAME}"

  echo "[shard $SHARD_ID] Submitting job: $JOB_NAME..."

  # CLI パラメータ構築
  EXTRACT_ARGS=(
    "--config" "$CONFIG"
    "--annot-dir" "$ANNOT_ROOT_DIR"
    "--video-dir" "$VIDEO_DATA_DIR"
    "--out-dir" "$OUT_DIR"
    "--batch-size" "$BATCH_SIZE"
    "--num-workers" "$NUM_WORKERS"
    "--shard-id" "$SHARD_ID"
    "--num-shards" "$NUM_SHARDS"
  )

  if [[ -n "$MAX_LIMIT" ]]; then
    EXTRACT_ARGS+=("--limit" "$MAX_LIMIT")
  fi

  if [[ -n "$OVERWRITE" ]]; then
    EXTRACT_ARGS+=("--overwrite")
  fi

  # sbatch 投下
  JOB_ID=$(sbatch \
    --job-name="$JOB_NAME" \
    --gpus="$GPU_PER_JOB" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --partition="$SLURM_PARTITION" \
    --output="${LOG_PREFIX}.out" \
    --error="${LOG_PREFIX}.err" \
    << SLURM_SCRIPT
#!/bin/bash
export UV_CACHE_DIR="$UV_CACHE_DIR"
export HF_HOME="$HF_HOME"
export ANNOT_ROOT_DIR="$ANNOT_ROOT_DIR"
export VIDEO_DATA_DIR="$VIDEO_DATA_DIR"

cd "$(pwd)"

echo "[shard $SHARD_ID] Starting on GPU(s): \$CUDA_VISIBLE_DEVICES"
echo "[shard $SHARD_ID] Job ID: \$SLURM_JOB_ID"

uv run --no-sync python -m VideoMAEv2.extract_features ${EXTRACT_ARGS[@]}

echo "[shard $SHARD_ID] Completed at \$(date)"
SLURM_SCRIPT
  )

  # ジョブIDを抽出 (Submitted batch job 12345678)
  JOB_ID=$(echo "$JOB_ID" | grep -oE "[0-9]+" | head -1)

  SUBMITTED_JOBS+=("$JOB_ID")
  echo "  → Job ID: $JOB_ID"
done

echo ""
echo "======================================================================="
echo "✓ Submitted $NUM_SHARDS jobs"
echo "======================================================================="
echo ""
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor progress:"
echo "  squeue -l -j ${SUBMITTED_JOBS[0]}"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/extract_shard*.out"
echo ""
echo "When complete, check:"
echo "  ls -lh $OUT_DIR/ | head -20"
echo "  cat $OUT_DIR/index.json | jq '.videos | length'"
