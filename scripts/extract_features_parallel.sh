#!/bin/bash
# VideoMAE v2 特徴抽出 — 複数GPU並列稼働スクリプト (手動投下)
#
# 目的: 63k動画を複数GPUで並列処理し、時間短縮
#
# 実行方法:
#   # 方法1: デフォルト (GPU 0,1,2,3)
#   bash scripts/extract_features_parallel.sh
#
#   # 方法2: GPU番号を指定 (例: GPU 4,5,6,7)
#   GPU="4,5,6,7" bash scripts/extract_features_parallel.sh
#
#   # 方法3: 環境変数で指定
#   export GPU="0,1,2,3"
#   bash scripts/extract_features_parallel.sh
#
#   # 方法4: MAX_STEPS制限 (デバッグ用、最初の N個の動画のみ処理)
#   NUM_SHARDS=2 MAX_LIMIT=100 bash scripts/extract_features_parallel.sh
#
# 設定可能な環境変数:
#   GPU              : 使用するGPU番号 (カンマ区切り, default: "0")
#   NUM_SHARDS       : 並列プロセス数 (default: 4)
#   BATCH_SIZE       : バッチサイズ (default: 16, A100なら安定)
#   NUM_WORKERS      : ビデオ読込ワーカー (default: 1)
#   MAX_LIMIT        : デバッグ用; 最初のN個ファイルのみ処理 (default: 未指定=全て)
#   OUT_DIR          : 出力ディレクトリ (default: data/features/30s_mae_b_16_2)
#   CONFIG           : 抽出設定ファイル (default: codes/VideoMAEv2/configs/extract_vit_b.yaml)
#   OVERWRITE        : 既存出力を上書き (default: false, 指定なしで skip)
#
# 内部:
#   PYTHONPATH       : codes を追加
#   CUDA_VISIBLE_DEVICES : GPU_IDS から自動設定

set -euo pipefail

# ============================================================================
# 設定
# ============================================================================

# 環境変数から GPU 設定を取得
GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_LIMIT="${MAX_LIMIT:-}"
OUT_DIR="${OUT_DIR:-data/features/30s_mae_b_16_2}"
CONFIG="${CONFIG:-codes/VideoMAEv2/configs/extract_vit_b.yaml}"
OVERWRITE="${OVERWRITE:-}"

# 共通環境変数（実行時必須）
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/work/mt/.uv-cache}"
export HF_HOME="${HF_HOME:-/lustre/work/mt/.cache/huggingface}"
export ANNOT_ROOT_DIR="${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}"
export VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks}"
export PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}"

# ============================================================================
# バリデーション & 準備
# ============================================================================

cd "$(dirname "$0")/.."  # プロジェクトルートへ

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -d "$ANNOT_ROOT_DIR" ]]; then
  echo "[ERROR] annotation root not found: $ANNOT_ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$VIDEO_DATA_DIR" ]]; then
  echo "[ERROR] video root not found: $VIDEO_DATA_DIR" >&2
  exit 1
fi

# GPU IDs をトリミング（スペース除去）
GPU_IDS=$(echo "$GPU" | tr -d ' ')
mapfile -t GPU_ARRAY < <(echo "$GPU_IDS" | tr ',' '\n')
NUM_GPUS=${#GPU_ARRAY[@]}

if (( NUM_GPUS < 1 )); then
  echo "[ERROR] GPU list is empty: $GPU" >&2
  exit 1
fi

if (( NUM_SHARDS > NUM_GPUS )); then
  echo "[WARN] NUM_SHARDS ($NUM_SHARDS) > available GPUs ($NUM_GPUS)" >&2
  echo "       Clamping NUM_SHARDS to $NUM_GPUS"
  NUM_SHARDS=$NUM_GPUS
fi

mkdir -p "$OUT_DIR"

# ============================================================================
# ログ・ヘッダー
# ============================================================================

echo "======================================================================="
echo "VideoMAE v2 特徴抽出 — 複数GPU並列処理"
echo "======================================================================="
echo "GPU (CUDA_VISIBLE_DEVICES): $GPU_IDS"
echo "  Available GPUs: ${GPU_ARRAY[*]}"
echo "  Num Shards: $NUM_SHARDS"
echo ""
echo "Config: $CONFIG"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "Output Dir: $OUT_DIR"
echo ""
echo "Annotation Root: $ANNOT_ROOT_DIR"
echo "Video Root: $VIDEO_DATA_DIR"
if [[ -n "$MAX_LIMIT" ]]; then
  echo "Max Limit: $MAX_LIMIT (debug mode)"
fi
if [[ -n "$OVERWRITE" ]]; then
  echo "Overwrite: enabled"
fi
echo "======================================================================="
echo ""

# ============================================================================
# 並列実行 — 各シャードをバックグラウンドで投下
# ============================================================================

PIDS=()

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  # GPU 割り当て（ラウンドロビン）
  GPU_IDX=$((SHARD_ID % NUM_GPUS))
  GPU_DEV="${GPU_ARRAY[$GPU_IDX]}"

  echo "[shard $SHARD_ID/$NUM_SHARDS] Launching on GPU $GPU_DEV..."

  # CLI パラメータ構築
  ARGS=(
    --config "$CONFIG"
    --annot-dir "$ANNOT_ROOT_DIR"
    --video-dir "$VIDEO_DATA_DIR"
    --out-dir "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --shard-id "$SHARD_ID"
    --num-shards "$NUM_SHARDS"
  )

  if [[ -n "$MAX_LIMIT" ]]; then
    ARGS+=(--limit "$MAX_LIMIT")
  fi

  if [[ -n "$OVERWRITE" ]]; then
    ARGS+=(--overwrite)
  fi

  # バックグラウンドで投下
  (
    export CUDA_VISIBLE_DEVICES="$GPU_DEV"
    export PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}"
    uv run --no-sync python -m VideoMAEv2.extract_features "${ARGS[@]}"
  ) &

  PIDS+=($!)
  sleep 1  # Stagger launches slightly to avoid contention
done

echo ""
echo "[main] Launched $NUM_SHARDS shards with PIDs: ${PIDS[*]}"
echo "[main] Waiting for all shards to complete..."
echo ""

# ============================================================================
# 全プロセスの終了を待機
# ============================================================================

FAILED=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  if wait "$PID"; then
    echo "[shard $i/$NUM_SHARDS] ✓ completed (PID $PID)"
  else
    EXIT_CODE=$?
    echo "[shard $i/$NUM_SHARDS] ✗ failed (PID $PID, exit code $EXIT_CODE)" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if (( FAILED == 0 )); then
  echo "======================================================================="
  echo "✓ All shards completed successfully"
  echo "======================================================================="
  echo ""
  echo "Output at: $OUT_DIR"
  echo "Index: $OUT_DIR/index.json"
  exit 0
else
  echo "======================================================================="
  echo "✗ $FAILED shard(s) failed"
  echo "======================================================================="
  exit 1
fi
