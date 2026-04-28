#!/bin/bash
# VideoMAE v2 特徴抽出 — 単一GPU テストスクリプト
#
# 目的: デバッグ・検証用に単一GPUで実行
#
# 実行方法:
#   # 方法1: GPU 0 (デフォルト)
#   bash scripts/extract_features_single_gpu.sh
#
#   # 方法2: GPU 5 を指定
#   GPU=5 bash scripts/extract_features_single_gpu.sh
#
#   # 方法3: 最初の10ファイルのみ処理
#   GPU=0 MAX_LIMIT=10 bash scripts/extract_features_single_gpu.sh
#
#   # 方法4: 既存出力を上書き
#   GPU=0 OVERWRITE=1 bash scripts/extract_features_single_gpu.sh
#
# 環境変数:
#   GPU              : GPU番号 (default: 0)
#   MAX_LIMIT        : 処理上限ファイル数 (default: 未指定 = 全て)
#   BATCH_SIZE       : バッチサイズ (default: 16)
#   NUM_WORKERS      : ビデオ読込ワーカー (default: 4)
#   OVERWRITE        : 既存を上書き (default: 0)
#   OUT_DIR          : 出力先 (default: data/features/30s_mae_b_16_2)
#   CONFIG           : 設定ファイル (default: codes/VideoMAEv2/configs/extract_vit_b.yaml)

set -euo pipefail

# ============================================================================
# 設定
# ============================================================================

GPU="${GPU:-0}"
MAX_LIMIT="${MAX_LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-0}"
OUT_DIR="${OUT_DIR:-data/features/30s_mae_b_16_2}"
CONFIG="${CONFIG:-codes/VideoMAEv2/configs/extract_vit_b.yaml}"

# 必須環境変数
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/work/mt/.uv-cache}"
export HF_HOME="${HF_HOME:-/lustre/work/mt/.cache/huggingface}"
export ANNOT_ROOT_DIR="${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}"
export VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="codes${PYTHONPATH:+:${PYTHONPATH}}"

# ============================================================================
# バリデーション & 準備
# ============================================================================

cd "$(dirname "$0")/.."  # プロジェクトルート

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

mkdir -p "$OUT_DIR"

# ============================================================================
# ログ・ヘッダー
# ============================================================================

echo "======================================================================="
echo "VideoMAE v2 特徴抽出 — 単一GPU (テスト・デバッグ)"
echo "======================================================================="
echo "GPU: $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""
echo "Config: $CONFIG"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "Output Dir: $OUT_DIR"
echo ""
echo "Annotation Root: $ANNOT_ROOT_DIR"
echo "Video Root: $VIDEO_DATA_DIR"
if [[ -n "$MAX_LIMIT" ]]; then
  echo "Max Limit: $MAX_LIMIT (debug mode - first N files only)"
fi
if (( OVERWRITE == 1 )); then
  echo "Overwrite: enabled"
fi
echo "======================================================================="
echo ""

# ============================================================================
# 実行
# ============================================================================

ARGS=(
  --config "$CONFIG"
  --annot-dir "$ANNOT_ROOT_DIR"
  --video-dir "$VIDEO_DATA_DIR"
  --out-dir "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$MAX_LIMIT" ]]; then
  ARGS+=(--limit "$MAX_LIMIT")
fi

if (( OVERWRITE == 1 )); then
  ARGS+=(--overwrite)
fi

echo "[main] Starting extraction..."
echo ""

uv run --no-sync python -m VideoMAEv2.extract_features "${ARGS[@]}"

echo ""
echo "======================================================================="
echo "✓ Extraction completed"
echo "======================================================================="
echo ""
echo "Output directory: $OUT_DIR"
echo "Index file: $OUT_DIR/index.json"
echo ""
echo "To inspect results:"
echo "  cat $OUT_DIR/index.json | jq '.videos | length'"
echo "  ls -lh $OUT_DIR/*.npy | head -5"
