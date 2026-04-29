#!/bin/bash
# Prepare ActionFormer training prerequisites for TAL.
#
# Steps:
# 1) uv sync
# 2) build action_type vocab
# 3) validate split/annotation/feature/vocab consistency
#
# Usage:
#   bash scripts/prepare_tal_training.sh
#   ANNOT_DIR=/path/to/annot FEAT_DIR=/path/to/features bash scripts/prepare_tal_training.sh

set -euo pipefail

cd "$(dirname "$0")/.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/work/mt/.uv-cache}"
export HF_HOME="${HF_HOME:-/lustre/work/mt/.cache/huggingface}"
export ANNOT_DIR="${ANNOT_DIR:-${ANNOT_ROOT_DIR:-/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2}}"
export FEAT_DIR="${FEAT_DIR:-data/features/30s_mae_b_16_2}"
export SPLIT_LIST_DIR="${SPLIT_LIST_DIR:-data/splits/v1}"
export VOCAB_JSON="${VOCAB_JSON:-data/aux_stats/v1/action_type_vocab.json}"
export MIN_COUNT="${MIN_COUNT:-1}"
export TOP_N="${TOP_N:-0}"

echo "[prepare] uv sync"
uv sync

echo "[prepare] build action_type vocab -> $VOCAB_JSON"
uv run python scripts/build_action_type_vocab.py \
  --annot-dir "$ANNOT_DIR" \
  --output-json "$VOCAB_JSON" \
  --min-count "$MIN_COUNT" \
  --top-n "$TOP_N"

echo "[prepare] validate training inputs"
uv run python scripts/check_tal_training_inputs.py \
  --split-list-dir "$SPLIT_LIST_DIR" \
  --annot-dir "$ANNOT_DIR" \
  --feat-dir "$FEAT_DIR" \
  --vocab-json "$VOCAB_JSON"

echo "[prepare] done"
