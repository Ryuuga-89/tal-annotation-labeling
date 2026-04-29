#! /bin/bash

uv run python scripts/train_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --output-folder outputs/tal_motion_experiments \
  --devices 0 \
  --tag experiment_v1 \
  --max-epochs 1 \
  --print-freq 10 \
  --split-list-dir /lustre/work/mt/okamura/tal-annotation-labeling/data/features/30s_mae_b_16_2/splits \
  --val-every 100 