#! /bin/bash

uv run python scripts/train_tal.py \
  --config codes/ActionFormer/configs/25k.yaml \
  --output-folder outputs/ActionFormer_train/25k \
  --devices 0 \
  --tag 25k \
  --unit epoch \
  --max-epochs 3 \
  --save-every 1 \
  --val-every 1 \
  --batch-size 64 \
  --print-freq 10 \
  --val-batch-size 64 \
  --split-list-dir /lustre/work/mt/okamura/tal-annotation-labeling/data/splits/25k
