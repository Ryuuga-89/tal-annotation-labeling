# W&B Run Templates

W&B naming rule:

- `{train|exp|test}_{description}_{timestamp}`

Project / Entity:

- project: `tal-annotation-labeling`
- entity: `models-institute-of-science-tokyo`

## Train

```bash
uv run python scripts/train_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --devices 0 \
  --wandb-run-type train \
  --wandb-run-desc combined164_stepval \
  --wandb-log-ckpt
```

## Inference

```bash
uv run python scripts/infer_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --ckpt outputs/tal_motion_experiments/<run>/step_00001000.pth.tar \
  --feature-list-txt data/splits/infer_targets.txt \
  --feat-dir data/features/30s_mae_b_16_2 \
  --output-json outputs/infer/detections.json \
  --wandb-run-type test \
  --wandb-run-desc infer_combined \
  --wandb-log-output
```

## Evaluation

```bash
uv run python scripts/eval_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --ckpt outputs/tal_motion_experiments/<run>/step_00001000.pth.tar \
  --output-dir outputs/eval/step_00001000 \
  --devices 0 \
  --wandb-run-type test \
  --wandb-run-desc eval_combined \
  --wandb-log-output
```
