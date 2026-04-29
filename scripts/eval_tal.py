#!/usr/bin/env python3
"""
Evaluation script for class-agnostic temporal action localization (TAL).

Evaluates ActionFormer on validation/test set with mAP@tIoU and AR metrics.

Usage:
    uv run python scripts/eval_tal.py \\
        --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \\
        --ckpt outputs/tal_motion_experiments/tal_motion_vit_b_experiment_v1/ \\
        --devices 0 1 6 7 \\
        --output-dir eval_results \\
        --topk 100

Environment variables (set before running):
    export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
    export HF_HOME=/lustre/work/mt/.cache/huggingface
    export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
    export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import wandb
from wandb_utils import add_wandb_cli_args, init_wandb_run

# Ensure ActionFormer imports work
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1] / "codes"  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from ActionFormer import _upstream  # noqa: F401 -- mount actionformer_libs
from actionformer_libs.core.config import load_config  # noqa: E402
from actionformer_libs.datasets import make_dataset, make_data_loader  # noqa: E402
from actionformer_libs.modeling import make_meta_arch  # noqa: E402
from actionformer_libs.utils import (  # noqa: E402
    valid_one_epoch,
    ANETdetection,
    fix_random_seed,
)


def _apply_multiclass_overrides(cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset", {})
    if dataset_cfg.get("label_mode", "binary") != "combined":
        return
    vocab_file = dataset_cfg.get("combined_vocab_file", "")
    if not vocab_file:
        raise ValueError("label_mode=combined requires dataset.combined_vocab_file")
    p = Path(vocab_file)
    if not p.exists():
        raise FileNotFoundError(f"combined_vocab_file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    label_to_id = payload.get("combined_label", {}).get("label_to_id", {})
    if not label_to_id:
        raise ValueError(f"Invalid combined vocab (combined_label.label_to_id): {p}")
    num_classes = max(label_to_id.values()) + 1
    cfg["dataset"]["num_classes"] = num_classes
    cfg["model"]["num_classes"] = num_classes
    cfg["test_cfg"]["multiclass_nms"] = True
    cfg["model"]["test_cfg"]["multiclass_nms"] = True
    print(f"[Eval] Combined multiclass mode enabled (num_classes={num_classes})")


def main(args):
    """Main evaluation function."""

    # =========================================================================
    # 1. Load config
    # =========================================================================
    print("[Eval] Loading config from:", args.config)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)
    _apply_multiclass_overrides(cfg)
    wandb_run, wandb_run_name = init_wandb_run(args, cfg, "eval_cfg")
    if wandb_run is not None:
        print(f"[Eval] W&B enabled: {wandb_run_name}")
    pprint(cfg)

    # =========================================================================
    # 2. Find checkpoint
    # =========================================================================
    ckpt_file = None
    if args.ckpt.endswith(".pth.tar"):
        # Explicit checkpoint file
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        ckpt_file = args.ckpt
    else:
        # Checkpoint directory: find latest or specific epoch
        if not os.path.isdir(args.ckpt):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt}")

        if args.epoch > 0:
            ckpt_file = os.path.join(args.ckpt, f"epoch_{args.epoch:03d}.pth.tar")
            if not os.path.exists(ckpt_file):
                raise FileNotFoundError(
                    f"Checkpoint for epoch {args.epoch} not found: {ckpt_file}"
                )
        else:
            # Find latest checkpoint
            ckpt_files = sorted(glob.glob(os.path.join(args.ckpt, "*.pth.tar")))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoints found in: {args.ckpt}")
            ckpt_file = ckpt_files[-1]

    print(f"[Eval] Using checkpoint: {ckpt_file}")

    # Override model config if needed
    if args.topk > 0:
        cfg["model"]["test_cfg"]["max_seg_num"] = args.topk
        print(f"[Eval] Set max_seg_num to {args.topk}")

    # =========================================================================
    # 3. Fix randomness
    # =========================================================================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    torch.cuda.set_device(args.devices[0])
    _ = fix_random_seed(0, include_cuda=False)
    torch.cuda.manual_seed(0)

    # =========================================================================
    # 4. Create dataset and dataloader
    # =========================================================================
    print("[Eval] Creating validation dataset...")
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    print(f"[Eval] Val samples: {len(val_dataset)}")

    # Set BS=1 for evaluation (matches upstream eval.py)
    val_loader = make_data_loader(
        val_dataset,
        False,
        None,
        batch_size=1,
        num_workers=cfg["loader"].get("num_workers", 0),
    )

    # =========================================================================
    # 5. Create model and load checkpoint
    # =========================================================================
    print("[Eval] Creating model...")
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = model.cuda(args.devices[0])
    model = nn.DataParallel(model, device_ids=args.devices)

    print(f"[Eval] Loading checkpoint: {ckpt_file}")
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(args.devices[0]),
    )

    # Load EMA model if available, otherwise regular state_dict
    if "state_dict_ema" in checkpoint:
        print("[Eval] Loading EMA model state")
        model.load_state_dict(checkpoint["state_dict_ema"])
    else:
        print("[Eval] Loading regular model state")
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # =========================================================================
    # 6. Create evaluator
    # =========================================================================
    val_db_vars = val_dataset.get_attributes()
    tiou_thresholds = val_db_vars.get(
        "tiou_thresholds", [0.3, 0.4, 0.5, 0.6, 0.7]
    )

    print(f"[Eval] tIoU thresholds: {tiou_thresholds}")

    # For tal_motion, we may need to handle evaluation differently
    # since it's class-agnostic. Let's create a simple evaluator.
    det_eval = None
    output_file = None

    if not args.saveonly:
        # Try to use standard ANETdetection evaluator
        # Note: this may require specific dataset format (json_file)
        try:
            det_eval = ANETdetection(
                val_dataset.json_file if hasattr(val_dataset, "json_file") else None,
                val_dataset.split[0] if hasattr(val_dataset, "split") else "val",
                tiou_thresholds=tiou_thresholds,
            )
        except Exception as e:
            print(f"[Eval] Warning: Could not create ANETdetection evaluator: {e}")
            print("[Eval] Will save predictions only")
            output_file = os.path.join(
                os.path.dirname(ckpt_file), "eval_results.pkl"
            )

    # =========================================================================
    # 7. Run evaluation
    # =========================================================================
    print("[Eval] Running validation...")
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,  # epoch (not used in eval mode)
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg.get("test_cfg", {}).get("ext_score_file", ""),
    )

    # =========================================================================
    # 8. Save results
    # =========================================================================
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = {
        "checkpoint": ckpt_file,
        "config": args.config,
        "model_name": cfg.get("model_name"),
        "dataset_name": cfg.get("dataset_name"),
        "val_split": cfg.get("val_split"),
        "tiou_thresholds": tiou_thresholds,
    }

    if isinstance(mAP, dict):
        results.update(mAP)
    else:
        results["mAP"] = float(mAP) if mAP is not None else 0.0

    results_json = os.path.join(args.output_dir, "eval_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Eval] Results saved to: {results_json}")
    print("[Eval] Results summary:")
    pprint(results)
    if wandb_run is not None:
        log_payload = {"eval/checkpoint": ckpt_file}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                log_payload[f"eval/{k}"] = v
        wandb.log(log_payload)
        if args.wandb_log_output and os.path.exists(results_json):
            artifact = wandb.Artifact(name=f"eval-results-{wandb_run.id}", type="evaluation_output")
            artifact.add_file(results_json)
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation for class-agnostic TAL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ActionFormer config YAML",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pth.tar or directory",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="CUDA device indices",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Specific epoch to evaluate (0 = latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Maximum number of predictions per video (0 = no limit)",
    )
    parser.add_argument(
        "--saveonly",
        action="store_true",
        help="Save predictions only, skip evaluation",
    )
    add_wandb_cli_args(parser, default_run_type="test", default_run_desc="eval-tal")
    parser.add_argument(
        "--wandb-log-output",
        action="store_true",
        help="Upload evaluation result JSON as W&B artifact",
    )

    args = parser.parse_args()
    main(args)
