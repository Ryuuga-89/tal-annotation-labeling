#!/usr/bin/env python3
"""
Training script for class-agnostic temporal action localization (TAL).

Trains ActionFormer on tal_motion dataset with optional resumption.

Usage:
    uv run python scripts/train_tal.py \\
        --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \\
        --output-folder outputs/tal_motion_experiments \\
        --devices 0 1 6 7 \\
        --tag experiment_v1 \\
        --max-epochs 100

Environment variables (set before running):
    export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
    export HF_HOME=/lustre/work/mt/.cache/huggingface
    export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
    export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks
"""
import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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
    train_one_epoch,
    valid_one_epoch,
    save_checkpoint,
    make_optimizer,
    make_scheduler,
    fix_random_seed,
    ModelEma,
)


def main(args):
    """Main training function."""

    # =========================================================================
    # 1. Setup config and output folder
    # =========================================================================
    print("[Training] Loading config from:", args.config)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)
    pprint(cfg)

    # Create output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    if len(args.tag) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            args.output_folder, f"{cfg_filename}_{ts}"
        )
    else:
        ckpt_folder = os.path.join(
            args.output_folder, f"{cfg_filename}_{args.tag}"
        )

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    print(f"[Training] Checkpoint folder: {ckpt_folder}")

    # Save config for reproducibility
    import yaml
    config_save_path = os.path.join(ckpt_folder, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Training] Config saved to: {config_save_path}")

    # TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    # Fix random seeds
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", 0), include_cuda=True)

    # Scale LR and workers based on number of GPUs
    num_gpus = len(args.devices)
    cfg["opt"]["learning_rate"] *= num_gpus
    cfg["loader"]["num_workers"] *= num_gpus
    print(f"[Training] Scaled learning rate x{num_gpus}, workers x{num_gpus}")

    # =========================================================================
    # 2. Create dataset and dataloader
    # =========================================================================
    print("[Training] Creating datasets...")
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars.get("empty_label_ids", [])

    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    val_db_vars = val_dataset.get_attributes()

    print(f"[Training] Train samples: {len(train_dataset)}")
    print(f"[Training] Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg["loader"]
    )
    val_loader = make_data_loader(
        val_dataset, False, rng_generator,
        batch_size=cfg["loader"].get("batch_size", 1),
        num_workers=cfg["loader"].get("num_workers", 0)
    )

    num_iters_per_epoch = len(train_loader)
    print(f"[Training] Iters per epoch: {num_iters_per_epoch}")

    # =========================================================================
    # 3. Create model, optimizer, scheduler
    # =========================================================================
    print("[Training] Creating model...")
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=args.devices)

    optimizer = make_optimizer(model, cfg["opt"])
    scheduler = make_scheduler(optimizer, cfg["opt"], num_iters_per_epoch)

    # Model EMA
    print("[Training] Enabling Model EMA...")
    model_ema = ModelEma(model)

    # =========================================================================
    # 4. Resume from checkpoint (optional)
    # =========================================================================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Training] Resuming from: {args.resume}")
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(args.devices[0]),
            )
            start_epoch = checkpoint.get("epoch", 0)
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"[Training] Resumed from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    # =========================================================================
    # 5. Training loop
    # =========================================================================
    max_epochs = args.max_epochs or cfg.get("max_epoch", 50)
    best_loss = float("inf")

    for epoch in range(start_epoch, max_epochs):
        print(f"\n[Epoch {epoch+1}/{max_epochs}]")

        # Train one epoch
        train_loss, train_lr = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            tb_writer,
        )

        # Update model EMA
        model_ema.update(model)

        # Validation
        print(f"[Validation] Running validation...")
        val_loss = valid_one_epoch(
            val_loader,
            model,
            epoch,
            tb_writer,
            evaluator=None,  # Skip evaluator for now (mAP calculation expensive)
        )

        # Log to TensorBoard
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/lr", train_lr, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.flush()

        print(
            f"[Epoch {epoch+1}] "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"lr={train_lr:.2e}"
        )

        # Save checkpoint every epoch (or only best)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        ckpt_path = os.path.join(ckpt_folder, f"epoch_{epoch+1:03d}.pth.tar")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_name": cfg["model_name"],
                "state_dict": model.state_dict(),
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            file_name=ckpt_path,
        )

        if is_best:
            print(f"[Epoch {epoch+1}] New best val_loss: {best_loss:.4f}")

    print(f"\n[Training] Finished training. Results in: {ckpt_folder}")
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training for class-agnostic TAL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ActionFormer config YAML",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="outputs",
        help="Root folder for checkpoints",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag for checkpoint folder (default: timestamp)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="CUDA device indices",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint for resumption",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs from config",
    )

    args = parser.parse_args()
    main(args)
