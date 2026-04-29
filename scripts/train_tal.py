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
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
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
    save_checkpoint,
    make_optimizer,
    make_scheduler,
    fix_random_seed,
    ModelEma,
)

class _AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


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
    print(f"[Training] Combined multiclass mode enabled (num_classes={num_classes})")


def _validate_max_seq_len_for_model(cfg: dict) -> None:
    """Pre-flight check for ActionFormer max_seq_len divisibility constraints."""
    model_cfg = cfg["model"]
    max_seq_len = int(model_cfg["max_seq_len"])
    scale_factor = int(model_cfg["scale_factor"])
    fpn_start = int(model_cfg["fpn_start_level"])
    last_level = int(model_cfg["backbone_arch"][-1])
    win_cfg = model_cfg["n_mha_win_size"]
    if isinstance(win_cfg, int):
        wins = [win_cfg] * (1 + last_level)
    else:
        wins = list(win_cfg)
    if len(wins) != (1 + last_level):
        raise ValueError(
            f"n_mha_win_size length mismatch: got {len(wins)}, expected {1 + last_level}"
        )
    for level, w in enumerate(wins[fpn_start:]):
        s = scale_factor ** (fpn_start + level)
        stride = s * (w // 2) * 2 if w > 1 else s
        if max_seq_len % stride != 0:
            raise ValueError(
                "Invalid config: dataset/model max_seq_len must be divisible by "
                f"effective stride={stride} (level={fpn_start + level}, "
                f"scale={s}, n_mha_win_size={w}). Got max_seq_len={max_seq_len}. "
                "Try 128/256/384... depending on your model settings."
            )


def _load_stems_from_path_list(path_list_txt: str) -> list[str]:
    stems: list[str] = []
    for ln in Path(path_list_txt).read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line or line.startswith("#"):
            continue
        stems.append(Path(line).stem)
    # de-duplicate while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_split_list_tmpdir(stems: list[str], split_name: str) -> tempfile.TemporaryDirectory:
    tmpdir = tempfile.TemporaryDirectory(prefix=f"tal_{split_name}_from_featlist_")
    split_file = Path(tmpdir.name) / f"{split_name}.txt"
    split_file.write_text("\n".join(f"{stem}.json" for stem in stems) + "\n", encoding="utf-8")
    return tmpdir


def _save_ckpt(ckpt_folder: str, cfg: dict, model: nn.Module, model_ema, optimizer, scheduler, epoch: int, global_step: int, save_model_only: bool) -> None:
    ckpt_state = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "model_name": cfg["model_name"],
        "state_dict": model.state_dict(),
        "state_dict_ema": model_ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    save_checkpoint(
        ckpt_state,
        False,
        file_folder=ckpt_folder,
        file_name=f"step_{global_step:08d}.pth.tar",
    )
    if save_model_only:
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_name": cfg["model_name"],
                "state_dict": ckpt_state["state_dict"],
                "state_dict_ema": ckpt_state["state_dict_ema"],
            },
            os.path.join(ckpt_folder, f"step_{global_step:08d}.model_only.pth"),
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
    # Optional dataset path overrides for small-scale experiments.
    if args.feat_dir:
        cfg["dataset"]["feat_folder"] = args.feat_dir
    if args.annot_dir:
        cfg["dataset"]["annot_dir"] = args.annot_dir
    if args.split_list_dir:
        cfg["dataset"]["split_list_dir"] = args.split_list_dir

    split_list_tmpdir = None
    if args.feature_list_txt:
        stems = _load_stems_from_path_list(args.feature_list_txt)
        if not stems:
            raise ValueError(f"No valid entries in feature list: {args.feature_list_txt}")
        split_list_tmpdir = _build_split_list_tmpdir(stems, "train")
        cfg["train_split"] = ["train"]
        cfg["dataset"]["split_list_dir"] = split_list_tmpdir.name
        print(
            f"[Training] Using feature-list based subset: {len(stems)} items "
            f"(split_list_dir={split_list_tmpdir.name})"
        )

    _apply_multiclass_overrides(cfg)
    pprint(cfg)
    _validate_max_seq_len_for_model(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but no GPU is visible.")
    n_visible = torch.cuda.device_count()
    bad_devices = [d for d in args.devices if d < 0 or d >= n_visible]
    if bad_devices:
        raise ValueError(
            f"Invalid device ids {bad_devices}; visible cuda devices: 0..{n_visible-1}"
        )
    primary_device = args.devices[0]
    torch.cuda.set_device(primary_device)
    print(f"[Training] Primary CUDA device: {primary_device}")
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    # Save base config for reproducibility
    import yaml
    config_save_path = os.path.join(ckpt_folder, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Training] Config saved to: {config_save_path}")
    wandb_run, wandb_run_name = init_wandb_run(args, cfg, "train_cfg")
    if wandb_run is not None:
        wandb_run.config.update({"checkpoint_folder": ckpt_folder}, allow_val_change=True)
        print(f"[Training] W&B enabled: {wandb_run_name}")

    # TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    # Fix random seeds
    # Avoid touching all GPUs (manual_seed_all) to prevent unintended GPU0 usage.
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", 0), include_cuda=False)
    torch.cuda.manual_seed(cfg.get("init_rand_seed", 0))

    # Scale LR and workers based on number of GPUs
    num_gpus = len(args.devices)
    cfg["opt"]["learning_rate"] *= num_gpus
    cfg["loader"]["num_workers"] *= num_gpus
    cfg["loader"]["num_workers"] = min(cfg["loader"]["num_workers"], args.max_workers)
    print(
        f"[Training] Scaled learning rate x{num_gpus}; "
        f"num_workers capped at {cfg['loader']['num_workers']}"
    )

    # =========================================================================
    # 2. Create dataset and dataloader
    # =========================================================================
    print("[Training] Creating datasets...")
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars.get("empty_label_ids", [])

    print(f"[Training] Train samples: {len(train_dataset)}")

    # Data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg["loader"]
    )
    num_iters_per_epoch = len(train_loader)
    print(f"[Training] Iters per epoch: {num_iters_per_epoch}")
    if num_iters_per_epoch <= 0:
        raise RuntimeError("No training iterations available. Check dataset/split/feature files.")

    # Optional validation dataset/loader
    val_loader = None
    if args.val_every > 0:
        val_cfg = deepcopy(cfg)
        if args.val_feat_dir:
            val_cfg["dataset"]["feat_folder"] = args.val_feat_dir
        if args.val_annot_dir:
            val_cfg["dataset"]["annot_dir"] = args.val_annot_dir
        if args.val_split_list_dir:
            val_cfg["dataset"]["split_list_dir"] = args.val_split_list_dir
        val_split = [args.val_split] if args.val_split else list(val_cfg.get("val_split", ["val"]))

        val_split_tmpdir = None
        if args.val_feature_list_txt:
            val_stems = _load_stems_from_path_list(args.val_feature_list_txt)
            if not val_stems:
                raise ValueError(f"No valid entries in val feature list: {args.val_feature_list_txt}")
            val_split_tmpdir = _build_split_list_tmpdir(val_stems, "val")
            val_split = ["val"]
            val_cfg["dataset"]["split_list_dir"] = val_split_tmpdir.name
            print(
                f"[Training] Using val feature-list based subset: {len(val_stems)} items "
                f"(split_list_dir={val_split_tmpdir.name})"
            )

        print(f"[Training] Creating validation dataset (every {args.val_every} epoch[s])...")
        val_dataset = make_dataset(
            val_cfg["dataset_name"], False, val_split, **val_cfg["dataset"]
        )
        print(f"[Training] Val samples: {len(val_dataset)}")
        val_loader = make_data_loader(
            val_dataset,
            False,
            None,
            batch_size=1,
            num_workers=val_cfg["loader"].get("num_workers", 0),
        )
        if val_split_tmpdir is not None:
            # Keep temporary directory alive during training loop.
            _ = val_split_tmpdir

    # =========================================================================
    # 3. Create model, optimizer, scheduler
    # =========================================================================
    print("[Training] Creating model...")
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = model.cuda(primary_device)
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
    global_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Training] Resuming from: {args.resume}")
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(args.devices[0]),
            )
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", start_epoch * num_iters_per_epoch)
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"[Training] Resumed from epoch {start_epoch}, step {global_step}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    # =========================================================================
    # 5. Training loop
    # =========================================================================
    max_epochs = args.max_epochs or cfg["opt"].get(
        "early_stop_epochs",
        cfg["opt"]["epochs"] + cfg["opt"].get("warmup_epochs", 0),
    )
    # Save resolved runtime config after device-aware scaling.
    runtime_config_save_path = os.path.join(ckpt_folder, "config.runtime.yaml")
    with open(runtime_config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[Training] Runtime config saved to: {runtime_config_save_path}")

    for epoch in range(start_epoch, max_epochs):
        print(f"\n[Epoch {epoch+1}/{max_epochs}]")
        model.train()
        batch_time = _AverageMeter()
        losses_tracker: dict[str, _AverageMeter] = {}
        start_time = time.time()

        for iter_idx, video_list in enumerate(train_loader, 0):
            optimizer.zero_grad(set_to_none=True)
            losses = model(video_list)
            losses["final_loss"].backward()
            if cfg["train_cfg"]["clip_grad_l2norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["train_cfg"]["clip_grad_l2norm"],
                )
            optimizer.step()
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)
            global_step += 1

            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = _AverageMeter()
                losses_tracker[key].update(float(value.item()))

            if tb_writer is not None:
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/learning_rate", lr, global_step)
                tb_writer.add_scalar("train/final_loss", losses_tracker["final_loss"].val, global_step)
            if wandb_run is not None:
                lr = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/learning_rate": lr,
                        "train/final_loss": losses_tracker["final_loss"].val,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if (iter_idx != 0) and (iter_idx % args.print_freq == 0):
                torch.cuda.synchronize()
                batch_time.update((time.time() - start_time) / args.print_freq)
                start_time = time.time()
                print(
                    f"Epoch[{epoch+1:03d}] Iter[{iter_idx:05d}/{num_iters_per_epoch:05d}] "
                    f"Step[{global_step}] Time {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                    f"Loss {losses_tracker['final_loss'].val:.4f} ({losses_tracker['final_loss'].avg:.4f})"
                )

            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                _save_ckpt(
                    ckpt_folder, cfg, model, model_ema, optimizer, scheduler,
                    epoch, global_step, args.save_model_only
                )

            if val_loader is not None and args.val_every > 0 and (global_step % args.val_every == 0):
                print(f"[Validation] Running at step {global_step} ...")
                val_output_file = os.path.join(ckpt_folder, f"val_pred_step_{global_step:08d}.pkl")
                val_map = valid_one_epoch(
                    val_loader,
                    model_ema.module,
                    global_step,
                    evaluator=None,
                    output_file=val_output_file,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                )
                print(
                    f"[Validation] step={global_step} mAP={val_map:.4f} "
                    f"(predictions saved: {val_output_file})"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "validation/mAP": float(val_map),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

        train_lr = float(scheduler.get_last_lr()[0])
        tb_writer.add_scalar("train/lr_epoch", train_lr, epoch)
        tb_writer.flush()
        print(f"[Epoch {epoch+1}] lr={train_lr:.2e}, global_step={global_step}")

        if args.save_every_steps <= 0:
            should_save_epoch = (
                (epoch + 1) == max_epochs
                or (args.ckpt_freq > 0 and ((epoch + 1) % args.ckpt_freq == 0))
            )
            if should_save_epoch:
                _save_ckpt(
                    ckpt_folder, cfg, model, model_ema, optimizer, scheduler,
                    epoch, global_step, args.save_model_only
                )

        if wandb_run is not None and args.wandb_log_ckpt and (
            args.save_every_steps > 0 or should_save_epoch
        ):
            ckpt_path = os.path.join(ckpt_folder, f"step_{global_step:08d}.pth.tar")
            if os.path.exists(ckpt_path):
                artifact = wandb.Artifact(
                    name=f"checkpoint-step-{global_step:08d}",
                    type="model",
                    metadata={"epoch": epoch + 1, "global_step": global_step},
                )
                artifact.add_file(ckpt_path)
                wandb_run.log_artifact(artifact)

    print(f"\n[Training] Finished training. Results in: {ckpt_folder}")
    tb_writer.close()
    if wandb_run is not None:
        wandb_run.summary["final_global_step"] = global_step
        wandb_run.summary["final_epoch"] = max_epochs
        wandb_run.finish()


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
    parser.add_argument(
        "--print-freq",
        type=int,
        default=20,
        help="Training log frequency (iterations)",
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        default="",
        help="Override dataset.feat_folder (useful for subset experiments)",
    )
    parser.add_argument(
        "--annot-dir",
        type=str,
        default="",
        help="Override dataset.annot_dir",
    )
    parser.add_argument(
        "--split-list-dir",
        type=str,
        default="",
        help="Override dataset.split_list_dir",
    )
    parser.add_argument(
        "--feature-list-txt",
        type=str,
        default="",
        help="Text file with one .json/.npy path per line; used to build train subset",
    )
    parser.add_argument(
        "--ckpt-freq",
        type=int,
        default=5,
        help="Checkpoint frequency in epochs (<=0 disables periodic saves)",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help=(
            "Save checkpoint every N optimizer steps (gradient updates). "
            "If >0, this takes priority over --ckpt-freq."
        ),
    )
    parser.add_argument(
        "--save-model-only",
        action="store_true",
        help="Additionally save a model-only checkpoint (no optimizer/scheduler states)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Upper bound for dataloader num_workers after GPU scaling",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Run validation every N optimizer steps (gradient updates); 0 disables validation",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="",
        help="Validation split name (default: use config val_split)",
    )
    parser.add_argument(
        "--val-feat-dir",
        type=str,
        default="",
        help="Override validation dataset.feat_folder",
    )
    parser.add_argument(
        "--val-annot-dir",
        type=str,
        default="",
        help="Override validation dataset.annot_dir",
    )
    parser.add_argument(
        "--val-split-list-dir",
        type=str,
        default="",
        help="Override validation dataset.split_list_dir",
    )
    parser.add_argument(
        "--val-feature-list-txt",
        type=str,
        default="",
        help="Validation subset txt (one .json/.npy path per line)",
    )
    add_wandb_cli_args(parser, default_run_type="train", default_run_desc="tal-motion")
    parser.add_argument(
        "--wandb-log-ckpt",
        action="store_true",
        help="Upload saved checkpoints as W&B artifacts",
    )

    args = parser.parse_args()
    main(args)
