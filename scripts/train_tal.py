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
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from wandb_utils import add_wandb_cli_args, init_wandb_run

# Ensure ActionFormer imports work
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1] / "codes"  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from ActionFormer import _upstream  # noqa: F401 -- mount actionformer_libs
from actionformer_libs.core.config import load_config  # noqa: E402
from actionformer_libs.datasets import make_dataset  # noqa: E402
from actionformer_libs.datasets.data_utils import (  # noqa: E402
    trivial_batch_collator,
    worker_init_reset_seed,
)
from actionformer_libs.modeling import make_meta_arch  # noqa: E402
from actionformer_libs.utils import (  # noqa: E402
    valid_one_epoch,
    ANETdetection,
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


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def _build_data_loader(
    dataset,
    *,
    is_training: bool,
    generator,
    batch_size: int,
    num_workers: int,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=(is_training and sampler is None),
        drop_last=is_training,
        generator=generator,
        sampler=sampler,
        persistent_workers=(num_workers > 0),
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


def _build_anet_gt_json_from_dataset(dataset, out_json: str, split_name: str) -> str:
    """Create ActivityNet-style GT json from tal_motion dataset object."""
    database = {}
    for item in dataset.data_list:
        video_id = item["id"]
        segments = item.get("segments")
        labels = item.get("labels")
        annotations = []
        if segments is not None and labels is not None and len(segments) == len(labels):
            for seg, lab in zip(segments, labels):
                annotations.append(
                    {
                        "segment": [float(seg[0]), float(seg[1])],
                        "label_id": int(lab),
                    }
                )
        database[video_id] = {
            "subset": split_name,
            "duration": float(item.get("duration", 0.0)),
            "annotations": annotations,
        }
    payload = {"database": database}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return out_json


def _run_validation_with_nms_fallback(
    *,
    val_loader,
    model_for_eval,
    global_step: int,
    val_evaluator,
    val_output_file: str,
    tb_writer,
    print_freq: int,
) -> float:
    """Run validation, falling back to NMS-free mode if extension is missing."""
    try:
        return float(
            valid_one_epoch(
                val_loader,
                model_for_eval,
                global_step,
                evaluator=val_evaluator,
                output_file=None if val_evaluator is not None else val_output_file,
                tb_writer=tb_writer,
                print_freq=print_freq,
            )
        )
    except RuntimeError as e:
        msg = str(e)
        if "nms_1d_cpu extension not built" not in msg:
            raise
        print(
            "[Validation] nms_1d_cpu extension is missing. "
            "Falling back to prediction-dump-only validation "
            "(skip mAP computation to avoid very slow/no-NMS evaluation)."
        )
        _ = valid_one_epoch(
            val_loader,
            model_for_eval,
            global_step,
            evaluator=None,
            output_file=val_output_file,
            tb_writer=tb_writer,
            print_freq=print_freq,
        )
        return 0.0


def _interval_to_steps(value: float, unit: str, updates_per_epoch: int, default_epochs: float = 0.5) -> int:
    """Convert epoch/step interval to step interval."""
    if value <= 0:
        value = default_epochs
        unit = "epoch"
    if unit == "epoch":
        return max(1, int(round(value * updates_per_epoch)))
    return max(1, int(round(value)))


def main(args):
    """Main training function."""
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size_env > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if ddp_enabled and not dist.is_initialized():
        # ddp-timeout-hours=0 means effectively disable timeout.
        dist_timeout = (
            datetime.timedelta(days=3650)
            if args.ddp_timeout_hours <= 0
            else datetime.timedelta(hours=args.ddp_timeout_hours)
        )
        dist.init_process_group(backend="nccl", timeout=dist_timeout)

    # =========================================================================
    # 1. Setup config and output folder
    # =========================================================================
    if _is_main_process():
        print("[Training] Loading config from:", args.config)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)
    # Dataset split directory override (generated by train_val_test_split.py).
    if args.split_list_dir:
        cfg["dataset"]["split_list_dir"] = args.split_list_dir
    cfg["loader"]["batch_size"] = args.batch_size

    _apply_multiclass_overrides(cfg)
    if _is_main_process():
        pprint(cfg)
    _validate_max_seq_len_for_model(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but no GPU is visible.")
    n_visible = torch.cuda.device_count()
    if ddp_enabled:
        if local_rank < 0 or local_rank >= n_visible:
            raise ValueError(
                f"Invalid LOCAL_RANK={local_rank}; visible cuda devices: 0..{n_visible-1}"
            )
    else:
        bad_devices = [d for d in args.devices if d < 0 or d >= n_visible]
        if bad_devices:
            raise ValueError(
                f"Invalid device ids {bad_devices}; visible cuda devices: 0..{n_visible-1}"
            )
    primary_device = local_rank if ddp_enabled else args.devices[0]
    torch.cuda.set_device(primary_device)
    if _is_main_process():
        print(f"[Training] Primary CUDA device: {primary_device}")
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create output folder
    if _is_main_process() and not os.path.exists(args.output_folder):
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

    if _is_main_process() and not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    if ddp_enabled:
        dist.barrier()
    if _is_main_process():
        print(f"[Training] Checkpoint folder: {ckpt_folder}")

    # Save base config for reproducibility
    import yaml
    if _is_main_process():
        config_save_path = os.path.join(ckpt_folder, "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[Training] Config saved to: {config_save_path}")
    wandb_run, wandb_run_name = (None, None)
    if _is_main_process():
        wandb_run, wandb_run_name = init_wandb_run(args, cfg, "train_cfg")
        if wandb_run is not None:
            wandb_run.config.update({"checkpoint_folder": ckpt_folder}, allow_val_change=True)
            print(f"[Training] W&B enabled: {wandb_run_name}")

    # TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs")) if _is_main_process() else None

    # Fix random seeds
    # Avoid touching all GPUs (manual_seed_all) to prevent unintended GPU0 usage.
    rng_generator = fix_random_seed(cfg.get("init_rand_seed", 0), include_cuda=False)
    torch.cuda.manual_seed(cfg.get("init_rand_seed", 0))

    # Scale LR and workers based on number of GPUs
    num_gpus = world_size_env if ddp_enabled else len(args.devices)
    cfg["loader"]["num_workers"] *= num_gpus
    cfg["loader"]["num_workers"] = min(cfg["loader"]["num_workers"], args.max_workers)
    if _is_main_process():
        print(
            f"[Training] DDP={ddp_enabled}, world_size={num_gpus}; "
            f"num_workers capped at {cfg['loader']['num_workers']}"
        )

    # =========================================================================
    # 2. Create dataset and dataloader
    # =========================================================================
    if _is_main_process():
        print("[Training] Creating datasets...")
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_db_vars = train_dataset.get_attributes()
    cfg["model"]["train_cfg"]["head_empty_cls"] = train_db_vars.get("empty_label_ids", [])

    if _is_main_process():
        print(f"[Training] Train samples: {len(train_dataset)}")

    # Data loaders
    if ddp_enabled:
        if args.batch_size % world_size_env != 0:
            raise ValueError(
                f"--batch-size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size_env}) in DDP mode"
            )
        per_rank_batch_size = args.batch_size // world_size_env
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size_env,
            rank=rank,
            shuffle=True,
            seed=cfg.get("init_rand_seed", 0),
            drop_last=True,
        )
    else:
        per_rank_batch_size = args.batch_size
        train_sampler = None
    train_loader = _build_data_loader(
        train_dataset,
        is_training=True,
        generator=rng_generator,
        batch_size=per_rank_batch_size,
        num_workers=cfg["loader"]["num_workers"],
        sampler=train_sampler,
    )
    num_iters_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(num_iters_per_epoch / args.grad_accum_steps)
    val_every_steps = _interval_to_steps(args.val_every, args.unit, updates_per_epoch, default_epochs=0.5)
    save_every_steps = _interval_to_steps(args.save_every, args.unit, updates_per_epoch, default_epochs=0.5)
    if _is_main_process():
        print(
            f"[Training] Iters per epoch: {num_iters_per_epoch}, "
            f"updates per epoch: {updates_per_epoch} "
            f"(grad_accum_steps={args.grad_accum_steps})"
        )
        print(
            f"[Training] unit={args.unit}, "
            f"validation every {val_every_steps} step(s), "
            f"checkpoint save every {save_every_steps} step(s)"
        )
    if num_iters_per_epoch <= 0:
        raise RuntimeError("No training iterations available. Check dataset/split/feature files.")

    # Optional validation dataset/loader
    val_loader = None
    val_evaluator = None
    if val_every_steps > 0:
        val_cfg = deepcopy(cfg)
        val_split = [args.val_split] if args.val_split else list(val_cfg.get("val_split", ["val"]))
        if _is_main_process():
            print(f"[Training] Creating validation dataset (every {val_every_steps} step[s])...")
        val_dataset = make_dataset(
            val_cfg["dataset_name"], False, val_split, **val_cfg["dataset"]
        )
        if _is_main_process():
            print(f"[Training] Val samples: {len(val_dataset)}")
        val_loader = _build_data_loader(
            val_dataset,
            is_training=False,
            generator=None,
            batch_size=args.val_batch_size,
            num_workers=args.val_loader_workers,
            sampler=None,
        )
        # Prefer true mAP evaluation when ANET evaluator can be constructed.
        try:
            val_db_vars = val_dataset.get_attributes()
            tiou_thresholds = val_db_vars.get(
                "tiou_thresholds", [0.3, 0.4, 0.5, 0.6, 0.7]
            )
            if hasattr(val_dataset, "json_file") and getattr(val_dataset, "json_file", None):
                ant_file = val_dataset.json_file
            else:
                ant_file = os.path.join(ckpt_folder, "val_gt_for_anet.json")
                split_name = val_split[0] if val_split else "val"
                _build_anet_gt_json_from_dataset(val_dataset, ant_file, split_name)
            val_evaluator = ANETdetection(
                ant_file,
                val_split[0] if val_split else "val",
                tiou_thresholds=tiou_thresholds,
                num_workers=args.val_eval_workers,
            )
            if _is_main_process():
                print(
                    f"[Training] Validation evaluator enabled "
                    f"(tIoU={tiou_thresholds}, gt_instances={len(val_evaluator.ground_truth)})"
                )
        except Exception as e:
            val_evaluator = None
            msg = (
                "[Training] Validation evaluator unavailable. "
                f"Reason: {e}"
            )
            if args.allow_val_dump_only:
                if _is_main_process():
                    print(msg + " -> fallback to prediction dump only.")
            else:
                raise RuntimeError(
                    msg
                    + " Set --allow-val-dump-only if you intentionally want dump-only validation."
                )
    # =========================================================================
    # 3. Create model, optimizer, scheduler
    # =========================================================================
    if _is_main_process():
        print("[Training] Creating model...")
    model = make_meta_arch(cfg["model_name"], **cfg["model"]).cuda(primary_device)
    if ddp_enabled:
        model = DDP(model, device_ids=[primary_device], output_device=primary_device)
    elif len(args.devices) > 1:
        model = nn.DataParallel(model, device_ids=args.devices)

    optimizer = make_optimizer(model, cfg["opt"])
    scheduler = make_scheduler(optimizer, cfg["opt"], updates_per_epoch)

    # Model EMA
    if _is_main_process():
        print("[Training] Enabling Model EMA...")
    model_ema = ModelEma(_unwrap_model(model))

    # =========================================================================
    # 4. Resume from checkpoint (optional)
    # =========================================================================
    start_epoch = 0
    global_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if _is_main_process():
                print(f"[Training] Resuming from: {args.resume}")
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(args.devices[0]),
            )
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", start_epoch * updates_per_epoch)
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if _is_main_process():
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
    if _is_main_process():
        runtime_config_save_path = os.path.join(ckpt_folder, "config.runtime.yaml")
        with open(runtime_config_save_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[Training] Runtime config saved to: {runtime_config_save_path}")

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if _is_main_process():
            print(f"\n[Epoch {epoch+1}/{max_epochs}]")
        model.train()
        batch_time = _AverageMeter()
        losses_tracker: dict[str, _AverageMeter] = {}
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for iter_idx, video_list in enumerate(train_loader, 0):
            losses = model(video_list)
            (losses["final_loss"] / args.grad_accum_steps).backward()

            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = _AverageMeter()
                losses_tracker[key].update(float(value.item()))

            should_update = ((iter_idx + 1) % args.grad_accum_steps == 0) or (
                (iter_idx + 1) == num_iters_per_epoch
            )
            if should_update:
                if cfg["train_cfg"]["clip_grad_l2norm"] > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg["train_cfg"]["clip_grad_l2norm"],
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(_unwrap_model(model))
                global_step += 1

                if tb_writer is not None and _is_main_process():
                    lr = scheduler.get_last_lr()[0]
                    tb_writer.add_scalar("train/learning_rate", lr, global_step)
                    tb_writer.add_scalar("train/final_loss", losses_tracker["final_loss"].val, global_step)
                if wandb_run is not None and _is_main_process():
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

                if _is_main_process() and (global_step != 0) and (global_step % args.print_freq == 0):
                    torch.cuda.synchronize()
                    batch_time.update((time.time() - start_time) / args.print_freq)
                    start_time = time.time()
                    print(
                        f"Epoch[{epoch+1:03d}] Iter[{iter_idx+1:05d}/{num_iters_per_epoch:05d}] "
                        f"Step[{global_step}] Time {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                        f"Loss {losses_tracker['final_loss'].val:.4f} ({losses_tracker['final_loss'].avg:.4f})"
                    )

                if _is_main_process() and (global_step % save_every_steps == 0):
                    _save_ckpt(
                        ckpt_folder, cfg, _unwrap_model(model), model_ema, optimizer, scheduler,
                        epoch, global_step, args.save_model_only
                    )

                if val_loader is not None and (global_step % val_every_steps == 0):
                    if ddp_enabled:
                        dist.barrier()
                    if _is_main_process():
                        print(f"[Validation] Running at step {global_step} ...")
                        val_output_file = os.path.join(ckpt_folder, f"val_pred_step_{global_step:08d}.pkl")
                        val_map = _run_validation_with_nms_fallback(
                            val_loader=val_loader,
                            model_for_eval=model_ema.module,
                            global_step=global_step,
                            val_evaluator=val_evaluator,
                            val_output_file=val_output_file,
                            tb_writer=tb_writer,
                            print_freq=args.print_freq,
                        )
                        print(
                            f"[Validation] step={global_step} mAP={val_map:.4f} "
                            f"(predictions saved: {val_output_file if val_evaluator is None else 'via evaluator'})"
                        )
                        if val_evaluator is not None and val_map <= 0.0:
                            print(
                                "[Validation] mAP is 0.0. This can happen early in training, "
                                "but if it persists check class mapping / split quality."
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
                    if ddp_enabled:
                        dist.barrier()

        train_lr = float(scheduler.get_last_lr()[0])
        if tb_writer is not None and _is_main_process():
            tb_writer.add_scalar("train/lr_epoch", train_lr, epoch)
            tb_writer.flush()
        if _is_main_process():
            print(f"[Epoch {epoch+1}] lr={train_lr:.2e}, global_step={global_step}")

        if _is_main_process() and ((global_step % save_every_steps) != 0):
            _save_ckpt(
                ckpt_folder, cfg, _unwrap_model(model), model_ema, optimizer, scheduler,
                epoch, global_step, args.save_model_only
            )

        if wandb_run is not None and args.wandb_log_ckpt:
            ckpt_path = os.path.join(ckpt_folder, f"step_{global_step:08d}.pth.tar")
            if os.path.exists(ckpt_path):
                artifact = wandb.Artifact(
                    name=f"checkpoint-step-{global_step:08d}",
                    type="model",
                    metadata={"epoch": epoch + 1, "global_step": global_step},
                )
                artifact.add_file(ckpt_path)
                wandb_run.log_artifact(artifact)

    if _is_main_process():
        print(f"\n[Training] Finished training. Results in: {ckpt_folder}")
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None and _is_main_process():
        wandb_run.summary["final_global_step"] = global_step
        wandb_run.summary["final_epoch"] = max_epochs
        wandb_run.finish()
    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


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
        "--batch-size",
        type=int,
        default=4,
        help="Train dataloader batch size",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (1 = no accumulation)",
    )
    parser.add_argument(
        "--split-list-dir",
        type=str,
        default="",
        help="Override dataset.split_list_dir (contains train/val/test txt files)",
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=["epoch", "step"],
        default="step",
        help="Unit for --save-every and --val-every",
    )
    parser.add_argument(
        "--save-every",
        type=float,
        default=0.0,
        help=(
            "Checkpoint interval in selected unit. "
            "If --unit=epoch, fractional values like 0.5 are allowed. "
            "Default 0 means auto: 0.5 epoch."
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
        type=float,
        default=0.0,
        help=(
            "Validation interval in selected unit. "
            "If --unit=epoch, fractional values like 0.5 are allowed. "
            "Default 0 means auto: 0.5 epoch."
        ),
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="",
        help="Validation split name (default: use config val_split)",
    )
    parser.add_argument(
        "--val-loader-workers",
        type=int,
        default=1,
        help="Validation DataLoader workers (set 0/1 on constrained systems)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation/Test batch size used during in-training validation",
    )
    parser.add_argument(
        "--allow-val-dump-only",
        action="store_true",
        help="Allow validation without evaluator (mAP will be fixed to 0.0)",
    )
    parser.add_argument(
        "--val-eval-workers",
        type=int,
        default=1,
        help="ANET evaluator worker processes for validation (default: 1 for stability)",
    )
    add_wandb_cli_args(parser, default_run_type="train", default_run_desc="tal-motion")
    parser.add_argument(
        "--wandb-log-ckpt",
        action="store_true",
        help="Upload saved checkpoints as W&B artifacts",
    )
    parser.add_argument(
        "--ddp-timeout-hours",
        type=int,
        default=6,
        help=(
            "Process group timeout in hours for DDP collectives. "
            "Set 0 to effectively disable timeout."
        ),
    )

    args = parser.parse_args()
    main(args)
