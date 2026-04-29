#!/usr/bin/env python3
"""Shared Weights & Biases helpers for train/infer/eval scripts."""
from __future__ import annotations

import argparse
import datetime

import wandb

WANDB_PROJECT = "tal-annotation-labeling"
WANDB_ENTITY = "models-institute-of-science-tokyo"


def add_wandb_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_run_type: str,
    default_run_desc: str,
) -> None:
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--wandb-run-type",
        type=str,
        default=default_run_type,
        choices=["train", "exp", "test"],
        help="Run name prefix in W&B ({train,exp,test})",
    )
    parser.add_argument(
        "--wandb-run-desc",
        type=str,
        default=default_run_desc,
        help="Run description used in W&B run name",
    )


def init_wandb_run(args, cfg: dict, cfg_key: str):
    """Initialize W&B run following repository naming conventions."""
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None, ""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.wandb_run_type}_{args.wandb_run_desc}_{ts}"
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        mode=args.wandb_mode,
        config={
            "cli_args": vars(args),
            cfg_key: cfg,
        },
    )
    return run, run_name
