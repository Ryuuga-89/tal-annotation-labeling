"""Phase 1: extract VideoMAE V2 features for the 30s/10s chunk dataset.

Per video, produce:
    <out_dir>/<stem>.npy   shape [N, embed_dim], dtype float16 by default
    <out_dir>/<stem>.json  meta: {target_fps, window_size, stride, num_steps,
                                  step_time, embed_dim, model, ckpt, ...}

A side-cart `index.json` summarises all processed videos for downstream code
to enumerate without rescanning the directory.

Run:
    PYTHONPATH=codes uv run python -m VideoMAEv2.extract_features \\
        --config codes/VideoMAEv2/configs/extract_vit_b.yaml \\
        --annot-dir /raid/.../annot/30s_chunks_action_31detail_2 \\
        --video-dir /raid/.../30s_chunks \\
        --out-dir   data/features/30s_mae_b_16_2 \\
        --ckpt-path /path/to/vit_b_k710_ft.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python -m VideoMAEv2.extract_features` from the project root.
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from VideoMAEv2.dataset import (  # noqa: E402
    AnnotationRecord,
    VideoChunkDataset,
    load_annotation,
)
from VideoMAEv2.dataset.chunk_dataset import collate_clips, pair_video_with_annotation  # noqa: E402
from VideoMAEv2.models import build_backbone  # noqa: E402


# ---------------------------------------------------------------------------
# Config


@dataclass
class ExtractConfig:
    # I/O
    annot_dir: str = ""
    video_dir: str = ""
    out_dir: str = ""
    video_suffix: str = ".mp4"
    # Model
    model_name: str = "vit_base_patch16_224"
    ckpt_path: str = ""
    drop_path_rate: float = 0.0
    # Sampling
    target_fps: float = 10.0
    window_size: int = 16
    stride: int = 2
    input_size: int = 224
    resize_mode: str = "squash"
    # Runtime
    batch_size: int = 16
    num_workers: int = 0  # decord readers are fine in single-process for sane sizes
    device: str = "cuda"
    dtype: str = "float16"
    save_dtype: str = "float16"
    # Selection
    limit: int | None = None  # process at most N annotation files (debug)
    overwrite: bool = False
    fail_on_missing_video: bool = False
    # Sharding: split the (sorted) annotation list across multiple processes.
    # Process k of K handles json_paths[k::K]. Indices are stable as long as
    # the underlying directory listing is stable, which `sorted()` guarantees.
    shard_id: int = 0
    num_shards: int = 1


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(cfg: ExtractConfig, override: dict) -> ExtractConfig:
    fields = {f for f in cfg.__dataclass_fields__}
    for k, v in override.items():
        if v is None:
            continue
        if k not in fields:
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg


def parse_args(argv: list[str] | None = None) -> ExtractConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--annot-dir", type=str, default=None)
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--video-suffix", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--ckpt-path", type=str, default=None)
    p.add_argument("--target-fps", type=float, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--resize-mode", type=str, default=None, choices=["squash", "shortside_crop"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--save-dtype", type=str, default=None, choices=["float32", "float16"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-on-missing-video", action="store_true")
    p.add_argument("--shard-id", type=int, default=None,
                   help="0-based shard index. Process picks json_paths[shard_id::num_shards].")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Total number of shards (= number of parallel processes).")

    args = p.parse_args(argv)
    file_cfg = _load_config(args.config)

    cli_cfg = {
        "annot_dir": args.annot_dir,
        "video_dir": args.video_dir,
        "out_dir": args.out_dir,
        "video_suffix": args.video_suffix,
        "model_name": args.model_name,
        "ckpt_path": args.ckpt_path,
        "target_fps": args.target_fps,
        "window_size": args.window_size,
        "stride": args.stride,
        "input_size": args.input_size,
        "resize_mode": args.resize_mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "limit": args.limit,
        "overwrite": True if args.overwrite else None,
        "fail_on_missing_video": True if args.fail_on_missing_video else None,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }

    cfg = ExtractConfig()
    cfg = _merge(cfg, file_cfg)
    cfg = _merge(cfg, cli_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Extraction


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _np_dtype(name: str) -> np.dtype:
    return {"float32": np.float32, "float16": np.float16}[name]


def _make_step_time_index(num_steps: int, target_fps: float, stride: int, window_size: int) -> list[list[float]]:
    return [
        [i * stride / target_fps, (i * stride + window_size) / target_fps]
        for i in range(num_steps)
    ]


def extract_one(
    backbone,
    video_path: Path,
    rec: AnnotationRecord,
    *,
    cfg: ExtractConfig,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    save_dtype: np.dtype,
) -> dict:
    out_npy = out_dir / f"{rec.video_stem}.npy"
    out_meta = out_dir / f"{rec.video_stem}.json"
    if (
        out_npy.exists()
        and out_meta.exists()
        and not cfg.overwrite
    ):
        return {"stem": rec.video_stem, "status": "skipped", "num_steps": -1}

    if not video_path.exists():
        msg = f"missing video: {video_path}"
        if cfg.fail_on_missing_video:
            raise FileNotFoundError(msg)
        return {"stem": rec.video_stem, "status": "missing_video", "msg": msg}

    ds = VideoChunkDataset(
        video_path,
        target_fps=cfg.target_fps,
        window_size=cfg.window_size,
        stride=cfg.stride,
        input_size=cfg.input_size,
        resize_mode=cfg.resize_mode,
    )
    if len(ds) == 0:
        return {
            "stem": rec.video_stem,
            "status": "too_short",
            "duration": ds.spec.duration_sec,
            "num_target_frames": ds.spec.num_target_frames,
        }

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_clips,
        pin_memory=(device.type == "cuda"),
    )

    feats = np.empty((len(ds), backbone.embed_dim), dtype=save_dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    for batch in loader:
        clips = batch.clips.to(device, non_blocking=True)
        idx = batch.step_idx.numpy()
        with torch.inference_mode():
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = backbone(clips)
            else:
                out = backbone(clips)
        feats[idx] = out.float().cpu().numpy().astype(save_dtype)

    np.save(out_npy, feats)

    meta = {
        "stem": rec.video_stem,
        "video_path": str(video_path),
        "annotation_path": str(rec.json_path),
        "duration_sec": ds.spec.duration_sec,
        "src_fps": ds.spec.src_fps,
        "src_num_frames": ds.spec.src_num_frames,
        "target_fps": cfg.target_fps,
        "num_target_frames": ds.spec.num_target_frames,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "input_size": cfg.input_size,
        "resize_mode": cfg.resize_mode,
        "num_steps": len(ds),
        "embed_dim": backbone.embed_dim,
        "model_name": cfg.model_name,
        "ckpt_path": cfg.ckpt_path,
        "save_dtype": cfg.save_dtype,
        "compute_dtype": cfg.dtype,
        # First step time span; downstream can derive others as i*stride/fps.
        "step_time_first": [0.0, cfg.window_size / cfg.target_fps],
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"stem": rec.video_stem, "status": "ok", "num_steps": len(ds)}


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    if not cfg.annot_dir or not cfg.video_dir or not cfg.out_dir:
        raise SystemExit("annot_dir, video_dir, out_dir are required")

    annot_dir = Path(cfg.annot_dir)
    video_dir = Path(cfg.video_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if cfg.limit is not None:
        json_paths = json_paths[: cfg.limit]
    if cfg.num_shards < 1:
        raise SystemExit(f"num_shards must be >= 1 (got {cfg.num_shards})")
    if not (0 <= cfg.shard_id < cfg.num_shards):
        raise SystemExit(
            f"shard_id={cfg.shard_id} not in [0, {cfg.num_shards})"
        )
    total_in_split = len(json_paths)
    if cfg.num_shards > 1:
        json_paths = json_paths[cfg.shard_id :: cfg.num_shards]
        print(
            f"[shard {cfg.shard_id}/{cfg.num_shards}] "
            f"processing {len(json_paths)} of {total_in_split} files",
            flush=True,
        )
    if not json_paths:
        raise SystemExit(f"no annotation json files under {annot_dir}")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device != "cuda" else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU", flush=True)
    dtype = _torch_dtype(cfg.dtype)
    save_dtype = _np_dtype(cfg.save_dtype)

    backbone = build_backbone(
        cfg.model_name,
        ckpt_path=cfg.ckpt_path or None,
        drop_path_rate=cfg.drop_path_rate,
    )
    backbone.eval()
    backbone.to(device)

    summary = {
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "embed_dim": backbone.embed_dim,
        "shard_id": cfg.shard_id,
        "num_shards": cfg.num_shards,
        "total_in_split": total_in_split,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "videos": [],
    }

    progress = tqdm(
        json_paths,
        desc=f"extract[{cfg.shard_id}/{cfg.num_shards}]"
        if cfg.num_shards > 1
        else "extract",
        dynamic_ncols=True,
    )
    for jp in progress:
        try:
            video_path, rec = pair_video_with_annotation(
                video_dir, jp, suffix=cfg.video_suffix
            )
        except Exception as e:
            summary["videos"].append({"json": str(jp), "status": "load_error", "msg": str(e)})
            continue
        try:
            res = extract_one(
                backbone,
                video_path,
                rec,
                cfg=cfg,
                out_dir=out_dir,
                device=device,
                dtype=dtype,
                save_dtype=save_dtype,
            )
        except Exception as e:
            res = {"stem": rec.video_stem, "status": "error", "msg": str(e)}
        summary["videos"].append(res)
        progress.set_postfix(last=res.get("status", "?"))

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if cfg.num_shards > 1:
        index_name = f"index.shard{cfg.shard_id:02d}of{cfg.num_shards:02d}.json"
    else:
        index_name = "index.json"
    with (out_dir / index_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for v in summary["videos"] if v.get("status") == "ok")
    n_skip = sum(1 for v in summary["videos"] if v.get("status") == "skipped")
    n_err = len(summary["videos"]) - n_ok - n_skip
    tag = (
        f"shard {cfg.shard_id}/{cfg.num_shards}"
        if cfg.num_shards > 1
        else "done"
    )
    print(f"{tag}: ok={n_ok} skipped={n_skip} other={n_err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
