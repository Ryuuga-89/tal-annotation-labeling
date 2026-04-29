"""Phase 1: extract VideoMAE V2 features for the 30s/10s chunk dataset.

Per video, produce:
    <out_dir>/<stem>.npy   shape [N, embed_dim], dtype float16 by default
    <out_dir>/<stem>.json  meta: {target_fps, window_size, stride, num_steps,
                                  step_time, embed_dim, model, ckpt, ...}

A side-cart `index.json` summarises all processed videos for downstream code
to enumerate without rescanning the directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# Tiny utility to tee stdout/stderr to a file while still printing to console.
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


# Allow `python -m VideoMAEv2.extract_features` from the project root.
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from VideoMAEv2.dataset import (  # noqa: E402
    AnnotationRecord,
    VideoChunkDataset,
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
    annot_list: str = ""

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
    num_workers: int = 0
    device: str = "cuda"
    gpu_id: int | None = None  # if specified, use cuda:gpu_id instead of default cuda
    dtype: str = "float16"
    save_dtype: str = "float16"

    # Selection
    limit: int | None = None
    overwrite: bool = False
    fail_on_missing_video: bool = False
    shuffle: bool = False
    seed: int = 42

    # Sharding
    shard_id: int = 0
    num_shards: int = 1

    # Housekeeping
    # Do GC / empty_cache every N batches (0 disables; end-of-video cleanup still runs)
    cleanup_interval_batches: int = 0
    # Do GC / empty_cache every N videos in the outer loop (0 disables).
    cleanup_interval_videos: int = 0
    # Print progress metrics every N videos (0 disables extra logs)
    progress_log_interval_videos: int = 20
    # Decord decode threads per video
    decode_threads: int = 2
    # Decode at input resolution for squash mode (faster, less memory)
    decode_resize_on_read: bool = True
    # "full": decode full target-fps video once, then unfold windows.
    # "batch": decode only frames needed per mini-batch.
    # "auto": use full for short clips, batch for long clips.
    decode_mode: str = "auto"
    # Threshold for decode_mode=auto. If num_target_frames is greater than this,
    # switch to batch decode to avoid expensive full-video materialization.
    auto_batch_threshold_frames: int = 320
    # If true on CUDA, choose batch size from available VRAM.
    auto_batch_size: bool = False
    # Upper bound for effective batch size when using decode_mode=batch.
    # Large values can increase decode latency without improving throughput.
    max_batch_size_batch_decode: int = 32
    # If true, insert cuda synchronize around timed blocks for accurate stage
    # profiling. This can reduce throughput; keep disabled for production runs.
    sync_cuda_timing: bool = False


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
    p.add_argument(
        "--annot-list",
        type=str,
        default=None,
        help="File listing annotation json filenames (one per line). Restricts processing to this subset.",
    )
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
    p.add_argument("--gpu-id", type=int, default=None, help="GPU device ID (e.g., 0, 1, 6, 7). Only used if device is cuda.")
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--save-dtype", type=str, default=None, choices=["float32", "float16"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-on-missing-video", action="store_true")
    p.add_argument("--shuffle", action="store_true", help="Shuffle processing order before sharding")
    p.add_argument("--seed", type=int, default=None, help="Random seed used with --shuffle")
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--cleanup-interval-batches", type=int, default=None)
    p.add_argument("--cleanup-interval-videos", type=int, default=None)
    p.add_argument("--progress-log-interval-videos", type=int, default=None)
    p.add_argument("--decode-threads", type=int, default=None)
    p.add_argument("--no-decode-resize-on-read", action="store_true")
    p.add_argument("--decode-mode", type=str, default=None, choices=["auto", "full", "batch"])
    p.add_argument("--auto-batch-threshold-frames", type=int, default=None)
    p.add_argument("--auto-batch-size", action="store_true")
    p.add_argument("--max-batch-size-batch-decode", type=int, default=None)
    p.add_argument("--sync-cuda-timing", action="store_true")

    args = p.parse_args(argv)
    file_cfg = _load_config(args.config)

    cli_cfg = {
        "annot_dir": args.annot_dir,
        "video_dir": args.video_dir,
        "out_dir": args.out_dir,
        "video_suffix": args.video_suffix,
        "annot_list": args.annot_list,
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
        "gpu_id": args.gpu_id,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "limit": args.limit,
        "overwrite": True if args.overwrite else None,
        "fail_on_missing_video": True if args.fail_on_missing_video else None,
        "shuffle": True if args.shuffle else None,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "cleanup_interval_batches": args.cleanup_interval_batches,
        "cleanup_interval_videos": args.cleanup_interval_videos,
        "progress_log_interval_videos": args.progress_log_interval_videos,
        "decode_threads": args.decode_threads,
        "decode_resize_on_read": False if args.no_decode_resize_on_read else None,
        "decode_mode": args.decode_mode,
        "auto_batch_threshold_frames": args.auto_batch_threshold_frames,
        "auto_batch_size": True if args.auto_batch_size else None,
        "max_batch_size_batch_decode": args.max_batch_size_batch_decode,
        "sync_cuda_timing": True if args.sync_cuda_timing else None,
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


def _maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _auto_batch_size_for_device(device: torch.device, fallback: int) -> int:
    if device.type != "cuda":
        return fallback
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    if total_gb >= 70:
        return max(fallback, 64)
    if total_gb >= 38:
        return max(fallback, 48)
    if total_gb >= 22:
        return max(fallback, 32)
    return fallback


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
    # Top-level timer for the whole video
    t_start = time.perf_counter()

    out_npy = out_dir / f"{rec.video_stem}.npy"
    out_meta = out_dir / f"{rec.video_stem}.json"
    if out_npy.exists() and out_meta.exists() and not cfg.overwrite:
        return {"stem": rec.video_stem, "status": "skipped", "num_steps": -1, "elapsed_sec": 0.0}

    if not video_path.exists():
        msg = f"missing video: {video_path}"
        if cfg.fail_on_missing_video:
            raise FileNotFoundError(msg)
        return {"stem": rec.video_stem, "status": "missing_video", "msg": msg, "elapsed_sec": 0.0}

    # --- stage: load dataset / video read ---
    t_ds_start = time.perf_counter()
    ds = VideoChunkDataset(
        video_path,
        target_fps=cfg.target_fps,
        window_size=cfg.window_size,
        stride=cfg.stride,
        input_size=cfg.input_size,
        resize_mode=cfg.resize_mode,
        decode_threads=cfg.decode_threads,
        decode_resize_on_read=cfg.decode_resize_on_read,
    )
    t_ds_end = time.perf_counter()
    t_load = t_ds_end - t_ds_start

    if len(ds) == 0:
        return {
            "stem": rec.video_stem,
            "status": "too_short",
            "duration": ds.spec.duration_sec,
            "num_target_frames": ds.spec.num_target_frames,
            "elapsed_sec": 0.0,
            "t_load": t_load,
        }

    num_steps = len(ds)
    feats = np.empty((num_steps, backbone.embed_dim), dtype=save_dtype)

    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    pin_memory = device.type == "cuda"

    use_direct_batching = int(cfg.num_workers or 0) == 0
    loader = None
    if not use_direct_batching:
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=int(cfg.num_workers or 0),
            pin_memory=pin_memory,
            collate_fn=collate_clips,
            persistent_workers=bool(cfg.num_workers and cfg.num_workers > 0),
        )

    # Per-stage accumulators
    t_preproc = 0.0
    t_infer = 0.0
    t_postproc = 0.0

    batch_i = 0
    if use_direct_batching:
        if cfg.decode_mode == "auto":
            # Keep "full" only for very short clips. For 30s clips at 10fps
            # (~300 frames), batch decode is substantially more stable/faster.
            effective_decode_mode = (
                "full" if ds.spec.num_target_frames <= cfg.auto_batch_threshold_frames else "batch"
            )
        else:
            effective_decode_mode = cfg.decode_mode

        effective_batch_size = cfg.batch_size
        if effective_decode_mode == "batch":
            effective_batch_size = min(cfg.batch_size, cfg.max_batch_size_batch_decode)

        if effective_decode_mode == "full":
            # Stable path for short clips (our 30s chunks): decode once and unfold.
            t_prep_start = time.perf_counter()
            all_clips = ds.windowed_clips()  # [N, 3, T, H, W]
            t_prep_end = time.perf_counter()
            t_preproc += (t_prep_end - t_prep_start)
            batch_iter = (
                (s, min(s + effective_batch_size, num_steps))
                for s in range(0, num_steps, effective_batch_size)
            )
        else:
            # Lower memory path: decode only frames needed per batch.
            all_clips = None
            batch_iter = (
                (s, min(s + effective_batch_size, num_steps))
                for s in range(0, num_steps, effective_batch_size)
            )

        for start, end in batch_iter:
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_prep_start = time.perf_counter()
            if effective_decode_mode == "full":
                clips_cpu = all_clips[start:end]
            else:
                clips_cpu = ds.batch_clips(start, end)  # [B, 3, T, H, W]
            clips = clips_cpu.to(device, non_blocking=True)
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_prep_end = time.perf_counter()
            t_preproc += (t_prep_end - t_prep_start)

            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                        t_inf_start = time.perf_counter()
                        out = backbone(clips)
                        _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                        t_inf_end = time.perf_counter()
                else:
                    _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                    t_inf_start = time.perf_counter()
                    out = backbone(clips)
                    _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                    t_inf_end = time.perf_counter()
            t_infer += (t_inf_end - t_inf_start)

            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_post_start = time.perf_counter()
            out_cpu = out.detach()
            if save_dtype == np.float16:
                feats[start:end] = out_cpu.to(torch.float16).cpu().numpy()
            else:
                feats[start:end] = out_cpu.to(torch.float32).cpu().numpy()
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_post_end = time.perf_counter()
            t_postproc += (t_post_end - t_post_start)

            del clips_cpu, clips, out, out_cpu
            batch_i += 1
            if cfg.cleanup_interval_batches and (batch_i % cfg.cleanup_interval_batches == 0):
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if all_clips is not None:
            del all_clips
    else:
        # We'll measure time spent waiting for the batch (this includes collate/preprocess)
        batch_fetch_start = time.perf_counter()
        for batch in loader:
            batch_fetched = time.perf_counter()
            t_preproc += (batch_fetched - batch_fetch_start)

            # move clips to device (non-blocking attempt)
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_move_start = time.perf_counter()
            clips = batch.clips.to(device, non_blocking=True)
            idx = batch.step_idx.numpy()
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_move_end = time.perf_counter()
            t_preproc += (t_move_end - t_move_start)

            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                        t_inf_start = time.perf_counter()
                        out = backbone(clips)
                        _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                        t_inf_end = time.perf_counter()
                else:
                    _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                    t_inf_start = time.perf_counter()
                    out = backbone(clips)
                    _maybe_cuda_sync(device, cfg.sync_cuda_timing)
                    t_inf_end = time.perf_counter()
            t_infer += (t_inf_end - t_inf_start)

            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_post_start = time.perf_counter()
            out_cpu = out.detach().to("cpu")
            if save_dtype == np.float16:
                feats[idx] = out_cpu.to(torch.float16).numpy()
            else:
                feats[idx] = out_cpu.to(torch.float32).numpy()
            _maybe_cuda_sync(device, cfg.sync_cuda_timing)
            t_post_end = time.perf_counter()
            t_postproc += (t_post_end - t_post_start)

            del clips, out, out_cpu, idx, batch
            batch_i += 1

            if cfg.cleanup_interval_batches and (batch_i % cfg.cleanup_interval_batches == 0):
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            batch_fetch_start = time.perf_counter()

    # --- stage: save to disk ---
    t_save_start = time.perf_counter()
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
        "num_steps": num_steps,
        "embed_dim": backbone.embed_dim,
        "model_name": cfg.model_name,
        "ckpt_path": cfg.ckpt_path,
        "save_dtype": cfg.save_dtype,
        "compute_dtype": cfg.dtype,
        "step_time": _make_step_time_index(num_steps, cfg.target_fps, cfg.stride, cfg.window_size),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    t_save_end = time.perf_counter()
    t_save = t_save_end - t_save_start

    # end-of-video cleanup (avoid forced per-video GC/empty_cache: expensive)
    ds.release_cache()
    del ds, feats, loader

    elapsed = time.perf_counter() - t_start
    # Return detailed stage timings for profiling
    return {
        "stem": rec.video_stem,
        "status": "ok",
        "num_steps": num_steps,
        "elapsed_sec": elapsed,
        "t_load": t_load,
        "t_preproc": t_preproc,
        "t_infer": t_infer,
        "t_postproc": t_postproc,
        "t_save": t_save,
        "clip_per_sec": (num_steps / elapsed) if elapsed > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    if not cfg.annot_dir or not cfg.video_dir or not cfg.out_dir:
        raise SystemExit("annot_dir, video_dir, out_dir are required")

    annot_dir = Path(cfg.annot_dir)
    video_dir = Path(cfg.video_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory and tee stdout/stderr to a shard-specific log file
    logs_dir = out_dir / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"shard{cfg.shard_id:02d}of{cfg.num_shards:02d}.log" if cfg.num_shards > 1 else "extract.log"
        log_path = logs_dir / log_name
        # open in append mode so multiple runs append
        fh = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, fh)
        sys.stderr = Tee(sys.__stderr__, fh)
        print(f"[log] writing logs to {log_path}", flush=True)
    except Exception as e:
        print(f"[warn] failed to create log file under {logs_dir}: {e}", flush=True)

    if cfg.annot_list:
        list_path = Path(cfg.annot_list)
        names = [
            ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        json_paths = []
        missing: list[str] = []
        for n in names:
            jp = annot_dir / Path(n).name
            if jp.exists():
                json_paths.append(jp)
            else:
                missing.append(n)
        json_paths.sort()
        if missing:
            print(
                f"[warn] {len(missing)}/{len(names)} entries in {list_path} "
                f"not found under {annot_dir} (showing up to 5): {missing[:5]}",
                flush=True,
            )
    else:
        json_paths = sorted(annot_dir.glob("*.json"))

    if cfg.limit is not None:
        json_paths = json_paths[: cfg.limit]
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(json_paths)
        print(
            f"[shuffle] randomized order enabled (seed={cfg.seed}); applied before sharding",
            flush=True,
        )
    if cfg.num_shards < 1:
        raise SystemExit(f"num_shards must be >= 1 (got {cfg.num_shards})")
    if not (0 <= cfg.shard_id < cfg.num_shards):
        raise SystemExit(f"shard_id={cfg.shard_id} not in [0, {cfg.num_shards})")

    total_in_split = len(json_paths)
    if cfg.num_shards > 1:
        json_paths = json_paths[cfg.shard_id :: cfg.num_shards]
        print(
            f"[shard {cfg.shard_id}/{cfg.num_shards}] processing {len(json_paths)} of {total_in_split} files",
            flush=True,
        )
    # When not overwriting, pre-filter already materialized items so tqdm "it/s"
    # reflects actual extraction throughput instead of mixing in near-zero-cost
    # skipped entries.
    if not cfg.overwrite:
        pending = []
        skipped_existing = 0
        for jp in json_paths:
            stem = jp.stem
            out_npy = out_dir / f"{stem}.npy"
            out_meta = out_dir / f"{stem}.json"
            if out_npy.exists() and out_meta.exists():
                skipped_existing += 1
                continue
            pending.append(jp)
        if skipped_existing > 0:
            print(
                f"[prefilter] existing outputs skipped before loop: "
                f"{skipped_existing}/{len(json_paths)}",
                flush=True,
            )
        json_paths = pending
    if not json_paths:
        raise SystemExit(f"no annotation json files under {annot_dir}")

    # Determine device: if gpu_id is specified and cuda is available, use cuda:gpu_id
    if cfg.device == "cuda" and cfg.gpu_id is not None:
        device = torch.device(f"cuda:{cfg.gpu_id}")
    elif cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    dtype = _torch_dtype(cfg.dtype)
    save_dtype = _np_dtype(cfg.save_dtype)
    
    if cfg.device == "cuda":
        gpu_msg = f" (GPU {cfg.gpu_id})" if cfg.gpu_id is not None else " (default GPU)"
        print(f"[device] using {device}{gpu_msg}", flush=True)
        # Throughput-oriented defaults for inference-only extraction.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if cfg.auto_batch_size:
            old_bs = cfg.batch_size
            cfg.batch_size = _auto_batch_size_for_device(device, cfg.batch_size)
            if cfg.batch_size != old_bs:
                print(
                    f"[tune] auto batch size enabled: {old_bs} -> {cfg.batch_size}",
                    flush=True,
                )

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
        desc=f"extract[{cfg.shard_id}/{cfg.num_shards}]" if cfg.num_shards > 1 else "extract",
        dynamic_ncols=True,
    )
    shard_started = time.perf_counter()
    ok_steps_acc = 0
    ok_videos_acc = 0
    load_acc = 0.0
    preproc_acc = 0.0
    infer_acc = 0.0
    postproc_acc = 0.0
    save_acc = 0.0
    timing_count = 0
    recent_ok = deque(maxlen=32)  # (elapsed_sec, num_steps)
    for i, jp in enumerate(progress, start=1):
        try:
            video_path, rec = pair_video_with_annotation(video_dir, jp, suffix=cfg.video_suffix)
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
        if res.get("status") == "ok":
            ok_videos_acc += 1
            ok_steps_acc += int(res.get("num_steps", 0))
            load_acc += float(res.get("t_load", 0.0))
            preproc_acc += float(res.get("t_preproc", 0.0))
            infer_acc += float(res.get("t_infer", 0.0))
            postproc_acc += float(res.get("t_postproc", 0.0))
            save_acc += float(res.get("t_save", 0.0))
            timing_count += 1
            recent_ok.append((float(res.get("elapsed_sec", 0.0)), int(res.get("num_steps", 0))))

        elapsed = time.perf_counter() - shard_started
        videos_per_sec = i / elapsed if elapsed > 0 else 0.0
        avg_sec_per_video = elapsed / i if i > 0 else 0.0
        eta_sec = (len(json_paths) - i) / videos_per_sec if videos_per_sec > 0 else 0.0
        clip_per_sec = ok_steps_acc / elapsed if elapsed > 0 else 0.0
        infer_clip_per_sec = ok_steps_acc / infer_acc if infer_acc > 0 else 0.0
        if recent_ok:
            recent_elapsed = sum(x[0] for x in recent_ok)
            recent_steps = sum(x[1] for x in recent_ok)
            recent_clip_per_sec = recent_steps / recent_elapsed if recent_elapsed > 0 else 0.0
        else:
            recent_clip_per_sec = 0.0

        progress.set_postfix(
            last=res.get("status", "?"),
            vps=f"{videos_per_sec:.3f}",
            cps=f"{clip_per_sec:.1f}",
            cps_recent=f"{recent_clip_per_sec:.1f}",
            cps_infer=f"{infer_clip_per_sec:.1f}",
            eta_min=f"{eta_sec / 60.0:.1f}",
        )

        if cfg.progress_log_interval_videos and (i % cfg.progress_log_interval_videos == 0):
            denom = max(timing_count, 1)
            print(
                f"[metrics shard {cfg.shard_id}/{cfg.num_shards}] "
                f"done={i}/{len(json_paths)} avg_s_per_video={avg_sec_per_video:.2f} "
                f"videos_per_min={videos_per_sec * 60.0:.2f} clip_per_sec={clip_per_sec:.2f} "
                f"clip_per_sec_recent={recent_clip_per_sec:.2f} clip_per_sec_infer={infer_clip_per_sec:.2f} "
                f"eta_min={eta_sec / 60.0:.1f} "
                f"stage_avg_sec(load={load_acc / denom:.2f}, preproc={preproc_acc / denom:.2f}, "
                f"infer={infer_acc / denom:.2f}, postproc={postproc_acc / denom:.2f}, save={save_acc / denom:.2f})",
                flush=True,
            )

        if cfg.cleanup_interval_videos and (i % cfg.cleanup_interval_videos == 0):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    index_name = (
        f"index.shard{cfg.shard_id:02d}of{cfg.num_shards:02d}.json"
        if cfg.num_shards > 1
        else "index.json"
    )
    with (out_dir / index_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for v in summary["videos"] if v.get("status") == "ok")
    n_skip = sum(1 for v in summary["videos"] if v.get("status") == "skipped")
    n_err = len(summary["videos"]) - n_ok - n_skip
    tag = f"shard {cfg.shard_id}/{cfg.num_shards}" if cfg.num_shards > 1 else "done"
    print(f"{tag}: ok={n_ok} skipped={n_skip} other={n_err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())