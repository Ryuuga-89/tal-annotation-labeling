#!/usr/bin/env python3
"""
Run checkpoint-based TAL inference test on randomly sampled videos.

This script:
1) scans all annotation JSONs under ANNOT_ROOT_DIR,
2) samples N videos with a fixed seed,
3) runs ActionFormer inference with a specified checkpoint,
4) writes a reproducible package under output-dir:
   - predictions.json
   - manifest.json
   - predictions_readable.txt
   - videos/<video_stem>.mp4 (copied file)
   - visualizations/<video_stem>.mp4 (prediction/GT interval dots)

Usage:
    uv run python scripts/run_checkpoint_infer_test.py \
        --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
        --checkpoint outputs/.../step_00012345.pth.tar \
        --num-videos 10 \
        --seed 42 \
        --output-dir outputs/infer_test_run_001
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project packages are importable.
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1] / "codes"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))
if str(_THIS.parents[1] / "scripts") not in sys.path:
    sys.path.insert(0, str(_THIS.parents[1] / "scripts"))

from ActionFormer import _upstream  # noqa: F401
from actionformer_libs.core.config import load_config
from actionformer_libs.utils import fix_random_seed
from VideoMAEv2.dataset.annotation import load_annotation
from infer_tal import (
    _load_combined_id_to_label,
    infer_one_video,
    load_features,
    load_model,
)


@dataclass(frozen=True)
class SampleEntry:
    video_stem: str
    annot_json: Path
    video_mp4: Path
    feature_npy: Path


def _must_get_env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"Environment variable is required: {name}")
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"{name} path does not exist: {p}")
    return p


def _scan_candidates(
    annot_root: Path,
    video_root: Path,
    feat_dir: Path,
    allowed_stems: set[str] | None = None,
) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    for annot_json in sorted(annot_root.glob("*.json")):
        stem = annot_json.stem
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        feat_npy = feat_dir / f"{stem}.npy"
        video_mp4 = video_root / f"{stem}.mp4"
        if not feat_npy.exists():
            continue
        if not video_mp4.exists():
            continue
        entries.append(
            SampleEntry(
                video_stem=stem,
                annot_json=annot_json,
                video_mp4=video_mp4,
                feature_npy=feat_npy,
            )
        )
    return entries


def _load_population_stems_from_txt(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Population list file not found: {path}")
    stems: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        stems.add(p.stem)
    if not stems:
        raise ValueError(f"Population list file has no valid entries: {path}")
    return stems


def _copy_or_replace_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _write_readable_predictions(output_path: Path, predictions: dict[str, dict]) -> None:
    lines: list[str] = []
    for video_stem in sorted(predictions.keys()):
        lines.append(f"[{video_stem}]")
        detections = predictions[video_stem].get("detections", [])
        sorted_detections = sorted(
            detections,
            key=lambda d: (float(d.get("start_time", 0.0)), float(d.get("end_time", 0.0))),
        )
        if not sorted_detections:
            lines.append("(no detections)")
            lines.append("")
            continue
        for det in sorted_detections:
            st = float(det["start_time"])
            ed = float(det["end_time"])
            score = float(det.get("score", 0.0))
            line = f"{st:.3f}-{ed:.3f} (score={score:.4f})"
            if "class_name" in det:
                line += f" class={det['class_name']}"
            elif "class_id" in det:
                line += f" class_id={int(det['class_id'])}"
            lines.append(line)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _is_active_interval(t_sec: float, start: float, end: float) -> bool:
    return start <= t_sec <= end


def _gt_active_with_contiguous_boundary_gaps(
    t_sec: float,
    gt_intervals: list[tuple[float, float]],
    fps: float,
    *,
    contiguous_eps: float = 1e-4,
) -> bool:
    """True if t is inside some GT segment, except a short blackout around
    boundaries where one segment ends exactly when the next starts (e.g. 1.6
    between [1.4,1.6] and [1.6,2.0]) so the dot blinks off for ~one frame.
    """
    if not gt_intervals:
        return False
    sorted_ivs = sorted(gt_intervals, key=lambda x: (x[0], x[1]))
    # Half-width of blackout centered on the shared boundary (~1 frame total).
    half_gap = max(0.5 / max(fps, 1e-6), 1e-6)
    for i in range(len(sorted_ivs) - 1):
        _s0, e0 = sorted_ivs[i]
        s1, _e1 = sorted_ivs[i + 1]
        if abs(e0 - s1) <= contiguous_eps:
            boundary = 0.5 * (e0 + s1)
            if abs(t_sec - boundary) <= half_gap:
                return False
    return any(_is_active_interval(t_sec, s, e) for s, e in gt_intervals)


def _render_interval_dots_video(
    src_video: Path,
    dst_video: Path,
    pred_intervals: list[tuple[float, float]],
    gt_intervals: list[tuple[float, float]],
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Read all frames once so we can retry writer with different codecs
    frames: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from source video: {src_video}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame_idx, frame in enumerate(frames):
        t_sec = frame_idx / fps
        pred_active = any(_is_active_interval(t_sec, s, e) for s, e in pred_intervals)
        gt_active = _gt_active_with_contiguous_boundary_gaps(
            t_sec, gt_intervals, fps
        )

        # GT interval dot (red): top-left
        cv2.circle(
            frame,
            (32, 32),
            10,
            (0, 0, 255) if gt_active else (80, 80, 80),
            thickness=-1,
        )
        cv2.putText(frame, "GT", (48, 37), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Prediction interval dot (green): below GT
        cv2.circle(
            frame,
            (32, 62),
            10,
            (0, 255, 0) if pred_active else (80, 80, 80),
            thickness=-1,
        )
        cv2.putText(frame, "Pred", (48, 67), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # First try imageio-ffmpeg for H.264/yuv420p MP4, which is generally the
    # most widely playable format.
    last_error = "unknown writer error"
    try:
        import imageio_ffmpeg

        tmp_path = dst_video.with_suffix(".tmp.mp4")
        writer = imageio_ffmpeg.write_frames(
            str(tmp_path),
            size=(width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            writer.send(frame)
        writer.close()
        if dst_video.exists():
            dst_video.unlink()
        tmp_path.replace(dst_video)
        check = cv2.VideoCapture(str(dst_video))
        ok_open = check.isOpened()
        ok_read, _ = check.read()
        check.release()
        if ok_open and ok_read:
            return
        last_error = "imageio_ffmpeg wrote undecodable mp4"
    except Exception as e:
        last_error = f"imageio_ffmpeg failed: {e}"

    # Fallback to OpenCV writer if imageio-ffmpeg path is unavailable.
    codec_candidates = ("avc1", "H264", "mp4v")
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            last_error = f"{last_error}; failed to open VideoWriter(codec={codec})"
            continue
        for frame in frames:
            writer.write(frame)
        writer.release()

        # Re-open once to ensure produced file is decodable.
        check = cv2.VideoCapture(str(dst_video))
        ok_open = check.isOpened()
        ok_read, _ = check.read()
        check.release()
        if ok_open and ok_read:
            return
        last_error = f"{last_error}; wrote but failed decode check(codec={codec})"

    raise RuntimeError(
        f"Failed to create playable visualization video: {dst_video} ({last_error})"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TAL inference test with checkpoint on random sampled videos."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ActionFormer config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth or .pth.tar)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        required=True,
        help="Number of videos to sample for inference test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for video sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save predictions, manifest, and copied videos",
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        default="",
        help=(
            "Directory with feature .npy files. "
            "If omitted, uses dataset.feat_folder from --config."
        ),
    )
    parser.add_argument(
        "--population-list-txt",
        type=str,
        default="data/splits/ALL/test.txt",
        help=(
            "Text file that defines the candidate population for random sampling "
            "(one json/npy path per line). Default: data/splits/ALL/test.txt"
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    if args.num_videos <= 0:
        raise ValueError("--num-videos must be > 0")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    videos_out_dir = output_dir / "videos"
    vis_out_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_out_dir.mkdir(parents=True, exist_ok=True)
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    id_to_label = _load_combined_id_to_label(cfg)

    feat_dir = Path(args.feat_dir) if args.feat_dir else Path(cfg["dataset"]["feat_folder"])
    if not feat_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feat_dir}")
    population_list_path = Path(args.population_list_txt)
    allowed_stems = _load_population_stems_from_txt(population_list_path)

    annot_root = _must_get_env_path("ANNOT_ROOT_DIR")
    video_root = _must_get_env_path("VIDEO_DATA_DIR")

    candidates = _scan_candidates(
        annot_root=annot_root,
        video_root=video_root,
        feat_dir=feat_dir,
        allowed_stems=allowed_stems,
    )
    if not candidates:
        raise RuntimeError(
            "No valid candidates found. Check ANNOT_ROOT_DIR/VIDEO_DATA_DIR and feature directory."
        )
    if args.num_videos > len(candidates):
        raise ValueError(
            f"--num-videos ({args.num_videos}) is larger than available candidates ({len(candidates)})"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(candidates, k=args.num_videos)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    _ = fix_random_seed(0, include_cuda=False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    model = load_model(cfg, str(checkpoint_path), args.device, torch.cuda.is_available())

    feat_stride_base = cfg["dataset"].get("feat_stride", 2)
    num_frames = cfg["dataset"].get("num_frames", 16)
    fps_default = float(cfg["dataset"].get("default_fps", 10.0))
    downsample_rate = cfg["dataset"].get("downsample_rate", 1)

    predictions: dict[str, dict] = {}
    manifest_items: list[dict] = []

    for entry in sampled:
        rec = load_annotation(entry.annot_json)
        fps = float(rec.video_fps) if rec.video_fps > 0 else fps_default
        feats, stride_mult = load_features(str(entry.feature_npy), downsample_rate)
        feat_stride = feat_stride_base * stride_mult
        duration = float(rec.video_duration) if rec.video_duration > 0 else (
            (feats.shape[1] * feat_stride + 0.5 * num_frames) / fps
        )

        detections = infer_one_video(
            model=model,
            video_id=entry.video_stem,
            feats=feats,
            duration=duration,
            feat_stride=feat_stride,
            num_frames=num_frames,
            fps=fps,
            score_thresh=args.score_thresh,
            id_to_label=id_to_label,
        )
        predictions[entry.video_stem] = {
            "fps": fps,
            "duration": duration,
            "detections": detections,
        }
        pred_intervals = [
            (float(det["start_time"]), float(det["end_time"])) for det in detections
        ]
        gt_intervals = [
            (float(action.start_time), float(action.end_time)) for action in rec.actions
        ]

        copied_video_path = videos_out_dir / f"{entry.video_stem}.mp4"
        _copy_or_replace_file(entry.video_mp4, copied_video_path)
        vis_video_path = vis_out_dir / f"{entry.video_stem}.mp4"
        _render_interval_dots_video(
            src_video=copied_video_path,
            dst_video=vis_video_path,
            pred_intervals=pred_intervals,
            gt_intervals=gt_intervals,
        )
        manifest_items.append(
            {
                "video_stem": entry.video_stem,
                "annotation_json": str(entry.annot_json.resolve()),
                "source_video": str(entry.video_mp4.resolve()),
                "copied_video": str(copied_video_path.resolve()),
                "visualization_video": str(vis_video_path.resolve()),
                "feature_npy": str(entry.feature_npy.resolve()),
            }
        )

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "num_videos": int(args.num_videos),
        "seed": int(args.seed),
        "score_thresh": float(args.score_thresh),
        "annot_root_dir": str(annot_root.resolve()),
        "video_data_dir": str(video_root.resolve()),
        "feature_dir": str(feat_dir.resolve()),
        "population_list_txt": str(population_list_path.resolve()),
        "sampled_videos": manifest_items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    readable_predictions_path = output_dir / "predictions_readable.txt"
    _write_readable_predictions(readable_predictions_path, predictions)

    print(f"[InferenceTest] candidates={len(candidates)} sampled={len(sampled)}")
    print(f"[InferenceTest] predictions: {predictions_path}")
    print(f"[InferenceTest] readable:    {readable_predictions_path}")
    print(f"[InferenceTest] manifest:    {manifest_path}")
    print(f"[InferenceTest] videos dir:  {videos_out_dir}")
    print(f"[InferenceTest] vis dir:     {vis_out_dir}")


if __name__ == "__main__":
    parser = _build_arg_parser()
    main(parser.parse_args())
