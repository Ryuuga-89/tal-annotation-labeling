#!/usr/bin/env python3
"""
Inference script for class-agnostic temporal action localization (TAL).

Loads pre-trained ActionFormer model + features, runs inference,
applies NMS, and outputs detections in JSON format.

Usage:
    uv run python scripts/infer_tal.py \\
        --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \\
        --ckpt outputs/tal_motion_vit_b/.../epoch_050.pth.tar \\
        --feat-dir data/features/30s_mae_b_16_2 \\
        --output-json detections.json \\
        --nms-thresh 0.5 \\
        --score-thresh 0.3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure ActionFormer imports work
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1] / "codes"  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from ActionFormer import _upstream  # noqa: F401 -- mount actionformer_libs
from actionformer_libs.core.config import load_config  # noqa: E402
from actionformer_libs.modeling import make_meta_arch  # noqa: E402
from actionformer_libs.utils import fix_random_seed  # noqa: E402


def _load_combined_id_to_label(cfg: dict) -> dict[int, str]:
    dataset_cfg = cfg.get("dataset", {})
    if dataset_cfg.get("label_mode", "binary") != "combined":
        return {}
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
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"[Inference] Combined multiclass mode enabled (num_classes={num_classes})")
    return id_to_label


def get_video_ids_from_dir(feat_dir: str) -> list[str]:
    """List all .npy files in feat_dir, return stems (without extension)."""
    feat_dir = Path(feat_dir)
    stems = []
    for npy_file in sorted(feat_dir.glob("*.npy")):
        stems.append(npy_file.stem)
    return stems


def load_model(cfg: dict, ckpt_path: str, device_index: int, use_cuda: bool) -> nn.Module:
    """Load ActionFormer model from config and checkpoint."""
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    if use_cuda:
        model = nn.DataParallel(model, device_ids=[device_index])

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    if use_cuda:
        checkpoint = torch.load(
            ckpt_path,
            map_location=lambda storage, loc: storage.cuda(device_index),
        )
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Try loading EMA model first, fall back to regular state_dict
    if "state_dict_ema" in checkpoint:
        print("  -> Loading EMA model state")
        model.load_state_dict(checkpoint["state_dict_ema"])
    else:
        print("  -> Loading regular model state")
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    return model


def load_features(feat_file: str, downsample_rate: int = 1) -> tuple[np.ndarray, int]:
    """Load features from .npy file and optionally downsample.

    Returns:
        feats: (T, C) numpy array (after downsampling)
        original_stride: feature stride after downsampling
    """
    feats = np.load(feat_file).astype(np.float32)  # (T, C)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {feats.shape}")
    # Transpose to (C, T) for ActionFormer
    feats = np.ascontiguousarray(feats.transpose())  # (C, T)
    if downsample_rate > 1:
        feats = feats[:, ::downsample_rate]
    return feats, downsample_rate


@torch.no_grad()
def infer_one_video(
    model: nn.Module,
    video_id: str,
    feats: np.ndarray,
    duration: float,
    feat_stride: int,
    num_frames: int,
    fps: float,
    score_thresh: float = 0.3,
    id_to_label: dict[int, str] | None = None,
) -> list[dict]:
    """Run inference on one video and return filtered detections."""
    feats_tensor = torch.from_numpy(feats)
    video_list = [{
        "video_id": video_id,
        "feats": feats_tensor,
        "segments": torch.zeros((0, 2), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "fps": float(fps),
        "duration": float(duration),
        "feat_stride": int(feat_stride),
        "feat_num_frames": int(num_frames),
    }]

    output = model(video_list)
    if not output:
        return []
    pred = output[0]
    segs = pred["segments"].numpy()
    scores = pred["scores"].numpy()
    labels = pred["labels"].numpy() if "labels" in pred else None
    detections = []
    for i in range(segs.shape[0]):
        score = float(scores[i])
        if score < score_thresh:
            continue
        start_time = max(0.0, float(segs[i, 0]))
        end_time = min(float(duration), float(segs[i, 1]))
        if end_time <= start_time:
            continue
        det = {"start_time": start_time, "end_time": end_time, "score": score}
        if labels is not None:
            class_id = int(labels[i])
            det["class_id"] = class_id
            if id_to_label is not None:
                det["class_name"] = id_to_label.get(class_id, "UNKNOWN")
        detections.append(det)
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections


def main(args):
    """Main inference loop."""
    print(f"[Inference] Config: {args.config}")
    print(f"[Inference] Checkpoint: {args.ckpt}")
    print(f"[Inference] Feature dir: {args.feat_dir}")

    # Load config
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = load_config(args.config)
    id_to_label = _load_combined_id_to_label(cfg)
    print(f"[Inference] Loaded config: {cfg['dataset_name']}")

    _ = fix_random_seed(0, include_cuda=torch.cuda.is_available())
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"[Inference] Using device: {device}")

    # Load model
    model = load_model(cfg, args.ckpt, args.device, torch.cuda.is_available())
    print("[Inference] Model loaded and set to eval mode")

    # Get video IDs
    if args.video_ids:
        video_ids = args.video_ids
    else:
        video_ids = get_video_ids_from_dir(args.feat_dir)
    print(f"[Inference] Found {len(video_ids)} videos to process")

    # Inference loop
    results = {}
    feat_stride_base = cfg["dataset"].get("feat_stride", 2)
    num_frames = cfg["dataset"].get("num_frames", 16)
    fps = float(cfg["dataset"].get("default_fps", 10.0))
    downsample_rate = cfg["dataset"].get("downsample_rate", 1)

    for video_id in video_ids:
        feat_file = os.path.join(args.feat_dir, f"{video_id}.npy")
        if not os.path.exists(feat_file):
            print(f"  [skip] {video_id}: feature file not found")
            continue

        # Load features
        feats, stride_mult = load_features(feat_file, downsample_rate)
        feat_stride = feat_stride_base * stride_mult
        duration = (feats.shape[1] * feat_stride + 0.5 * num_frames) / fps

        # Run inference
        detections = infer_one_video(
            model,
            video_id,
            feats,
            duration,
            feat_stride,
            num_frames,
            fps,
            score_thresh=args.score_thresh,
            id_to_label=id_to_label,
        )

        results[video_id] = {
            "fps": float(fps),
            "duration": float(duration),
            "detections": detections,
        }
        print(
            f"  [{video_id}] {len(detections)} detections "
            f"(score_thresh={args.score_thresh})"
        )

    # Save results
    output_json = args.output_json
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Inference] Results saved to {output_json}")
    print(f"[Inference] Total videos: {len(results)}")
    total_dets = sum(len(r["detections"]) for r in results.values())
    print(f"[Inference] Total detections: {total_dets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for class-agnostic TAL"
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
        help="Path to checkpoint .pth.tar",
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        required=True,
        help="Directory containing .npy feature files",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="detections.json",
        help="Output JSON file for detections",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="*",
        default=None,
        help="Video IDs to process (default: all)",
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
    args = parser.parse_args()
    main(args)
