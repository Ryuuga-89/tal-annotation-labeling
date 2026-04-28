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
from typing import Any

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
from actionformer_libs.utils import nms_1d_cpu  # noqa: E402, F401


def get_video_ids_from_dir(feat_dir: str) -> list[str]:
    """List all .npy files in feat_dir, return stems (without extension)."""
    feat_dir = Path(feat_dir)
    stems = []
    for npy_file in sorted(feat_dir.glob("*.npy")):
        stems.append(npy_file.stem)
    return stems


def load_model(cfg: dict, ckpt_path: str, device: str) -> nn.Module:
    """Load ActionFormer model from config and checkpoint."""
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=[int(device)])
    model.to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(
        ckpt_path,
        map_location=lambda storage, loc: storage.cuda(int(device)),
    )
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


def nms_1d_cpu_wrapper(
    detections: list[dict],
    nms_thresh: float,
) -> list[dict]:
    """Apply NMS on 1D detections using actionformer_libs.utils.nms_1d_cpu.

    Args:
        detections: List of dicts with 'start_time', 'end_time', 'score'.
        nms_thresh: IoU threshold for NMS.

    Returns:
        Filtered detections after NMS.
    """
    if not detections:
        return []

    # Convert to (N, 3) array: [start, end, score]
    dets = np.asarray(
        [[d["start_time"], d["end_time"], d["score"]] for d in detections],
        dtype=np.float32,
    )

    # nms_1d_cpu expects (N, 2) for positions and (N,) for scores separately
    # Let's use a simple IoU-based NMS instead (ActionFormer's NMS is 2D)
    keep = soft_nms_1d(dets, nms_thresh)
    return [detections[i] for i in keep]


def soft_nms_1d(dets: np.ndarray, iou_thresh: float) -> list[int]:
    """Soft-NMS for 1D temporal segments.

    Args:
        dets: (N, 3) array [start, end, score]
        iou_thresh: IoU threshold

    Returns:
        List of indices to keep
    """
    if dets.size == 0:
        return []

    dets = dets.copy()
    keep = []

    # Sort by score descending
    order = np.argsort(-dets[:, 2])

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with all remaining detections
        start = np.maximum(dets[i, 0], dets[order[1:], 0])
        end = np.minimum(dets[i, 1], dets[order[1:], 1])
        inter = np.maximum(0, end - start)
        union = (dets[i, 1] - dets[i, 0]) + (dets[order[1:], 1] - dets[order[1:], 0]) - inter
        iou = inter / (union + 1e-6)

        # Remove detections with high IoU
        mask = iou <= iou_thresh
        order = order[1:][mask]

    return keep


@torch.no_grad()
def infer_batch(
    model: nn.Module,
    feats_tensor: torch.Tensor,
    feat_stride: int,
    num_frames: int,
    fps: float,
    score_thresh: float = 0.3,
) -> list[dict]:
    """Run inference on a single sample and extract detections.

    Args:
        model: ActionFormer model in eval mode
        feats_tensor: (1, C, T) tensor (batch size = 1)
        feat_stride: stride between feature frames
        num_frames: window size used for feature extraction
        fps: original video FPS
        score_thresh: confidence threshold for filtering

    Returns:
        List of dicts: {'start_time', 'end_time', 'score'}
    """
    output = model(feats_tensor)
    # output should be (B, 2, H) where:
    #   output[:, 0, :] = regression (start/end offsets in feat-grid)
    #   output[:, 1, :] = classification scores

    # Assuming ActionFormer outputs (B, 2, T_out) or similar
    # For class-agnostic, we expect 2 channels: regression + 1 class
    # This depends on ActionFormer's head implementation.

    # Typical ActionFormer output:
    # - output['cls_logits']: (B, num_classes, T)
    # - output['reg_preds']: (B, 2, T)  [start/end offsets]
    # OR model outputs dict with these keys

    if isinstance(output, dict):
        cls_logits = output.get("cls_logits")  # (B, num_classes, T)
        reg_preds = output.get("reg_preds")  # (B, 2, T)
    else:
        # Assume tuple: (cls_logits, reg_preds) or similar
        # For class-agnostic, might be (B, 1, T) and (B, 2, T)
        # This is model-dependent; adjust as needed
        raise NotImplementedError(
            "Model output type not recognized. Expecting dict with "
            "'cls_logits' and 'reg_preds' keys."
        )

    if cls_logits is None or reg_preds is None:
        return []

    # Extract from batch (assume B=1)
    cls_logits = cls_logits[0]  # (num_classes, T)
    reg_preds = reg_preds[0]  # (2, T)

    # For class-agnostic (num_classes=1), cls_logits is (1, T)
    scores = torch.sigmoid(cls_logits[0])  # (T,)

    # Find peaks above threshold
    mask = scores > score_thresh
    if not mask.any():
        return []

    # Extract temporal positions
    feat_offset = 0.5 * num_frames / feat_stride
    indices = torch.where(mask)[0].cpu().numpy()
    detections = []

    for idx in indices:
        # Regression prediction: [start_offset, end_offset]
        start_offset, end_offset = reg_preds[:, idx].cpu().numpy()

        # Convert to feature-grid coordinates (assuming regression is delta-based)
        # If regression is absolute offsets from center:
        start_grid = idx + start_offset
        end_grid = idx + end_offset

        # Clamp to valid range
        start_grid = max(0, float(start_grid))
        end_grid = min(float(cls_logits.shape[1]), float(end_grid))

        # Convert from feature-grid to seconds
        start_time = (start_grid + feat_offset) * feat_stride / fps
        end_time = (end_grid + feat_offset) * feat_stride / fps

        score = float(scores[idx])

        detections.append(
            {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "score": float(score),
            }
        )

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
    print(f"[Inference] Loaded config: {cfg['dataset_name']}")

    # Determine device
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"[Inference] Using device: {device}")

    # Load model
    model = load_model(cfg, args.ckpt, args.device)
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
    fps = cfg["dataset"].get("default_fps", 10.0)
    downsample_rate = cfg["dataset"].get("downsample_rate", 1)

    for video_id in video_ids:
        feat_file = os.path.join(args.feat_dir, f"{video_id}.npy")
        if not os.path.exists(feat_file):
            print(f"  [skip] {video_id}: feature file not found")
            continue

        # Load features
        feats, stride_mult = load_features(feat_file, downsample_rate)
        feat_stride = feat_stride_base * stride_mult
        feats_tensor = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1, C, T)

        # Run inference
        detections = infer_batch(
            model,
            feats_tensor,
            feat_stride,
            num_frames,
            fps,
            score_thresh=args.score_thresh,
        )

        # Apply NMS
        if detections:
            detections = nms_1d_cpu_wrapper(detections, args.nms_thresh)

        results[video_id] = {
            "fps": float(fps),
            "detections": detections,
        }
        print(
            f"  [{video_id}] {len(detections)} detections "
            f"(before NMS: ~{len(detections)})"
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
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for NMS",
    )

    args = parser.parse_args()
    main(args)
