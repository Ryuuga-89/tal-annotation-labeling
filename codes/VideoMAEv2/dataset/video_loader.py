"""Video IO that resamples to a fixed target fps and applies VideoMAE preprocessing.

The fps of source videos is heterogeneous (10 fps, 30 fps, ...). To keep the
feature time axis comparable across all videos, we sample frames at evenly
spaced timestamps that correspond to `target_fps`, regardless of source fps.

Spatial preprocessing follows the official `extract_tad_feature.py` recipe:
direct Resize((224, 224)) (squash). Aspect ratio is intentionally not
preserved, matching the K710 fine-tuning preprocessing of VideoMAE V2 so the
backbone sees the same distribution it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import decord
from decord import VideoReader, cpu

# AVION-style optimization: keep decoding output as torch tensor on CPU,
# avoiding repeated numpy->torch conversions.
decord.bridge.set_bridge("torch")


@dataclass
class VideoSpec:
    """Static info about a video, after target-fps resampling."""

    path: Path
    src_num_frames: int
    src_fps: float
    duration_sec: float
    target_fps: float

    @property
    def num_target_frames(self) -> int:
        # Use floor: never sample past EOF.
        return int(np.floor(self.duration_sec * self.target_fps))

    def num_steps(self, *, window_size: int, stride: int) -> int:
        """Number of sliding-window clips for the resampled video."""
        n = self.num_target_frames
        if n < window_size:
            return 0
        return (n - window_size) // stride + 1


def probe_video(video_path: str | Path, *, target_fps: float) -> VideoSpec:
    """Open the video briefly to read length / fps without loading all frames."""
    video_path = Path(video_path)
    vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    src_num_frames = len(vr)
    src_fps = float(vr.get_avg_fps())
    duration_sec = src_num_frames / src_fps if src_fps > 0 else 0.0
    return VideoSpec(
        path=video_path,
        src_num_frames=src_num_frames,
        src_fps=src_fps,
        duration_sec=duration_sec,
        target_fps=target_fps,
    )


def _target_to_source_indices(
    spec: VideoSpec,
    start_target_idx: int,
    n: int,
) -> np.ndarray:
    """Map n consecutive target-fps indices (starting at start_target_idx) to
    nearest source frame indices."""
    target_idx = np.arange(start_target_idx, start_target_idx + n, dtype=np.float64)
    # target time t = i / target_fps
    # source frame = round(t * src_fps)
    if spec.src_fps <= 0:
        # Pathological: assume 1:1.
        src_idx = target_idx.astype(np.int64)
    else:
        src_idx = np.round(target_idx * (spec.src_fps / spec.target_fps)).astype(np.int64)
    src_idx = np.clip(src_idx, 0, spec.src_num_frames - 1)
    return src_idx


def _resize_squash(frames_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """frames_uint8: [T, H, W, 3] uint8 tensor -> [3, T, size, size] float."""
    t = frames_uint8.to(torch.float32) / 255.0  # [T,H,W,3]
    t = t.permute(0, 3, 1, 2)  # [T,3,H,W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.permute(1, 0, 2, 3).contiguous()  # [3,T,size,size]
    return t


def _resize_shortside_then_center_crop(
    frames_uint8: torch.Tensor, size: int
) -> torch.Tensor:
    t = frames_uint8.to(torch.float32) / 255.0  # [T,H,W,3]
    t = t.permute(0, 3, 1, 2)  # [T,3,H,W]
    _, _, h, w = t.shape
    if h <= w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # center crop
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    t = t[:, :, top : top + size, left : left + size]
    t = t.permute(1, 0, 2, 3).contiguous()  # [3,T,size,size]
    return t


def load_target_fps_frames(
    spec: VideoSpec,
    *,
    start_target_idx: int,
    num_frames: int,
    vr: VideoReader | None = None,
) -> torch.Tensor:
    """Load `num_frames` consecutive frames on the target-fps timeline.

    The returned tensor keeps the original uint8 layout [T, H, W, 3].
    """
    if vr is None:
        vr = VideoReader(str(spec.path), num_threads=1, ctx=cpu(0))
    src_idx = _target_to_source_indices(spec, start_target_idx, num_frames)
    # Decord can become unstable/slow for very large index arrays on some videos.
    # Read in chunks to bound per-call latency and memory.
    chunk_size = 256
    if src_idx.shape[0] <= chunk_size:
        return vr.get_batch(src_idx)

    parts = []
    for i in range(0, src_idx.shape[0], chunk_size):
        part = vr.get_batch(src_idx[i : i + chunk_size])
        parts.append(part)
    return torch.cat(parts, dim=0)


def load_video_at_target_fps(
    spec: VideoSpec,
    *,
    input_size: int = 224,
    resize_mode: str = "squash",
    normalize: bool = True,
    vr: VideoReader | None = None,
) -> torch.Tensor:
    """Load the whole video once at target fps and return [3, T, H, W]."""
    frames = load_target_fps_frames(
        spec,
        start_target_idx=0,
        num_frames=spec.num_target_frames,
        vr=vr,
    )
    if resize_mode == "squash":
        clip = _resize_squash(frames, input_size)
    elif resize_mode == "shortside_crop":
        clip = _resize_shortside_then_center_crop(frames, input_size)
    else:
        raise ValueError(f"unknown resize_mode={resize_mode!r}")
    if normalize:
        clip = (clip - 0.5) / 0.5
    return clip


def load_clip_at_target_fps(
    spec: VideoSpec,
    *,
    start_target_idx: int,
    window_size: int,
    input_size: int = 224,
    resize_mode: str = "squash",
    vr: VideoReader | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load a single clip aligned to target fps and produce a model-ready tensor.

    Returns a tensor of shape (3, window_size, input_size, input_size).
    `normalize=True` applies VideoMAE's mean=std=0.5 normalization.
    """
    frames = load_target_fps_frames(
        spec,
        start_target_idx=start_target_idx,
        num_frames=window_size,
        vr=vr,
    )
    if resize_mode == "squash":
        clip = _resize_squash(frames, input_size)
    elif resize_mode == "shortside_crop":
        clip = _resize_shortside_then_center_crop(frames, input_size)
    else:
        raise ValueError(f"unknown resize_mode={resize_mode!r}")
    if normalize:
        # VideoMAE pretrain/finetune normalization: mean=std=0.5 per channel.
        clip = (clip - 0.5) / 0.5
    return clip
