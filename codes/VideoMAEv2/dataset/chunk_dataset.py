"""Per-video sliding-window clip dataset for feature extraction.

Yields fixed-shape clips (one per sliding window step). The Dataset is
"per-video, per-step" so it composes naturally with a DataLoader to batch
multiple steps from the same video.

For multi-video extraction we wrap a list of (video, json) pairs and a
batched extractor in `extract_features.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from .annotation import AnnotationRecord, load_annotation
from .video_loader import VideoSpec, load_video_at_target_fps, probe_video


class ClipBatch(NamedTuple):
    """Default collation output."""

    clips: torch.Tensor  # [B, 3, T, H, W]
    step_idx: torch.Tensor  # [B]


@dataclass
class _DatasetConfig:
    target_fps: float
    window_size: int
    stride: int
    input_size: int
    resize_mode: str
    decode_threads: int
    decode_resize_on_read: bool


class VideoChunkDataset(Dataset):
    """Sliding-window clips for a single video.

    Each item is one (clip_tensor, step_idx) pair. Length equals the number of
    sliding-window steps after target-fps resampling.

    Note: a fresh `VideoReader` is created in `__getitem__` because decord's
    reader is not safe to share across DataLoader worker processes. For the
    single-process path we still amortize by caching one reader on the dataset.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        target_fps: float = 10.0,
        window_size: int = 16,
        stride: int = 2,
        input_size: int = 224,
        resize_mode: str = "squash",
        decode_threads: int = 2,
        decode_resize_on_read: bool = True,
        spec: VideoSpec | None = None,
    ):
        self.video_path = Path(video_path)
        self.cfg = _DatasetConfig(
            target_fps=target_fps,
            window_size=window_size,
            stride=stride,
            input_size=input_size,
            resize_mode=resize_mode,
            decode_threads=decode_threads,
            decode_resize_on_read=decode_resize_on_read,
        )
        self.spec = spec or probe_video(self.video_path, target_fps=target_fps)
        self._n_steps = self.spec.num_steps(window_size=window_size, stride=stride)
        self._vr: VideoReader | None = None
        self._video_tensor: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._n_steps

    @property
    def num_steps(self) -> int:
        return self._n_steps

    def _get_reader(self) -> VideoReader:
        if self._vr is None:
            # For squash mode we can decode directly at model resolution and skip
            # expensive CPU-side interpolation of full-resolution frames.
            if self.cfg.resize_mode == "squash" and self.cfg.decode_resize_on_read:
                self._vr = VideoReader(
                    str(self.video_path),
                    num_threads=max(1, int(self.cfg.decode_threads)),
                    width=int(self.cfg.input_size),
                    height=int(self.cfg.input_size),
                    ctx=cpu(0),
                )
            else:
                self._vr = VideoReader(
                    str(self.video_path),
                    num_threads=max(1, int(self.cfg.decode_threads)),
                    ctx=cpu(0),
                )
        return self._vr

    def _get_video_tensor(self) -> torch.Tensor:
        if self._video_tensor is None:
            self._video_tensor = load_video_at_target_fps(
                self.spec,
                input_size=self.cfg.input_size,
                resize_mode=self.cfg.resize_mode,
                vr=self._get_reader(),
            )
        return self._video_tensor

    def release_cache(self) -> None:
        """Release cached reader/tensor eagerly to avoid resource buildup."""
        self._video_tensor = None
        self._vr = None

    def batch_clips(self, start_step: int, end_step: int) -> torch.Tensor:
        """Return clips for step range [start_step, end_step) as [B,3,T,H,W].

        This decodes only frames needed by the batch window span instead of
        materializing the full target-fps video tensor.
        """
        if start_step < 0 or end_step > self._n_steps or end_step <= start_step:
            raise ValueError(f"invalid step range: {start_step}:{end_step} for n_steps={self._n_steps}")

        # Fallback path for non-squash mode: keep behavior simple/correct.
        if self.cfg.resize_mode != "squash":
            return self.windowed_clips()[start_step:end_step]

        t_start = start_step * self.cfg.stride
        t_end = (end_step - 1) * self.cfg.stride + self.cfg.window_size  # exclusive

        target_idx = np.arange(t_start, t_end, dtype=np.float64)
        if self.spec.src_fps <= 0:
            src_idx = target_idx.astype(np.int64)
        else:
            src_idx = np.round(target_idx * (self.spec.src_fps / self.spec.target_fps)).astype(np.int64)
        src_idx = np.clip(src_idx, 0, self.spec.src_num_frames - 1)

        # Decode unique source frames once, then remap to target timeline.
        unique_src, inverse = np.unique(src_idx, return_inverse=True)
        frames = self._get_reader().get_batch(unique_src)  # [U,H,W,3] uint8 torch tensor
        t = frames.to(torch.float32) / 255.0  # [U,H,W,3]
        t = t.permute(0, 3, 1, 2)  # [U,3,H,W]
        if not (
            self.cfg.decode_resize_on_read
            and int(t.shape[-2]) == int(self.cfg.input_size)
            and int(t.shape[-1]) == int(self.cfg.input_size)
        ):
            t = F.interpolate(
                t,
                size=(self.cfg.input_size, self.cfg.input_size),
                mode="bilinear",
                align_corners=False,
            )
        # VideoMAE normalization
        t = (t - 0.5) / 0.5

        seq = t[inverse]  # [L,3,H,W], L == (t_end - t_start)
        seq = seq.permute(1, 0, 2, 3).contiguous()  # [3,L,H,W]
        clips = seq.unfold(1, self.cfg.window_size, self.cfg.stride).permute(1, 0, 4, 2, 3)
        return clips

    def windowed_clips(self) -> torch.Tensor:
        """Return all sliding-window clips as a view of shape [N, 3, T, H, W]."""
        video_tensor = self._get_video_tensor()
        return video_tensor.unfold(1, self.cfg.window_size, self.cfg.stride).permute(1, 0, 4, 2, 3)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx < 0 or idx >= self._n_steps:
            raise IndexError(idx)
        start_target_idx = idx * self.cfg.stride
        video_tensor = self._get_video_tensor()
        clip = video_tensor[:, start_target_idx : start_target_idx + self.cfg.window_size]
        # Keep as a view to avoid an extra CPU copy per step; batching handles packing.
        return clip, idx


def collate_clips(items: list[tuple[torch.Tensor, int]]) -> ClipBatch:
    clips = torch.stack([c for c, _ in items], dim=0)
    step_idx = torch.tensor([i for _, i in items], dtype=torch.long)
    return ClipBatch(clips=clips, step_idx=step_idx)


def pair_video_with_annotation(
    video_root: str | Path,
    annotation_json: str | Path,
    *,
    suffix: str = ".mp4",
) -> tuple[Path, AnnotationRecord]:
    """Resolve `<video_root>/<stem><suffix>` for a given annotation json.

    The CLAUDE.md layout has annotations at .../annot/.../<stem>.json and videos
    at .../30s_chunks/<stem>.mp4 (matching stems). The `video_path` field inside
    the json points to a different (intermediate analysis) location and is
    therefore *not* used here.
    """
    rec = load_annotation(annotation_json)
    video_path = Path(video_root) / f"{rec.video_stem}{suffix}"
    return video_path, rec
