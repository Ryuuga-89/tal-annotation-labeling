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

import torch
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
        spec: VideoSpec | None = None,
    ):
        self.video_path = Path(video_path)
        self.cfg = _DatasetConfig(
            target_fps=target_fps,
            window_size=window_size,
            stride=stride,
            input_size=input_size,
            resize_mode=resize_mode,
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
            self._vr = VideoReader(str(self.video_path), num_threads=1, ctx=cpu(0))
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
        return clip.contiguous(), idx


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
