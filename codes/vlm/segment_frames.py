"""Sample RGB frames from a video clip using decord (native wall-clock times)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def sample_frames_equidistant(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    *,
    decode_threads: int = 2,
) -> list[Image.Image]:
    """Take ``num_frames`` frames at equally spaced times in ``[start_sec, end_sec]``.

    Frame indices use ``round(time * src_fps)`` clipped to valid range.
    """
    if num_frames < 1 or end_sec < start_sec:
        return []

    video_path = Path(video_path)
    vr = VideoReader(str(video_path), num_threads=max(1, int(decode_threads)), ctx=cpu(0))
    n = len(vr)
    if n <= 0:
        return []

    fps = float(vr.get_avg_fps())
    if fps <= 0:
        fps = 25.0

    times = np.linspace(float(start_sec), float(end_sec), num=int(num_frames), dtype=np.float64)
    idxs: list[int] = []
    for t in times:
        fi = int(round(float(t) * fps))
        fi = max(0, min(n - 1, fi))
        idxs.append(fi)

    batch = vr.get_batch(np.asarray(idxs, dtype=np.int32))
    if hasattr(batch, "asnumpy"):
        arr = batch.asnumpy()
    else:
        arr = np.asarray(batch)

    out: list[Image.Image] = []
    for i in range(arr.shape[0]):
        out.append(Image.fromarray(arr[i]))
    return out
