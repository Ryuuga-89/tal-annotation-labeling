"""Generate a tiny synthetic video + annotation for smoke-testing the pipeline.

Usage:
    uv run python codes/VideoMAEv2/tests/make_smoke_data.py <out_dir>

Produces:
    <out_dir>/video/smoke_001.mp4   (10s, 30fps, 320x240, moving rectangle)
    <out_dir>/annot/smoke_001.json  (3 fake action segments)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(out_root: str) -> None:
    out_root = Path(out_root)
    (out_root / "video").mkdir(parents=True, exist_ok=True)
    (out_root / "annot").mkdir(parents=True, exist_ok=True)

    stem = "smoke_001"
    video_path = out_root / "video" / f"{stem}.mp4"
    fps = 30
    sec = 10
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    n = fps * sec
    for i in range(n):
        img = np.full((h, w, 3), 32, dtype=np.uint8)
        x = int(20 + (w - 60) * (i / max(1, n - 1)))
        cv2.rectangle(img, (x, 90), (x + 40, 150), (200, 30, 30), -1)
        vw.write(img)
    vw.release()

    ann = {
        "mode": "action_description",
        "video_path": str(video_path),
        "video_duration": float(sec),
        "video_fps": float(fps),
        "analysis_result": {
            "actions": [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "body_part": "右手",
                    "action_type": "指差す",
                    "target_object": "左",
                    "motion_detail": "synthetic-A",
                    "grip_or_contact": "非接触",
                    "speed_or_force": "通常",
                    "posture_change": "無",
                },
                {
                    "start_time": 4.5,
                    "end_time": 6.2,
                    "body_part": "右足",
                    "action_type": "押す",
                    "target_object": "ペダル",
                    "motion_detail": "synthetic-B",
                    "grip_or_contact": "面接触",
                    "speed_or_force": "力強く",
                    "posture_change": "底屈",
                },
                {
                    "start_time": 8.5,
                    "end_time": 9.2,
                    "body_part": "右手",
                    "action_type": "滑らせる",
                    "target_object": "太もも",
                    "motion_detail": "synthetic-C",
                    "grip_or_contact": "接触",
                    "speed_or_force": "軽く",
                    "posture_change": "無",
                },
            ]
        },
        "usage_metadata": {},
    }
    ann_path = out_root / "annot" / f"{stem}.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    print(f"wrote {video_path} and {ann_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_smoke_data.py <out_dir>")
    main(sys.argv[1])
