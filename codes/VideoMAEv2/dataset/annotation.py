"""Annotation IO and time<->step index conversion.

Annotation schema (subset, see CLAUDE.md for the full example):

    {
      "mode": "action_description",
      "video_path": "/.../<stem>_<...>.mp4",
      "video_duration": 10.124,
      "video_fps": 10,
      "analysis_result": {
        "actions": [
          {
            "start_time": 0.0,
            "end_time": 1.0,
            "body_part": "...",
            "action_type": "...",
            "target_object": "...",
            "motion_detail": "...",
            "grip_or_contact": "...",
            "speed_or_force": "...",
            "posture_change": "..."
          },
          ...
        ]
      },
      ...
    }

Step index convention (matches the feature extractor in `extract_features.py`):

    step i corresponds to frames [i * stride, i * stride + window) at target_fps.
    its time span is T_i = [i * stride / target_fps, (i * stride + window) / target_fps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Action:
    """One action segment from analysis_result.actions[]."""

    start_time: float
    end_time: float
    body_part: str = ""
    action_type: str = ""
    target_object: str = ""
    motion_detail: str = ""
    grip_or_contact: str = ""
    speed_or_force: str = ""
    posture_change: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            start_time=float(d["start_time"]),
            end_time=float(d["end_time"]),
            body_part=str(d.get("body_part", "")),
            action_type=str(d.get("action_type", "")),
            target_object=str(d.get("target_object", "")),
            motion_detail=str(d.get("motion_detail", "")),
            grip_or_contact=str(d.get("grip_or_contact", "")),
            speed_or_force=str(d.get("speed_or_force", "")),
            posture_change=str(d.get("posture_change", "")),
        )


@dataclass
class AnnotationRecord:
    """Parsed annotation file."""

    json_path: Path
    video_stem: str
    video_path_in_json: str
    video_duration: float
    video_fps: float
    actions: list[Action] = field(default_factory=list)

    @property
    def num_actions(self) -> int:
        return len(self.actions)


def load_annotation(json_path: str | Path) -> AnnotationRecord:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    actions_raw = d.get("analysis_result", {}).get("actions", []) or []
    actions = [Action.from_dict(a) for a in actions_raw]
    return AnnotationRecord(
        json_path=json_path,
        video_stem=json_path.stem,
        video_path_in_json=str(d.get("video_path", "")),
        video_duration=float(d.get("video_duration", 0.0)),
        video_fps=float(d.get("video_fps", 0.0)),
        actions=actions,
    )


def iter_annotations(annot_dir: str | Path) -> Iterable[Path]:
    """Yield *.json paths under annot_dir (non-recursive, matches the data layout)."""
    annot_dir = Path(annot_dir)
    yield from sorted(annot_dir.glob("*.json"))


@dataclass
class StepGTSegment:
    """Action segment expressed in step index space.

    step_start and step_end are inclusive. If the action does not cover any
    step under the chosen overlap rule, step_start > step_end and step_len = 0.
    """

    action: Action
    step_start: int
    step_end: int

    @property
    def step_len(self) -> int:
        return max(0, self.step_end - self.step_start + 1)


def step_time_span(
    step: int,
    *,
    target_fps: float,
    stride: int,
    window_size: int,
) -> tuple[float, float]:
    """Return (start_time, end_time) in seconds for step index `step`."""
    s = step * stride / target_fps
    e = (step * stride + window_size) / target_fps
    return s, e


def time_to_step(
    t: float,
    *,
    target_fps: float,
    stride: int,
    window_size: int,
    mode: str = "center",
) -> float:
    """Convert a wall-clock time (seconds) to a (possibly fractional) step index.

    mode:
        "start"  : t = i*stride/fps
        "center" : t = (i*stride + window/2)/fps
        "end"    : t = (i*stride + window)/fps
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    f = t * target_fps
    if mode == "start":
        return f / stride
    if mode == "center":
        return (f - window_size / 2.0) / stride
    if mode == "end":
        return (f - window_size) / stride
    raise ValueError(f"unknown mode={mode!r}")


def actions_to_step_segments(
    actions: list[Action],
    *,
    num_steps: int,
    target_fps: float,
    stride: int,
    window_size: int,
    overlap: str = "any",
) -> list[StepGTSegment]:
    """Map action time intervals to inclusive step index ranges.

    overlap:
        "any"    : step is positive if T_i overlaps [start_time, end_time).
        "center" : step is positive if T_i's center is in [start_time, end_time).
        "inside" : step is positive only if T_i is fully in [start_time, end_time).

    Implementation: enumerate all steps. N is small (~140 for 30s @ stride=2),
    so the linear pass is cheap and avoids tricky boundary inversion.
    """
    if num_steps <= 0:
        return []
    if overlap not in {"any", "center", "inside"}:
        raise ValueError(f"unknown overlap={overlap!r}")

    spans = [
        step_time_span(i, target_fps=target_fps, stride=stride, window_size=window_size)
        for i in range(num_steps)
    ]

    out: list[StepGTSegment] = []
    for a in actions:
        s_lo = max(0.0, a.start_time)
        s_hi = max(s_lo, a.end_time)
        i_min = num_steps  # sentinel: "no step matched"
        i_max = -1
        for i, (t_lo, t_hi) in enumerate(spans):
            t_center = 0.5 * (t_lo + t_hi)
            if overlap == "any":
                hit = (t_lo < s_hi) and (t_hi > s_lo)
            elif overlap == "center":
                hit = (s_lo <= t_center < s_hi)
            else:  # "inside"
                hit = (t_lo >= s_lo) and (t_hi <= s_hi)
            if hit:
                if i < i_min:
                    i_min = i
                if i > i_max:
                    i_max = i
        out.append(StepGTSegment(action=a, step_start=i_min, step_end=i_max))
    return out
