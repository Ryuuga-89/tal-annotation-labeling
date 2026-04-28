"""Aggregate per-action analysis rows into reports/annotation_analyze.csv.

One row per action segment. Per-video aggregates (n_actions, dominant_*, etc.)
are denormalized onto every row of the same video so the CSV can be loaded as
a single dataframe. Step indices follow the extractor convention
(target_fps=10, window=16, stride=2 by default).

No video probing — purely JSON-based.

Run:
    PYTHONPATH=codes uv run python -m VideoMAEv2.tools.build_annotation_csv \\
        --annot-dir "$ANNOT_ROOT_DIR" \\
        --out-csv   reports/annotation_analyze.csv \\
        [--target-fps 10 --window-size 16 --stride 2]
        [--limit 100]   # debug
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[2]  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from VideoMAEv2.dataset.annotation import (  # noqa: E402
    Action,
    actions_to_step_segments,
    load_annotation,
)


# ---------------------------------------------------------------------------
# Helpers


_DIGIT_RE = re.compile(r"[0-9０-９]")
# Stem looks like: <hash>_<HH-MM-SS.mmm>_<HH-MM-SS.mmm>_<idx>_<dur>
_STEM_TAIL_RE = re.compile(
    r"^(?P<group>[^_]+)_"
    r"(?P<start_hms>\d{2}-\d{2}-\d{2}(?:\.\d+)?)_"
    r"(?P<end_hms>\d{2}-\d{2}-\d{2}(?:\.\d+)?)_"
    r"(?P<idx>\d+)_(?P<dur>\d+)$"
)

# Heuristic threshold: Gemini 2.5 Flash typical max output ~8192 tokens.
_NEAR_OUTPUT_TOKEN_LIMIT = 8000


def parse_stem(stem: str) -> dict:
    """Return {group_key, chunk_start_in_source_sec, chunk_idx_in_source_label,
    chunk_dur_label}. Falls back gracefully on unexpected stems."""
    m = _STEM_TAIL_RE.match(stem)
    if not m:
        return {
            "group_key": stem.split("_", 1)[0],
            "chunk_start_in_source_sec": "",
            "chunk_idx_in_source_label": "",
            "chunk_dur_label": "",
        }
    hh, mm, ss = m.group("start_hms").split("-")
    start_sec = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return {
        "group_key": m.group("group"),
        "chunk_start_in_source_sec": round(start_sec, 3),
        "chunk_idx_in_source_label": int(m.group("idx")),
        "chunk_dur_label": int(m.group("dur")),
    }


def is_missing(s: str) -> bool:
    return (s is None) or (s.strip() == "")


def has_digit(s: str) -> bool:
    return bool(_DIGIT_RE.search(s or ""))


def num_target_frames(duration: float, target_fps: float) -> int:
    n = int(duration * target_fps)
    return max(0, n)


def num_steps(num_frames: int, window_size: int, stride: int) -> int:
    if num_frames < window_size:
        return 0
    return (num_frames - window_size) // stride + 1


# ---------------------------------------------------------------------------
# Column schema (kept explicit so the CSV header is stable & human-readable).


COLUMNS: tuple[str, ...] = (
    # ---- identity ----
    "json_filename",
    "video_stem",
    "group_key",
    "chunk_start_in_source_sec",
    "chunk_idx_in_source_label",
    "chunk_dur_label",
    "n_chunks_in_group",
    "action_idx",
    # ---- per-video meta (denormalized) ----
    "mode",
    "video_duration_meta",
    "video_fps_meta",
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
    "near_output_token_limit",
    # ---- per-video aggregates (denormalized) ----
    "n_actions",
    "total_action_duration",
    "coverage_ratio",
    "mean_action_duration",
    "max_action_duration",
    "min_action_duration",
    "dominant_action_type",
    "dominant_action_type_share",
    "dominant_body_part",
    "dominant_body_part_share",
    "n_unique_action_types",
    "n_unique_body_parts",
    "num_target_frames",
    "num_steps",
    "pos_step_ratio_any",
    # ---- per-action raw ----
    "start_time",
    "end_time",
    "duration",
    "body_part",
    "action_type",
    "target_object",
    "motion_detail",
    "grip_or_contact",
    "speed_or_force",
    "posture_change",
    # ---- per-action derived ----
    "motion_detail_char_len",
    "motion_detail_has_number",
    "target_object_char_len",
    "body_part_is_missing",
    "action_type_is_missing",
    "target_object_is_missing",
    "grip_or_contact_is_missing",
    "speed_or_force_is_missing",
    "posture_change_is_missing",
    "posture_change_is_no_change",
    # ---- step indices ----
    "step_start_any",
    "step_end_any",
    "step_count_any",
    "step_start_center",
    "step_end_center",
    "step_count_center",
    "is_too_short_for_step",
    # ---- sanity flags ----
    "is_zero_duration",
    "is_negative_duration",
    "out_of_video",
    "overlaps_prev",
    "overlaps_next",
    "gap_to_prev_sec",
    "gap_to_next_sec",
)


# ---------------------------------------------------------------------------
# Parsing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", type=str, required=True)
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--target-fps", type=float, default=10.0)
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV file encoding. utf-8-sig = UTF-8 BOM (Excel-friendly).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if args.limit is not None:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"no annotation json under {annot_dir}")

    # Pass 1: count chunks per group_key (cheap — filenames only).
    chunks_per_group: dict[str, int] = defaultdict(int)
    for jp in json_paths:
        chunks_per_group[parse_stem(jp.stem)["group_key"]] += 1

    n_rows = 0
    n_videos = 0
    n_skipped = 0

    with out_csv.open("w", encoding=args.encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="raise")
        writer.writeheader()

        for jp in tqdm(json_paths, desc="aggregate", dynamic_ncols=True):
            try:
                raw = json.loads(jp.read_text(encoding="utf-8"))
                rec = load_annotation(jp)
            except Exception as e:
                n_skipped += 1
                tqdm.write(f"[skip] {jp.name}: {e}")
                continue

            stem_info = parse_stem(rec.video_stem)
            usage = raw.get("usage_metadata") or {}
            cand_tokens = int(usage.get("candidates_token_count") or 0)

            # ---- per-video aggregates ----
            actions: list[Action] = rec.actions
            n_actions = len(actions)
            durations = [max(0.0, a.end_time - a.start_time) for a in actions]
            total_dur = sum(durations)
            video_dur = float(rec.video_duration or 0.0)

            type_counter: Counter[str] = Counter(
                a.action_type for a in actions if not is_missing(a.action_type)
            )
            part_counter: Counter[str] = Counter(
                a.body_part for a in actions if not is_missing(a.body_part)
            )
            dom_type, dom_type_n = (type_counter.most_common(1) or [("", 0)])[0]
            dom_part, dom_part_n = (part_counter.most_common(1) or [("", 0)])[0]

            ntf = num_target_frames(video_dur, args.target_fps)
            ns = num_steps(ntf, args.window_size, args.stride)

            # any-overlap and center-overlap step segments for the whole video
            segs_any = actions_to_step_segments(
                actions,
                num_steps=ns,
                target_fps=args.target_fps,
                stride=args.stride,
                window_size=args.window_size,
                overlap="any",
            )
            segs_center = actions_to_step_segments(
                actions,
                num_steps=ns,
                target_fps=args.target_fps,
                stride=args.stride,
                window_size=args.window_size,
                overlap="center",
            )
            # union of any-overlap step ranges -> positive-step ratio
            pos_steps: set[int] = set()
            for s in segs_any:
                if s.step_len > 0:
                    pos_steps.update(range(s.step_start, s.step_end + 1))
            pos_step_ratio_any = (len(pos_steps) / ns) if ns > 0 else 0.0

            video_row = {
                "json_filename": jp.name,
                "video_stem": rec.video_stem,
                "group_key": stem_info["group_key"],
                "chunk_start_in_source_sec": stem_info["chunk_start_in_source_sec"],
                "chunk_idx_in_source_label": stem_info["chunk_idx_in_source_label"],
                "chunk_dur_label": stem_info["chunk_dur_label"],
                "n_chunks_in_group": chunks_per_group[stem_info["group_key"]],
                # per-video meta
                "mode": str(raw.get("mode", "")),
                "video_duration_meta": video_dur,
                "video_fps_meta": float(rec.video_fps or 0.0),
                "prompt_token_count": int(usage.get("prompt_token_count") or 0),
                "candidates_token_count": cand_tokens,
                "thoughts_token_count": int(usage.get("thoughts_token_count") or 0),
                "total_token_count": int(usage.get("total_token_count") or 0),
                "near_output_token_limit": int(cand_tokens >= _NEAR_OUTPUT_TOKEN_LIMIT),
                # per-video aggregates
                "n_actions": n_actions,
                "total_action_duration": round(total_dur, 4),
                "coverage_ratio": round(total_dur / video_dur, 4) if video_dur > 0 else "",
                "mean_action_duration": round(total_dur / n_actions, 4) if n_actions else "",
                "max_action_duration": round(max(durations), 4) if durations else "",
                "min_action_duration": round(min(durations), 4) if durations else "",
                "dominant_action_type": dom_type,
                "dominant_action_type_share": (
                    round(dom_type_n / n_actions, 4) if n_actions else ""
                ),
                "dominant_body_part": dom_part,
                "dominant_body_part_share": (
                    round(dom_part_n / n_actions, 4) if n_actions else ""
                ),
                "n_unique_action_types": len(type_counter),
                "n_unique_body_parts": len(part_counter),
                "num_target_frames": ntf,
                "num_steps": ns,
                "pos_step_ratio_any": round(pos_step_ratio_any, 4),
            }

            n_videos += 1
            if n_actions == 0:
                # Emit a single placeholder row so the video is still represented.
                row = dict(video_row)
                row.update({c: "" for c in COLUMNS if c not in video_row})
                row["action_idx"] = -1
                writer.writerow(row)
                n_rows += 1
                continue

            # ---- per-action rows ----
            for i, a in enumerate(actions):
                seg_any = segs_any[i]
                seg_ctr = segs_center[i]
                dur = max(0.0, a.end_time - a.start_time)
                gap_prev = (
                    round(a.start_time - actions[i - 1].end_time, 4) if i > 0 else ""
                )
                gap_next = (
                    round(actions[i + 1].start_time - a.end_time, 4)
                    if i < n_actions - 1
                    else ""
                )

                row = dict(video_row)
                row.update(
                    {
                        "action_idx": i,
                        "start_time": round(a.start_time, 4),
                        "end_time": round(a.end_time, 4),
                        "duration": round(dur, 4),
                        "body_part": a.body_part,
                        "action_type": a.action_type,
                        "target_object": a.target_object,
                        "motion_detail": a.motion_detail,
                        "grip_or_contact": a.grip_or_contact,
                        "speed_or_force": a.speed_or_force,
                        "posture_change": a.posture_change,
                        # derived
                        "motion_detail_char_len": len(a.motion_detail or ""),
                        "motion_detail_has_number": int(has_digit(a.motion_detail)),
                        "target_object_char_len": len(a.target_object or ""),
                        "body_part_is_missing": int(is_missing(a.body_part)),
                        "action_type_is_missing": int(is_missing(a.action_type)),
                        "target_object_is_missing": int(is_missing(a.target_object)),
                        "grip_or_contact_is_missing": int(is_missing(a.grip_or_contact)),
                        "speed_or_force_is_missing": int(is_missing(a.speed_or_force)),
                        "posture_change_is_missing": int(is_missing(a.posture_change)),
                        "posture_change_is_no_change": int(
                            (a.posture_change or "").strip() == "変化なし"
                        ),
                        # step indices
                        "step_start_any": seg_any.step_start if seg_any.step_len > 0 else "",
                        "step_end_any": seg_any.step_end if seg_any.step_len > 0 else "",
                        "step_count_any": seg_any.step_len,
                        "step_start_center": (
                            seg_ctr.step_start if seg_ctr.step_len > 0 else ""
                        ),
                        "step_end_center": (
                            seg_ctr.step_end if seg_ctr.step_len > 0 else ""
                        ),
                        "step_count_center": seg_ctr.step_len,
                        "is_too_short_for_step": int(seg_any.step_len == 0),
                        # sanity
                        "is_zero_duration": int(dur == 0.0),
                        "is_negative_duration": int(a.end_time < a.start_time),
                        "out_of_video": int(
                            (a.start_time < 0)
                            or (video_dur > 0 and a.end_time > video_dur + 1e-6)
                        ),
                        "overlaps_prev": int(
                            i > 0 and a.start_time < actions[i - 1].end_time - 1e-6
                        ),
                        "overlaps_next": int(
                            i < n_actions - 1
                            and a.end_time > actions[i + 1].start_time + 1e-6
                        ),
                        "gap_to_prev_sec": gap_prev,
                        "gap_to_next_sec": gap_next,
                    }
                )
                writer.writerow(row)
                n_rows += 1

    print(
        f"[build_annotation_csv] videos={n_videos} skipped={n_skipped} "
        f"rows={n_rows} -> {out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
