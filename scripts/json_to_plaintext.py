#!/usr/bin/env python3
"""
Convert TAL/caption JSON output into plain-text lines.

Default line format:
    {start_time:.3f}-{end_time:.3f}: {text}

If `motion_detail` exists in each detection item, it is used as text.
Otherwise score is used as fallback text.
"""
import argparse
import json
from pathlib import Path


def _line_for_detection(det: dict, text_key: str) -> str:
    start = float(det.get("start_time", 0.0))
    end = float(det.get("end_time", 0.0))
    if text_key in det and det[text_key]:
        text = str(det[text_key])
    elif "score" in det:
        text = f"score={float(det['score']):.4f}"
    else:
        text = ""
    return f"{start:.3f}-{end:.3f}: {text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TAL JSON to plain text")
    parser.add_argument("--input-json", type=str, required=True, help="Input JSON path")
    parser.add_argument("--output-txt", type=str, required=True, help="Output text path")
    parser.add_argument(
        "--text-key",
        type=str,
        default="motion_detail",
        help="Detection key used as caption text",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="",
        help="Optional single video_id filter",
    )
    args = parser.parse_args()

    in_path = Path(args.input_json)
    out_path = Path(args.output_txt)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    video_ids = [args.video_id] if args.video_id else sorted(data.keys())
    for video_id in video_ids:
        if video_id not in data:
            continue
        lines.append(f"[{video_id}]")
        detections = data[video_id].get("detections", [])
        for det in detections:
            lines.append(_line_for_detection(det, args.text_key))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote plain text to: {out_path}")


if __name__ == "__main__":
    main()
