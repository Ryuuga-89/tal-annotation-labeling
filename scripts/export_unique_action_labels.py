#!/usr/bin/env python3
"""Export unique labels from annotation JSON files.

This script scans annotation files and writes:
  - body_part_unique.txt
  - action_type_unique.txt
  - combined_label_unique.txt
  - combined_label_counts.csv
  - combined_label_vocab.json
  - unique_labels_summary.txt

Each output file is UTF-8 text with one label per line.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BODY_PART_MERGE_ORDER: tuple[str, ...] = (
    "両手",
    "両足",
    "頭部",
    "右手",
    "左手",
    "右足",
    "左足",
)
BODY_PART_OTHER = "その他"

ACTION_TYPE_MERGE_ORDER: tuple[str, ...] = (
    "運ぶ",
    "伸ばす",
    "つかむ",
    "位置決め",
    "離す",
    "保持",
    "押す",
    "回す",
    "滑らせる",
    "持ち上げる",
    "引く",
    "位置ぎめ",
    "置く",
    "押し込む",
    "移動",
    "向きを変える",
    "曲げる",
    "視線を向ける",
    "待機",
    "動かす",
    "使用",
    "検査",
    "組み立て",
    "切る",
    "叩く",
)
ACTION_TYPE_OTHER = "その他"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unique body_part/action_type labels to text files."
    )
    parser.add_argument(
        "--annot-dir",
        type=str,
        required=True,
        help="Directory containing annotation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output text files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: scan only first N files (0 means all files).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty/blank labels instead of dropping them.",
    )
    parser.add_argument(
        "--disable-merge",
        action="store_true",
        help="Disable label merging; export raw unique labels only.",
    )
    return parser.parse_args()


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _merge_by_contains(label: str, ordered_targets: tuple[str, ...], other_label: str) -> str:
    """Merge a raw label by ordered substring match (priority = list order)."""
    for target in ordered_targets:
        if target in label:
            return target
    return other_label


def _write_lines(path: Path, values: set[str]) -> None:
    sorted_values = sorted(values)
    path.write_text("\n".join(sorted_values) + ("\n" if sorted_values else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"No annotation json files found in: {annot_dir}")

    body_parts: set[str] = set()
    action_types: set[str] = set()
    combined_counter: Counter[str] = Counter()
    bad_files = 0
    total_actions = 0

    for json_path in json_paths:
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            bad_files += 1
            continue

        actions = (obj.get("analysis_result") or {}).get("actions") or []
        total_actions += len(actions)
        for action in actions:
            bp = _normalize_label(action.get("body_part", ""))
            at = _normalize_label(action.get("action_type", ""))
            if not args.disable_merge:
                bp = _merge_by_contains(bp, BODY_PART_MERGE_ORDER, BODY_PART_OTHER) if bp else (
                    bp if args.keep_empty else BODY_PART_OTHER
                )
                at = _merge_by_contains(at, ACTION_TYPE_MERGE_ORDER, ACTION_TYPE_OTHER) if at else (
                    at if args.keep_empty else ACTION_TYPE_OTHER
                )
            if args.keep_empty or bp:
                body_parts.add(bp)
            if args.keep_empty or at:
                action_types.add(at)
            if (args.keep_empty or bp) and (args.keep_empty or at):
                combined_counter[f"{bp}|{at}"] += 1

    body_part_path = output_dir / "body_part_unique.txt"
    action_type_path = output_dir / "action_type_unique.txt"
    combined_path = output_dir / "combined_label_unique.txt"
    combined_counts_path = output_dir / "combined_label_counts.csv"
    combined_vocab_path = output_dir / "combined_label_vocab.json"
    summary_path = output_dir / "unique_labels_summary.txt"

    _write_lines(body_part_path, body_parts)
    _write_lines(action_type_path, action_types)
    _write_lines(combined_path, set(combined_counter.keys()))
    combined_counts_lines = ["label,count"]
    for label, count in combined_counter.most_common():
        safe_label = label.replace('"', '""')
        combined_counts_lines.append(f"\"{safe_label}\",{count}")
    combined_counts_path.write_text("\n".join(combined_counts_lines) + "\n", encoding="utf-8")
    label_to_id = {"OTHER": 0}
    for i, label in enumerate(sorted(combined_counter.keys()), start=1):
        label_to_id[label] = i
    combined_vocab = {
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
        "num_classes": len(label_to_id),
    }
    combined_vocab_path.write_text(
        json.dumps(combined_vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        f"annot_dir: {annot_dir}",
        f"files_scanned: {len(json_paths)}",
        f"bad_files: {bad_files}",
        f"actions_seen: {total_actions}",
        f"merge_enabled: {not args.disable_merge}",
        f"unique_body_part: {len(body_parts)}",
        f"unique_action_type: {len(action_types)}",
        f"unique_combined_label: {len(combined_counter)}",
        f"body_part_file: {body_part_path}",
        f"action_type_file: {action_type_path}",
        f"combined_label_file: {combined_path}",
        f"combined_counts_file: {combined_counts_path}",
        f"combined_vocab_file: {combined_vocab_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[export_unique_action_labels] scanned={len(json_paths)} bad={bad_files}")
    print(f"  unique body_part   : {len(body_parts)} -> {body_part_path}")
    print(f"  unique action_type : {len(action_types)} -> {action_type_path}")
    print(f"  unique combined    : {len(combined_counter)} -> {combined_path}")
    print(f"  combined counts    : {combined_counts_path}")
    print(f"  combined vocab     : {combined_vocab_path} (num_classes={len(label_to_id)})")
    print(f"  summary            : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
