#!/usr/bin/env python3
"""Build combined (body_part x action_type) vocabulary JSON for training.

Default behavior:
- Apply merge rules (substring match with priority order) to body_part/action_type
- Combine as "{body_part}|{action_type}"
- Reserve id=0 for "その他|その他"

Output JSON format:
{
  "combined_label": {
    "label_to_id": {"その他|その他": 0, "右手|押す": 1, ...},
    "id_to_label": {"0": "その他|その他", "1": "右手|押す", ...},
    "num_classes": 123,
    "min_count": 5,
    "top_n": 300,
    "merge_enabled": true
  }
}
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
COMBINED_OTHER = f"{BODY_PART_OTHER}|{ACTION_TYPE_OTHER}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", type=str, required=True, help="Annotation directory")
    p.add_argument("--output-json", type=str, required=True, help="Output vocab JSON")
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Keep labels with count >= min-count",
    )
    p.add_argument("--top-n", type=int, default=0, help="Keep top-N labels (0 = keep all)")
    p.add_argument("--limit", type=int, default=0, help="Scan first N files (0 = all)")
    p.add_argument(
        "--disable-merge",
        action="store_true",
        help="Disable merge rules and use raw labels before combination",
    )
    return p.parse_args()


def _merge_by_contains(label: str, ordered_targets: tuple[str, ...], other_label: str) -> str:
    for target in ordered_targets:
        if target in label:
            return target
    return other_label


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"No json files found in {annot_dir}")

    counts: Counter[str] = Counter()
    bad_files = 0
    for jp in json_paths:
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            bad_files += 1
            continue
        actions = (obj.get("analysis_result") or {}).get("actions") or []
        for action in actions:
            bp = str(action.get("body_part", "")).strip()
            at = str(action.get("action_type", "")).strip()

            if args.disable_merge:
                if not bp:
                    bp = BODY_PART_OTHER
                if not at:
                    at = ACTION_TYPE_OTHER
            else:
                bp = _merge_by_contains(bp, BODY_PART_MERGE_ORDER, BODY_PART_OTHER) if bp else BODY_PART_OTHER
                at = _merge_by_contains(at, ACTION_TYPE_MERGE_ORDER, ACTION_TYPE_OTHER) if at else ACTION_TYPE_OTHER

            counts[f"{bp}|{at}"] += 1

    kept = [label for label, c in counts.most_common() if c >= args.min_count]
    if args.top_n > 0:
        kept = kept[: args.top_n]

    label_to_id = {COMBINED_OTHER: 0}
    next_id = 1
    for label in kept:
        if label == COMBINED_OTHER:
            continue
        label_to_id[label] = next_id
        next_id += 1

    payload = {
        "combined_label": {
            "label_to_id": label_to_id,
            "id_to_label": {str(v): k for k, v in label_to_id.items()},
            "num_classes": len(label_to_id),
            "min_count": args.min_count,
            "top_n": args.top_n,
            "merge_enabled": not args.disable_merge,
        }
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[build_combined_vocab] files={len(json_paths)} bad={bad_files} "
        f"classes={len(label_to_id)} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
