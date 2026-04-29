#!/usr/bin/env python3
"""Visualize label distributions for body_part/action_type/combined labels.

Outputs (under --output-dir):
- body_part_counts.csv
- action_type_counts.csv
- combined_counts.csv
- body_part_counts.png
- action_type_counts.png
- combined_counts_topN.png
- label_distribution_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

try:
    import japanize_matplotlib  # noqa: F401
    _JAPANIZE_ENABLED = True
except Exception:
    _JAPANIZE_ENABLED = False

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
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", type=str, required=True, help="Annotation directory")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--limit", type=int, default=0, help="Scan first N files (0 = all)")
    p.add_argument(
        "--disable-merge",
        action="store_true",
        help="Disable merge rules and use raw labels",
    )
    p.add_argument(
        "--top-n-combined",
        type=int,
        default=50,
        help="Top-N combined labels to plot in PNG",
    )
    return p.parse_args()


def _setup_japanese_font() -> str:
    """Enable Japanese font rendering for matplotlib."""
    # japanize-matplotlib already sets IPAexGothic if available.
    if _JAPANIZE_ENABLED:
        rcParams["axes.unicode_minus"] = False
        return "japanize_matplotlib"

    # Fallback to common JP fonts if japanize-matplotlib is unavailable.
    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Yu Gothic",
        "Meiryo",
        "TakaoGothic",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return name
    # Last resort: keep matplotlib default.
    rcParams["axes.unicode_minus"] = False
    return "default"


def _merge_by_contains(label: str, ordered_targets: tuple[str, ...], other_label: str) -> str:
    for target in ordered_targets:
        if target in label:
            return target
    return other_label


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _write_counter_csv(path: Path, counter: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in counter.most_common():
            writer.writerow([label, count])


def _plot_counter(
    counter: Counter[str],
    title: str,
    output_png: Path,
    top_n: int = 0,
    horizontal: bool = True,
) -> None:
    items = counter.most_common(top_n if top_n > 0 else None)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    if not labels:
        return

    plt.figure(figsize=(12, max(4, len(labels) * 0.35)))
    if horizontal:
        y = list(range(len(labels)))
        plt.barh(y, values)
        plt.yticks(y, labels, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel("Count")
    else:
        x = list(range(len(labels)))
        plt.bar(x, values)
        plt.xticks(x, labels, rotation=90, fontsize=8)
        plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font_name = _setup_japanese_font()

    json_paths = sorted(annot_dir.glob("*.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"No json files found in {annot_dir}")

    body_part_counter: Counter[str] = Counter()
    action_type_counter: Counter[str] = Counter()
    combined_counter: Counter[str] = Counter()

    bad_files = 0
    actions_seen = 0
    for jp in json_paths:
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            bad_files += 1
            continue

        actions = (obj.get("analysis_result") or {}).get("actions") or []
        actions_seen += len(actions)
        for action in actions:
            body_part = _normalize(action.get("body_part", ""))
            action_type = _normalize(action.get("action_type", ""))

            if args.disable_merge:
                body_part = body_part if body_part else BODY_PART_OTHER
                action_type = action_type if action_type else ACTION_TYPE_OTHER
            else:
                body_part = (
                    _merge_by_contains(body_part, BODY_PART_MERGE_ORDER, BODY_PART_OTHER)
                    if body_part
                    else BODY_PART_OTHER
                )
                action_type = (
                    _merge_by_contains(action_type, ACTION_TYPE_MERGE_ORDER, ACTION_TYPE_OTHER)
                    if action_type
                    else ACTION_TYPE_OTHER
                )

            body_part_counter[body_part] += 1
            action_type_counter[action_type] += 1
            combined_counter[f"{body_part}|{action_type}"] += 1

    body_csv = out_dir / "body_part_counts.csv"
    action_csv = out_dir / "action_type_counts.csv"
    combined_csv = out_dir / "combined_counts.csv"
    _write_counter_csv(body_csv, body_part_counter)
    _write_counter_csv(action_csv, action_type_counter)
    _write_counter_csv(combined_csv, combined_counter)

    body_png = out_dir / "body_part_counts.png"
    action_png = out_dir / "action_type_counts.png"
    combined_png = out_dir / "combined_counts_topN.png"
    _plot_counter(body_part_counter, "Body Part Distribution", body_png)
    _plot_counter(action_type_counter, "Action Type Distribution", action_png)
    _plot_counter(
        combined_counter,
        f"Combined Label Distribution (Top {args.top_n_combined})",
        combined_png,
        top_n=args.top_n_combined,
    )

    summary_path = out_dir / "label_distribution_summary.txt"
    summary_lines = [
        f"annot_dir: {annot_dir}",
        f"files_scanned: {len(json_paths)}",
        f"bad_files: {bad_files}",
        f"actions_seen: {actions_seen}",
        f"merge_enabled: {not args.disable_merge}",
        f"matplotlib_font: {font_name}",
        f"unique_body_part: {len(body_part_counter)}",
        f"unique_action_type: {len(action_type_counter)}",
        f"unique_combined: {len(combined_counter)}",
        f"body_part_csv: {body_csv}",
        f"action_type_csv: {action_csv}",
        f"combined_csv: {combined_csv}",
        f"body_part_png: {body_png}",
        f"action_type_png: {action_png}",
        f"combined_png: {combined_png}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[plot_label_distribution] files={len(json_paths)} bad={bad_files} actions={actions_seen}")
    print(f"  body_part unique={len(body_part_counter)} -> {body_csv}")
    print(f"  action_type unique={len(action_type_counter)} -> {action_csv}")
    print(f"  combined unique={len(combined_counter)} -> {combined_csv}")
    print(f"  plots: {body_png}, {action_png}, {combined_png}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
