"""Scan annotation JSONs and compute frequency distributions for the auxiliary
fields used as side-supervision in Phase 2b (ActionFormer multi-task heads).

Outputs (under --out-dir):
    aux_stats.json      full counts per field, plus per-action stats (n_actions
                        per video, duration distribution, ...)
    aux_vocab.json      top-N + OTHER mapping per field, ready to be loaded as
                        the categorical label space at training time

Run:
    PYTHONPATH=codes uv run python -m VideoMAEv2.tools.scan_aux_labels \\
        --annot-dir "$ANNOT_ROOT_DIR" \\
        --out-dir   data/aux_stats \\
        --top-n 30
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Fields we plan to use as categorical auxiliary heads.
AUX_FIELDS: tuple[str, ...] = (
    "body_part",
    "action_type",
    "grip_or_contact",
    "speed_or_force",
    "posture_change",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Keep top-N most frequent values per field; rest collapsed to OTHER.",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=50,
        help="Additional cutoff: a value must occur >= min-count times to be "
             "kept even if it would fit in the top-N.",
    )
    p.add_argument("--limit", type=int, default=None, help="debug: scan first N JSONs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if args.limit is not None:
        json_paths = json_paths[: args.limit]
    if not json_paths:
        raise SystemExit(f"no annotation json under {annot_dir}")

    counters: dict[str, Counter] = {f: Counter() for f in AUX_FIELDS}
    n_videos = 0
    n_videos_with_actions = 0
    n_actions_total = 0
    n_actions_per_video: list[int] = []
    durations: list[float] = []
    bad_files: list[str] = []

    for jp in tqdm(json_paths, desc="scan", dynamic_ncols=True):
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            bad_files.append(f"{jp.name}: {e}")
            continue
        actions = (obj.get("analysis_result") or {}).get("actions") or []
        n_videos += 1
        if actions:
            n_videos_with_actions += 1
        n_actions_per_video.append(len(actions))
        for a in actions:
            n_actions_total += 1
            try:
                durations.append(float(a["end_time"]) - float(a["start_time"]))
            except (KeyError, TypeError, ValueError):
                pass
            for f in AUX_FIELDS:
                v = a.get(f)
                # Treat None or blank/whitespace-only strings as missing
                if v is None:
                    counters[f]["__MISSING__"] += 1
                else:
                    sv = str(v).strip()
                    if sv == "":
                        counters[f]["__MISSING__"] += 1
                    else:
                        counters[f][sv] += 1

    # ---- summary stats ----
    def quantiles(xs: list[float], qs=(0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
        if not xs:
            return {}
        s = sorted(xs)
        out = {}
        for q in qs:
            i = min(len(s) - 1, int(round(q * (len(s) - 1))))
            out[f"q{q:.2f}"] = s[i]
        return out

    stats = {
        "n_videos": n_videos,
        "n_videos_with_actions": n_videos_with_actions,
        "n_actions_total": n_actions_total,
        "actions_per_video": {
            "mean": (sum(n_actions_per_video) / max(1, len(n_actions_per_video))),
            **quantiles([float(x) for x in n_actions_per_video]),
        },
        "action_duration_sec": quantiles(durations),
        "n_bad_files": len(bad_files),
        "fields": {},
    }
    for f, c in counters.items():
        stats["fields"][f] = {
            "n_unique": len(c),
            "top50": c.most_common(50),
        }

    (out_dir / "aux_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if bad_files:
        (out_dir / "aux_bad_files.txt").write_text(
            "\n".join(bad_files), encoding="utf-8"
        )

    # ---- vocab: top-N + OTHER (per field) ----
    vocab = {}
    for f, c in counters.items():
        kept: list[str] = []
        for value, count in c.most_common():
            if value == "__MISSING__":
                continue
            if count < args.min_count:
                break
            if len(kept) >= args.top_n:
                break
            kept.append(value)
        # Reserve label 0 for OTHER (unknown/rare/missing) so unseen values at
        # training time map cleanly. label_to_id: OTHER=0, kept[i] -> i+1.
        label_to_id = {"OTHER": 0}
        for i, v in enumerate(kept):
            label_to_id[v] = i + 1
        vocab[f] = {
            "label_to_id": label_to_id,
            "id_to_label": {i: v for v, i in label_to_id.items()},
            "num_classes": len(label_to_id),
            "top_n": args.top_n,
            "min_count": args.min_count,
            "coverage": (
                sum(c[v] for v in kept)
                / max(1, sum(v for k, v in c.items() if k != "__MISSING__"))
            ),
        }
    (out_dir / "aux_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[scan_aux_labels] videos={n_videos} actions={n_actions_total} "
        f"bad={len(bad_files)} -> {out_dir}/{{aux_stats.json,aux_vocab.json}}"
    )
    for f, info in vocab.items():
        print(
            f"  {f:18s} num_classes={info['num_classes']:3d} "
            f"coverage={info['coverage']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
