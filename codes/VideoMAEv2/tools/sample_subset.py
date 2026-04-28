"""Random group-split of the annotation set + subset list for Phase 2b training.

The 30s/10s chunk dataset is built by slicing *source* videos into chunks; many
chunk stems share the same source-video hash (the prefix before the first `_`).
A naive random split would put chunks of the same source video into both
train and val, which inflates eval. We split by *group* (= source-video hash).

Outputs (under --out-dir):
    split.json              full split summary (counts, seed, hash buckets)
    train.txt val.txt test.txt
                            one annotation-json filename per line (no path).
                            train.txt is sub-sampled to --train-size when given.
    subset.txt              union of (sub-sampled) train + val + test, suitable
                            as input to extract_features.py --annot-list.

Run:
    PYTHONPATH=codes uv run python -m VideoMAEv2.tools.sample_subset \\
        --annot-dir "$ANNOT_ROOT_DIR" \\
        --out-dir   data/splits/v1 \\
        --train-size 5000 \\
        --val-frac 0.05 --test-frac 0.05 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="If set, randomly sub-sample this many *chunks* from the train "
             "split. None = keep all train chunks.",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def group_key(stem: str) -> str:
    """Return the source-video hash from a chunk stem.

    Chunk stems look like:
        <hash>_<HH-MM-SS.mmm>_<HH-MM-SS.mmm>_<idx>_<dur>
    so the group key is the prefix before the first underscore.
    """
    return stem.split("_", 1)[0]


def assign_split(group: str, val_frac: float, test_frac: float, seed: int) -> str:
    """Deterministic split using a salted hash of the group key.

    This is stable across runs (so re-running with the same seed reproduces the
    same split) and independent of the annotation directory listing order.
    """
    h = hashlib.sha1(f"{seed}:{group}".encode("utf-8")).digest()
    # take first 8 bytes, normalize to [0, 1)
    x = int.from_bytes(h[:8], "big") / (1 << 64)
    if x < val_frac:
        return "val"
    if x < val_frac + test_frac:
        return "test"
    return "train"


def main() -> int:
    args = parse_args()
    annot_dir = Path(args.annot_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(annot_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"no annotation json under {annot_dir}")

    groups: dict[str, list[str]] = defaultdict(list)
    for jp in json_paths:
        stem = jp.stem
        groups[group_key(stem)].append(jp.name)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    group_split: dict[str, str] = {}
    for g, names in groups.items():
        s = assign_split(g, args.val_frac, args.test_frac, args.seed)
        group_split[g] = s
        splits[s].extend(names)

    # Sort for determinism, then sub-sample train if requested.
    for s in splits:
        splits[s].sort()
    rng = random.Random(args.seed)
    train_full = list(splits["train"])
    if args.train_size is not None and args.train_size < len(train_full):
        splits["train"] = sorted(rng.sample(train_full, args.train_size))

    for s, items in splits.items():
        (out_dir / f"{s}.txt").write_text(
            "\n".join(items) + ("\n" if items else ""), encoding="utf-8"
        )

    subset = sorted(set(splits["train"]) | set(splits["val"]) | set(splits["test"]))
    (out_dir / "subset.txt").write_text(
        "\n".join(subset) + ("\n" if subset else ""), encoding="utf-8"
    )

    summary = {
        "annot_dir": str(annot_dir),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_total_chunks": len(json_paths),
        "n_groups": len(groups),
        "n_chunks": {s: len(items) for s, items in splits.items()},
        "n_groups_per_split": {
            s: sum(1 for v in group_split.values() if v == s)
            for s in ("train", "val", "test")
        },
        "train_size_arg": args.train_size,
        "n_train_full": len(train_full),
        "n_subset": len(subset),
    }
    (out_dir / "split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[sample_subset] wrote {out_dir}/{{train,val,test,subset}}.txt + split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
