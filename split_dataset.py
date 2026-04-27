"""Split annotations into train/val/test sets at the video-source group level.

Groups are defined by factory*worker (for factory videos) or by 64-char hash
(for hash-based videos). This prevents data leakage between splits.

Usage:
    uv run python split_dataset.py
    uv run python split_dataset.py --annot-dir /path/to/annots --video-dir /path/to/videos
"""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from prepare_dataset import ANNOT_DIR, VIDEO_DIR, extract_group_key


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--annot-dir", default=ANNOT_DIR)
    parser.add_argument("--video-dir", default=VIDEO_DIR)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Collect valid annotations (those with existing video files)
    annot_files = sorted(Path(args.annot_dir).glob("*.json"))
    print(f"Found {len(annot_files)} annotation files")

    valid_annots = []
    for af in annot_files:
        with open(af) as f:
            data = json.load(f)
        video_stem = Path(data["video_path"]).stem
        video_path = os.path.join(args.video_dir, f"{video_stem}.mp4")
        actions = data.get("analysis_result", {}).get("actions", [])
        if os.path.exists(video_path) and actions:
            valid_annots.append(af.name)

    print(f"Valid annotations (video exists + has actions): {len(valid_annots)}")

    # 2. Group by video source
    groups: dict[str, list[str]] = defaultdict(list)
    for name in valid_annots:
        key = extract_group_key(name)
        groups[key].append(name)

    factory_groups = {k: v for k, v in groups.items() if k.startswith("factory")}
    hash_groups = {k: v for k, v in groups.items() if not k.startswith("factory")}
    print(f"Groups: {len(factory_groups)} factory/worker, {len(hash_groups)} hash-based")

    # 3. Shuffle groups deterministically
    rng = random.Random(args.seed)

    factory_keys = sorted(factory_groups.keys())
    hash_keys = sorted(hash_groups.keys())
    rng.shuffle(factory_keys)
    rng.shuffle(hash_keys)

    # 4. Select test groups (~100 samples, balanced between factory and hash)
    def select_groups_for_n_samples(keys: list[str], group_dict: dict, target: int) -> tuple[list[str], list[str]]:
        """Select groups until we reach target sample count. Returns (selected_keys, remaining_keys)."""
        selected = []
        count = 0
        remaining = []
        for k in keys:
            if count < target:
                selected.append(k)
                count += len(group_dict[k])
            else:
                remaining.append(k)
        return selected, remaining

    # Split test target evenly between factory and hash
    factory_test_target = args.test_samples // 2
    hash_test_target = args.test_samples - factory_test_target

    factory_test_keys, factory_remaining = select_groups_for_n_samples(
        factory_keys, factory_groups, factory_test_target
    )
    hash_test_keys, hash_remaining = select_groups_for_n_samples(
        hash_keys, hash_groups, hash_test_target
    )

    test_annots = []
    for k in factory_test_keys:
        test_annots.extend(factory_groups[k])
    for k in hash_test_keys:
        test_annots.extend(hash_groups[k])

    # 5. Split remaining into val (~10%) and train (~90%) at group level
    all_remaining_keys = []
    remaining_groups = {}
    for k in factory_remaining:
        all_remaining_keys.append(k)
        remaining_groups[k] = factory_groups[k]
    for k in hash_remaining:
        all_remaining_keys.append(k)
        remaining_groups[k] = hash_groups[k]

    rng.shuffle(all_remaining_keys)
    n_val_groups = max(1, int(len(all_remaining_keys) * args.val_ratio))

    val_keys = all_remaining_keys[:n_val_groups]
    train_keys = all_remaining_keys[n_val_groups:]

    val_annots = []
    for k in val_keys:
        val_annots.extend(remaining_groups[k])
    train_annots = []
    for k in train_keys:
        train_annots.extend(remaining_groups[k])

    # Sort for reproducibility
    train_annots.sort()
    val_annots.sort()
    test_annots.sort()

    print(f"\nSplit results:")
    print(f"  Train: {len(train_annots)} samples ({len(train_keys)} groups)")
    print(f"  Val:   {len(val_annots)} samples ({len(val_keys)} groups)")
    print(f"  Test:  {len(test_annots)} samples ({len(factory_test_keys) + len(hash_test_keys)} groups)")

    # 6. Verify no group overlap
    train_group_keys = set(train_keys)
    val_group_keys = set(val_keys)
    test_group_keys = set(factory_test_keys + hash_test_keys)
    assert train_group_keys.isdisjoint(val_group_keys), "Train/val group overlap!"
    assert train_group_keys.isdisjoint(test_group_keys), "Train/test group overlap!"
    assert val_group_keys.isdisjoint(test_group_keys), "Val/test group overlap!"
    print("  No group overlap between splits ✓")

    # 7. Save
    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in [("train", train_annots), ("val", val_annots), ("test", test_annots)]:
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {path}")

    meta = {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_samples_target": args.test_samples,
        "train_samples": len(train_annots),
        "val_samples": len(val_annots),
        "test_samples": len(test_annots),
        "train_groups": len(train_keys),
        "val_groups": len(val_keys),
        "test_groups": len(factory_test_keys) + len(hash_test_keys),
        "total_valid_annots": len(valid_annots),
        "total_groups": len(groups),
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")


if __name__ == "__main__":
    main()
