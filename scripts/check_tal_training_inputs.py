#!/usr/bin/env python3
"""Validate TAL training prerequisites for ActionFormer.

Checks:
- split files exist (`train.txt`, `val.txt`)
- annotation json exists for each split entry
- feature `.npy` exists for each split entry
- action_type vocab file exists and has valid mapping
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_split(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line).name)
    return items


def _check_vocab(vocab_path: Path) -> int:
    raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    action_type = raw.get("action_type", {})
    label_to_id = action_type.get("label_to_id", {})
    if not isinstance(label_to_id, dict) or not label_to_id:
        raise ValueError(f"Invalid vocab: action_type.label_to_id missing in {vocab_path}")
    if "OTHER" not in label_to_id:
        raise ValueError(f"Invalid vocab: OTHER not found in {vocab_path}")
    return max(label_to_id.values()) + 1


def _summarize_split(
    split_name: str, names: list[str], annot_dir: Path, feat_dir: Path
) -> tuple[int, int]:
    missing_annot = 0
    missing_feat = 0
    for name in names:
        stem = Path(name).stem
        if not (annot_dir / name).exists():
            missing_annot += 1
        if not (feat_dir / f"{stem}.npy").exists():
            missing_feat += 1
    print(
        f"[check] split={split_name} items={len(names)} "
        f"missing_annot={missing_annot} missing_feat={missing_feat}"
    )
    return missing_annot, missing_feat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split-list-dir", type=str, required=True)
    p.add_argument("--annot-dir", type=str, required=True)
    p.add_argument("--feat-dir", type=str, required=True)
    p.add_argument("--vocab-json", type=str, required=True)
    args = p.parse_args()

    split_list_dir = Path(args.split_list_dir)
    annot_dir = Path(args.annot_dir)
    feat_dir = Path(args.feat_dir)
    vocab_path = Path(args.vocab_json)

    for req in ("train.txt", "val.txt"):
        pth = split_list_dir / req
        if not pth.exists():
            raise FileNotFoundError(f"Missing split file: {pth}")
    if not annot_dir.exists():
        raise FileNotFoundError(f"annot_dir not found: {annot_dir}")
    if not feat_dir.exists():
        raise FileNotFoundError(f"feat_dir not found: {feat_dir}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab_json not found: {vocab_path}")

    num_classes = _check_vocab(vocab_path)
    print(f"[check] action_type num_classes={num_classes}")

    train_names = _read_split(split_list_dir / "train.txt")
    val_names = _read_split(split_list_dir / "val.txt")

    t_ma, t_mf = _summarize_split("train", train_names, annot_dir, feat_dir)
    v_ma, v_mf = _summarize_split("val", val_names, annot_dir, feat_dir)

    total_missing = t_ma + t_mf + v_ma + v_mf
    if total_missing > 0:
        print(f"[check] FAILED: total_missing={total_missing}")
        return 1

    print("[check] OK: training inputs are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
