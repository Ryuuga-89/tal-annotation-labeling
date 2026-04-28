"""Dataset for class-agnostic temporal action localization on the 30s/10s
chunk dataset, plus optional auxiliary categorical labels (body_part /
action_type / grip_or_contact / speed_or_force / posture_change).

Plugs into upstream ActionFormer via its `register_dataset` registry.

Per-sample dict (consumed by the model + loss):
    video_id        : str
    feats           : Tensor (C, T)               C=embed_dim, T=num_steps
    segments        : Tensor (N, 2)               in feature-grid units
    labels          : LongTensor (N,)             always 0 (single class "action")
    aux_labels      : dict[str, LongTensor (N,)]  per-segment aux class id
                                                  (0 == OTHER for unknown values)
    fps             : float                       target_fps used at extraction
    duration        : float                       video_duration_meta (seconds)
    feat_stride     : int                         frames per feat step
    feat_num_frames : int                         frames per feat clip (= window)

`aux_labels` is omitted when no aux_vocab_file is provided; downstream code
should treat its absence as "no aux supervision".

Time -> feat-grid mapping (matches upstream THUMOS dataset):
    seg_grid = seg_sec * fps / feat_stride - feat_offset
    feat_offset = 0.5 * feat_num_frames / feat_stride

For our extractor (target_fps=10, window=16, stride=2) this evaluates to:
    feat_offset = 0.5 * 16 / 2 = 4
    seg_grid    = sec * 10 / 2 - 4 = 5*sec - 4
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure VideoMAEv2 dataset utilities and upstream ActionFormer libs are importable.
_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[2]  # = <project>/codes
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from ActionFormer import _upstream  # noqa: F401, E402  -- mounts actionformer_libs
from actionformer_libs.datasets.datasets import register_dataset  # noqa: E402

from VideoMAEv2.dataset.annotation import Action, load_annotation  # noqa: E402


AUX_FIELDS_DEFAULT: tuple[str, ...] = (
    "body_part",
    "action_type",
    "grip_or_contact",
    "speed_or_force",
    "posture_change",
)


# ---------------------------------------------------------------------------
# Helpers


def _read_split_list(path: str | None) -> list[str] | None:
    """Read a list of annotation json filenames, one per line. None or '' = no filter."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"split list not found: {p}")
    items: list[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        items.append(Path(ln).name)
    return items


def _load_aux_vocab(path: str | None) -> dict[str, dict[str, int]] | None:
    """Load `aux_vocab.json` produced by tools/scan_aux_labels.py.

    Returns dict mapping field -> label_to_id (with OTHER=0). None if no path.
    """
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {field: info["label_to_id"] for field, info in raw.items()}


def _encode_aux(value: str, label_to_id: dict[str, int]) -> int:
    if value is None:
        return 0  # OTHER
    return label_to_id.get(str(value).strip(), 0)


def _truncate_feats_with_aux(
    data_dict: dict,
    max_seq_len: int,
    trunc_thresh: float,
    offset: float,
    crop_ratio: tuple[float, float] | None = None,
    max_num_trials: int = 200,
) -> dict:
    """Same logic as upstream `data_utils.truncate_feats`, but also filters
    `aux_labels` (dict of LongTensors) with the same `seg_idx` mask."""
    feat_len = data_dict["feats"].shape[1]
    num_segs = data_dict["segments"].shape[0]

    # If short enough, optionally do random cropping (matches upstream).
    if feat_len <= max_seq_len:
        if crop_ratio is None:
            return data_dict
        max_seq_len = random.randint(
            max(round(crop_ratio[0] * feat_len), 1),
            min(round(crop_ratio[1] * feat_len), feat_len),
        )
        if feat_len == max_seq_len:
            return data_dict

    data_dict = copy.deepcopy(data_dict)

    st = 0
    ed = feat_len
    seg_idx = torch.zeros(num_segs, dtype=torch.bool)
    left = data_dict["segments"][:, 0].clone()
    right = data_dict["segments"][:, 1].clone()

    for _ in range(max_num_trials):
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32).repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict["segments"][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict["segments"][:, 1])
        inter = (right - left).clamp(min=0)
        area = torch.abs(data_dict["segments"][:, 1] - data_dict["segments"][:, 0])
        # avoid div-by-zero on zero-length annotations
        ratio = inter / area.clamp(min=1e-6)
        seg_idx = ratio >= trunc_thresh
        if seg_idx.sum().item() > 0:
            break

    data_dict["feats"] = data_dict["feats"][:, st:ed].clone()
    data_dict["segments"] = torch.stack((left[seg_idx], right[seg_idx]), dim=1) - st
    data_dict["labels"] = data_dict["labels"][seg_idx].clone()
    if "aux_labels" in data_dict and data_dict["aux_labels"]:
        data_dict["aux_labels"] = {
            k: v[seg_idx].clone() for k, v in data_dict["aux_labels"].items()
        }
    return data_dict


# ---------------------------------------------------------------------------
# Dataset


@register_dataset("tal_motion")
class TalMotionDataset(Dataset):
    """30s/10s chunk dataset for class-agnostic TAL with optional aux heads.

    Constructor args follow upstream's `make_dataset` convention. New args
    (relative to THUMOS14Dataset) are at the bottom.
    """

    def __init__(
        self,
        is_training: bool,
        split,                # tuple/list of split names (e.g., ('train',))
        feat_folder: str,
        json_file: str,       # ignored (kept for upstream make_dataset signature)
        feat_stride: int,
        num_frames: int,
        default_fps: float | None,
        downsample_rate: int,
        max_seq_len: int,
        trunc_thresh: float,
        crop_ratio,
        input_dim: int,
        num_classes: int,
        file_prefix,
        file_ext: str,
        force_upsampling: bool,  # ignored (no upsampling for now)
        # ---- our additions ----
        annot_dir: str = "",
        split_list_dir: str = "",   # contains {train,val,test}.txt
        aux_vocab_file: str | None = None,
        aux_fields: tuple[str, ...] = AUX_FIELDS_DEFAULT,
        skip_no_actions: bool = True,
        tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
    ):
        super().__init__()
        # class-agnostic: enforce 1 class
        assert num_classes == 1, (
            f"TalMotionDataset is class-agnostic; expected num_classes=1, got {num_classes}"
        )
        if not annot_dir:
            raise ValueError("annot_dir is required for TalMotionDataset")
        if not split_list_dir:
            raise ValueError("split_list_dir is required for TalMotionDataset")

        self.is_training = is_training
        self.split = tuple(split)
        self.feat_folder = feat_folder
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps if default_fps is not None else 10.0
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.file_prefix = file_prefix or ""
        self.file_ext = file_ext
        self.annot_dir = Path(annot_dir)
        self.split_list_dir = Path(split_list_dir)
        self.aux_fields = tuple(aux_fields)
        self.skip_no_actions = skip_no_actions

        # aux vocab: {field: {label_str: id, "OTHER": 0}}
        self.aux_vocab = _load_aux_vocab(aux_vocab_file) or {}
        self.aux_num_classes = {
            f: max(self.aux_vocab.get(f, {"OTHER": 0}).values()) + 1
            for f in self.aux_fields
        }

        # Collect annotation filenames from union of named splits.
        names: list[str] = []
        for s in self.split:
            lst = _read_split_list(self.split_list_dir / f"{s}.txt")
            if lst is None:
                raise FileNotFoundError(self.split_list_dir / f"{s}.txt")
            names.extend(lst)
        # de-dupe while keeping deterministic order
        seen: set[str] = set()
        dedup: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                dedup.append(n)
        self._names = dedup

        # Build the in-memory database (filenames + metadata only; features
        # are loaded lazily in __getitem__).
        self.data_list = self._build_db()

        self.db_attributes = {
            "dataset_name": "tal_motion",
            "tiou_thresholds": np.asarray(tiou_thresholds, dtype=np.float32),
            "empty_label_ids": [],
        }

    def get_attributes(self):
        return self.db_attributes

    # ------------------------------------------------------------------
    # DB construction

    def _build_db(self) -> tuple:
        out: list[dict] = []
        n_missing_feat = 0
        n_no_action = 0
        for name in self._names:
            jp = self.annot_dir / name
            if not jp.exists():
                continue
            try:
                rec = load_annotation(jp)
            except Exception:
                continue

            stem = rec.video_stem
            feat_file = os.path.join(
                self.feat_folder, self.file_prefix + stem + self.file_ext
            )
            if not os.path.exists(feat_file):
                n_missing_feat += 1
                continue

            actions: list[Action] = rec.actions
            if not actions and self.skip_no_actions:
                n_no_action += 1
                continue

            if actions:
                segments = np.asarray(
                    [[a.start_time, a.end_time] for a in actions], dtype=np.float32
                )
                # all class-agnostic -> label 0
                labels = np.zeros(len(actions), dtype=np.int64)
                aux: dict[str, np.ndarray] = {}
                for f in self.aux_fields:
                    if f not in self.aux_vocab:
                        continue
                    vocab = self.aux_vocab[f]
                    aux[f] = np.asarray(
                        [_encode_aux(getattr(a, f, ""), vocab) for a in actions],
                        dtype=np.int64,
                    )
            else:
                segments = None
                labels = None
                aux = {}

            out.append(
                {
                    "id": stem,
                    "feat_file": feat_file,
                    "fps": self.default_fps,
                    "duration": float(rec.video_duration or 0.0),
                    "segments": segments,
                    "labels": labels,
                    "aux_labels": aux,
                }
            )

        if n_missing_feat or n_no_action:
            print(
                f"[tal_motion] split={self.split} "
                f"kept={len(out)} missing_feat={n_missing_feat} no_action={n_no_action}",
                flush=True,
            )
        return tuple(out)

    # ------------------------------------------------------------------
    # Pytorch Dataset interface

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        item = self.data_list[idx]
        feats = np.load(item["feat_file"]).astype(np.float32)
        # downsampling support
        feats = feats[:: self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        if item["segments"] is not None:
            segments = torch.from_numpy(
                item["segments"] * item["fps"] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(item["labels"])
            aux_labels = {
                f: torch.from_numpy(v) for f, v in item["aux_labels"].items()
            }
        else:
            segments = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            aux_labels = {}

        data_dict = {
            "video_id": item["id"],
            "feats": feats,                     # C x T
            "segments": segments,               # N x 2 (in feat-grid)
            "labels": labels,                   # N
            "aux_labels": aux_labels,           # {field: LongTensor (N,)}
            "fps": item["fps"],
            "duration": item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": self.num_frames,
        }

        if self.is_training and segments.shape[0] > 0:
            data_dict = _truncate_feats_with_aux(
                data_dict,
                self.max_seq_len,
                self.trunc_thresh,
                feat_offset,
                self.crop_ratio,
            )
        return data_dict
