#!/usr/bin/env python3
"""
End-to-end pipeline: VideoMAE V2 features -> ActionFormer TAL -> local VLM captions.

VideoMAE settings must match training (see ``codes/VideoMAEv2/configs/extract_vit_b.yaml``).
ActionFormer checkpoint example::

    models/ActionFormer/checkpoint.pth.tar

Example::

    export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
    export HF_HOME=/lustre/work/mt/.cache/huggingface

    uv run python scripts/pipeline_tal_vlm.py \\
        --video /path/to/clip.mp4 \\
        --af-config codes/ActionFormer/configs/tal_motion_vit_b.yaml \\
        --af-ckpt models/ActionFormer/checkpoint.pth.tar \\
        --mae-config codes/VideoMAEv2/configs/extract_vit_b.yaml \\
        --vlm-model google/gemma-3n-E4B-it \\
        --output-dir outputs/pipeline_demo

Outputs ``result.json`` (structured) and ``lines.txt`` (one ``{s}-{e}: text`` per line, sorted by start).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
_PKG_PARENT = _ROOT / "codes"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from ActionFormer import _upstream  # noqa: F401, E402
from VideoMAEv2.extract_features import (  # noqa: E402
    ExtractConfig,
    _merge,
    extract_features_array,
)
from VideoMAEv2.models import build_backbone  # noqa: E402
from actionformer_libs.core.config import load_config  # noqa: E402
from actionformer_libs.utils import fix_random_seed  # noqa: E402
from infer_tal import (  # noqa: E402
    _load_combined_id_to_label,
    infer_one_video,
    load_model,
)
from vlm import QwenLocalVLM, sample_frames_equidistant  # noqa: E402


DEFAULT_VLM_INSTRUCTION = (
    "以下の画像列は、ひとつながりの動作の一部を時系列に並べたものです。"
    "主語は省略し、動作とその対象を含む短い日本語1文（最大40文字）で説明してください。"
    "1文のみを出力し、箇条書きや補足説明はしないでください。"
    "時刻・秒数・フレーム番号は出力に含めないでください。"
)


def _load_mae_config(path: Path) -> ExtractConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExtractConfig()
    return _merge(cfg, raw)


def _resolve_path_maybe_relative(p: str | Path, base: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _format_line(start_time: float, end_time: float, description: str) -> str:
    s = round(float(start_time), 1)
    e = round(float(end_time), 1)
    return f"{s:.1f}-{e:.1f}: {description.strip()}"


def _short_one_sentence(text: str, max_chars: int = 40) -> str:
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    for sep in ("。", "！", "？", ".", "!", "?"):
        idx = t.find(sep)
        if idx != -1:
            t = t[: idx + 1]
            break
    if len(t) > max_chars:
        t = t[:max_chars].rstrip("、,;: ") + "。"
    return t


def _prep_segment(
    video_path: Path,
    det: dict,
    num_frames: int,
    decode_threads: int,
) -> tuple[dict, list]:
    st = float(det["start_time"])
    et = float(det["end_time"])
    if et <= st:
        return det, []
    imgs = sample_frames_equidistant(
        video_path,
        st,
        et,
        num_frames,
        decode_threads=decode_threads,
    )
    return det, imgs


def main() -> int:
    parser = argparse.ArgumentParser(description="VideoMAE + ActionFormer + Gemma/Qwen VL pipeline")
    parser.add_argument("--video", type=str, required=True, help="Input mp4 path")
    parser.add_argument(
        "--af-config",
        type=str,
        default=str(_ROOT / "codes/ActionFormer/configs/tal_motion_vit_b.yaml"),
    )
    parser.add_argument(
        "--af-ckpt",
        type=str,
        default=str(_ROOT / "models/ActionFormer/checkpoint.pth.tar"),
    )
    parser.add_argument(
        "--mae-config",
        type=str,
        default=str(_ROOT / "codes/VideoMAEv2/configs/extract_vit_b.yaml"),
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        required=True,
        help="HuggingFace model id or local path (e.g. google/gemma-3n-E4B-it)",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Directory for result.json and lines.txt")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--num-vlm-frames", type=int, default=8)
    parser.add_argument(
        "--vlm-instruction",
        type=str,
        default=DEFAULT_VLM_INSTRUCTION,
        help="Japanese instruction (no timestamps in prompt)",
    )
    parser.add_argument("--vlm-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=4,
        help="Independent VLM batch size (default: 4, 0 or negative = all segments at once)",
    )
    parser.add_argument("--device-mae", type=int, default=0)
    parser.add_argument("--device-af", type=int, default=0)
    parser.add_argument("--device-vlm", type=int, default=0)
    parser.add_argument("--prep-workers", type=int, default=8, help="Thread workers for frame decode")
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=2,
        help="decord threads per video open in segment frame sampling",
    )
    parser.add_argument(
        "--timing-json",
        type=str,
        default="",
        help="Optional path to write timing details as JSON",
    )
    args = parser.parse_args()
    if "gemma" in args.vlm_model.lower() and not os.environ.get("HF_TOKEN", "").strip():
        print(
            "ERROR: Gemmaモデルは gated のため HF_TOKEN が必要です。"
            " `export HF_TOKEN=...` を設定して再実行してください。",
            file=sys.stderr,
        )
        return 1

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else _ROOT / "outputs" / f"pipeline_{video_path.stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_path = Path(args.timing_json) if args.timing_json else out_dir / "timings.json"
    t_total_start = time.perf_counter()
    timings: dict[str, float | int] = {}

    mae_cfg = _load_mae_config(Path(args.mae_config))
    mae_cfg.ckpt_path = str(_resolve_path_maybe_relative(mae_cfg.ckpt_path, _ROOT))
    mae_cfg.decode_threads = int(args.decode_threads)

    device_mae = torch.device(f"cuda:{args.device_mae}" if torch.cuda.is_available() else "cpu")
    device_vlm = f"cuda:{args.device_vlm}" if torch.cuda.is_available() else "cpu"

    from VideoMAEv2.extract_features import _np_dtype, _torch_dtype  # noqa: WPS433

    dtype = _torch_dtype(mae_cfg.dtype)
    save_dtype = _np_dtype(mae_cfg.save_dtype)

    if device_mae.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(device_mae)

    print("[1/4] VideoMAE feature extraction (in-memory)...", flush=True)
    t_stage_start = time.perf_counter()
    backbone = build_backbone(
        mae_cfg.model_name,
        ckpt_path=mae_cfg.ckpt_path or None,
        drop_path_rate=mae_cfg.drop_path_rate,
    )
    backbone.eval()
    backbone.to(device_mae)

    feats_tc, mae_meta = extract_features_array(
        video_path,
        backbone=backbone,
        cfg=mae_cfg,
        device=device_mae,
        dtype=dtype,
        save_dtype=save_dtype,
    )
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if feats_tc is None:
        print(f"Feature extraction failed: {mae_meta}", file=sys.stderr)
        return 1

    print(f"  steps={mae_meta.get('num_steps')} embed_dim={mae_meta.get('embed_dim')}", flush=True)
    timings["stage1_feature_extract_sec"] = time.perf_counter() - t_stage_start

    print("[2/4] ActionFormer inference...", flush=True)
    t_stage_start = time.perf_counter()
    if not Path(args.af_config).is_file():
        print(f"AF config not found: {args.af_config}", file=sys.stderr)
        return 1
    if not Path(args.af_ckpt).is_file():
        print(f"AF checkpoint not found: {args.af_ckpt}", file=sys.stderr)
        return 1

    af_cfg = load_config(args.af_config)
    id_to_label = _load_combined_id_to_label(af_cfg)
    _ = fix_random_seed(0, include_cuda=False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    af_model = load_model(af_cfg, args.af_ckpt, args.device_af, torch.cuda.is_available())

    feats = np.ascontiguousarray(feats_tc.transpose())
    downsample_rate = af_cfg["dataset"].get("downsample_rate", 1)
    if downsample_rate > 1:
        feats = feats[:, ::downsample_rate]
    feat_stride_base = af_cfg["dataset"].get("feat_stride", 2)
    num_frames = af_cfg["dataset"].get("num_frames", 16)
    fps = float(af_cfg["dataset"].get("default_fps", 10.0))
    feat_stride = feat_stride_base * downsample_rate
    duration = (feats.shape[1] * feat_stride + 0.5 * num_frames) / fps

    detections = infer_one_video(
        af_model,
        video_path.stem,
        feats,
        duration,
        feat_stride,
        num_frames,
        fps,
        score_thresh=args.score_thresh,
        id_to_label=id_to_label or None,
    )
    del af_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  detections (thresh={args.score_thresh}): {len(detections)}", flush=True)
    timings["stage2_actionformer_sec"] = time.perf_counter() - t_stage_start

    dets_sorted = sorted(detections, key=lambda d: float(d["start_time"]))

    print("[3/4] Segment frames + local VLM...", flush=True)
    t_stage_start = time.perf_counter()
    prep_workers = max(1, int(args.prep_workers))
    t_prep_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=prep_workers) as ex:
        packed = list(
            ex.map(
                lambda d: _prep_segment(
                    video_path, d, args.num_vlm_frames, args.decode_threads
                ),
                dets_sorted,
            )
        )
    timings["stage3a_frame_prep_sec"] = time.perf_counter() - t_prep_start

    t_vlm_load_start = time.perf_counter()
    vlm = QwenLocalVLM(args.vlm_model, device=device_vlm, torch_dtype="auto")
    timings["stage3b_vlm_load_sec"] = time.perf_counter() - t_vlm_load_start

    segments_out: list[dict] = []
    segment_timings: list[dict[str, float | int]] = []
    t_vlm_infer_start = time.perf_counter()
    infer_inputs = [(det, imgs, len(imgs), time.perf_counter()) for det, imgs in packed]
    infer_results: list[tuple[dict, str, int, float]] = []
    requested_bs = int(args.vlm_batch_size)
    bs = len(infer_inputs) if requested_bs <= 0 else max(1, requested_bs)
    for i in range(0, len(infer_inputs), bs):
        chunk = infer_inputs[i : i + bs]
        dets = [x[0] for x in chunk]
        img_groups = [x[1] for x in chunk]
        num_imgs = [x[2] for x in chunk]
        starts = [x[3] for x in chunk]
        insts = [args.vlm_instruction] * len(chunk)

        descs = vlm.describe_batch(
            img_groups,
            insts,
            max_new_tokens=int(args.vlm_max_new_tokens),
        )
        if len(descs) < len(chunk):
            descs = descs + [""] * (len(chunk) - len(descs))
        descs = [_short_one_sentence(d, max_chars=40) for d in descs[: len(chunk)]]
        infer_results.extend(zip(dets, descs, num_imgs, starts))

    for det, desc, num_imgs, t_seg_start in infer_results:
        line = _format_line(det["start_time"], det["end_time"], desc or "")
        seg = {
            "start_time": float(det["start_time"]),
            "end_time": float(det["end_time"]),
            "score": float(det.get("score", 0.0)),
            "description": desc.strip(),
            "line": line,
        }
        if "class_id" in det:
            seg["class_id"] = det["class_id"]
        if "class_name" in det:
            seg["class_name"] = det["class_name"]
        segments_out.append(seg)
        segment_timings.append(
            {
                "start_time": float(det["start_time"]),
                "end_time": float(det["end_time"]),
                "elapsed_sec": time.perf_counter() - t_seg_start,
                "num_frames": int(num_imgs),
            }
        )
    timings["stage3c_vlm_infer_sec"] = time.perf_counter() - t_vlm_infer_start
    if segment_timings:
        seg_elapsed = np.array([float(x["elapsed_sec"]) for x in segment_timings], dtype=np.float64)
        timings["vlm_segment_count"] = int(seg_elapsed.size)
        timings["vlm_seg_p50_sec"] = float(np.percentile(seg_elapsed, 50))
        timings["vlm_seg_p90_sec"] = float(np.percentile(seg_elapsed, 90))
        timings["vlm_seg_p95_sec"] = float(np.percentile(seg_elapsed, 95))
        total_vlm_sec = float(timings["stage3c_vlm_infer_sec"])
        timings["vlm_segments_per_sec"] = float(seg_elapsed.size / total_vlm_sec) if total_vlm_sec > 0 else 0.0
    else:
        timings["vlm_segment_count"] = 0
        timings["vlm_seg_p50_sec"] = 0.0
        timings["vlm_seg_p90_sec"] = 0.0
        timings["vlm_seg_p95_sec"] = 0.0
        timings["vlm_segments_per_sec"] = 0.0

    t_vlm_unload_start = time.perf_counter()
    vlm.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timings["stage3d_vlm_unload_sec"] = time.perf_counter() - t_vlm_unload_start
    timings["stage3_total_sec"] = time.perf_counter() - t_stage_start

    segments_out.sort(key=lambda s: s["start_time"])
    lines = [s["line"] for s in segments_out]

    t_stage_start = time.perf_counter()
    payload = {
        "video": str(video_path),
        "mae_meta": {k: v for k, v in mae_meta.items() if k != "step_time"},
        "actionformer": {
            "config": str(Path(args.af_config).resolve()),
            "checkpoint": str(Path(args.af_ckpt).resolve()),
            "score_thresh": args.score_thresh,
            "fps": fps,
            "duration": float(duration),
            "feat_stride": feat_stride,
        },
        "vlm_model": args.vlm_model,
        "num_vlm_frames": args.num_vlm_frames,
        "detections_raw": detections,
        "segments": segments_out,
        "lines": lines,
        "timings": {
            **timings,
            "segment_vlm": segment_timings,
        },
    }

    result_json = out_dir / "result.json"
    lines_txt = out_dir / "lines.txt"
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    timings["stage4_output_write_sec"] = time.perf_counter() - t_stage_start
    timings["total_sec"] = time.perf_counter() - t_total_start
    timing_path.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[4/4] Wrote {result_json} and {lines_txt}", flush=True)
    print(f"[timing] total={timings['total_sec']:.3f}s", flush=True)
    print(
        "[timing] stage1={:.3f}s stage2={:.3f}s stage3={:.3f}s stage4={:.3f}s".format(
            float(timings["stage1_feature_extract_sec"]),
            float(timings["stage2_actionformer_sec"]),
            float(timings["stage3_total_sec"]),
            float(timings["stage4_output_write_sec"]),
        ),
        flush=True,
    )
    print(
        "[timing] vlm_infer={:.3f}s seg_count={} seg/s={:.3f} p50={:.3f}s p90={:.3f}s p95={:.3f}s batch_size={}".format(
            float(timings["stage3c_vlm_infer_sec"]),
            int(timings["vlm_segment_count"]),
            float(timings["vlm_segments_per_sec"]),
            float(timings["vlm_seg_p50_sec"]),
            float(timings["vlm_seg_p90_sec"]),
            float(timings["vlm_seg_p95_sec"]),
            int(args.vlm_batch_size),
        ),
        flush=True,
    )
    print("[tuning] vlm_batch_size を 2/4/8 で比較し最速値を採用してください", flush=True)
    print(f"[timing] details -> {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
