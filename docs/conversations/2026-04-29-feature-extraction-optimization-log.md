# 2026-04-29 Feature Extraction Optimization Log

## Goal

`codes/VideoMAEv2/extract_features.py` の前処理ボトルネック（`preproc`）を削減し、  
2GPU 並列でのスループットを安定化する。

---

## What Was Changed

### 1) Timing / Throughput 設計

- `sync_cuda_timing` を追加（デフォルト `False`）
  - `torch.cuda.synchronize()` を通常実行で無効化
  - 必要時のみ `--sync-cuda-timing` で精密計測

### 2) Decode Mode 自動切替

- `decode_mode` に `auto` を導入（運用デフォルト）
- `auto_batch_threshold_frames` を追加
  - `num_target_frames` に応じて `full` / `batch` を切替
  - 試行の結果、30s クリップに対しては `threshold=320` が実運用で安定

### 3) Batch / VRAM チューニング

- `auto_batch_size` を追加（VRAM ベースのバッチ自動拡大）
- `max_batch_size_batch_decode` を追加
  - `decode_mode=batch` 時の過大バッチを抑制

### 4) 2GPU 並列ランチャー

- `scripts/run_extract_features_2gpu.sh` を追加
  - `--num-shards 2` / `--shard-id 0,1` で 2GPU 同時実行
  - 共通引数を一元化
- `trap` を追加
  - `Ctrl+C` 時に子プロセスも停止するよう改善

### 5) Random Order 実装（再現可能）

- `extract_features.py` に以下を追加:
  - `--shuffle`
  - `--seed`
- シャッフルは **sharding 前** に適用
  - 再現性を維持したまま 2GPU 分割可能

---

## Key Log Findings

### Initial State

- `cps_infer` は高いが、`preproc` が支配的
- 一部 run で `decord.get_batch` 側の待ちが長く、体感で停止に近い挙動

### After Optimization Iterations

- 2GPU 並列で合算スループットは単GPU比で改善
- 実運用で安定した帯域:
  - `clip_per_sec`（各 shard）: おおむね `~8.2 - 9.9`
  - 合算: `~16+ clips/sec`
- 依然ボトルネックは `preproc` だが、悪化せず安定化

---

## Tuned Configuration History (Main Trials)

1. **Aggressive**
   - `batch-size` 大、`decode-threads` 大
   - 初速は高いが 40 step 以降で低下しやすい
2. **Balanced (stable)**
   - `batch-size` と `decode-threads` を抑制
   - `cps_recent` の持続性が改善
3. **Current practical**
   - `decode_mode=auto`
   - `auto_batch_threshold_frames=320`
   - `auto_batch_size` 有効
   - 必要に応じて `batch-size` と `decode-threads` を微調整

---

## Operational Notes

- 本処理順はデフォルトで決定的（ソート順）  
  `--shuffle --seed <int>` で再現可能ランダム順に変更可能
- 停止時に残プロセスがある場合は `scancel <jobid>` を優先
- 監視ポイント:
  - `clip_per_sec_recent`
  - `stage_avg_sec(preproc=...)`
  - `clip_per_sec_infer` との差分

---

## Related Files

- `codes/VideoMAEv2/extract_features.py`
- `codes/VideoMAEv2/dataset/chunk_dataset.py`
- `codes/VideoMAEv2/dataset/video_loader.py`
- `scripts/run_extract_features_2gpu.sh`
- `data/features/30s_mae_b_16_2/logs/shard00of02.log`
- `data/features/30s_mae_b_16_2/logs/shard01of02.log`
