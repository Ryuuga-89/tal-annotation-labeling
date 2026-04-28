# 2026-04-28 実装計画: Phase 2 推論ラッパー + 学習パイプライン

前提: [2026-04-28-progress.md](2026-04-28-progress.md) 参照  
参照設計: [2026-04-27-tal-design.md](2026-04-27-tal-design.md)

---

## 📋 実装範囲

### Phase 2: 区間検出 (Temporal Action Localization)
- **推論ラッパー**: `.npy` 特徴 → ActionFormer → JSON 検出結果
- **学習パイプライン**: train/val ループ
- **評価スクリプト**: mAP@tIoU, AR@10 計測

**スコープ外 (Phase 3)**:
- キャプション生成 (motion_detail → caption)
- RoIAlign, Q-Former, Qwen LoRA
- 補助分類ヘッド (body_part, action_type 等)

---

## 🎯 実装タスク

### Task 1: 推論スクリプト (`scripts/infer_tal.py`)

**目的**: 学習済みモデル + `.npy` 特徴 → JSON 検出結果

**入力**:
```
--ckpt /path/to/checkpoint.pth.tar
--config codes/ActionFormer/configs/tal_motion_vit_b.yaml
--feat-dir data/features/30s_mae_b_16_2
--video-ids test_video_001 test_video_002 ...  # オプション、未指定=全て
--output-json detections.json
--nms-thresh 0.5 (オプション)
```

**出力** (`detections.json`):
```json
{
  "video_id": {
    "fps": 10.0,
    "duration": 30.0,
    "detections": [
      {
        "start_time": 1.2,
        "end_time": 3.4,
        "score": 0.95
      },
      ...
    ]
  },
  ...
}
```

**実装詳細**:
- モデル読み込み → eval mode
- 特徴読み込み (NumPy `.npy`)
- 推論 (バッチ処理; BS=8 程度)
- NMS 適用 (IoU threshold + score threshold)
- JSON 出力

**コード構成**:
```python
def load_model(config_path, ckpt_path, device):
def infer_features(model, feats, fps, duration, feat_stride, num_frames):
def apply_nms(detections, nms_thresh, score_thresh):
def main():
```

---

### Task 2: 学習スクリプト (`scripts/train_tal.py`)

**目的**: ActionFormer + tal_motion dataset で学習

**入力**:
```
--config codes/ActionFormer/configs/tal_motion_vit_b.yaml
--output-folder outputs/tal_motion_vit_b
--devices 0 1 6 7  (CUDA_VISIBLE_DEVICES 代わり)
--resume outputs/tal_motion_vit_b/.../epoch_010.pth.tar  (オプション)
--tag experiment_v1  (タイムスタンプの代わりにタグ)
```

**学習ハイパーパラメータ** (A100 80GB 制約):
- **バッチサイズ**: 32 (4 GPU × 8)
- **学習率**: 1e-4 (初期値)
- **エポック**: 50-100
- **ウォームアップ**: 5 エポック
- **スケジューラ**: MultiStepLR (decay@[30, 60])
- **最適化**: AdamW (weight_decay=1e-5)
- **損失**: focal loss + DIoU regression
- **EMA**: enabled (decay=0.9998)

**出力**:
```
outputs/tal_motion_vit_b/tal_motion_vit_b_experiment_v1/
├── configs.yaml              # 実行時config (再現性)
├── logs/
│   └── events.out.tfevents   # TensorBoard
├── epoch_001.pth.tar         # チェックポイント
├── epoch_002.pth.tar
└── ...
```

**実装詳細**:
- ActionFormer 標準の `train.py` ロジック流用
- YAML config 読み込み
- DataLoader (train/val)
- Model, Optimizer, Scheduler 初期化
- Train/Val epoch loop
- EMA model 管理
- チェックポイント保存

**コード構成**:
```python
def main(args):
    # 1. config load
    # 2. dataset / dataloader
    # 3. model, optimizer, scheduler
    # 4. resume (optional)
    # 5. training loop
    #    - train_one_epoch()
    #    - valid_one_epoch()
    #    - save checkpoint
```

---

### Task 3: 評価スクリプト (`scripts/eval_tal.py`)

**目的**: 検証セットで mAP@tIoU, AR@10 計測

**入力**:
```
--config codes/ActionFormer/configs/tal_motion_vit_b.yaml
--ckpt outputs/tal_motion_vit_b/.../epoch_050.pth.tar  (or フォルダ → 最新)
--devices 0 1 6 7
--topk 100  (検出上限数)
--output-dir eval_results
```

**出力** (JSON + stdout):
```json
{
  "mAP@0.3": 0.65,
  "mAP@0.4": 0.60,
  "mAP@0.5": 0.55,
  "mAP@0.6": 0.45,
  "mAP@0.7": 0.35,
  "average_mAP": 0.52,
  "AR@10": 0.78
}
```

**実装詳細**:
- ActionFormer 標準の `eval.py` ロジック流用
- ANETdetection evaluator (ActionFormer の実装)
- mAP@tIoU 計算 (tIoU = 0.3, 0.4, 0.5, 0.6, 0.7)
- AR@10 計算
- 結果出力 (JSON + console log)

**コード構成**:
```python
def main(args):
    # 1. config load
    # 2. dataset / dataloader (val split)
    # 3. model load + ckpt
    # 4. evaluator init
    # 5. valid_one_epoch() with eval mode
    # 6. result output
```

---

### Task 4: テスト実行

#### 4a. 推論テスト
```bash
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

cd /lustre/work/mt/okamura/tal-annotation-labeling

# 既存ダミーモデル (または最初の学習エポック後)
uv run python scripts/infer_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --ckpt models/dummy/epoch_001.pth.tar \
  --feat-dir data/features/30s_mae_b_16_2 \
  --video-ids test_001 test_002 \
  --output-json /tmp/infer_test.json
```

**期待出力**: `/tmp/infer_test.json` に検出結果

#### 4b. 学習テスト (1エポック)
```bash
uv run python scripts/train_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --output-folder outputs/test_run \
  --devices 0 \
  --tag smoke_test \
  --max-epochs 1
```

**期待出力**: 
- `outputs/test_run/tal_motion_vit_b_smoke_test/`
- チェックポイント + logs

#### 4c. 評価テスト
```bash
uv run python scripts/eval_tal.py \
  --config codes/ActionFormer/configs/tal_motion_vit_b.yaml \
  --ckpt outputs/test_run/tal_motion_vit_b_smoke_test/epoch_001.pth.tar \
  --devices 0 \
  --output-dir /tmp/eval_test
```

**期待出力**: 
- JSON with mAP@tIoU, AR@10
- Console log with results

---

## 📁 ファイル配置

```
/lustre/work/mt/okamura/tal-annotation-labeling/
├── scripts/
│   ├── infer_tal.py              ← 新規
│   ├── train_tal.py              ← 新規
│   ├── eval_tal.py               ← 新規
│   ├── extract_features_parallel.sh
│   └── ...
├── codes/
│   └── ActionFormer/
│       └── configs/
│           └── tal_motion_vit_b.yaml  ✅ (既存)
├── data/
│   ├── features/30s_mae_b_16_2/  ✅ (サンプル特徴)
│   ├── splits/v1/                ⏳ (GPU側生成待ち)
│   └── aux_stats/v1/             ⏳ (GPU側生成待ち)
└── outputs/                       ← 学習出力先
```

---

## 🔧 実装優先度

| # | タスク | 優先度 | 依存 |
|---|---|---|---|
| 1 | Task 1: 推論 | 🔴 高 | config + tal_motion.py |
| 2 | Task 2: 学習 | 🔴 高 | 推論スクリプト (validate code quality) |
| 3 | Task 3: 評価 | 🟡 中 | 学習スクリプト |
| 4 | Task 4: テスト | 🟡 中 | 1, 2, 3 全て |

---

## 💾 設定ファイル確認

現在の ActionFormer config (`codes/ActionFormer/configs/tal_motion_vit_b.yaml`):
- ✅ 存在確認必要
- dataset_name: "tal_motion"
- num_classes: 1 (class-agnostic)
- input_dim: 768 (VideoMAE v2-B)
- feat_stride: 2
- max_seq_len: 192 (30s)

---

## ✅ チェックリスト

- [x] Task 1: 推論スクリプト実装 (`scripts/infer_tal.py`)
- [x] Task 2: 学習スクリプト実装 (`scripts/train_tal.py`)
- [x] Task 3: 評価スクリプト実装 (`scripts/eval_tal.py`)
- [ ] Task 4a: 推論テスト実行
- [ ] Task 4b: 学習テスト実行 (1 epoch)
- [ ] Task 4c: 評価テスト実行
- [ ] 設定最適化 (バッチサイズ、学習率等)
- [ ] ドキュメント更新

---

## 📝 実装完了確認（2026-04-28）

### Task 1: 推論スクリプト (`scripts/infer_tal.py`) ✅
- 機能:
  - `.npy` 特徴ファイル読み込み
  - ActionFormer モデルで推論実行
  - NMS（Soft-NMS）適用
  - JSON 出力形式: `{video_id: {fps, detections: [{start_time, end_time, score}, ...]}}`
- 主要関数:
  - `load_model()`: config + checkpoint からモデル読み込み
  - `load_features()`: `.npy` ファイル読み込みとdownsampling
  - `infer_batch()`: バッチ推論実行
  - `soft_nms_1d()`: 1D temporal NMS
- CLI 引数:
  - `--config`: ActionFormer config YAML
  - `--ckpt`: checkpoint .pth.tar
  - `--feat-dir`: 特徴ファイルディレクトリ
  - `--output-json`: 出力 JSON ファイル
  - `--score-thresh`: 信頼度閾値 (default: 0.3)
  - `--nms-thresh`: NMS IoU 閾値 (default: 0.5)
  - `--device`: CUDA device (default: 0)

### Task 2: 学習スクリプト (`scripts/train_tal.py`) ✅
- 機能:
  - ActionFormer + tal_motion dataset で学習
  - Train/Val loop
  - Model EMA 管理
  - TensorBoard ロギング
  - チェックポイント自動保存（毎 epoch）
- 主要設定:
  - バッチサイズ: 32 (config で調整可能)
  - 学習率: 1e-4 × (GPU数)
  - スケジューラ: MultiStepLR
  - EMA decay: 0.9998
  - 出力: `outputs/<cfg_name>_<tag>/epoch_*.pth.tar`
- CLI 引数:
  - `--config`: ActionFormer config YAML
  - `--output-folder`: チェックポイント出力先 (default: "outputs")
  - `--tag`: 実験タグ (default: timestamp)
  - `--devices`: CUDA device indices (e.g., 0 1 6 7)
  - `--resume`: チェックポイントファイル (optional)
  - `--max-epochs`: 最大エポック数 (config 優先)

### Task 3: 評価スクリプト (`scripts/eval_tal.py`) ✅
- 機能:
  - 検証/テストセットで評価
  - mAP@tIoU (tIoU = 0.3, 0.4, 0.5, 0.6, 0.7) 計算
  - AR@10 計算
  - JSON + console ログ出力
  - EMA モデル優先ロード
- 出力形式:
  - JSON: `{mAP@0.3, mAP@0.4, ..., mAP@0.7, average_mAP, AR@10, ...}`
- CLI 引数:
  - `--config`: ActionFormer config YAML
  - `--ckpt`: checkpoint .pth.tar or directory (最新自動選択)
  - `--devices`: CUDA device indices
  - `--epoch`: 特定エポック指定 (default: latest)
  - `--output-dir`: 結果出力ディレクトリ
  - `--topk`: 最大検出数 (default: 100)
  - `--saveonly`: 予測保存のみ、評価スキップ

---

**作成日時**: 2026-04-28  
**次ステップ**: Task 1 実装開始
