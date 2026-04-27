# codes/VideoMAEv2

VideoMAE V2 を特徴抽出器として使う Phase 1 実装。30s/10s チャンクの mp4 と
`30s_chunks_action_31detail_2/*.json` のアノテーションを入力に、
sliding-window 特徴 `[N, embed_dim]` を npy で出力する。

## 構成

```
codes/VideoMAEv2/
├── dataset/
│   ├── annotation.py      # JSON パース + 区間 ↔ step index 変換
│   ├── video_loader.py    # decord + target_fps リサンプリング + 224x224 squash
│   └── chunk_dataset.py   # 1 動画 = 1 PyTorch Dataset (clip ごとの item)
├── models/
│   └── backbone.py        # repos/VideoMAEv2/models のラッパー (ViT-B / ViT-g 切替)
├── configs/
│   ├── extract_vit_b.yaml
│   └── extract_vit_g.yaml
├── scripts/
│   └── extract.sh
└── extract_features.py    # 全動画分の特徴抽出を実行するエントリポイント
```

## サンプリング規約

- **target_fps = 10**: 全動画をこの fps に揃えて時間ベース sampling。10fps 動画は
  そのまま、30fps 動画は 3 frame に 1 枚使用。アノテーションが秒単位なので、特徴の
  時間軸を全動画で統一する目的。
- **window = 16, stride = 2** (= 1 step が 0.2 秒、1 window が 1.6 秒)。
- **空間**: `Resize((224, 224))` で直接 squash (公式 `extract_tad_feature.py` と同一)。
  アスペクト比は崩れるが、VideoMAE V2 の K710 fine-tune 前処理と一致するので特徴
  分布が保たれる。`resize_mode: shortside_crop` も config で選択可能。

step `i` の時間スパン:

```
T_i = [i * stride / target_fps, (i * stride + window) / target_fps)
    = [0.2 * i, 0.2 * i + 1.6)
```

## 出力フォーマット

```
<out_dir>/<stem>.npy    # shape [N, embed_dim], float16 (デフォ)
<out_dir>/<stem>.json   # 抽出時のメタ (target_fps, stride, num_steps, embed_dim, ...)
<out_dir>/index.json    # 全動画分のサマリ (config と各動画の status)
```

## 実行

公式 K710 蒸留済重み (`vit_b_k710_dl_from_giant.pth`) を `models/vit_b/` に配置済の場合:

```bash
# 単一 GPU (デフォルト = GPU 0)
bash codes/VideoMAEv2/scripts/extract.sh vit_b

# 複数 GPU (動画リストをシャード分割して並列実行)
GPU=0,1,2,3 bash codes/VideoMAEv2/scripts/extract.sh vit_b

# 物理 GPU 番号を指定 (例: 2 と 5)
GPU=2,5 bash codes/VideoMAEv2/scripts/extract.sh vit_b

# 4 動画だけドライラン
GPU=0 bash codes/VideoMAEv2/scripts/extract.sh vit_b "" 4

# CPU で動作確認
GPU=cpu bash codes/VideoMAEv2/scripts/extract.sh vit_b
```

環境変数:

- `GPU`: `0` / `0,1,2,3` / `cpu`。デフォルト `0`。
- `ANNOT_ROOT_DIR` / `VIDEO_DATA_DIR`: アノテーション・動画ディレクトリ。
- `OUT_DIR`: 特徴量出力先。デフォルトは `data/features/30s_mae_{b,g}_16_2`。

直接 Python で実行する場合 (デバッグ用):

```bash
PYTHONPATH=codes uv run python -m VideoMAEv2.extract_features \
  --config codes/VideoMAEv2/configs/extract_vit_b.yaml \
  --annot-dir "$ANNOT_ROOT_DIR" \
  --video-dir "$VIDEO_DATA_DIR" \
  --out-dir   data/features/30s_mae_b_16_2 \
  --ckpt-path models/vit_b/vit_b_k710_dl_from_giant.pth \
  --shard-id 0 --num-shards 1
```

`--limit N` で先頭 N 動画だけ処理 (動作確認用)。
`--overwrite` で既存 `.npy` を上書き。

### 並列化の仕組み

複数 GPU 指定時は **DDP ではなくシャード並列**:

- `K` 個の GPU を指定すると、`K` 個の Python プロセスが並行起動。
- プロセス `k` (0-based) は `sorted(annot_dir/*.json)[k::K]` の動画だけを処理。
- 各プロセスは異なる `CUDA_VISIBLE_DEVICES` で物理 GPU 1 枚に固定される。
- 出力 npy は同一の `out_dir/` にフラットに保存 (動画 stem ごとに 1 ファイル
  なので衝突しない)。
- 進捗・ログは `out_dir/logs/shard{k}of{K}.log` に分離。
- サマリ json は `index.shardKKofNN.json` として shard 単位に書き出し。
  全動画ぶんの一覧が必要なら `*.json` を後でマージするか、`out_dir/*.npy` を
  そのまま enumerate すれば良い。

DDP を使わない理由: 抽出は per-video で gradient 不要 = 通信不要。
シャード並列の方が起動コスト・依存・ログが単純。

## アノテーションを step index に変換する

```python
import sys; sys.path.insert(0, "codes")
from VideoMAEv2.dataset.annotation import load_annotation, actions_to_step_segments

rec = load_annotation("/raid/.../00011ec9....json")
segs = actions_to_step_segments(
    rec.actions,
    num_steps=143,         # メタ json の num_steps
    target_fps=10.0,
    stride=2,
    window_size=16,
    overlap="any",          # "any" | "center" | "inside"
)
for s in segs:
    print(s.action.start_time, s.action.end_time, "→ steps", s.step_start, s.step_end)
```

## スモークテスト (合成データ)

GPU・実データなしで動作確認する場合:

```bash
uv run python codes/VideoMAEv2/tests/make_smoke_data.py /tmp/tal_smoke
PYTHONPATH=codes uv run python -m VideoMAEv2.extract_features \
  --config codes/VideoMAEv2/configs/extract_vit_b.yaml \
  --annot-dir /tmp/tal_smoke/annot \
  --video-dir /tmp/tal_smoke/video \
  --out-dir   /tmp/tal_smoke/features \
  --device cpu --dtype float32 --batch-size 4
```

## 現状の制約 / 既知の事項

- 重み未指定 (`--ckpt-path` 省略 or 空文字) でも実行は通るが、ランダム重みのため
  特徴は意味を持たない。**本番抽出時は必ず checkpoint を指定**。
- timm 0.6+ では `timm.create_model` 経由で本リポジトリのモデルを作るとエラー
  (`pretrained_cfg` kwarg 非対応) になるため、ファクトリ関数を直接呼ぶラッパー
  (`models/backbone.py`) を経由する。
- `dataset/__init__.py` を import すると decord が必須になるので、annotation だけ
  使いたい場合は `from VideoMAEv2.dataset.annotation import ...` と直接 import すると軽い。
