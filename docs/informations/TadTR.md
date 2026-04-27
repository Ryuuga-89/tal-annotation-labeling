# TadTR プロジェクト概要

`repos/TadTR/` に配置されているのは、論文 **"End-to-end Temporal Action Detection with Transformer"** (IEEE TIP 2022, Liu et al.) の公式実装である。

- 論文: https://arxiv.org/abs/2106.10271
- 元リポジトリ: https://github.com/xlliu7/TadTR

## 概要

TadTR (Temporal action detection TRansformer) は、動画内の**時間的行動検出 (Temporal Action Detection, TAD)** を End-to-End で行う Transformer ベースの検出器。DETR / Deformable DETR の思想を時間軸方向に拡張したものに相当する。

### 特徴
- **Simple**: set-prediction による単一ネットワーク構成。プロポーザル生成段を持たない。
- **Flexible**: アンカー設計や NMS などの手作業設計を排除。
- **Sparse**: ActivityNet で 10 個程度の検出のみを生成し、計算コストが低い。
- **Strong**: HACS / THUMOS14 で SOTA。RTD-Net や AGT などの同時期 Transformer 系手法を上回る。

### 主要結果 (公式数値)
| Dataset | Feature | Avg. mAP |
|---|---|---|
| THUMOS14 | I3D 2stream | 56.7 |
| HACS Segments | I3D RGB | 32.09 |
| ActivityNet-1.3 | TSP | 36.75 |

## ディレクトリ構成

```
TadTR/
├── main.py              # 学習・評価のエントリポイント
├── engine.py            # train / eval ループ
├── opts.py              # 各種オプション・設定
├── demo.py              # 簡易動作確認スクリプト
├── configs/
│   └── thumos14_i3d2s_tadtr.yml   # 実験設定 (現状 THUMOS14 のみ)
├── datasets/            # TAD データセット読み込み・評価コード
│   ├── tad_dataset.py
│   └── tad_eval.py
├── models/              # モデル本体
│   ├── tadtr.py         # TadTR 本体 (set prediction head 等)
│   ├── transformer.py   # Deformable Transformer
│   ├── matcher.py       # Hungarian matcher
│   ├── position_encoding.py
│   ├── custom_loss.py
│   └── ops/             # CUDA 拡張 (RoIAlign1D 等)
├── scripts/
│   ├── run_parallel.sh
│   └── test_reference_models.sh
├── Evaluation/          # 公式評価ツール
└── requirements.txt
```

## 入力データ仕様

公式コードは**動画そのものではなく、事前抽出された特徴量 (I3D 2-stream など)** を入力とする (E2E ではない版)。動画から直接学習するフルエンドツーエンド版は別リポジトリ [E2E-TAD](https://github.com/xlliu7/E2E-TAD)。

THUMOS14 用には以下を `data/thumos14/` 以下に配置する想定:
- I3D 2-stream 特徴量 (`I3D_2stream_Pth/`)
- アノテーション JSON (`th14_annotations_with_fps_duration.json`)
- 特徴量メタ情報 (`th14_i3d2s_ft_info.json`)
- 事前学習済モデル (`thumos14_tadtr_reference.pth`)

## 実行方法 (公式)

学習:
```bash
python main.py --cfg configs/thumos14_i3d2s_tadtr.yml
```

事前学習モデルでの評価:
```bash
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

CUDA 拡張 (RoIAlign1D) は `models/ops/` で `python setup.py build_ext --inplace` によりビルドする。GPU が無い場合は `opts.py` 内 `disable_cuda=True` を設定する (ただし actionness regression は無効化され性能低下)。

## 本プロジェクト (tal-annotation-labeling) との関連

本リポジトリのタスクは「30 秒程度の動画に対し TAL 技術でアノテーションラベルを生成する」ことであり、TadTR はその**ベース手法候補**として参照されているとみられる。利用に際しての論点:

- 公式 TadTR は事前抽出特徴量入力前提のため、生 mp4 (`30s_chunks/*.mp4`) を直接扱うには別途特徴抽出パイプラインか、E2E-TAD 版の利用が必要。
- 想定 GPU は A100 80GB 単発であり、TadTR 自体は軽量なので単一 GPU で十分動作する。
- 既定実装は THUMOS14 のみサポート。本プロジェクトのアノテーション形式 (`30s_chunks_action_31detail_2/*.json`) に合わせたデータローダ実装が別途必要となる。
