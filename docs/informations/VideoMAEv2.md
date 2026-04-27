# VideoMAEv2 プロジェクト概要

`repos/VideoMAEv2/` に配置されているのは、論文 **"VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking"** (CVPR 2023, Wang et al.) の公式実装である。

- 論文: https://arxiv.org/abs/2303.16727
- 元リポジトリ: https://github.com/OpenGVLab/VideoMAEv2
- 事前学習済みモデル: https://huggingface.co/OpenGVLab/VideoMAE2

## 概要

VideoMAE V2 は、動画向けの **Masked Autoencoder (MAE) による自己教師あり事前学習**手法をスケール化したもの。ViT を ViT-giant (約 1B パラメータ) まで拡張し、UnlabeledHybrid データセット (約 135 万動画) で大規模事前学習を行う。

### 特徴
- **Dual Masking**: encoder/decoder の双方にマスキングを適用し、giant モデルでも事前学習の計算/メモリコストを抑える。
- **段階的学習レシピ**: 大規模未ラベル動画で事前学習 → K710 など教師ラベル付きでポストプリトレ → 各下流タスクで fine-tune。
- **強い汎化**: 行動認識 (K400/600/700, SSv2, UCF, HMDB)、時空間検出 (AVA)、時間的行動検出 (THUMOS14, FineAction) で SOTA を達成。
- 蒸留版 (ViT-S, ViT-B) も公開されており、giant 教師から軽量モデルへ転移可能。

### 主要結果 (公式数値抜粋)
| タスク | データセット | Top-1 / mAP |
|---|---|---|
| Action Classification | Kinetics-400 | 88.6 (ViT-g) |
| Action Classification | Kinetics-600 | 88.8 (ViT-g) |
| Spatio-Temporal Detection | AVA v2.2 | 42.6 mAP |
| Temporal Action Detection | THUMOS14 (+ActionFormer) | 69.6 mAP |
| Temporal Action Detection | FineAction (+ActionFormer) | 18.2 mAP |

## ディレクトリ構成

```
VideoMAEv2/
├── run_mae_pretraining.py       # 自己教師あり事前学習エントリポイント
├── run_class_finetuning.py      # 行動認識 fine-tuning エントリポイント
├── extract_tad_feature.py       # TAD 用に動画特徴量を抽出するスクリプト
├── engine_for_pretraining.py    # 事前学習ループ
├── engine_for_finetuning.py     # fine-tune / 評価ループ
├── optim_factory.py             # optimizer / LR scheduler 構築
├── utils.py
├── models/
│   ├── modeling_pretrain.py     # MAE 事前学習用モデル (encoder + decoder + dual mask)
│   └── modeling_finetune.py     # 下流タスク用 ViT (vit_small/base/large/huge/giant)
├── dataset/
│   ├── datasets.py              # Kinetics 等 fine-tune 用データセット
│   ├── pretrain_datasets.py     # 事前学習用 (UnlabeledHybrid 等)
│   ├── masking_generator.py     # tube/random masking
│   ├── loader.py                # decord ベース動画ローダ
│   └── (video_)transforms.py 等
├── scripts/
│   ├── pretrain/                # 事前学習用 shell スクリプト
│   └── finetune/                # fine-tune 用 shell スクリプト
├── docs/
│   ├── INSTALL.md / DATASET.md
│   ├── PRETRAIN.md / FINETUNE.md
│   ├── MODEL_ZOO.md             # 事前学習済み重みの一覧
│   └── TAD.md                   # TAD 用特徴量抽出の手順
└── requirements.txt
```

## 入出力仕様

- **入力**: mp4 等の動画。`dataset/loader.py` が decord で読み込み、16 frames × 224×224 のクリップを生成 (デフォルト)。tubelet_size=2 で時空間トークン化。
- **出力 (事前学習)**: マスクされたパッチの再構成。
- **出力 (fine-tune)**: `num_classes` ロジット。
- **出力 (TAD 特徴抽出)**: `extract_tad_feature.py` が `forward_features` の global feature を `[N, C]` の `.npy` として保存。N はクリップ数、C は ViT 次元 (giant=1408)。

## 提供モデル (MODEL_ZOO 抜粋)

| Model | 事前学習データ | 用途 |
|---|---|---|
| `vit_small_patch16_224` (蒸留) | UnlabeledHybrid → K710 | 軽量 fine-tune |
| `vit_base_patch16_224` (蒸留) | UnlabeledHybrid → K710 | 標準 |
| `vit_large_patch16_224` | UnlabeledHybrid | 高精度 |
| `vit_huge_patch16_224` | UnlabeledHybrid | 高精度 |
| `vit_giant_patch14_224` | UnlabeledHybrid (1200ep) → K710 ft | TAD 特徴抽出に推奨 |

重みは Hugging Face `OpenGVLab/VideoMAE2` から取得。

## TAD への利用 (本プロジェクトとの関連)

本プロジェクト (tal-annotation-labeling) のタスクは「30 秒程度の動画に対し TAL 技術でアノテーションラベルを生成する」ことであり、VideoMAE V2 は**強力な動画特徴抽出バックボーン**として位置づけられる。

典型的な使い方は以下:

1. `extract_tad_feature.py` で 30s 動画から ViT-giant 特徴量 (`[N, 1408]` 等) を `.npy` として事前抽出。
   - THUMOS14 設定では sliding window: 16 frames × stride 4 (`range(0, num_frames-15, 4)`)。
   - 30s/10fps の動画 (~300 frame) なら ~71 クリップ分の特徴ベクトルが得られる。
2. その特徴を `repos/TadTR/` などの TAD ヘッド (TadTR / ActionFormer 系) に入力し、行動区間と caption を予測。
3. 公式論文では VideoMAE V2-g + ActionFormer の組合せが THUMOS14 で 69.6 mAP を達成しており、TadTR より高精度な代替パイプラインとなり得る。

### 注意点
- `vit_giant_patch14_224` の重み (約 4GB 弱) と推論時 VRAM が必要。A100 80GB 単発であれば余裕。
- `extract_tad_feature.py` はデフォルトで `--data_set THUMOS14|FINEACTION` を仮定するため、本プロジェクトの 30s チャンク用には sliding-window stride を適切に設定する追加実装が必要。
- 本実装はあくまで**特徴抽出 + 分類**までであり、行動区間検出 (TAD) ヘッド自体は含まない。区間検出は別リポジトリ (TadTR / ActionFormer 等) と組み合わせる前提。
- 本プロジェクトのターゲット出力は `{start}-{end}: {motion_detail}` のような**自由記述キャプション**であり、固定クラスを前提とする VideoMAE V2 の fine-tune パイプラインそのままでは使えない。VideoMAE V2 は**特徴抽出器**として用い、キャプション生成は別途テキストデコーダ等で実装する必要がある。
