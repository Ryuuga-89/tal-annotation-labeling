# E2E-TAD プロジェクト概要

`repos/E2E-TAD/` は、論文 **"An Empirical Study of End-to-end Temporal Action Detection"** (CVPR 2022, Liu et al.) の公式実装。TadTR の拡張版であり、特徴量入力に加えて **生フレームからのフルエンドツーエンド学習** をサポートする。

- 論文: https://arxiv.org/abs/2204.02932
- 元リポジトリ: https://github.com/xlliu7/E2E-TAD

## 概要

End-to-End な Temporal Action Detection (TAD) について以下を実証研究したもの:

- **End-to-End 学習の有効性**: 特徴量固定方式に対し最大 11% の性能向上を確認。
- **設計選択肢の影響**を体系的に検証: detection head, video encoder, 空間/時間解像度, フレームサンプリング法, multi-scale 特徴融合 など。
- **ベースライン検出器の確立**: SlowFast + TadTR ヘッドで RGB のみで MUSES / AFSD を上回り、TITAN Xp 単発で 5076 fps を達成。

本コードは **TadTR の上位互換** であり、動画入力・特徴量入力の双方をサポートする。

### 主要結果
| Dataset | Encoder | Head | SR / TR | Avg. mAP | GPU Mem |
|---|---|---|---|---|---|
| THUMOS14 | SlowFast R50 4x16 | TadTR | 96 / 10FPS | 54.2 | 7.6G |
| ActivityNet | SlowFast R50 4x16 | TadTR | 96 / 384 | 35.10 | 11G |
| ActivityNet | TSM R50 | TadTR | 96 / 96 | 34.14 | 10G |

SR = 空間解像度, TR = 時間解像度 (THUMOS14 はサンプリング FPS、ActivityNet は1動画あたりサンプリングフレーム数)。

## ディレクトリ構成

```
E2E-TAD/
├── main.py              # 学習・評価エントリポイント
├── engine.py            # train / eval ループ
├── opts.py              # オプション・設定
├── demo.py              # 簡易動作確認
├── configs/
│   ├── thumos14_e2e_slowfast_tadtr.yml   # E2E (SlowFast + TadTR) 設定
│   └── thumos14_i3d2s_tadtr.yml          # 特徴量入力 (TadTR 互換) 設定
├── datasets/
│   ├── tad_dataset.py
│   ├── tad_eval.py
│   └── e2e_lib/         # 動画フレーム I/O・サンプリング関連
├── models/
│   ├── tadtr.py         # 検出ヘッド (TadTR)
│   ├── transformer.py   # Deformable Transformer
│   ├── matcher.py       # Hungarian matcher
│   ├── position_encoding.py
│   ├── custom_loss.py
│   ├── video_encoder.py # 動画エンコーダ統合層
│   ├── video_encoder_archs/
│   │   ├── slowfast.py
│   │   ├── resnet3d.py
│   │   └── tsm.py
│   └── ops/             # CUDA 拡張 (RoIAlign1D 等)
├── tools/
│   ├── extract_frames.py   # 動画 → フレーム抽出
│   ├── prepare_data.py     # メタ情報 JSON 生成
│   ├── flop_count.py
│   └── test_runtime.py
├── scripts/
├── docs/
│   └── 1_train_on_your_dataset.md   # カスタムデータセット利用ガイド
├── Evaluation/
└── requirements.txt
```

## 入力データ仕様

E2E モードの入力は **動画フレーム画像列** (mp4 そのものではなく、事前に抽出したフレーム)。`tools/extract_frames.py` で動画 → フレーム抽出、`tools/prepare_data.py` でメタ情報 JSON を生成できる。

THUMOS14 用ファイル例:
- `data/thumos14/th14_annotations_with_fps_duration.json` — 行動区間アノテーション
- `data/thumos14/th14_img10fps_info.json` — 各動画の FPS / フレーム数メタ
- 10fps 抽出フレーム (`thumos14_img10fps.tar`)
- 事前学習モデル `thumos14_e2e_slowfast_tadtr_reference.pth`

なお、特徴量入力モード (TadTR 互換) もそのまま利用可能。

## 実行方法 (公式)

評価:
```bash
python main.py --cfg configs/thumos14_e2e_slowfast_tadtr.yml \
               --eval --resume CKPT_PATH
```

簡易テスト:
```bash
python demo.py --cfg configs/thumos14_e2e_slowfast_tadtr.yml
```

CUDA 拡張 (RoIAlign1D 等) は `models/ops/` で `python setup.py build_ext --inplace` でビルドする。

**学習コードは README 上「TBD」とされているが、リポジトリ内には学習用コード (`main.py` の train 経路、`engine.py`) が含まれており動作する状態とみられる**。READMEの記述に従えば公式サポートは未確定なので、利用時は要検証。

## TadTR との違い

| 項目 | TadTR | E2E-TAD |
|---|---|---|
| 入力 | 事前抽出特徴量のみ | 生フレーム / 特徴量 両対応 |
| Video Encoder | なし (外部で抽出) | SlowFast / TSM / ResNet3D を内蔵 |
| 学習方式 | ヘッドのみ学習 | エンコーダ含めた E2E 学習可 |
| IO 要件 | 軽い | フレーム読み込み多発、SSD 推奨 |
| 計算コスト | 軽量 | 重い (GPU メモリ 7〜11G 程度) |

## 本プロジェクト (tal-annotation-labeling) との関連

本プロジェクトは 30 秒程度の生 mp4 (`30s_chunks/*.mp4`) に対して TAL でアノテーションラベルを生成する。E2E-TAD は **生動画 (フレーム) から直接学習・推論できる** 点で TadTR より本タスクに適合する候補。検討論点:

- フレーム抽出パイプラインを `tools/extract_frames.py` 等で 63k 動画分構築する必要がある (SSD 推奨)。
- データローダおよびアノテーション JSON を本プロジェクト形式 (`30s_chunks_action_31detail_2/*.json`) → E2E-TAD 想定形式へ変換する必要あり。`docs/1_train_on_your_dataset.md` がカスタムデータ向けガイド。
- A100 80GB 単発であれば SlowFast R50 4x16 構成 (約 7.6G) でも十分余裕がある。バッチサイズ・空間/時間解像度を増やす方向の調整が可能。
- 30 秒という比較的短い動画長は ActivityNet 設定 (固定フレーム数サンプリング) の流儀が相性良いと考えられる。
