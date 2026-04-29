# 2026-04-29 進捗更新（step基準化 + W&B統合）

## 目的

- 学習・検証・保存を実運用向けに整理し、  
  大規模データ（約6万件）でも運用可能な形にする。
- W&Bで学習/推論/評価を統一的に追跡できるようにする。

## 実施内容

### 1. 入力指定の拡張（txtベース）

- `scripts/train_tal.py`
  - `--feature-list-txt` を追加（1行1パス）
  - `.json/.npy` のどちらでも stem 解決して train subset を構成
- `scripts/infer_tal.py`
  - `--feature-list-txt` を追加
  - `.npy` 行は直接使用、`.json` 行は `--feat-dir/{stem}.npy` へ解決

### 2. GPU0の不要使用対策

- `manual_seed_all` による全GPU初期化を回避
- 主GPUを `torch.cuda.set_device(...)` で明示指定
- train/eval/infer 全スクリプトで初期化経路を調整

### 3. 学習step定義の修正（重要）

- 旧: `1 step = 1 epoch`（不適切）
- 新: **`1 step = 1 optimizer update`**
  - `global_step` を勾配更新ごとにインクリメント

### 4. 保存処理のstep基準化

- `scripts/train_tal.py`
  - `--save-every-steps` を追加
  - 指定stepごとに checkpoint 保存
  - 保存名を step基準に変更: `step_00000042.pth.tar`
  - `--save-model-only` で軽量モデル保存にも対応

### 5. 検証処理のstep基準化

- `scripts/train_tal.py`
  - `--val-every` を step基準に変更
  - 指定stepごとに `valid_one_epoch` 実行
  - 出力名: `val_pred_step_00000042.pkl`
  - 検証データは `--val-*` 引数で個別指定可能
  - `--val-feature-list-txt` による subset 検証にも対応

### 6. 多クラス（combined）対応の整備

- `codes/ActionFormer/dataset/tal_motion.py`
  - `label_mode: binary|combined`
  - `combined_vocab_file` 読み込み
  - `merge_combined_labels` のON/OFF切替
- `codes/ActionFormer/configs/tal_motion_vit_b.yaml`
  - combinedモード設定を反映
  - `num_classes` を combined vocab に合わせる構成へ

### 7. ラベル統合・語彙・可視化スクリプト

- `scripts/export_unique_action_labels.py`
  - body_part/action_type/combined のユニーク抽出
  - 統合ON/OFF（デフォルトON）
- `scripts/build_action_type_vocab.py`
  - action_type統合ロジックを追加
- `scripts/build_combined_vocab.py`（新規）
  - `body_part|action_type` vocab 生成
- `scripts/plot_label_distribution.py`（新規）
  - 件数CSV + PNG 可視化
  - 日本語フォント表示調整済み

## 実測（統合後）

- `body_part` ユニーク数: **8**
- `action_type` ユニーク数: **25**
- `combined` クラス数: **164**（目標100〜300内）

## W&B統合

### train

- `scripts/train_tal.py`
  - 学習loss/lr、検証mAP、checkpoint artifact を記録可能

### infer / eval

- `scripts/infer_tal.py`
  - 推論件数メトリクス、出力JSON artifact
- `scripts/eval_tal.py`
  - 評価メトリクス、`eval_results.json` artifact

### 共通化

- `scripts/wandb_utils.py`（新規）
  - 共通CLI引数追加
  - run初期化処理を共通化
  - project/entity/run命名規則を統一
- `docs/wandb-run-templates.md`（新規）
  - train/infer/eval 実行テンプレートを記載

## 補足

- 評価器は `ANETdetection`（`eval_tal.py`）
- `README.md` 指定のW&B設定:
  - project: `tal-annotation-labeling`
  - entity: `models-institute-of-science-tokyo`
  - run名: `{train,exp,test}_{description}_{timestamp}`
