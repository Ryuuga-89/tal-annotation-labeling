# 2026-04-29 マルチクラス移行ログ（body_part × action_type）

## 背景

- もともと ActionFormer は `binary`（action vs background）で運用。
- 方針変更により、`body_part` と `action_type` を組み合わせた多クラス分類へ移行。
- 目標クラス数は `100~300` 程度。

## 実施内容（時系列）

1. **会話記録の追加**
   - `docs/conversations/2026-04-29-actionformer-action-type-plan.md`
   - `docs/conversations/2026-04-29-actionformer-binary-mode.md`

2. **2値モードの再整理**
   - 一時的に `binary` 構成へ戻し、学習パイプラインを安定化。
   - `tal_motion` 登録漏れ（`KeyError: tal_motion`）を修正。

3. **学習スクリプト最適化**
   - `scripts/train_tal.py` に以下を追加:
     - CUDA可視性・device idチェック
     - `cudnn.benchmark=True`, TF32有効化
     - workers上限 (`--max-workers`)
     - checkpoint頻度指定 (`--ckpt-freq`)
     - runtime config保存 (`config.runtime.yaml`)
   - `max_seq_len` の整合チェックを追加。

4. **max_seq_len不整合の修正**
   - `n_mha_win_size=9` と `backbone_arch=[2,2,4]` に対し、
     `max_seq_len=192` は不適合。
   - `codes/ActionFormer/configs/tal_motion_vit_b.yaml` の
     `dataset.max_seq_len` を `256` に更新。

5. **ラベル統合基盤の実装**
   - `scripts/export_unique_action_labels.py` を実装・拡張:
     - `body_part` / `action_type` のユニーク抽出
     - 統合ON/OFF（デフォルトON）
     - 組み合わせラベル (`body_part|action_type`) のユニーク・頻度・vocab出力
   - 統合ルールは部分一致 + 優先順で実装。

6. **vocab生成スクリプトの拡張**
   - `scripts/build_action_type_vocab.py`
     - 同様の統合処理を追加（デフォルトON、`--disable-merge`対応）
   - `scripts/build_combined_vocab.py` を新規追加
     - `body_part|action_type` の vocab を生成
     - `その他|その他` を id=0 予約
     - `min_count`, `top_n` 対応

7. **ラベル分布可視化**
   - `scripts/plot_label_distribution.py` を新規追加
     - CSV + PNG 出力
     - `body_part`, `action_type`, `combined` を集計
   - 日本語表示調整:
     - `japanize-matplotlib` 優先
     - フォントフォールバック実装

8. **多クラス学習への本格対応**
   - `codes/ActionFormer/dataset/tal_motion.py`
     - `label_mode: binary|combined` 対応
     - `combined_vocab_file` 読み込み
     - `merge_combined_labels` で統合ON/OFF
     - `labels` を `combined_label` の class id に変換
   - `codes/ActionFormer/configs/tal_motion_vit_b.yaml`
     - `label_mode: combined`
     - `combined_vocab_file: data/aux_stats/v1/combined_vocab_merged.json`
     - `num_classes: 164`
     - `multiclass_nms: True`
   - `scripts/train_tal.py` / `scripts/eval_tal.py` / `scripts/infer_tal.py`
     - `combined` モード時に vocab から `num_classes` を自動反映
   - `scripts/infer_tal.py`
     - 推論出力に `class_id` と `class_name` を付与

## 実測結果（統合後）

- `body_part` ユニーク数: **8**
- `action_type` ユニーク数: **25**
- `combined (body_part|action_type)` クラス数:
  - `combined_vocab_merged.json` の `num_classes = 164`
  - 目標レンジ（100~300）内

## 評価指標（検証）

- 検証スクリプト: `scripts/eval_tal.py`
- evaluator: `ANETdetection`（ActionFormer標準）
- 指標: `mAP@tIoU`（設定: 0.3, 0.4, 0.5, 0.6, 0.7）および平均 mAP

## 主要追加/更新ファイル

- 追加:
  - `scripts/export_unique_action_labels.py`
  - `scripts/build_combined_vocab.py`
  - `scripts/plot_label_distribution.py`
  - `docs/conversations/2026-04-29-actionformer-action-type-plan.md`
  - `docs/conversations/2026-04-29-actionformer-binary-mode.md`
- 更新:
  - `scripts/train_tal.py`
  - `scripts/infer_tal.py`
  - `scripts/eval_tal.py`
  - `scripts/build_action_type_vocab.py`
  - `codes/ActionFormer/dataset/tal_motion.py`
  - `codes/ActionFormer/configs/tal_motion_vit_b.yaml`
  - `codes/ActionFormer/__init__.py`
  - `codes/ActionFormer/README.md`

## 次ステップ（提案）

1. `train_tal.py` で 1GPU smoke 学習（1〜3 epoch）
2. `eval_tal.py` で `mAP@tIoU` を取得し、クラス混同行動を確認
3. `combined_counts.csv` を用いて `min_count`/`top_n` を再調整（必要時）
