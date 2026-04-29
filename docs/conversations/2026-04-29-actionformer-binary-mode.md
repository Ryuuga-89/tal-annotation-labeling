# 2026-04-29 ActionFormer 2値モード切り替え

## 決定

- `action_type` 多クラスではなく、まずは **2値（action / background）** で学習・推論する。
- 目的は区間検出の安定化。caption は後段 LLM が生成する。

## 反映内容

- `TalMotionDataset` を class-agnostic (`num_classes=1`) に戻し、`labels` は全て 0 に設定。
- `train_tal.py` / `infer_tal.py` / `eval_tal.py` から、
  `action_type` vocab による `num_classes` 自動上書き処理を削除。
- `tal_motion_vit_b.yaml` の `num_classes` コメントを
  class-agnostic 用途に更新。

## 備考

- `scripts/build_action_type_vocab.py` は将来の多クラス化用として保持。
