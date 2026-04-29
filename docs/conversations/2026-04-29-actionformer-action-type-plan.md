# 2026-04-29 ActionFormer 学習方針（action_type分類）

## 合意内容

- ActionFormer は `body_part` と `action_type` を扱える設計にしつつ、**学習ターゲットは `action_type` のみ**とする。
- 最終的な caption は LLM が生成するため、ActionFormer 側は **区間検出（start/end）** を主目的とする。
- 推論出力の必須項目は `start_time`, `end_time`, `score`（必要に応じて `action_type` を保持）。

## 採用したラベル方針

- 選択肢 B を採用:
  - **分類ラベル = `action_type`**
  - `body_part` は将来拡張または補助情報として扱う

## 実装に反映した内容

1. `TalMotionDataset` を class-agnostic 固定から解除し、`action_type` を `labels` に割り当て。
2. `aux_vocab_file` 内の `action_type.label_to_id` を参照して、`num_classes` と整合チェックを実施。
3. `train_tal.py` / `infer_tal.py` / `eval_tal.py` で、`aux_vocab_file` から `num_classes` を自動反映。
4. `action_type` 語彙生成用に `scripts/build_action_type_vocab.py` を追加。

## 実行手順（要約）

```bash
uv sync
uv run python scripts/build_action_type_vocab.py \
  --annot-dir "$ANNOT_ROOT_DIR" \
  --output-json data/aux_stats/v1/action_type_vocab.json \
  --min-count 1
```

> 注: 既存 config の `aux_vocab_file` は `action_type` を含む vocab JSON を指すように設定する。

## 次アクション

- 特徴抽出完了後に vocab を生成
- smoke 学習（1 epoch）で loss と checkpoint 出力を確認
- 本学習投入後、推論結果を LLM caption パイプラインへ接続
