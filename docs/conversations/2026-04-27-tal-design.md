# 30s 動画への TAL アノテーション生成: 設計議論ログ

日付: 2026-04-27

本プロジェクト (tal-annotation-labeling) における TAL パイプライン設計の検討記録。`repos/` 配下の既存リポジトリ (VideoMAEv2 / TadTR / E2E-TAD / OpenTAL / VCLTDR-GMG / DiffusionTAD) を踏まえた段階的な設計議論。

---

## Q1. VideoMAEv2 についての説明

### 概要
- 論文: VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking (CVPR 2023, Wang et al.)
- 動画向け Masked Autoencoder の大規模化 (ViT-giant, ~1B params)。
- Dual Masking で encoder/decoder 双方をマスキングし、giant モデルの事前学習コストを抑える。
- UnlabeledHybrid (~135 万動画) で事前学習 → K710 等でポストプリトレ → 下流タスクで fine-tune。
- 行動認識 (K400/600, SSv2)、時空間検出 (AVA)、TAD (THUMOS14, FineAction) で SOTA。

### 提供モデル
| Model | embed_dim | 用途 |
|---|---|---|
| ViT-S (蒸留) | 384 | 軽量 fine-tune |
| ViT-B (蒸留) | 768 | 標準 |
| ViT-L | 1024 | 高精度 |
| ViT-H | 1280 | 高精度 |
| ViT-g | 1408 | TAD 特徴抽出に推奨 |

### 本プロジェクトでの位置づけ
- 強力な動画特徴抽出バックボーン。
- `extract_tad_feature.py` で 30s 動画から sliding window 特徴 `[N, C]` を npy 化 → 下流の TAL ヘッドに渡す 2 段構成。
- 区間検出ヘッド自体は含まないため、TadTR / ActionFormer 系と組み合わせる前提。
- 出力目標が自由記述キャプション (`{start}-{end}: {motion_detail}`) のため、固定クラス fine-tune そのままでは使えず、特徴抽出器としての利用が中心。

詳細は `docs/informations/VideoMAEv2.md` 参照。

---

## Q2. 特徴量検出器として用いた場合の出力と使い方

### `forward_features` の出力
`models/modeling_finetune.py:429-448` の実装より:

```python
fc_norm( tokens.mean(dim=1) )   # use_mean_pooling=True のとき
```

全パッチトークン (時空間) の平均 → LayerNorm を取った **1 クリップ 1 ベクトル** の global feature。

### 形状
- 入力: `[B, 3, 16, 224, 224]` (16 frames, tubelet=2)。ViT-g (patch14) で `8×16×16=2048` トークン、ViT-B (patch16) で `8×14×14=1568` トークン。
- 出力: `[B, embed_dim]`。
- `extract_tad_feature.py` は 1 動画あたり sliding window でクリップ特徴を vstack し `[N, embed_dim]` の `.npy` として保存。

### 時間軸対応
THUMOS14 設定: `range(0, num_frames-15, 4)` (16f window, stride 4)。
30s/10fps (~300 frame) なら:
- stride 4: N ≈ 71 → `[71, 1408]`
- stride 16: N ≈ 18 → `[18, 1408]`

各行は元動画の `[4i, 4i+15]` フレーム (~1.6 秒幅) に対応する 1 ベクトル。動画を時間方向に圧縮した可変長特徴系列。

### 使い方
1. **特徴抽出 → ファイル化**: `extract_tad_feature.py` を本プロジェクト向けに stride 調整 (2〜4) して実行。
2. **TAL ヘッドに入力**:
   - TadTR: I3D 特徴の代わりに食わせ、segment 回帰 + 分類で区間予測。
   - ActionFormer: マルチスケール 1D Transformer で boundary/class 回帰。VideoMAE V2-g + ActionFormer は THUMOS14 で 69.6 mAP。
3. **本プロジェクト (caption 付き TAL)**:
   - 2 段構成 (TAL → text decoder)、Q-Former 経由 LLM 生成、frozen backbone + 特徴キャッシュなど。

### 注意点
- `forward_features` は mean pooling 後の 1 ベクトル/クリップ。空間/時間トークンを残したいなら `model.blocks` 最終出力を直接取る改造が必要。
- L2 正規化されていない (`fc_norm` = LayerNorm)。cosine 比較には自前で正規化。
- tubelet=2 のため時間解像度がクリップ内 8 step に落ちる。フレーム精度が必要なら stride を小さく。

---

## Q3. 区間検出パイプラインの性質 (個数 / 基準 / fine-tune 可否)

### 1. 検出される区間の数
- **複数を同時に検出**するのが TAL の標準。
- **TadTR**: 固定 `num_queries` (THUMOS14 は 40)。常に 40 候補 → スコア閾値で絞り、最終的に数個〜十数個。NMS なしでも重複しにくい DETR 設計。
- **ActionFormer**: 各時刻で dense 予測 → Soft-NMS で統合。最終出力数は可変。
- 30s 動画 (例: 16 actions) なら `num_queries` をその 1.5〜2 倍 (30〜50) が目安。

### 2. 検出基準 (=何をイベントと見なすか)
- VideoMAE V2 自体は基準を持たない。
- **TAL ヘッドの教師データ (アノテーション) が基準そのもの**。
- TadTR/ActionFormer のロス: classification (前景 vs クラス) + regression (区間)。Hungarian matcher / center-based assigner が GT 区間にクエリを対応付け。
- 「正解として与えた区間が行動」「それ以外は背景」と暗黙に学習。

### 3. fine-tune による調整
- できる。むしろ調整しないと使えない。
- 調整可能な軸:
  - **(a) アノテーション差し替え**: GT 区間/クラス集合を変えれば検出対象も変わる。
  - **(b) 時間解像度・ロス**: stride、`num_queries`、FPN レベル、tIoU 閾値、center-radius。
  - **(c) 重なり処理**: NMS 設定。TadTR は重なり対応が自然。
  - **(d) backbone fine-tune**: 計算余裕があれば backbone も学習。A100 80GB なら ViT-B まで実用、ViT-g は frozen + LoRA 推奨。

### 本プロジェクトでの「基準」候補
| 方針 | クラス定義 | 出力 |
|---|---|---|
| A. action_type のみ分類 | 31 クラス | `[start, end, action_type]` |
| B. body_part × action_type | 数百クラス | より細かい |
| C. class-agnostic | 1 クラス「動作」 | 区間のみ → caption は別 |

**C 案がキャプション生成ゴールと相性が良い** (TAL は区間だけ、motion_detail は別の言語デコーダで生成)。

---

## Q4. 推奨設計案: 2 段構成の Dense Action Captioning

### 全体パイプライン

```
mp4 (30s, 10fps)
  │
  ├─[1]─► VideoMAE V2-B (frozen, 16f window, stride 2)
  │        → 特徴系列 [N≈140, 768]
  │
  ├─[2]─► ActionFormer 系 TAL ヘッド (class-agnostic, 1 クラス "action")
  │        → 区間集合 {(start_i, end_i, score_i)}  典型 10〜30 区間
  │
  └─[3]─► 各区間ごとに:
           RoIAlign-1D で区間内特徴 → Q-Former (32 query) → 軽量 LLM デコーダ
           → 構造化 JSON: {body_part, action_type, target_object, motion_detail, ...}
           → motion_detail を最終 caption として採用
```

### [1] バックボーン: VideoMAE V2-Base
- ViT-g は強力だが学習コスト過大。ViT-B (768d) で十分。
- frozen で特徴を事前抽出 → npy キャッシュ。
- stride 2 (= 0.2 秒/step)。json の最短区間 0.3 秒に対応。30s 動画 → N ≈ 140。

### [2] TAL ヘッド: ActionFormer (class-agnostic)
- 短く密な区間に強い (TadTR は fixed `num_queries` で取りこぼしやすい)。
- class-agnostic にして motion_detail の表現力は caption 側に全振り。
- focal loss + DIoU loss、top-k = 30、score 閾値 0.3 から PR 曲線で調整。

### [3] キャプション生成: 構造化フィールド予測 → motion_detail 生成
- 6 フィールドのうち 4 つは半閉集合 → 分類ヘッド。
- target_object と motion_detail のみテキスト生成。motion_detail は他フィールドを prefix にして LLM で言い換え生成。
- アーキ: 区間特徴 → RoIAlign-1D (固定長 8) → Q-Former (32 query) → Qwen2.5-0.5B/1.5B (LoRA only)。

### 学習スケジュール
1. **Phase 1**: 特徴抽出 (1 回限り)。
2. **Phase 2**: TAL ヘッド学習 (10〜20 epoch)。指標は mAP@tIoU=0.5、AR@10。
3. **Phase 3**: キャプションヘッド学習 (Phase 2 frozen)。GT 区間で教師強制 → 最後に予測区間でスケジュールサンプリング。

### 評価指標
- 検出: mAP@tIoU={0.3, 0.5, 0.7}, AR@10。
- caption: tIoU≥0.5 マッチ後の構造化フィールド EM + motion_detail の CIDEr/BERTScore。
- E2E: SODA_c。

### end-to-end (Vid2Seq 系) を採らない理由
- 1 動画 16 区間 × 63k = 100 万区間規模で dense すぎ、time token 方式が破綻。
- 構造化フィールドが既にあるのを活かさない手はない。
- 2 段にすれば失敗の切り分けが楽。

### 撤退ライン
- TAL recall 不足 → stride 1、ViT-g 昇格、backbone LoRA fine-tune。
- caption 定型化 → デコーダ大型化、温度調整、contrastive loss。
- 稀少クラス → balanced sampling、synonym 統合。

---

## Q5. 新リポジトリ追加後の戦略更新

### 各リポジトリの位置づけ

| Repo | 種別 | 採否 | 理由 |
|---|---|---|---|
| **VCLTDR-GMG (CLTDR-GMG)** | ActionFormer 系後継 (TriDet/TemporalMaxer ベース) | **採用 — Phase 2 第一候補に昇格** | ActionFormer/TriDet の累積改良。入力 I/F 互換、VideoMAE V2 特徴をそのまま投入可。RTX 4090 動作 = A100 80GB に余裕。 |
| **OpenTAL** | Open-set TAL (EDL で未知棄却) | **部分採用 — 棄却機構として参考** | AFSD ベース乗り換えはコスト高。actionness (PU 学習) + EDL uncertainty は class-agnostic と相性◎。precision 補強の予備。 |
| **DiffusionTAD (DiffTAD)** | 拡散モデル TAD | **不採用** | コード未公開 (README + assets のみ)。再現実装はコスト過大。 |
| E2E-TAD | 生動画 → 検出 | **不採用維持** | 2 段案と方向性が逆。 |

### 戦略の更新点

#### 1. Phase 2 を VCLTDR-GMG ベースに変更
- ActionFormer 単体より精度が高い後継系。入力形式互換でコスト変わらず精度向上。
- class-agnostic (1 class) 設定の yaml を新設して回す方針は維持。
- ActionFormer は対照群として温存。

#### 2. OpenTAL の EDL/actionness をオプション追加
- precision 問題が出た場合の補強案。
- 静止フレームや姿勢維持での誤検出抑制に effective。
- 実装は OpenTAL のロス設計 (PU loss + EDL loss) を CLTDR-GMG ヘッドに移植する形で。最初から入れず、AR は出るが FP 多いと判明したら追加。

#### 3. DiffTAD はマイルストーンから外す
- コードがない以上、DDIM denoising decoder のゼロ実装はスコープ外。
- 論文アイデアだけメモに残す。

### 更新後パイプライン (差分のみ)

```
[1] VideoMAE V2-B (frozen, stride 2)  ← 変更なし
       ↓
[2] VCLTDR-GMG (class-agnostic)       ← ActionFormer から差し替え
     + (任意) OpenTAL の actionness/EDL でフィルタ
       ↓
[3] RoIAlign-1D + Q-Former + Qwen LoRA ← 変更なし
```

### 撤退ライン (更新)
- VCLTDR-GMG で AR@10 不足 → ActionFormer / TriDet 単独で原因切り分け → OpenTAL actionness 移植 → backbone を ViT-g に昇格、の順。
- DiffTAD はコード公開時に再検討 (proposal denoising は短く密な区間に効く可能性)。

---

## 最初の 1〜2 週の最小ゴール
1. VideoMAE V2-B で 30s 動画特徴を npy 化
2. VCLTDR-GMG の yaml を本データ用に作成 (1 クラス, stride 2)
3. AR@10 / mAP@0.5 を測る

ここで TAL 単体が立たないと Phase 3 を作っても無駄になるため、まず Phase 1+2 の検証を優先する。
