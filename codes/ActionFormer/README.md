# codes/ActionFormer

Phase 2b: ActionFormer ベースの TAL を、本タスクのデータ
(VideoMAE V2 ViT-B 特徴) 上で学習する。**upstream の `repos/ActionFormer/`
は変更せず**、必要な追加分だけ本ディレクトリに置く。

## ディレクトリ

```
codes/ActionFormer/
├── _upstream.py          # repos/ActionFormer/libs を `actionformer_libs` として import 可能にする
├── dataset/
│   └── tal_motion.py     # @register_dataset("tal_motion") — 本データ用 Dataset
├── configs/
│   └── tal_motion_vit_b.yaml
└── README.md
```

## データ I/O

- 特徴: `data/features/30s_mae_b_16_2/subset_v1/<stem>.npy` ([N, 768], float16)
- アノテーション: `$ANNOT_ROOT_DIR/<stem>.json`
- split list: `data/splits/v1/{train,val,test}.txt` (1 行 1 json filename)
- aux vocab: `data/aux_stats/v1/aux_vocab.json` (top-N + OTHER, 各 field)

## 設計方針

- **メインの分類 head は class-agnostic (num_classes=1)**: 区間検出の品質に集中。
- **caption 生成は LLM 側で実施**: ActionFormer は区間推定を主担当。
- **補助 head は将来拡張**: `body_part / action_type` 等を使う multi-task は必要時に追加。
- 時間軸: `feat_stride=2 / num_frames=16 / fps=10` → 1 step = 0.2 秒。
  `max_seq_len=192` は 30s 動画 (~143 step) に余裕を持って収まる。
- regression range は step 単位で短時間〜長時間をカバー
  (1-2 step = 0.2-0.4 秒 〜 16+ step = 3.2 秒以上)。

## 残タスク (本コミット時点で未実装)

`aux_vocab.json` (各 field の num_classes) が確定してから実装する:

1. **Multi-task head 拡張**: `PtTransformerClsHead` を継承し、各 aux field 用の
   1D Conv 分類 head を追加。FPN level の各点に aux logit を出力。
2. **Loss**: 既存の focal (cls) + DIoU (reg) に加え、各 GT segment が
   割り当てられた point の aux logit に対する CE loss を `λ_aux` で加算。
   GT 0 件 (= seg なし) の動画は aux loss は 0。
3. **train.py / eval.py のラッパー**: upstream の `train.py` をそのまま動かす
   ことも可能だが、本タスクでは aux head 込みの forward が必要なので、
   `codes/ActionFormer/scripts/train.py` を別途用意 (upstream の loop 流用)。

## 現状での Dataset 単体の動作確認 (smoke)

```python
import sys; sys.path.insert(0, "codes")
from ActionFormer import _upstream  # noqa
from ActionFormer.dataset import tal_motion  # noqa: registers "tal_motion"
from actionformer_libs.datasets.datasets import make_dataset

ds = make_dataset(
    "tal_motion",
    is_training=True,
    split=("train",),
    feat_folder="data/features/30s_mae_b_16_2/subset_v1",
    json_file="",
    feat_stride=2, num_frames=16, default_fps=10.0,
    downsample_rate=1, max_seq_len=192, trunc_thresh=0.5,
    crop_ratio=(0.9, 1.0),
    input_dim=768, num_classes=1,
    file_prefix=None, file_ext=".npy", force_upsampling=False,
    annot_dir="/raid/.../30s_chunks_action_31detail_2",
    split_list_dir="data/splits/v1",
    aux_vocab_file="data/aux_stats/v1/aux_vocab.json",
)
print(len(ds), ds[0]["feats"].shape, ds[0]["segments"].shape, ds[0]["aux_labels"].keys())
```

## アクション 0 件動画

`skip_no_actions=True` (default) で除外。Caption 学習データとしても役に立たない
ので Phase 2b 段階では捨てる。Phase 3 で empty-prediction の評価に使う場合は
False に切り替えて再構築する。
