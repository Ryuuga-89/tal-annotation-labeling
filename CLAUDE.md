# CLAUDE.md

## Project

A repository that uses TAL technology to create annotation labels for video data approximately 30 seconds long. Target GPU: single A100 80GB — all training configs, batch sizes, and memory estimates must fit within this constraint.

## Python

- Use `uv sync` to install dependencies (not pip).
- Run scripts via `uv run python ...`.

## Behavior

- When anything is ambiguous — requirements, parameter choices, data format, training config — always ask the user before proceeding. Do not guess.

## Data

- Annotations: `/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2/*.json`
- Videos: `/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks/{video-stem}.mp4`
- ~63k annotation files

## 環境

### 各種環境変数

```bash
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks
```

### モデルへのパス

#### 特徴量検出用：vit_b

/lustre/work/mt/okamura/tal-annotation-labeling/models/vit_b/vit_b_k710_dl_from_giant.pth

## 指示

/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2/*にannotation file が {video-stem}.json で入っている。
この json は {
  "mode": "action_description",
  "video_path": "/tmp/analyze_30s_chunks/00011ec9fdeeb5a304ee7f109a7a8aad418c07c4f705dfee7fbc728bca681f6b_00-01-28.360_00-01-38.440_0_10.mp4",
  "video_duration": 10.124,
  "video_fps": 10,
  "analysis_result": {
    "actions": [
      {
        "start_time": 0.0,
        "end_time": 1.0,
        "body_part": "右手の人差し指",
        "action_type": "指差す",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "人差し指を伸ばし、白いペダルに向けて手首を上下に2回ほど振って指し示す",
        "grip_or_contact": "非接触",
        "speed_or_force": "リズミカルに",
        "posture_change": "変化なし"
      },
      ...
      {
        "start_time": 7.8,
        "end_time": 8.5,
        "body_part": "右手",
        "action_type": "置く",
        "target_object": "右太もも",
        "motion_detail": "手を下ろし、右太ももの上に手のひらを置く",
        "grip_or_contact": "接触",
        "speed_or_force": "ゆっくり",
        "posture_change": "リラックスした姿勢"
      },
      {
        "start_time": 8.5,
        "end_time": 9.2,
        "body_part": "右手",
        "action_type": "滑らせる",
        "target_object": "右太ももの衣服",
        "motion_detail": "太ももの上で手を前後に軽く滑らせる（払うような動作）",
        "grip_or_contact": "接触",
        "speed_or_force": "軽く",
        "posture_change": "変化なし"
      }
    ]
  },
  "usage_metadata": {
    "prompt_token_count": 8229,
    "candidates_token_count": 2214,
    "thoughts_token_count": 2775,
    "total_token_count": 13218
  }
}

このような形式。これの motion detail をtarget caption として、 video path が与えられた時に
{start-time}-{end-time}: {motion detail}
...

となるようにする。
これをTAL技術を用いて実現できるようにする。

必要なライブラリはpyproject.toml + uv で管理して

# wandb