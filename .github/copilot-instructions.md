# copilot-instructions.md

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

### 各種環境変数(実行時に設定必須)

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
      {
        "start_time": 1.0,
        "end_time": 1.3,
        "body_part": "右手",
        "action_type": "運ぶ",
        "target_object": "左側の空間",
        "motion_detail": "右から左へ約20cm水平移動させ、指先を青いペダルに向ける",
        "grip_or_contact": "非接触",
        "speed_or_force": "スムーズに",
        "posture_change": "右肩をわずかに内側へ入れる"
      },
      {
        "start_time": 1.3,
        "end_time": 1.9,
        "body_part": "右手の人差し指",
        "action_type": "指差す",
        "target_object": "左側の青いフットペダル（四角形）",
        "motion_detail": "人差し指で青いペダルを指し示し、手首を小さく振って強調する",
        "grip_or_contact": "非接触",
        "speed_or_force": "明確に",
        "posture_change": "変化なし"
      },
      {
        "start_time": 1.9,
        "end_time": 2.4,
        "body_part": "右手",
        "action_type": "触れる",
        "target_object": "右足首付近",
        "motion_detail": "手を下ろし、右足首の外側付近に軽く触れて動きを促すような動作",
        "grip_or_contact": "軽い接触",
        "speed_or_force": "軽く",
        "posture_change": "上半身を少し右に傾ける"
      },
      {
        "start_time": 2.4,
        "end_time": 2.8,
        "body_part": "右足",
        "action_type": "運ぶ",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "床から足を持ち上げ、右斜め前方の白いペダルの上空へ約15cm移動させる",
        "grip_or_contact": "非接触（空中移動）",
        "speed_or_force": "通常速度",
        "posture_change": "右膝を上げる"
      },
      {
        "start_time": 2.8,
        "end_time": 3.2,
        "body_part": "右足の裏",
        "action_type": "位置決めする",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "ペダルの表面に合わせて足を下ろし、踏む位置を定める",
        "grip_or_contact": "面接触（足裏全体）",
        "speed_or_force": "慎重に",
        "posture_change": "変化なし"
      },
      {
        "start_time": 3.2,
        "end_time": 4.1,
        "body_part": "右足",
        "action_type": "押す",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "つま先に体重をかけ、ペダルを底まで押し下げる（ストローク約5cm）",
        "grip_or_contact": "圧力を伴う接触",
        "speed_or_force": "力強く",
        "posture_change": "足首を底屈させる"
      },
      {
        "start_time": 4.1,
        "end_time": 5.1,
        "body_part": "右足",
        "action_type": "戻す",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "足の力を徐々に緩め、ペダルのバネの力に合わせて足を元の高さまで上昇させる",
        "grip_or_contact": "接触維持",
        "speed_or_force": "ゆっくりと制御しながら",
        "posture_change": "足首を背屈させ元に戻す"
      },
      {
        "start_time": 5.1,
        "end_time": 5.4,
        "body_part": "右足",
        "action_type": "離す",
        "target_object": "右側の白いフットペダル",
        "motion_detail": "ペダルから足を垂直に持ち上げ、接触を絶つ",
        "grip_or_contact": "分離",
        "speed_or_force": "素早く",
        "posture_change": "膝を少し持ち上げる"
      },
      {
        "start_time": 5.4,
        "end_time": 5.8,
        "body_part": "右足",
        "action_type": "運ぶ",
        "target_object": "左側の青いフットペダル",
        "motion_detail": "左方向へ約20cm水平移動し、青いペダルの上空へ運ぶ",
        "grip_or_contact": "非接触",
        "speed_or_force": "スムーズに",
        "posture_change": "股関節を内転させる"
      },
      {
        "start_time": 5.8,
        "end_time": 6.0,
        "body_part": "右足の裏",
        "action_type": "位置決めする",
        "target_object": "左側の青いフットペダル",
        "motion_detail": "足を下ろし、青いペダルの表面に足裏を接地させる",
        "grip_or_contact": "面接触",
        "speed_or_force": "静かに",
        "posture_change": "変化なし"
      },
      {
        "start_time": 6.0,
        "end_time": 6.5,
        "body_part": "右手の人差し指",
        "action_type": "指差す",
        "target_object": "左側の青いフットペダル（右足が乗っている状態）",
        "motion_detail": "右足が乗っている青いペダルを指し示す",
        "grip_or_contact": "非接触",
        "speed_or_force": "通常速度",
        "posture_change": "上体をわずかに前傾"
      },
      {
        "start_time": 6.5,
        "end_time": 7.0,
        "body_part": "右手",
        "action_type": "戻す",
        "target_object": "右膝の上",
        "motion_detail": "指差しを終え、手を引いて右膝の上付近に戻す",
        "grip_or_contact": "非接触",
        "speed_or_force": "リラックスして",
        "posture_change": "上体を起こす"
      },
      {
        "start_time": 7.0,
        "end_time": 7.8,
        "body_part": "右手の人差し指",
        "action_type": "ジェスチャー",
        "target_object": "空中",
        "motion_detail": "人差し指を立てて、空中で小さく振る（注意を促す動作）",
        "grip_or_contact": "非接触",
        "speed_or_force": "軽く",
        "posture_change": "変化なし"
      },
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
