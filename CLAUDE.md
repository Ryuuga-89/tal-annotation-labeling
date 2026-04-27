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

