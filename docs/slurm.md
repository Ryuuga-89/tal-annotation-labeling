# slurm使用方法

## 全体の注意点

partitionは必ず104-patition
CPUコア数を必ず明示的に表示

## bashで入ってその上で実行

### コマンド

```bash
srun -p 104-partition -N1 -n1 -c16 -J okamura --gpus=2 --pty bash
```

-c値、--gpus値は実行するプログラムに応じて調整。

### 注意点

プログラムの実行前に

```bash
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks
```

を必ず実施。

## batchで投げる

### コマンド

```bash
sbatch example.sh
```

### ジョブスクリプト作成のルール

ジョブスクリプトは以下の形式で、値は必要に応じて変更する。環境変数の設定は必ず実行する。
`sbatch` 実行時は `$0` が一時ファイルになるため、`dirname "$0"` を基準に `cd` すると失敗することがある。
作業ディレクトリは `SLURM_SUBMIT_DIR`（`sbatch` を実行した場所）を使って明示的に移動すること。
`batch_logs` は実行前に作成しておくこと。

```bash
#!/bin/bash
#SBATCH -p 104-partition  # 固定
#SBATCH -J vlm_train_task  # 適切なジョブ名に変更
#SBATCH -n 1  # 通常は1 (DDP等の並列化はプログラム側で制御)
#SBATCH --gpus=2  # 使用するGPU数
#SBATCH -c 16  # 1枚のGPUにつき4コア程度が目安 (8枚なら32-64)
#SBATCH -o batch_logs/%j_%x.out  # 標準出力の出力先（ジョブID_名前）
#SBATCH -e batch_logs/%j_%x.err  # 標準エラー出力の出力先（ジョブID_名前）
#SBATCH --mail-type=ALL  # 固定
#SBATCH --mail-user=eyesharp06@gmail.com  # 固定

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

# --- 環境変数の設定 ---
# キャッシュ場所を共有ストレージに固定（ノード間の整合性と容量確保）
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

# --- 仮想環境の構築・同期 ---
# uvを利用して再現性を確保
uv sync

# --- 実行コマンド ---
# 8枚のGPUを認識させるために、必要に応じて torchrun 等を使用
# 例: uv run torchrun --nproc_per_node=8 train.py
uv run python train.py
```

### 失敗しやすいポイント（今回の検証結果）

- `No pyproject.toml found ...` で即失敗する場合は、作業ディレクトリがずれている可能性が高い。`cd "${SLURM_SUBMIT_DIR:-$PWD}"` を使う。
- GPUは `CUDA_VISIBLE_DEVICES` と `SLURM_GPUS_ON_NODE` の値をログ出力して確認する。
- まず `scripts/sbatch/sbatch_test.sh` で、`uv sync` と `torch.cuda.device_count()` が期待通りかを確認してから本学習を投げる。

`scripts/sbatch/sbatch_test.sh` は以下の通り

```bash
#!/bin/bash
#SBATCH -p 104-partition
#SBATCH -J sbatch_env_test
#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH -c 16
#SBATCH -o batch_logs/%j_%x.out
#SBATCH -e batch_logs/%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyesharp06@gmail.com

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p batch_logs

# --- Required environment variables ---
export UV_CACHE_DIR=/lustre/work/mt/.uv-cache
export HF_HOME=/lustre/work/mt/.cache/huggingface
export ANNOT_ROOT_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2
export VIDEO_DATA_DIR=/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks

echo "===== [sbatch test] start ====="
date
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$(pwd)"
echo "USER=${USER:-}"
echo

echo "===== [paths] ====="
echo "which uv: $(command -v uv || true)"
echo "which python: $(command -v python || true)"
echo "which nvidia-smi: $(command -v nvidia-smi || true)"
echo

echo "===== [env vars] ====="
echo "UV_CACHE_DIR=${UV_CACHE_DIR:-}"
echo "HF_HOME=${HF_HOME:-}"
echo "ANNOT_ROOT_DIR=${ANNOT_ROOT_DIR:-}"
echo "VIDEO_DATA_DIR=${VIDEO_DATA_DIR:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo

echo "===== [filesystem checks] ====="
if [[ -f pyproject.toml ]]; then
  echo "pyproject.toml: found"
else
  echo "pyproject.toml: NOT found"
fi
if [[ -d "$ANNOT_ROOT_DIR" ]]; then
  echo "ANNOT_ROOT_DIR: found"
else
  echo "ANNOT_ROOT_DIR: NOT found"
fi
if [[ -d "$VIDEO_DATA_DIR" ]]; then
  echo "VIDEO_DATA_DIR: found"
else
  echo "VIDEO_DATA_DIR: NOT found"
fi
echo

echo "===== [gpu checks] ====="
nvidia-smi || true
echo

echo "===== [uv sync] ====="
uv sync
echo

echo "===== [python/torch checks] ====="
uv run python - <<'PY'
import os
import torch

print("python ok")
print("torch.__version__ =", torch.__version__)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
PY
echo

echo "===== [sbatch test] done ====="
date
```

## その他便利コマンド

### 実行中のジョブ一覧を確認
```bash
squeue
```

### 計算ノードのリソースの空きを確認
全情報取得
```bash
scontrol show node fcdgx00196
```

簡略表記
```bash
sinfo -p 104-partition -o "%15N %10c %10m %25f %10G %20C"
```

### 実行中のジョブに用いているGPUの状態を確認
```bash
srun --jobid ジョブID --overlap --partition 104-partition nvidia-smi
```

リアルタイムで確認したい場合は
```bash
srun --jobid ジョブID --overlap --partition 104-partition watch -n 1 nvidia-smi
```