from huggingface_hub import snapshot_download

# 保存先のディレクトリを指定
target_dir = "/lustre/work/mt/okamura/tal-annotation-labeling/models"

model_path = snapshot_download(
    repo_id="OpenGVLab/VideoMAEv2-Base",
    local_dir=target_dir,
    local_dir_use_symlinks=False  # ディレクトリ内に実体ファイルをコピーする場合
)

print(f"モデルはここに保存されました: {model_path}")