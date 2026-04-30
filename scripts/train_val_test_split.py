"""
train_val_test_split.pyは、featureディレクトリと分割比率ratioとseedとoutput_dirをコマンドライン引数で受け取り、
featureディレクトリ内にあるファイルを分割比率ratioに従って分割して、trainデータセットとvalデータセットとtestデータセットに分割する。

featureディレクトリは、data/features/30s_mae_b_16_2にあるようなもので、
映像から抽出された特徴量を格納した.npyファイルと、それとstemを同じとする.jsonファイルのセットが大量に入っている。
ここでは.jsonファイルを基準に分割を行う。

つまり、.jsonファイルの数をNとして、0からN-1の整数をランダムに分割比率ratioに従って分割して、trainデータセット、valデータセット、testデータセットに分割する。

train.txt, val.txt, test.txtは、それぞれtrainデータセット、valデータセット、testデータセットのjsonファイルのファイルパスを行ごとに羅列したファイルである。
全て以下のような形式になる。

/path/to/feature/file1.json
/path/to/feature/file2.json
...
/path/to/feature/fileN.json

seedは、ランダムシードである。
output_dirは、train.txt, val.txt, test.txtを保存するディレクトリである。

例えば、
```bash
uv run python scripts/train_val_test_split.py \
  --feature_dir data/features/30s_mae_b_16_2 \
  --ratio 0.8 0.1 0.1 \
  --seed 42 \
  --output_dir data/splits/30s_mae_b_16_2
```

これは、data/features/30s_mae_b_16_2内にある.jsonファイルを0.8:0.1:0.1の比率で分割して、train.txt, val.txt, test.txtに保存する。
ランダムシードは42である。
分割されたデータセットは、data/splits/30s_mae_b_16_2に保存される。
"""

import argparse
import os
import glob
import random
import json
from pathlib import Path

VIDEO_BASE_DIR = "/raid/containers/enroot/data/b100/ft_fact_videos/30s_chunks"
ANNOT_BASE_DIR = "/raid/containers/enroot/data/b100/ft_fact_videos/annot/30s_chunks_action_31detail_2"


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _rewrite_paths_in_json(json_path: str) -> tuple[bool, str | None]:
    """Rewrite video_path/annotation_path base directories in-place.

    Returns:
        (changed, error)
        - changed: True if file content changed
        - error: None if parsed successfully, otherwise error message
    """
    p = Path(json_path)
    try:
        raw = p.read_text(encoding="utf-8")
        if not raw.strip():
            return False, "empty file"
        data = json.loads(raw)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    changed = False

    old_video_path = data.get("video_path")
    if isinstance(old_video_path, str) and old_video_path:
        new_video_path = str(Path(VIDEO_BASE_DIR) / Path(old_video_path).name)
        if new_video_path != old_video_path:
            data["video_path"] = new_video_path
            changed = True

    old_annot_path = data.get("annotation_path")
    if isinstance(old_annot_path, str) and old_annot_path:
        new_annot_path = str(Path(ANNOT_BASE_DIR) / Path(old_annot_path).name)
        if new_annot_path != old_annot_path:
            data["annotation_path"] = new_annot_path
            changed = True

    if changed:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed, None


def _is_valid_json_file(json_path: str) -> tuple[bool, str | None]:
    p = Path(json_path)
    try:
        raw = p.read_text(encoding="utf-8")
        if not raw.strip():
            return False, "empty file"
        _ = json.loads(raw)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--ratio", type=float, nargs=3, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--change_path",
        type=_to_bool,
        default=False,
        help="If True, rewrite video_path/annotation_path base directories in json files.",
    )
    args = parser.parse_args()

    feature_dir = args.feature_dir
    ratio = args.ratio
    seed = args.seed
    output_dir = args.output_dir
    change_path = args.change_path

    # ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ディレクトリ内のファイルを取得
    json_files_all = sorted(glob.glob(os.path.join(feature_dir, "*.json")))

    # Always filter out invalid JSON files from split candidates.
    json_files = []
    invalid_jsons: list[tuple[str, str]] = []
    for jf in json_files_all:
        ok, err = _is_valid_json_file(jf)
        if ok:
            json_files.append(jf)
        else:
            invalid_jsons.append((jf, err or "unknown"))
    if invalid_jsons:
        invalid_log = os.path.join(output_dir, "invalid_json_files.txt")
        with open(invalid_log, "w", encoding="utf-8") as f:
            for path, err in invalid_jsons:
                f.write(f"{path}\t{err}\n")
        print(f"skipped_invalid_json_files_for_split: {len(invalid_jsons)}")
        print(f"invalid_json_log: {invalid_log}")

    if change_path:
        changed_count = 0
        skipped_count = 0
        for jf in json_files:
            changed, error = _rewrite_paths_in_json(jf)
            if error is not None:
                skipped_count += 1
                continue
            if changed:
                changed_count += 1
        print(f"changed_path_files: {changed_count}")
        print(f"skipped_invalid_json_files: {skipped_count}")

    # ファイルを分割する
    random.seed(seed)
    random.shuffle(json_files)
    train_files = json_files[:int(len(json_files) * ratio[0])]
    val_files = json_files[int(len(json_files) * ratio[0]):int(len(json_files) * (ratio[0] + ratio[1]))]
    test_files = json_files[int(len(json_files) * (ratio[0] + ratio[1])):]

    # ファイルを保存する
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for file in train_files:
            f.write(file + "\n")
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for file in val_files:
            f.write(file + "\n")
    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for file in test_files:
            f.write(file + "\n")

    print(f"train_files: {len(train_files)}")
    print(f"val_files: {len(val_files)}")
    print(f"test_files: {len(test_files)}")
    print(f"valid_json_files: {len(json_files)} / all_json_files: {len(json_files_all)}")
    print(f"train_files: {train_files}")
    print(f"val_files: {val_files}")
    print(f"test_files: {test_files}")
    print(f"output_dir: {output_dir}")
    print(f"ratio: {ratio}")
    print(f"seed: {seed}")

if __name__ == "__main__":
    main()
