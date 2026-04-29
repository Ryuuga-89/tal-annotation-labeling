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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--ratio", type=float, nargs=3, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    feature_dir = args.feature_dir
    ratio = args.ratio
    seed = args.seed
    output_dir = args.output_dir

    # ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ディレクトリ内のファイルを取得
    json_files = glob.glob(os.path.join(feature_dir, "*.json"))

    # ファイルを分割する
    random.seed(seed)
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
    print(f"train_files: {train_files}")
    print(f"val_files: {val_files}")
    print(f"test_files: {test_files}")
    print(f"output_dir: {output_dir}")
    print(f"ratio: {ratio}")
    print(f"seed: {seed}")

if __name__ == "__main__":
    main()
