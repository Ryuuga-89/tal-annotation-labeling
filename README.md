# プロジェクト概要

TAl技術を用いた映像からのアノテーション出力タスク用リポジトリ

# ディレクトリ構造

* docs/ 各種ドキュメント
* repos/ 外部リポジトリ。サブモジュール化してある。

# 注意点

uvを用いてパッケージ管理
wandbを用いて学習などを管理。設定は以下を参照。

* プロジェクト名：tal-annotation-labeling
* リンク：https://wandb.ai/models-institute-of-science-tokyo/tal-annotation-labeling
* run命名規則：{train, exp, test}_{runの説明}_{タイムスタンプ}