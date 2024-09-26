# ANZ LLM ワークショップ リポジトリ 🧠

このリポジトリは、ANZ LLM ワークショップ シリーズ用です。

## セットアップ

この一連のノートブックは、Databricks ML Runtime 14.3 で開発およびテストされています。

これらは、Databricks Provisioned Throughput Foundation Model API と並行して実行するように設計されています。
参照：[Databricks AWS ドキュメント](https://docs.databricks.com/ja/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html)

DBRX / Mistral / Llama 2 などのチャットモデルを使用してモデルエンドポイントをデプロイできます。
参照：[モデルエンドポイントの作成](https://docs.databricks.com/ja/machine-learning/model-serving/create-foundation-model-endpoints.html)

## 教材の概要

`0.1_lab_setup(instructor_only)` ノートブックは講師によって実行されます。これにより、HuggingFace モデルと作業に使用するサンプルドキュメントがダウンロードされます。ワークスペースは `*.huggingface.co` へのアクセスと、pdf データ用の wikipedia およびその他のウェブサイトへのアクセスが必要です。

`0.x_` シリーズのノートブックは、LLM の基本を通じて、HuggingFace オープンソースモデルを使用した基本的な RAG アプリのセットアップを行います。\
`1.x_` シリーズのノートブックは、RAG アーキテクチャの構築とチューニングについて、より詳細にカバーします。

## 追加情報

Databricks のドライバーノードでアプリケーションを実行することが可能です。`app` フォルダには、これを行う方法の例が含まれています。

## 録画

2023年版のこれらの教材はウェビナーで紹介されました。参照：
[LLM 基礎](https://vimeo.com/857791352) 0.x_ 教材
[LLM 上級](https://vimeo.com/862303088) 1.x_ 教材


## さらなる読み物と学習
- Data and AI Summit での LLM 関連の素晴らしいカタログがあります [リンクはこちら](https://www.databricks.com/dataaisummit/llm/)
- これらの LLM をファインチューニングする素晴らしい例のセットについては、[Databricks ML 例のリポジトリ](https://github.com/databricks/databricks-ml-examples/tree/master)をお勧めします