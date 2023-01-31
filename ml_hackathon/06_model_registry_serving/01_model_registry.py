# Databricks notebook source
# MAGIC %md
# MAGIC # 【step6】モデルレジストリへのモデルの登録
# MAGIC 
# MAGIC モデルレジストリにモデルを登録することで、Databricksのどこからでもモデルを容易に参照できるようになり、一貫性を持って機械学習モデルのステータスを管理できるようになります。さらには、REST APIエンドポイントにモデルをデプロイして呼び出せる様になります。これを「モデルサービング」と呼びます。
# MAGIC 
# MAGIC - [DatabricksにおけるMLflowモデルレジストリ](https://qiita.com/taka_yayoi/items/e7a4bec6420eb7069995)
# MAGIC - [Databricksにおけるモデルサービング](https://qiita.com/taka_yayoi/items/b5a5f83beb4c532cf921)
# MAGIC 
# MAGIC ![](https://qiita-user-contents.imgix.net/https%3A%2F%2Fsajpstorage.blob.core.windows.net%2Fyayoi%2Fmodel_registry.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=68317e9e4977d9ce53b1b21da3bbd12c)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# モデルレジストリに登録するモデルの名称
model_name = f"accommodation_fee_model_{team_name}"
print("model_name:", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベストモデルの特定
# MAGIC 
# MAGIC MLflowのAPI`search_runs`を用いてベストなパフォーマンスを示したモデルを特定します。

# COMMAND ----------

import mlflow

# ワークフローエクスペリメントの選択
mlflow.set_experiment(experiment_id=my_exp_id)

# COMMAND ----------

# エクスペリメントに記録されているランをすべて表示します
mlflow.search_runs()

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 以下のセルを修正して、ベストモデルを取得してください。
# MAGIC 
# MAGIC [Search Runs — MLflow 2\.0\.1 documentation](https://www.mlflow.org/docs/latest/search-runs.html)
# MAGIC 
# MAGIC **ヒント**
# MAGIC - メトリクスの`Valid-rmse`を使ってください。
# MAGIC - 並び替え条件に指定するメトリクス名に記号が含まれる場合には **`** で囲んでください。

# COMMAND ----------

# Valid-rmseが一番小さいモデルを特定
best_run = mlflow.search_runs(order_by=[<FILL_IN>]).iloc[0]
print(f'ベストランのValid-rmse: {best_run["metrics.Valid-rmse"]}')

# COMMAND ----------

## 解答編 ########################################################################################
# Valid-rmseが一番小さいモデルを特定
best_run = mlflow.search_runs(order_by=['metrics.`Valid-rmse` ASC']).iloc[0]
print(f'ベストランのValid-rmse: {best_run["metrics.Valid-rmse"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルレジストリへの登録

# COMMAND ----------

# モデルレジストリにモデルを登録します
model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# COMMAND ----------

# モデルの説明文を追加します
client = mlflow.tracking.MlflowClient()
client.update_registered_model(name=model_name, description="""
# 民泊サービスの物件データを使って、宿泊価格を予測するモデル

![](https://static.signate.jp/competitions/266/background/9933zZ8vCOSHyGlu5BqV7PyWyEhRso8qZSfVMgl4.jpeg)

**データ概要**
- 課題種別：回帰
- データ種別：多変量
- 学習データサンプル数：55583
- 説明変数の数：27
- 欠損値：有り
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. サイドメニューの**モデル**にアクセスして、上のコマンドで登録したモデルにアクセスしてください。
# MAGIC 1. モデルレジストリの画面から、どの様な操作ができるのかを試してチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのステータスの変更
# MAGIC 
# MAGIC モデルレジストリ上のモデルのステータスは以下のいずれかとなります。
# MAGIC 
# MAGIC - **None** ステータスなし
# MAGIC - **Staging** テスト段階
# MAGIC - **Production** 本格運用
# MAGIC - **Archived** アーカイブ(お蔵入り)
# MAGIC 
# MAGIC 次のセルを実行して、レジストリに登録したモデルのステータスを**Production**に移行します。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC このようにモデルのステータスを変更することの影響、対策に関してチームで議論してください。モデルレジストリの画面も参考にしてください。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step7】モデルサービング]($02_model_serving)に続きます。これが最後です！
