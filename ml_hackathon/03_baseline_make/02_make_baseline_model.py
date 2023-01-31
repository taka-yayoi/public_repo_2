# Databricks notebook source
# MAGIC %md
# MAGIC # 【step3-2】数値データだけを用いたBaselineモデルの作成
# MAGIC 
# MAGIC このノートブックでは、Feature Storeに保存した情報を利用して、Baselineモデルを作成します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインポート

# COMMAND ----------

import mlflow
import lightgbm as lgb

import pandas as pd
import numpy as np

# データベースの選択
spark.sql(f"USE {team_name}_hackathon")

# ワークスペースエクスペリメントの選択
mlflow.set_experiment(experiment_id=my_exp_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC MLflowエクスペリメントには**ノートブックエクスペリメント**と**ワークスペースエクスペリメント**があります。今回、**ワークスペースエクスペリメント**を使用する理由についてチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Storeから特徴量を呼び出し、必要な情報とマージしてトレーニングに必要なデータを準備
# MAGIC 
# MAGIC [Databricks Feature Storeを用いたモデルのトレーニング](https://qiita.com/taka_yayoi/items/c6785fc3c93c89a204f6)

# COMMAND ----------

from databricks.feature_store import FeatureLookup, FeatureStoreClient
fs = FeatureStoreClient()

# 特徴量データの読み込み
df_train = fs.read_table(name=f'{team_name}_hackathon.numeric_features_train')
df_test = fs.read_table(name=f'{team_name}_hackathon.numeric_features_test')

# 訓練データセットに結合したい情報を指定
feature_lookups = [
    FeatureLookup(
      table_name = f'{team_name}_hackathon.data_split',
      feature_name = 'split', # 今回は訓練データと評価データの分割情報を付与する
      lookup_key = 'id'
    )]

# 訓練データセットに結合したい情報を指定
training_set = fs.create_training_set(
  df_train,
  feature_lookups = feature_lookups,
  label = "y", # 目的変数があれば指定する
  exclude_columns = ['id'] # モデリングに不要なカラムはあらかじめ除外できる / IDの学習は必要ないので除外しておく。
)

# 実データをインスタンス化して、pandas形式に変換
df_train = training_set.load_df().toPandas()

# 作成したデータを確認
display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの学習とMLflowでの実験管理を行う

# COMMAND ----------

# 先ほど付加した情報を利用して、trainとvalidに分割する
df_train_ = df_train[df_train["split"]=="train"].drop(["split"], axis=1)
df_valid_ = df_train[df_train["split"]=="valid"].drop(["split"], axis=1)

# COMMAND ----------

mlflow.lightgbm.autolog()
with mlflow.start_run(run_name="baseline_lgb"):

    # データセットの作成
    # 最終列が目的変数になっているのでデータセットから除外し、ラベルに目的変数を設定します
    train_data = lgb.Dataset(df_train_.iloc[:, :-1], label=df_train_.iloc[:, -1])
    valid_data = lgb.Dataset(df_valid_.iloc[:, :-1], label=df_valid_.iloc[:, -1])

    # 必要最小限のパラメータを設定
    params = {
        "random_state": 1234,
        "verbose": 0,
        "metrics": "rmse",
    }

    model = lgb.train(
        params=params,  # ハイパーパラメータをセット
        train_set=train_data,  # 訓練データを訓練用にセット
        valid_sets=[train_data, valid_data],  # 訓練データとテストデータをセット
        valid_names=["Train", "Valid"],  # データセットの名前をそれぞれ設定
        num_boost_round=10000,  # 計算回数
        early_stopping_rounds=10,  # アーリーストッピング設定
        verbose_eval=0,  # ログを最後の1つだけ表示
    )

    # 提出用の推論結果を記録する
    # 推論結果をcsvとしてartifactに保存します。
    # 以下の iloc[:, 1:] は、最初の列idを除外しています
    # Python初心者向け：loc,ilocの使い方を基本から解説 | happy analysis https://happy-analysis.com/python/python-topic-loc-iloc.html
    pd.DataFrame(model.predict(df_test.toPandas().iloc[:, 1:])).to_csv(
        "predict.csv", header=False
    )
    # 予測結果をアーティファクトとしてMLflowに記録
    mlflow.log_artifact("predict.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. MLflowがない状態でモデルの管理を行おうとすると何が起きるのかをチームで議論してください。
# MAGIC 1. 記録されたモデルにアクセスして、GUIを操作してみてください。
# MAGIC 1. 予測結果を[【練習問題】民泊サービスの宿泊価格予測 \| SIGNATE \- Data Science Competition](https://signate.jp/competitions/266#disclosure-policy)に提出してみてください。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step4-1】カテゴリデータから構成される特徴量テーブルの作成]($../04_add_categorical_features/01_make_categorical_feature_store)に続きます。
