# Databricks notebook source
# MAGIC %md
# MAGIC # 【step4-2】数値データとカテゴリデータの両方を用いたBaselineモデルの作成
# MAGIC 
# MAGIC このノートブックでは、Feature Storeに保存した数値・カテゴリ双方の特徴量を利用して、Baselineモデルを作成します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

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
# MAGIC ## Feature Storeから特徴量を呼び出し、必要な情報とマージしてトレーニングに必要なデータを準備
# MAGIC 
# MAGIC [Databricks Feature Storeを用いたモデルのトレーニング](https://qiita.com/taka_yayoi/items/c6785fc3c93c89a204f6)

# COMMAND ----------

from databricks.feature_store import FeatureLookup, FeatureStoreClient

fs = FeatureStoreClient()

# 数値のみの特徴量テーブルの読み込み
df_train = fs.read_table(name=f"{team_name}_hackathon.numeric_features_train")
df_test = fs.read_table(name=f"{team_name}_hackathon.numeric_features_test")

# 訓練データセットに結合したい情報を指定
feature_lookups_train = [
    # idをキーとしてカテゴリー特徴量テーブルと結合
    FeatureLookup(
        table_name=f"{team_name}_hackathon.categorical_features_train", lookup_key="id"
    ),
    FeatureLookup(
        table_name=f"{team_name}_hackathon.data_split",
        feature_name="split",  # 今回は訓練データと評価データの分割情報を付与する
        lookup_key="id",
    ),
]

# テストデータはidをキーとしてカテゴリー特徴量テーブルと結合
feature_lookups_test = [
    FeatureLookup(
        table_name=f"{team_name}_hackathon.categorical_features_test", lookup_key="id"
    )
]

# 訓練データセットに結合したい情報を指定
training_set = fs.create_training_set(
    df_train, feature_lookups=feature_lookups_train, label="y"
)

test_set = fs.create_training_set(
    df_test, feature_lookups=feature_lookups_test, label=None
)

# 実データをインスタンス化して、pandas形式に変換
df_train = training_set.load_df()
df_test = test_set.load_df().toPandas()

# リークを防止
df_train_ = df_train[df_train["split"] == "train"].toPandas().drop(["split"], axis=1)
df_valid_ = df_train[df_train["split"] == "valid"].toPandas().drop(["split"], axis=1)

# 作成したデータを確認
display(df_train)

# COMMAND ----------

display(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングの実行

# COMMAND ----------

mlflow.lightgbm.autolog()
with mlflow.start_run(run_name="lgb_with_categorical"):

    # データセットの作成
    train_data = lgb.Dataset(df_train_.iloc[:, 1:-1], label=df_train_.iloc[:, -1])
    valid_data = lgb.Dataset(df_valid_.iloc[:, 1:-1], label=df_valid_.iloc[:, -1])

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
    # まずは一度signateに提出してみましょう
    pd.DataFrame(model.predict(df_test.iloc[:, 1:])).to_csv("predict.csv", header=False)
    mlflow.log_artifact("predict.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 予測結果を[【練習問題】民泊サービスの宿泊価格予測 \| SIGNATE \- Data Science Competition](https://signate.jp/competitions/266#disclosure-policy)に提出してみてください。前回の評価結果と比較してみましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step5】hyperoptによるパラメータチューニング]($../05_hyper_param_tune/01_hyper_param_tune_with_hyperopt)に続きます。
