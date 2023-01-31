# Databricks notebook source
# MAGIC %md
# MAGIC # 【step5】hyperoptによるパラメータチューニング
# MAGIC 
# MAGIC このノートブックでは、hyperoptにより、step4にて作成したモデルのハイパーパラメータチューニングを実施します。

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
# ワークフローエクスペリメントの選択
mlflow.set_experiment(experiment_id=my_exp_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Storeからデータを読み込み
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

# COMMAND ----------

train_data = lgb.Dataset(df_train_.iloc[:, 1:-1], 
                         label=df_train_.iloc[:, -1])
valid_data = lgb.Dataset(df_valid_.iloc[:, 1:-1], 
                         label=df_valid_.iloc[:, -1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## hyperoptによるハイパーパラメータチューニング
# MAGIC 
# MAGIC - [Hyperoptのコンセプト](https://qiita.com/taka_yayoi/items/238ecf8b038151b84bc1)
# MAGIC - [Pythonにおける機械学習モデルチューニングのためのHyperoptのスケーリング](https://qiita.com/taka_yayoi/items/16ba48e84245c31aca21)

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

# COMMAND ----------

# 最適化するハイパーパラメータ空間の定義
param_space = {
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
        'num_leaves': hp.quniform('num_leaves', 4, 100, 4),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'subsample_freq': hp.quniform('subsample_freq', 1, 20, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
        'min_child_samples': hp.quniform('min_child_samples', 1, 50, 1),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-3), np.log(1e+1)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-2), np.log(1e+3)),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(1e-1))
 }

# COMMAND ----------

# パラメータを入力として、lossを返す関数を定義
def train_and_eval(params):
    # オートロギングの有効化
    mlflow.lightgbm.autolog(exclusive=False)
  
    # 整数値が必要なパラメータを整数に変換
    params["num_leaves"] = int(params["num_leaves"])
    params["min_child_samples"] = int(params["min_child_samples"])
    params["subsample_freq"] = int(params["subsample_freq"])
    params["n_estimators"] = int(params["n_estimators"])

    # 各試行に共通のパラメータを追加
    params["objective"] = "rmse"
    params["metrics"] = "rmse"
    params["feature_pre_filter"] = False
    params["random_state"] = 1234

    model = lgb.train(
        params=params,  # ハイパーパラメータをセット
        train_set=train_data,  # 訓練データを訓練用にセット
        valid_sets=[train_data, valid_data],  # 訓練データとテストデータをセット
        valid_names=["Train", "Valid"],  # データセットの名前をそれぞれ設定
        num_boost_round=10000,  # 計算回数
        early_stopping_rounds=10,  # アーリーストッピング設定
        verbose_eval=0,  # ログを最後の1つだけ表示
    )

    # 推論データの保存
    pd.DataFrame(model.predict(df_test.iloc[:, 1:])).to_csv("predict.csv", header=False)
    mlflow.log_artifact("predict.csv")

    # Validのrmseを最小化するパラメータを探索する
    valid_score = model.best_score["Valid"]["rmse"]
    return {"loss": valid_score, "status": STATUS_OK}

# COMMAND ----------

with mlflow.start_run() as run:

    argmin = fmin(
        fn=train_and_eval, # 評価関数
        space=param_space, # ハイパーパラメーター探索空間
        algo=tpe.suggest, # 探索アルゴリズム
        max_evals=15, # 評価回数
        trials=SparkTrials(parallelism=2), # SparkTrialsによる並列処理
        verbose=True,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. ハイパーパラメータチューニングのメリットをチームで議論してください。
# MAGIC 1. SparkTrialsとhyperoptでハイパーパラメータチューニングを分散処理する際の注意点をチームで議論してください。
# MAGIC 
# MAGIC **ヒント**: 並列処理によるトレーニングにはトレードオフが存在します。
# MAGIC 
# MAGIC [Pythonにおける機械学習モデルチューニングのためのHyperoptのスケーリング \- Qiita](https://qiita.com/taka_yayoi/items/16ba48e84245c31aca21)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step6】モデルレジストリへのモデルの登録]($../06_model_registry_serving/01_model_registry)に続きます。
