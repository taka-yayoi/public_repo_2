# Databricks notebook source
# MAGIC %md
# MAGIC # 【step1】データをDelta形式で保存する
# MAGIC 
# MAGIC このNotebookでは、今回利用するcsvファイルを読み込み、Delta形式で保存します。
# MAGIC 
# MAGIC 1. csvファイルを読み込む
# MAGIC 1. データを可視化する(ビジュアライゼーション)
# MAGIC 1. 読み込んだcsvをDelta形式で保存する
# MAGIC 1. 訓練データを**訓練データ**と**評価データ**に分割するためのindexを作成してfeature storeに保存する。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定
# MAGIC 
# MAGIC 環境設定を行います。

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# MAGIC %md
# MAGIC ハッカソンで使用するデータを確認します。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/tables/ds_hackashon/

# COMMAND ----------

# MAGIC %md
# MAGIC ## データベースの準備
# MAGIC 
# MAGIC ハッカソンで使用するデータは`<チーム名>_hackathon`というデータベースに格納されます。

# COMMAND ----------

db_name = f"{team_name}_hackathon" # データベース名を定義
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE") # 以前に作成したデータベースがあれば削除
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}") # 空のデータベースを作成
spark.sql(f"USE {db_name}") # 利用するデータベースを指定
print("database name: " + db_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## csvファイルを読み込む
# MAGIC 
# MAGIC 事務局で事前に`/dbfs/FileStore/tables/ds_hackashon`に配置してあるデータを読み込みます。
# MAGIC 
# MAGIC ここでは、pandasデータフレームとして読み込んだデータをPandas on Sparkデータフレームに変換しています。
# MAGIC 
# MAGIC [Apache Spark™ 3\.2におけるPandas APIのサポート \- Qiita](https://qiita.com/taka_yayoi/items/63a21a0e5113e33ad6a3)

# COMMAND ----------

import pyspark.pandas as ps
import pandas as pd

train_path = "/dbfs/FileStore/tables/ds_hackashon/hackathon_train.csv" # 訓練データのpath
test_path = "/dbfs/FileStore/tables/ds_hackashon/hackathon_test.csv" # テストデータのpath

# データの読み込み
df_train = ps.from_pandas(pd.read_csv(train_path, sep=",", parse_dates=["first_review", "host_since", "last_review"]))
df_test = ps.from_pandas(pd.read_csv(test_path, sep=",", parse_dates=["first_review", "host_since", "last_review"]))

# データのクレンジング
df_train["host_response_rate"] = df_train["host_response_rate"].str.strip('%').astype("int") # 百分率の%がついているので削除
df_test["host_response_rate"] = df_test["host_response_rate"].str.strip('%').astype("int") # 百分率の%がついているので削除

# COMMAND ----------

# まずは今回のデータを確認
display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. 上のセルを操作して、**ビジュアライゼーション**を追加してください。表示するデータや可視化の方法は任意です。
# MAGIC 1. 上のセルを操作して、**データプロファイル**を追加してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta形式で読み込んだcsvを保存する
# MAGIC 
# MAGIC `to_table`メソッドを用いることでデータフレームをテーブルに保存することができます。

# COMMAND ----------

# Deltaテーブルとして保存
df_train.to_table("train", mode="overwrite")
df_test.to_table("test", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. サイドメニューを操作して、上で作成したテーブルのメタデータにアクセスしてください。
# MAGIC 1. 次のセルに、上で保存した`train`テーブルのすべての列を10行取得するSQLを記述して実行してください。

# COMMAND ----------

# MAGIC %sql
# MAGIC <<FILL_IN>>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 解答編 ----------------------------------------------------------------------------
# MAGIC SELECT *
# MAGIC FROM train
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC なぜ、上のセルでSQLを実行することができたのかをチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 訓練データを訓練データ(train)と評価データ(valid)に分割するためのインデックスを作成してFeature Storeに保存
# MAGIC 
# MAGIC 特徴量の作成時にリーク(評価データを使ってトレーニングしてしまう)することを防ぐため、訓練データと評価データのインデックスを作成してFeature Storeに保存します。
# MAGIC 
# MAGIC [機械学習用データの収集と準備 \| Google Cloud 公式ブログ](https://cloud.google.com/blog/ja/products/gcp/preparing-and-curating-your-data-for-machine-learning?hl=ja)
# MAGIC 
# MAGIC > データ リークというのは、予測時には入手できない情報を入力の一部としてモデルの構築に利用した際に発生する問題です。
# MAGIC 
# MAGIC このデータを特徴量ストアに保持しておき、トレーニングの際に結合することで、リークが起きないことを常に保証できる様になります。

# COMMAND ----------

from sklearn.model_selection import train_test_split

# 訓練データと評価データを分割します
train_idx, test_idx = train_test_split(df_train["id"].to_pandas().copy(), 
                                   test_size=0.2,
                                   random_state=1)

# 訓練データのインデックス
train_idx = pd.DataFrame(train_idx)
train_idx["split"] = "train"

# 評価データのインデックス
test_idx = pd.DataFrame(test_idx)
test_idx["split"] = "valid"

df_idx = spark.createDataFrame(pd.concat([train_idx, test_idx]).sort_values("id"))
display(df_idx) # 中身を確認しておきます。

# COMMAND ----------

# 作成したDataFrameをfreature storeに保存します。
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
fs.create_table(
    name="data_split",
    primary_keys=["id"],
    df=df_idx,
    description="訓練データと評価データのインデックス")

# COMMAND ----------

# MAGIC %md
# MAGIC Feature StoreはDelta形式で特徴量を保持するので、SQLでクエリすることができます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. 以下のセルに、特徴量テーブル`data_split`の`split`ごとの件数を集計するSQLを記述して実行してください。
# MAGIC 1. 好きな方法で結果をビジュアライズしてください。

# COMMAND ----------

# MAGIC %sql
# MAGIC <<FILL_IN>>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 解答編 ------------------------------------------------------------------------------------------
# MAGIC SELECT split, count(id)
# MAGIC FROM data_split
# MAGIC GROUP BY split
# MAGIC ORDER BY split ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. 上のSQLの結果を用いて円グラフを追加してください。
# MAGIC 1. **特徴量ストア**のGUIから上で作成した特徴量テーブルにアクセスしてください。そして、GUIから特徴量ストアが提供する機能を類推してください。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step2】bamboolibを用いて、GUIによるEDAを実施する]($../02_EDA/01_bamboolib_eda)に続きます。
