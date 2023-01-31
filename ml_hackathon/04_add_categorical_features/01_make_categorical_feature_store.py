# Databricks notebook source
# MAGIC %md
# MAGIC # 【step4-1】カテゴリデータから構成される特徴量テーブルの作成
# MAGIC 
# MAGIC このノートブックでは、Deltaテーブルからカテゴリデータのみを抜き出し、インデックス変換を行いFeature Storeに保存します。
# MAGIC 
# MAGIC 1. カテゴリデータのみを抜粋したデータフレームを作成
# MAGIC 1. インデックスに変換
# MAGIC 1. Feature Storeに保存

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC なぜ、数値データだけでなくてカテゴリデータからも特徴量を生成するのかをチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

spark.sql(f"USE {team_name}_hackathon")
import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC ## カテゴリデータのみを抜粋したDataframeを作成

# COMMAND ----------

# MAGIC %sql
# MAGIC -- categorical_featuresのみを抽出したtemp viewを作成
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW categorical_features_train AS
# MAGIC SELECT
# MAGIC   id,
# MAGIC   bed_type, 
# MAGIC   cancellation_policy,
# MAGIC   city, 
# MAGIC   cleaning_fee, 
# MAGIC   instant_bookable,
# MAGIC   property_type, 
# MAGIC   room_type
# MAGIC FROM
# MAGIC   train;
# MAGIC   
# MAGIC   
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW categorical_features_test AS
# MAGIC SELECT
# MAGIC   id,
# MAGIC   bed_type, 
# MAGIC   cancellation_policy,
# MAGIC   city, 
# MAGIC   cleaning_fee, 
# MAGIC   instant_bookable,
# MAGIC   property_type, 
# MAGIC   room_type
# MAGIC FROM
# MAGIC   test;

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. 上で作成した一時ビュー`categorical_features_train`の全レコードを取得するSQLを次のセルに記述し実行してください。
# MAGIC 1. そして、結果の`city`ごとの件数を表示する棒グラフを追加してください。

# COMMAND ----------

# MAGIC %sql
# MAGIC <<FILL_IN>>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 解答編 --------------------------------------------------------------------------------------
# MAGIC SELECT
# MAGIC  *
# MAGIC FROM
# MAGIC   categorical_features_train

# COMMAND ----------

# まずはFeature Storeからtrain or valid の情報を取得して、データフレームに追加します。
from databricks.feature_store import FeatureLookup, FeatureStoreClient
fs = FeatureStoreClient()

df_train = spark.read.table(f'categorical_features_train')
df_test = spark.read.table(f'categorical_features_test')

# データセットに結合したい情報を指定
# idをキーとしてdata_splitと結合します
feature_lookups = [
    FeatureLookup(
      table_name = f'{team_name}_hackathon.data_split',
      feature_name = 'split', # 今回は訓練と評価データの分割情報を付与する
      lookup_key = 'id'
    )]

training_set = fs.create_training_set(
  df_train,
  feature_lookups = feature_lookups,
  label = None
)
df_train = training_set.load_df()

# リークを防ぐために分割します
df_train_ = df_train[df_train["split"]=="train"]
df_valid_ = df_train[df_train["split"]=="valid"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## インデックスへの変換

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

"""
df_train_.columns[1:-1]の中身
['bed_type',
 'cancellation_policy',
 'city',
 'cleaning_fee',
 'instant_bookable',
 'property_type',
 'room_type']
"""

categoricalCols = df_train_.columns[1:-1] # categorical_featureを指定する。

# stringIndexerをインスタンス化
stringIndexer = StringIndexer(inputCols=categoricalCols, 
                              outputCols=[x + "_label" for x in categoricalCols],
                              handleInvalid="keep" # 訓練データに含まれないラベルに対する挙動を指定する。
                             )

# カテゴリ変数に変換する
stringIndexerModel = stringIndexer.fit(df_train_) # 訓練データのみでindexを学習

df_train_category = stringIndexerModel.transform(df_train)
df_test_category = stringIndexerModel.transform(df_test)

# COMMAND ----------

display(df_train_category)

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. なぜ、カテゴリー変数ではインデックスへの変換が必要なのかをチームで議論してください。
# MAGIC 1. 今回はインデックス変換に`stringIndexer`を使用していますが、これに問題がないか、ある場合にはどの様なアプローチが考えられるかをチームで議論して下さい。
# MAGIC 
# MAGIC [stringIndexer - 特徴の抽出、変形および選択 \- Spark 2\.2\.0 ドキュメント 日本語訳](http://mogile.web.fc2.com/spark/spark220/ml-features.html#stringindexer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Storeへ保存

# COMMAND ----------

df_train_for_fs = df_train_category.drop(*df_train_.columns[1:-1]).drop("split") # feature storeへは、indexに変換したカラムのみを保存する
df_test_for_fs = df_test_category.drop(*df_train_.columns[1:-1]).drop("split") # feature storeへは、indexに変換したカラムのみを保存する

# COMMAND ----------

# 特徴量テーブル categorical_features_train の作成
fs.create_table(
    name=f"categorical_features_train",
    primary_keys=["id"],
    df=df_train_for_fs,
    description="カテゴリカルデータのみの特徴量テーブルを作成(訓練データ)"
)

# 特徴量テーブル categorical_features_test の作成
fs.create_table(
    name=f"categorical_features_test",
    primary_keys=["id"],
    df=df_test_for_fs,
    description="カテゴリカルデータのみの特徴量テーブルを作成(テストデータ)")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 特徴量ストアのGUIにアクセスして、ここまでの操作で特徴量テーブルが合計5個作成されていることを確認してください。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step4-2】数値データとカテゴリデータの両方を用いたBaselineモデルの作成]($02_make_categorical_feature_model)に続きます。
