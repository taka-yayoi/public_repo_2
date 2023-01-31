# Databricks notebook source
# MAGIC %md
# MAGIC # 【step3-1】数値データから構成される特徴量テーブルの作成
# MAGIC 
# MAGIC このノートブックでは、Deltaテーブルから数値データのみを抜き出して、Feature Storeに保存します
# MAGIC 
# MAGIC 1. 数値データのみを抜粋したデータフレームを作成
# MAGIC 1. Feature Storeに保存

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# MAGIC %md
# MAGIC ## データベースの選択

# COMMAND ----------

spark.sql(f"USE {team_name}_hackathon")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 数値データのみを抜粋したデータフレームを作成
# MAGIC 1. 数値データのみを抜粋した TEMPORARY VIEW(一時ビュー) を作成
# MAGIC 2. TEMPORARY VIEW をデータフレームとして読み込み

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 数値データのみを抜粋した TEMPORARY VIEW を作成します。
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW numeric_features_train AS
# MAGIC SELECT
# MAGIC   id, 
# MAGIC   accommodates,
# MAGIC   bathrooms, 
# MAGIC   bedrooms, 
# MAGIC   beds, 
# MAGIC   host_response_rate, 
# MAGIC   number_of_reviews,
# MAGIC   review_scores_rating, 
# MAGIC   y
# MAGIC FROM
# MAGIC   train;
# MAGIC 
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW numeric_features_test AS
# MAGIC SELECT
# MAGIC   id, 
# MAGIC   accommodates,
# MAGIC   bathrooms, 
# MAGIC   bedrooms, 
# MAGIC   beds, 
# MAGIC   host_response_rate, 
# MAGIC   number_of_reviews,
# MAGIC   review_scores_rating
# MAGIC   
# MAGIC FROM
# MAGIC   test

# COMMAND ----------

# MAGIC %sql
# MAGIC -- temp viewの内容を確認
# MAGIC SELECT *
# MAGIC FROM numeric_features_train

# COMMAND ----------

# TEMPORARY VIEW をデータフレームとして読み込みます
df_numeric_train = spark.read.table(f"numeric_features_train")
df_numeric_test = spark.read.table(f"numeric_features_test")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 一時ビューと通常のビューの違い、ビューとテーブルの違いをチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Storeに保存
# MAGIC 
# MAGIC PythonからFeatureStoreを操作するには`FeatureStoreClient`のインスタンスを作成します。

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# 特徴量テーブル numeric_features_train の作成
fs.create_table(
    name=f"numeric_features_train",
    primary_keys=["id"],
    df=df_numeric_train,
    description="数値データのみの特徴量テーブル(訓練データ)"
)

# 特徴量テーブル numeric_features_test の作成
fs.create_table(
    name=f"numeric_features_test",
    primary_keys=["id"],
    df=df_numeric_test,
    description="数値データのみの特徴量テーブル(テストデータ)")

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 特徴量ストア(Feature Store)を使うことの意義をチームで議論してください。
# MAGIC 
# MAGIC **ヒント**
# MAGIC - [DatabricksのFeature Store \- Qiita](https://qiita.com/taka_yayoi/items/88ddec323537febf7784)
# MAGIC - [Databricks Feature Storeのコンセプト \- Qiita](https://qiita.com/taka_yayoi/items/1afe4bc4e720b6b50b33)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step3-2】数値データだけを用いたBaselineモデルの作成]($02_make_baseline_model)に続きます。
