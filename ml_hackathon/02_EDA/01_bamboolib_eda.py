# Databricks notebook source
# MAGIC %md
# MAGIC # 【step2】bamboolibを用いて、GUIによるEDAを実施する

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# MAGIC %md
# MAGIC ## bamboolibのインストール

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 上のセルでは`%pip install`でライブラリをインストールしましたが、Databricksでライブラリをインストールする方法が他にないかチームで議論してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## bamboolibの起動

# COMMAND ----------

import bamboolib as bam
bam

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. 前のステップで作成したテーブルをbamboolibでロードし、データの加工や可視化をしてください。
# MAGIC 1. bamboolibを活用することのメリットをチームで議論してください。
# MAGIC 
# MAGIC **ヒント**
# MAGIC - データをロードする際、Catalogには`spark_catalog`を選択します。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [【step3-1】数値データから構成される特徴量テーブルの作成]($../03_baseline_make/01_make_baseline_feature_store)に続きます。
