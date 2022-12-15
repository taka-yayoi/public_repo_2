# Databricks notebook source
# MAGIC %md
# MAGIC # Retail CDC Data Generator
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fdlt_cdc%2Fnotebook_generator&dt=DLT_CDC">
# MAGIC <!-- [metadata={"description":"Process CDC from external system and save them as a Delta Table. BRONZE/SILVER.<br/><i>Usage: demo CDC flow.</i>",
# MAGIC  "authors":["mojgan.mazouchi@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["autoloader", "copy into", "cdc", "cdf"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fakerのインストール

# COMMAND ----------

# MAGIC %pip install Faker

# COMMAND ----------

# MAGIC %md
# MAGIC ## ダミーデータの生成

# COMMAND ----------

from pyspark.sql import functions as F
from faker import Faker
from collections import OrderedDict 
import uuid

# データの格納パス: 適宜変更してください
folder = "/tmp/takaaki.yayoi@databricks.com/demo/cdc_raw"
#dbutils.fs.rm(folder, True)
try:
  dbutils.fs.ls(folder)
except:
  print("フォルダーが存在しません、データを生成中...")  
  fake = Faker()
  fake_firstname = F.udf(fake.first_name)
  fake_lastname = F.udf(fake.last_name)
  fake_email = F.udf(fake.ascii_company_email)
  fake_date = F.udf(lambda:fake.date_time_this_month().strftime("%m-%d-%Y %H:%M:%S"))
  fake_address = F.udf(fake.address)
  operations = OrderedDict([("APPEND", 0.5),("DELETE", 0.1),("UPDATE", 0.3),(None, 0.01)])
  fake_operation = F.udf(lambda:fake.random_elements(elements=operations, length=1)[0])
  fake_id = F.udf(lambda: str(uuid.uuid4()))

  df = spark.range(0, 100000)
  df = df.withColumn("id", fake_id())
  df = df.withColumn("firstname", fake_firstname())
  df = df.withColumn("lastname", fake_lastname())
  df = df.withColumn("email", fake_email())
  df = df.withColumn("address", fake_address())
  df = df.withColumn("operation", fake_operation())
  df = df.withColumn("operation_date", fake_date())

  df.repartition(100).write.format("json").mode("overwrite").save(folder+"/customers")
  
  df = spark.range(0, 10000)
  df = df.withColumn("id", fake_id())
  df = df.withColumn("transaction_date", fake_date())
  df = df.withColumn("amount", F.round(F.rand()*1000))
  df = df.withColumn("item_count", F.round(F.rand()*10))
  df = df.withColumn("operation", fake_operation())
  df = df.withColumn("operation_date", fake_date())

  # 同じIDが生成された顧客とJoin
  df = df.withColumn("t_id", F.monotonically_increasing_id()).join(spark.read.json(folder+"/customers").select("id").withColumnRenamed("id", "customer_id").withColumn("t_id", F.monotonically_increasing_id()), "t_id").drop("t_id")
  df.repartition(10).write.format("json").mode("overwrite").save(folder+"/transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの確認

# COMMAND ----------

spark.read.json(folder+"/customers").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプラインの初期設定

# COMMAND ----------

# MAGIC %sql
# MAGIC -- データベースの作成: パイプラインのターゲットに指定してください
# MAGIC DROP DATABASE IF EXISTS cdc_data_taka CASCADE;
# MAGIC CREATE DATABASE cdc_data_taka;

# COMMAND ----------

# DLTパイプラインのストレージの初期化: パイプラインのストレージに指定してください
dbutils.fs.rm("/tmp/takaaki.yayoi@databricks.com/demo/dlt_cdc", True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC [Python]($./2-Retail_DLT_CDC_Python)か、[SQL]($./2-Retail_DLT_CDC_sql)のノートブックを確認してください。
