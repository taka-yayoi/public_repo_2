# Databricks notebook source
# MAGIC %md
# MAGIC ## データの抽出(Extract)

# COMMAND ----------

import pandas as pd

# pandasを使ってURLから直接データを取得します
pdf = pd.read_csv("https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv")
# データを表示します
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの変換(Transform)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ構造の変換
# MAGIC 
# MAGIC このデータは、そのままでも都道府県ごとの日々の感染者数の推移を確認することはできます。しかし、このデータは各列が都道府県名となっている**横持ちのデータフレーム(Wide dataframe)** となっています。特定の県のデータを取り出すためには、列名を指定しなくてはならず、where句を使うSQLとの相性は良いとはいえません。
# MAGIC 
# MAGIC そこで、ここではいろいろな切り口で分析をしやすくなるように、**縦持ちのデータフレーム(Long dataframe)** に変換します。
# MAGIC 
# MAGIC ![](https://www.salesanalytics.co.jp/wp-content/uploads/2021/04/795316b92fc766b0181f6fef074f03fa-5.jpg)
# MAGIC 
# MAGIC [便利だけど分かりにくいデータフレームを再構築するPandasのMelt\(\)関数のお話し – セールスアナリティクス](https://www.salesanalytics.co.jp/datascience/datascience021/#melt)

# COMMAND ----------

# 縦長に変換
# id_vars: Dateをidとして、変換後のデータフレームにそのまま残します
# var_name: variable変数の列名をPrefectureにします
# value_name: value_name変数はCaseとします
long_pdf = pdf.melt(id_vars=["Date"], var_name="Prefecture", value_name="Cases")
display(long_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ型の確認
# MAGIC 
# MAGIC 次にデータ型を確認します。
# MAGIC 
# MAGIC [pandasのデータ型dtype一覧とastypeによる変換（キャスト） \| note\.nkmk\.me](https://note.nkmk.me/python-pandas-dtype-astype/)

# COMMAND ----------

long_pdf.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### フレームワークの選択、データ型の変換、フィルタリング
# MAGIC 
# MAGIC ここでは、データエンジニアの立場に立って以下の検討を行うものとします。
# MAGIC 
# MAGIC - pandasでは豊富なメソッドが提供されていますが、大量データの取り扱いには不得手です。このため、ここからは大量データを容易に取り扱うことができるSparkデータフレームで操作を行なっていきます。
# MAGIC - `object`はオブジェクト型であることを示しています。日付を示す`Date`がオブジェクト型だと後程の処理で都合が悪くなるのでタイムスタンプ型に変換します。
# MAGIC - `Prefecture`が`ALL`は全国の感染者数を示しています。全都道府県のデータの合計を取れば全国分は計算できるので除外します。
# MAGIC - 全期間のデータは不要なので２０２２年以降のデータに限定します。
# MAGIC 
# MAGIC このように、データエンジニアリングの過程では、後段での処理で求められる要件を踏まえてデータを加工していきます。これと並行して、性能要件も考慮してどのようなアプローチを取ったらいいのかも検討していきます。

# COMMAND ----------

# pandasデータフレームをSparkデータフレームに変換します
sdf = spark.createDataFrame(long_pdf)

# COMMAND ----------

from pyspark.sql.functions import *

# PythonのSpark API(pyspark)を用いて、データ型の変換を行います
# 変換後に元のDate列は削除します
# Prefecture ALLを除外します
# 2022年以降のデータに限定します
df = (sdf
       .withColumn("date_timestamp", to_timestamp(col("Date"), "yyyy/M/d"))
       .drop("Date")
       .filter("Prefecture != 'ALL' AND date_timestamp >= '2022-1-1'")
     )

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データの拡張
# MAGIC 
# MAGIC これで変換処理は概ね完了しましたが、都道府県がどの地方に属しているのかが分かった方が分析の幅が広がるので、地方名などでデータを拡張します。
# MAGIC 
# MAGIC 以下のサイトのデータをお借りして、補強するデータを準備します。
# MAGIC 
# MAGIC [【Excel】都道府県名の隣のセルに地方名を表示させる方法｜近未来スライム記](https://xenonhyx.com/excel-tihoumei/)
# MAGIC 
# MAGIC 以下で使用している`pref_augment.csv`は[こちら](https://github.com/taka-yayoi/public_repo/blob/main/covid-19%20analytics%20with%20lakehouse/pref_augment.csv)においてあります。

# COMMAND ----------

from pyspark.sql.types import *

schema = StructType([
    StructField("pref_no", IntegerType(), True),
    StructField("Prefecture", StringType(), True),
    StructField("Area", StringType(), True),
])

# COMMAND ----------

# 拡張用のデータを読み込みます
augment_df = spark.read.csv("dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/pref_augment.csv", header=True, schema=schema)
display(augment_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 元のデータと拡張用のデータを結合(join)します。

# COMMAND ----------

# 結合キーはPrefecture、内部結合(inner join)します
final_df = df.join(augment_df, on="Prefecture", how="inner")
display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データのロード(Load)
# MAGIC 
# MAGIC これでデータの準備が整いました。ETL(Extract-Transform-Load)で言うところのTransformまでが終わった形になります。最後にBIなどでこのデータを活用できるようにデータベースにロードします。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名
db_name = f"japan_covid_{username}"

# Hiveメタストアのデータベースの準備:データベースの作成
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# Hiveメタストアのデータベースの選択
spark.sql(f"USE {db_name}")

print("database name: " + db_name)

# COMMAND ----------

# データベースに登録
final_df.write.format("delta").mode("overwrite").saveAsTable("covid_cases")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM japan_covid_takaakiyayoidatabrickscom.covid_cases

# COMMAND ----------

# MAGIC %md
# MAGIC # END
