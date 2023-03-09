# Databricks notebook source
# MAGIC %md
# MAGIC # Apache Sparkのご紹介
# MAGIC 
# MAGIC * [Intro to Spark slides](https://github.com/databricks/tech-talks/blob/master/2020-04-29%20%7C%20Intro%20to%20Apache%20Spark/Intro%20to%20Spark.pdf)
# MAGIC * Sparkデータフレームとは何か？
# MAGIC   * [NYTデータセット](https://github.com/nytimes/covid-19-data)の読み込み
# MAGIC * どのように分散カウント処理を実行するのか？
# MAGIC * トランスフォーメーション vs. アクション
# MAGIC * Spark SQL
# MAGIC 
# MAGIC [Sparkドキュメント](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html)

# COMMAND ----------

# MAGIC %fs ls databricks-datasets/COVID/covid-19-data/

# COMMAND ----------

# MAGIC %md
# MAGIC ## このデータはどのように表現されるのか？

# COMMAND ----------

# MAGIC %md
# MAGIC ![Unified Engine](https://files.training.databricks.com/images/105/unified-engine.png)
# MAGIC 
# MAGIC 
# MAGIC #### 最初はRDDがありました...
# MAGIC * **R**esilient: 耐障害性
# MAGIC * **D**istributed: 複数ノードに分散
# MAGIC * **D**ataset: パーティション分けされたデータのコレクション
# MAGIC 
# MAGIC RDDは作成されると不変となり、問題特定を可能にするために自分のリネージを追跡し続けます。
# MAGIC 
# MAGIC ####... そして、データフレームが生まれました。
# MAGIC * 高レベルのAPI
# MAGIC * ユーザーフレンドリー
# MAGIC * 最適化、パフォーマンス改善
# MAGIC 
# MAGIC ![RDD vs DataFrames](https://files.training.databricks.com/images/105/rdd-vs-dataframes.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NYT COVIDデータからデータフレームの作成

# COMMAND ----------

covid_df = spark.read.csv("dbfs:/databricks-datasets/COVID/covid-19-data/us-counties.csv")
covid_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC CSVリーダーにどのようなオプションを指定できるのかを確認するために、[Sparkドキュメント](https://spark.apache.org/docs/latest/index.html)を参照しましょう。

# COMMAND ----------

covid_df = spark.read.csv("dbfs:/databricks-datasets/COVID/covid-19-data/us-counties.csv", header=True, inferSchema=True)
covid_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### レコード数はいくつでしょうか？
# MAGIC * M&Mを数えるのではなく、データフレームの行数をカウントしましょう。
# MAGIC 
# MAGIC ### ここでのSparkジョブはどのようなものになるのでしょうか？
# MAGIC * ステージ数はいくつでしょうか？

# COMMAND ----------

covid_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sparkコードを書いてみましょう！
# MAGIC 
# MAGIC * 住んでいる郡(Los Angeles)の情報のみを参照したいです。
# MAGIC * 最新の情報を一番上に表示したいです。

# COMMAND ----------

(covid_df
 .sort(covid_df["date"].desc()) 
 .filter(covid_df["county"] == "Los Angeles")) 

# COMMAND ----------

# MAGIC %md
# MAGIC **...何も起きません。なぜでしょうか？**

# COMMAND ----------

# MAGIC %md
# MAGIC ## トランスフォーメーション vs アクション
# MAGIC 
# MAGIC Sparkでは2つのタイプのオペレーションがあります: トランスフォーメーションとアクションです。
# MAGIC 
# MAGIC Apache Sparkの基本は以下のように説明されます
# MAGIC * トランスフォーメーションは **怠惰(LAZY)** です
# MAGIC * アクションは **懸命(EAGER)** です

# COMMAND ----------

# 上と同じオペレーションです
(covid_df
 .sort(covid_df["date"].desc()) 
 .filter(covid_df["county"] == "Los Angeles")) 

# COMMAND ----------

# MAGIC %md
# MAGIC なぜ結果が表示されないのでしょうか？**Sort**と**filter**は、Sparkでは遅延評価される`トランスフォーメーション`です。
# MAGIC 
# MAGIC 遅延評価にはいくつかのメリットがあります。
# MAGIC * 最初からすべてのデータをロードする必要がありません。
# MAGIC   * **本当に**大きなデータセットでは技術的に不可能です。
# MAGIC * オペレーションの並列化が容易です。
# MAGIC   * 単一マシン、単一スレッド、単一のデータ要素に対して、N個の異なるトランスフォーメーションを処理することが可能です。
# MAGIC * 最も重要なことですが、これによってこのフレームワーク様々な最適化処理を自動で適用できるようになります。
# MAGIC   * これもまた我々がデータフレームを活用する理由なのです！
# MAGIC 
# MAGIC Sparkの**Catalyst**オプティマイザーができることは様々です。この状況にのみフォーカスしていきましょう。詳細はこちらの[ブログ記事](https://qiita.com/taka_yayoi/items/311692dd034763caa4ed)をご覧ください！
# MAGIC   
# MAGIC ![Catalyst](https://files.training.databricks.com/images/105/catalyst-diagram.png)

# COMMAND ----------

(covid_df
 .sort(covid_df["date"].desc()) 
 .filter(covid_df["county"] == "Los Angeles") 
 .show())  # これがアクションです！

# COMMAND ----------

# MAGIC %md
# MAGIC ### 実際に最適化処理を確認することができます！
# MAGIC * Spark UIに移動します
# MAGIC * Sparkジョブに関連づけられているSQLクエリーをクリックします
# MAGIC * 論理的、物理的プランを確認します！
# MAGIC   * フィルタリングとソートが入れ替えられています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark SQL

# COMMAND ----------

covid_df.createOrReplaceTempView("covid")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM covid
# MAGIC 
# MAGIC -- keys = date, grouping = county, values = cases

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM covid 
# MAGIC WHERE county = "Los Angeles"
# MAGIC 
# MAGIC -- keys = date, grouping = county, values = cases, deaths

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT max(cases) AS max_cases, max(deaths) AS max_deaths, county 
# MAGIC FROM covid 
# MAGIC GROUP BY county 
# MAGIC ORDER BY max_cases DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### 自分で分析してみましょう！
# MAGIC * トライしてみるアイデアがいくつかあります。
# MAGIC * [こちら](https://qiita.com/taka_yayoi/items/3d62c4dbdc0e39e4772c)に更なるサンプルがあります。

# COMMAND ----------

# MAGIC %md
# MAGIC **これはcensus.govから取得した国勢調査データです**
# MAGIC * NYTデータに対応するfipsコードカラムを構成するのに十分なデータが含まれています。

# COMMAND ----------

# MAGIC %sh wget https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv && cp co-est2019-alldata.csv /dbfs/tmp

# COMMAND ----------

census_df = spark.read.csv("dbfs:/tmp/co-est2019-alldata.csv", header=True, inferSchema=True)

# display()はDatabricksのみで使用できる関数です。show()のようにデータを表示しますが、上のSQLのセクションで見たようにビジュアライゼーションのオプションを提供しています。
display(census_df)

# COMMAND ----------

# MAGIC %md
# MAGIC NYTデータにマッチするfipsカラムを追加できるように上のデータフレームを調整します。こちらが[ユーザー定義関数(UDF)](https://docs.databricks.com/spark/latest/spark-sql/udf-python.html)に関するドキュメントです。

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def make_fips(state_code, county_code):
  if len(str(county_code)) == 1:
    return str(state_code) + "00" + str(county_code)
  elif len(str(county_code)) == 2:
    return str(state_code) + "0" + str(county_code)
  else:
    return str(state_code) + str(county_code)

make_fips_udf = udf(make_fips, StringType())
  
census_df = census_df.withColumn("fips", make_fips_udf(census_df.STATE, census_df.COUNTY))

# COMMAND ----------

# MAGIC %md
# MAGIC 同じカラムを持つ国勢調査データとCOVIDのデータの準備ができたので、2つのデータフレームをjoinしましょう。

# COMMAND ----------

covid_with_census = (covid_df
                     .na.drop(subset=["fips"])
                     .join(census_df.drop("COUNTY", "STATE"), on=['fips'], how='inner'))

# COMMAND ----------

# MAGIC %md
# MAGIC 最も人口の多い郡では感染者数はどのようになっているでしょうか？

# COMMAND ----------

display(covid_with_census.filter("POPESTIMATE2019 > 2000000").select("county", "cases", "date"))

# keys = date, grouping = county, values = cases

# COMMAND ----------

# MAGIC %md
# MAGIC NYTデータセットは日毎に新しい行が追加されるので、感染者数は日毎に増加します。郡ごとの最新の数のみを取得しましょう。
# MAGIC * 以下では、カラムを参照するために`col`関数を使用しています。これは`df["column_name"]`と同じようなものです。
# MAGIC * 郡ごとの最新の行を取得するために、[ウィンドウ関数](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=window#pyspark.sql.Window)を活用しています。

# COMMAND ----------

from pyspark.sql.functions import row_number, col
from pyspark.sql import Window

w = Window.partitionBy("fips").orderBy(col("date").desc())
current_covid_rates = (covid_with_census
                       .withColumn("row_num", row_number().over(w))
                       .filter(col("row_num") == 1)
                       .drop("row_num"))

# COMMAND ----------

# MAGIC %md
# MAGIC 感染者数を人口にスケールさせた場合、最も困難に直面した郡はどこでしょうか？

# COMMAND ----------

current_covid_rates = (current_covid_rates
                       .withColumn("case_rates_percent", 100*(col("cases")/col("POPESTIMATE2019")))
                       .sort(col("case_rates_percent").desc()))

# トップ10の郡を参照します
display(current_covid_rates.select("county", "state", "cases", "POPESTIMATE2019", "case_rates_percent").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas API on Spark
# MAGIC 
# MAGIC ここまではPySparkを用いてデータを処理してきましたが、`PySparkの文法を覚えるのは大変...`という意見もあるかと思います。そこで、Pandas APIを利用しつつもSparkを操作できるAPIとして**Pandas API on Spark**(旧Koalas)があります。
# MAGIC 
# MAGIC - [SparkにおけるPandas API](https://qiita.com/taka_yayoi/items/db8e1ea52afe5c282c94)
# MAGIC - [Supported pandas API — PySpark 3\.3\.2 documentation](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/supported_pandas_api.html)
# MAGIC - [pandasユーザーがPandas API on Sparkでつまづいたあれこれ \- KAKEHASHI Tech Blog](https://kakehashi-dev.hatenablog.com/entry/2022/12/24/090000)

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

# Pandas on Sparkデータフレームに変換
psdf = current_covid_rates.pandas_api()

# COMMAND ----------

# pandasのお作法でカラムにアクセスします
psdf['state']

# COMMAND ----------

psdf.head(2)

# COMMAND ----------

psdf.describe()

# COMMAND ----------

psdf.sort_values(by='deaths', ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
