# Databricks notebook source
# MAGIC %md 
# MAGIC # MLflowとDelta LakeによるMLモデルトレーニングのトラッキング
# MAGIC 
# MAGIC データチームがモデルをトレーニングし、それをプロダクションにデプロイして、しばらくの間は上手くいくと言うのはよくある話です。そして、モデルが変な予測をし始めると、モデルを調査し、デバッグしなくてはなりません。
# MAGIC 
# MAGIC このノートブックでは、デバッグを容易にするためにモデルのトレーニングを容易に追跡、可視化、再現するために、どのように[MLflow](http://mlflow.org)と[Delta Lake](http://delta.io)を活用するのかをデモンストレーションします。
# MAGIC 
# MAGIC 1. MLパイプラインの構築に使用したデータの正確なスナップショットを追跡し、再現する。
# MAGIC 1. 特定のデータのスナップショットでトレーニングを行ったモデルを特定する。
# MAGIC 1. (例：古いモデルを再現するために)過去のデータのスナップショットに対してトレーニングを再実行する。
# MAGIC 
# MAGIC このノートブックでは、データのバージョン管理と「タイムトラベル」機能を提供するDelta Lakeを使用し、データを追跡し、特定のデータセットを使用したランをクエリーするためにMLflowを活用します。
# MAGIC 
# MAGIC **要件**:
# MAGIC * Databricksランタイム7.0以降、Mavenライブラリ`org.mlflow:mlflow-spark:2.0.0`がインストールされているクラスター。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 問題定義: 貸し手向け「悪いローン」の分類
# MAGIC 
# MAGIC このノートブックでは、クレジットのスコア、クレジット履歴などその他の特徴量に基づいて、「悪いローン」(利益を産まない可能性があるローン)の特定をゴールとして、Lending Clubデータセットにおける分類問題に取り組みます。
# MAGIC 
# MAGIC 最終的なゴールは、ローンを承認するかどうかを決定する前に、ローンの係員が使用する解釈可能なモデルを生成することです。この様なモデルは貸し手に対して、情報を提供するビューとなり、見込みのある借り手を即座に評価し、レスポンスできる様にします。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### データ
# MAGIC 
# MAGIC 使用するデータはLending Clubの公開データです。これには、2012年から2017年に融資されたローンが含まれています。それぞれのローンには、申請者によって提供された申込者情報と、現在のローンのステータス(遅延なし、遅延、完済など)、最新の支払い情報が含まれています。データに対する完全なビューは[データ辞書](https://resources.lendingclub.com/LCDataDictionary.xlsx)をご覧ください。
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC 
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

# MAGIC %md ### セットアップ: DBFSにDeltaテーブルを作成
# MAGIC 
# MAGIC DBFSに格納されている既存のParquetテーブルを変換することで、Delta Lakeフォーマットでいくつかのサンプルデータを生成します。

# COMMAND ----------

from pyspark.sql.functions import *

# テーブルが存在する場合には削除
DELTA_TABLE_DEFAULT_PATH = "/ml/loan_stats.delta" # 適宜変更してください
dbutils.fs.rm(DELTA_TABLE_DEFAULT_PATH, recurse=True)

# Lending Clubデータをロード&加工し、Delta LakeフォーマットでDBFSに保存します
lspq_path = "/databricks-datasets/samples/lending_club/parquet/"
data = spark.read.parquet(lspq_path)

# 必要なカラムを選択し、他の前処理を適用
features = ["loan_amnt",  "annual_inc", "dti", "delinq_2yrs","total_acc", "total_pymnt", "issue_d", "earliest_cr_line"]
raw_label = "loan_status"
loan_stats_ce = data.select(*(features + [raw_label]))
print("------------------------------------------------------------------------------------------------")
print("悪いローンのラベルを作成、これにはチャージオフ、デフォルト、ローンの支払い遅延が含まれます...")
loan_stats_ce = loan_stats_ce.filter(loan_stats_ce.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
                       .withColumn("bad_loan", (~(loan_stats_ce.loan_status == "Fully Paid")).cast("string"))
loan_stats_ce = loan_stats_ce.orderBy(rand()).limit(10000) # Community Editionでも実行できる様にロードする行を限定
print("------------------------------------------------------------------------------------------------")
print("数値のカラムを適切な型にキャスト...")
loan_stats_ce = loan_stats_ce.withColumn('issue_year',  substring(loan_stats_ce.issue_d, 5, 4).cast('double')) \
                       .withColumn('earliest_year', substring(loan_stats_ce.earliest_cr_line, 5, 4).cast('double')) \
                       .withColumn('total_pymnt', loan_stats_ce.total_pymnt.cast('double'))
loan_stats_ce = loan_stats_ce.withColumn('credit_length_in_years', (loan_stats_ce.issue_year - loan_stats_ce.earliest_year))   
# Delta Lakeフォーマットでテーブルを保存
loan_stats_ce.write.format("delta").mode("overwrite").save(DELTA_TABLE_DEFAULT_PATH)

# COMMAND ----------

# MAGIC %md ## 1. 再現性確保のためにデータバージョンとロケーションをトラッキング
# MAGIC 
# MAGIC このノートブックではウィジェット経由でデータのバージョンとパスを受け入れ、将来的に明示的にデータバージョンとパスを指定することでノートブックの実行を再現できる様になっています。データバージョンを指定できることはDelta Lakeを活用することのメリットであり、後ほどレストアできる様に以前のバージョンのデータセットを保持します。

# COMMAND ----------

# ノートブックのパラメーターからデータのパスとバージョンを取得
dbutils.widgets.text(name="deltaVersion", defaultValue="1", label="テーブルのバージョン、デフォルトは最新")
dbutils.widgets.text(name="deltaPath", defaultValue="", label="テーブルのパス")

data_version = None if dbutils.widgets.get("deltaVersion") == "" else int(dbutils.widgets.get("deltaVersion"))
#DELTA_TABLE_DEFAULT_PATH = "/ml/loan_stats.delta" # 適宜変更してください
data_path = DELTA_TABLE_DEFAULT_PATH if dbutils.widgets.get("deltaPath")  == "" else dbutils.widgets.get("deltaPath")

print("テーブルのバージョン:", data_version)
print("テーブルのパス:", data_path)

# COMMAND ----------

# MAGIC %md ### Deltaテーブルからデータをロード
# MAGIC 
# MAGIC ウィジェットで指定されたデータパスとバージョンを用いて、Delta Lakeフォーマットでデータをロードします。

# COMMAND ----------

# バージョンパラメーターが明示的に指定されていない場合、デフォルトでは最新バージョンのテーブルを使用します
if data_version is None:
  from delta.tables import DeltaTable  
  delta_table = DeltaTable.forPath(spark, data_path)
  version_to_load = delta_table.history(1).select("version").collect()[0].version  
else:
  version_to_load = data_version

loan_stats = spark.read.format("delta").option("versionAsOf", version_to_load).load(data_path)  

# データの確認
display(loan_stats)

# COMMAND ----------

# MAGIC %md ### Deltaテーブルの履歴を確認
# MAGIC 
# MAGIC 初期状態のデータ追加、アップデート、削除、マージ、追加を含むこのテーブルに対するすべてのトランザクションはテーブルに記録されます。

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS loan_stats")
spark.sql("CREATE TABLE loan_stats USING DELTA LOCATION '" + DELTA_TABLE_DEFAULT_PATH + "'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY loan_stats

# COMMAND ----------

# MAGIC %md ### ハイパーパラメーターチューニングのために交差検証を用いたモデルのトレーニング
# MAGIC 
# MAGIC Spark MLlibを用いてMLパイプラインをトレーニングします。後で調査できる様に、チューニングの実行におけるメトリクスとパラメーターは、自動でMLflowによってトラッキングされます。

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow.spark
from pyspark.sql import SparkSession

# data_version, data_pathを含むパラメーターを自動で記録するためにautolog APIを使います
mlflow.spark.autolog()

def _fit_crossvalidator(train, features, target):
  """
  `features`のカラムを用いて、渡されたトレーニング用データフレームの`target`の2値ラベルを予測するために
  CrossValidatorモデルをフィッティングするヘルパー関数
  :param: train: トレーニングデータを格納するSparkデータフレーム
  :param: features: `train`から特徴量として使用するカラム名を含む文字列のリスト
  :param: target: 予測する`train`の2値ターゲットカラムの名前
  """
  train = train.select(features + [target])
  model_matrix_stages = [
    Imputer(inputCols = features, outputCols = features),
    VectorAssembler(inputCols=features, outputCol="features"),
    StringIndexer(inputCol="bad_loan", outputCol="label")
  ]
  lr = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features")
  pipeline = Pipeline(stages=model_matrix_stages + [lr])
  paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=5)

  cvModel = crossval.fit(train)
  return cvModel.bestModel

# COMMAND ----------

# モデルのフィッティングを行いROCを表示します
features = ["loan_amnt",  "annual_inc", "dti", "delinq_2yrs","total_acc", "credit_length_in_years"]
glm_model = _fit_crossvalidator(loan_stats, features, target="bad_loan")
lr_summary = glm_model.stages[len(glm_model.stages)-1].summary
display(lr_summary.roc)

# COMMAND ----------

print("MLパイプラインの精度: %s" % lr_summary.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowエクスペリメントランサイドバーでトレーニング結果を参照
# MAGIC 
# MAGIC 上のモデルトレーニングコードは、MLflowのランの中にメトリクスやパラメーターを自動で記録し、[MLflowランサイドバー](https://databricks.com/blog/2019/04/30/introducing-mlflow-run-sidebar-in-databricks-notebooks.html)で参照することができます。エクスペリメントランサイドバーを表示するには右上のフラスコアイコンをクリックします。

# COMMAND ----------

# MAGIC %md ### 特徴量エンジニアリング: データスキーマを進化
# MAGIC 
# MAGIC データセットの過去のバージョンを追跡するDelta Lakeを用いて、モデルパフォーマンスを改善するためにいくつかの特徴量エンジニアリングを行うことができます。最初に、ローンごとに支払い金額とローン金額の合計を捉える特徴量を追加します。

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("ローンごとの支払い、ローン金額の合計を計算...")
loan_stats_new = loan_stats.withColumn('net', round( loan_stats.total_pymnt - loan_stats.loan_amnt, 2))

# COMMAND ----------

# MAGIC %md スキーマを安全に進化させるように、`mergeSchema`を指定して更新したテーブルを保存します。

# COMMAND ----------

loan_stats_new.write.option("mergeSchema", "true").format("delta").mode("overwrite").save(DELTA_TABLE_DEFAULT_PATH)

# COMMAND ----------

# オリジナルのスキーマと更新したスキーマの違いを確認します
set(loan_stats_new.schema.fields) - set(loan_stats.schema.fields)

# COMMAND ----------

# MAGIC %md 
# MAGIC 更新したデータでモデルを再トレーニングし、オリジナルのモデルとパフォーマンスを比較します。

# COMMAND ----------

# ROCを表示
glm_model_new = _fit_crossvalidator(loan_stats_new, features + ["net"], target="bad_loan")
lr_summary_new = glm_model_new.stages[len(glm_model_new.stages)-1].summary
display(lr_summary_new.roc)

# COMMAND ----------

print("MLパイプラインの精度: %s" % lr_summary_new.accuracy)

# COMMAND ----------

# MAGIC %md ## 2. オリジナルのデータバージョンを使用したランの特定
# MAGIC 
# MAGIC 特徴量エンジニアリングのステップを経て、モデルの精度は ~80% から ~95% に改善しました。この様に思うかもしれません: オリジナルのデータセットで構築したすべてのモデルを特徴量エンジニアリングしたデータセットで再トレーニングしたらどうなるのだろう？モデルパフォーマンスに動揺の改善が見られるのだろうか？
# MAGIC 
# MAGIC オリジナルデータセットに対して行われた他のランを特定するには、MLflowの`mlflow.search_runs` APIを使います。

# COMMAND ----------

mlflow.search_runs(filter_string="tags.sparkDatasourceInfo LIKE '%path=dbfs:{path}%'".format(path=data_path, version=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. データのスナップショットをロードし、ランを再現
# MAGIC 
# MAGIC 最後に、モデルの再トレーニングに使うデータの特定のバージョンをロードすることができます。これを行うには、シンプルに上のウィジェットでデータバージョン1(特徴量エンジニアリングをおこなったデータに対応)を指定し、ノートブックのセクション1を再実行します。
