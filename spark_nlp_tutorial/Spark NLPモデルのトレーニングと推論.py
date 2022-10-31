# Databricks notebook source
# MAGIC %md
# MAGIC # Spark NLPモデルのトレーニングと推論
# MAGIC 
# MAGIC このノートブックではSpark NLPによる固有表現抽出(Named Entity Extraction)を行います。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 要件
# MAGIC 
# MAGIC Spark NLPを使うには、[互換性のあるランタイムバージョン](https://nlp.johnsnowlabs.com/docs/en/install#databricks-support)が稼働するクラスターを作成、あるいは使用する必要があります。`com.johnsnowlabs.nlp:spark-nlp_2.12:4.1.0`のようなMavenコーディネートを用いてSpark NLPをクラスターにインストールしてください。

# COMMAND ----------

# MAGIC %pip install sparknlp mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## サンプルのトレーニング、評価データのロード

# COMMAND ----------

!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/conll2003/eng.train
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/src/test/resources/conll2003/eng.testa

# COMMAND ----------

from sparknlp.training import CoNLL
training_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.train')
test_data = CoNLL().readDataset(spark, 'file:/databricks/driver/eng.testa')

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータにパイプラインをフィット

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

import mlflow
mlflow_run = mlflow.start_run()

max_epochs=1
lr=0.003
batch_size=32
random_seed=0
verbose=1
validation_split= 0.2
evaluation_log_extended= True
enable_output_logs= True
include_confidence= True
output_logs_path="dbfs:/ner_logs"

dbutils.fs.mkdirs(output_logs_path)

nerTagger = NerDLApproach()\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(max_epochs)\
  .setLr(lr)\
  .setBatchSize(batch_size)\
  .setRandomSeed(random_seed)\
  .setVerbose(verbose)\
  .setValidationSplit(validation_split)\
  .setEvaluationLogExtended(evaluation_log_extended)\
  .setEnableOutputLogs(enable_output_logs)\
  .setIncludeConfidence(include_confidence)\
  .setOutputLogsPath(output_logs_path)

# MLflowにモデルトレーニングパラメーターを記録
mlflow.log_params({
  "max_epochs": max_epochs,
  "lr": lr,
  "batch_size": batch_size,
  "random_seed": random_seed,
  "verbose": verbose,
  "validation_split": validation_split,
  "evaluation_log_extended": evaluation_log_extended,
  "enable_output_logs": enable_output_logs,
  "include_confidence": include_confidence,
  "output_logs_path": output_logs_path
})

# トレーニングデータ、評価用データはすでにトークナイズされているので、エンべディングモデルを直接適用し、
# エンベンディングに固有エンティティ識別器をフィットします
glove_embeddings = WordEmbeddingsModel.pretrained('glove_100d')\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

ner_pipeline = Pipeline(stages=[
          glove_embeddings,
          nerTagger
 ])

ner_model = ner_pipeline.fit(training_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テストデータによる評価

# COMMAND ----------

predictions = ner_model.transform(test_data)

# COMMAND ----------

# DBTITLE 1,トークン、正解データ、予測
import pyspark.sql.functions as F
display(predictions.select(F.col('token.result').alias("tokens"),
                           F.col('label.result').alias("ground_truth"),
                           F.col('ner.result').alias("predictions")).limit(3))

# COMMAND ----------

# 評価のために行あたり1トークンになる様にデータを再フォーマット
predictions_pandas = predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                                     predictions.label.result,
                                                     predictions.ner.result)).alias("cols")) \
                              .select(F.expr("cols['0']").alias("token"),
                                      F.expr("cols['1']").alias("ground_truth"),
                                      F.expr("cols['2']").alias("prediction")).toPandas()

# COMMAND ----------

# DBTITLE 1,最初の20トークンとラベル
display(predictions_pandas.head(20))

# COMMAND ----------

from sklearn.metrics import classification_report

# 分類レポートの生成
report = classification_report(predictions_pandas['ground_truth'], predictions_pandas['prediction'], output_dict=True)

# 直接MLflowに精度を記録
mlflow.log_metric("accuracy", report["accuracy"])
# MLflowにアーティファクトとしてトークンタイプごとの完全な分類を記録
mlflow.log_dict(report, "classification_report.yaml")

# ノートブックで参照するためにレポートを出力
print (classification_report(predictions_pandas['ground_truth'], predictions_pandas['prediction']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## テキストの予測パイプラインの構築および記録

# COMMAND ----------

document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentence')

token = Tokenizer()\
    .setInputCols(['sentence'])\
    .setOutputCol('token')

# パイプラインからモデルを取得
loaded_ner_model = ner_model.stages[1]

converter = NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_span")

ner_prediction_pipeline = Pipeline(
    stages = [
        document,
        sentence,
        token,
        glove_embeddings,
        loaded_ner_model,
        converter])

# COMMAND ----------

# 空のデータフレームにフィットさせることで、モデルを再トレーニングすることなしにパイプラインモデルを構築することができます
empty_data = spark.createDataFrame([['']]).toDF("text")
prediction_model = ner_prediction_pipeline.fit(empty_data)

# COMMAND ----------

# Databricksランタイム11.2や11.2MLでは、モデルロギングはDatabricks MLflowユーティリティを用いて記録されます。
# Databrikcsランタイム11.2のDBFS向けDatabricks MLflowユーティリティは、
# Spark NLPがモデルのシリアライズ化に使用するすべてのファイルシステムコールをサポートしていません。
# 以下のコマンドではMLflowユーティリティの使用を無効化し、標準のDBFSサポートを使用する様にしています。
import os
if os.environ["DATABRICKS_RUNTIME_VERSION"].startswith('11.'):
  os.environ["DISABLE_MLFLOWDBFS"] = "True"

# COMMAND ----------

## モデルをMLflowに記録し、モデルURIへのリファレンスを構築します
model_name = "NerPipelineModel"
mlflow.spark.log_model(prediction_model, model_name)
mlflow.end_run()
mlflow_model_uri = "runs:/{}/{}".format(mlflow_run.info.run_id, model_name)
display(mlflow_model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをテキストに適用

# COMMAND ----------

# サンプルテキストの作成
text = "From the corner of the divan of Persian saddle-bags on which he was lying, smoking, as was his custom, innumerable cigarettes, Lord Henry Wotton could just catch the gleam of the honey-sweet and honey-coloured blossoms of a laburnum, whose tremulous branches seemed hardly able to bear the burden of a beauty so flamelike as theirs; and now and then the fantastic shadows of birds in flight flitted across the long tussore-silk curtains that were stretched in front of the huge window, producing a kind of momentary Japanese effect, and making him think of those pallid, jade-faced painters of Tokyo who, through the medium of an art that is necessarily immobile, seek to convey the sense of swiftness and motion. The sullen murmur of the bees shouldering their way through the long unmown grass, or circling with monotonous insistence round the dusty gilt horns of the straggling woodbine, seemed to make the stillness more oppressive. The dim roar of London was like the bourdon note of a distant organ."
sample_data = spark.createDataFrame([[text]]).toDF("text")

# モデルをロードして適用
mlflow_model = mlflow.spark.load_model(mlflow_model_uri)
predictions = mlflow_model.transform(sample_data)

# COMMAND ----------

# DBTITLE 1,生の予測結果
display(predictions)

# COMMAND ----------

# DBTITLE 1,抽出されたエンティティ
display(predictions.select(F.explode(F.arrays_zip(predictions.ner_span.result,predictions.ner_span.metadata)).alias("entities")) 
      .select(F.expr("entities['0']").alias("chunk"),
              F.expr("entities['1'].entity").alias("entity")))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
