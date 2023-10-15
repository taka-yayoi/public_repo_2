# Databricks notebook source
# MAGIC %md
# MAGIC # コンピュータビジョンとリアルタイムサーバレス推論を用いたプリント基盤(PCB)の製品品質調査
# MAGIC
# MAGIC <div style="float:right">
# MAGIC <img width="500px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/PCB1.png">
# MAGIC </div>
# MAGIC
# MAGIC このソリューションアクセラレータでは、製品の品質調査のためのエンドツーエンドのパイプラインのデプロイにおいて、Databricksがどのように役立つのかを説明します。モデルは[サーバレスリアルタイム推論](https://docs.databricks.com/archive/serverless-inference-preview/serverless-real-time-inference.html)を用いてデプロイされます。
# MAGIC
# MAGIC [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/)検知データセットを用い、PCB画像の以上を検知するためのパイプラインを構築します。
# MAGIC
# MAGIC ## なぜ画像による品質調査を？
# MAGIC
# MAGIC 画像による品質調査は製造業の文脈においては一般的な課題となっています。スマートマニュファクチャリングを提供する際の鍵となります。
# MAGIC
# MAGIC ## プロダクションレベルのパイプラインの実装
# MAGIC
# MAGIC 事前学習済みディープラーニングモデル、転送学習、高レベルのフレームワークによって、近年では画像分類問題は簡単になってきています。データサイエンスチームはそのようなモデルをクイックにデプロイすることはできますが、プロダクションレベルのエンドツーエンドのパイプラインの実装、画像の利用、MLOps/ガバナンスの必要性、最終的な結果の提供においては、依然として本当の課題が存在し続けています。
# MAGIC
# MAGIC Databricksレイクハウスは、このような全体的なプロセスをシンプルにするように設計されており、データサイエンティストはコアのユースケースにフォーカスすることができます。
# MAGIC
# MAGIC 品質調査モデルを構築するために、Torchvisionを使用します。しかし、他のライブラリで同じアーキテクチャを活用することも可能です。TorchvisionライブラリはPyTorchプロジェクトの一部であり、ディープラーニングで人気のフレームワークとなっています。Torchvisionは、モデルアーキテクチャ、よく使われるデータセット、画像のトランスフォーマーと共に提供されています。
# MAGIC
# MAGIC パイプライン構築の最初のステップは、データの取り込みです。Databricksでは、画像(非構造化データ)を含む任意のソースのデータをロードすることができます。これは、効率的かつ分散された方法で画像のコンテンツとともにテーブルに格納され、ラベルと関連づけられます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 品質調査画像パイプライン
# MAGIC
# MAGIC これが我々が構築するパイプラインです。2つのデータセットを取り込みます。すなわち:
# MAGIC
# MAGIC * PCBを含む生のサテライトイメージ(jpg)
# MAGIC * CSVとして保存されている不良のタイプを示すラベル
# MAGIC
# MAGIC 最初にこのデータをインクリメンタルにロードするデータパイプラインにフォーカスし、最終的なゴールドテーブルを作成します。
# MAGIC
# MAGIC このテーブルは、我々の画像からリアルタイムで異常を検知するために、ML分離モデルをトレーニングするために活用されます！
# MAGIC
# MAGIC <img width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/pipeline.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### https://registry.opendata.aws/visa/ からデータセットをダウンロード
# MAGIC
# MAGIC [https://registry.opendata.aws/visa/](https://registry.opendata.aws/visa/)からデータセットをダウンロードするために`bash`コマンドを使います。
# MAGIC
# MAGIC データはAWS S3に格納されているので、AWS CLIライブラリ(`awscli`)をインストールする必要があります。

# COMMAND ----------

# MAGIC %pip install awscli

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/data
# MAGIC aws s3 cp --no-progress --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar /tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/data
# MAGIC tar xf /tmp/VisA_20220922.tar --no-same-owner -C /tmp/data/ 

# COMMAND ----------

# MAGIC %md
# MAGIC ## いくつかのPCB画像を見てみましょう
# MAGIC
# MAGIC ネイティブなPythonの方法で`matplotlib`を使って画像を表示することができます。
# MAGIC
# MAGIC 正常な画像がどのようなものであるのか、異常があるものがどのようなものであるかを見てみましょう。

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=300):
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.imshow(img, interpolation="nearest", aspect="auto")


display_image("/tmp/data/pcb1/Data/Images/Normal/0000.JPG")
display_image("/tmp/data/pcb1/Data/Images/Anomaly/000.JPG")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データをDBFSに移動しましょう
# MAGIC
# MAGIC クイックな覚書: Databricksファイルシステム(DBFS)は、Databricksワークスペースにマウントされ、Databricksクラスターから利用できる分散ファイルシステムです。DBFSは、ネイティブなクラウドストレージのAPIコールをUnixライクなファイルシステムコールにマッピングする、スケーラブルなオブジェクトストレージ上の抽象化レイヤーとなります。

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /dbfs/pcb1
# MAGIC mkdir -p /dbfs/pcb1/labels 
# MAGIC cp -r /tmp/data/pcb1/Data/Images/ /dbfs/pcb1/
# MAGIC cp /tmp/data/pcb1/image_anno.csv /dbfs/pcb1/labels/

# COMMAND ----------

# MAGIC %sql
# MAGIC USE takaakiyayoi_catalog.pcb;
# MAGIC DROP TABLE IF EXISTS circuit_board;
# MAGIC DROP TABLE IF EXISTS circuit_board_gold;
# MAGIC DROP TABLE IF EXISTS circuit_board_label;

# COMMAND ----------

cloud_storage_path="/pcb1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto LoaderによるCSVラベルファイルをロード
# MAGIC
# MAGIC Databricksの[Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html)を用いることで、CSVファイルを簡単にロードすることができます。

# COMMAND ----------

from pyspark.sql.functions import substring_index, col

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", True)
    .option("cloudFiles.schemaLocation", f"{cloud_storage_path}/circuit_board_label_schema")
    .load(f"{cloud_storage_path}/labels/")
    .withColumn("filename", substring_index(col("image"), "/", -1))
    .select("filename", "label")
    .withColumnRenamed("label", "labelDetail")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{cloud_storage_path}/circuit_board_label_checkpoint")
    .toTable("circuit_board_label")
    .awaitTermination()
)
display(spark.table("circuit_board_label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto Loaderによるバイナリーファイルのロード
# MAGIC
# MAGIC これで、画像をロードするためにAuto Loaderを用い、ラベルのカラムを作成するためにspark関数を活用することができます。
# MAGIC また、テーブルとして画像のコンテンツとラベルを簡単に表示することができます。

# COMMAND ----------

from pyspark.sql.functions import substring_index, col, when

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.JPG")
    .option("recursiveFileLookup", "true")
    .option("cloudFiles.schemaLocation", f"{cloud_storage_path}/circuit_board_schema")
    .load(f"{cloud_storage_path}/Images/")
    .withColumn("filename", substring_index(col("path"), "/", -1))
    .withColumn(
        "labelName",
        when(col("path").contains("Anomaly"), "anomaly").otherwise("normal"),
    )
    .withColumn("label", when(col("labelName").eqNullSafe("anomaly"), 1).otherwise(0))
    .select("filename", "content", "label", "labelName")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{cloud_storage_path}/circuit_board_checkpoint")
    .toTable("circuit_board")
    .awaitTermination()
)
display(spark.table("circuit_board"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ラベルと画像テーブルをマージしましょう

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE circuit_board_gold as (
# MAGIC   select
# MAGIC     cb.*,
# MAGIC     labelDetail
# MAGIC   from
# MAGIC     circuit_board cb
# MAGIC     inner join circuit_board_label cbl on cb.filename = cbl.filename
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 画像テーブルに対する自動最適化を有効化することができます
# MAGIC
# MAGIC 自動最適化は2つの補完的な機能となります: 最適化書き込みとオートコンパクションです。

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE circuit_board_gold SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true);
# MAGIC ALTER TABLE circuit_board SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true)

# COMMAND ----------

# MAGIC %md
# MAGIC このテーブルに対して任意のSQLコマンドを実行することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   circuit_board_gold
# MAGIC limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### データサインスチームが活用するデータセットの準備ができました
# MAGIC
# MAGIC これですべてです！プロダクションレベルのデータ取り込みパイプラインをデプロイしました。
# MAGIC
# MAGIC 我々の画像はインクリメンタルに取り込まれ、ラベルデータセットと結合されます。
# MAGIC
# MAGIC 異常検知に必要な[モデルを構築]($./01_ImageClassificationPytorch)するために、データサイエンティストによってこのデータがどのように活用されるのかを見てみましょう。
