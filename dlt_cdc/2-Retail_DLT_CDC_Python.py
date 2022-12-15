# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # DLTパイプラインにおけるCDCの実装: チェンジデータキャプチャ
# MAGIC 
# MAGIC -----------------
# MAGIC ###### By Morgan Mazouchi
# MAGIC 
# MAGIC ###### Resource [Change data capture with Delta Live Tables](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-cdc.html)
# MAGIC -----------------
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/morganmazouchi/Delta-Live-Tables/main/Images/dlt%20end%20to%20end%20flow.png">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## チェンジデータキャプチャ(CDC)の重要性
# MAGIC 
# MAGIC チェンジデータキャプチャ(CDC)はデータベースやデータウェハウスのようなデータストレージにおけるレコードの変更をキャプチャするプロセスです。これらの変更は通常、データの削除、追加、更新のようなオペレーションとみなされます。
# MAGIC 
# MAGIC データベースをエクスポートするデータベースのダンプを取得し、レイクハウス/データウェアハウス/データレイクにインポートするデータレプリケーションはシンプルな方法ですが、これはスケーラブルなアプローチとは言えません。
# MAGIC 
# MAGIC データベースでなされた変更のみをキャプチャし、これらの変更をターゲットデータベースに適用するのがチェンジデータキャプチャです。CDCはオーバーヘッドを削減し、リアルタイム分析をサポートします。バルクロードによる更新をすることなしに、インクリメンタルなロードを実現します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### CDCのアプローチ
# MAGIC 
# MAGIC **1 - 内製CDCプロセスの開発:** 
# MAGIC 
# MAGIC ***複雑なタスク:*** CDCのデータレプリケーションは、一度切りの簡単なソリューションではありません。データベースプロバイダーによる差異から、レコードのフォーマットは異なり、ログレコードへのアクセスが不便であることからCDCは困難なものとなります。
# MAGIC 
# MAGIC ***定期的なメンテナンス:*** CDCプロセスのスクリプトの記述は最初の一歩です。上述の変化を定期的にマッピングできるカスタマイズされたソリューションをメンテナンスしなくてはなりません。これには、多くの時間とリソースを必要とします。
# MAGIC 
# MAGIC ***過度の負担:*** 企業の開発者はすでに公式なクエリーの付加に晒されています。カスタムのCDCソリューションを構築する追加の工数は、既存の収益を生み出しているプロジェクトに影響を与えます。
# MAGIC 
# MAGIC **2 - CDCツールの活用:** Debezium, Hevo Data, IBM Infosphere, Qlik Replicate, Talend, Oracle GoldenGate, StreamSetsなど
# MAGIC 
# MAGIC このデモリポジトリでは、CDCツールから到着するCDCデータを活用します。CDCツールはデータベースログを読み込むので、特定カラムを更新する際に開発者を頼る必要がありません。
# MAGIC 
# MAGIC — DebeziumのようなCDCツールは変更されたすべての行をキャプチャします。Kafkaログにおいて、アプリケーションが利用し始めた以降のデータ変更履歴を記録します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## セットアップ/要件:
# MAGIC 
# MAGIC パイプラインとしてこのノートブックを実行する前に、生成されたCDCデータに対してこのノートブックが動作するように、DLTパイプラインに[1-CDC_DataGenerator]($./1-CDC_DataGenerator)ノートブックのパスを含めるようにしてください。
# MAGIC 
# MAGIC ```
# MAGIC "configuration": {
# MAGIC         "source": "/tmp/takaaki.yayoi@databricks.com/demo/cdc_raw"
# MAGIC     }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## お使いのSQLデータベースをどのようにレイクハウスに同期するのか？
# MAGIC 
# MAGIC CDCツール、Auto Loader、DLTパイプラインを用いたCDCフロー:
# MAGIC 
# MAGIC - CDCツールがデータベースログを読み込み、変更を含むJSONメッセージを生成し、Kafkaに対して変更説明を伴うレコードをストリーミング
# MAGIC - KafkaがINSERT, UPDATE, DELETEオペレーションを含むメッセージをストリーミングし、クラウドオブジェクトストレージ(S3、ADLSなど)に格納
# MAGIC - Auto Loaderを用いてクラウドオブジェクトストレージからメッセージをインクリメンタルにロードし、生のメッセージとして保存するためにブロンズテーブルに格納
# MAGIC - 次に、クレンジングされたブロンズレイヤーテーブルに APPLY CHANGES INTO を実行し、後段のシルバーテーブルに最新の更新データを伝搬
# MAGIC 
# MAGIC 外部データベースからCDCデータを処理するための実装を以下に示します。入力はKafkaのようなメッセージキューを含む任意のフォーマットになり得ることに注意してください。
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/morganmazouchi/Delta-Live-Tables/main/Images/cdc_flow_new.png" alt='Make all your data ready for BI and ML'/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### DebeziumのようなCDCツールの出力はどのようなものか？
# MAGIC 
# MAGIC 変更データを表現するJSONメッセージは、以下の一覧と同じような興味深いフィールドを持っています:
# MAGIC 
# MAGIC - operation: オペレーションのコード(DELETE, APPEND, UPDATE, CREATE)
# MAGIC - operation_date: それぞれのオペレーションのアクションがあった日付、タイムスタンプ
# MAGIC 
# MAGIC Debeziumの出力には以下のようなフィールドが含まれます(このデモには含めていません):
# MAGIC 
# MAGIC - before: 変更前の行
# MAGIC - after: 変更後の行
# MAGIC 
# MAGIC 想定されるフィールドに関しては、[こちらのリファレンス](https://debezium.io/documentation/reference/stable/connectors/postgresql.html#postgresql-update-events)をチェックしてみてください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Auto Loader(cloud_files)を用いたインクリメンタルなデータロード
# MAGIC 
# MAGIC <div style="float:right">
# MAGIC   <img width="700px" src="https://raw.githubusercontent.com/morganmazouchi/Delta-Live-Tables/main/Images/DLT_CDC.png"/>
# MAGIC </div>
# MAGIC スキーマの更新によって、外部システムとの連携は困難となり得ます。外部データベースではスキーマの更新やカラムの追加、更新があり、我々のシステムはこれらの変更に対して頑健である必要があります。DatabricksのAuto Loader(`cloudFiles`)は、すぐにスキーマ推定とスキーマ進化を取り扱うことができます。
# MAGIC 
# MAGIC Auto Loaderを用いることで、クラウドストレージから数百万のファイルを取り込むことができ、大規模なスキーマ推定や進化をサポートすることができます。このノートブックでは、ストリーミング(とバッチ)データを取り扱うためにAuto Loaderを活用します。
# MAGIC 
# MAGIC パイプラインを作成し、外部のプロバイダーによってデリバリーされる生のJSONデータを取り込むためにAuto Loaderを使いましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Pythonの構文
# MAGIC 
# MAGIC 関連メソッドを使うには、`dlt` Pythonモジュールをインポートする必要があります。ここでは、`pyspark.sql.functions`もインポートします。
# MAGIC 
# MAGIC DLTのテーブル、ビュー、関連設定は[デコレーター](https://www.python.org/dev/peps/pep-0318/#current-syntax)を用いて設定されます。
# MAGIC 
# MAGIC Pythonのデコレーターを触ったことがない場合には、Pythonスクリプトで次に表現される関数とやり取りを行う、`@`で始まる関数やクラスであると考えてください。
# MAGIC 
# MAGIC `@dlt.table`デコレーターは、Python関数をDelta Liveテーブルに変換する基本的なメソッドとなります。
# MAGIC 
# MAGIC 以下では到着データを探索していきます。

# COMMAND ----------

# DBTITLE 1,ブロンズテーブル - Auto Loader & DLT
## ストレージパスから取得する生のJSONデータを含むブロンズテーブルの作成
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

source = spark.conf.get("source")

@dlt.table(name="customer_bronze",
                  comment = "クラウドオブジェクトストレージのランディングゾーンからインクリメンタルに取り込まれる新規顧客",
  table_properties={
    "quality": "bronze"
  }
)

def customer_bronze():
  return (
    spark.readStream.format("cloudFiles") \
      .option("cloudFiles.format", "json") \
      .option("cloudFiles.inferColumnTypes", "true") \
      .load(f"{source}/customers")
  )

# COMMAND ----------

# DBTITLE 1,シルバーレイヤー - クレンジングされたテーブル (制約の適用)
"""
@dlt.view(name="customer_bronze_clean_v",
  comment="クレンジングされたブロンズ顧客ビュー(シルバーになるビューです)")

@dlt.expect_or_drop("valid_id", "id IS NOT NULL")
@dlt.expect("valid_address", "address IS NOT NULL")
@dlt.expect_or_drop("valid_operation", "operation IS NOT NULL")

def customer_bronze_clean_v():
  return dlt.read_stream("customer_bronze") \
            .select("address", "email", "id", "firstname", "lastname", "operation", "operation_date", "_rescued_data")
"""

# COMMAND ----------

@dlt.table(name="customer_bronze_clean",
  comment="クレンジングされたブロンズ顧客ビュー(シルバーになるテーブルです)")

@dlt.expect_or_drop("valid_id", "id IS NOT NULL")
@dlt.expect("valid_address", "address IS NOT NULL")
@dlt.expect_or_drop("valid_operation", "operation IS NOT NULL")

def customer_bronze_clean():
  return dlt.read_stream("customer_bronze") \
            .select("address", "email", "id", "firstname", "lastname", "operation", "operation_date", "_rescued_data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## シルバーテーブルのマテリアライズ
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/morganmazouchi/Delta-Live-Tables/main/Images/cdc_silver_layer.png" alt='Make all your data ready for BI and ML' style='float: right' width='1000'/>
# MAGIC 
# MAGIC シルバーテーブルである`customer_silver`には、最新のビューが含まれます。オリジナルテーブルの複製となります。
# MAGIC 
# MAGIC 後段の`シルバー`レイヤーに`Apply Changes Into`オペレーションを伝播させるには、DLTパイプライン設定で`applyChanges`設定を追加して有効化することで、明示的にこの機能を有効化する必要があります。

# COMMAND ----------

# DBTITLE 1,不要な顧客レコードの削除 - シルバーテーブル - DLT Python
dlt.create_target_table(name="customer_silver",
  comment="クレンジング、マージされた顧客",
  table_properties={
    "quality": "silver"
  }
)

# COMMAND ----------

dlt.apply_changes(
  target = "customer_silver", # マテリアライズされる顧客テーブル
  source = "customer_bronze_clean", # 入力のCDC
  keys = ["id"], # upsert/deleteするために行をマッチする際の主キー
  sequence_by = col("operation_date"), # 最新の値を取得するためにオペレーション日による重複排除
  apply_as_deletes = expr("operation = 'DELETE'"), # DELETEの条件
  except_column_list = ["operation", "operation_date", "_rescued_data"] # メタデータカラムの削除
)

# COMMAND ----------

# MAGIC %md
# MAGIC 次のステップでは、DLTパイプラインを作成し、このノートブックのパスを追加し、**applychangesをtrueにする設定を追加します**。詳細に関しては、ノートブック`PipelineSettingConfiguration.json`を参照ください。
# MAGIC 
# MAGIC パイプラインを実行したら、イベントログとリネージュデータをモニタリングするために、[3. Retail_DLT_CDC_Monitoring]($./3. Retail_DLT_CDC_Monitoring)をチェックしてください。
