# Databricks notebook source
# MAGIC %md # Delta Live Tables - モニタリング
# MAGIC 
# MAGIC それぞれのDLTパイプラインは、パイプラインで定義されたストレージロケーションに自信のイベントテーブルを持ちます。このテーブルから、何が起きているのか、パイプラインを通過するデータの品質を確認することができます。
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/morganmazouchi/Delta-Live-Tables/main/Images/dlt%20end%20to%20end%20flow.png"/>

# COMMAND ----------

# MAGIC %md ## 01 - 設定

# COMMAND ----------

dbutils.widgets.removeAll()
# -- 古いウィジェットを削除

# COMMAND ----------

# パスはパイプラインの設定に合わせて適宜変更してください
dbutils.widgets.text('storage_path','/tmp/takaaki.yayoi@databricks.com/demo/dlt_cdc')

# COMMAND ----------

# MAGIC %md ## 02 - セットアップ

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- 適宜データベースを指定してください
# MAGIC CREATE TABLE IF NOT EXISTS cdc_data_taka.demo_cdc_dlt_system_event_log_raw using delta LOCATION '$storage_path/system/events';
# MAGIC select * from cdc_data_taka.demo_cdc_dlt_system_event_log_raw;

# COMMAND ----------

# MAGIC %md ## Delta Live Tablesのエクスペクテーション分析
# MAGIC 
# MAGIC Delta Live Tablesはエクスペクテーションを通じてデータ品質を追跡します。これらのエクスペクテーションはDLTのログイベントとともに技術的なテーブルとして格納されます。この情報を分析するために、シンプルにビューを作成することができます。
# MAGIC 
# MAGIC **ウィジェットでDLTのストレージパスを設定するようにしてください!**
# MAGIC 
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fretail%2Fdlt%2Fnotebook_quality_expectations&dt=DATA_PIPELINE">
# MAGIC <!-- [metadata={"description":"Notebook extracting DLT expectations as delta tables used to build DBSQL data quality Dashboard.",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{"Dashboards": ["DLT Data Quality Stats"]},
# MAGIC  "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["autoloader", "copy into"]},
# MAGIC  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1 - イベントログの分析
# MAGIC 
# MAGIC `details`カラムにはイベントログに送信されたイベントごとのメタデータが含まれています。イベントのタイプに応じてフィールドが異なります。行く疲れの例を示します:
# MAGIC 
# MAGIC | イベントのタイプ | 挙動 |
# MAGIC | --- | --- |
# MAGIC | `user_action` | パイプラインの作成のようなアクションが行われた際に生じるイベント |
# MAGIC | `flow_definition`| パイプラインのデプロイメントやアップデートが行われた際に生じるイベントであり、リネージュ、スキーマ、実行計画情報を持ちます |
# MAGIC | `output_dataset` と `input_datasets` | 出力のテーブル/ビュー、前段のテーブル/ビュー |
# MAGIC | `flow_type` | コンプリートフローか追加のフローか |
# MAGIC | `explain_text` | Sparkの実行計画 |
# MAGIC | `flow_progress`| データフローがデータバッチの処理を開始あるいは完了した際に生じるイベント |
# MAGIC | `metrics` | 現在は`num_output_rows`が含まれています |
# MAGIC | `data_quality` (`dropped_records`), (`expectations`: `name`, `dataset`, `passed_records`, `failed_records`)| この特定のデータセットに対するデータ品質ルールの結果の配列が含まれます   * `expectations`|

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cdc_data_taka.demo_cdc_dlt_system_event_log_raw

# COMMAND ----------

# DBTITLE 1,イベントログ - タイムスタンプで並び替えられた生のイベント
# MAGIC %sql
# MAGIC -- 適宜データベースを指定してください
# MAGIC SELECT 
# MAGIC        id,
# MAGIC        timestamp,
# MAGIC        sequence,
# MAGIC        event_type,
# MAGIC        message,
# MAGIC        level, 
# MAGIC        details
# MAGIC   FROM cdc_data_taka.demo_cdc_dlt_system_event_log_raw
# MAGIC  ORDER BY timestamp ASC
# MAGIC ;  

# COMMAND ----------

# MAGIC %md ### 2 - DLTのリネージュ

# COMMAND ----------

# MAGIC %sql
# MAGIC -- タイプと最新の変更ごとに出力データセットを一覧します
# MAGIC -- 適宜データベースを指定してください
# MAGIC create or replace temp view cdc_dlt_expectations as (
# MAGIC   SELECT 
# MAGIC     id,
# MAGIC     timestamp,
# MAGIC     details:flow_progress.metrics.num_output_rows as output_records,
# MAGIC     details:flow_progress.data_quality.dropped_records,
# MAGIC     details:flow_progress.status as status_update,
# MAGIC     explode(from_json(details:flow_progress.data_quality.expectations
# MAGIC              ,'array<struct<dataset: string, failed_records: bigint, name: string, passed_records: bigint>>')) expectations
# MAGIC   FROM cdc_data_taka.demo_cdc_dlt_system_event_log_raw
# MAGIC   where details:flow_progress.data_quality.expectations is not null
# MAGIC   ORDER BY timestamp);
# MAGIC 
# MAGIC select * from cdc_dlt_expectations

# COMMAND ----------

# MAGIC %sql
# MAGIC ----------------------------------------------------------------------------------------
# MAGIC -- リネージュ
# MAGIC ----------------------------------------------------------------------------------------
# MAGIC SELECT max_timestamp,
# MAGIC        details:flow_definition.output_dataset,
# MAGIC        details:flow_definition.input_datasets,
# MAGIC        details:flow_definition.flow_type,
# MAGIC        details:flow_definition.schema,
# MAGIC        details:flow_definition.explain_text,
# MAGIC        details:flow_definition
# MAGIC   FROM cdc_data_taka.demo_cdc_dlt_system_event_log_raw e
# MAGIC  INNER JOIN (
# MAGIC               SELECT details:flow_definition.output_dataset output_dataset,
# MAGIC                      MAX(timestamp) max_timestamp
# MAGIC                 FROM cdc_data_taka.demo_cdc_dlt_system_event_log_raw
# MAGIC                WHERE details:flow_definition.output_dataset IS NOT NULL
# MAGIC                GROUP BY details:flow_definition.output_dataset
# MAGIC             ) m
# MAGIC   WHERE e.timestamp = m.max_timestamp
# MAGIC     AND e.details:flow_definition.output_dataset = m.output_dataset
# MAGIC --    AND e.details:flow_definition IS NOT NULL
# MAGIC  ORDER BY e.details:flow_definition.output_dataset
# MAGIC ;

# COMMAND ----------

# MAGIC %md ### 3 - 品質メトリクス

# COMMAND ----------

# MAGIC %sql 
# MAGIC select sum(expectations.failed_records) as failed_records, 
# MAGIC sum(expectations.passed_records) as passed_records, 
# MAGIC expectations.name 
# MAGIC from cdc_dlt_expectations 
# MAGIC group by expectations.name

# COMMAND ----------

# MAGIC %md ### 4. ビジネス集計情報のチェック

# COMMAND ----------

# MAGIC %python 
# MAGIC import plotly.express as px
# MAGIC expectations_metrics = spark.sql("""select sum(expectations.failed_records) as failed_records, 
# MAGIC                                  sum(expectations.passed_records) as passed_records, 
# MAGIC                                  expectations.name 
# MAGIC                                  from cdc_dlt_expectations
# MAGIC                                  group by expectations.name""").toPandas()
# MAGIC px.bar(expectations_metrics, x="name", y=["passed_records", "failed_records"], title="DLT expectations metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
