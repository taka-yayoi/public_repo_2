# Databricks notebook source
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,ドキュメントパスの設定
config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
# ベクトルストアの格納パス
config['vector_store_path'] = '/dbfs/tmp/takaaki.yayoi@databricks.com/qabot_open/vector_store_jpn' # /dbfs/... はローカルファイルシステムにおける表現です

# COMMAND ----------

# DBTITLE 1,データベースの作成
config['database_name'] = 'qabot_taka_yayoi_jpn_open' # 適宜変更してください

# 存在しない場合にはデータベースを作成
_ = spark.sql(f"create database if not exists {config['database_name']}")

# 現在のデータベースコンテキストの設定
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,mlflowの設定
import mlflow
config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator_taka_jpn_open' # 適宜変更してください
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,OpenAIモデルの設定
# エンべディング変換を担うLLM
config['hf_embedding_model'] = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'
# 要約処理を担うLLM
config['hf_chat_model'] = "inu-ai/dolly-japanese-gpt-1b"

# プロンプトテンプレート
config['prompt_template'] = r"<s>\nあなたはDatabricksによって開発された有能なアシスタントであり、指定された文脈に基づいて質問に回答することを得意としており、文脈は文書です。文脈が回答を決定するのに十分な情報を提供しない場合には、わかりませんと言ってください。文脈が質問に適していない場合には、わかりませんと言ってください。文脈から良い回答が見つからない場合には、わかりませんと言ってください。問い合わせが完全な質問になっていない場合には、わからないと言ってください。あなたは同じ言葉を繰り返しません。以下は、文脈を示す文書と、文脈のある質問の組み合わせです。文書を要約することで質問を適切に満たす回答をしなさい。\n[SEP]\n文書:\n{context}\n[SEP]\n質問:\n{question}\n[SEP]\n回答:\n"
config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,評価の設定
config["eval_dataset_path"]= "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,デプロイメントの設定
config['serving_endpoint_name'] = "llm-qabot-endpoint-taka-jpn-open" # サービングエンドポイント名
