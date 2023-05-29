# Databricks notebook source
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,ドキュメントパスの設定
config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
# ベクトルストアの格納パス
config['vector_store_path'] = '/dbfs/tmp/takaaki.yayoi@databricks.com/qabot/vector_store_jpn' # /dbfs/... はローカルファイルシステムにおける表現です

# COMMAND ----------

# DBTITLE 1,データベースの作成
config['database_name'] = 'qabot_taka_yayoi_jpn' # 適宜変更してください

# 存在しない場合にはデータベースを作成
_ = spark.sql(f"create database if not exists {config['database_name']}")

# 現在のデータベースコンテキストの設定
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,トークンのための環境変数の設定
import os

# 実際に設定したシークレットのスコープとキーを指定します
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("demo-token-takaaki.yayoi", "openai_api_key")

# COMMAND ----------

# DBTITLE 1,mlflowの設定
import mlflow
config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator_taka_jpn' # 適宜変更してください
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,OpenAIモデルの設定
config['openai_embedding_model'] = 'text-embedding-ada-002'
config['openai_chat_model'] = "gpt-3.5-turbo"
config['system_message_template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question."""
config['human_message_template'] = """Given the context: {context}. Answer the question {question}."""
config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,評価の設定
config["eval_dataset_path"]= "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,デプロイメントの設定
config['openai_key_secret_scope'] = "demo-token-takaaki.yayoi" # シークレットスコープの手順に関しては `./RUNME` ノートブックをご覧ください。実際に使用するシークレットスコープと一貫していることを確認してください。
config['openai_key_secret_key'] = "openai_api_key" #シークレットスコープの手順に関しては `./RUNME` ノートブックをご覧ください。実際に使用するシークレットキーと一貫していることを確認してください。
config['serving_endpoint_name'] = "llm-qabot-endpoint-taka-jpn" # サービングエンドポイント名
