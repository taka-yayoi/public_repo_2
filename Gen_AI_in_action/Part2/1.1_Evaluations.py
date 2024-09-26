# Databricks notebook source
# MAGIC %md
# MAGIC # 評価
# MAGIC RAGに対する評価の実行はまだ科学というよりアートです。
# MAGIC
# MAGIC llama_indexを使用して評価用の質問を生成します。そして、llama_indexの組み込み評価プロンプトを使用します。

# COMMAND ----------

# MAGIC %pip install llama_index==0.10.25 langchain==0.1.13 llama-index-llms-langchain llama-index-embeddings-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import nest_asyncio

# 非同期呼び出しが必要となります
nest_asyncio.apply()

# COMMAND ----------

# MAGIC %md
# MAGIC # Llama Indexのご紹介
# MAGIC langchainと同様に、llama_indexもLLMロジックのオーケストレーション層です\
# MAGIC 異なる点は、llama_indexはRAGやインテリジェントなインデキシングにより特化していることです\
# MAGIC Langchainはより汎用的で、複雑なワークフローを可能にすることに焦点を当てています
# MAGIC
# MAGIC このノートブックで使用するLlama Indexのいくつかの重要な概念:
# MAGIC - サービスコンテキスト - llmモデル/エンベディングを保持するラッパークラス
# MAGIC - インデックス - これはllama indexのコアです。基本的に、インデックスはテキストとエンベディングを含む複雑なノード構造から成り立っています

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Llama Indexのデフォルトモデルの設定
# MAGIC デフォルトの設定を変更しない場合、Llama_indexはデフォルトでOpenAIにアクセスします

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding

# エンべディングモデルとモデル名の設定
embedding_model = 'databricks-bge-large-en'
model_name = 'databricks-dbrx-instruct'

# LLMモデルの初期化
llm_model = ChatDatabricks(
  target_uri='databricks',
  endpoint=model_name,
  temperature=0.1
)

# エンべディングモデルの初期化
embeddings = DatabricksEmbeddings(endpoint=embedding_model)

# LangChainLLMとLangchainEmbeddingの初期化
llama_index_chain = LangChainLLM(llm=llm_model)
llama_index_embeddings = LangchainEmbedding(langchain_embeddings=embeddings)

# 設定の更新
Settings.llm = llama_index_chain 
Settings.embed_model = llama_index_embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC # ドキュメントの読み込みとチャンク化
# MAGIC まず、単純なデフォルトのチャンキング戦略を使用して、サンプルドキュメントを読み込みます

# COMMAND ----------

vol_path = f'/Volumes/{db_catalog}/{db_schema}/{db_volume}/'

# ファイルがあることを確認
os.listdir(vol_path)

# COMMAND ----------

from llama_index.core import (
  SimpleDirectoryReader, VectorStoreIndex, Response   
)

# SimpleDirectoryReaderを使ってvol_pathからドキュメントを読み込む
reader = SimpleDirectoryReader(vol_path)
documents = reader.load_data()

# COMMAND ----------

# ここでは、シンプルなインメモリのベクトルストアをセットアップします
index = VectorStoreIndex.from_documents(documents)

# ベクトルストアをクエリーエンジンに変換します
query_engine = index.as_query_engine()

# 全てが動作していることを確認しましょう
reply = query_engine.query('what is a neural network?')

print(reply.response)

# COMMAND ----------

# MAGIC %md
# MAGIC # 評価用の質問を作成する
# MAGIC 評価を実行するためには、モデルに供給するための実用的な質問が必要です \
# MAGIC 質問を手動で作成するのは時間がかかるため、LLMを使用します \
# MAGIC ただし、生成される質問の種類には制限があることに注意してください

# COMMAND ----------

from llama_index.core.evaluation import DatasetGenerator

data_generator = DatasetGenerator.from_documents(documents)

# これは質問を生成するための呼び出しです
# 数値を設定するとマルチスレッドで高速に動作します
eval_questions = data_generator.generate_questions_from_nodes(num=64)
eval_questions

# これらの質問のいくつかはあまり有用なものでは無いかもしれません。それは、生成に使用しているモデルのせいかもしれません
# あるいは、適切にチャンクが作成されていないことによる場合があります

# COMMAND ----------

# ラボ環境で実行する際、クラスの先頭で事前生成して、リロードのために格納しておくこともできます
#question_frame = spark.sql(f"SELECT * FROM {db_catalog}.{db_schema}.evaluation_questions").toPandas()
question_frame = pd.DataFrame(eval_questions, columns=["eval_questions"])
dataframe = spark.createDataFrame(question_frame)

dataframe.write.mode("overwrite").saveAsTable(f"{db_catalog}.{db_schema}.evaluation_questions")
display(dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC # 質問を使用して評価を生成する
# MAGIC
# MAGIC 質問を生成したので、これらを用いてレスポンスを取得することができます。
# MAGIC
# MAGIC 次のステップは遅い場合があるため、ここでは1つの質問のみを使用します。その後、`ResponseEvaluator`を使用して、クエリが応答で回答されているかどうかを確認できます

# COMMAND ----------

import pandas as pd
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import EvaluationResult

#eval_questions = eval_questions[0:20]
eval_question = eval_questions[0]

# はい、ここではLLMの評価にLLMを使用します
## これを行う際には、あなたの入力の品質を評価するために通常は、よりパワフルでより高価な評価器を使いたいと考えるでしょう
evaluator = RelevancyEvaluator(llm=llama_index_chain)

# jupyter display関数を定義
def display_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)

# COMMAND ----------

# クエリ文字列を定義
#query_str = (
#    "What is the best approach to finetuning llms?"
#)

# クエリエンジンを初期化
query_engine = index.as_query_engine()
# クエリエンジンを使用してクエリを実行し、レスポンスベクトルを取得
response_vector = query_engine.query(eval_question)
# 評価器を使用してレスポンスを評価
eval_result = evaluator.evaluate_response(
    query=eval_question, response=response_vector
)

# COMMAND ----------

display_eval_df(eval_question, response_vector, eval_result)

# COMMAND ----------


