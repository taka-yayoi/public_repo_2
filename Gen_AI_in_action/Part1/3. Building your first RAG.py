# Databricks notebook source
# MAGIC %md # 初めてのRAG構築
# MAGIC
# MAGIC これは、あなたの初めてのRAGアプリケーションを作成するためのノートブックです。

# COMMAND ----------

# MAGIC %pip install -U langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow==2.11.0 databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,パラメーターのセットアップ
db_catalog = 'users'
db_schema = 'takaaki_yayoi'
db_table = "arxiv_parse"

endpoint_name = "dbdemos_vs_endpoint"
vs_index = f"{db_table}_bge_index"
vs_index_fullname = f"{db_catalog}.{db_schema}.{vs_index}"

# temp need to change later
embedding_model = "databricks-bge-large-en"
chat_model = "databricks-mixtral-8x7b-instruct"

# COMMAND ----------

# LLMチェーンの構築とテスト
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import DatabricksEmbeddings

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# これはDB FM API向けのもので、Provisioned Throughput向けではありません
# 現時点ではFM APIは特定のリージョンで利用できます
chat = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)

# 動作確認
chat([HumanMessage(content="hello")])

# COMMAND ----------

# エンべディングモデルの構築とテスト
embeddings = DatabricksEmbeddings(endpoint=embedding_model)
embeddings.embed_query("hello")[:3]

# COMMAND ----------

# MAGIC %md ## チェーンのセットアップ

# COMMAND ----------

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=endpoint_name,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a response to answer this question")
    ])

retriever_chain = create_history_aware_retriever(chat, retriever, retriever_prompt)

# COMMAND ----------

prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         You are a research and eduation assistant that takes a conversation between a learner and yourself and answer their questions based on the below context:
         
         <context>
         {context}
         </context> 
         
         <instructions>
         - Focus your answers based on the context but provide additional helpful notes from your background knowledge caveat those notes though.
         - If the context does not seem relevant to the answer say as such as well.
         </instructions>
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

stuff_documents_chain = create_stuff_documents_chain(chat, prompt_template)

retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

# COMMAND ----------

# チェーンのテスト
retrieval_chain.invoke({"chat_history": [], 
                              "input": "Tell me about tuning LLMs", 
                              "context": ""})

# COMMAND ----------

# MAGIC %md ## MLflowのロギング
# MAGIC
# MAGIC チェーンに対する評価を実行し、MLflowに結果を記録することができます。

# COMMAND ----------

import mlflow
import pandas as pd

experiment_name = 'workshop_rag_evaluations'

username = spark.sql("SELECT current_user()").first()['current_user()']
mlflow_dir = f'/Users/{username}/experiment_name'
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

eval_questions = [
    "Can you describe the process of Asymmetric transitivity preserving graph embedding as mentioned in reference [350]?",
    "What is the main idea behind Halting in random walk kernels as discussed in reference [351]?",
    "What is the title of the paper authored by Ledig et al. in CVPR, as mentioned in the context information?",
    'Who are the authors of the paper "Invertible conditional gans for image editing"?',
    'In which conference was the paper "Generating videos with scene dynamics" presented?',
    'What is the name of the algorithm developed by Tulyakov et al. for video generation?',
    'What is the main contribution of the paper "Unsupervised learning of visual representations using videos" by Wang and Gupta?',
    'What is the title of the paper authored by Wei et al. in CVPR, as mentioned in the context information?',
    'What is the name of the algorithm developed by Ahsan et al. for video action recognition?',
    'What is the main contribution of the paper "Learning features by watching objects move" by Pathak et al.?'
]

sample_questions = pd.DataFrame(
    eval_questions, columns = ['eval_questions']
)

# if we have these saved out we can do this
#sample_questions = spark.sql(f"SELECT * FROM {db_catalog}.{db_schema}.evaluation_questions").toPandas()

# COMMAND ----------

def eval_pipe(inputs):
    answers = []
    for index, row in inputs.iterrows():
        answer = retrieval_chain.invoke({"chat_history": [], 
                              "input": row.item(), 
                              "context": ""})
        
        # pipe([HumanMessage(content=prompt)], max_tokens=100)
        # result = pipe( [HumanMessage(content=row.item())], max_tokens=100)
        # answer = result.content
        answers.append(answer['answer'])
    
    return answers

# COMMAND ----------

with mlflow.start_run(run_name='basic_rag'):
  results = mlflow.evaluate(eval_pipe, 
                          data=sample_questions[0:30], 
                          model_type='text')

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、評価結果とRAGがうまく動いているかどうかを確認できるようになります。
