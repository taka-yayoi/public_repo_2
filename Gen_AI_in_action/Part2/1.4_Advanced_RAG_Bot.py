# Databricks notebook source
# MAGIC %md
# MAGIC # 高度なRAGシステムの構築
# MAGIC
# MAGIC 複数のファイルといくつかのより複雑なロジックを持つ高度なRAGシステムを構築します
# MAGIC
# MAGIC ここでは、インストールと実行速度を速めるために「Llama_index」と「Unstructured」をスキップします

# COMMAND ----------

# DBTITLE 1,追加ライブラリのインストール
# MAGIC %pip install -U pymupdf typing_extensions sqlalchemy>=2.0.25 langchain==0.1.16 databricks-vectorsearch==0.23 flashrank mlflow==2.12.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,セットアップユーティリティ
# MAGIC %run ./utils

# COMMAND ----------

# 設定を上書き
vector_search_endpoint = 'dbdemos_vs_endpoint'
db_catalog = 'users'
db_schema = 'takaaki_yayoi'

# COMMAND ----------

# MAGIC %md
# MAGIC # Vector Storeとインデックスの構築

# COMMAND ----------

# DBTITLE 1,環境とエンべディングのセットアップ
# 少し興味深いものにして、コストを削減するためにローカルのエンべディングモデルを活用します
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain.schema import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts.prompt import PromptTemplate

# メッセージ履歴を持つように、履歴を認識するリトリーバーを必要とします
# 基本的には質問と履歴を受け取ります
# - 次に再計算するようにLLMに指示します
# - そして、リトリーバーに更新されたLLM生成の質問を送信します
from langchain.chains import create_history_aware_retriever

chat_model = 'databricks-dbrx-instruct'
embedding_model_name = 'databricks-bge-large-en'
index_name = 'arxiv_parse_bge_index'

vsc = VectorSearchClient()
vs_index_fullname = f'{db_catalog}.{db_schema}.{index_name}'

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)
embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

# インデックスがない場合には、それを検知してエラーにすべきです

# COMMAND ----------

# ロジックのセットアップ

# vector searchの設定
index = vsc.get_index(endpoint_name=vector_search_endpoint,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever(search_kwargs={"k": 10})

# リランクモジュール
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# コンテキストのフォーマット
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

## 履歴の追加
# リトリーバーに入力する前に(存在する場合には)chat_historyのコンテキストを用いて入力された質問を再計算します
contextualize_q_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="<s> [INST] Your job is to reformulate a question given a user question and the prior conversational history. DO NOT answer the question. If there is no chat history pass through the question [/INST] </s> \n [INST] Question: {input} \nHistory: {chat_history} \nAnswer: [/INST]"
)

history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

rag_prompt = PromptTemplate(input_variables=['context', 'input', 'chat_history'],
                                      template="<s> [INST] You are a helpful personal assistant who helps users find what they need from documents. Be conversational, polite and use the following pieces of retrieved context and the conversational history to help answer the question. <unbreakable-instruction> ANSWER ONLY FROM THE CONTEXT </unbreakable-instruction> <unbreakable-instruction> If you don't know the answer, just say that you don't know. </unbreakable-instruction> Keep the answer concise. [/INST] </s> \n[INST] Question: {input} \nContext: {context} \nHistory: {chat_history} \nAnswer: [/INST]")


chain = (
    {'context': history_aware_retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | rag_prompt
    | llm 
    | StrOutputParser()
)

# COMMAND ----------

# エンドポイントのテスト
chain.invoke({'input': 'tell me about llms', 'chat_history': ''})

# チャット履歴を追加するには、`AiMessage` と `HumanMessage` エントリーを変更してリストオブジェクトを含める必要があります

# COMMAND ----------

# MAGIC %md # プロダクション化
# MAGIC
# MAGIC Langchainとllama_indexは常に変化しており、統合が壊れることがよくありますので、追加の柔軟性のために独自のラッパーを作成することをお勧めします。\
# MAGIC 詳細については、[MLflow Pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)を参照してください。

# COMMAND ----------

import mlflow

class AdvancedLangchainBot(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        Databricksモデルサービングでモデルのインスタンスが作成される際、
        この関数が最初に実行されます。

        すべてのLangchainのコンポーネントがシリアライズされるわけでは無いので、
        チェーン全体のインスタンスを作成するには、この関数を使う必要があります

        以下は単に上のセルの内容を貼り付けたものです
        """

        from langchain_community.chat_models import ChatDatabricks
        from langchain_community.embeddings import DatabricksEmbeddings
        from databricks.vector_search.client import VectorSearchClient
        from langchain_community.vectorstores import DatabricksVectorSearch
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import FlashrankRerank

        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        from langchain_core.prompts.prompt import PromptTemplate

        from langchain.chains import create_history_aware_retriever

        chat_model = 'databricks-dbrx-instruct'
        embedding_model_name = 'databricks-bge-large-en'

        vsc = VectorSearchClient()
        vs_index_fullname = f'{db_catalog}.{db_schema}.{index_name}'

        llm = ChatDatabricks(
            target_uri="databricks",
            endpoint=chat_model,
            temperature=0.1,
        )

        embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

        index = vsc.get_index(endpoint_name=vector_search_endpoint,
                      index_name=vs_index_fullname)

        retriever = DatabricksVectorSearch(
            index, text_column="page_content", 
            embedding=embeddings, columns=["source_doc"]
        ).as_retriever(search_kwargs={"k": 10})

        # リランクモジュール
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        # コンテキストの整形
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        ## 履歴の追加
        # リトリーバーに入力する前に(存在する場合には)chat_historyのコンテキストを用いて入力された質問を再計算します
        contextualize_q_prompt = PromptTemplate(
            input_variables=["input", "chat_history"],
            template="<s> [INST] Your job is to reformulate a question given a user question and the prior conversational history. DO NOT answer the question. If there is no chat history pass through the question [/INST] </s> \n [INST] Question: {input} \nHistory: {chat_history} \nAnswer: [/INST]"
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, compression_retriever, contextualize_q_prompt
        )

        rag_prompt = PromptTemplate(input_variables=['context', 'input', 'chat_history'],
                                              template="<s> [INST] You are a helpful personal assistant who helps users find what they need from documents. Be conversational, polite and use the following pieces of retrieved context and the conversational history to help answer the question. <unbreakable-instruction> ANSWER ONLY FROM THE CONTEXT </unbreakable-instruction> <unbreakable-instruction> If you don't know the answer, just say that you don't know. </unbreakable-instruction> Keep the answer concise. [/INST] </s> \n[INST] Question: {input} \nContext: {context} \nHistory: {chat_history} \nAnswer: [/INST]")


        # predict関数で使用できるように、ここでは self.chain を使います
        self.chain = (
            {'context': history_aware_retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | rag_prompt
            | llm 
            | StrOutputParser()
        )

    def process_row(self, row):
       return self.chain.invoke({'input': row['input'],
                                 'chat_history': row['chat_history']})
    
    def predict(self, context, data):
        """
        これはもう一つの重要な関数であり、入力を処理しチェーンに送信します
        """
        results = data.apply(self.process_row, axis=1) 

        # Databricksを使う場合には .content を削除します
        results_text = results.apply(lambda x: x)
        return results_text 


# COMMAND ----------

# まず初めに、ラッパーの動作確認を行います
import pandas as pd

sample_input = 'Tell me about how good ChatGPT is across various tasks in a Zero shot Prompting paradigm?'

mlflow_pyfunc_model = AdvancedLangchainBot()
mlflow_pyfunc_model.load_context(context='')

# TODO verify if the pandas gets done by Model Serving when deploy ie we just send json?
response = mlflow_pyfunc_model.predict(
  data=pd.DataFrame.from_records({'input': [sample_input], 'chat_history': [[]]}),
  context='')
response.iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC もし動作するなら、評価のためのサンプルと共にmlflowにモデルを記録することができます \
# MAGIC 注記 - エンドポイントとしてデプロイする場合、モデルはDatabricksのモデルにアクセスし、自身を認証するために2つの環境変数 `DATABRICKS_HOST` と `DATABRICKS_TOKEN` を設定する必要があります

# COMMAND ----------

# 以前と同じデータセットを使っているので、ノートブック 0.4の質問を再利用します

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

data = {'input': [[x] for x in eval_questions],
        'chat_history': [[[]] for x in eval_questions]}


sample_questions = pd.DataFrame(data)
sample_questions

# COMMAND ----------

def eval_pipe(inputs):
    print(inputs)
    answers = []
    for index, row in inputs.iterrows():
        #answer = {'answer': 'test'}
        #print(row)
        dict_obj = {"chat_history": row['input'], 
                    "input": row['chat_history']}
        answer = chain.invoke(dict_obj)
        
        answers.append(answer) #['answer'])
    
    return answers

# COMMAND ----------

experiment_name = 'workshop_rag_evaluations'

username = spark.sql("SELECT current_user()").first()['current_user()']
mlflow_dir = f'/Users/{username}/{experiment_name}'
mlflow.set_experiment(mlflow_dir)

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run(run_name='advanced_rag'):
  
    model = AdvancedLangchainBot()

    example_input = 'Tell me about how good ChatGPT is across various tasks in a Zero shot Prompting paradigm?'
    input_json = {'input': [example_input,example_input], 
                  'chat_history': [
                        [{'role':'user', 'content': 'Hello'},
                         {'role':'assistant', 'content': 'Hello'}],
                        None
                    ]}

    langchain_signature = mlflow.models.infer_signature(
        model_input=input_json,
        model_output=[response.iloc[0]]
    )

    mlflow_result = mlflow.pyfunc.log_model(
        python_model = model,
        extra_pip_requirements = ['langchain==0.1.16', 
                                'sqlalchemy==2.0.29', 
                                'mlflow==2.12.2', 
                                'databricks-vectorsearch==0.23', 
                                'flashrank==0.2.0'],
        artifact_path = 'langchain_pyfunc',
        signature = langchain_signature,
        input_example = input_json,
        registered_model_name = f'{db_catalog}.{db_schema}.adv_langchain_model'
    )

    # TODO Fix the evals potentially by just using the chain from above?
    eval_results = mlflow.evaluate(eval_pipe, 
                          data=sample_questions, 
                          model_type='text')

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントへのリクエストの適切な送信方法は以下の通りです:
# MAGIC ```
# MAGIC {
# MAGIC "input": ["What is the main idea behind Halting in random walk kernels as discussed in reference [351]?"],
# MAGIC "chat_history": [[{"role": "user", "content": "I like beef"}]]
# MAGIC }
# MAGIC ````
# MAGIC
# MAGIC エンドポイントにクエリーを行うために、Pythonのrequestsを使うことができます。
