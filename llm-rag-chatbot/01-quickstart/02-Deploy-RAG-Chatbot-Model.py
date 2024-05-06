# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 2/ Retrieval Augmented Generation (RAG)とDBRX Instructによるチャットbotの作成
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC Vector Search Indexの準備ができました！
# MAGIC
# MAGIC RAGを実行するための新たなモデルサービングエンドポイントを作成して、デプロイしましょう。
# MAGIC
# MAGIC 流れは以下の通りとなります:
# MAGIC
# MAGIC - ユーザーが質問します
# MAGIC - 質問がサーバレスChatbot RAGエンドポイントに送信されます
# MAGIC - エンドポイントがエンべディングを計算し、Vector Search Indexを活用して質問に類似した文書を検索します。
# MAGIC - エンドポイントは、文書で補強されたプロンプトを生成します
# MAGIC - プロンプトはDBRX Instruct Foundation Modelサービングエンドポイントに送信されます
# MAGIC - ユーザーにアウトプットを出力します！
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC *注意: RAGはDatabricks Vector Searchを用いて文書を検索します。このノートブックでは、search indexの準備ができていることを前提としています。以前の[01-Data-Preparation-and-Index]($./01-Data-Preparation-and-Index)ノートブックを必ず実行してください。*

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC   
# MAGIC ### このデモを動作させるにはシークレットが必要です:
# MAGIC
# MAGIC 使用するモデルサービングエンドポイントには、あなたのVector Search Indexに対して認証を行うためのシークレットが必要です。([ドキュメント](https://docs.databricks.com/ja/security/secrets/secrets.html)をご覧ください)<br/>
# MAGIC **共有のデモワークスペースを使用している場合、シークレットがセットアップされている場合があるので、以下のステップを実行して値を上書きしないようにしてください。**<br/>
# MAGIC
# MAGIC - ラップトップあるいはクラスターのターミナルで[Databricks CLIをセットアップ](https://docs.databricks.com/en/dev-tools/cli/install.html)する必要があります: <br/>
# MAGIC `pip install databricks-cli` <br/>
# MAGIC - CLIを設定します。お使いのワークスペースのURLや自分のプロフィールページからPATトークンが必要となります。<br>
# MAGIC `databricks configure`
# MAGIC - シークレットのスコープを作成します:<br/>
# MAGIC `databricks secrets create-scope dbdemos`
# MAGIC - サービスプリンシパルのシークレットを保存します。モデルのエンドポイントが認証を行うために使用されます。これがデモやテストの場合、ご自身の[PATトークン](https://docs.databricks.com/ja/dev-tools/auth/pat.html)を使うことができます。<br>
# MAGIC `databricks secrets put-secret dbdemos rag_sp_token`
# MAGIC
# MAGIC *Note: お使いのサービスプリンシパルがVector Search indexへのアクセス権を持つことを確認してください:*
# MAGIC
# MAGIC ```
# MAGIC spark.sql('GRANT USAGE ON CATALOG <catalog> TO `<YOUR_SP>`');
# MAGIC spark.sql('GRANT USAGE ON DATABASE <catalog>.<db> TO `<YOUR_SP>`');
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import databricks.sdk.service.catalog as c
# MAGIC WorkspaceClient().grants.update(c.SecurableType.TABLE, <index_name>, 
# MAGIC                                 changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="<YOUR_SP>")])
# MAGIC   ```

# COMMAND ----------

# DBTITLE 1,お使いのSPがご自身のVector Search Indexに読み取り権限があることを確認してください
index_name=f"{catalog}.{db}.databricks_documentation_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

test_demo_permissions(host, secret_scope="demo-token-takaaki.yayoi", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="taka-text-embedding-ada-002")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Langchain retriever
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Langchain retrieverの構築を始めましょう。 
# MAGIC
# MAGIC これは以下のことに責任を持ちます:
# MAGIC
# MAGIC * 入力の質問を作成します (我々のマネージドVector Search Indexがエンべディングの計算を行なってくれます)
# MAGIC * プロンプトを拡張するための類似文書を検索するためにvector search indexを呼び出します 
# MAGIC
# MAGIC Databricks Langchainラッパーは、1ステップでこれを行うことを簡単にし、背後のロジックやAPIの呼び出しの面倒を見てくれます。

# COMMAND ----------

# DBTITLE 1,モデルの認証のセットアップ
# サーバレスエンドポイントからモデルにリクエストを送るために使用されるURL
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("demo-token-takaaki.yayoi", "rag_sp_token")

# COMMAND ----------

# DBTITLE 1,Databricks Embedding Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# エンべディングLangchainモデルのテスト
# 注意: お使いの質問のエンべディングモデルは以前のノートブックのチャンクで用いられたものとマッチしなくてはなりません
embedding_model = DatabricksEmbeddings(endpoint="taka-text-embedding-ada-002")
print(f"Test embeddings: {embedding_model.embed_query('Sparkとは?')[:20]}...")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    # vector search indexの取得
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # retrieverの作成
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# retrieverのテスト
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Databricksクラスターとは？")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Databricks DBRX Instruct基盤モデルに問い合わせを行うDatabricks Chatモデルの構築
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-3.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC 我々のチャットボットは回答を提供するために、Databricks DBRX Instruct基盤モデルを使用します。DBRX Instructは、企業レベルのGenAIアプリケーションの開発のために構築された汎用LLMであり、これまではクローズドモデルのAPIに限定されていた機能で、皆様のユースケースを解放します。
# MAGIC
# MAGIC 我々の計測によれば、DBRXはGPT-3.5を上回っており、Gemini 1.0 Proと並ぶものとなっています。汎用LLMとしての強みに加えて、優れたコードモデルであり、プログラミングに特化したCodeLLaMA-70Bのような特化モデルと比類するものとなっています。
# MAGIC
# MAGIC *注意: 複数タイプのエンドポイントやlangchainモデルを活用することができます:*
# MAGIC
# MAGIC - Databricks基盤モデル **(こちらを使います)**
# MAGIC - ご自身でファインチューンしたモデル
# MAGIC - 外部のモデルプロバイダー (Azure OpenAIなど)

# COMMAND ----------

# Databricks Foundation LLMモデルのテスト
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)
print(f"Test chat model: {chat_model.predict('Apache Sparkとは？日本語で教えて')}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 完全なRAGチェーンの構築
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC 単一のLangchainのチェーンにretrieverとモデルをまとめましょう。
# MAGIC
# MAGIC アシスタントが適切な回答を提供できるように、カスタムのlangchainテンプレートを使用します。
# MAGIC
# MAGIC ご自身の要件に基づき、様々なテンプレートを試し、アシスタントの口調や個性を調整するのにある程度の時間を取るようにしてください。

# COMMAND ----------

# DBTITLE 1,Databricksアシスタントのチェーン
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """私はDatabricksユーザーのためのアシスタントです。Python、コーディング、SQL、データエンジニアリング、Spark、データサイエンス、DWおよびプラットフォーム、APIまたはインフラ管理に関連する質問に回答します。これらのトピックに関連しない質問の場合は、丁重に回答を辞退します。答えがわからない場合は、「わかりません」とだけ言います。回答はできるだけ簡潔な日本語にしてください。
最後に質問に回答するために以下のコンテキストのピースを使ってください:
{context}
質問: {question}
回答:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,ノートブックから直接チャットbotを試しましょう
# langchain.debug = True # チェーンの詳細や送信される完全なプロンプトを確認するにはコメントを解除してください
#question = {"query": "How can I track billing usage on my workspaces?"}
question = {"query": "ワークスペースの使用量を追跡するにはどうしたらいいですか？"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unity Catalogレジストリにモデルを保存
# MAGIC
# MAGIC モデルの準備ができたので、Unity Catalogのスキーマにモデルを登録することができます:

# COMMAND ----------

# DBTITLE 1,MLFlowにチェーンを登録
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.dbdemos_chatbot_model"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  #(認証のために)シークレットとして環境変数　DATABRICKS_TOKEN とretrieverをロード
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ### サーバレスモデルエンドポイントにChatモデルをデプロイ
# MAGIC
# MAGIC モデルがUnity Catalogに保存されたので、最後のステップはこれをモデルサービングにデプロイすることとなります。
# MAGIC
# MAGIC これによって、アシスタントのフロントエンドからリクエストを送信できるようになります。

# COMMAND ----------

# サービングエンドポイントの作成、更新
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = f"dbdemos_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/demo-token-takaaki.yayoi/rag_sp_token}}",  # アクセストークンを保持する <scope>/<secret>
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントがデプロイされました！[Serving Endpoint UI](#/mlflow/endpoints)でエンドポイントを検索し、パフォーマンスを可視化することができます！
# MAGIC
# MAGIC PythonでRESTクエリーを実行して試してみましょう。

# COMMAND ----------

# DBTITLE 1,チャットbotにクエリーを送信してみましょう
#question = "How can I track billing usage on my workspaces?"
question = "ワークスペースの使用量を追跡するにはどうしたらいいですか？"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])

# COMMAND ----------

question = "サーバレスを使うことのメリットは"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## おめでとうございます！最初のGenAI RAGモデルをデプロイしました！
# MAGIC
# MAGIC Lakehouse AIを活用することで、皆様の内部の知識に対して同じロジックをデプロイすることができる準備ができました。
# MAGIC
# MAGIC 皆様のGenAIの課題の解決にLakehouse AIがどのようにユニークに位置づけられるのかを見てきました:
# MAGIC
# MAGIC - Databricksのエンジニアリング機能でデータの取り込みと準備をシンプルに
# MAGIC - 完全にマネージドなインデックスでVector Searchのデプロイメントを加速
# MAGIC - Databricks DBRX Instruct基盤モデルエンドポイントの活用
# MAGIC - RAGを実行し、Q&A機能を提供するためのリアルタイムモデルエンドポイントのデプロイ
# MAGIC
# MAGIC Lakehouse AIは皆様のGenAIデプロイメントを加速するためのユニークなソリューションです。

# COMMAND ----------

# MAGIC %md # クリーンアップ
# MAGIC
# MAGIC リソースを解放するには、以下のセルのコメントを解除して実行してください。

# COMMAND ----------

# /!\ THIS WILL DROP YOUR DEMO SCHEMA ENTIRELY /!\ 
# cleanup_demo(catalog, db, serving_endpoint_name, f"{catalog}.{db}.databricks_documentation_vs_index")
