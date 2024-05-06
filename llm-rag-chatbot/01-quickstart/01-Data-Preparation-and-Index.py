# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ LLM Chatbot RAGのデータ準備
# MAGIC
# MAGIC ## ナレッジベースを構築し、Databricks Vector Searchのドキュメントを準備する
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-1.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC このノートブックでは、チャットボットがより良い回答をできるように、ドキュメントのページを取り込みんで、Vector Search Indexでインデックスを作成します。
# MAGIC
# MAGIC みなさまのチャットbotのパフォーマンスにおいては、高品質なデータの準備がキーとなります。ご自身のデータセットを用いてこれを実装することに時間を費やすことをお勧めします。
# MAGIC
# MAGIC 嬉しいことに、Lakehouse AIは皆様のAIとLLMプロジェクトを加速する最先端のソリューションを提供し、大規模なデータの取り込みと準備をシンプルにします。
# MAGIC
# MAGIC この例では、[docs.databricks.com](docs.databricks.com)のDatabricksドキュメントを使用します:
# MAGIC - Webページのダウンロード
# MAGIC - 小規模なチャンクにページを分割
# MAGIC - HTMLコンテンツからテキストを抽出
# MAGIC - DeltaテーブルをベースとしたVector Search Indexを作成
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F01-Data-Preparation-and-Index&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F01-Data-Preparation-and-Index&version=1">

# COMMAND ----------

# DBTITLE 1,必要となる外部ライブラリのインストール
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,リソースとカタログの初期化
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricksドキュメントのサイトマップとページの抽出
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC 最初に、Deltaテーブルとして生のデータセットを作成しましょう。
# MAGIC
# MAGIC このデモでは、`docs.databricks.com`のドキュメントページからいくつかのドキュメントを直接ダウンロードし、HTMLコンテンツとして保存します。
# MAGIC
# MAGIC こちらが主要なステップです:
# MAGIC
# MAGIC - `sitemap.xml`ファイルからページのURLを抽出するためのクイックスクリプトを実行します
# MAGIC - Webページをダウンロードします
# MAGIC - 文書本体を抽出するためにBeautifulSoupを活用します
# MAGIC - Delta Lakeテーブルに結果を保存します

# COMMAND ----------

if not table_exists("raw_documentation") or spark.table("raw_documentation").isEmpty():
    # Databricksのドキュメントをデータフレームにダウンロード (詳細は _resources/00-init をご覧ください)
    doc_articles = download_databricks_documentation_articles()
    # raw_documentationテーブルに保存
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation")

display(spark.table("raw_documentation").limit(2))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### ドキュメントページを小規模なチャンクに分割
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC 通常、LLMモデルはに最大の入力ウィンドウサイズがあり、非常に長いテキストのエンべディングを計算することができません。さらに、コンテキストが大きくなるほど、モデルがレスポンスを生成するのに必要とする時間が長くなります。
# MAGIC
# MAGIC ドキュメントの準備はモデルが適切に動作するためには重要であり、皆様のデータセットに応じて複数の戦略が存在します:
# MAGIC
# MAGIC - 小規模なチャンク(パラグラフ, h2...)にドキュメントを分割します
# MAGIC - ドキュメントを固定長に切り取ります
# MAGIC - チャンクサイズは皆様のコンテンツとプロンプトを作成するためにどのように活用するのかに依存します。プロンプトに複数の小規模なドキュメントチャンクを追加することで、大規模なチャンクを送信するのとは異なる結果を生成します。
# MAGIC - 高速なライブ推論のために、大規模なチャンクに分割し、ワンオフのジョブとしてそれぞれのチャンクの要約するようにモデルに依頼します。
# MAGIC - それぞれの大規模なドキュメントを並列で評価するために複数のエージェントを作成し、回答を生成するために最終的なエージェントに依頼します...

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 大規模なドキュメントページを小規模なチャンクに分割 (h2セクション)
# MAGIC
# MAGIC このデモでは、我々のモデルには大きすぎる大規模なドキュメントがいくつか存在します。これらの文書をHTMLの`h2`タグの間で分割し、それぞれのチャンクが4000トークン以上にならないようにします。
# MAGIC
# MAGIC また、モデルに平文テキストを送信するようにHTMLタグも除外します。
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC <br/>
# MAGIC このデモでは、我々のモデルには大きすぎる大規模なドキュメントがいくつか存在します。
# MAGIC
# MAGIC 最大の入力サイズを超えてしまうため、複数のドキュメントをRAGのコンテキストとしては使用できません。いくつかの最近の研究でも、LLMはあなたのプロンプトの最初と最後にフォーカスする傾向があるため、大きなウィンドウサイズが常に優れているとは限らないと述べています。
# MAGIC
# MAGIC 我々のケースでは、LangChainを用いて、HTMLの`h2`タグの間の文で分割を行ってHTMLを除外し、それぞれのチャンクが500トークンに収まるようにします。
# MAGIC
# MAGIC ### LLMウィンドウサイズとトークナイザー
# MAGIC
# MAGIC 同じセンテンスであっても、モデルが変わると返却されるトークンも異なります。LLMは、指定されたセンテンスに対していくつかのトークンが生成されるのかをカウント(通常は単語数以上のものです)するために活用できる`Tokenizer`も提供しています。([Hugging Faceドキュメント](https://huggingface.co/docs/transformers/main/tokenizer_summary) や [OpenAI](https://github.com/openai/tiktoken)をご覧ください)
# MAGIC
# MAGIC 使用するトークナイザーとコンテキストサイズの上限がエンべディングモデルとマッチするようにしてください。Databricks DBRX InstructはGPT4と同じトークナイザーを使用します。DBRX Instructのトークンとそのトークナイザーを考慮するように`transformers`ライブラリを使用います。また、これによって、ドキュメントのトークンサイズをエンべディングの最大サイズ(1024)に収めるようにします。
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>お知らせ</strong><br/>
# MAGIC   以降のステップはお使いのデータセット固有になることを覚えておいてください。これは、適切なRAGアシスタントの構築において重要な部分となります。
# MAGIC   <br/> 作成されたチャンクを手動で確認し、それらが適切な情報を含んでいることを確認するために、常に時間を費やすようにしてください。
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,HTMLページを小規模なチャンクに分割
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2で分割しますが、あまり小さすぎないように小さなh2チャンクはマージします
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # チャンクを結合し、h2の前にテキストを追加することでチャンクを結合し、小さすぎる文書を回避します
  for c in h2_chunks:
    # h2の結合 (注意: 重複したh2を回避するために以前のチャンクを削除することもできます)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # 小さすぎるチャンクの破棄
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 
# チャンク処理関数を試しましょう
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### チャンクを作成し、Deltaテーブルに保存
# MAGIC
# MAGIC 最後のステップは、ドキュメントのすべてのテキストにUDFを適用し、`databricks_documentation`テーブルに保存することです。
# MAGIC
# MAGIC *このパートは多くの場合、新規ドキュメントのページが更新されるとすぐに実行されるプロダクションレベルのジョブであるケースが多いことに注意してください。
# MAGIC <br/>これは、更新をインクリメンタルに処理するためのDelta Live Tableパイプラインである場合があります。*

# COMMAND ----------

# DBTITLE 1,databricks_documentationを格納する最終的なテーブルの作成
# MAGIC %sql
# MAGIC --インデックスを作成するためには、テーブルでチェンジデータフィードを有効化する必要があることに注意してください
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# sparkですべてのドキュメントのチャンクを作成するためのユーザー定義関数(UDF)を作成しましょう
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("raw_documentation")
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Vector Search Indexに必要なもの
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-managed-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricksは複数のタイプのVector Search Indexを提供します:
# MAGIC
# MAGIC - **マネージドのエンべディング**: テキストのカラムとエンドポイント名を指定すると、DatabricksはDeltaテーブルとインデックスを同期します  **(このデモではこちらを使います)**
# MAGIC - **セルフマネージドのエンべディング**: エンべディングをご自身で計算し、Deltaテーブルのフィールドとして保存します。Databricksはインデックスと同期を行います。
# MAGIC - **ダイレクトインデックス**: Deltaテーブルを使うことなしにインデックスを更新したい場合に使用します。
# MAGIC
# MAGIC このデモでは、**マネージドのエンべディング**のインデックスのセットアップ方法を説明します。 *(セルフマネージドのエンべディングはadvancedデモでカバーされています)*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricks BGE Embeddings Foundation Modelエンドポイントのご紹介
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC 基盤モデルはDatabricksによって提供され、すぐに利用することができます。
# MAGIC
# MAGIC Databricksでは、エンべディングを計算したり、モデルを評価するためのいくつかのエンドポイントタイプがサポートされています:
# MAGIC - **基盤モデルエンドポイント**はDatabricksによって提供されます(例: llama2-70B, MPT, BGE)。
# MAGIC - **外部エンドポイント**は外部モデルへのゲートウェイとして動作します(例: Azure OpenAI)。**このデモではこちらを使います。**
# MAGIC - **カスタム**はDatabricksのモデルサービスにホスティングされるファインチューンされたモデルです。
# MAGIC
# MAGIC 基盤モデルを探索、トライするために[モデルサービングエンドポイントページ](/ml/endpoints)を開きましょう。
# MAGIC
# MAGIC このデモでは、`text-embedding-ada-002` (embeddings) と `llama2-70B` (chat)を使います。 <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

# DBTITLE 1,エンべディングとは何か
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# エンべディングエンドポイントはテキストをベクトル(floatの配列)に変換します。ここでは text-embedding-ada-002 を使います:
response = deploy_client.predict(endpoint="taka-text-embedding-ada-002", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### マネージドエンべディングを用いたVector Search Indexの作成
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC マネージドエンべディングを用いることで、Databricksが自動でエンべディングを計算します。Databricksを使い始めるにはより簡単なモードとなっています。
# MAGIC
# MAGIC Vector Search Indexはエンべディングを提供するために、**Vector search endpoint**を使用します(あなたのVector Search APIエンドポイントと考えることができます)。
# MAGIC
# MAGIC 複数のインデックスが同じエンドポイントを使うことができます。
# MAGIC
# MAGIC 一つ作ってみましょう。

# COMMAND ----------

# DBTITLE 1,Vector Searchエンドポイントの作成
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
#    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

#wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
#print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Vector Search Indexの作成
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/index_creation.gif?raw=true" width="600px" style="float: right">
# MAGIC
# MAGIC [Vector SearchエンドポイントのUI](#/setting/clusters/vector-search)からエンドポイントを確認することができます。エンドポイントによって提供されるすべてのインデックスを確認するためにエンドポイント名をくりっっくします。
# MAGIC
# MAGIC それでは、Databricksにインデックスの作成を指示しましょう。
# MAGIC
# MAGIC マネージドエンべディングインデックスなので、テキストのカラムとエンべディング基盤モデルを指定するだけです。Databricksが自動でエンべディングを計算します。
# MAGIC
# MAGIC これは、APIやUnity Catalogのカタログエクスプローラからの数クリックで行うことができます。

# COMMAND ----------

# DBTITLE 1,エンドポイントを用いたマネージドvector searchの作成
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# インデックスを作成したいテーブル
source_table_fullname = f"{catalog}.{db}.databricks_documentation"
# インデックスの格納場所
vs_index_fullname = f"{catalog}.{db}.databricks_documentation_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', # テキストを格納しているカラム
    embedding_model_endpoint_name='taka-text-embedding-ada-002' # エンべディング作成に用いるエンべディングエンドポイント
  )
  # すべてのエンべディングが作成され、インデックスが作成されるのを待ちます
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  # Vector Searchのコンテンツを更新し、新規データを保存するように同期処理を起動
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 類似コンテンツの検索
# MAGIC
# MAGIC やらなくてはならないことはこれで全てです。DatabricksはDelta Live Tablesを用いて、新規のエントリーを自動で捕捉し同期します。
# MAGIC
# MAGIC データセットのサイズやモデルのサイズに応じて、エンべディングの作成とインデックス作成に数秒を要することに注意してください。
# MAGIC
# MAGIC 類似コンテンツの検索を試してみましょう。
# MAGIC
# MAGIC *注意: `similarity_search`は、filtersパラメーターもサポートしています。これはRAGシステムにセキュリティレイヤーを追加するのに有用です: 誰が呼び出しを行なっているのかに基づいて、センシティブなコンテンツを除外することができます(例えば、ユーザーの設定に基づき特定の部署にフィルタリングするなど)。*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#question = "How can I track billing usage on my workspaces?"
question = "Databricksクラスターとは？"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 次のステップ: DBRXを用いたRAGチャットボットモデルのデプロイ
# MAGIC
# MAGIC DatabricksのLakehouse AIによって、容易に文書を取り込んで準備することができ、数行のコードと設定だけでそれらをベースとしたVector Search Indexをデプロイできることを見てきました。
# MAGIC
# MAGIC これは、皆様が次のステップにフォーカスできるように、データプロジェクトをシンプルかつ加速します: 次のステップとは、適切に作成されたプロンプト拡張によるリアルタイムチャットボットエンドポイントの作成です。
# MAGIC
# MAGIC チャットbotのエンドポイントを作成するために、[02-Deploy-RAG-Chatbot-Model]($./02-Deploy-RAG-Chatbot-Model)ノートブックを開きましょう。
