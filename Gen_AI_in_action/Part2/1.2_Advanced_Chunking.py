# Databricks notebook source
# MAGIC %md
# MAGIC # 高度なパースとチャンキング
# MAGIC
# MAGIC より良いクエリエンジンを構築するために、抽出とチャンキングのプロセスを改善する必要があります。
# MAGIC - 全ての情報を適切に抽出していますか？
# MAGIC - ドキュメントを適切なチャンクに分割していますか？
# MAGIC - モデルのコンテキストを適切に収め、十分な抽出を提供するためにどのサイズのチャンクが必要ですか？
# MAGIC
# MAGIC このプロセスには2つのステップがあります。パースとチャンクです。
# MAGIC - パースでは、できるだけ多くのテキストと関連するメタデータを抽出する必要があります。
# MAGIC - チャンクでは、パースした結果を受け取り、LLMのプロンプティングで活用しやすいセクションに分割します。
# MAGIC
# MAGIC デフォルトの手法はシンプルで、文字制限や単語での分割に基づいています。
# MAGIC
# MAGIC unstructuredというライブラリを活用しますが、他にも多くのオプションがあります。

# COMMAND ----------

# MAGIC %sh
# MAGIC # ここでの作業の多くのケースでpopplerが必要となります
# MAGIC apt update
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install pymupdf llama_index==0.10.25 langchain==0.1.13 llama-index-llms-langchain poppler-utils unstructured[pdf,txt]==0.13.0 databricks-vectorsearch==0.23 llama-index-embeddings-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,環境のセットアップ
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,設定
import os
from langchain_community.chat_models import ChatDatabricks
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sample_file_to_load = f'/Volumes/{db_catalog}/{db_schema}/{db_volume}/2302.06476.pdf'
print(f'We will use {sample_file_to_load} to review chunking open it alongside to see how different algorithms work')
print(f'You can access it here https://arxiv.org/pdf/2302.06476.pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC # 基本的なファイルの読み込み
# MAGIC このステージでは、基本的な pymupdf ローダーを使用します。\
# MAGIC load_and_split 関数がすべての設定を処理します。

# COMMAND ----------

loader = PyMuPDFLoader(sample_file_to_load)
docu_split = loader.load_and_split()
docu_split

# COMMAND ----------

# 最初のページのすべての内容が単一のページに結合されていることがわかります
# "reasonable performance"を検索すると、フッターもパラグラフにマージされていることを確認できます
Intro = docu_split[0].page_content
Intro

# COMMAND ----------

# ここでは、ページ1の最後のパラグラフの一部とフッターがマージされているようです
Weird_snippet = docu_split[1].page_content
Weird_snippet

# COMMAND ----------

# 表では /n のセパレーターをピックアップしているようです
# おそらく、説明文のあるテーブルを一つのチャンクとし、残りは除外したいと考えることでしょう
Table = docu_split[36].page_content
Table

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDFの手動読み込みと解析
# MAGIC
# MAGIC PDF解析プリミティブの探索 \
# MAGIC 生のPDF解析プリミティブを使用する実験が可能ですが、このプロセスは時間を要するものになることでしょう。

# COMMAND ----------

import fitz 

doc = fitz.open(sample_file_to_load)

for page in doc:
  page_dict = page.get_text("dict")
  blocks = page_dict["blocks"]
  print(blocks)
  break

# COMMAND ----------

# MAGIC %md
# MAGIC PyMuPDFの生のテキストブロックには、多くの情報が格納されていることがわかります
# MAGIC テキストの位置情報などがあります

# COMMAND ----------

# これらのオブジェクトに何が含まれているのかを見てみましょう
print(page_dict.keys())

# 幾つのブロックがあるのかを見てみましょう
print(len(page_dict['blocks']))

# ブロックに何が含まれるのかを見てみましょう
print(page_dict['blocks'])

# COMMAND ----------

# タイトル
page_dict['blocks'][0]

# COMMAND ----------

# 最初の行の著者
page_dict['blocks'][1]

# COMMAND ----------

# 2行目の著者
page_dict['blocks'][2]

# COMMAND ----------

# 画像
page_dict['blocks'][5]

# COMMAND ----------

# MAGIC %md
# MAGIC コンテキスト情報を保持し、それを活用するためには何が必要ですか？
# MAGIC ドキュメントによっては、ドキュメントの構造を解析して理解するためのカスタムロジックを記述する必要があります。
# MAGIC
# MAGIC パース手法に関する詳細情報に関しては、[PyMuPDF Docs](https://pymupdf.readthedocs.io/en/latest/tutorial.html)をご覧ください。
# MAGIC
# MAGIC 解析方法の代替案：
# MAGIC - ドキュメントスキャンモデル（例：LayoutLM）を使用する
# MAGIC - PDFからHTMLに変換してからHTMLファイルを解析する
# MAGIC   - 例：\<p>、\<h1>、\<h2>など。ただし、各PDFからHTMLへの変換ツールは少し異なる動作をするかもしれません...
# MAGIC
# MAGIC 改良されたパーサーを使用すると、次のことができます：
# MAGIC - pysparkのpandas_udfとして記述し、pdfドキュメントを標準のDeltaテーブルにパースし、それをDatabricks VectorSearchと組み合わせることができます

# COMMAND ----------

# MAGIC %md
# MAGIC # PDFの高度な解析
# MAGIC 手動のコーディングの代わりに、より新しい、より高度なパーサーを試すことができます。
# MAGIC
# MAGIC Unstructuredは1つのオプションです。OSS Unstructuredライブラリには2つの動作モードがあります。
# MAGIC - 基本的な解析では、生のPDF構造を読み取り、見出し、段落などを分析し、論理的にグループ化しようとします。
# MAGIC - OCRモードでは、データ抽出を支援するためにコンピュータビジョンモデルを適用します。
# MAGIC   - nltkが必要であり、ライブラリは事前にインストールされている必要があります。
# MAGIC   - OCR抽出には、コンピュータビジョン（Pytorch）ベースのライブラリを正しく設定する必要があります。
# MAGIC
# MAGIC インストールに関する詳細は、[Unstructured Docs](https://unstructured-io.github.io/unstructured/installation/full_installation.html)を参照してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unstructured PDF Reader
# MAGIC
# MAGIC まずは、LangChainと統合する前に、リーダーを単独で使用して抽出される内容を確認しましょう。

# COMMAND ----------

from unstructured.partition.pdf import partition_pdf
from collections import Counter

# COMMAND ----------

elements = partition_pdf(sample_file_to_load)

# 背後にある構造が分類されていることを確認できます
display(Counter(type(element) for element in elements))

# COMMAND ----------

# フロントページの著者はTitleセクションに出現しています  
display(*[(type(element), element.text) for element in elements[0:13]])

# COMMAND ----------

# ナラティブなテキストとしてセクションが抽出されています
display(*[(type(element), element.text) for element in elements[400:410]])

# COMMAND ----------

# MAGIC %md
# MAGIC # Llama_indexによるUnstructuredの活用
# MAGIC
# MAGIC カスタムロジックを追加することでパース方法を改善することは、パフォーマンス向上の一つの方法です。\
# MAGIC 2023年と比較して、モデルは奇妙なチャンクや断片的な段落もますます処理できるようになっています。
# MAGIC
# MAGIC パフォーマンスをさらに向上させるもう一つの方法は、Llama_indexのようなライブラリを利用してチャンクのより賢い構造化を行うことです
# MAGIC
# MAGIC Llama Indexは、チャンク（Llama_indexの用語では`Nodes`）がその空間的な関係性を理解できるように構造化できます
# MAGIC 参照：[Llama Index Types](https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide/)

# COMMAND ----------

# DBTITLE 1,Llama_indexデフォルトモードのセットアップ
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
import nltk

nltk.download('averaged_perceptron_tagger')
model_name = 'databricks-dbrx-instruct'
embedding_model = 'databricks-bge-large-en'

llm_model = ChatDatabricks(
  target_uri='databricks',
  endpoint = model_name,
  temperature = 0.1
)
embeddings = DatabricksEmbeddings(endpoint=embedding_model)

llama_index_chain = LangChainLLM(llm=llm_model)
llama_index_embeddings = LangchainEmbedding(embeddings)
Settings.llm = llama_index_chain
Settings.embed_model = llama_index_embeddings

# COMMAND ----------

# DBTITLE 1,データローダー
# OCR解析を行う必要がある場合はコンピュータービジョンモデルをダウンロードするので、こちらの実行にはある程度の時間を要することに注意してください
from llama_index.core import VectorStoreIndex
from pathlib import Path
from llama_index.readers.file.unstructured import UnstructuredReader

unstruct_loader = UnstructuredReader()
unstructured_document = unstruct_loader.load_data(sample_file_to_load)

# COMMAND ----------

# DBTITLE 1,インデックスの生成
unstructured_index = VectorStoreIndex.from_documents(unstructured_document)
unstructured_query = unstructured_index.as_query_engine()

# COMMAND ----------

# DBTITLE 1,クエリー
question = 'Are there any weak points in ChatGPT for Zero Shot Learning?'
unstructured_result = unstructured_query.query(question)
print(unstructured_result.response)

# COMMAND ----------

# MAGIC %md 他のタイプのインデックスも試してみてください。また、拡張として、複数のドキュメントでのパフォーマンスがどのようになるかも確認してみてください。\
# MAGIC 今のところ単一のドキュメントのみを見てきましたが、複数ドキュメントの状況で最適なドキュメントを特定することは別のものとなります
