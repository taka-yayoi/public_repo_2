# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC
# MAGIC ドキュメントのインデックスを作成したので、コアアプリケーションのロジックの構築にフォーカスすることができます。このロジックは、ユーザーによる質問に基づいてベクトルストアからドキュメントを取得します。ドキュメントと質問にコンテキストが追加され、レスポンスを生成するためにモデルに送信されるプロンプトを構成するためにそれらを活用します。</p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_application.png' width=900>
# MAGIC
# MAGIC </p>
# MAGIC このノートブックでは、最初に何が行われているのかを把握するために一度ステップをウォークスルーします。そして、我々の作業をより簡単にカプセル化するためにクラスオブジェクトとしてこのロジックを再度パッケージングします。そして、このアクセラレーターの最後のノートブックで、モデルのデプロイをアシストするMLflowの中にモデルとしてこのオブジェクトを永続化します。

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install mlflow==2.4.0
# MAGIC %pip install langchain==0.0.166 tiktoken==0.4.0 faiss-cpu==1.7.4
# MAGIC %pip install sentence_transformers fugashi ipadic
# MAGIC # openai==0.27.6

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインポート
import re
import time
import pandas as pd
import mlflow

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: 回答生成の探索
# MAGIC
# MAGIC まず初めに、どのようにしてユーザーが指定した質問に反応して回答を導き出すのかを探索しましょう。ここでは質問を定義するところからスタートします：

# COMMAND ----------

# DBTITLE 1,質問の指定
question = "Delta Lakeのログ保持期間は"

# COMMAND ----------

# MAGIC %md 
# MAGIC 以前のノートブックで構築したベクトルストアを用いて、質問に適したドキュメントのチャンクを取得します：
# MAGIC
# MAGIC **注意** 設定値を取得するための上での呼び出しを通じて、OpenAIEmbeddingsオブジェクトで使用されるOpenAI APIキーが環境に設定されます。
# MAGIC
# MAGIC [Embeddings — 🦜🔗 LangChain 0\.0\.190](https://python.langchain.com/en/latest/reference/modules/embeddings.html)

# COMMAND ----------

# DBTITLE 1,適切なドキュメントの取得
# エンべディングにアクセスするためにベクトルストアをオープン
embeddings = HuggingFaceEmbeddings(model_name=config['hf_embedding_model'])

vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])

# ドキュメント取得の設定 
n_documents = 5 # 取得するドキュメントの数 
retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # 取得メカニズムの設定

# 適切なドキュメントの取得
docs = retriever.get_relevant_documents(question)
for doc in docs: 
  print(doc,'\n') 

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、モデルに送信されるプロンプトにフォーカスすることができます。このプロンプトには、ユーザーが送信する *question* と、回答の *context* を提供すると信じるドキュメントのプレースホルダーが必要です。
# MAGIC
# MAGIC プロンプトは複数のプロンプト要素から構成され、[prompt templates](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html)を用いて定義されることに注意してください。簡単に言えば、プロンプトテンプレートによって、プロンプトの基本的な構造を定義し、レスポンスをトリガーするために容易に変数データで置き換えることができるようになります。ここで示しているシステムメッセージは、モデルにどのように反応して欲しいのかの指示を当てます。人間によるメッセージテンプレートは、ユーザーが発端となるリクエストに関する詳細情報を提供します。
# MAGIC
# MAGIC プロンプトに対するレスポンスを行うモデルに関する詳細とプロンプトは、[LLMChain object](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html)にカプセル化されます。このオブジェクトはクエリーの解決とレスポンスの返却に対する基本構造をシンプルに定義します：
# MAGIC
# MAGIC **ToDo**: LangChain連携。現状はHuggingFace Pipelinesを使用。
# MAGIC
# MAGIC - [LangChain \+ GPT\-NEOX\-Japanese\-2\.7b で日本語 LLM やりとり整備するメモ \- Qiita](https://qiita.com/syoyo/items/d0fb68d5fe1127276e2a)
# MAGIC - [How to create a custom prompt template — 🦜🔗 LangChain 0\.0\.191](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/examples/custom_prompt_template.html)
# MAGIC - [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

# COMMAND ----------

import torch
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# システムレベルの指示の定義
system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_message_template'])

# 人間駆動の指示の定義
human_message_prompt = HumanMessagePromptTemplate.from_template(config['human_message_template'])

# 単一のプロンプトに指示を統合
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# プロンプトに反応するモデルを定義
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

tokenizer = AutoTokenizer.from_pretrained(config['hf_chat_model'], use_fast=False)
model = AutoModelForCausalLM.from_pretrained(config['hf_chat_model'])#.to(device)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=pipe)

# 作業単位(chain)にプロンプトとモデルを統合
qa_chain = LLMChain(
  llm = llm,
  prompt = chat_prompt, verbose=True
  )

# COMMAND ----------

output = qa_chain.generate([{'context': "テスト", 'question': "Delta Lakeとは"}])
 
# 結果から回答の取得
generation = output.generations[0][0]
answer = generation.text
print(answer)

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("inu-ai/dolly-japanese-gpt-1b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("inu-ai/dolly-japanese-gpt-1b").to(device)

# COMMAND ----------

MAX_ASSISTANT_LENGTH = 100
MAX_INPUT_LENGTH = 1024
INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n'
NO_INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n応答:\n'
USER_NAME = "User"
ASSISTANT_NAME = "Assistant"

def prepare_input(role_instruction, conversation_history, new_conversation):
    instruction = "".join([f"{text} " for text in role_instruction])
    instruction += " ".join(conversation_history)
    input_text = f"{USER_NAME}:{new_conversation}"

    return INPUT_PROMPT.format(instruction=instruction, input=input_text)

def format_output(output):
    output = output.lstrip("<s>").rstrip("</s>").replace("[SEP]", "").replace("\\n", "\n")
    return output

def generate_response(role_instruction, conversation_history, new_conversation):
    # 入力トークン数1024におさまるようにする
    for _ in range(8):
        input_text = prepare_input(role_instruction, conversation_history, new_conversation)
        token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
        n = len(token_ids[0])
        if n + MAX_ASSISTANT_LENGTH <= MAX_INPUT_LENGTH:
            break
        else:
            conversation_history.pop(0)
            conversation_history.pop(0)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            min_length=n,
            max_length=min(MAX_INPUT_LENGTH, n + MAX_ASSISTANT_LENGTH),
            temperature=0.7,
            repetition_penalty=1.0, # 数値を大きくすると、文字列の繰り返しが減る
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    formatted_output_all = format_output(output)

    response = f"{ASSISTANT_NAME}:{formatted_output_all.split('応答:')[-1].strip()}"
    conversation_history.append(f"{USER_NAME}:{new_conversation}".replace("\n", "\\n"))
    conversation_history.append(response.replace("\n", "\\n"))

    return formatted_output_all, response 

role_instruction = [
    f"{USER_NAME}:きみは「ずんだもん」なのだ。東北ずん子の武器である「ずんだアロー」に変身する妖精またはマスコットなのだ。一人称は「ボク」で語尾に「なのだー」を付けてしゃべるのだ。",
    f"{ASSISTANT_NAME}:了解したのだ。",
    f"{USER_NAME}:きみは同じ言葉を繰り返さず、何でも正確に要約して答えられるのだ。",
    f"{ASSISTANT_NAME}:了解したのだ。",
]

conversation_history = [
]

questions = [
    "日本で一番高い山は？",
    "日本で一番広い湖は？",
    "冗談を言ってください。",
    "世界で一番高い山は？",
    "世界で一番広い湖は？",
    "最初の質問は何ですか？",
    "今何問目？",
    "自己紹介をしてください。",
]

# 各質問に対して応答を生成して表示
for question in questions:
    formatted_output_all, response = generate_response(role_instruction, conversation_history, question)
    print(f"{USER_NAME}:{question}\n{response}\n---")


# COMMAND ----------

llm(context="Delta Time Travelを使用して、デルタテーブルの以前のバージョンにアクセスすることができます。Delta Lakeは、デフォルトで30日間のバージョン履歴を保持しますが、必要であればより長い履歴を保持することもできます。", question="Deltaのログ保持期間は")

# COMMAND ----------

# DBTITLE 1,レスポンスの生成
# 指定されたドキュメントのそれぞれに対して
for doc in docs:

  # ドキュメントテキストの取得
  text = doc.page_content

  # レスポンスの生成
  prompt = f"ユーザー: あなたはDatabricksによって開発された有能なアシスタントであり、指定されたコンテキストに基づいて質問に回答することを得意としています。コンテキストはドキュメントです。コンテキストが回答を決定するのに十分な情報を提供しない場合には、「わかりません」と言ってください。コンテキストが質問に適していない場合には、「わかりません」と言ってください。コンテキストから良い回答が見つからない場合には、「わかりません」と言ってください。問い合わせが完全な質問になっていない場合には、「わかりません」と言ってください。コンテキストから良い回答が得られた場合には、質問に回答するためにコンテキストを要約してください。<NL>システム: 質問は何ですか？<NL>ユーザー: {question}<NL>システム: コンテキストは何ですか？<NL>ユーザー: {text}<NL>システム: "
  #print(prompt)

  output = llm(prompt)
 
  # 結果から回答の取得
  #generation = output.generations[0][0]
  #answer = generation.text

  # 回答の表示
  if output is not None:
    print(f"Question: {question}", '\n', f"Answer: {output}")
    break

# COMMAND ----------

# MAGIC %md ##Step 2: デプロイするモデルの構築
# MAGIC
# MAGIC レスポンス生成に関連する基本的なステップを探索したら、デプロイメントを容易にするためにクラスの中にロジックをラップしましょう。我々のクラスは、LLMモデル定義、ベクトルストアの収集器、クラスに対するプロンプトを渡すことでインスタンスを生成します。*get_answer*メソッドは、質問を送信してレスポンスを取得するための主要なメソッドとして機能します：

# COMMAND ----------

# DBTITLE 1,QABotクラスの定義
class QABot():


  def __init__(self, llm, retriever):
    self.llm = llm
    self.retriever = retriever
    self.abbreviations = { # 置換したい既知の略語
      "DBR": "Databricks Runtime",
      "ML": "Machine Learning",
      "UC": "Unity Catalog",
      "DLT": "Delta Live Table",
      "DBFS": "Databricks File Store",
      "HMS": "Hive Metastore",
      "UDF": "User Defined Function"
      } 


  def _is_good_answer(self, answer):

    ''' 回答が妥当かをチェック '''

    result = True # デフォルトのレスポンス

    badanswer_phrases = [ # モデルが回答を生成しなかったことを示すフレーズ
      "わかりません", "コンテキストがありません", "知りません", "答えが明確でありません", "すみません", 
      "答えがありません", "説明がありません", "リマインダー", "コンテキストが提供されていません", "有用な回答がありません", 
      "指定されたコンテキスト", "有用でありません", "適切ではありません", "質問がありません", "明確でありません",
      "十分な情報がありません", "適切な情報がありません", "直接関係しているものが無いようです"
      ]
    
    if answer is None: # 回答がNoneの場合は不正な回答
      results = False
    else: # badanswer phraseを含んでいる場合は不正な回答
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' タイムアウトハンドリングありのLLMからの回答取得 '''

    # デフォルトの結果
    result = None

    # 終了時間の定義
    end_time = time.time() + timeout_sec

    # タイムアウトに対するトライ
    while time.time() < end_time:

      # レスポンス取得の試行
      try: 
        prompt = f"ユーザー: あなたはDatabricksによって開発された有能なアシスタントであり、指定されたコンテキストに基づいて質問に回答することを得意としています。コンテキストはドキュメントです。コンテキストが回答を決定するのに十分な情報を提供しない場合には、「わかりません」と言ってください。コンテキストが質問に適していない場合には、「わかりません」と言ってください。コンテキストから良い回答が見つからない場合には、「わかりません」と言ってください。問い合わせが完全な質問になっていない場合には、「わかりません」と言ってください。コンテキストから良い回答が得られた場合には、質問に回答するためにコンテキストを要約してください。<NL>システム: 質問は何ですか？<NL>ユーザー: {question}<NL>システム: コンテキストは何ですか？<NL>ユーザー: {context}<NL>システム: "

        result =  self.llm(prompt)
        break # レスポンスが成功したらループをストップ

      # レートリミットのエラーが起きたら...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # 時間があるのであればsleep
          time.sleep(2)
          continue
        else: # そうでなければ例外を発生
          raise rate_limit_error

      # その他のエラーでも例外を発生
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' 指定された質問の回答を取得 '''

    # デフォルトの結果
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # 質問から一般的な略語を削除
    for abbreviation, full_text in self.abbreviations.items():
      pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
      question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # 適切なドキュメントの取得
    docs = self.retriever.get_relevant_documents(question)

    # それぞれのドキュメントごとに ...
    for doc in docs:

      # ドキュメントのキー要素を取得
      text = doc.page_content
      source = doc.metadata['source']

      # LLMから回答を取得
      output = self._get_answer(text, question)
 
      # 結果からアウトプットを取得
      answer = output

      # no_answer ではない場合には結果を構成
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        break # 良い回答であればループをストップ
      
    return result

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、以前インスタンス化したオブジェクトを用いてクラスをテストすることができます：

# COMMAND ----------

# DBTITLE 1,QABotクラスのテスト
# botオブジェクトのインスタンスを作成
qabot = QABot(llm, retriever)

# 質問に対するレスポンスの取得
qabot.get_answer(question) 

# COMMAND ----------

# MAGIC %md ##Step 3: MLflowにモデルを永続化
# MAGIC
# MAGIC 我々のbotクラスが定義、検証されたので、MLflowにこれを永続化します。MLflowはモデルのトラッキングとロギングのためのオープンソースのリポジトリです。Databricksプラットフォームにはデフォルトでデプロイされており、簡単にモデルを記録することができます。
# MAGIC
# MAGIC 今では、MLflowはOpenAIとLangChainの両方のモデルフレーバーを[サポート](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html)していますが、我々のbotアプリケーションではカスタムロジックを記述しているので、より汎用的な[pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models)モデルフレーバーを活用しなくてはなりません。このモデルフレーバーによって、標準的なMLflowのデプロイメントメカニズムを通じてデプロイされた際に、モデルがどのように反応するのかに関して非常に多くのコントロールを行えるように、モデルに対するカスタムラッパーを記述できるようになります。
# MAGIC
# MAGIC カスタムMLflowモデルを作成するのに必要なことは、*mlflow.pyfunc.PythonModel*のタイプのカスタムラッパーを定義することだけです。 *\_\_init__* メソッドは、*QABot*クラスのインスタンスを初期化し、クラス変数に永続化します。そして、 *predict* メソッドは、レスポンス生成の標準的なインタフェースとして動作します。このメソッドはpandasデータフレームとして入力を受け付けますが、ユーザーから一度に一つの質問を受け取るという知識を用いてロジックを記述することができます：

# COMMAND ----------

# DBTITLE 1,モデルのMLflowラッパーの定義
class MLflowQABot(mlflow.pyfunc.PythonModel):

  def __init__(self, llm, retriever):
    self.qabot = QABot(llm, retriever)

  def predict(self, context, inputs):
    questions = list(inputs['question'])

    # 回答の返却
    return [self.qabot.get_answer(q) for q in questions]

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、以下のようにモデルのインスタンスを作成し、[MLflow registry](https://docs.databricks.com/mlflow/model-registry.html)に記録します：

# COMMAND ----------

# DBTITLE 1,MLflowにモデルを永続化
# mlflowモデルのインスタンスを作成
model = MLflowQABot(llm, retriever)

# mlflowにモデルを永続化
with mlflow.start_run():
  _ = (
    mlflow.pyfunc.log_model(
      python_model=model,
      extra_pip_requirements=['langchain==0.0.166', 'openai==0.27.6', 'tiktoken==0.4.0', 'faiss-cpu==1.7.4'],
      artifact_path='model',
      await_registration_for = 1200, # モデルサイズが大きいので長めの待ち時間にします
      registered_model_name=config['registered_model_name']
      )
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowが始めてであれば、ロギングが何の役に立つのかと思うかもしれません。このノートブックに関連づけられているエクスペリメントに移動して、*log_model*の呼び出しによって記録されたものに対する詳細を確認するために、最新のエクスペリメントをクリックすることができます。エクスペリメントにアクセスするにはDatabricks環境の右側のナビゲーションにあるフラスコアイコンをクリックします。モデルのアーティファクトを展開すると、以前インスタンスを作成したMLflowQABotモデルのpickleを表現する*python_model.pkl*を確認することができます。これが(後で)本環境あるいは別環境でモデルをロードする際に取得されるモデルとなります：
# MAGIC </p>
# MAGIC
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/bot_mlflow_log_model.PNG" width=1000>

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowのモデルレジストリは、CI/CDワークフローを移動する際に登録されたモデルを管理するメカニズムを提供します。モデルを直接プロダクションのステータスにプッシュ(デモでは構いませんが、現実世界のシナリオでは推奨しません)したいのであれば、以下のようにプログラムから行うことができます：

# COMMAND ----------

# DBTITLE 1,モデルをプロダクションステータスに昇格
# mlflowに接続
client = mlflow.MlflowClient()

# 最新モデルバージョンの特定
latest_version = client.get_latest_versions(config['registered_model_name'], stages=['None'])[0].version

# モデルをプロダクションに移行
client.transition_model_version_stage(
    name=config['registered_model_name'],
    version=latest_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、レスポンスを確認するために、レジストリからモデルを取得し、いくつかの質問を送信することができます：

# COMMAND ----------

# DBTITLE 1,モデルのテスト
# mlflowからモデルを取得
model = mlflow.pyfunc.load_model(f"models:/{config['registered_model_name']}/Production")

# 質問入力の構築
queries = pd.DataFrame({'question':[
  "Delta Sharingとは何？",
  "MLflowのメリットは？",
  "Unity Catalogのセットアップ方法"
]})

# レスポンスの取得
model.predict(queries)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
