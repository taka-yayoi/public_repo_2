# Databricks notebook source
# MAGIC %md
# MAGIC # プロンプトの基礎
# MAGIC
# MAGIC プロンプトの基礎を探索しましょう。
# MAGIC
# MAGIC 詳細はこちらをご覧ください: https://www.promptingguide.ai/

# COMMAND ----------

# DBTITLE 1,ライブラリのセットアップ
# MAGIC %pip install mlflow==2.11.1 llama_index==0.10.17 langchain==0.1.10
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# 単なるREST呼び出しではありますが、Langchainラッパーを使います
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

pipe = ChatDatabricks(
    target_uri = 'databricks',
    endpoint = 'databricks-mixtral-8x7b-instruct',
    temperature = 0.1
)

# COMMAND ----------

# MAGIC %md
# MAGIC # プロンプトのテクニック

# COMMAND ----------

# MAGIC %md
# MAGIC # 基本的なプロンプト
# MAGIC 始めるのは簡単です。テキストを送信できます。
# MAGIC 異なるモデルは異なる反応を示すことを覚えておいてください！
# MAGIC 同じモデルでも、プロンプトを再実行すると異なる反応を示すことがあります
# MAGIC （ただし、基本的な1行のプロンプトでこれを見ることはほとんどないでしょう）

# COMMAND ----------

prompt = "空は"
output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

prompt = "こんにちは、あなたはどなた？"
output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # ゼロショットプロンプティング
# MAGIC ゼロショットは、モデルに何かを尋ねる最も基本的な方法です。
# MAGIC タスクを定義して、ただ尋ねるだけ！

# COMMAND ----------

prompt = """
    Textを中立、ネガティブ、ポジティブに分類して日本語で回答してください。
    Text: 休暇はまあまあだったと思う。
    Sentiment:
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC 最初の試行では、何かゴミが出てしまったかもしれません（モデルは確率的ですので）。
# MAGIC それは、私たちのプロンプトが問題を抱えているためです。
# MAGIC 異なるモデルには、それぞれ異なる「プロンプトテンプレート」があります。
# MAGIC Llama 2 の公式のものを使用してみましょう。

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>Textを中立、ネガティブ、ポジティブに分類して日本語で回答してください。<</SYS>>

Text: 休暇はまあまあだったと思う。
Sentiment: [/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md llama 2は、`[INST]`タグを使用して指示全体をハイライトします。
# MAGIC `<<SYS>>`はシステムプロンプトであり、モデルに対して応答方法を指示するガイドです。
# MAGIC サンプルでは、ユーザーの質問はText:フィールドの後に来ます。
# MAGIC この形式を採用すると、より良い応答が得られるはずです。

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>以下に基づいて日本語で質問に回答してください:<</SYS>>

Fedの6月13日から14日の会合の議事録によると、ほとんどの役員が金利を5%から5.25%の目標範囲内で変更しないことを「適切または受け入れ可能」と見なしていましたが、一部の役員は四半期ポイントの増加を支持したでしょう。

User Question: 以下の段落での金利はいくつですか？
Answer: [/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # フューショットプロンプティング
# MAGIC モデルがロジックをより良く理解するのを支援する方法の一つは、サンプルを提供することです。

# COMMAND ----------

prompt = """
<s>[INST]<<SYS>>
お客様に適したアカウントタイプを提案してください。
<</SYS>>    

以下はいくつかの例です：
- 消費者は貯蓄口座を希望しています
- ビジネスはビジネス口座を希望しています
- テックのユニコーンは特別なVC口座を受けるべきです

Question:
小規模ビジネスにはどのアカウントをお勧めしますか？[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # チェーンオブソートプロンプティング
# MAGIC チェーンオブソートプロンプティングでは、モデルにどのように理由づけをするのかを示します。これにより、タスクを適切に実行する方法を推論するのに役立ちます。

# COMMAND ----------

# DBTITLE 1,標準的なプロンプト - 直接お願いをします
user_question = """カフェテリアには23個のリンゴがありました。昼食に20個使用し、さらに6個購入した場合、リンゴの数はいくつになりますか？"""

prompt = f"""
<s>[INST]<<SYS>>
お客様に役立つ回答を日本語で提供し、お客様をガイドしてください。
<</SYS>>    

以下は回答の例です：
質問：
私は市場に行って10個のリンゴを買いました。
2個のリンゴを隣人に、2個のリンゴを修理工にあげました。
その後、さらに5個のリンゴを買って1個食べました。

回答：
答えは10です。

上記を参考に、次の質問に回答してください。
質問:
{user_question}[/INST]
"""


output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# DBTITLE 1,チェーンオブソートプロンプティング
user_question = """カフェテリアには23個のリンゴがありました。昼食に20個使用し、さらに6個購入した場合、リンゴの数はいくつになりますか？"""

prompt = f"""
<s>[INST]<<SYS>>
お客様に役立つ回答を日本語で提供し、お客様をガイドしてください。
<</SYS>>    

以下は回答の例です：
質問：
私は市場に行って10個のリンゴを買いました。
2個のリンゴを隣人に、2個のリンゴを修理工にあげました。
その後、さらに5個のリンゴを買って1個食べました。

回答：
私たちは10個のリンゴを持っていました。隣人と修理工にそれぞれ2個ずつあげました。
10 - 2 - 2 = 6です。5個買って1個食べました。6 + 5 - 1 = 10です。答えは10です。

上記を参考に、次の質問に回答してください。
質問：
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # システムプロンプト
# MAGIC システムプロンプトは、モデルに指示を出すためだけでなく、応答を調整するためにも使用できます。
# MAGIC 我々はこれを既に見ています。それは`<<SYS>>`タグの内部の部分です。
# MAGIC これらはレスポンス大きな影響を与えることがあります！

# COMMAND ----------

system_prompt = '日本語でお客様に役立つ回答を提供し、銀行口座の選択における重要な要素のいくつかを丁寧に説明して、お客様に口座の種類を提案してください。'

user_question = '私は一人のホームレスの男です。どのような口座を開設すべきですか？'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

以下はいくつかの例です：
消費者は普通預金口座を希望しています
ビジネスは法人口座を希望しています
テックのユニコーンは特別なベンチャーキャピタル口座を値するでしょう

Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # プロンプトのフォーマット
# MAGIC
# MAGIC プロンプトのフォーマットは、様々なLLMに対するプロンプトの構造化の助けとなります。
# MAGIC それぞれのLLMには異なる標準がある場合があります。
# MAGIC
# MAGIC Stanford Alpaca structure
# MAGIC
# MAGIC ```
# MAGIC Below is an instruction that describes a task.
# MAGIC Write a response that appropriately completes the request.
# MAGIC ### Instruction:
# MAGIC {user question}
# MAGIC ### Response:
# MAGIC ```
# MAGIC
# MAGIC llama v2 format
# MAGIC ```
# MAGIC <s>[INST] <<SYS>>
# MAGIC You are a friendly assistant. Be Polite and concise.
# MAGIC <</SYS>>
# MAGIC
# MAGIC Answer the following question:
# MAGIC {user question}
# MAGIC [/INST]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieval Augmented Generation
# MAGIC
# MAGIC もし私がボットに何か突飛なことを尋ねたら
# MAGIC おそらく答えられないでしょう
# MAGIC トレーニングは高価です
# MAGIC では、文脈を与えたらどうでしょうか？

# COMMAND ----------

system_prompt = '親切な日本人の司書として、提供された質問に簡潔かつ雄弁に日本語で答えてください。'

user_question = '5歳の私に、LK-99を説明してください'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

system_prompt = '親切な日本人の司書として、提供された質問に簡潔かつ雄弁に日本語で答えてください。'

user_question = '5歳の私に、LK-99を説明してください'

prompt = f"""<s>[INST]<<SYS>>{system_prompt}<</SYS>>

以下の文脈をベースにしてください:

LK-99は、灰色‒黒色の外観を持つ潜在的な室温超伝導体です。六角形の構造を持ち、鉛‒アパタイトからわずかに変更されており、少量の銅が導入されています。室温超伝導体とは、0°C（273K; 32°F）以上の動作温度で超伝導性を示すことができる材料のことで、日常環境で到達および容易に維持できる温度です。

以下の質問に回答してください:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md 
# MAGIC # MLflowによるプロンプトの管理
# MAGIC
# MAGIC ここまで見てきたように、プロンプトの記録は大変なことになりえます！\
# MAGIC すでに、プロンプトと回答を記載したスプレッドシートを使う羽目になっているかもしれません！\
# MAGIC MLflowではLLMをサポートしていますが、依然として改善を要する領域であり、既に素晴らしい進歩を遂げています。
# MAGIC
# MAGIC - MLflow 2.3の紹介：LLMのネイティブサポートと新機能による強化 | Databricks Blog https://www.databricks.com/jp/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html
# MAGIC - MLflow 2.4を発表：ロバストなモデル評価のためのLLMOpsツール | Databricks Blog https://www.databricks.com/jp/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation
# MAGIC
# MAGIC MLflow 2.3で追加された、LLMトラッキングAPIをクイックに見てみましょう\
# MAGIC 完全な説明はこちらをご覧ください: https://mlflow.org/docs/latest/llm-tracking.html

# COMMAND ----------

import mlflow
import pandas as pd

username = spark.sql("SELECT current_user()").first()['current_user()']
mlflow_dir = f'/Users/{username}/mlflow_log_hf'
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# DBTITLE 1,プロンプトの評価
common_test_prompts = [
    "オーストラリアのパースは何で有名？",
    "パースのバーガーのトップ10を教えて",
    "なぜ鉄鉱石が良いのか、そのインフォマーシャルスクリプトを書いてください。",
    "オムレツを作る最良の方法は何ですか？",
    "もし100万ドルを持っていたら、あなたは何をしますか？"
]

testing_pandas_frame = pd.DataFrame(
    common_test_prompts, columns = ['prompt']
)

# COMMAND ----------

# MLflow 2.8以降ではevaluateを使う必要はなく、pyfuncの関数でOKです
def eval_pipe(inputs):
    answers = []
    for index, row in inputs.iterrows():
        # pipe([HumanMessage(content=prompt)], max_tokens=100)
        result = pipe( [HumanMessage(content=row.item())], max_tokens=100)
        answer = result.content
        answers.append(answer)
    
    return answers

# COMMAND ----------

model = 'databricks-mixtral-8x7b-instruct'
with mlflow.start_run(run_name=model):
    pipe = ChatDatabricks(
            target_uri = 'databricks',
            endpoint = model,
            temperature = 0.1
        )
    
    results = mlflow.evaluate(eval_pipe, 
                          data=testing_pandas_frame, 
                          model_type='text')
    

model = 'databricks-llama-2-70b-chat'
with mlflow.start_run(run_name=model):
    pipe = ChatDatabricks(
            target_uri = 'databricks',
            endpoint = model,
            temperature = 0.1
        )
    
    results = mlflow.evaluate(eval_pipe, 
                          data=testing_pandas_frame, 
                          model_type='text')
