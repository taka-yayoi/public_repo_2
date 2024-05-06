# Databricks notebook source
# MAGIC %md
# MAGIC ## Python User-Defined Table Functions (UDTFs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### イントロダクション
# MAGIC
# MAGIC [Python user-defined table function (UDTF)](https://spark.apache.org/docs/latest/api/python/user_guide/sql/python_udtf.html)は、出力を単一のスカラー結果値ではなく、テーブル全体を返却する新たなタイプの関数です。登録すると、SQLクエリーの`FROM`句で使用できるようになります。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python UDTFの作成

# COMMAND ----------

from pyspark.sql.functions import udtf

# UDTFクラスの定義、必要となる `eval` メソッドの実装
@udtf(returnType="num: int, squared: int")
class SquareNumbers:
    """(num, squared)ペアのシーケンスを生成"""
    def eval(self, start: int, end: int):
        for num in range(start, end + 1):
            yield (num, num * num)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PythonでのPython UDTFの利用

# COMMAND ----------

from pyspark.sql.functions import lit

SquareNumbers(lit(1), lit(3)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQLでのPython UDTFの利用

# COMMAND ----------

# Spark SQLで利用できるためにUDTFを登録
spark.udtf.register("square_numbers", SquareNumbers)

# COMMAND ----------

spark.sql("SELECT * FROM square_numbers(1, 3)").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arrow最適化Python UDTF
# MAGIC
# MAGIC Apache ArrowはJavaプロセスとPythonプロセス間の効率的なデータ転送を実現する、インメモリの列指向データフォーマットです。
# MAGIC
# MAGIC UDTFが大量の行を出力する際に、劇的にパフォーマンスをブーストします。Arrow最適化は`useArrow=True`を用いることで有効化することができます。

# COMMAND ----------

@udtf(returnType="num: int, squared: int", useArrow=True)
class SquareNumbersArrow:
    def eval(self, start: int, end: int):
        for num in range(start, end + 1):
            yield (num, num * num)

# COMMAND ----------

# 通常のPython UDTF
SquareNumbers(lit(0), lit(10000000)).show()

# COMMAND ----------

# Arrow最適化Python UDTF
SquareNumbersArrow(lit(0), lit(10000000)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LangChainを用いた現実世界のユースケース
# MAGIC
# MAGIC Python UDTFとOpenAI、LangChainを組み合わせた、より面白い例を深掘りしていきましょう。

# COMMAND ----------

# MAGIC %pip install openai langchain

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pyspark.sql.functions import lit, udtf

#MODEL_NAME = "gpt-4"
MODEL_NAME = "gpt-3.5-turbo-instruct"
#MODEL_NAME = "gpt-4-turbo"
API_KEY = dbutils.secrets.get("demo-token-takaaki.yayoi", "openai_api_key")

@udtf(returnType="keyword: string")
class KeywordsGenerator:
    """
    LLMを用いてトピックに関するカンマ区切りのキーワードのリストを生成します。
    キーワードのみを出力します。
    """
    def __init__(self):
        print(MODEL_NAME)
        llm = OpenAI(model_name=MODEL_NAME, openai_api_key=API_KEY)
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="generate a couple of comma separated keywords about {topic}. Output only the keywords."
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def eval(self, topic: str):
        response = self.chain.run(topic)
        keywords = [keyword.strip() for keyword in response.split(",")]
        for keyword in keywords:
            yield (keyword, )

# COMMAND ----------

KeywordsGenerator(lit("apache spark")).show(truncate=False)

# COMMAND ----------

KeywordsGenerator(lit("日本")).show(truncate=False)

# COMMAND ----------

@udtf(returnType="keyword: string")
class KeywordsGenerator:
    """
    LLMを用いてトピックに関するカンマ区切りのキーワードのリストを生成します。
    キーワードのみを出力します。
    """
    def __init__(self):
        print(MODEL_NAME)
        llm = OpenAI(model_name=MODEL_NAME, openai_api_key=API_KEY)
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="{topic}に関する半角カンマ区切りのキーワードをいくつか生成してください。生成結果にはキーワードのみを含めてください。"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def eval(self, topic: str):
        response = self.chain.run(topic)
        keywords = [keyword.strip() for keyword in response.split(",")]
        for keyword in keywords:
            yield (keyword, )

# COMMAND ----------

KeywordsGenerator(lit("日本文化")).show(truncate=False)

# COMMAND ----------


