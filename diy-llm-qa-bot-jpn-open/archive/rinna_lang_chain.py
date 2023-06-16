# Databricks notebook source
# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install langchain==0.0.191 tiktoken==0.4.0 faiss-cpu==1.7.4
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

#from langchain.chat_models import ChatOpenAI
#from langchain.embeddings.openai import OpenAIEmbeddings

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

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# COMMAND ----------

# プロンプトに反応するモデルを定義
llm = HuggingFaceHub(repo_id="rinna/japanese-gpt-neox-3.6b-instruction-ppo", model_kwargs={"temperature":config['temperature']})

# テンプレートの準備
template = """ユーザー: あなたは劇作家です。劇のタイトルが与えられた場合、そのタイトルのあらすじを書くのがあなたの仕事です。<NL>タイトル: {title}<NL>システム: """

# プロンプトテンプレートの準備
prompt_template = PromptTemplate(
    input_variables=["title"], 
    template=template
)

# LLMChainの準備
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# COMMAND ----------

prompt_template

# COMMAND ----------

review = synopsis_chain.predict(title="浜辺の夕暮れ時の悲劇")
review

# COMMAND ----------

question = "Deltaログの保持期間"

# COMMAND ----------

prompt = [
    {
        "speaker": "ユーザー",
        "text": "あなたはDatabricksによって開発された有能なアシスタントであり、指定されたコンテキストに基づいて質問に回答することを得意としており、コンテキストはドキュメントです。コンテキストが回答を決定するのに十分な情報を提供しない場合には、わかりませんと言ってください。コンテキストが質問に適していない場合には、わかりませんと言ってください。コンテキストから良い回答が見つからない場合には、わかりませんと言ってください。問い合わせが完全な質問になっていない場合には、わからないと言ってください。コンテキストから良い回答が得られた場合には、質問に回答するためにコンテキストを要約してみてください。"
    },
    {
        "speaker": "システム",
        "text": "どのような質問でしょうか？"
    },
    {
        "speaker": "ユーザー",
        "text": f"{question}"
    },
    {
        "speaker": "システム",
        "text": "どのようなコンテキストでしょうか？"
    },
     {
        "speaker": "ユーザー",
        "text": "Delta Time Travelを使用して、デルタテーブルの以前のバージョンにアクセスすることができます。Delta Lakeは、デフォルトで30日間のバージョン履歴を保持しますが、必要であればより長い履歴を保持することもできます。 "
    },
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "<NL>".join(prompt)
prompt = (
    prompt
    + "<NL>"
    + "システム: "
)

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo")

if torch.cuda.is_available():
    model = model.to("cuda")

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
output = output.replace("<NL>", "\n")
print(output)

# COMMAND ----------

from langchain import HuggingFaceHub

repo_id = "databricks/dolly-v2-3b"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_length":64})
llm

# COMMAND ----------

# MAGIC %md
# MAGIC - [Timeout when running hugging face LLMs for ConversationRetrivalChain · Issue \#3275 · hwchase17/langchain](https://github.com/hwchase17/langchain/issues/3275)
# MAGIC - [google/flan\-t5\-xxl · Error raised by inference API: Model google/flan\-t5\-xl time out](https://huggingface.co/google/flan-t5-xxl/discussions/43)

# COMMAND ----------

from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))

# COMMAND ----------

import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# COMMAND ----------

print(llm_chain.predict(instruction="Explain to me the difference between nuclear fission and fusion.").lstrip())

# COMMAND ----------


