# Databricks notebook source
# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install langchain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC [Hugging Face Hub — 🦜🔗 LangChain 0\.0\.194](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html)

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

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# COMMAND ----------

from langchain import HuggingFaceHub

repo_id = "stabilityai/stablelm-tuned-alpha-3b" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

# COMMAND ----------

from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))

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


