# Databricks notebook source
# MAGIC %md
# MAGIC ## Driver Proxyへの接続確認

# COMMAND ----------

import requests
import json

def request_rag_chatbot(prompt, temperature=1.0, max_new_tokens=1024):
  token = "" # TODO: Driver Proxyが動作しているクラスターと別のクラスター、ローカルからアクセスする際にはパーソナルアクセストークンが必要です
  url = "http://127.0.0.1:7777/" # Driver Proxyが動作しているポートを指定します
  
  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "prompt": prompt,
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
  }

  response = requests.post(url, headers=headers, data=json.dumps(data))
  return response

# COMMAND ----------

# Driver Proxyへのリクエスト
response = request_rag_chatbot("Databricksとは？")
response_json = response.json()

print("回答: " + response_json["answer"])
print("ソース: " + response_json["source"])

# COMMAND ----------


