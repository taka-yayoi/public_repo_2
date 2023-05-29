# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC
# MAGIC このノートブックでは、以前のノートブックでMLflowに登録したカスタムモデルを、Databricksのモデルサービング([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/))にデプロイします。Databricksモデルサービングは、認証されたアプリケーションがREST API経由で登録されたモデルとインタラクションできる、コンテナ化されたデプロイメントオプションを提供します。これによって、MLOpsチームはモデルを簡単にデプロイ、管理し、様々なアプリケーションとこれらのモデルをインテグレーションできるようになります。

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# DBTITLE 1,インポート
import mlflow
import requests
import json
import time
from mlflow.utils.databricks_utils import get_databricks_host_creds

# COMMAND ----------

# DBTITLE 1,デプロイする最新Productionモデルのバージョンを取得
latest_version = mlflow.MlflowClient().get_latest_versions(config['registered_model_name'], stages=['Production'])[0].version

# COMMAND ----------

# MAGIC %md ##Step 1: モデルサービングエンドポイントのデプロイ
# MAGIC
# MAGIC 通常、モデルはDatabricksワークスペースのUIかREST APIを用いて、モデル共有エンドポイントにデプロイされます。我々のモデルはセンシティブな環境変数のデプロイメントに依存しているので、現在REST API経由でのみ利用できる比較的新しいモデルサービングの機能を活用する必要あります。
# MAGIC
# MAGIC 以下でサーブされるモデルの設定を見ると、サーブされるモデルの設定の一部に`env_vars`があることに気づくでしょう。これで、シークレットスコープにキーを格納し、環境変数としてモデルサービングエンドポイントに引き渡すことができます。

# COMMAND ----------


served_models = [
    {
      "name": "current",
      "model_name": config['registered_model_name'],
      "model_version": latest_version,
      "workload_size": "Small",
      "scale_to_zero_enabled": "true",
      "env_vars": [{
        "env_var_name": "OPENAI_API_KEY",
        "secret_scope": config['openai_key_secret_scope'],
        "secret_key": config['openai_key_secret_key'],
      }]
    }
]
traffic_config = {"routes": [{"served_model_name": "current", "traffic_percentage": "100"}]}

# COMMAND ----------

# DBTITLE 1,仕様に合わせてエンドポイントを作成、更新する関数の定義
def endpoint_exists():
  """serving_endpoint_nameの名前のエンドポイントが存在するかどうかをチェック"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

def wait_for_endpoint():
  """デプロイメントの準備ができるまで待ち、エンドポイント設定を返却"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # エンドポイントの準備ができていない、あるいは、設定更新中
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

def create_endpoint():
  """サービングエンドポイントを作成し、準備ができるまで待つ"""
  print(f"Creating new serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": config['serving_endpoint_name'], "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")
  
def update_endpoint():
  """サービングエンドポイントを更新し、準備ができるまで待つ"""
  print(f"Updating existing serving endpoint: {config['serving_endpoint_name']}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{config['serving_endpoint_name']}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint()
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{config['serving_endpoint_name']}" target="_blank">{config['serving_endpoint_name']}</a> serving endpoint""")

# COMMAND ----------

# DBTITLE 1,エンドポイントの作成、更新に定義した関数を使用
# APIが必要とするその他の入力を収集
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

# エンドポイントの作成/更新のスタート
if not endpoint_exists():
  create_endpoint()
else:
  update_endpoint()

# COMMAND ----------

# MAGIC %md 
# MAGIC 作成したばかりのモデルサービングエンドポイントにアクセスするには上のリンクを使用できます。 
# MAGIC
# MAGIC <img src='https://github.com/databricks-industry-solutions/diy-llm-qa-bot/raw/main/image/model_serving_ui.png'>

# COMMAND ----------

# MAGIC %md ##Step 2: エンドポイントAPIのテスト

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、このエンドポイントにクエリーを行う関数をセットアップするために以下のコードを使用します。このコードは、サービングエンドポイントページでアクセスできる*Query Endpoint*のUIでアクセスするコードから若干変更しています：

# COMMAND ----------

# DBTITLE 1,エンドポイントにクエリーする関数の定義
import os
import requests
import numpy as np
import pandas as pd
import json

endpoint_url = f"""https://{serving_host}/serving-endpoints/{config['serving_endpoint_name']}/invocations"""


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = endpoint_url
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

# MAGIC %md これでエンドポイントをテストすることができます：

# COMMAND ----------

# DBTITLE 1,モデルサービングエンドポイントのテスト
# 質問入力の構築
queries = pd.DataFrame({'question':[
  "Databricksのレイクハウスとは？"
]})

score_model( 
   queries
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC いくかの制限があります：
# MAGIC - エンドポイントをゼロまでスケールするようにすると、クエリーがない際のbotのコストを削減できます。しかし、長い期間の後の最初のリクエストは、エンドポイントがゼロノードからスケールアップする必要があるため、数分を要します。
# MAGIC - サーバレスモデルサービングリクエストのタイムアウトリミットは60秒です。同じリクエストで3つの質問が送信されると、モデルがタイムアウトすることがあります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Streamlitとの連携
# MAGIC
# MAGIC せっかくですので、GUIからエンドポイント上のモデルを呼び出せるようにしてみましょう。Streamlitがインストールされている環境で以下のコードを`streamlit run databricks_qa.py`として保存して以下を実行します。
# MAGIC
# MAGIC ```
# MAGIC streamlit run databricks_qa.py
# MAGIC ```

# COMMAND ----------

import streamlit as st 
import numpy as np 
from PIL import Image
import base64
import io

import os
import requests
import numpy as np
import pandas as pd

import json

st.header('Databricks Q&A bot')
st.write('''
- [カスタマーサービスとサポートにおける大規模言語モデルの革命をドライブする \- Qiita](https://qiita.com/taka_yayoi/items/447ab95af2b8493a04dd)
''')


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(question):
  # 1. パーソナルアクセストークンを設定してください
  # 今回はデモのため平文で記載していますが、実際に使用する際には環境変数経由で取得する様にしてください。
  token = "<パーソナルアクセストークン>"
  #token = os.environ.get("DATABRICKS_TOKEN")

  # 2. モデルエンドポイントのURLを設定してください
  url = '<エンドポイントのURL>'
  headers = {'Authorization': f'Bearer {token}',
             "Content-Type": "application/json",}

  dataset = pd.DataFrame({'question':[question]})

  ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method="POST", headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(
       f"Request failed with status {response.status_code}, {response.text}"
    )
  
  return response.json()

question = st.text_input("質問")

if question != "":
  response = score_model(question)

  answer = response['predictions'][0]["answer"]
  source = response['predictions'][0]["source"]

  st.write(f"回答: {answer}")
  st.write(f"ソース: [{source}]({source})")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC Q&Aチャットbotアプリの完成です！
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/databricks_qa.png)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
