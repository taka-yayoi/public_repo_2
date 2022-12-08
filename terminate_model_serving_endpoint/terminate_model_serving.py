# Databricks notebook source
# MAGIC %md
# MAGIC # MLflowモデルサービングエンドポイントの停止
# MAGIC 
# MAGIC モデルサービングのREST APIエンドポイントは、明示的に停止しない限り稼働し続け、コスト増につながります。このノートブックでは、すべてのエンドポイントを停止します。
# MAGIC 
# MAGIC **注意**
# MAGIC - すべてのエンドポイントを停止するには、管理者権限を持つユーザーでこのノートブックを実行してください。
# MAGIC - ジョブでこのノートブックを実行する際には、ユーザーにエンドポイントの運用を周知(xx時間ごとに停止ジョブが実行される等)してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ワークスペースURLとトークンの取得
# MAGIC 
# MAGIC - [How can I make Databricks API calls from notebook?](https://community.databricks.com/s/question/0D53f00001GHVZiCAP/how-can-i-make-databricks-api-calls-from-notebook)
# MAGIC - [MLflow API 2\.0 \| Databricks on AWS](https://docs.databricks.com/dev-tools/api/latest/mlflow.html)

# COMMAND ----------

databricksURL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
myToken = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## エンドポイント一覧の取得
# MAGIC 
# MAGIC 稼働中のエンドポイント一覧を取得します。

# COMMAND ----------

import requests
import json

header = {'Authorization': 'Bearer {}'.format(myToken)}
endpoint = '/api/2.0/mlflow/endpoints/list'
payload = """{}"""
#print(payload)
 
resp = requests.get(
  databricksURL + endpoint,
  data=payload,
  headers=header
)

data = resp.json()
print (json.dumps(data, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## エンドポイントの停止

# COMMAND ----------

for rest_endpoint in data["endpoints"]:
  print(rest_endpoint["registered_model_name"], ":", rest_endpoint["state"])

  # REST APIの停止
  header = {'Authorization': 'Bearer {}'.format(myToken)}
  endpoint = '/api/2.0/mlflow/endpoints/disable'
  payload = {"registered_model_name": rest_endpoint["registered_model_name"]}
  # JSON形式の文字列に変換
  payload = json.dumps(payload)
  #print(payload)

  resp = requests.post(
      databricksURL + endpoint,
      data=payload,
      headers=header
    )

  result_data = resp.json()
  print (json.dumps(result_data, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
