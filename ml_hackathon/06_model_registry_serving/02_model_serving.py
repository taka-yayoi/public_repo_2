# Databricks notebook source
# MAGIC %md
# MAGIC # 【step7】モデルサービング
# MAGIC 
# MAGIC トレーニングした機械学習モデルをバッチ推論に用いることができますが、モデルサービングを用いてREST APIを公開することで、機械学習モデルの活用の幅が広がります。
# MAGIC 
# MAGIC - [Databricksにおけるモデルサービング](https://qiita.com/taka_yayoi/items/b5a5f83beb4c532cf921)
# MAGIC - [Databricksのサーバレスリアルタイム推論エンドポイントを使ってみる](https://qiita.com/taka_yayoi/items/ef2fb0856c70ced0e57a)
# MAGIC 
# MAGIC このノートブックではモデルサービングを有効化し、Streamlitで構築したGUIから機械学習モデルを呼び出してみます。
# MAGIC 
# MAGIC [StreamlitからDatabricksでサービングしている機械学習モデルを呼び出す](https://qiita.com/taka_yayoi/items/ce79c29df59a99bf872b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境設定

# COMMAND ----------

# MAGIC %run "../99_config"

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルサービングの有効化
# MAGIC 
# MAGIC 1. 前のステップで登録したモデルにアクセスし、**サービング**タブをクリックします。
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/yayoi/serving_1.png" width=70%>
# MAGIC 
# MAGIC 1. **サービングを有効化**をクリックします。
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/yayoi/serving_2.png" width=70%>
# MAGIC 
# MAGIC 1. プロダクションのモデルが**Ready**になるまで待ちます。これで、REST API経由でモデルを呼び出せる様になりました。右側に表示されるURL`https://.../model/.../Production/invocations`をメモしておきます。
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/yayoi/serving_3.png" width=70%>

# COMMAND ----------

# MAGIC %md
# MAGIC ## パーソナルアクセストークンの取得
# MAGIC 
# MAGIC REST APIを呼び出す際にはパーソナルアクセストークンが必要となります。
# MAGIC 
# MAGIC 1. 画面右上のユーザー名をクリックし、**ユーザー設定**を選択します。
# MAGIC 1. **新規トークンを作成**をクリックし、表示されるトークンをメモしておきます。
# MAGIC ![](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F1168882%2F8109f057-98ea-2f1d-86f3-f0457a1cff26.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=ecbeb0e8383b13f7c41770bbfdcedee9)
# MAGIC 
# MAGIC **注意**
# MAGIC パーソナルアクセストークンは厳重に管理してください。第三者に教えたりしないでください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## テストデータの準備
# MAGIC 
# MAGIC 特徴量ストアからテストデータを取得します。

# COMMAND ----------

from databricks.feature_store import FeatureLookup, FeatureStoreClient

fs = FeatureStoreClient()

# 特徴量ストアからテスト用データを取得
df_test = fs.read_table(name=f"{team_name}_hackathon.numeric_features_test")

# カテゴリー特徴量の検索条件を指定
feature_lookups_test = [
    FeatureLookup(
        table_name=f"{team_name}_hackathon.categorical_features_test", lookup_key="id"
    )
]

# テストデータセットに結合したい情報を指定
test_set = fs.create_training_set(
    df_test, feature_lookups=feature_lookups_test, label=None
)

# 実データをインスタンス化して、pandas形式に変換
df_test = test_set.load_df().toPandas()

# id列を除外
df_test = df_test.iloc[:, 1:]

display(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルのエンドポイントにリクエストするためには、Databricksのトークンが必要です。(右上のプロファイルアイコンの下の)User Settingページでトークンを生成することができます。
# MAGIC 
# MAGIC 今回は簡便な方法をとりますが、本来はトークンなど機密性の高い情報はノートブックに記述すべきではありません。シークレットに保存するようにしてください。
# MAGIC 
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

#import os

# 事前にCLIでシークレットにトークンを登録しておきます
#token = dbutils.secrets.get("demo-token-takaaki.yayoi", "token")

#os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def process_input(dataset):
  if isinstance(dataset, pd.DataFrame):
    return {"dataframe_split": dataset.to_dict(orient='split') }
  elif isinstance(dataset, str):
    return dataset
  else:
    return create_tf_serving_json(dataset)

def score_model(dataset):
  
  # パーソナルアクセストークン
  # for production
  #token = os.environ.get("DATABRICKS_TOKEN")
  # for hackathon
  # 上のステップで取得したパーソナルアクセストークンを指定してください
  token = "<パーソナルアクセストークン>"
  
  # モデルエンドポイントのURL
  url = '<モデルエンドポイントのURL>'
  headers = {'Authorization': f'Bearer {token}'}
  data_json = process_input(dataset)
  
  #print(data_json)
  
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# 動作確認
num_predictions = 1
served_predictions = score_model(df_test[:num_predictions])
served_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streamlitの設定
# MAGIC 
# MAGIC SreamlitはPythonで実行できるWebアプリケーション開発のフレームワークです。Webサーバ機能や開発フレームワークがオールインワンでパッケージされていて、PythonだけでWebアプリケーションが簡単に公開できます。
# MAGIC 
# MAGIC - [【Streamlit】インストールしてみた \- Qiita](https://qiita.com/tanktop-kun/items/1be2f24b9a38c76fec95)
# MAGIC - [Python Streamlitを簡単に使ってみる \- Qiita](https://qiita.com/Nao_Ishimatsu/items/2f272b6ad34695600be7)
# MAGIC - [Streamlit、Python、GridDBを使ってインタラクティブなダッシュボードを作成する \- Qiita](https://qiita.com/GridDBnet/items/2f03024c8b10f327d5ca)
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/challenge.png)
# MAGIC 
# MAGIC 1. Streamlitがインストールされているマシンで、以下のセルのコードを`app.py`として保存してください。
# MAGIC 1. `app.py`ファイルを開き以下のパラメーターを指定してください。
# MAGIC     - モデルのURL
# MAGIC     - パーソナルアクセストークン
# MAGIC 1. コマンドプロンプト(ターミナル)で以下のコマンドを実行してください。ブラウザで画面が表示されるはずです。
# MAGIC     ```
# MAGIC     cd <app.pyの配置されているディレクトリ>
# MAGIC     streamlit run app.py
# MAGIC     ```
# MAGIC 1. 画面を操作してモデルを呼び出してみてください。

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

st.header('宿泊価格予測モデル')
st.write('![](https://sajpstorage.blob.core.windows.net/yayoi/hotel_image.png)')
st.write('''
- [Databricksにおける機械学習モデル構築のエンドツーエンドのサンプル \- Qiita](https://qiita.com/taka_yayoi/items/f48ccd35e0452611d81b)
- [【練習問題】民泊サービスの宿泊価格予測 \| SIGNATE \- Data Science Competition](https://signate.jp/competitions/266)'
''')

# Copy and paste this code from the MLflow real-time inference UI. Make sure to save Bearer token from 
def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def process_input(dataset):
  if isinstance(dataset, pd.DataFrame):
    return {"dataframe_split": dataset.to_dict(orient='split') }
  elif isinstance(dataset, str):
    return dataset
  else:
    return create_tf_serving_json(dataset)

def score_model(dataset):
  # 1. パーソナルアクセストークンを設定してください
  # 今回はハッカソンのため平文で記載していますが、実際に使用する際には環境変数経由で取得する様にしてください。
  token = "<パーソナルアクセストークン>"
  #token = os.environ.get("DATABRICKS_TOKEN")

  # 2. モデルエンドポイントのURLを設定してください
  url = '<モデルエンドポイントのURL>'
  headers = {'Authorization': f'Bearer {token}'}
  
  data_json = process_input(dataset)
   
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

st.subheader('宿泊価格を予測したい物件の情報')

accommodates = st.slider("収容可能人数", 1, 10, 1)
bathrooms = st.slider("バスルームの数", 0, 5, 1)
bedrooms = st.slider("ベッドルームの数", 1, 5, 1)
beds = st.slider("ベッドの数", 1, 10, 1)
host_response_rate = st.slider("ホストの反応率(%)", 0, 100, 50)
reviews = st.slider("レビューの数", 0, 20, 1)
review_score = st.slider("レビューのスコア", 0, 100, 50)

parameter_df = pd.DataFrame(
    data={'accommodates': accommodates, 
          'bathrooms': bathrooms, 
          'bedrooms': bedrooms,
          'beds': beds,
          'host_response_rate': host_response_rate,
          'number_of_reviews': reviews,
          'review_scores_rating': review_score,
          'bed_type_label': 0,
          'cancellation_policy_label': 0,
          'city_label': 0,
          'cleaning_fee_label': 0,
          'instant_bookable_label': 0,
          'property_type_label': 0,
          'room_type_label': 0},index=[0]
)

response = score_model(parameter_df)
response_df = pd.DataFrame(response) 
parameter_df['predictions'] = response_df

estimated_accommodation_fee = int(parameter_df['predictions'][0])

st.subheader('モデルの予測結果')
st.metric(label="この物件の宿泊価格", value=f"{estimated_accommodation_fee}ドル")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC お疲れ様でした！
