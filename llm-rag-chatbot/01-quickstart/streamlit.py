import streamlit as st 
import numpy as np 
import json
import requests
import pandas as pd

from databricks.sdk.runtime import dbutils

st.title('Databricks Q&A bot')
#st.header('Databricks Q&A bot')

def generate_answer(question):
  # Driver Proxyと異なるクラスター、ローカルからDriver Proxyにアクセスする際にはパーソナルアクセストークンを設定してください
  token = dbutils.secrets.get("demo-token-takaaki.yayoi", "rag_sp_token")
  url = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/dbdemos_endpoint_takaakiyayoi_catalog_rag_chatbot/invocations"

  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "query": question
  }

  prompt = pd.DataFrame(
        {"query": [question]}
    )
  #print(prompt)
  ds_dict = {"dataframe_split": prompt.to_dict(orient="split")}

  response = requests.post(url, headers=headers, data=json.dumps(ds_dict))
  if response.status_code != 200:
    raise Exception(
       f"Request failed with status {response.status_code}, {response.text}"
    )
  
  response_json = response.json()
  #print(response_json)
  return response_json


if "messages" not in st.session_state:
    st.session_state.messages = []

# アプリの再実行の際に履歴のチャットメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力に対する反応
if prompt := st.chat_input("Databricksに関して何を知りたいですか？"):
    # チャットメッセージコンテナにユーザーメッセージを表示
    st.chat_message("user").markdown(prompt)
    # チャット履歴にユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('回答を生成中...'):
      bot_response = generate_answer(prompt)
      #answer = bot_response["answer"]
      answer = bot_response["predictions"][0]
      #source = bot_response["source"]

      response = f"""{answer}"""

#**ソース:** {source}"""

    # チャットメッセージコンテナにアシスタントのレスポンスを表示
    with st.chat_message("assistant"):
      st.markdown(response)

    # チャット履歴にアシスタントのレスポンスを追加
    st.session_state.messages.append({"role": "assistant", "content": response})


#with st.chat_message("user"):
#    st.write("Hello 👋")

#question = st.text_input("**質問**")

#if question != "":
#  response = generate_answer(question)

#  answer = response["answer"]
#  source = response["source"]

#  st.write(f"**回答:** {answer}")
#  st.write(f"**ソース:** [{source}]({source})")