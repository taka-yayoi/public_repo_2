import streamlit as st 
import numpy as np 
import json
import requests

st.title('Databricks Q&A bot')
#st.header('Databricks Q&A bot')

def generate_answer(question):
  # Driver Proxyと異なるクラスター、ローカルからDriver Proxyにアクセスする際にはパーソナルアクセストークンを設定してください
  token = "" 
  url = "http://127.0.0.1:7777/"

  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "prompt": question
  }

  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code != 200:
    raise Exception(
       f"Request failed with status {response.status_code}, {response.text}"
    )
  
  response_json = response.json()
  return response_json

question = st.text_input("**質問**")

if question != "":
  response = generate_answer(question)

  answer = response["answer"]
  source = response["source"]

  st.write(f"**回答:** {answer}")
  st.write(f"**ソース:** [{source}]({source})")