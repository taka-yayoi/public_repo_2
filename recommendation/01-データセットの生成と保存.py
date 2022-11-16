# Databricks notebook source
# MAGIC %md
# MAGIC # データセットの生成とDeltaテーブルへの保存
# MAGIC 
# MAGIC このノートブックでは5つのDeltaテーブルを生成します:
# MAGIC 
# MAGIC * `user_profile`: user_id と静的なプロファイル
# MAGIC * `item_profile`: item_id と静的なプロファイル
# MAGIC * `user_item_interaction`: ユーザーとアイテムのインタラクションが発生するイベント
# MAGIC   * このテーブルはモデルトレーニングと評価のために3つのテーブルに分割されます: `train`, `val`, `test`
# MAGIC 
# MAGIC シンプルにするために、ユーザーとアイテムのプロファイルは2つの属性のみを持ちます: 年齢とトピックです。`user_age`カラムはユーザーの年齢であり、`item_age`カラムはアイテムとインタラクションしたユーザーの平均年齢です。`user_topic`カラムはユーザーが好きなトピックであり、`item_topic`カラムはアイテムに最も適したトピックとなります。また、これもシンプルにするために、`user_item_interaction`テーブルでは、イベントのタイムスタンプは無視して`user_id`と`item_id`と、ユーザーがアイテムとインタラクションしたかどうかを表現する2値ラベルのカラムを持ちます。
# MAGIC 
# MAGIC ## どの様にラベルを計算するのか
# MAGIC 
# MAGIC このノートブックでは、ユーザーがアイテムとインタラクションしたのかどうかを表現するラベルをランダムに割り当てます。ラベルは、年齢とトピックの属性によって決定されるユーザーとアイテムの類似性に基づいています。
# MAGIC 
# MAGIC この計算処理は、ユーザーを3つの年齢レンジ(under 18, 18-34, 35-60)に分割します。`user_age`と`item_age`が同じレンジに属する場合、インタラクションの確率が高くなります。
# MAGIC 
# MAGIC * 同じ年齢レンジ: P(interact_age) = 0.3
# MAGIC * 異なる年齢レンジ: P(interact_age) = 0.1
# MAGIC 
# MAGIC トピックは10個のカテゴリーから構成されます。この計算処理では、トピック(1,2,4)、(3,6,9)、(0,5,7,8)のグループでそれぞれが関連していると仮定しています。
# MAGIC 
# MAGIC * 関連トピック: P(interact_topic) = 0.3
# MAGIC * 異なるトピック: P(interact_topic) = 0.05
# MAGIC 
# MAGIC ユーザーがアイテムとインタラクションする全体的な確率は P(interact) = P(interact_age OR interact_topic) となります。このノートブックでは、この確率に基づいてラベルをランダムに生成します。

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split   

from pyspark.sql.functions import *

# COMMAND ----------

NUM_USERS = 400
NUM_ITEMS = 2000
NUM_INTERACTIONS = 4000
NUM_TOPICS = 10
MAX_AGE = 60

DATA_DBFS_ROOT_DIR = '/tmp/recommender/data'

def export_pd_in_delta(pdf, name):
  spark.createDataFrame(pdf).write.format("delta").mode("overwrite").save(f"{DATA_DBFS_ROOT_DIR}/{name}")

# COMMAND ----------

# MAGIC %md # 特徴量の生成

# COMMAND ----------

user_pdf = pd.DataFrame({
  "user_id": [i for i in range(NUM_USERS)],
  "user_age": [np.random.randint(MAX_AGE) for _ in range(NUM_USERS)],
  "user_topic": [np.random.randint(NUM_TOPICS) for _ in range(NUM_USERS)],
})

# COMMAND ----------

item_pdf = pd.DataFrame({
  "item_id": [i for i in range(NUM_ITEMS)],
  "item_age": [np.random.randint(MAX_AGE) for _ in range(NUM_ITEMS)],
  "item_topic": [np.random.randint(NUM_TOPICS) for _ in range(NUM_ITEMS)],
})

# COMMAND ----------

export_pd_in_delta(item_pdf, "item_profile")
export_pd_in_delta(user_pdf, "user_profile")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/recommender/data

# COMMAND ----------

# MAGIC %md
# MAGIC # ラベルの生成

# COMMAND ----------

user_id = [np.random.randint(NUM_USERS) for _ in range(NUM_INTERACTIONS)]
item_id = [np.random.randint(NUM_ITEMS) for _ in range(NUM_INTERACTIONS)]
pdf = pd.DataFrame({"user_id": user_id, "item_id": item_id})

# COMMAND ----------

all_pdf = pdf \
    .set_index('item_id') \
    .join(item_pdf.set_index('item_id'), rsuffix='_it').reset_index() \
    .set_index('user_id') \
    .join(user_pdf.set_index('user_id'), rsuffix='_us').reset_index()

# COMMAND ----------

display(all_pdf)

# COMMAND ----------

def get_range(age):
  # <18, 18-34, 35-60
  if age < 18:
    return 0
  if age < 35:
    return 1
  return 2

#  (1,2,4) が関連、(3,6,9)が関連、(0,5,7,8)が関連
d = {1:0, 2:0, 4:0, 3:1, 6:1, 9:1, 0:2, 5:2, 7:2, 8:2}
  
def calc_clicked(ad_age, ad_topic, disp_age, disp_topic):
  if get_range(ad_age) == get_range(disp_age):
    age_not_click = 0.7
  else:
    age_not_click = 0.9
  if d[ad_topic] == d[disp_topic]:
    disp_not_click = 0.7
  else:
    disp_not_click = 0.95
  overall_click = 1 - age_not_click * disp_not_click
  return 1 if np.random.rand() < overall_click else 0

# COMMAND ----------

all_pdf['label'] = all_pdf.apply(lambda row: calc_clicked(
  row['item_age'], row['item_topic'], row['user_age'], row['user_topic']
), axis=1)

# COMMAND ----------

export_pdf = all_pdf[['user_id', 'item_id', 'label']]
display(export_pdf)

# COMMAND ----------

export_pdf.groupby(['user_id']).sum().describe()[['label']]

# COMMAND ----------

export_pdf.groupby(['user_id', 'item_id']).sum().describe()

# COMMAND ----------

train, test = train_test_split(export_pdf, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
export_pd_in_delta(train, "user_item_interaction_train")
export_pd_in_delta(val, "user_item_interaction_val")
export_pd_in_delta(test, "user_item_interaction_test")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/recommender/data

# COMMAND ----------


