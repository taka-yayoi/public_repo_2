# Databricks notebook source
# MAGIC %md
# MAGIC # プロダクション環境での推論にモデルを活用
# MAGIC
# MAGIC これまでのノートブックで、ディープラーニングモデルをトレーニングし、モデルレジストリを用いてデプロイを行いました。ここでは、推論でモデルをどのように活用するのかを見ていきます。
# MAGIC
# MAGIC 最初のステップでは、MLflowのリポジトリからモデルをダウンロードする必要があります。

# COMMAND ----------

import os
import torch
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

model_name = "cv_pcb_classification_taka" # 適宜変更


local_path = ModelsArtifactRepository(
    f"models:/{model_name}/Production"
).download_artifacts(
    ""
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## PCB画像の分類
# MAGIC
# MAGIC PCB画像を分類するために使用するUDF関数を作成します。

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator
from io import BytesIO
from PIL import Image
from torchvision.models import ViT_B_16_Weights
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_model = torch.load(
    local_path + "data/model.pth", map_location=torch.device(device)
)

weights = ViT_B_16_Weights.DEFAULT
feature_extractor = weights.transforms()

feature_extractor_b = sc.broadcast(feature_extractor)
model_b = sc.broadcast(loaded_model)

@pandas_udf("struct<score: float, label: int, labelName: string>")
def apply_vit(images_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:

    model = model_b.value
    feature_extractor = feature_extractor_b.value
    model = model.to(torch.device("cuda"))
    model.eval()
    id2label = {0: "normal", 1: "anomaly"}
    with torch.set_grad_enabled(False):
        for images in images_iter:
            pil_images = torch.stack(
                [
                    feature_extractor(Image.open(BytesIO(b)).convert("RGB"))
                    for b in images
                ]
            )
            pil_images = pil_images.to(torch.device(device))
            outputs = model(pil_images)
            preds = torch.max(outputs, 1)[1].tolist()
            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
            yield pd.DataFrame(
                [
                    {"score": prob, "label": pred, "labelName": id2label[pred]}
                    for pred, prob in zip(preds, probs)
                ]
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## バッチサイズの設定
# MAGIC
# MAGIC SparkのデータパーティションがArrowのレコードバッチに変換される際に、`maxRecordsPerBatch`パラメータを用いてバッチサイズを64に設定しましょう。

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 64)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測テーブル
# MAGIC
# MAGIC これで、すべての画像に対する予測結果を持つ新規テーブルを計算することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC USE takaakiyayoi_catalog.pcb;

# COMMAND ----------

spark.sql("drop table IF EXISTS circuit_board_prediction")
spark.table("circuit_board_gold").withColumn(
    "prediction", apply_vit("content")
).write.saveAsTable("circuit_board_prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 誤ってラベル付けされた画像を表示
# MAGIC
# MAGIC シンプルなSQLで誤ったラベルを持つ画像を表示します。

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   circuit_board_prediction
# MAGIC where
# MAGIC   labelName != prediction.labelName

# COMMAND ----------

# MAGIC %md
# MAGIC ### RESTサーバレスリアルタイム推論エンドポイントにモデルをデプロイ
# MAGIC
# MAGIC RESTサーバレスリアルタイム推論エンドポイントにモデルをデプロイしましょう。
# MAGIC
# MAGIC しかし、最初に、base64画像を入力として受け入れられるように、ラッパーモデルを作成し、MLflowに公開しましょう。

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
import base64
from PIL import Image
import io
import mlflow
from io import BytesIO

from torchvision.models import ViT_B_16_Weights



class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # 評価モードでモデルのインスタンスを作成
        model.to(torch.device("cpu"))
        self.model = model.eval()
        weights = ViT_B_16_Weights.DEFAULT
        self.feature_extractor = weights.transforms()

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          id2label = {0: "normal", 1: "anomaly"}
          pil_images = torch.stack(
              [
                  self.feature_extractor(
                      Image.open(BytesIO(base64.b64decode(row[0]))).convert("RGB")
                  )
                  for _, row in images.iterrows()
              ]
          )
          pil_images = pil_images.to(torch.device("cpu"))
          outputs = self.model(pil_images)
          preds = torch.max(outputs, 1)[1]
          probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
          labels = [id2label[pred] for pred in preds.tolist()]

          return pd.DataFrame( data=dict(
            score=probs,
            label=preds,
            labelName=labels)
          )

# COMMAND ----------

loaded_model = torch.load(
    local_path + "data/model.pth", map_location=torch.device("cpu")
)
wrapper = CVModelWrapper(loaded_model)
images = spark.table("circuit_board_gold").take(25)

b64image1 = base64.b64encode(images[0]["content"]).decode("ascii")
b64image2 = base64.b64encode(images[1]["content"]).decode("ascii")
b64image3 = base64.b64encode(images[3]["content"]).decode("ascii")
b64image4 = base64.b64encode(images[4]["content"]).decode("ascii")
b64image24 = base64.b64encode(images[24]["content"]).decode("ascii")

df_input = pd.DataFrame(
    [b64image1, b64image2, b64image3, b64image4, b64image24], columns=["data"]
)
df = wrapper.predict("", df_input)
display(df)

# COMMAND ----------

username = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
mlflow.set_experiment("/Users/{}/pcbqi".format(username))
model_name = "cv_pcb_classification_rt_taka" # 適宜変更
with mlflow.start_run(run_name=model_name) as run:
    mlflowModel = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapper,
        input_example=df_input,
        registered_model_name=model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0].version
client.transition_model_version_stage(
    name=model_name, version=latest_version, stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC これで新たなモデルをサーバレスリアルタイムサービングエンドポイントにデプロイすることができます。
# MAGIC
# MAGIC サンプルを用いて"send request"をクリックします。
# MAGIC
# MAGIC <img width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/serving.png">

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # 結論
# MAGIC
# MAGIC これですべてです！データセットをインクリメンタルに取り込み、クレンジングし、ディープラーニングモデルをトレーニングするエンドツーエンドのパイプラインを構築しました。プロダクションレベルのパイプラインとモデルはデプロイされ、活用できる状態です。
# MAGIC
# MAGIC Databricksレイクハウスは、皆様のチームのスピードを改善し、プロダクションへの移行をシンプルにします:
# MAGIC
# MAGIC * Auto Loaderによるユニークなデータ取り込み、データ準備機能は誰でもデータエンジニアリングにアクセスできるようにします
# MAGIC * 構造化データ、非構造化データを取り込み、処理できることで、すべてのユースケースをサポートします
# MAGIC * MLトレーニングのための高度なML機能
# MAGIC * データサイエンティストがオペレーションのタスクではなく、(皆様のビジネスを改善するために)本当に問題になっていることにフォーカスできるようにするMLOpsのカバレッジ
# MAGIC * 外部ツールなしに、あなたのすべてのユースケースをカバーするように、すべてのタイプのプロダクションデプロイメントをサポート
# MAGIC * データセキュリティからモデルガバナンスに至る全てをカバーするセキュリティとコンプライアンス
# MAGIC
# MAGIC これによって、Databricksを活用しているチームは、データの取り込みからモデルのデプロイメントに至る高度なMLプロジェクトを数週間でプロダクションに到達させることが可能となり、ビジネスを劇的に加速させます。
