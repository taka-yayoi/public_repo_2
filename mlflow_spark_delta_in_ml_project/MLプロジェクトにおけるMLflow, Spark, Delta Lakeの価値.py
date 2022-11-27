# Databricks notebook source
# MAGIC %md
# MAGIC **自己紹介**
# MAGIC 
# MAGIC ![](https://media-exp1.licdn.com/dms/image/C5603AQGephOQ9YaqEQ/profile-displayphoto-shrink_800_800/0/1608023126792?e=1674691200&v=beta&t=7TD0kwDHdccvAOOGVaYVnnmrSNhvLDPLlgTOmRjJoeo)
# MAGIC <br><br>
# MAGIC - 弥生 隆明(Taka Yayoi)
# MAGIC - <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140"> シニアソリューションアーキテクト
# MAGIC   - 2020年からデータブリックス ジャパンにおいて、プレセールスやPOCに従事
# MAGIC   - 前職はコンサルティングファーム、総合電機メーカーにてデータ分析・Webサービス構築などに従事。インド赴任経験あり。
# MAGIC   - 前職まではJupyter notebookを使って機械学習モデル構築を行なっていました。

# COMMAND ----------

# MAGIC %md
# MAGIC # MLプロジェクトにおけるMLflow/Spark/Delta Lakeの価値
# MAGIC 
# MAGIC 機械学習(ML)モデルを構築・運用するMLプロジェクトにおいて、どのように[Apache Spark](https://www.databricks.com/jp/glossary/apache-spark-as-a-service)や[Delta Lake](https://delta.io/)、[MLflow](https://mlflow.org/)を活用するのかを説明します。
# MAGIC 
# MAGIC **[Apache Spark](https://qiita.com/taka_yayoi/items/bf5fb09a0108aa14770b)とは？**
# MAGIC 
# MAGIC <img width=200 src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/512px-Apache_Spark_logo.svg.png?20210416091439">
# MAGIC 
# MAGIC Apache Sparkは、大規模なデータの高速リアルタイム処理を実現するオープンソースのクラスタコンピューティングフレームワークです。大量なデータを並列で処理することで、非常に高いパフォーマンスを発揮することができます。データ加工だけでなく、機械学習モデルのトレーニングやハイパーパラメーターチューニングを並列処理することが可能です。
# MAGIC 
# MAGIC **[MLflow](https://qiita.com/taka_yayoi/items/1a4e82f7e20c56ba4f72)とは？**
# MAGIC 
# MAGIC <img width=200 src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" title="MLflow Documentation — MLflow 1.15.0 documentation">
# MAGIC 
# MAGIC 機械学習モデルのライフサイクル管理のためのフレームワークを提供するソフトウェアです。機械学習のトラッキング、集中管理のためのモデルレジストリといった機能を提供します。Databricksでは、マネージドサービスとしてMLflowを利用できる様になっていますので、Databricksノートブック上でトレーニングした機械学習は自動でトラッキングされます。
# MAGIC 
# MAGIC **[Delta Lake](https://qiita.com/taka_yayoi/items/345f503d5f8177084f24)とは？**
# MAGIC 
# MAGIC <img width=200 src="https://docs.delta.io/latest/_static/delta-lake-logo.png">
# MAGIC 
# MAGIC データレイクに格納されているデータに対して高速なデータ処理、強力なデータガバナンスを提供するストレージレイヤーソフトウェアです。ACIDトランザクションやデータのバージョン管理、インデックス作成機能などを提供します。機械学習の文脈ではデータのバージョン管理が重要な意味を持つことになります。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/databricks_oss.png)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2022/11/26</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>11.3ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLプロジェクトとは？
# MAGIC 
# MAGIC MLプロジェクトは機械学習(ML)モデルを構築することが目的ではなく、ビジネス課題を解決するために立ち上げるのが一般的です。
# MAGIC ![](https://www.databricks.com/jp/wp-content/uploads/2021/06/ibjp-big-data-img-1.png)
# MAGIC 
# MAGIC MLプロジェクトの一般的なフローを示します。ここでは赤いボックスにフォーカスします。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ml_project_overview.png)
# MAGIC 
# MAGIC このノートブックでは以下のステップをウォークスルーします。
# MAGIC </p>
# MAGIC 
# MAGIC 1. データのインポート
# MAGIC 1. Seabornとmatplotlibによるデータの可視化
# MAGIC 1. ベースラインのトレーニング(MLflow)
# MAGIC 1. データパス、バージョンの確認(Delta, MLflow)
# MAGIC 1. 特徴量エンジニアリング(Spark, Delta)
# MAGIC 1. 機械学習モデルをトレーニングする際に用いるハイパーパラメーター探索を並列で実行(Spark, MLflow)
# MAGIC 1. ハイパーパラメーター探索結果をMLflowで確認(MLflow)
# MAGIC 1. MLflowにベストモデルを登録(MLflow)
# MAGIC 1. 登録済みモデルをSpark UDFとして別のデータセットに適用(Spark, MLflow)
# MAGIC 1. 低レーテンシーリクエストに対応するためのモデルサービングの実行(MLflow)
# MAGIC 
# MAGIC この例では、ワインの性質に基づいて、ポルトガルの"Vinho Verde"ワインの品質を予測するモデルを構築します。
# MAGIC 
# MAGIC この例では、UCI機械学習リポジトリのデータ[*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009]を活用します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowによる機械学習モデルのライフサイクル管理
# MAGIC 
# MAGIC MLflowを活用することで、データサイエンティストやMLエンジニアによる機械学習モデルの実験段階から本格運用までをサポートすることができます。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/mlflow_lifecycle.png)

# COMMAND ----------

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# ファイル格納パス
work_path = f"dbfs:/tmp/databricks_handson/{username}"
# Delta Lakeテーブルパス
DELTA_TABLE_DEFAULT_PATH = f"{work_path}/data.delta"

# データベース名
database_name = "takaakiyayoi_db"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
spark.sql(f"USE {database_name}")

# モデル名
model_name = f"wine_quality_{username}"

# パスとモデル名を表示
print(f"database_name: {database_name}")
print(f"table_path_name: {DELTA_TABLE_DEFAULT_PATH}")
print(f"model_name: {model_name}")

# COMMAND ----------

# MAGIC %md ## データのインポート
# MAGIC 
# MAGIC このセクションでは、サンプルデータからpandasデータフレームにデータを読み込みます。

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md
# MAGIC ワインが赤ワインか白ワインかを示す"is_red"カラムを追加して、2つのデータフレームを1つのデータセットにマージします。

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# カラム名から空白を削除
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC データの中身を確認します。Databricksでは`display`関数を用いることで、簡単にデータを可視化することができます。

# COMMAND ----------

# 中身を確認します
display(data)

# COMMAND ----------

# MAGIC %md ## データの可視化
# MAGIC 
# MAGIC モデルをトレーニングする前に、Seaborn、matplotlibを用いてデータを可視化します。お使いの可視化ライブラリを活用できることに加え、Databricksではデータの傾向把握を支援する機能を提供しています。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 目的変数のqualityのヒストグラムをプロットします。

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC qualityは3から9に正規分布しているように見えます。
# MAGIC 
# MAGIC quality >= 7のワインを高品質と定義します。

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md 
# MAGIC 特徴量と2値ラベルの間の相関を見るにはボックスプロットが有用です。

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
    if col == "is_red" or col == "quality":
        continue  # カテゴリ変数にボックスプロットは使用できません
    sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
    axis_j += 1
    if axis_j == dims[1]:
        axis_i += 1
        axis_j = 0

# COMMAND ----------

# MAGIC %md 上のボックスプロットから、いくつかの変数がqualityに対する単変量予測子として優れていることがわかります。
# MAGIC <br><br>
# MAGIC - alcoholのボックスプロットにおいては、高品質ワインのアルコール含有量の中央値は、低品質のワインの75%パーセンタイルよりも大きな値となっています。
# MAGIC - densityのボックスプロットにおいては、低品質ワインの密度は高品質ワインよりも高い値を示しています。密度は品質と負の相関があります。

# COMMAND ----------

# MAGIC %md ## データの前処理
# MAGIC 
# MAGIC モデルのトレーニングの前に、欠損値のチェックを行い、データをトレーニングデータとバリデーションデータに分割します。

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md 欠損値はありませんでした。このようなデータの欠損値や統計は、`dbutils.data.summarize`や`display`関数の結果からもアクセスすることができます。
# MAGIC 
# MAGIC `dbulits`は[Databricksユーティリティ](https://qiita.com/taka_yayoi/items/3717623187859809515d)であり、これ以外にもファイルシステムの操作やウィジェットの追加など便利な機能を提供しています。

# COMMAND ----------

dbutils.data.summarize(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lakeにデータを保存
# MAGIC 
# MAGIC データを[Delta Lake](https://qiita.com/taka_yayoi/items/6b423cca1f25d424a908)に保存します。機械学習モデルをトレーニングする際、**どの時点のデータを使ったのか**という情報は、実験の再現性確保の観点でも重要となりますが、Delta Lakeでデータを管理することで、データのバージョン管理機能(タイムトラベル)を活用できる様になります。さらにMLflowと組み合わせることで、機械学習モデルのトレーニングに使用したデータのパスやバージョンを記録することができるので、容易に実験を再現できる様になります。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/time_travel.png)
# MAGIC 
# MAGIC [今日の機械学習と明日の効率的な機械学習の間のDelta](https://qiita.com/taka_yayoi/items/9170d290864aa45146e9)

# COMMAND ----------

# データをリセット
dbutils.fs.rm(DELTA_TABLE_DEFAULT_PATH, True)
spark.sql("DROP TABLE IF EXISTS wine")

sdf = spark.createDataFrame(data)
# Delta Lake形式でデータを保存
sdf.write.format("delta").mode("overwrite").save(DELTA_TABLE_DEFAULT_PATH)
# SQLでアクセスできる様にメタストアに登録
spark.sql("CREATE TABLE wine USING DELTA LOCATION '" + DELTA_TABLE_DEFAULT_PATH + "'")

# COMMAND ----------

# MAGIC %md
# MAGIC Deltaテーブルに対する変更はすべて記録されます。

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY wine;

# COMMAND ----------

# MAGIC %md ## ベースラインモデルの構築
# MAGIC 
# MAGIC 出力が2値であり、複数の変数間での相互関係がある可能性があることから、このタスクにはランダムフォレスト分類器が適しているように見えます。
# MAGIC 
# MAGIC 以下のコードでは、scikit-learnを用いてシンプルな分類器を構築します。モデルの精度を追跡するためにMLflowを用い、後ほど利用するためにモデルを保存します。この際には以下のデータがMLflowによって記録されます。
# MAGIC 
# MAGIC - 機械学習モデル本体
# MAGIC - ハイパーパラメーター
# MAGIC - モデルの精度指標(メトリクス)
# MAGIC - トレーニングに使用したデータに関する情報(パス、バージョン)
# MAGIC 
# MAGIC MLflowでは1つの機械学習トレーニングを**ラン**という単位で管理し、複数のランを**エクスペリメント**という単位で管理します。ここでは、特定のランに上記の情報が記録されます。
# MAGIC 
# MAGIC データに関する情報を自動で記録される様にするには、[mlflow.spark.autolog()](https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.autolog)を呼び出します。

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# sklearnのRandomForestClassifierのpredictメソッドは、2値の分類結果(0、1)を返却します。
# 以下のコードでは、それぞれのクラスに属する確率を返却するpredict_probaを用いる、ラッパー関数SklearnModelWrapperを構築します。
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# data_version, data_pathを含むパラメーターを自動で記録するためにautolog APIを使います
mlflow.spark.autolog()
  
# mlflow.start_runは、このモデルのパフォーマンスを追跡するための新規MLflowランを生成します。
# コンテキスト内で、使用されたパラメーターを追跡するためにmlflow.log_param、精度のようなメトリクスを追跡するために
# mlflow.log_metricを呼び出します。
with mlflow.start_run(run_name='untuned_random_forest') as run1:
  from sklearn.model_selection import train_test_split

  # Delta Lake形式のデータを読み込みます
  sdf = spark.read.format("delta").load(DELTA_TABLE_DEFAULT_PATH)

  # トレーニングデータセットとテスト用データセットを準備します
  data = sdf.toPandas() 
  train, test = train_test_split(data, random_state=123)
  X_train = train.drop(["quality"], axis=1)
  X_test = test.drop(["quality"], axis=1)
  y_train = train.quality
  y_test = test.quality
  
  # トレーニングを実施します
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_probaは[prob_negative, prob_positive]を返却するので、出力を[:, 1]でスライスします。
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # メトリックとしてROC曲線のAUCを使用します。
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  
  # モデルの入出力スキーマを定義するシグネチャをモデルとともに記録します。
  # モデルがデプロイされた際に、入力を検証するためにシグネチャが用いられます。
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflowにはモデルをサービングする際に用いられるconda環境を作成するユーティリティが含まれています。
  # 必要な依存関係がconda.yamlに保存され、モデルとともに記録されます。
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC データチェックとして、モデルによって出力される特徴量の重要度を確認します。

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC 先ほどボックスプロットで見たように、品質を予測するのにアルコールと密度が重要であることがわかります。

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowにROC曲線のAUCを記録しました。右上の**フラスコアイコン**をクリックして、エクスペリメントランのサイドバーを表示します。
# MAGIC 
# MAGIC このモデルはAUC0.89を達成しました。
# MAGIC 
# MAGIC ランダムな分類器のAUCは0.5となり、それよりAUCが高いほど優れていると言えます。詳細は、[Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)を参照ください。

# COMMAND ----------

# MAGIC %md #### MLflowモデルレジストリにモデルを登録
# MAGIC 
# MAGIC モデルレジストリにモデルを登録することで、Databricksのどこからでもモデルを容易に参照できるようになり、一貫性を持って機械学習モデルのステータスを管理できるようになります。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/model_registry.png)
# MAGIC 
# MAGIC 以下のセクションでは、どのようにプログラム上から操作をするのかを説明しますが、UIを用いてモデルを登録することもできます。"Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui))を参照ください。

# COMMAND ----------

# MAGIC %md
# MAGIC まず、MLflowのAPI`mlflow.search_runs`を用いて、ランに登録されたデータソース情報を確認します。

# COMMAND ----------

dataSourceInfo = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0]["tags.sparkDatasourceInfo"]

param_array = dataSourceInfo.split(",")
data_path = param_array[0]
data_version = param_array[1]
data_format = param_array[2]

print("このトレーニング(ラン)に記録されたデータソース情報")
print("データパス:", data_path)
print("データバージョン:", data_version)
print("データフォーマット:", data_format)

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、ランを特定するIDを取得します。

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# MAGIC %md
# MAGIC **注意**
# MAGIC 最初のセルで設定しているように、モデル名は `wine_quality_<ユーザー名から記号を除外したもの>` となります。

# COMMAND ----------

# モデルレジストリにモデルを登録します
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# COMMAND ----------

# モデルの説明文を追加します
client = mlflow.tracking.MlflowClient()
client.update_registered_model(name=model_name, description="""**ワイン品質予測モデル**

![](https://sajpstorage.blob.core.windows.net/demo20210903-ml/22243068_s.jpg)

- **特徴量** ワインの特性を示す特徴量
- **出力** ワインが高品質である確率
- **承認者** Taro Yamada
""")

# COMMAND ----------

# MAGIC %md 
# MAGIC Modelsページでモデルを確認できるはずです。Modelsページを表示するには、左のサイドバーでModelsアイコンをクリックします。
# MAGIC 
# MAGIC 次に、このモデルをproductionに移行し、モデルレジストリからモデルをこのノートブックにロードします。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Modelsページでは、モデルバージョンが`Production`ステージにあると表示されます。
# MAGIC 
# MAGIC これで、`models:/wine_quality/production`のパスでモデルを参照することができます。

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# サニティチェック: この結果はMLflowで記録されたAUCと一致すべきです
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## 新たなモデルを用いたエクスペリメント
# MAGIC 
# MAGIC ハイパーパラメーターチューニングを行わなくても、ベースラインのランダムフォレストモデルはうまく動きました。
# MAGIC 
# MAGIC 以下のコードでは、より精度の高いモデルをトレーニングするためにxgboostライブラリを使用します。HyperoptとSparkTrialsを用いて、複数のモデルを並列にトレーニングするために、ハイパーパラメーター探索を並列で処理します。上のコードと同様に、パラメーター設定、パフォーマンスをMLflowでトラッキングします。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/scaleup_scaleout.png)
# MAGIC 
# MAGIC **注意**
# MAGIC - 今回は特徴量エンジニアリングとハイパーパラメーターチューニングを一緒に行っていますが、検証のためには個々にトレーニングを実行し、精度を評価してください。
# MAGIC - 時間の都合上、`max_evals`を`4`にしていますが、実際にご利用いただく際にはパフォーマンスチューニングの効果を出すために十分に大きな値を指定してください。

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # トレーニングの再現性を確保するためにシードを設定します。
}

def train_model(params):
  # MLflowのオートロギングによって、ハイパーパラメーターとトレーニングしたモデルは自動的にMLflowに記録されます。
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # xgbが評価メトリクスを追跡できるようにテストセットを渡します。XGBoostは、評価メトリクスに改善が見られなくなった際にトレーニングを中止します。
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # fminがauc_scoreを最大化するようにlossに-1*auc_scoreを設定します。
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# 並列度が高いほどスピードを改善できますが、ハイパーパラメータの探索において最適とは言えません。
# max_evalsの平方根が並列度の妥当な値と言えます。
spark_trials = SparkTrials(parallelism=10)

# "xgboost_models"という親のランの子ランとして、それぞれのハイパーパラメーターの設定が記録されるようにMLflowランのコンテキスト内でfminを実行します。
with mlflow.start_run(run_name='xgboost_models'):
  # Delta Lake形式のデータを読み込みます
  sdf = spark.read.format("delta").load(DELTA_TABLE_DEFAULT_PATH)

  # トレーニングデータセットとテスト用データセットを準備します
  data = sdf.toPandas() 
  train, test = train_test_split(data, random_state=123)
  X_train = train.drop(["quality"], axis=1)
  X_test = test.drop(["quality"], axis=1)
  y_train = train.quality
  y_test = test.quality
  
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=4, # originally 96
    trials=spark_trials, 
    rstate=np.random.default_rng(123)
  )

# COMMAND ----------

# MAGIC %md #### MLflowを用いて結果を確認
# MAGIC 
# MAGIC Experiment Runsサイドバーを開いて、ランを参照します。メニューを表示するために、下向き矢印の隣にあるDateをクリックし`auc`を選択し、aucメトリックの順でランを並び替えます。一番高いaucは0.92となっています。ベースラインモデルを上回りました！
# MAGIC 
# MAGIC MLflowはそれぞれのランのパフォーマンスメトリクスとパラメーターをトラッキングします。Experiment Runsサイドバーの一番上にある右上向きの矢印アイコン<img src="https://docs.databricks.com/_static/images/icons/external-link.png"/>をクリックすることで、MLflowランの一覧に移動することができます。

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、どのようにハイパーパラメータの選択がAUCと相関しているのかを見てみましょう。"+"アイコンをクリックして、親のランを展開し、親以外の全てのランを選択し、"Compare"をクリックします。Parallel Coordinates Plotを選択します。
# MAGIC 
# MAGIC メトリックに対するパラメーターのインパクトを理解するために、Parallel Coordinates Plotは有用です。プロットの右上にあるピンクのスライダーをドラッグすることで、AUCの値のサブセット、対応するパラメーターの値をハイライトすることができます。以下のプロットでは、最も高いAUCの値をハイライトしています。
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC 
# MAGIC 最もパフォーマンスの良かったランの全てが、`reg_lambda`と`learning_rate`において低い値を示していることに注意してください。
# MAGIC 
# MAGIC これらのパラメーターに対してより低い値を探索するために、さらなるハイパーパラメーターチューニングを実行することもできますが、ここではシンプルにするために、そのステップをデモに含めていません。

# COMMAND ----------

# MAGIC %md 
# MAGIC それぞれのハイパーパラメーターの設定において生成されたモデルを記録するためにMLflowを用いました。以下のコードでは、最も高いパフォーマンスを示したランを検索し、モデルレジストリにモデルを登録します。

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### MLflowモデルレジストリのProductionステージにある`wine_quality`モデルを更新
# MAGIC 
# MAGIC はじめに、`wine_quality_<ユーザー名>`という名前でベースラインモデルをモデルレジストリに保存しました。さらに精度の高いモデルができましたので、`wine_quality_<ユーザー名>`を更新します。

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC 左のサイドバーで**Models**をクリックし、`wine_quality_<ユーザー名>`に二つのバージョンが存在することを確認します。
# MAGIC 
# MAGIC 以下のコードで新バージョンをproductionに移行します。

# COMMAND ----------

# 古いモデルバージョンをアーカイブします。
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# 新しいモデルバージョンをProductionに昇格します。
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md load_modelを呼び出すクライアントは新たなモデルを受け取ります。

# COMMAND ----------

# このコードは上の"ベースラインモデルの構築"と同じものです。新たなモデルを利用するためにクライアント側での変更は不要です！
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## MLflowとSparkによるバッチ推論
# MAGIC 
# MAGIC 新たなデータのコーパスに対してモデルを評価したいというシナリオは数多く存在します。例えば、新たなデータバッチを手に入れたり、同じデータコーパスに対して二つのモデルを比較することなどが考えられます。
# MAGIC 
# MAGIC 以下のコードでは、並列に処理を行うためにSparkを用い、Deltaテーブルに格納されたデータに対してモデルの評価を行います。

# COMMAND ----------

# 新たなデータコーパスをシミュレートするために、既存のX_trainデータをDeltaテーブルに保存します。
# 実際の環境では、本当に新たなデータバッチとなります。
spark_df = spark.createDataFrame(X_train)
# Deltaテーブルの保存先
table_path = f"{work_path}/delta/wine_data"
# すでにコンテンツが存在する場合には削除します
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md MLflowに記録された機械学習モデルをSparkのUDF(ユーザー定義関数)としてロードし、Deltaテーブルに適用できるようにします。pandasであればデータの各行に対して、予測処理を逐次実行しなくてはなりませんが、このようにすることで、Sparkの並列分散処理能力を活用して予測を分散処理し、大量データであっても高速に予測結果を得ることが可能になります。

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# 新規データをDeltaから読み込みます
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# 新規データにモデルを適用します
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# それぞれの行には予測結果が紐づけられています。
# xgboostの関数はデフォルトでは確率を出力せず、予測結果が[0, 1]に限定されないことに注意してください。
display(new_data)

# COMMAND ----------

# クリーンアップ
dbutils.fs.rm(table_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルサービング
# MAGIC 
# MAGIC 低レーテンシーでの予測を行うようにモデルを運用するためには、MLflowのモデルサービング([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving))を利用して、モデルをエンドポイントにデプロイします。
# MAGIC 
# MAGIC 以下のコードでは、どのようにREST APIを用いてデプロイしたモデルから予測結果を得るのかを説明しています。

# COMMAND ----------

# MAGIC %md
# MAGIC モデルのエンドポイントにリクエストするためには、Databricksのトークンが必要です。(右上のプロファイルアイコンの下の)User Settingページでトークンを生成することができます。
# MAGIC 
# MAGIC トークンなど機密性の高い情報はノートブックに記述すべきではありません。シークレットに保存するようにしてください。
# MAGIC 
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

import os

# 事前にCLIでシークレットにトークンを登録しておきます
token = dbutils.secrets.get("demo-token-takaaki.yayoi", "token")

os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------

# MAGIC %md
# MAGIC 左のサイドバーで**Models**をクリックし、登録されているワインモデルに移動します。servingタブをクリックし、**Enable Serving**をクリックします。
# MAGIC 
# MAGIC 次に、**Call The Model**で、リクエストを送信するためのPythonコードスニペットを表示するために**Python**ボタンをクリックします。コードをこのノートブックにコピーします。次のセルと同じようなコードになるはずです。
# MAGIC 
# MAGIC Databricksの外からリクエストするために、このトークンを利用することもできます。

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
  
  #print(dataset)
  url = 'https://e2-demo-west.cloud.databricks.com/model/wine_quality_takaakiyayoidatabrickscom/Production/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = process_input(dataset)
  
  #print(data_json)
  
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントから得られるモデルの予測結果と、ローカルでモデルを評価した結果は一致すべきです。

# COMMAND ----------

# モデルサービングは、比較的小さいデータバッチにおいて低レーテンシーで予測するように設計されています。
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])

# トレーニングしたモデルとデプロイされたモデルの結果を比較します。
df1 = pd.DataFrame(model_evaluations)
df2 = pd.DataFrame(served_predictions)

df1.rename(columns={0: "Model Prediction"}, inplace=True)
df2.rename(columns={"predictions": "Served Model Prediction"}, inplace=True)

pd.concat([df1, df2], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 
# MAGIC これらすべては、データサイエンティスト、MLエンジニアの皆様が、生産性高く機械学習の取り組みを進められる様にするためのものです。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ml_value_proposition.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
