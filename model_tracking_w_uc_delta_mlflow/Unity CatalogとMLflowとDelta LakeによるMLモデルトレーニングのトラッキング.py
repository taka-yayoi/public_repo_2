# Databricks notebook source
# MAGIC %md
# MAGIC #  Unity CatalogとMLflowとDelta Lakeによる機械学習モデルトレーニングのトラッキング
# MAGIC
# MAGIC データチームがモデルをトレーニングし、それをプロダクションにデプロイして、しばらくの間は上手くいくと言うのはよくある話です。そして、モデルが変な予測をし始めると、モデルを調査してデバッグしなくてはなりません。
# MAGIC
# MAGIC このノートブックでは、デバッグを容易にするためにモデルのトレーニングを容易に追跡、可視化、再現するために、どのように[Unity Catalog](https://www.databricks.com/jp/product/unity-catalog)と[MLflow](http://mlflow.org)と[Delta Lake](http://delta.io)を活用するのかをデモンストレーションします。
# MAGIC
# MAGIC - **Delta Lake:** データのバージョン管理と「タイムトラベル」機能を提供
# MAGIC - **MLflow:** モデルやパラメーター、精度を記録
# MAGIC - **Unity Catalog:** モデルのライフサイクルやリネージの管理
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/delta_mlflow_uc.png)
# MAGIC
# MAGIC このデモでは以下のステップをカバーします。
# MAGIC 1. データのインポート
# MAGIC 1. Seabornとmatplotlibによるデータの可視化
# MAGIC 1. 予測精度を改善するための特徴量エンジニアリングとDeltaによるバージョン追跡
# MAGIC 1. 特徴量エンジニアリングの効果をMLflowで確認
# MAGIC 1. Unity Catalogにベストモデルを登録
# MAGIC 1. 登録済みモデルをSpark UDFとして別のデータセットに適用
# MAGIC
# MAGIC この例では、融資のデータから「悪いローン」を予測するモデルを構築します。
# MAGIC
# MAGIC ## 要件
# MAGIC - Unity Catalog対応クラスター
# MAGIC - Databricks MLランタイム
# MAGIC
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2024/08/26</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>15.4ML</td></tr>
# MAGIC </table>
# MAGIC
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 問題定義: 貸し手向け「悪いローン」の分類
# MAGIC
# MAGIC このノートブックでは、クレジットのスコア、クレジット履歴などその他の特徴量に基づいて、「悪いローン」(利益を産まない可能性があるローン)の特定をゴールとして、Lending Clubデータセットにおける分類問題に取り組みます。
# MAGIC
# MAGIC 最終的なゴールは、ローンを承認するかどうかを決定する前に、ローンの係員が使用する解釈可能なモデルを生成することです。この様なモデルは貸し手に対して、情報を提供するビューとなり、見込みのある借り手を即座に評価し、レスポンスできる様にします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ
# MAGIC
# MAGIC 使用するデータはLending Clubの公開データです。これには、2012年から2017年に融資されたローンが含まれています。それぞれのローンには、申請者によって提供された申込者情報と、現在のローンのステータス(遅延なし、遅延、完済など)、最新の支払い情報が含まれています。データに対する完全なビューは[データ辞書](https://resources.lendingclub.com/LCDataDictionary.xlsx)をご覧ください。
# MAGIC
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# 事前にカタログを作成しておいてください
catalog_name = "takaakiyayoi_catalog"
schema_name = f"loan_{username}"
table_name = "loan_stats"
model_name = "loan_model"

# Unity Catalogにおける登録モデルのパス
MODEL_PATH = f"{catalog_name}.{schema_name}.{model_name}"

print(f"{catalog_name=}")
print(f"{schema_name=}")
print(f"{table_name=}")
print(f"{MODEL_PATH=}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md ## データのインポート
# MAGIC
# MAGIC このセクションでは、サンプルデータからSparkデータフレームにデータを読み込みます。

# COMMAND ----------

# テーブルが存在する場合には削除
spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# Lending Clubデータをロード&加工し、Delta LakeフォーマットでUCに保存します
lspq_path = "/databricks-datasets/samples/lending_club/parquet/"
data = spark.read.parquet(lspq_path)
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの前処理

# COMMAND ----------

import pyspark.sql.functions as F

# 必要なカラムを選択し、他の前処理を適用
features = [
   "loan_amnt",
   "annual_inc",
   "dti",
   "delinq_2yrs",
   "total_acc",
   "total_pymnt",
   "issue_d",
   "earliest_cr_line",
]

raw_label = "loan_status"
loan_stats_ce = data.select(*(features + [raw_label]))
print(
    "------------------------------------------------------------------------------------------------"
)
print("悪いローンのラベルを作成、これにはチャージオフ、デフォルト、ローンの支払い遅延が含まれます...")
# ローンのステータスに基づいて悪いローンのラベルを作成
loan_stats_ce = loan_stats_ce.filter(
    loan_stats_ce.loan_status.isin(["Default", "Charged Off", "Fully Paid"])
).withColumn("bad_loan", (~(loan_stats_ce.loan_status == "Fully Paid")).cast("int"))
loan_stats_ce = loan_stats_ce.orderBy(F.rand()).limit(
    10000
)  # Community Editionでも実行できる様にロードする行を限定

print(
    "------------------------------------------------------------------------------------------------"
)
print("数値のカラムを適切な型にキャスト...")
loan_stats_ce = (
    loan_stats_ce.withColumn(
        "issue_year", F.substring(loan_stats_ce.issue_d, 5, 4).cast("double")
    )  # 文字列から年のみを取り出しdoubleに変換
    .withColumn(
        "earliest_year",
        F.substring(loan_stats_ce.earliest_cr_line, 5, 4).cast("double"),
    )  # 文字列から年のみを取り出しdoubleに変換
    .withColumn("total_pymnt", loan_stats_ce.total_pymnt.cast("double"))
)
# ローン期間を計算
loan_stats_ce = loan_stats_ce.withColumn(
    "credit_length_in_years", (loan_stats_ce.issue_year - loan_stats_ce.earliest_year)
)

print(
    "------------------------------------------------------------------------------------------------"
)

display(loan_stats_ce)

# COMMAND ----------

# MAGIC %md ## データの可視化
# MAGIC
# MAGIC モデルをトレーニングする前に、Seaborn、matplotlibを用いてデータを可視化します。
# MAGIC
# MAGIC まず、目的変数の`bad_loan`の分布を確認します。

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

loan_stats_pdf = loan_stats_ce.toPandas()
bad_loan_count = loan_stats_pdf.groupby("bad_loan")['bad_loan'].count()

sns.set_style("whitegrid") # チャートのスタイルを設定
plt.figure(figsize=(6,6)) # 図のサイズを設定
plt.pie(bad_loan_count, labels=bad_loan_count.index, autopct='%1.1f%%') # 円グラフを作成
plt.show() # グラフを表示

# COMMAND ----------

# MAGIC %md 
# MAGIC 特徴量と2値ラベルの間の相関を見るにはボックスプロットが有用です。

# COMMAND ----------

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in loan_stats_pdf.columns:
    if col in ["issue_d", "earliest_cr_line", "loan_status", "bad_loan"]:
        continue  # カテゴリ変数にボックスプロットは使用できません
    sns.boxplot(x=loan_stats_pdf.bad_loan, y=loan_stats_pdf[col], ax=axes[axis_i, axis_j])
    axis_j += 1
    if axis_j == dims[1]:
        axis_i += 1
        axis_j = 0

# COMMAND ----------

# MAGIC %md 上のボックスプロットから、いくつかの変数が`bad_loan`に対する単変量予測子として優れていることがわかります。
# MAGIC - `total_pyment`のボックスプロットにおいて、デフォルトになるケースでの支払い金額が少ない傾向が見て取れます。
# MAGIC   - `total_pymnt`: 融資された総額に対して当日までに支払われた金額。
# MAGIC - `dti`には、デフォルトになるケースとの逆の相関が認められます。
# MAGIC   - `dti`: 借り手の自己申告による月収に対し、住宅ローンとリクエストされたLCローンを除く全ての債務に対する月々の支払い総額を使用して計算される比率。

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC モデルのトレーニングの前に、欠損値のチェックを行います。

# COMMAND ----------

loan_stats_pdf.isna().any()

# COMMAND ----------

# MAGIC %md サンプリングされたデータには欠損値が存在する場合があります。このようなデータの欠損値や統計は、`dbutils.data.summarize`や`display`関数の結果からもアクセスすることができます。

# COMMAND ----------

dbutils.data.summarize(loan_stats_ce)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 機械学習モデル構築で用いるテーブルとデータバージョンのトラッキング
# MAGIC
# MAGIC 機械学習モデルの構築においては、トレーニングデータに対する試行錯誤が伴います。データバージョンを管理・指定できることはDelta Lakeを活用することのメリットであり、後ほどレストアできる様に以前のバージョンのデータセットを保持します。
# MAGIC
# MAGIC トレーニングデータとテストデータを分割して、トレーニングデータを保存して準備は完了です。

# COMMAND ----------

from sklearn.model_selection import train_test_split

# 使用する変数
features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "total_acc",
    "credit_length_in_years",
    "bad_loan",
]

# 欠損値を含む行を削除
loan_stats_pdf.dropna(inplace=True)

# 説明変数を選択して、トレーニング・テストデータに分割
train, test = train_test_split(loan_stats_pdf[features], random_state=123)
X_train = train.drop(["bad_loan"], axis=1)
X_test = test.drop(["bad_loan"], axis=1)
y_train = train.bad_loan
y_test = test.bad_loan

# トレーニングデータの保存
training_df = spark.createDataFrame(X_train)
training_df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deltaテーブルの履歴を確認
# MAGIC
# MAGIC Deltaテーブルでは、初期状態および、以降のデータ追加、アップデート、削除、マージ、追加を含むこのテーブルに対するすべてのトランザクションがテーブルに記録されます。テーブルの履歴にアクセスするには、`DESCRIBE HISTORY`を使用します。

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY loan_stats;

# COMMAND ----------

# MAGIC %md ## ベースラインモデルの構築
# MAGIC
# MAGIC 出力が2値であり、複数の変数間での相互関係がある可能性があることから、このタスクにはランダムフォレスト分類器が適しているように見えます。
# MAGIC
# MAGIC 以下のコードでは、scikit-learnを用いてシンプルな分類器を構築します。モデルの精度を追跡するためにMLflowを用い、後ほど利用するためにモデルを保存します。
# MAGIC
# MAGIC また、DeltaTableのAPIを用いてDeltaテーブルの最新バージョンを取得します。
# MAGIC
# MAGIC [Table utility commands — Delta Lake Documentation](https://docs.delta.io/latest/delta-utility.html#retrieve-delta-table-history)

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
from delta.tables import DeltaTable

# UC配下のモデルとして登録
mlflow.set_registry_uri("databricks-uc")

# sklearnのRandomForestClassifierのpredictメソッドは、2値の分類結果(0、1)を返却します。
# 以下のコードでは、それぞれのクラスに属する確率を返却するpredict_probaを用いる、ラッパー関数SklearnModelWrapperを構築します。

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

# トレーニングデータセット、テストデータセットのロード、トレーニングデータセットのバージョンを受け取りトレーニング、ロギングを行う
def train_and_log_model(X_train, y_train, X_test, y_test, version_to_load):

  # mlflow.start_runは、このモデルのパフォーマンスを追跡するための新規MLflowランを生成します。
  # コンテキスト内で、使用されたパラメーターを追跡するためにmlflow.log_param、精度のようなメトリクスを追跡するためにmlflow.log_metricを呼び出します。
  with mlflow.start_run(run_name='untuned_random_forest'):
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

    # トレーニングデータのロギング
    dataset = mlflow.data.load_delta(table_name=f"{catalog_name}.{schema_name}.{table_name}", version=version_to_load)
    mlflow.log_input(dataset, context="training")
  
    # MLflowにはモデルをサービングする際に用いられるconda環境を作成するユーティリティが含まれています。
    # 必要な依存関係がconda.yamlに保存され、モデルとともに記録されます。
    conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature, )

    return model

# COMMAND ----------

# Deltaテーブルの最新バージョンの取得
delta_table = DeltaTable.forName(spark, table_name)
version_to_load = delta_table.history(1).select("version").collect()[0].version
print(f"{version_to_load=}")

model = train_and_log_model(X_train, y_train, X_test, y_test, version_to_load)

# COMMAND ----------

# MAGIC %md
# MAGIC データチェックとして、モデルによって出力される特徴量の重要度を確認します。

# COMMAND ----------

import pandas as pd

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC 先ほどボックスプロットで見たように、デフォルトを予測するのに`dti`、そして、ボックスプロットでは特定が難しかった`annual_inc`などが重要であることがわかります。

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowにROC曲線のAUCを記録しました。右上の**フラスコマーク**をクリックして、エクスペリメントランのサイドバーを表示します。
# MAGIC
# MAGIC このモデルはAUC0.58を達成しました。
# MAGIC
# MAGIC ランダムな分類器のAUCは0.5となり、それよりAUCが高いほど優れていると言えます。詳細は、[Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)を参照ください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow APIによるモデルの操作
# MAGIC
# MAGIC モデルの記録・管理を担うMLflowではAPIが提供されているので、PythonからMLflowにアクセスしてさまざまな処理を行うことができます。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

# 登録モデルの最新バージョンを取得
def get_latest_model_version(model_name):  
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)

  return max([model_version_info.version for model_version_info in model_version_infos])

# 登録モデルの初期化
try:
  client.delete_registered_model(name=MODEL_PATH)
except:
  pass

# COMMAND ----------

# MAGIC %md ### Unity Catalogモデルレジストリにモデルを登録
# MAGIC
# MAGIC モデルレジストリにモデルを登録することで、Databricksのどこからでもモデルを容易に参照できるようになります。
# MAGIC
# MAGIC 以下のセクションでは、どのようにプログラム上から操作をするのかを説明しますが、UIを用いてモデルを登録することもできます。"Unity Catalogでモデルのライフサイクルを管理する" ([AWS](https://docs.databricks.com/ja/machine-learning/manage-model-lifecycle/index.html)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/manage-model-lifecycle/))を参照ください。

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
run_id

# COMMAND ----------

# モデルレジストリにモデルを登録します
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルページでモデルを確認できるはずです。モデルページを表示するには、左のサイドバーで**モデル**をクリックします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### エイリアスによるモデルの管理
# MAGIC
# MAGIC モデルの[エイリアス](https://docs.databricks.com/ja/machine-learning/manage-model-lifecycle/index.html#deploy-and-organize-models-with-aliases-and-tags)を使用すると、登録済みモデルの特定のバージョンに変更可能な名前付きリファレンスを割り当てることができます。 エイリアスを使用して、モデルバージョンのデプロイステータスを示すことができます。 たとえば、現在本番運用にあるモデルバージョンに`"Champion"`エイリアスを割り当て、本番運用モデルを使用するワークロードでこのエイリアスをターゲットにすることができます。 その後、`"Champion"`エイリアスを別のモデル バージョンに再割り当てすることで、本番運用モデルを更新できます。
# MAGIC
# MAGIC 以下のコードで、このモデルバージョンに`"Champion"`エイリアスを割り当てます。これによって、エイリアスを指定したモデルのロードが可能となります。

# COMMAND ----------

latest_version = get_latest_model_version(MODEL_PATH)
client.set_registered_model_alias(MODEL_PATH, "Champion", latest_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルバージョンの説明の追加
# MAGIC
# MAGIC トレーニングしたモデルバージョンを特定し、モデルバージョンに説明文を追加するためにMLflow APIを活用することができます。
# MAGIC
# MAGIC はじめに、登録モデル自身に説明文を付与します。

# COMMAND ----------

client.update_registered_model(
  name=MODEL_PATH,
  description="このモデルは、ローン申請者の特徴量から「悪いローン」(利益を産まない可能性があるローン)の特定を行います。"
)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルバージョンに説明文を付与します。

# COMMAND ----------

client.update_model_version(
  name=MODEL_PATH,
  version=latest_version,
  description="このモデルバージョンは、ローンごとのloan_amnt, annual_inc, dti, delinq_2yrs, total_acc, credit_length_in_yearsの特徴量を用いてscikit-learnでトレーニングされました。"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルページでは、モデルバージョンが`Champion`のエイリアスが付与されていると表示されます。
# MAGIC
# MAGIC これで、`models:/<モデルパス>@Champion`のパスでモデルを参照することができます。

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{MODEL_PATH}@Champion")

# サニティチェック: この結果はMLflowで記録されたAUCと一致すべきです
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特徴量エンジニアリング: データスキーマを進化
# MAGIC
# MAGIC データセットの過去のバージョンを追跡するDelta Lakeを用いて、モデルパフォーマンスを改善するためにいくつかの特徴量エンジニアリングを行うことができます。ここでは、ローンごとの支払い金額とローン金額の損益の合計を捉える特徴量`net`を追加します。
# MAGIC <br><br>
# MAGIC ```
# MAGIC net = total_pymnt - loan_amnt
# MAGIC ```
# MAGIC
# MAGIC - `total_pymnt`: 融資された総額に対して当日までに支払われた金額。
# MAGIC - `loan_amnt`: 借り手が申請したローンのリストされた金額です。ある時点で信用部門がローン額を減額した場合、その変更はこの値に反映されます。
# MAGIC
# MAGIC `net`が0を上回っていれば健全と言えますが、マイナスの場合は負債を背負っていることになります。

# COMMAND ----------

print(
    "------------------------------------------------------------------------------------------------"
)
print("ローンごとの支払い、ローン金額の合計を計算...")
loan_stats_new = loan_stats_ce.withColumn(
    "net",
    F.round(loan_stats_ce.total_pymnt - loan_stats_ce.loan_amnt, 2)).select(
        (features + ["net"]
    ),
)
display(loan_stats_new)

# COMMAND ----------

# 欠損値を含む行を削除
loan_stats_pdf.dropna(inplace=True)

# 説明変数を選択して、トレーニング・テストデータに分割
train, test = train_test_split(loan_stats_new.toPandas(), random_state=123)
X_train = train.drop(["bad_loan"], axis=1)
X_test = test.drop(["bad_loan"], axis=1)
y_train = train.bad_loan
y_test = test.bad_loan

# トレーニングデータの保存
training_df = spark.createDataFrame(X_train)
training_df.write.option("mergeSchema", "true").format("delta").mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deltaテーブルの履歴を確認
# MAGIC
# MAGIC Deltaテーブルのバージョンが上がっていることを確認できます。

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY loan_stats;

# COMMAND ----------

# MAGIC %md ## 新たなデータを用いたエクスペリメント

# COMMAND ----------

# Deltaテーブルの最新バージョンの取得
delta_table = DeltaTable.forName(spark, table_name)
version_to_load = delta_table.history(1).select("version").collect()[0].version
print(f"{version_to_load=}")

model = train_and_log_model(X_train, y_train, X_test, y_test, version_to_load)

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md ### MLflowを用いて結果を確認
# MAGIC
# MAGIC MLflowエクスペリメントサイドバーを開いて、ランを参照します。メニューを表示するために、下向き矢印の隣にあるDateをクリックし`auc`を選択し、aucメトリックの順でランを並び替えます。aucは0.97となっています。ベースラインモデルを上回りました！
# MAGIC
# MAGIC MLflowはそれぞれのランのパフォーマンスメトリクスとパラメーターをトラッキングします。MLflowエクスペリメントサイドバーの一番上にある右上向きの矢印アイコン<img src="https://docs.databricks.com/_static/images/icons/external-link.png"/>をクリックすることで、MLflowランの一覧に移動することができます。
# MAGIC
# MAGIC 以下のコードでは、最も高いパフォーマンスを示したランを検索しています。

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md ### モデルのエイリアスを更新
# MAGIC
# MAGIC はじめに`loan_model`という名前でベースラインモデルをモデルレジストリに保存しました。さらに精度の高いモデルができましたので、`loan_model`のエイリアスを更新します。

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/random_forest_model", MODEL_PATH)

# COMMAND ----------

# MAGIC %md 
# MAGIC 左のサイドバーで**モデル**をクリックし、二つのバージョンのモデルが存在することを確認します。
# MAGIC
# MAGIC 以下のコードで新バージョンをChampionに移行します。

# COMMAND ----------

client.set_registered_model_alias(
  name=MODEL_PATH,
  alias="Champion",
  version=new_model_version.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルバージョンに説明文を記載します。

# COMMAND ----------

client.update_model_version(
  name=MODEL_PATH,
  version=new_model_version.version,
  description="このモデルバージョンは、ローンごとの支払い金額とローン金額の損益の合計を捉える特徴量を追加してトレーニングされました。"
)

# COMMAND ----------

# MAGIC %md `load_model`を呼び出すクライアントはURIを変更することなしに新たなモデルにアクセスできます。

# COMMAND ----------

# このコードは上の"ベースラインモデルの構築"と同じものです。新たなモデルを利用するためにクライアント側での変更は不要です！
model = mlflow.pyfunc.load_model(f"models:/{MODEL_PATH}@Champion")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## バッチ推論
# MAGIC
# MAGIC 新たなデータのコーパスに対してモデルを評価したいというシナリオは数多く存在します。例えば、新たなデータバッチを手に入れたり、同じデータコーパスに対して二つのモデルを比較することなどが考えられます。
# MAGIC
# MAGIC 以下のコードでは、並列に処理を行うためにSparkを用い、Deltaテーブルに格納されたデータに対してモデルの評価を行います。

# COMMAND ----------

# 新たなデータコーパスをシミュレートするために、既存のX_trainデータをDeltaテーブルに保存します。
# 実際の環境では、本当に新たなデータバッチとなります。
spark_df = spark.createDataFrame(X_train)
# Deltaテーブルの保存先
table_path = f"{catalog_name}.{schema_name}.loan_inference"
# すでにコンテンツが存在する場合には削除します
spark.sql(f"DROP TABLE IF EXISTS {table_path}")
spark_df.write.format("delta").saveAsTable(table_path)

# COMMAND ----------

# MAGIC %md モデルをSparkのUDF(ユーザー定義関数)としてロードし、Deltaテーブルに適用できるようにします。

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@Champion")

# COMMAND ----------

# 新規データをDeltaから読み込みます
new_data = spark.read.format("delta").table(table_path)

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
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## カタログエクスプローラでの探索
# MAGIC
# MAGIC ここまで作成してきたテーブルやモデルは全てUnity Catalogで管理されています。以下のコマンドを実行することで表示されるリンクから、Unity CatalogのGUIである[カタログエクスプローラ](https://docs.databricks.com/ja/catalog-explorer/index.html)にアクセスして以下のようなポイントをチェックしてみましょう。
# MAGIC
# MAGIC - あるモデルバージョンは、どのバージョンのDeltaテーブルを用いてトレーニングされたのか
# MAGIC - モデルトレーニングの際のパラメーターや精度はどうなっているのか
# MAGIC - モデルバージョン間で精度はどのように変化したのか
# MAGIC - モデル本体がどのように記録されているのか
# MAGIC - 現在`Champion`のモデルはどれか
# MAGIC - Deltaテーブルにはどのような操作が加えられたのか

# COMMAND ----------

displayHTML(f"<a href='/explore/data/{catalog_name}/{schema_name}'>カタログエクスプローラ</a>")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
