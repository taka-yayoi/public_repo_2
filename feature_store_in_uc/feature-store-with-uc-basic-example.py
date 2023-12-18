# Databricks notebook source
# MAGIC %md # Unity Catakigにおける特徴量エンジニアリングの基本的なサンプル
# MAGIC
# MAGIC このノートブックでは、MLモデルのトレーニングとバッチ推論を行うために、推論時にのみ利用できる特徴量を含むUnity Catalogの特徴量の作成、格納、管理を行うためのUnity Catalogの特徴量エンジニアリングの使い方を説明します。このサンプルでは、さまざまな静的なワインの特徴量やリアルタイムのインプットとMLモデルを用いてワインの品質を予測することがゴールとなります。
# MAGIC
# MAGIC このノートブックでは以下の方法を説明します:
# MAGIC
# MAGIC - 機械学習モデル向けのトレーニングデータセットを構築するための特徴量テーブルの作成
# MAGIC - 新バージョンのモデルを作成するために、特徴量テーブルを編集し、アップデートされたテーブルを使用
# MAGIC - 特徴量とモデルがどのような関係にあるのかを特定するためにDatabricksの特徴量UIを使用
# MAGIC - 自動特徴量検索を用いたバッチスコアリングの実行
# MAGIC
# MAGIC ## 要件
# MAGIC
# MAGIC - Databricks機械学習ランタイム13.2以降
# MAGIC   - Databricks機械学習ランタイムにアクセスできない場合には、Databricksランタイム13.2以降でこのノートブックを実行することができます。この際には、ノートブックの最初で`%pip install databricks-feature-engineering`を実行します。

# COMMAND ----------

import pandas as pd

from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md ## データセットのロード
# MAGIC
# MAGIC 以下のセルのコードでは、データセットをロードして少々のデータ準備を行います: それぞれの観測値に対してユニークなIDを作成し、カラム名から空白を除外します。ユニークIDのカラム(`wine_id`)は特徴量テーブルの主キーとなり、特徴量の検索に使用されます。

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv",format="csv",sep=";",inferSchema="true",header="true" )

def addIdColumn(dataframe, id_column_name):
    """データフレームに id カラムを追加"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """UCのFeature Engineeringと互換性を持つようにカラム名を変更"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# 関数の実行
renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, 'wine_id')

# 特徴量テーブルに含めないターゲットカラム ('quality') を削除します
features_df = df.drop('quality')
display(features_df)

# COMMAND ----------

# MAGIC %md ## 新規のカタログの作成、あるいは既存カタログの再利用
# MAGIC
# MAGIC 新規にカタログを作成するには、メタストアに対する`CREATE CATALOG`権限が必要です。既存のカタログを使用する場合には、カタログに対する`USE CATALOG`権限が必要です。

# COMMAND ----------

catalog_name = "takaakiyayoi_catalog"

# 新規カタログを作成:
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
# spark.sql(f"USE CATALOG {catalog_name}")

# あるいは、既存カタログを再利用:
spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md ## カタログに新規スキーマを作成
# MAGIC
# MAGIC カタログに新規スキーマを作成するには、カタログに対する`CREATE SCHEMA`権限が必要です。

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS wine_db")
spark.sql("USE SCHEMA wine_db")

# それぞれの実行ごとにユニークなテーブル名を作成。複数回ノートブックを実行する際のエラーを回避します。
table_name = f"{catalog_name}.wine_db.wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md ## 特徴量テーブルの作成

# COMMAND ----------

# MAGIC %md 最初のステップでは`FeatureEngineeringClient`を作成します。

# COMMAND ----------

fe = FeatureEngineeringClient()

# ノートブックでfeature engineering client APIの関数のヘルプを取得できます :
# help(fe.<function_name>)

# 例:
# help(fe.create_table)

# COMMAND ----------

# MAGIC %md 特徴量テーブルを作成します。完全なAPIリファレンスについては([AWS](https://docs.databricks.com/machine-learning/feature-store/python-api.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/python-api)|[GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html))をご覧ください。

# COMMAND ----------

fe.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="ワインの特徴量"
)

# COMMAND ----------

# MAGIC %md
# MAGIC データフレームを指定せずに`create_table`を使うことができ、後で`fe.write_table`を使って特徴量テーブルにデータを追加することができます。
# MAGIC
# MAGIC 例:
# MAGIC
# MAGIC ```
# MAGIC fe.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["wine_id"],
# MAGIC     schema=features_df.schema,
# MAGIC     description="wine features"
# MAGIC )
# MAGIC
# MAGIC fe.write_table(
# MAGIC     name=table_name,
# MAGIC     df=features_df,
# MAGIC     mode="merge"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Unity Catalogの特徴量エンジニアリングを用いたモデルのトレーニング

# COMMAND ----------

# MAGIC %md 
# MAGIC 特徴量テーブルには予測ターゲットは含まれません。しかし、トレーニングデータセットには予測ターゲットの値が必要です。また、モデルが推論で使用されるまで利用できない特徴量が存在する場合があります。
# MAGIC
# MAGIC この例では、推論時にのみ観測できるワインの特性を表現する特徴量 **`real_time_measurement`** を使用します。この特徴量はトレーニングで使用され、推論時にはワインの特徴量の値として提供されます。

# COMMAND ----------

## inference_data_df には、 wine_id (主キー)、quality (予測ターゲット)、リアルタイムの特徴量が含まれます
inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 特徴量テーブルから特徴量を検索するために指定された`lookup_key`とオンライン特徴量`real_time_measurement`を使用するトレーニングデータセットを構築するために`FeatureLookup`を使用します。`feature_names`パラメータを指定しない場合には、主キーを除くすべての特徴量が返却されます。

# COMMAND ----------

def load_data(table_name, lookup_key):
    # FeatureLookupで`feature_names`パラメータを指定しない場合、主キーを除くすべての特徴量が返却されます
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_setはinference_data_dfと主キーがマッチするmodel_feature_lookupsの特徴量を検索します
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # トレーニングデータセット、テストデータセットを作成
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

# トレーニングデータセットとテストデータセットを作成
X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Unity CatalogのモデルにアクセスするようにMLflowクライアントを設定
mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog_name}.wine_db.wine_model"

client = MlflowClient()

try:
    client.delete_registered_model(model_name) # 作成済みの場合にはモデルを削除
except:
    None

# COMMAND ----------

# MAGIC %md
# MAGIC 次のセルのコードはscikit-learnのRandomForestRegressorモデルをトレーニングし、UCのFeature Engineeringを用いてモデルを記録します。
# MAGIC
# MAGIC このコードはトレーニングのパラメータと結果を追跡するためのMLflowエクスペリメントをスタートします。モデルのオートロギングをオフ(`mlflow.sklearn.autolog(log_models=False)`)にしていることに注意してください。これは、モデルは`fe.log_model`を用いて記録されるためです。

# COMMAND ----------

# MLflowオートロギングを無効化して、UCのFeature Engineeringを用いてモデルを記録
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## モデルのフィッティングと記録
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="wine_quality_prediction",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md
# MAGIC 記録されたモデルを参照するには、このノートブックのMLflowエクスペリメントページに移動します。エクスペリメントページにアクセスするには、左のナビゲーションバーのエクスペリメントアイコンをクリックします: <img src="https://docs.databricks.com/_static/images/icons/experiments-icon.png"/>
# MAGIC
# MAGIC リストからノートブックのエクスペリメントを探します。ノートブックと同じ名前になっており、この場合`feature-store-with-uc-basic-example`となります。
# MAGIC
# MAGIC エクスペリメントページを表示するにはエクスペリメント名をクリックします。このページの**Artifacts**セクションには、`fe.log_model`を呼び出した際に作成された、パッケージングされたUCのFeature Engineeringモデルが表示されます。
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/basic-fs-nb-artifact.png"/>
# MAGIC
# MAGIC また、このモデルは自動的にUnity Catalogに登録されます。

# COMMAND ----------

# MAGIC %md ## バッチスコアリング
# MAGIC
# MAGIC 推論において新規データに対して、パッケージングされたFeature Engineering in UCモデルを適用するには、`score_batch`を使用します。入力データには主キーのカラム`wine_id`とリアルタイムの特徴量である`real_time_measurement`のみが必要となります。モデルは自動で特徴量テーブルからすべてのその他の特徴量を検索します。

# COMMAND ----------

# ヘルパー関数
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

## シンプルにするために、この例では予測の入力データとしてinference_data_dfを使います
batch_input_df = inference_data_df.drop("quality") # ラベルカラムの削除

latest_model_version = get_latest_model_version(model_name)

predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)

display(predictions_df["wine_id", "prediction"])

# COMMAND ----------

# MAGIC %md ## 特徴量テーブルの修正
# MAGIC
# MAGIC 新たな特徴量を追加することでデータフレームを修正したとします。特徴量テーブルを更新するには`mode="merge"`で`fe.write_table`を使用します。

# COMMAND ----------

## 特徴量を保持するデータフレームの修正
so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]
new_features_df = (features_df.withColumn("average_so2", expr("+".join(so2_cols)) / 2))

display(new_features_df)

# COMMAND ----------

# MAGIC %md `fe.write_table`で`mode="merge"`を指定して特徴量テーブルを更新します。

# COMMAND ----------

fe.write_table(
    name=table_name,
    df=new_features_df,
    mode="merge"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 特徴量テーブルから特徴量を読み込むには`fe.read_table()`を使用します。

# COMMAND ----------

# 最新バージョンの特徴量テーブルを表示します
# 現行バージョンで削除された特徴量は表示されますが、値はnullとなります
display(fe.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md ## 更新された特徴量テーブルを用いた新たなモデルバージョンのトレーニング

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fe.create_training_set は inference_data_df とキーがマッチする model_feature_lookups 特徴量を検索します
    training_set = fe.create_training_set(df=inference_data_df, feature_lookups=model_feature_lookups, label="quality", exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # トレーニングデータセットとテストデータセットの作成
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC トレーニングデータセットを構築する際、特徴量を検索するために指定された `key` を使用します。

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fe):
    ## モデルのフィッティングと記録
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fe.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=model_name,
        )

train_model(X_train, X_test, y_train, y_test, training_set, fe)

# COMMAND ----------

# MAGIC %md **`score_batch`** を用いて特徴量に最新バージョンの登録MLflowモデルを適用します。

# COMMAND ----------

## シンプルにするために、この例では予測の入力データとしてinference_data_dfを使います
batch_input_df = inference_data_df.drop("quality") # ラベルカラムの削除
latest_model_version = get_latest_model_version(model_name)
predictions_df = fe.score_batch(model_uri=f"models:/{model_name}/{latest_model_version}", df=batch_input_df)
display(predictions_df["wine_id","prediction"])

# COMMAND ----------

# MAGIC %md ## 特徴量テーブルの権限コントロールと削除
# MAGIC
# MAGIC - Unity Catalogの特徴量テーブルに誰がアクセスできるのかをコントロールするには、カタログエクスプローラのテーブル詳細ページにある**Permissions**ボタンを使います。
# MAGIC - Unity Catalog特徴量テーブルを削除するには、カタログエクスプローラのテーブル詳細ページにあるケバブメニューをクリックし、**Delete**を選択します。UIを用いてUnity Catalog特徴量テーブルを削除すると、対応するDeltaテーブルも削除されます。
