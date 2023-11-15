# Databricks notebook source
# MAGIC %md # Databricks MLクイックスタート: モデルのトレーニング
# MAGIC
# MAGIC このノートブックでは、Databricksにおける機械学習モデルトレーニングの概要を説明します。モデルをトレーニングするには、Databricks機械学習ランタイムにプレインストールされているscikit-learnのようなライブラリを活用することができます。さらに、トレーニングしたモデルの追跡にMLflowを活用し、ハイパーパラメータチューニングをスケールさせるためにSparkTrialsとHyperoptを活用することもできます。
# MAGIC
# MAGIC このチュートリアルでは以下をカバーします:
# MAGIC - Part 1: MLflowトラッキングを用いたシンプルな分類モデルのトレーニング
# MAGIC - Part 2: Hyperoptを用いたより性能の良いモデルをトレーニングするためのハイパーパラメーターチューニング
# MAGIC
# MAGIC モデルライフサイクル管理とモデル推論を含む、Databricksにおける機械学習のプロダクション化の詳細に関しては、MLのエンドツーエンドのサンプルをご覧ください ([AWS](https://docs.databricks.com/ja/mlflow/end-to-end-example.html)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/mlflow/end-to-end-example)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/end-to-end-example.html))。
# MAGIC
# MAGIC ### 要件
# MAGIC - Databricksランタイム7.5 ML以降が稼働しているクラスター

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ライブラリ
# MAGIC
# MAGIC 必要なライブラリをインポートします。これらのライブラリは、Databricks機械学習ランタイム([AWS](https://docs.databricks.com/ja/machine-learning/index.html)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/)|[GCP](https://docs.gcp.databricks.com/runtime/mlruntime.html))クラスターにプレインストールされており、互換性とパフォーマンスにチューニングされています。

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md 
# MAGIC ### データのロード
# MAGIC
# MAGIC このチュートリアルでは、様々なワインのサンプルを記述しているデータセットを使用します。この[データセット](https://archive.ics.uci.edu/ml/datasets/Wine)はUCI機械学習リポジトリから取得しているものであり、DBFS ([AWS](https://docs.databricks.com/ja/dbfs/index.html)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/dbfs/)|[GCP](https://docs.gcp.databricks.com/data/databricks-file-system.html))に格納されています。赤ワインと白ワインを品質に基づいて分類することがゴールとなります。
# MAGIC
# MAGIC アップロードや他のデータソースからのロードに関する詳細については、データ取扱に関するドキュメント([AWS](https://docs.databricks.com/ja/data/index.html)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/data/)|[GCP](https://docs.gcp.databricks.com/data/index.html))をご覧ください。

# COMMAND ----------

# データのロードと前処理
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# ワイン品質に基づいた分類ラベルの定義
data_labels = data_df['quality'] >= 7
data_df = data_df.drop(['quality'], axis=1)

# 80/20でトレーニング/テストデータセットを分割
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md ## Part 1. 分類モデルのトレーニング

# COMMAND ----------

# MAGIC %md ### MLflowトラッキング
# MAGIC
# MAGIC [MLflowトラッキング](https://www.mlflow.org/docs/latest/tracking.html)を用いることで、お使いの機械学習トレーニングコード、パラメータ、モデルを整理することができます。
# MAGIC
# MAGIC [*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)を用いることで、自動でのMLflowトラッキングを有効化することができます。

# COMMAND ----------

# このノートブックでのMLflow autologgingを有効化
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、トレーニングされたモデルや関連付けられるメトリクスやパラメータが自動で記録されるMLflowランのコンテキスト内で分類器をトレーニングします。
# MAGIC
# MAGIC テストデータに対するモデルのAUCスコアのようなその他のメトリクスを追加することも可能です。

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # モデル、パラメータ、トレーニングメトリクスが自動でトラッキングされます
  model.fit(X_train, y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # テストデータに対するAUCスコアは自動で記録されないので、手動で記録します
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC このモデルのパフォーマンスに満足しない場合には、異なるハイパーパラメーターを用いて別のモデルをトレーニングします。

# COMMAND ----------

# 新たなランをスタートし、後でわかるようにrun_nameを割り当てます
with mlflow.start_run(run_name='gradient_boost') as run:
  model_2 = sklearn.ensemble.GradientBoostingClassifier(
    random_state=0, 
    
    # n_estimatorsで新たなパラメータ設定をトライします
    n_estimators=200,
  )
  model_2.fit(X_train, y_train)

  predicted_probs = model_2.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md ###MLflowランの参照
# MAGIC
# MAGIC 記録されたトレーニングランを参照するには、エクスペリメントサイドバーを表示するためにノートブック右上の**Experiment**アイコンをクリックします。必要であれば、最新のランを取得、監視するためにリフレッシュアイコンをクリックします。
# MAGIC
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC
# MAGIC より詳細なMLflowエクスペリメントページ([AWS](https://docs.databricks.com/ja/mlflow/tracking.html#notebook-experiments)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/mlflow/tracking#notebook-experiments)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#notebook-experiments))を表示するために、エクスペリメントページアイコンをクリックすることもできます。このページでは、複数のランを比較し、特定のランの詳細を表示することができます。
# MAGIC
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのロード
# MAGIC
# MAGIC また、MLflow APIを用いて特定のランの結果にアクセスすることができます。以下のセルのコードでは、特定のMLflowランでトレーニングされたモデルのロード方法と、予測での利用方法を説明しています。また、MLflowランのページ([AWS](https://docs.databricks.com/ja/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment))では特定のモデルをロードするためのコードスニペットを参照することができます。

# COMMAND ----------

# モデルが記録された後では、別のノートブックやジョブでモデルをロードすることができます
# mlflow.pyfunc.load_modelを用いることで、共通APIを通じてモデルの予測を行うことができます
model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

predictions_loaded = model_loaded.predict(X_test)
predictions_original = model_2.predict(X_test)

# ロードされたモデルはオリジナルと一致すべきです
assert(np.array_equal(predictions_loaded, predictions_original))

# COMMAND ----------

# MAGIC %md ## Part 2. ハイパーパラメーターチューニング
# MAGIC
# MAGIC この時点で、シンプルなモデルをトレーニングし、皆様の取り組み結果を整理するためにMLflowトラッキングサービスを活用しました。このセクションでは、Hyperoptを用いてより洗練されたチューニングを実行する方法をカバーします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### HyperoptとSparkTrialsによる並列トレーニング
# MAGIC
# MAGIC
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/)はハイパーパラメーターチューニングのためのPythonライブラリです。DatabricksにおけるHyperopt活用の詳細については、ドキュメント([AWS](https://docs.databricks.com/ja/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://learn.microsoft.com/ja-jp/azure/databricks/machine-learning/automl-hyperparam-tuning/#hyperparameter-tuning-with-hyperopt)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt))を参照ください。
# MAGIC
# MAGIC 並列でのハイパーパラメーター探索と複数のモデルのトレーニング実行を行うために、HyperoptとSparkTrialsを活用することができます。これによって、モデルパフォーマンスの最適化に要する時間を削減します。モデルとパラメータを自動で記録するように、MLflowトラッキングはHyperoptとインテグレーションされています。

# COMMAND ----------

# 探索する検索空間の定義
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # それぞれのワーカーでautologgingを有効化
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # テストデータに対するAUCに基づくチューニング
    # プロダクション環境では、代わりに別の検証用データセットを活用することができます
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # fminがauc_scoreを最大化するように、lossを -1*auc_score に設定します
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrialsがSparkワーカーを用いてチューニングを分散させます
# 並列度を高めると処理を加速しますが、それぞれのハイパーパラメーターのトライアルが他のトライアルから得られる情報が減少します
# 小規模なクラスターやDatabricks Community Editionでは parallelism=2 に設定してください
spark_trials = SparkTrials(
  parallelism=8
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # 最大のAUCを達成するパラメーターを特定するためhyperoptを使用します
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### ベストモデルを取得するためにランを検索します
# MAGIC
# MAGIC すべてのランはMLflowで追跡されるので、テストデータに対する最大のAUCを持つチューニングのランを特定するために、MLflowのsearch runs APIを用いて、ベストなランのメトリクスとパラメーターを取得することができます。
# MAGIC
# MAGIC このチューニングされたモデルは、パート1でトレーニングされたよりシンプルなモデルよりも良いパフォーマンスであるべきです。

# COMMAND ----------

# テストデータに対するAUCでランをソートします。値が同じ場合には最新のランを使用します
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)
best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md ### UIで複数のランを比較
# MAGIC
# MAGIC パート1と同じように、**Experiment**サイドバーの上部にある外部リンクアイコン経由でMLflowエクスペリメントの詳細ページでランを参照、比較できます。
# MAGIC
# MAGIC エクスペリメント詳細ページでは、親のランを展開するために　"+"　アイコンをクリックし、親を除くすべてのランを選択し、**Compare**をクリックします。メトリックに対する様々なパラメーターの値のインパクトを表示するparallel coordinates plotを用いて、様々なランを可視化することができます。
# MAGIC
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>
