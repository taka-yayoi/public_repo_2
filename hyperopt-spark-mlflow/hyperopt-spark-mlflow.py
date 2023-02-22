# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperoptの分散処理と自動化されたMLflowトラッキング
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt)はハイパーパラメーターチューニングのためのPythonライブラリです。Databricks機械学習ランタイムには、最適化かつ強化されたバージョンのHyperopt、分散チューニング向けの自動MLflowトラッキング、`SparkTrials`クラスが含まれています。
# MAGIC 
# MAGIC このノートブックでは、シングルマシン向けのPython機械学習アルゴリズムのハイパーパラメーターチューニングをどのようにスケールアップし、MLflowを用いて結果をトラッキングするのかを説明します。パート1では、シングルマシンのHyperoptワークフローを作成します。パート2では、Sparkクラスターでワークフローの計算処理を分散させるために`SparkTrials`クラスの使用法を学びます。
# MAGIC 
# MAGIC - [Parallelize hyperparameter tuning with scikit\-learn and MLflow \| Databricks on AWS](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html#)
# MAGIC - [Part 2: Hyperopt\.\. In this blog series, I am comparing… \| by Jakub Czakon \| Towards Data Science](https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324)
# MAGIC - [Python: Hyperopt で機械学習モデルのハイパーパラメータを選ぶ \- CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/hyperopt)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2023/02/21</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>11.3ML</td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md ## 必要なパッケージのインポートとデータセットのロード

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

# Databricks機械学習ランタイムを実行している場合、`mlflow`はインストール済みですので以下の行をスキップすることができます。
import mlflow

# COMMAND ----------

# scikit-learnからirisデータセットをロード
iris = iris = load_iris()
X = iris.data
y = iris.target

# COMMAND ----------

# MAGIC %md ## パート1. シングルマシンのHyperoptワークフロー
# MAGIC 
# MAGIC Hyperoptのワークフローにおけるステップは以下の通りです:
# MAGIC 1. 最小化する関数を定義
# MAGIC 1. ハイパーパラメーターに対する探索空間を定義
# MAGIC 1. 探索アルゴリズムを選択
# MAGIC 1. Hyperoptの`fmin()`を用いてチューニングアルゴリズムを実行
# MAGIC 
# MAGIC 詳細は[Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)をご覧ください。

# COMMAND ----------

# MAGIC %md ### 最小化する関数の定義
# MAGIC 
# MAGIC このサンプルでは、サポートベクトルマシンの分類器を使用します。ここでのゴールは、正則化パラメーター`C`の最適な値を見つけ出すことです。
# MAGIC 
# MAGIC Hyperoptワークフローのコードの大部分は目的関数です。このサンプルでは、[support vector classifier from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)を用いています。

# COMMAND ----------

def objective(C):
    # サポートベクトル分類モデルの作成
    clf = SVC(C)
    
    # モデルのパフォーマンスを比較するために交差検証の精度を使用
    accuracy = cross_val_score(clf, X, y).mean()
    
    # Hyperoptは目的関数を最小化しようとします。高い精度の値は優れたモデルであることを意味するので、負の精度を返却するようにしないといけません。
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ### ハイパーパラメーターの探索空間の定義
# MAGIC 
# MAGIC 探索空間とパラメーターの表現の定義の詳細については[Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)をご覧下さい。hyperoptでは正規分布など確率分布を考慮してハイパーパラメーターを探索することができます。
# MAGIC 
# MAGIC hyperoptの最適化アルゴリズムで認識される確率論的な表現には以下のものがあります:
# MAGIC 
# MAGIC - `hp.choice(label, options)`
# MAGIC 
# MAGIC     選択肢の一つを返却し、listかtubleである必要があります。選択肢の要素を[ネストされた]確率論的表現にすることも可能です。この場合、選択肢のいくつかにおける確率論的な選択肢は条件付きパラメータとなります。
# MAGIC 
# MAGIC - `hp.randint(label, upper)`
# MAGIC 
# MAGIC     範囲[0, upper)のランダムな整数値を返却します。この分布のセマンティクスは、ロス関数において、離れた整数値と比較して、近隣の整数値との間に相関がないというものです。これは、例えばランダムのシードを記述する際に適切な分布です。ロス関数が近隣の整数値とより相関している場合には、`quniform`, `qloguniform`, `qnormal`や`qlognormal`のように「quantized(量子化された)」連続分布を使用しなくてはならないかも知れません。
# MAGIC 
# MAGIC - `hp.uniform(label, low, high)`
# MAGIC 
# MAGIC     - `low`と`high`の間にある均等な値を返却します。
# MAGIC     - 最適化を行う際、この変数は2面性の間隔に制約を受けます。
# MAGIC 
# MAGIC - `hp.quniform(label, low, high, q)`
# MAGIC 
# MAGIC     - round(uniform(low, high) / q) * q のような値を返却します。
# MAGIC     - 目的変数が依然として「スムーズ」な離散値で上限値、下限値がある場合に適しています。
# MAGIC 
# MAGIC - `hp.loguniform(label, low, high)`
# MAGIC 
# MAGIC     - 戻り値の対数値が均一に分散するように、`exp(uniform(low, high))`から導かれる値を返却します。
# MAGIC     - 最適化の際、この変数は間隔`[exp(low), exp(high)]`の制約を受けます。
# MAGIC 
# MAGIC - `hp.qloguniform(label, low, high, q)`
# MAGIC 
# MAGIC     - `round(exp(uniform(low, high)) / q) * q`のような値を返却します。
# MAGIC     - 目的変数が「スムーズ」で値のサイズに応じてよりスムーズになるような離散値で上限値、下限値の制約を受ける場合に適しています。
# MAGIC 
# MAGIC - `hp.normal(label, mu, sigma)`
# MAGIC 
# MAGIC     - 平均値の mu と標準偏差の sigma の正規分布の実数を返却します。最適化の際は制約のない変数となります。
# MAGIC 
# MAGIC - `hp.qnormal(label, mu, sigma, q)`
# MAGIC 
# MAGIC     - `round(normal(mu, sigma) / q) * q`のような値を返却します。
# MAGIC     - mu 周辺の値を取り、基本的に未制限の離散値の場合に適しています。
# MAGIC 
# MAGIC - `hp.lognormal(label, mu, sigma)`
# MAGIC 
# MAGIC     - 戻り値の対数が正規分布になるように、`exp(normal(mu, sigma))`から導かれた値を返却します。最適化の際、この変数は正の数であるという制約を受けます。
# MAGIC 
# MAGIC - `hp.qlognormal(label, mu, sigma, q)`
# MAGIC 
# MAGIC     - `round(exp(normal(mu, sigma)) / q) * q`のような値を返却します。
# MAGIC     - 目的変数が「スムーズ」で値のサイズに応じてよりスムーズになるような離散値で、一方向でのみ制約を受ける場合に適しています。

# COMMAND ----------

# 対数正規分布
search_space = hp.lognormal('C', 0, 1.0)

# 正規分布
#search_space = hp.normal('C', 0, 1.0)

# 均一分布
#search_space = hp.uniform('C', 0, 1.0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 探索空間の可視化
# MAGIC 
# MAGIC Cmd 9で定義した探索空間がどのようになっているのかを確認するために、探索空間からデータポイントを抽出します。
# MAGIC 
# MAGIC [Defining search spaces \- Hyperopt Documentation](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/)

# COMMAND ----------

import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt

values = []

# 100ポイントをサンプリング
for i in range(100):
  sample = hyperopt.pyll.stochastic.sample(search_space)
  #print("sample:", sample)
  values.append(sample)
  
plt.hist(values, density=False, bins=30)  # density=False would make counts

# COMMAND ----------

# MAGIC %md ### 探索アルゴリズムの選択
# MAGIC 
# MAGIC 主な選択肢は以下の2つです:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators、過去の結果に基づいて探索する新たなハイパーパラメーターの設定を繰り返しかつ適合的に選択するベイジアンアプローチ
# MAGIC * `hyperopt.rand.suggest`: ランダムサーチ、探索空間のサンプリングを行う非適合型のアプローチ

# COMMAND ----------

algo = tpe.suggest

# COMMAND ----------

# MAGIC %md ### Hyperoptの`fmin()`を用いてチューニングアルゴリズムを実行
# MAGIC 
# MAGIC テストすべきハイパーパラメーター空間の最大ポイント数、すなわち、フィッテイングして評価するモデルの最大数を`max_evals`で設定します。

# COMMAND ----------

argmin = fmin(fn=objective, space=search_space, algo=algo, max_evals=16)

# COMMAND ----------

# ベストなCの値を表示
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md ## パート2. Apache SparkとMLflowを用いた分散チューニング
# MAGIC 
# MAGIC チューニングを分散するには、`fmin()`の引数に`SparkTrials`と呼ばれる`Trials`クラスを追加します。
# MAGIC 
# MAGIC `SparkTrials`は2つのオプションの引数を受け取ります:
# MAGIC * `parallelism`: 同時にフィット、評価するモデルの数。デフォルトは利用可能なSparkタスクのスロット数です。
# MAGIC * `timeout`: `fmin()`を実行できる最大時間(秒数)です。デフォルトは時間制限はありません。
# MAGIC 
# MAGIC このサンプルでは、Cmdで定義された非常にシンプルな目的関数を使用します。この場合、関数はクイックに実行され、Sparkジョブの起動のオーバーヘッドが計算時間の大部分を占めるので、分散処理の場合は計算処理はさらに時間がかかります。典型的な現実的なユースケースでは、目的関数はより複雑となり、分散させるために`SparkTrails`を用いることで、シングルマシンのチューニングよりも計算処理が高速になります。
# MAGIC 
# MAGIC デフォルトで自動MLflowトラッキングが有効化されています。使用するには、サンプルで示しているように`fmin()`を呼び出す前に`mlflow.start_run()`を呼び出してください。

# COMMAND ----------

from hyperopt import SparkTrials

# SparkTrialsクラスのAPIドキュメントを表示するには、以下の行のコメントを解除してください。
# help(SparkTrials)

# COMMAND ----------

spark_trials = SparkTrials()

with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)

# COMMAND ----------

# ベストなCの値を表示
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md 
# MAGIC ノートブックに関連づけられているMLflowエクスペリメントを参照するには、ノートブック右側にあるフラスコアイコンをクリックします。そこでは、すべてのMLflowランを参照することができます。MLflow UIからランを参照するには、MLflowランの右端にある右上向き矢印のアイコンをクリックします。
# MAGIC 
# MAGIC `C`のチューニングによる効果を検証するには:
# MAGIC 1. 生成されたランを選択し、**Compare**をクリックします。
# MAGIC 1. 散布図でX軸に**C**を選択、Y軸に**loss**を選択します。
