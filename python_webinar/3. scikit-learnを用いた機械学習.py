# Databricks notebook source
# MAGIC %md
# MAGIC # scikit-learnを用いた機械学習
# MAGIC 
# MAGIC * 機械学習とは何か？
# MAGIC   * 機械学習のタイプ
# MAGIC * トレーニング - テストデータセットの分割
# MAGIC * 線形回帰モデルを構築するために `sklearn` を活用
# MAGIC * ワンホットエンコーディング
# MAGIC * パイプライン
# MAGIC * 評価メトリクス
# MAGIC 
# MAGIC 必要に応じて[scikit-learnのドキュメント](https://scikit-learn.org/stable/user_guide.html)や[pandasのドキュメント](https://pandas.pydata.org/pandas-docs/stable/index.html)を参照します。そして、 https://github.com/nytimes/covid-19-data にあるNew York Times COVID-19 US Statesデータセットのデータを分析します。
# MAGIC 
# MAGIC **免責: このデータセットにおいては線形回帰が最適のアルゴリズムではありませんが、ここではscikit-learnの使い方を説明するために線形回帰を使用します。**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 機械学習とは何か？
# MAGIC 
# MAGIC * 明示的なプログラミングを行うことなしにデータからパターンを学習します。
# MAGIC * 特徴量をアウトプットにマッピングする関数です。
# MAGIC 
# MAGIC ![](https://brookewenig.com/img/DL/al_ml_dl.png)
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [今さら聞けない機械学習](https://qiita.com/taka_yayoi/items/51583a581ce5a6ba6558)
# MAGIC - [今さら聞けないシリーズ 機械学習とMLOpsとは – Databricks](https://www.databricks.com/p/webinar/jp-mlops-beginner-series)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 機械学習のタイプ
# MAGIC * 教師あり学習
# MAGIC   * 回帰 <img src="https://miro.medium.com/max/640/1*LEmBCYAttxS6uI6rEyPLMQ.png" style="height: 250px; padding: 10px"/>
# MAGIC   * 分類
# MAGIC     <img src="https://cdn3-www.dogtime.com/assets/uploads/2018/10/puppies-cover.jpg" style="height: 250px; padding: 10px"/>
# MAGIC     <img src="https://images.unsplash.com/photo-1529778873920-4da4926a72c2?ixlib=rb-1.2.1&w=1000&q=80" style="height: 250px; padding: 10px"/>
# MAGIC * 教師なし学習
# MAGIC <img src="https://www.iotforall.com/wp-content/uploads/2018/01/Screen-Shot-2018-01-17-at-8.10.14-PM.png" style="height: 250px; padding: 10px"/>
# MAGIC * 強化学習
# MAGIC <img src="https://brookewenig.com/img/ReinforcementLearning/Rl_agent.png" style="height: 250px; padding: 10px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 本日はシンプルにスタートし、教師あり学習(回帰)問題にフォーカスします。ここでは、COVID-19による死者数を予測する線形回帰モデルを使用します。

# COMMAND ----------

# MAGIC %fs ls databricks-datasets/COVID/covid-19-data/us-states.csv

# COMMAND ----------

import pandas as pd

df = pd.read_csv("/dbfs/databricks-datasets/COVID/covid-19-data/us-states.csv")
df.head()
df.date.max()

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## 感染者数と死者数の関係

# COMMAND ----------

# ノートブックでプロットを出力できるようにします
%matplotlib inline

# COMMAND ----------

# 2020-05-01のデータにフィルタリングします
df_05_01 = df[df["date"] == "2020-05-01"]

ax = df_05_01.plot(x="cases", y="deaths", kind="scatter", 
                   figsize=(12,8), s=100, title="Deaths vs Cases on 2020-05-01 - All States")

df_05_01[["cases", "deaths", "state"]].apply(lambda row: ax.text(*row), axis=1);

# COMMAND ----------

# MAGIC %md
# MAGIC ## ニューヨークとニュージャージーは外れ値と言えます

# COMMAND ----------

# New YorkとNew Jersey以外の州にフィルタリングします
not_ny = df[(df["state"] != "New York") & (df["state"] != "New Jersey")]
not_ny.head()

# COMMAND ----------

# 2020-05-01のデータにフィルタリングします
not_ny_05_01 = not_ny[not_ny["date"] == "2020-05-01"]

ax = not_ny_05_01.plot(x="cases", y="deaths", kind="scatter", 
                   figsize=(12,8), s=50, title="Deaths vs Cases on 2020-05-01 - All States but NY and NJ")

not_ny_05_01[["cases", "deaths", "state"]].apply(lambda row: ax.text(*row), axis=1);

# COMMAND ----------

# MAGIC %md
# MAGIC ## New YorkとCaliforniaにおけるCOVID-19の死者数の比較

# COMMAND ----------

df_ny_cali = df[(df["state"] == "New York") | (df["state"] == "California")]

# 両方の州における死者数の時間変換をプロットできるように、df_ny_caliデータフレームをピボットしましょう
df_ny_cali_pivot = df_ny_cali.pivot(index='date', columns='state', values='deaths').fillna(0)
df_ny_cali_pivot

# COMMAND ----------

df_ny_cali_pivot.plot.line(title="Deaths 2020-01-25 to 2020-05-01 - CA and NY", figsize=(12,8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング - テストデータセットの分割
# MAGIC 
# MAGIC ![](https://brookewenig.com/img/IntroML/trainTest.png)

# COMMAND ----------

# MAGIC %md
# MAGIC これは時系列データなのでランダムに分割するのではなく、モデルのトレーニングには3/1から4/7のデータを使い、4/8から4/14の値を予測することでモデルをテストします。

# COMMAND ----------

train_df = df[(df["date"] >= "2020-03-01") & (df["date"] <= "2020-04-07")]
test_df = df[df["date"] > "2020-04-07"]

X_train = train_df[["cases"]]
y_train = train_df["deaths"]

X_test = test_df[["cases"]]
y_test = test_df["deaths"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 線形回帰
# MAGIC 
# MAGIC * ゴール: 最もフィットする直線を見つけ出す
# MAGIC $$\hat{y} = w_0 + w_1x$$
# MAGIC 
# MAGIC $$\{y} ≈ \hat{y} + ϵ$$
# MAGIC * *x*: 特徴量
# MAGIC * *y*: ラベル
# MAGIC 
# MAGIC ![](https://miro.medium.com/max/640/1*LEmBCYAttxS6uI6rEyPLMQ.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ここでは、scikit-learnの[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)モデルをフィッティングします。

# COMMAND ----------

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)
print(f"num_deaths = {lr.intercept_:.4f} + {lr.coef_[0]:.4f}*cases")

# COMMAND ----------

# MAGIC %md
# MAGIC むむむ...感染者数が`0`の場合、COVID-19の死者はいないはずですので切片は`0`であるべきです。

# COMMAND ----------

lr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
print(f"num_deaths = {lr.coef_[0]:.4f}*cases")

# COMMAND ----------

# MAGIC %md
# MAGIC これによって、使用しているデータセットにおいては死亡率が3.5%であることを示唆しています。しかし、我々はいくつかの州においてはさらに高い死亡率になっていることを知っています。それでは州を特徴量として追加しましょう！

# COMMAND ----------

# MAGIC %md
# MAGIC ## ワンホットエンコーディング
# MAGIC 
# MAGIC 州のように数値では無い特徴量をどのように扱ったらいいのでしょうか？
# MAGIC 
# MAGIC あるアイデア:
# MAGIC * 非数値を表現する単一の数値特徴量を作成する
# MAGIC * カテゴリー変数の特徴量を作成する:
# MAGIC   * state = {'New York', 'California', 'Louisiana'}
# MAGIC   * 'New York' = 1, 'California' = 2, 'Louisiana' = 3
# MAGIC   
# MAGIC しかし、これではCaliforniaがNew Yorkの2倍ということになってしまいます！
# MAGIC 
# MAGIC より良いアイデア:
# MAGIC * カテゴリーごとの`ダミー`特徴量を作成する
# MAGIC * 'New York' => [1, 0, 0], 'California' => [0, 1, 0], 'Louisiana' => [0, 0, 1]
# MAGIC 
# MAGIC このテクニックは["One Hot Encoding"](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)として知られるものです。

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

X_train = train_df[["cases", "state"]]
X_test = test_df[["cases", "state"]]

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc.fit(X_train).transform(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 形状を確認してみましょう。

# COMMAND ----------

enc.fit(X_train).transform(X_train).shape

# COMMAND ----------

# MAGIC %md
# MAGIC うわっ、感染者数の変数もワンホットエンコーディングしてしまいました。

# COMMAND ----------

enc.categories_

# COMMAND ----------

# MAGIC %md
# MAGIC 特定のカラムにのみワンホットエンコーディングを適用するように、[column transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)が必要となります。

# COMMAND ----------

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("enc", enc, ["state"])], remainder="passthrough")
ct.fit_transform(X_train).shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプライン
# MAGIC 
# MAGIC [パイプライン](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)を用いることで、一連のデータ変換処理をチェーンすることができます。また、このようにすることで、トレーニングセットに適用したすべてのオペレーションが、同じ順序でテストセットに適用されることを保証できます。

# COMMAND ----------

from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[("ct", ct), ("lr", lr)])
pipeline_model = pipeline.fit(X_train, y_train)

y_pred = pipeline_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 州が異なるとどうなるのか？
# MAGIC 
# MAGIC 特徴量を追加することで、感染者数特徴量の係数も変化していることに気づきます。

# COMMAND ----------

print(f"num_deaths = {pipeline_model.steps[1][1].coef_[-1]:.4f}*cases + state_coef")

# COMMAND ----------

import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)

categories = pipeline_model.steps[0][1].transformers[0][1].categories_[1]

pd.DataFrame(zip(categories, pipeline_model.steps[1][1].coef_[:-1]), columns=["State", "Coefficient"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価メトリクス
# MAGIC 
# MAGIC ![](https://brookewenig.com/img/IntroML/RMSE.png)

# COMMAND ----------

# MAGIC %md
# MAGIC [sklearn.metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mean_squared_error)を用いて、データセットのMSEとRMSEを計算しましょう。

# COMMAND ----------

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE is {mse:.1f}, RMSE is {rmse:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測結果の可視化

# COMMAND ----------

pred = pd.concat([test_df.reset_index(drop=True), pd.DataFrame(y_pred, columns=["predicted_deaths"])], axis=1)
pred

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC やりました！scikit-learnを用いて機械学習パイプラインを構築することができました！
# MAGIC 
# MAGIC scikit-learnを探索し続ける場合には、[UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)や[Kaggle](https://www.kaggle.com/)のデータセットをチェックしてみてください！
