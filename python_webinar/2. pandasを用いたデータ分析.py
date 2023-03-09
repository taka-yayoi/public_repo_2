# Databricks notebook source
# MAGIC %md
# MAGIC # 野心的なデータサイエンティスト向けデータ分析のご紹介
# MAGIC ## `pandas`を用いたデータ分析
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下を学びます:<br>
# MAGIC  - `pandas`を使う動機づけ
# MAGIC  - `pandas`とその歴史
# MAGIC  - COVID-19データセットのインポート
# MAGIC    * `pd.read_csv()`
# MAGIC  - データの要約
# MAGIC    * `head`, `tail`, `shape`
# MAGIC    * `sum`, `min`, `count`, `mean`, `std`
# MAGIC    * `describe`
# MAGIC  - データのスライスと加工
# MAGIC    * スライス, `loc`, `iloc`
# MAGIC    * `value_counts`
# MAGIC    * `drop`
# MAGIC    * `sort_values`
# MAGIC    * フィルタリング
# MAGIC  - データのグルーピングおよび集計関数の実行
# MAGIC    * `groupby`
# MAGIC  - 欠損値、重複への対応
# MAGIC    * `isnull`
# MAGIC    * `unique`, `drop_duplicates`
# MAGIC    * `fillna`
# MAGIC  - 可視化
# MAGIC    * ヒストグラム
# MAGIC    * 散布図
# MAGIC    * 折れ線グラフ
# MAGIC  
# MAGIC [pandasチートシート](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)が役に立ちます。 [`pandas`のドキュメント](https://pandas.pydata.org/docs/)もご覧ください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### `pandas`を使う動機づけ
# MAGIC 
# MAGIC 大きな絵からスタートしましょう...<br><br>
# MAGIC 
# MAGIC * 人類は道具を使う動物です
# MAGIC * コンピューターは我々が作り出した最もパワフルなツールの一つです
# MAGIC * コードを書くことで、これらのツールのフルパワーを解き放つことができます

# COMMAND ----------

# MAGIC %md
# MAGIC Ok、クールだね。でもなぜ`pandas`?<br><br>
# MAGIC 
# MAGIC * これまで以上にデータは意思決定において重要となっています。
# MAGIC * Excelは素晴らしいものですが、もし...
# MAGIC   - 毎日新規のデータに対して再実行できるように分析を自動化したいとしたら？
# MAGIC   - 同僚と共有できるようにコードベースを構築したいとしたら？
# MAGIC   - ビジネス上の意思決定につながるより堅牢な分析を必要としたら？
# MAGIC   - 機械学習を行いたいとしたら？
# MAGIC * Pythonにおけるデータ分析やデータサイエンティストによって使用されるコアのライブラリの一つが
# MAGIC 
# MAGIC `pandas`にようこそ...

# COMMAND ----------

# MAGIC %md
# MAGIC ### `pandas`とその歴史
# MAGIC 
# MAGIC `pandas`は、Pythonプログラミング言語上で開発された、高速、パワフル、柔軟かつ、簡単に使用できるオープンソースのデータ分析、データ操作ツールです。
# MAGIC 
# MAGIC ハイライト:
# MAGIC 
# MAGIC - 2008年に開発され、2009年にオープンソース化されました。
# MAGIC - インデックスがインテグレーションされたデータ操作のための高速かつ効率的な**データフレームオブジェクト**
# MAGIC - インメモリーのデータ構造と様々なフォーマット間の**データ読み書き**ツール: CSV、テキストファイル、Microsoft Excel、SQLデータベース、高速なHDF5フォーマット。
# MAGIC - インテリジェントな**データアライメント**とインテグレーションされた**欠損データ**への対応: 計算時に自動でラベルベースのアライメントを取得し、汚いデータを容易に綺麗な状態に変換。
# MAGIC - データセットに対する柔軟な **リシェイプとピボット**。
# MAGIC - 大規模データセットに対するインテリジェントなラベルベースの **スライス、ファンシーインデックス、サブセット作成**。
# MAGIC - **サイズ可変性**を持つデータ構造に対するカラムの追加・削除。
# MAGIC - パワフルな**group by**を用いたデータの集計、変換によるデータセットに対するsplit-apply-combineオペレーション。
# MAGIC - データセットに対する高性能の**マージとジョイン**。
# MAGIC - 階層型軸**インデックス**による、低次元データ構造における直感的な高次元データ操作手段の提供。
# MAGIC - **時系列**機能: 日付範囲の生成、頻度変換、移動ウィンドウ統計情報、日付シフト、ラギング。ドメイン固有の時間オフセットの作成、データ損失なしの時系列データのジョインもサポート。
# MAGIC - Cython、Cで記述されたクリティカルなコードパスにより高度に**最適化された**パフォーマンス。
# MAGIC - Pythonとpandasは、金融、神経科学、経営、統計学、広告、Web分析などを含むさまざまな**学術、商用の領域**で活用されています。
# MAGIC 
# MAGIC [書籍もチェックしてみてください。](https://www.amazon.com/gp/product/1491957662/)

# COMMAND ----------

# MAGIC %md
# MAGIC ### COVID-19データセットのインポート

# COMMAND ----------

# MAGIC %md
# MAGIC フォルダー構造を検索するために `%sh ls` を使います。

# COMMAND ----------

# MAGIC %sh ls /dbfs/databricks-datasets/COVID/

# COMMAND ----------

# MAGIC %sh ls /dbfs/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports

# COMMAND ----------

# MAGIC %md
# MAGIC CSVファイルの最初の数行を表示するために `%sh head` を使います。

# COMMAND ----------

# MAGIC %sh head /dbfs/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/04-11-2020.csv

# COMMAND ----------

# MAGIC %md
# MAGIC `pandas`をインポートします。別名を`pd`とします。

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC CSVファイルを読み込みます。これによって`データフレーム`が生成されます。

# COMMAND ----------

dbutils.fs.cp("dbfs:/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/04-11-2020.csv", "file:/tmp/covid.csv")
pd.read_csv("file:/tmp/covid.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC それでは数行のコードを組み合わせて、再利用できるように`データフレーム`を保存してみましょう。

# COMMAND ----------

import pandas as pd

df = pd.read_csv("file:/tmp/covid.csv")

df

# COMMAND ----------

# MAGIC %md
# MAGIC ### データの要約

# COMMAND ----------

# MAGIC %md
# MAGIC 最初にtabによるオートコンプリートについて話させてください。

# COMMAND ----------

#df. # この行のコメントを解除し、"."の後にカーソルを移動し'tab'を押します。」

# COMMAND ----------

# MAGIC %md
# MAGIC ヘルプが必要ですか？

# COMMAND ----------

help(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC データの最初の数行と最後の数行を見てみましょう。

# COMMAND ----------

df.head()

# COMMAND ----------

df.tail(2)

# COMMAND ----------

# MAGIC %md
# MAGIC データセットは何行でしょうか？

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC データを要約しましょう。

# COMMAND ----------

# df.sum()
# df.min()
# df.max()
# df.count()
df.mean()
# df.std()

# COMMAND ----------

# MAGIC %md
# MAGIC これらのサマリー統計情報を集計して表示することができます...

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### データのスライスと加工

# COMMAND ----------

# MAGIC %md
# MAGIC 感染者数だけを取り出します。

# COMMAND ----------

df['Confirmed']

# COMMAND ----------

# MAGIC %md
# MAGIC 国と感染者数を取り出します。

# COMMAND ----------

df.columns

# COMMAND ----------

df[['Country_Region', 'Confirmed']]

# COMMAND ----------

# MAGIC %md
# MAGIC 新たなカラム `Date` を作成します。

# COMMAND ----------

import datetime

df["Date"] = datetime.date(2020, 4, 11)

# COMMAND ----------

df["Date"].head()

# COMMAND ----------

# MAGIC %md
# MAGIC データフレームの最初の10行を取り出してスライスします。

# COMMAND ----------

df.loc[:10, ['Country_Region', 'Confirmed']]
# df.loc[0:10, ['Country_Region', 'Confirmed']] # 同じ処理です

# COMMAND ----------

# MAGIC %md
# MAGIC 最初の行の最初のカラムだけを返却します。

# COMMAND ----------

df.iloc[0, 0]

# COMMAND ----------

# MAGIC %md
# MAGIC 国ごとにいくつの地域があるのでしょうか？

# COMMAND ----------

df["Country_Region"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC FIPSとはなんでしょうか？不要なので削除します。

# COMMAND ----------

df = df.drop("FIPS", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 感染者数でソートします。

# COMMAND ----------

df.sort_values("Confirmed", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC USで起きていることだけを見てみましょう。

# COMMAND ----------

df[df["Country_Region"] == "US"]

# COMMAND ----------

# MAGIC %md
# MAGIC 特定の場所で起きていることだけを見てみましょう。

# COMMAND ----------

df[
    (df["Country_Region"] == "US")
    & (df["Province_State"] == "California")
    & (df["Admin2"] == "San Francisco")
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### データのグルーピングと集計関数の実行

# COMMAND ----------

# MAGIC %md
# MAGIC 感染者数が最も多い国はどこでしょうか？

# COMMAND ----------

df.groupby("Country_Region")

# COMMAND ----------

# MAGIC %md
# MAGIC データをグルーピングし合計を計算します。**集計関数はスカラー値(単一の値)を返却することに注意してください。**

# COMMAND ----------

df.groupby("Country_Region")["Confirmed"].sum().sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 感染者数が最も多いUSの州はどこでしょうか？

# COMMAND ----------

df[df['Country_Region'] == "US"].groupby("Province_State")["Confirmed"].sum().sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 欠損データと重複への対応

# COMMAND ----------

# MAGIC %md
# MAGIC null値はあるのでしょうか？

# COMMAND ----------

df.isnull().tail()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ユニークな国の数はいくつでしょうか？

# COMMAND ----------

df['Country_Region'].unique().shape

# COMMAND ----------

# MAGIC %md
# MAGIC 同じことを行う別の方法です。

# COMMAND ----------

df['Country_Region'].drop_duplicates()

# COMMAND ----------

df.fillna("NO DATA AVAILABLE").tail(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 可視化
# MAGIC    * ヒストグラム
# MAGIC    * 散布図
# MAGIC    * 折れ線グラフ

# COMMAND ----------

import matplotlib.pyplot as plt

%matplotlib inline

# COMMAND ----------

us_subset_df = df[df["Country_Region"] == "US"]

# COMMAND ----------

# MAGIC %md
# MAGIC USの州と地域ごとの死者数の _分布_ はどうなっているのでしょうか？

# COMMAND ----------

us_subset_df.groupby("Province_State")["Deaths"].sum().hist()

# COMMAND ----------

us_subset_df.groupby("Province_State")["Deaths"].sum().hist(bins=30)

# COMMAND ----------

# MAGIC %md
# MAGIC 死者数に比べて感染者数はどうなっているのでしょうか？

# COMMAND ----------

us_subset_df.plot.scatter(x="Confirmed", y="Deaths")

# COMMAND ----------

us_subset_df[us_subset_df["Deaths"] < 1000].plot.scatter(x="Confirmed", y="Deaths")

# COMMAND ----------

# MAGIC %md
# MAGIC 利用できるすべての日のデータをインポートします。

# COMMAND ----------

src_path_base = "dbfs:/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/"
dest_path_base = "file:/tmp/covid_daily_reports/"

files = [
 '11-21-2020.csv',
 '11-22-2020.csv',
 '11-23-2020.csv',
 '11-24-2020.csv',
 '11-25-2020.csv',
 '11-26-2020.csv',
 '11-27-2020.csv',
 '11-28-2020.csv',
 '11-29-2020.csv',
 '11-30-2020.csv'
]

dfs = []

for file in files:
  filename = dest_path_base+file
  dbutils.fs.cp(src_path_base+file, filename)
  
  temp_df = pd.read_csv(filename)
  temp_df.columns = [c.replace("/", "_") for c in temp_df.columns]
  temp_df.columns = [c.replace(" ", "_") for c in temp_df.columns]
  
  month, day, year = filename.split("/")[-1].replace(".csv", "").split("-")
  d = datetime.date(int(year), int(month), int(day))
  temp_df["Date"] = d

  dfs.append(temp_df)
  
all_days_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
all_days_df = all_days_df.drop(["Lat", "Long_", "FIPS", "Combined_Key", "Last_Update"], axis=1)

# COMMAND ----------

all_days_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 時間と共に感染病はどのように広がったのでしょうか？

# COMMAND ----------

all_days_df.groupby("Date")["Confirmed"].sum().plot(title="Confirmed Cases over Time", rot=45)

# COMMAND ----------

# MAGIC %md
# MAGIC これをケースのタイプごとにブレークダウンします。

# COMMAND ----------

all_days_df.groupby("Date")["Confirmed", "Deaths", "Recovered"].sum().plot(
    title="Confirmed, Deaths, Recovered over Time", rot=45
)

# COMMAND ----------

# MAGIC %md
# MAGIC 特定の場所における増加状況はどうなっているでしょうか？

# COMMAND ----------

(
    all_days_df[
        (all_days_df["Country_Region"] == "US")
        & (all_days_df["Province_State"] == "California")
        & (all_days_df["Admin2"] == "San Francisco")
    ]
    .groupby("Date")["Confirmed", "Deaths", "Recovered"]
    .sum()
    .plot(title="Confirmed, Deaths, Recovered over Time", rot=45)
)

# COMMAND ----------

# MAGIC %md
# MAGIC 関数でラッピングして自分で実行してみましょう！

# COMMAND ----------

def plotMyCountry(Country_Region, Province_State, Admin2):
    (
        all_days_df[
            (all_days_df["Country_Region"] == Country_Region)
            & (all_days_df["Province_State"] == Province_State)
            & (all_days_df["Admin2"] == Admin2)
        ]
        .groupby("Date")["Confirmed", "Deaths", "Recovered"]
        .sum()
        .plot(title="Confirmed, Deaths, Recovered over Time", rot=45)
    )


plotMyCountry("US", "New York", "Suffolk")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
