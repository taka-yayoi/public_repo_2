// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC ## GraphFramesによるグラフ分析
// MAGIC 
// MAGIC このノートブックでは、[spark-packages.orgで利用できるGraphFramesパッケージ](http://spark-packages.org/package/graphframes/graphframes)を用いた基本的なグラフ分析をウォークスルーします。このノートブックのゴールは、グラフ分析を行うためにどの様にGraphFramesを使うのかを説明することです。[Kaggle](https://www.kaggle.com/benhamner/sf-bay-area-bike-share/downloads/sf-bay-area-bike-share.zip)のベイエリアのバイクシェアのデータを用いてこれを行います。
// MAGIC 
// MAGIC #### グラフ理論とグラフ処理
// MAGIC 
// MAGIC グラフ処理は多くのユースケースに適用される重要な分析の観点です。基本的に、グラフ理論とグラフ処理は、異なるノードとエッジ間の関係性の定義に関するものです。ノード(vertex)はユニットであり、エッジはそれらの間に定義される関係性です。これは、関係性を理解し、重みづけを行うソーシャルネットワーク分析や、[PageRank](https://en.wikipedia.org/wiki/PageRank)のようなアルゴリズムの実行において非常に役立ちます。
// MAGIC 
// MAGIC いくつかのビジネスユースケースにおいては、ソーシャルネットワークの中心人物[友達グループで誰が一番人気なのか]、文献ネットワークにおける論文の重要性[どの論文が最も引用されているのか]、Webページのランキングの原因を探すと言ったことが考えられます！
// MAGIC 
// MAGIC #### グラフとバイク移動データ
// MAGIC 
// MAGIC 上で述べた様に、このサンプルではベイエリアのバイクシェアデータを使います。皆様が分析に慣れる様に、すべてのノードをステーションにし、2つのステーションを結ぶエッジが個々の移動になります。これは、*有向*グラフを作成します。
// MAGIC 
// MAGIC **その他のリファレンス:**
// MAGIC * [Graph Theory on Wikipedia](https://en.wikipedia.org/wiki/Graph_theory)
// MAGIC * [PageRank on Wikipedia](https://en.wikipedia.org/wiki/PageRank)
// MAGIC 
// MAGIC #### **目次**
// MAGIC * **データフレームの作成**
// MAGIC * **インポート**
// MAGIC * **グラフの構築**
// MAGIC * **ステーション間の移動**
// MAGIC * **入次数(in degree)と出次数(out degree)**

// COMMAND ----------

// MAGIC %md ### データフレームの作成

// COMMAND ----------

val bikeStations = spark.sql("SELECT * FROM takaakiyayoi_db.station_csv")
val tripData = spark.sql("SELECT * FROM takaakiyayoi_db.trip_csv")

// COMMAND ----------

display(bikeStations)

// COMMAND ----------

display(tripData)

// COMMAND ----------

// MAGIC %md 
// MAGIC 適切な型が適切なカラムに割り当てられていることを確認するために、正確なスキーマを確認することは多くの場合有用です。

// COMMAND ----------

bikeStations.printSchema()
tripData.printSchema()

// COMMAND ----------

// MAGIC %md 
// MAGIC ### インポート
// MAGIC 
// MAGIC 続ける前にいくつかのインポートが必要です。データフレームの操作を簡単にするさまざまなSQL関数ををインポートし、GraphFramesに必要な全てをインポートします。

// COMMAND ----------

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import org.graphframes._

// COMMAND ----------

// MAGIC %md
// MAGIC ### グラフの構築
// MAGIC 
// MAGIC データをインポートした後に必要なのはグラフの構築です。これを行うために2つのことが必要です。ノード(vertex)の構造を構築し、エッジの構造を構築します。GraphFramesの素晴らしいところは、このプロセスが信じられないほどシンプルだということです。必要なのは、ノードのテーブルの別個の**id**値を取得し、エッジテーブルの起点と終点のステーションをそれぞれ**src**と**dst**に変更するということです。これは、GraphFramesにおけるノードとエッジに必要な決まり事です。

// COMMAND ----------

val stationVertices = bikeStations
  .distinct()

val tripEdges = tripData
  .withColumnRenamed("start_station_name", "src")
  .withColumnRenamed("end_station_name", "dst")

// COMMAND ----------

display(stationVertices)

// COMMAND ----------

display(tripEdges)

// COMMAND ----------

// MAGIC %md 
// MAGIC これでグラフを構築できます。
// MAGIC 
// MAGIC また、ここでグラフへの入力データフレームをキャッシュしておきます。

// COMMAND ----------

val stationGraph = GraphFrame(stationVertices, tripEdges)

tripEdges.cache()
stationVertices.cache()

// COMMAND ----------

println("Total Number of Stations: " + stationGraph.vertices.count)
println("Total Number of Trips in Graph: " + stationGraph.edges.count)
println("Total Number of Trips in Original Data: " + tripData.count)// sanity check

// COMMAND ----------

// MAGIC %md
// MAGIC ### ステーション間の移動
// MAGIC 
// MAGIC よく尋寝られる質問は、データセットにおいて最も共通している目的地が何かということです。グルーピングオペレーターとエッジのカウントを組み合わせることで、これを行うことができます。これは、エッジを除外した新たなグラフを作り出し、意味的に同じエッジすべての合計値となります。この様に考えてみましょう: 全く同じステーションAからステーションBへの移動回数が存在し、単にこれらをカウントするだけです！
// MAGIC 
// MAGIC 以下のクエリーでは、最も共通するステーション間移動を抽出し、トップ10を表示しています。

// COMMAND ----------

val topTrips = stationGraph
  .edges
  .groupBy("src", "dst")
  .count()
  .orderBy(desc("count"))
  .limit(10)

display(topTrips)

// COMMAND ----------

// MAGIC %md 
// MAGIC 上の結果から、特定のノードがカルトレインのステーションであることが重要であることがわかります！これらは自然な接続点であり、車を使わない方法で、これらのバイクシェアプログラムを用いてAからBに移動するには最も人気のある利用法なのでしょう！

// COMMAND ----------

// MAGIC %md 
// MAGIC ### 入次数(in degree)と出次数(out degree)
// MAGIC 
// MAGIC この例では有向グラフを使っていることを思い出してください。これは、移動には方向があること - ある地点からある地点へ - を意味します。これによって、あなたが活用できる分析はさらにリッチなものになります。特定のステーションに移動する数や、特定のステーションから移動する数を見つけ出すことができます。
// MAGIC 
// MAGIC 通常、この情報で並び替えを行い、インバウンドとアウトバウンドの移動が多いステーションを見つけ出すことができます！詳細に関しては、[Vertex Degrees](http://mathworld.wolfram.com/VertexDegree.html)の定義をチェックしてみてください。
// MAGIC 
// MAGIC このプロセスを定義したので、次に進んでインバウンドとアウトバウンドの移動が多いステーションを見つけましょう。

// COMMAND ----------

val inDeg = stationGraph.inDegrees
display(inDeg.orderBy(desc("inDegree")).limit(5))

// COMMAND ----------

val outDeg = stationGraph.outDegrees
display(outDeg.orderBy(desc("outDegree")).limit(5))

// COMMAND ----------

// MAGIC %md 
// MAGIC もう一つの興味深い質問は、入次数が最も高いが、出次数が少ないステーションがどれかということです。すなわち、どのステーションが純粋な移動のシンクになっているかということです。移動がそこで終了しますが、ほとんどそこから移動を開始しない場所です。

// COMMAND ----------

val degreeRatio = inDeg.join(outDeg, inDeg.col("id") === outDeg.col("id"))
  .drop(outDeg.col("id"))
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio")

degreeRatio.cache()
  
display(degreeRatio.orderBy(desc("degreeRatio")).limit(10))

// COMMAND ----------

// MAGIC %md
// MAGIC 出次数に対する入次数の比率が最も低いステーションを得ることで、同様のことを行うことができます。これは、そのステーションからの移動開始が多いが、移動がそこで終わることがあまりないことを意味します。これは、基本的に上で見たのと逆の内容になります。

// COMMAND ----------

display(degreeRatio.orderBy(asc("degreeRatio")).limit(10))

// COMMAND ----------

// MAGIC %md
// MAGIC 上の分析から得られる結論は比較的わかりやすいものです。高い値は出てくるよりも入る移動が多いことを意味し、低い値はそのステーションからの移動開始が多いが、移動終了は少ないということです！
// MAGIC 
// MAGIC このノートブックから何かしらの価値を得ていただけたら幸いです！グラフ構造は探し始めるとどこにでもあり、GraphFramesが分析を容易にしてくれることを願っています！
