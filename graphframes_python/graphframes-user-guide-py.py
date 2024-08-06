# Databricks notebook source
# MAGIC %md # GraphFramesユーザーガイド (Python)
# MAGIC
# MAGIC このノートブックでは、[GraphFrames User Guide](https://graphframes.github.io/graphframes/docs/_site/user-guide.html)の例をデモンストレーションします。
# MAGIC
# MAGIC ## 要件
# MAGIC
# MAGIC このノートブックでは、機械学習ランタイムが必要となります。

# COMMAND ----------

from functools import reduce
from pyspark.sql import functions as F
from graphframes import GraphFrame

# COMMAND ----------

# MAGIC %md
# MAGIC [python \- PYSPARK: how to visualize a GraphFrame? \- Stack Overflow](https://stackoverflow.com/questions/45720931/pyspark-how-to-visualize-a-graphframe)

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt

def PlotGraph(edge_list):
    Gplot = nx.Graph()
    for row in edge_list.select("src", "dst").take(1000):
        Gplot.add_edge(row["src"], row["dst"])

    plt.subplot(121)
    nx.draw(Gplot, with_labels=True, font_weight="bold")

# COMMAND ----------

# MAGIC %md ## GraphFramesの作成
# MAGIC
# MAGIC ユーザーは頂点(vertex)とエッジ(edge)のデータフレームからGraphFramesを作成できます。
# MAGIC
# MAGIC * Vertexデータフレーム: Vertexデータフレームには、グラフにおけるそれぞれの頂点に対するユニークなIDを示す"id"という名前の特殊なカラムを含める必要があります。
# MAGIC * Edgeデータフレーム: Edgeデータフレームには、2つの特殊なカラムが必要です: "src" (エッジのソースとなる頂点のID) と "dst" (エッジのディスティネーションとなる頂点のID)です。
# MAGIC
# MAGIC 両方のデータフレームには、任意のその他のカラムを含めることができます。これらのカラムでは、頂点やエッジの属性を表現することができます。

# COMMAND ----------

# MAGIC %md はじめに頂点を作成します:

# COMMAND ----------

vertices = spark.createDataFrame([
    ("a", "Alice", 34),
    ("b", "Bob", 36),
    ("c", "Charlie", 30),
    ("d", "David", 29),
    ("e", "Esther", 32),
    ("f", "Fanny", 36),
    ("g", "Gabby", 60)],
    ["id", "name", "age"])

# COMMAND ----------

# MAGIC %md 次に幾つかのエッジを作成します:

# COMMAND ----------

edges = spark.createDataFrame([
    ("a", "b", "friend"),
    ("b", "c", "follow"),
    ("c", "b", "follow"),
    ("f", "c", "follow"),
    ("e", "f", "follow"),
    ("e", "d", "friend"),
    ("d", "a", "friend"),
    ("a", "e", "friend")], 
    ["src", "dst", "relationship"])

# COMMAND ----------

# MAGIC %md これらの頂点とエッジからグラフを作成します:

# COMMAND ----------

g = GraphFrame(vertices, edges)
print(g)

# COMMAND ----------

PlotGraph(g.edges)

# COMMAND ----------

# このサンプルnのグラフはGraphFramesパッケージでも提供されています。
from graphframes.examples import Graphs
same_g = Graphs(spark).friends()
print(same_g)

# COMMAND ----------

# MAGIC %md ## 基本的なグラフとデータフレームのクエリー
# MAGIC
# MAGIC GraphFramesでは、ノードの度数のようないくつかのシンプルなグラフクエリーを提供します。
# MAGIC
# MAGIC また、GraphFramesはグラフを頂点とエッジデータフレームのペアとして表現するので、頂点とエッジのデータフレームに対して直接パワフルなクエリーを容易に実行することができます。これらのデータフレームは、GraphFrameのverticesやedgesフィールドから利用できます。

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

# MAGIC %md 頂点の内向きの度数:

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

# MAGIC %md 頂点の外向きの度数:

# COMMAND ----------

display(g.outDegrees)

# COMMAND ----------

# MAGIC %md 頂点の度数:

# COMMAND ----------

display(g.degrees)

# COMMAND ----------

# MAGIC %md 
# MAGIC verticesデータフレームに直接クエリーを実行できます。例えば、グラフにおいて最も若い人物の年齢を特定できます:

# COMMAND ----------

youngest = g.vertices.groupBy().min("age")
display(youngest)

# COMMAND ----------

# MAGIC %md
# MAGIC また、edgesデータフレームにクエリーを行うことができます。例えば、グラフにおける _follow_ リレーションシップの数をカウントします:

# COMMAND ----------

numFollows = g.edges.filter("relationship = 'follow'").count()
print("The number of follow edges is", numFollows)

# COMMAND ----------

# MAGIC %md ## モチーフの特定
# MAGIC
# MAGIC モチーフを用いることで、エッジと頂点を含むより複雑なリレーションシップを構築できます。以下のセルでは、両方向で接続されている頂点とエッジのペアを特定しています。結果は、モチーフのキーによって指定されるカラム名を持つデータフレームとなります。
# MAGIC
# MAGIC APIの詳細に関しては、[GraphFrame User Guide](https://graphframes.github.io/graphframes/docs/_site/user-guide.html#motif-finding)をご覧ください。

# COMMAND ----------

# 両方向で接続されている頂点とエッジのペアを検索します。
motifs = g.find("(a)-[e]->(b); (b)-[e2]->(a)")
display(motifs)

# COMMAND ----------

# MAGIC %md 
# MAGIC 結果はデータフレームとなるので、モチーフをベースにしてより複雑なクエリーを構築することができます。以下のセルでは、どちらかが30歳を上回っているすべての相互リレーションシップを検索します:

# COMMAND ----------

filtered = motifs.filter("b.age > 30 or a.age > 30")
display(filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ステートフルなクエリー
# MAGIC
# MAGIC 上の例のように、ほとんどのモチーフクエリーはステートレスで表現が容易です。次の例では、モチーフのパスを通じて状態を運ぶより複雑なクエリーをデモンストレーションします。このようなクエリーは、GraphFrameモチーフの検索と、検索結果のデータフレームのカラムに対して適用される後続のオペレーションのフィルターを組み合わせることで表現できます。
# MAGIC
# MAGIC 例えば、一連の関数によって定義されるいくつかのプロパティを持つ、4つの頂点のチェーンを特定したいものとします。すなわち、4つの頂点のチェーン`a->b->c->d`において、このような複雑なフィルターにマッチするチェーンのサブセットを特定します:
# MAGIC
# MAGIC * パスの状態の初期化
# MAGIC * 頂点aに基づいて状態を更新
# MAGIC * 頂点bに基づいて状態を更新
# MAGIC * cやdも同様
# MAGIC
# MAGIC 最終的な状態が何かしらの条件に合致したら、フィルターはチェーンを受け入れます。
# MAGIC
# MAGIC 以下のコードスニペットではこのプロセスを説明しています。このコードでは、3つのエッジのうち少なくとも2つが"friend"リレーションシップである、4つの頂点のチェーンを特定します。この例では、状態は現時点での"friend"エッジのカウントとなります。通常、これはデータフレームのカラムとなります。

# COMMAND ----------

# 4つの頂点を持つチェーンを特定します。
chain4 = g.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d)")

# 状態(cnt)とともにシーケンスをクエリー
#  (a) モチーフの次の要素に基づいて状態を更新するメソッドを定義します。
def cumFriends(cnt, edge):
    relationship = F.col(edge)["relationship"]
    return F.when(relationship == "friend", cnt + 1).otherwise(cnt)

#  (b) モチーフの要素のシーケンスに対してメソッドを適用するためにシーケンスオペレーションを活用します。
#   この場合、要素は3つのエッジとなります。
edges = ["ab", "bc", "cd"]
numFriends = reduce(cumFriends, edges, F.lit(0))
    
chainWith2Friends2 = chain4.withColumn("num_friends", numFriends).where(numFriends >= 2)
display(chainWith2Friends2)

# COMMAND ----------

# MAGIC %md ## サブグラフ
# MAGIC
# MAGIC GraphFramesはエッジや頂点に基づいたフィルタリングによってサブグラフを構築するAPIを提供します。これらのフィルターは、以下のような30歳を超える人、かつ、友達も30歳を超えている人のみを含むサブグラフのように、組み合わせることが可能です。

# COMMAND ----------

g2 = g.filterEdges("relationship = 'friend'").filterVertices("age > 30").dropIsolatedVertices()

# COMMAND ----------

display(g2.vertices)

# COMMAND ----------

display(g2.edges)

# COMMAND ----------

# MAGIC %md ## 標準的なグラフアルゴリズム
# MAGIC
# MAGIC GraphFramesでは、数多くの標準的なグラフアルゴリズムがビルトインされています:
# MAGIC
# MAGIC * Breadth-first search (BFS)
# MAGIC * Connected components
# MAGIC * Strongly connected components
# MAGIC * Label Propagation Algorithm (LPA)
# MAGIC * PageRank (regular and personalized)
# MAGIC * Shortest paths
# MAGIC * Triangle count

# COMMAND ----------

# MAGIC %md ###Breadth-first search (BFS)
# MAGIC
# MAGIC age < 32のユーザーから"Esther"を検索します。

# COMMAND ----------

paths = g.bfs("name = 'Esther'", "age < 32")
display(paths)

# COMMAND ----------

# MAGIC %md
# MAGIC エッジのフィルターや最大パス長で検索を制限することもできます。

# COMMAND ----------

filteredPaths = g.bfs(
    fromExpr = "name = 'Esther'",
    toExpr = "age < 32",
    edgeFilter = "relationship != 'friend'",
    maxPathLength = 3)
display(filteredPaths)

# COMMAND ----------

# MAGIC %md ### 接続されたコンポーネント(Connected components)
# MAGIC
# MAGIC それぞれの頂点で接続されたコンポーネントのメンバーシップを計算し、コンポーネントIDが割り当てられたそれぞれの頂点を持つデータフレームを返却します。GraphFramesのConnected componentsの実装では、パフォーマンスを改善するためにチェックポイントを活用することができます。

# COMMAND ----------

sc.setCheckpointDir("/tmp/graphframes-example-connected-components")
result = g.connectedComponents()
display(result)

# COMMAND ----------

# MAGIC %md ### 強く接続されたコンポーネント(Strongly connected component)
# MAGIC
# MAGIC それぞれの頂点のstrongly connected component (SCC)を計算し、当該の頂点を含むSCCに割り当てられたそれぞれの頂点を持つデータフレームを返却します。

# COMMAND ----------

result = g.stronglyConnectedComponents(maxIter=10)
display(result.select("id", "component"))

# COMMAND ----------

# MAGIC %md ### ラベルの伝播
# MAGIC
# MAGIC ネットワークにおけるコミュニティを検知するために、静的なラベル伝播アルゴリズムを実行します。
# MAGIC
# MAGIC ネットワークにおけるそれぞれのノードは、最初に自身のコミュニティに割り当てられます。すべてのスーパーステップにおいて、ノードは全ての隣人にコミュニティへの協力関係を依頼し、到着するメッセージから最も高頻度なコミュニティの協力依頼に自身の状態を更新します。
# MAGIC
# MAGIC LPAは、グラフにおける標準的なコミュニティ検知アルゴリズムです。これは、(1) 収束が保証されません (2) つまらない回答になる場合があります(全てのノードが単独なコミュニティとして識別される) が、計算量的には安価なものとなっています。

# COMMAND ----------

result = g.labelPropagation(maxIter=5)
display(result)

# COMMAND ----------

# MAGIC %md ### PageRank
# MAGIC
# MAGIC 接続性に基づいてグラフにおいて重要な頂点を特定します。

# COMMAND ----------

results = g.pageRank(resetProbability=0.15, tol=0.01)
display(results.vertices)

# COMMAND ----------

display(results.edges)

# COMMAND ----------

# 固定回数のイテレーションでPageRankを実行します。
results = g.pageRank(resetProbability=0.15, maxIter=10)
display(results.vertices)

# COMMAND ----------

# 頂点"a"にパーソナライズしたPageRankの実行
results = g.pageRank(resetProbability=0.15, maxIter=10, sourceId="a")
display(results.vertices)

# COMMAND ----------

# MAGIC %md ### 最短パス
# MAGIC
# MAGIC 頂点IDで指定されるランドマークの頂点のセットへの最短パスを計算します。

# COMMAND ----------

results = g.shortestPaths(landmarks=["a", "d"])
display(results)

# COMMAND ----------

# MAGIC %md ### 三角形のカウント
# MAGIC
# MAGIC それぞれの頂点を通過する三角形の数を計算します。

# COMMAND ----------

results = g.triangleCount()
display(results)
