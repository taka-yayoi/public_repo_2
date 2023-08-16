# Databricks notebook source
# MAGIC %md
# MAGIC # DatabricksとApache Sparkのご紹介
# MAGIC
# MAGIC 本ノートブックでは、DatabricksとApache Sparkの概要を説明するとともに、PySpark、SparkR、Koalas、Delta Lakeのハンズオンを行います。
# MAGIC
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2023/08/16</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.1</td></tr>
# MAGIC   <tr><td>クラスター</td><td>13.2</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricksにようこそ！
# MAGIC
# MAGIC このノートブックは、DatabricksでApache Sparkを使いこなすための方法を学ぶ最初の一歩として作成されました。ノートブックを通じて、コアのコンセプト、基本的な概要、必要となるツールをご紹介します。このノートブックは、Apache Sparkの開発者から提供されるコアコンセプトとベスプラクティスを提供するものです。
# MAGIC
# MAGIC まず初めに、Databricksを説明させてください。DatabricksはApache Sparkを実行するためのマネージドプラットフォームです。つまり、Sparkを利用するために、複雑なクラスター管理の考え方や、面倒なプラットフォームの管理タスクを学ぶ必要がないということです。また、Databricksは、Sparkを利用したワークロードを円滑にするための機能も提供しています。GUIでの操作をこのむデータサイエンティストやデータアナリストの方向けに、マウスのクリックで操作できるプラットフォームとなっています。しかし、UIに加えて、データ処理のワークロードをジョブで自動化したい方向けには、洗練されたAPIも提供しています。エンタープライズでの利用に耐えるために、Databricksにはロールベースのアクセス制御や、使いやすさを改善するためだけではなく、管理者向けにコストや負荷軽減のための最適化が図られています。
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210421-spark-introduction/lakehouse.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricksにおける用語
# MAGIC
# MAGIC Databricksには理解すべきキーコンセプトが存在します。これらの多くは画面左側のサイドバーにアイコンとして表示されています。これらは、エンドユーザーであるあなたのために用意された基本的なツールとなります。これらはウェブアプリケーションのUIからも利用できますし、REST APIからも利用できます。<br><br>
# MAGIC
# MAGIC - **Workspaces(ワークスペース)**
# MAGIC   - Databricksで行う作業はワークスペース上で実施することになります。あなたのコンピューターのフォルダー階層のように、**notebooks(ノートブック)** や **libraries(ライブラリ)** を保存でき、それらを他のユーザーと共有することができます。ワークスペースはデータとは別となっており、データを格納するために用いるものではありません。これらは単に **notebooks(ノートブック)** や **libraries(ライブラリ)** を保存するためのものであり、データを操作するためにこれらを使用することになります。
# MAGIC - **Notebooks(ノートブック)**
# MAGIC   - ノートブックはコマンドを実行するためのセルの集合です。セルには以下の言語を記述することができます： `Scala`、`Python`、`R`、`SQL`、`Markdown`。ノートブックにはデフォルトの言語が存在しますが、セルレベルで言語を指定することも可能です。これは、セルの先頭に`%[言語名]`十することで実現できます。例えば、`%python`です。すぐにこの機能を利用することになります。
# MAGIC   - コマンドを実行するためには、ノートブックは **cluster(クラスター)** に接続される必要がありますが、永久につながっている必要はありません。これによって、ウェブ経由で共有したり、ローカルマシンにダウンロードすることができます。
# MAGIC   - ノートブックのデモ動画を[こちら](http://www.youtube.com/embed/MXI0F8zfKGI)から参照できます。    
# MAGIC   - **Dashboards(ダッシュボード)**
# MAGIC     - **Dashboards(ダッシュボード)** は **notebooks(ノートブック)** から作成することができ、ダッシュボードを生成したコードを非表示にして結果のみを表示する手段として利用することができます。
# MAGIC   - **Notebooks(ノートブック)** は、ワンクリックで **jobs(ジョブ)** としてスケジュールすることができ、データパイプラインの実行、機械学習の更新、ダッシュボードの更新などを行うことができます。
# MAGIC - **Libraries(ライブラリ)**
# MAGIC   - ライブラリは、あなたが問題を解決するために必要となる追加機能を提供するパッケージあるいはモジュールです。これらは、ScalaやJava jarによるカスタムライブラリ、Pythonのeggsやカスタムのパッケージとなります。あなた自身の手でライブラリを記述し手動でアップロードできますし、pypiやmavenなどのパッケージ管理ユーティリティ経由で直接インストールすることもできます。
# MAGIC - **Tables(テーブル)**
# MAGIC   - テーブルはあなたとあなたのチームが分析に使うことになる構造化データです。テーブルはいくつかの場所に存在します。テーブルはAmazon S3やAzure Blob Storageに格納できますし、現在使用しているクラスターにも格納できます。あるいは、メモリーにキャッシュすることも可能です。詳細は[Databricksにおけるデータベースおよびテーブル \- Qiita](https://qiita.com/taka_yayoi/items/e7f6982dfbee7fc84894)を参照ください。
# MAGIC - **Clusters(クラスター)**
# MAGIC   - クラスターは、あなたが単一のコンピューターとして取り扱うことのできるコンピューター群です。Databricksにおいては、効果的に20台のコンピューターを1台としてと扱えることを意味します。クラスターを用いることで、あなたのデータに対して **notebooks(ノートブック)** や **libraries(ライブラリ)** のコードを実行することができます。これらのデータはS3に格納されている構造化データかもしれませんし、作業しているクラスターに対して **table(テーブル)** としてアップロードされた非構造化データかもしれません。
# MAGIC   - クラスターにはアクセス制御機能があることに注意してください。
# MAGIC   - クラスターのデモ動画は[こちら](http://www.youtube.com/embed/2-imke2vDs8)となります。
# MAGIC - **Jobs(ジョブ)**
# MAGIC   - ジョブによって、既存の **cluster(クラスター)** あるいはジョブ専用のクラスター上で実行をスケジュールすることができます。実行対象は **notebooks(ノートブック)** 、jars、pythonスクリプトとなります。ジョブは手動で作成できますが、REST API経由でも作成できます。
# MAGIC   - ジョブのデモ動画は[こちら](<http://www.youtube.com/embed/srI9yNOAbU0)となります。
# MAGIC - **Apps(アプリケーション)**
# MAGIC   - AppsはDatabricksプラットフォームと連携するサードパーティツールです。Tableauなどが含まれます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## DatabricksとApache Sparkのヘルプリソース
# MAGIC
# MAGIC Databricksには、Apache SparkとDatabriksを効果的に使うために学習する際に、助けとなる様々なツールが含まれています。Databricksには膨大なApache Sparkのドキュメントが含まれており。Webのどこからでも利用可能です。リソースには大きく二つの種類があります。Apace SparkとDatabricksの使い方を学ぶためのものと、基本を理解した方が参照するためのリソースです。
# MAGIC
# MAGIC これらのリソースにアクセスするには、画面右上にあるクエスチョンマークをクリックします。サーチメニューでは、以下のドキュメントを検索することができます。
# MAGIC
# MAGIC ![img](https://sajpstorage.blob.core.windows.net/demo20210421-spark-introduction/help_menu.png)
# MAGIC
# MAGIC - **Help Center(ヘルプセンター)**
# MAGIC   - [Help Center \- Databricks](https://help.databricks.com/s/)にアクセスして、ドキュメント、ナレッジベース、トレーニングなどのリソースにアクセスできます。
# MAGIC - **Release Notes(リリースノート)**
# MAGIC   - 定期的に実施される機能アップデートの内容を確認できます。
# MAGIC - **Documentation(ドキュメント)**
# MAGIC   - マニュアルにアクセスできます。
# MAGIC - **Knowledge Base(ナレッジベース)**
# MAGIC   - 様々なノウハウが蓄積されているナレッジベースにアクセスできます。
# MAGIC - **Feedback(フィードバック)**
# MAGIC   - 製品に対するフィードバックを投稿できます。
# MAGIC - **Shortcuts(ショートカット)**
# MAGIC   - キーボードショートカットを表示します。
# MAGIC     
# MAGIC また、Databricksを使い始める方向けに資料をまとめたDatabricksクイックスタートガイドもご活用ください。<br><br>
# MAGIC
# MAGIC - [Databricksクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/125231c126a602693610)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DatabricksとApache Sparkの概要
# MAGIC
# MAGIC ここまで、用語と学習のためのリソースを説明してきました。ここからは、Apache SparkとDatabricksの基本を説明していきます。もしかしたら、Sparkのコンセプトはご存知かもしれませんが、皆様が同じ理解をしているのかどうかを確認するための時間をいただければと思います。また、せっかくですので、ここでSparkの歴史も学びましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apache Sparkプロジェクトの歴史
# MAGIC
# MAGIC ![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/320px-Apache_Spark_logo.svg.png)
# MAGIC
# MAGIC SparkはDatabricksの創始者たちがUC Berkeleyにいるときに誕生しました。Sparkプロジェクトは2009年にスタートし、2010年にオープンソース化され、2013年にApacheにコードが寄贈されApache Sparkになりました。Apache Sparkのコードの75%以上がDatabricksの従業員の手によって書かれており、他の企業に比べて10倍以上の貢献をし続けています。Apache Sparkは、多数のマシンにまたがって並列でコードを実行するための、洗練された分散処理フレームワークです。概要とインタフェースはシンプルですが、クラスターの管理とプロダクションレベルの安定性を確実にすることはそれほどシンプルではありません。Databricksにおいては、Apache Sparkをホストするソリューションとして提供することで、ビッグデータをシンプルなものにします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apache Sparkとは？
# MAGIC
# MAGIC [Apache Spark](https://spark.apache.org/)は、オンプレミスのデータセンターやクラウドで利用できる大規模分散データ処理のために設計された統合エンジンです。
# MAGIC
# MAGIC Sparkは中間結果を保持するためのインメモリーを提供するのでHadoop MapReduceよりもはるかに高速になります。機械学習(MLlib)、インタラクティブなクエリーのためのSQL(Spark SQL)、リアルタイムデータを操作するためのストリーム処理(構造化ストリーミング)、グラフ処理(GraphX)で利用可能なAPIを持つライブラリと連携します。
# MAGIC
# MAGIC Sparkの設計哲学は4つのキーとなる特性を中心としています:
# MAGIC
# MAGIC - **スピード**: Sparkではさまざまな方法でスピードを追求しています。大量の安価なハードウェアを活用することで、効率的なマルチスレッド、並列処理を実現しています。すべての中間結果がメモリーに保持され、ディスクのI/Oを削減することで高いパフォーマンスを実現しています。
# MAGIC - **使いやすさ**: データフレームやデータセットのような高レベルのデータ構造がベースとしているRDD(Resilient Distributed Dataset)を用いて、シンプルな論理的データ構造を提供することでシンプルさを実現しています。一連の*トランスフォーメーション*と*アクション*を提供することで、慣れ親しんだ言語でビッグデータアプリケーションの構築で活用できるシンプルなプログラミングモデルの恩恵を受けることができます。
# MAGIC - **モジュール性**: Scala, Java, SQL, Rから利用でき、Spark SQL, 構造化ストリーミング, MLlib, GraphXなどと連携して活用することができます。
# MAGIC - **拡張可能性**: さまざまなエコシステムとの連携が可能です。

# COMMAND ----------

# MAGIC %md
# MAGIC ### コンテキスト、環境
# MAGIC
# MAGIC DatabricksとSparkの効果的な使い方を理解するために必要なピース全てに慣れ親しめるように、Apache Sparkのコアコンセプトを学んでいきましょう。
# MAGIC
# MAGIC 歴史的に、Apache Sparkにはユーザーが利用できる2つのコアコンテキストが存在していました。`sc`として利用できる`sparkContext`と`sqlContext`として利用できる`SQLContext`です。これらのコンテキストによって、ユーザーは様々な関数、情報を利用することができます。`sqlContext`では多くのデータフレームに関する機能を利用できる一方、`sparkContext`ではApache Sparkのエンジン自身に焦点を当てていました。
# MAGIC
# MAGIC しかし、Apache Spark 2.X以降は、一つのコンテキストになります。それが`SparkSession`です。ノートブック上では`spark`でアクセスすることができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データインタフェース
# MAGIC
# MAGIC Sparkを使う際に理解すべき、いくつかのキーインタフェースが存在します。
# MAGIC
# MAGIC - **Dataset(データセット)**
# MAGIC   - データセットはApache Sparkにおける最新の分散コレクションであり、データフレームとRDDを組み合わせたものと考えることができます。RDDで利用できる型インタフェースを提供しつつも、データフレームの利便性を兼ね備えています。これは、今後コアの概念になるでしょう。
# MAGIC - **Dataframe(データフレーム)**
# MAGIC   - データフレームは、分散された`Row`タイプのコレクションです。これにより、柔軟なインタフェースを保ちながらも、pythonのpandasやR言語で慣れ親しんだデータフレームのコンセプトをほぼそのまま活用することができます。
# MAGIC - **RDD(Resilient Distributed Dataset)**
# MAGIC   - Apache Sparkにおける最初の抽象化はRDDでした。基本的に、これはクラスターにある複数のマシンにまたがって配置される、1種類以上の型から構成されるデータオブジェクトのリストに対するインタフェースとなります。RDDは様々な方法で作成することができ、ユーザーに対して低レベルのAPIを提供します。これは利用可能なオリジナルのデータ構造ではありますが、新たなユーザーは、RDDの機能のスーパーセットとなるデータセットにフォーカスすべきです。

# COMMAND ----------

# MAGIC %md
# MAGIC # コーディングを始めましょう!
# MAGIC
# MAGIC やれやれ、これまでに多くのことをカバーしてきました！でも、これでようやくApache SparkとDatabricksのパワーを体感できるデモに進むことができます。しかし、その前にいくつかの作業が必要となります。最初にすべきシンプルなことは、このノートブックをあなたの環境に取り込むことです。
# MAGIC
# MAGIC ノートブックをダウンロードした後の手順は[ノートブックのインポート](https://qiita.com/taka_yayoi/items/c306161906d6d34e8bd5#%E3%83%8E%E3%83%BC%E3%83%88%E3%83%96%E3%83%83%E3%82%AF%E3%81%AE%E3%82%A4%E3%83%B3%E3%83%9D%E3%83%BC%E3%83%88)を参照ください。

# COMMAND ----------

# MAGIC %md ## クラスターのアタッチ
# MAGIC
# MAGIC ノートブック上のコマンドを実行するには、ノートブックをSparkクラスターにアタッチする必要があります。画面左上にある「Detached」をクリックしてリストを展開し、クラスターを選択します。

# COMMAND ----------

# MAGIC %md ## PySpark
# MAGIC
# MAGIC PySparkはPythonからSparkを利用するためのAPIを提供します。Sparkの分散処理機能を活用することで、ローカルマシンのメモリーに乗り切らないような大量データであっても、容易に集計、分析、さらには機械学習が可能となります。
# MAGIC
# MAGIC <table>
# MAGIC   <tr><th>pandas</th><th>Apache Spark(PySpark)</th></tr>
# MAGIC <tr>
# MAGIC     
# MAGIC <td>
# MAGIC   データセットが小さい場合はpandasが正しい選択となります。
# MAGIC </td>
# MAGIC
# MAGIC <td>
# MAGIC   大きなデータに対する「フィルタリング」「クリーニング」「集計」などの処理が必要な場合は、Apache Sparkのような並列データフレームを使用することで線形の高速化が期待できます。
# MAGIC </td>
# MAGIC </tr>
# MAGIC   
# MAGIC </table>  
# MAGIC
# MAGIC 参考資料
# MAGIC - [PySparkとは \- Databricks](https://databricks.com/jp/glossary/pyspark)
# MAGIC - [Databricks Apache Sparkクイックスタート \- Qiita](https://qiita.com/taka_yayoi/items/bf5fb09a0108aa14770b)
# MAGIC - [Databricks Apache Sparkデータフレームチュートリアル \- Qiita](https://qiita.com/taka_yayoi/items/2a7e9bb792eba316de4b)
# MAGIC - [オープンソースのPandasとApache Sparkを比較](https://www.ossnews.jp/compare/Pandas/Apache_Spark)
# MAGIC - [最新のApache Spark v2\.4にふれてみよう: 概要と新機能の紹介 \| by Takeshi Yamamuro \| nttlabs \| Medium](https://medium.com/nttlabs/apache-spark-v24-159ab8983ead)
# MAGIC
# MAGIC 最初に上で説明した`SparkSession`に触れてみましょう。`spark`変数を介してアクセスすることができます。説明したように、SparkセッションはApache Sparkに関する情報が格納される重要な場所となります。
# MAGIC
# MAGIC セルのコマンドは、セルが選択されている状態で`Shift+Enter`を押すことで実行できます。

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md 
# MAGIC 情報にアクセスするためにSparkコンテキストを利用できますが、コレクションを並列化するためにも利用できます。こちらでは、`DataFrame`を返却するpythonの小規模のrangeを並列化します。

# COMMAND ----------

firstDataFrame = spark.range(1000000)
print(firstDataFrame)

# COMMAND ----------

# MAGIC %md ## トランスフォーメーションとアクション
# MAGIC
# MAGIC 上で`print`を実行した際に、並列化した`DataFrame`の値が表示されるはずだと思ったかもしれません。しかし、Apache Sparkはそのように動作しません。Sparkには、2種類の明確に異なるユーザーのオペレーションが存在します。それが**transformations(トランスフォーメーション)** と **actions(アクション)** です。
# MAGIC
# MAGIC ### Transformations(トランスフォーメーション)
# MAGIC
# MAGIC トランスフォーメーションを記述したセルを実行した時点では処理が完了しないオペレーションです。これらは **action(アクション)** が呼ばれたときにのみ実行されます。トランスフォーメーションの例は、integerをfloatへ変換、値のフィルタリングなどです。
# MAGIC
# MAGIC ### Actions(アクション)
# MAGIC
# MAGIC アクションは実行された瞬間にSparkによって処理が行われます。アクションは、実際の結果を取得するために、前にあるトランスフォーメーション全てを実行することから構成されます。アクションは一つ以上のジョブから構成され、ジョブはワーカーノードにおいて可能であれば並列で実行される複数のタスクから構成されます。
# MAGIC
# MAGIC こちらがシンプルなトランスフォーメーションとアクションの例です。これらが**全てのトランスフォーメーション、アクションではない**ことに注意してください。これはほんのサンプルです。なぜ、Apache Sparkがこのように設計されているのかについてはすぐにご説明します！
# MAGIC
# MAGIC **Transformations**
# MAGIC
# MAGIC - `orderby()`
# MAGIC - `groupby()`
# MAGIC - `filter()`
# MAGIC - `select()`
# MAGIC - `join()`
# MAGIC
# MAGIC **Actions**
# MAGIC
# MAGIC - `show()`
# MAGIC - `take()`
# MAGIC - `count()`
# MAGIC - `collect()`
# MAGIC - `save()`

# COMMAND ----------

# トランスフォーメーションの例
# IDカラムを選択し、2倍します
secondDataFrame = firstDataFrame.selectExpr("(id * 2) as value")

# COMMAND ----------

# アクションの例
# firstDataFrameの最初の５行を取得します
print(firstDataFrame.take(5))
# secondDataFrameの最初の５行を取得します
print(secondDataFrame.take(5))

# COMMAND ----------

# display()コマンドでsecondDataFrameを表示します
display(secondDataFrame)

# COMMAND ----------

# MAGIC %md ## Apache Sparkのアーキテクチャ
# MAGIC
# MAGIC ここまでで、Sparkにはアクションとトランスフォーメーションがあることがわかりました。なぜこれが必要なのかを説明します。これは、個々の処理のピースを最適化するのではなく、処理全体のパイプラインを最適化するためのシンプルな方法だからです。適切な処理を一度に実行できるため、特定のタイプの処理において、処理が劇的に高速になります。技術的に言えば、Sparkは、以下の図に示すように処理を`pipelines(パイプライン化)`します。すなわち、逐次的に処理を実行するのではなく、(フィルタリングやマッピングなどを)一括で処理するということです。
# MAGIC
# MAGIC ![transformations and actions](https://sajpstorage.blob.core.windows.net/yayoi/spark_dag.png)
# MAGIC
# MAGIC Apache Sparkは、他のフレームワークのようにそれぞれのタスクの都度ディスクに結果を書き込むのではなく、メモリー上に結果を保持します。
# MAGIC
# MAGIC サンプルを進める前に、Apache Sparkのアーキテクチャを見てみましょう。上で述べたように、Apache Sparkは大量のマシンを一つのマシンとして取り扱えるようにしてくれます。これは、クラスターに`driver(ドライバー)`ノードと、付随する`worker(ワーカー)`ノードから構成されるドライバー・ワーカータイプのアーキテクチャによって実現されています。ドライバーノードがワーカーノードに作業を割り振り、ワーカーノードに対して、メモリーあるいはディスク（あるいはS3やRedshift）からデータを取得するように指示します。
# MAGIC
# MAGIC 以下の図では、ワーカー(executor)ノードとやりとりをするドライバーノードから構成されるApache Sparkクラスターの例を示しています。それぞれのexecutorノードには、処理を実行するコアに該当するスロットが存在します。
# MAGIC
# MAGIC ![spark-architecture](https://sajpstorage.blob.core.windows.net/yayoi/spark_cluster.png)
# MAGIC
# MAGIC 処理を実行する際、ドライバーノードは空いているワーカーノードのスロットにタスクを割り当てます。
# MAGIC
# MAGIC Apache SparkのWeb UIで、あなたのApache Sparkアプリケーションの詳細を参照することができます。Web UIに移動するには、"Clusters"をクリックし、参照したいクラスターの"Spark UI"リンクをクリックします。あるいは、このノートブック画面の左上に表示されているクラスターをクリックして"Spark UI"リンクをクリックします。
# MAGIC
# MAGIC ハイレベルにおいては、全てのApache Sparkアプリケーションは、クラスターあるいはローカルマシンの同一マシンで動作するワーカーのJava Virtual Macines(JVMs)における並列処理を起動するドライバープログラムから構成されています。Databricksでは、ノートブックのインタフェースがドライバープログラムとなっています。このドライバープログラムはプログラムのmainループから構成されており、クラスター上に分散データセットを作成し、それらのデータセットにオペレーション(トランスフォーメーションとアクション)を適用します。
# MAGIC
# MAGIC ドライバープログラムは、デプロイされている場所に関係なく`SparkSession`オブジェクトを介してApache Sparkにアクセスします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## トランスフォーメーションとアクションの実例
# MAGIC
# MAGIC これらのアーキテクチャ、適切な **transformations(トランスフォーメーション)** と **actions(アクション)** を説明するために、より詳細な例を見ていきましょう。今回は、`DataFrames(データフレーム)`とcsvファイルを使用します。
# MAGIC
# MAGIC データフレームとSparkSQLはここまで説明した通りに動作します。どのようにデータにアクセスするのかの計画を組み立て、最終的にはそのプランをアクションによって実行します。これらのプロセスを以下の図に示します。クエリを分析し、プランを立て、比較を行い、最終的に実行することで、全体のプロセスをなぞります。
# MAGIC
# MAGIC このプロセスがどのように動作するのかについて、あまり詳細には踏み込みませんが、詳細な内容は[Databricks blog](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html)で読むことができます。このプロセスを通じてどのようにApache Sparkが動作しているのかを知りたい方には、こちらの記事を読むことをお勧めします。
# MAGIC
# MAGIC 以降では、Databricksで利用可能なパブリックなデータセットにアクセスすることになります。Databricksデータセットは、Webから集められた、小規模かつ整理されたデータセットとなっています。これらのデータは、[Databricks File System](https://docs.databricks.com/data/databricks-file-system.html)を通じて利用できます。よく使われるダイアモンドのデータセットを、Sparkの`DataFrame`としてロードしてみましょう。まずは、作業することになるデータセットを見てみましょう。

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/datasets.csv

# COMMAND ----------

# MAGIC %md 以下のコードでは、前の行から継続していることを示すためにバックスラッシュ `\` を使用していることに注意してください。Sparkのコマンドは、多くのケースで複数のオペレーションのチェーンを構築することになります(例：`.option(...)`)。コマンドが一行に収まらない場合、バックスラッシュを使うことでコードをきれいに保つことができます。

# COMMAND ----------

dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = spark.read.format("csv")\
  .option("header","true")\
  .option("inferSchema", "true")\
  .load(dataPath)
  
# inferSchemaはデータを読み込んで自動的にカラムの型を識別するオプションです。データを読み込む分のコストがかかります。

# COMMAND ----------

# MAGIC %md 
# MAGIC データをロードしたので、計算に取りかかります。この作業を通じて、基礎的な機能とDatabricks上で動作するSparkの実行をシンプルにする素敵な機能のいくつかを体験することができます。計算を行うためには、データを理解する必要があります。これは`display`関数で可能となります。

# COMMAND ----------

display(diamonds)

# COMMAND ----------

# MAGIC %md 
# MAGIC `display`の素晴らしいところは、グラフアイコンをクリックするだけで以下のような素晴らしいグラフを容易に作成できるところです。以下のプロットによって、価格、色、カットを比較することができます。

# COMMAND ----------

display(diamonds)

# COMMAND ----------

# MAGIC %md 
# MAGIC データを探索しましたので、**transformations**と**actions**の理解に戻りましょう。ここではいくつかのトランスフォーメーションを作成し、アクションを呼び出します。その後で、内部で何が行われたのかを見ていきます。
# MAGIC
# MAGIC これらのトランスフォーメーションはシンプルなものです。最初に二つの変数、カットとカラーでグルーピングします。そして、平均価格を計算します。そして、`color`カラムでオリジナルのデータセットと`inner join`を行います。そして、新たなデータセットからカラットと平均価格を選択します。

# COMMAND ----------

df1 = diamonds.groupBy("cut", "color").avg("price") # シンプルなグルーピング

df2 = df1\
  .join(diamonds, on='color', how='inner')\
  .select("`avg(price)`", "carat")
# シンプルなjoin及び列の選択

# COMMAND ----------

# MAGIC %md 
# MAGIC これらのトランスフォーメーションはある意味完成していますが何も起きません。上に表示されているように何の結果も表示されていません！
# MAGIC
# MAGIC これは、ユーザーによって要求される処理の初めから終わりまでの全体のデータフローを構築するために、これらの処理が*lazy(怠惰)*となっているためです。
# MAGIC
# MAGIC これは、二つの理由からスマートな最適化と言えます。第一に、途中でエラーが発生した場合や、ワーカーノードが処理に手間取った場合、どこから再計算すればいいのか追跡できます。第二に、上で述べたようにデータと処理がパイプライン化されるように、処理を最適化することができます。このため、それぞれのトランスフォーメーションに対して、Apache Sparkはどのように処理を行うのか計画を立てます。
# MAGIC
# MAGIC [Sparkの内部処理を理解する \- Qiita](https://qiita.com/uryyyyyyy/items/ba2dceb709f8701715f7)
# MAGIC
# MAGIC 計画がどのようなものなのかを理解するために、`explain`メソッドを使用します。この時点ではまだ何の処理も実行されていないことに注意してください。このため、explainメソッドが教えてくれることは、このデータセットに対してどのように処理を行うのかの証跡(leneage)となります。

# COMMAND ----------

df2.explain()

# COMMAND ----------

# MAGIC %md 
# MAGIC 上の結果が何を意味しているのかは、この導入のチュートリアルの範疇外となりますが、中身を読むのは自由です。ここで導き出されることは、Sparkは与えられたクエリーを実行する際には実行計画を生成するということです。上のプランを実行に移すためにアクションを実行しましょう。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC これにより、Apache Sparkが上で構築したプランが実行されます。実行後に、`(1) Spark Jobs`と言った表示の隣にある小さい矢印をクリックし、`View`リンクをクリックし、さらに`DAG Visualization`をクリックします。これにより、ノートブックの右側にApache Spark Web UIが表示されます。ノートブックの上部にあるクラスターをクリックしてもこの画面にアクセスすることができます。Spark UIでは、以下のような図を見ることができます。
# MAGIC
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/yayoi/spark_dag_full.png" width=600>
# MAGIC
# MAGIC これが、結果を得るために実行された処理全ての有向非巡回グラフ(DAG)となります。この可視化によって、Sparkがデータを最終的な形に持っていくまでのステップ全てを参照することができます。
# MAGIC
# MAGIC 繰り返しになりますが、このDAGはトランスフォーメーションが*lazy*であるため生成されます。これら一連のステップを生成する過程で、Sparkは多くの最適化を行い、そのためのコードをも生成します。このことによって、ユーザはレガシーなRDD APIではなくデータフレームやデータセットにフォーカスすることができます。データフレームやデータセットを用いることで、Apache Sparkは内部で全てのステップ、全てのクエリープラン、パイプラインを最適化することができます。プランの中に`WholeStageCodeGen`や`tungsten`を見ることになるかと思います。これらは[improvements in Spark SQL, which you can read more about on the Databricks blog.](https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html)の一部です。
# MAGIC
# MAGIC 上の図では、左側のCSVから始まり、いくつかの処理を経て、別のCSVファイル(これはオリジナルのデータフレームから作成したものです)にマージし、これらをジョインした上で最終的な結果を得るためにいくつかの集計処理を行っています。

# COMMAND ----------

# MAGIC %md ## キャッシュ
# MAGIC
# MAGIC Apache Sparkの重要な機能の一つとして、計算の過程でメモリーにデータを格納できるということが挙げられます。これは、よく検索されるテーブルやデータに対するアクセスを高速にする際に活用できるテクニックです。また、同じデータに対して繰り返しアクセスするような繰り返しの処理を行うアルゴリズムにも有効です。全ての性能問題に対する万能薬かと思うかもしれませんが、利用できるツールの一つとして捉えるべきです。データのパーティショニングやクラスタリング、バケッティングなどの他の重要なコンセプトの方が、キャッシングよりも高い性能を実現する場合があります。とは言え、これら全てのツールが利用可能であることを覚えておいて下さい。
# MAGIC
# MAGIC データフレームやRDDをキャッシュするには、単に`cache`メソッドを呼ぶだけです。

# COMMAND ----------

df2.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC キャッシュはトランスフォーメーションのようにlazyに評価されます。データセットに対するアクションが呼び出されるまでメモリーにデータはキャッシュされません。
# MAGIC
# MAGIC 簡単な例で説明します。これまでに我々はデータフレーム`df2`を作成しました。これは本質的には、データフレームをどのように計算するのかを示すロジカルプランです。Apahce Sparkに対してこのデータをキャッシュするように初めて伝えました。countによるデータに対するフルスキャンを2回実行しましょう。最初の時には、データフレームを作成しメモリーにキャッシュし、結果を返却します。2回目は、全てのデータフレームを計算するのではなく、メモリーに存在するバージョンを取得します。
# MAGIC
# MAGIC どのように動作しているのかを見てみましょう。一回目のアクション`count`が呼ばれた時点で`df2`はキャッシュされます。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC キャッシュした後では、同じクエリーに要する処理時間が大きく減少していることがわかります。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 上の例では、データを生成するのに必要な時間を劇的に短縮できた様子が見て取れます。少なくとも一桁レベルで削減できています。より大規模かつ複雑なデータ分析においては、キャッシングによる恩恵はより大きなものになります。
# MAGIC
# MAGIC キャッシュを削除するには`unpersist`メソッドを呼び出します。

# COMMAND ----------

df2.unpersist()

# COMMAND ----------

# MAGIC %md ## SparkR
# MAGIC
# MAGIC SparkRは、RからApache Sparkを利用することができる軽量フロントエンドを提供するRパッケージです。MLlibによる分散機械学習もサポートしています。
# MAGIC
# MAGIC Spark 2.2以降では、デフォルトではSparkRはノートブックにインポートされません。これは、他のパッケージとの名前の競合が起きるためです。SparkRを使うにはノートブック上で明示的に`library(SparkR)`を実行する必要があります。
# MAGIC
# MAGIC Databricksのノートブックではセルレベルで言語を切り替えることができます。Rを使うにはセルの先頭に`%r`と記述します。
# MAGIC
# MAGIC 参考資料
# MAGIC - [SparkR overview \| Databricks on AWS](https://docs.databricks.com/spark/latest/sparkr/overview.html)
# MAGIC - [混成言語（言語マジックコマンド）](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc#%E6%B7%B7%E6%88%90%E8%A8%80%E8%AA%9E)

# COMMAND ----------

# MAGIC %md ### Rデータフレームの作成
# MAGIC
# MAGIC ローカルのRデータフレーム、データソース、あるいはSpark SQLクエリーからSparkのデータフレームを作成することができます。

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC # ローカルのRデータフレームからSparkデータフレームを作成
# MAGIC # faithfulはイエローストーン国立公園の間欠泉の噴出期間を記録したデータです
# MAGIC df <- createDataFrame(faithful)
# MAGIC
# MAGIC # 標準出力にデータフレームを表示
# MAGIC head(df)

# COMMAND ----------

# MAGIC %r
# MAGIC # データソースからSparkデータフレームを作成
# MAGIC diamondsDF <- read.df("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", source = "csv", header="true", inferSchema = "true")
# MAGIC head(diamondsDF)

# COMMAND ----------

# MAGIC %md ### データフレームの操作

# COMMAND ----------

# MAGIC %r
# MAGIC # Sparkデータフレームの作成
# MAGIC df <- createDataFrame(faithful)
# MAGIC
# MAGIC # "eruptions"列のみをSELECT
# MAGIC head(select(df, df$eruptions))

# COMMAND ----------

# MAGIC %r
# MAGIC # 噴出間隔が50分未満の行のみをフィルタリングします
# MAGIC head(filter(df, df$waiting < 50))

# COMMAND ----------

# MAGIC %md ## Pandas API on Spark(旧Koalas)
# MAGIC
# MAGIC ![](https://koalas.readthedocs.io/en/v1.6.0/_static/koalas-logo-docs.png)
# MAGIC
# MAGIC [Koalas](https://github.com/databricks/koalas)は、[pandas](https://pandas.pydata.org/)の補完材を提供するオープンソースプロジェクトです。主にデータサイエンティストによって用いられるpandasは、簡単に使えるデータ構造とPython言語向けのデータ分析ツールを提供するPythonのパッケージです。しかし、pandasは大量データに対してスケールしません。KoalasはApache Sparkで動作するpandasと同等のAPIを提供することでこのギャップを埋めます。Koalasはpandasユーザーにとって有益であるだけではなく、Koalasは例えばPySparkデータフレームから直接データをプロットするなど、PySparkで実行するのが困難なタスクをサポートするので、PySparkユーザーにも役立ちます。
# MAGIC
# MAGIC [Koalasのご紹介 \- Qiita](https://qiita.com/taka_yayoi/items/5bbb3280940e73395bf5)

# COMMAND ----------

# MAGIC %md ### オブジェクトの作成

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

# pandasのシリーズの作成
pser = pd.Series([1, 3, 5, np.nan, 6, 8]) 
# Koalasのシリーズの作成
kser = ps.Series([1, 3, 5, np.nan, 6, 8])
# pansasシリーズを渡してKoalasシリーズを作成
kser = ps.Series(pser)
kser = ps.from_pandas(pser)

# COMMAND ----------

pser # pandasオブジェクト

# COMMAND ----------

kser # Koalasオブジェクト

# COMMAND ----------

kser.sort_index() # pandasで提供されているsort_indexを利用できます

# COMMAND ----------

# pandasデータフレームの作成
pdf = pd.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})
# Koalasデータフレームの作成
kdf = ps.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})
# pandasデータフレームを渡してKoalasデータフレームを作成
kdf = ps.DataFrame(pdf)
kdf = ps.from_pandas(pdf)

# COMMAND ----------

pdf # pandasデータフレーム

# COMMAND ----------

kdf.sort_index() # Koalasデータフレーム

# COMMAND ----------

# MAGIC %md ### データの参照

# COMMAND ----------

kdf.head(2)

# COMMAND ----------

kdf.describe()

# COMMAND ----------

kdf.sort_values(by='B')

# COMMAND ----------

kdf.transpose()

# COMMAND ----------

# MAGIC %md ## まとめ
# MAGIC
# MAGIC このノートブックでは多くの話題をカバーしました！ しかし、まだあなたはSparkとDatabricksを学ぶ道の入り口に立ったところです！ このノートブックを修了することで、SparkとDatabricksのコアの概念に慣れ親しんだに違いありません。

# COMMAND ----------

# MAGIC %md # END
