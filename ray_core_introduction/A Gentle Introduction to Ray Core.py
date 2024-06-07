# Databricks notebook source
# MAGIC %md
# MAGIC [A Gentle Introduction to Ray Core by Example — Ray 2\.23\.0](https://docs.ray.io/en/latest/ray-core/examples/gentle_walkthrough.html)
# MAGIC
# MAGIC Rayがどのように動作するのか、そして基本コンセプトを理解するために、Ray Coreで関数を実装します。経験は少ないけど高度なタスクに興味があるPythonプログラマーは、Ray Core APIを学ぶことで、Pythonを用いた分散コンピューティングを始めることができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rayのインストール
# MAGIC
# MAGIC Databricksランタイム15.0にはすでにRayがインストールされています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray Core
# MAGIC
# MAGIC 以下のコマンドを実行することでローカルクラスターを起動します。

# COMMAND ----------

# MAGIC %md
# MAGIC 2台のRayワーカーノードを持つRayクラスターをセットアップします。それぞれのワーカーノードには4個のGPUコアが割り当てられています。クラスターが起動すると、Rayクラスターダッシュボードを参照するために、リンク"Open Ray Cluster Dashboard in a new tab"をクリックすることができます。

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=4,
  collect_log_to_path="/dbfs/tmp/raylogs",
)

# COMMAND ----------

import ray
ray.init()
ray.cluster_resources()

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、Ray Core APIの簡単なご紹介をしていきますが、これをRay APIとして参照します。Ray APIはPythonプログラマーに馴染みのあるデコレーター、関数、関数のようなコンセプトをベースとして構築されています。これは分散コンピューティングにおける一般的なプログラミングインタフェースとなっています。このエンジンが複雑な作業に対応するので、開発者は既存のPythonライブラリやシステムと共にRayを活用することができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 初めてのRay APIサンプル
# MAGIC
# MAGIC 以下の関数では、データベースからデータを取得し処理を行います。ダミーの`database`は[“Learning Ray” book](https://www.amazon.com/Learning-Ray-Flexible-Distributed-Machine/dp/1098117220/)のタイトルに含まれる単語を含むプレーンなPythonリストとなっています。データベースのデータへのアクセス、処理のコストをシミュレートするために、`sleep`関数は一定の期間ポーズしています。

# COMMAND ----------

import time

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]


def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]

# COMMAND ----------

# MAGIC %md
# MAGIC インデックス5のタスクが0.5秒`(5 / 10.)`要している場合、逐次的にすべての8つのアイテムを取得するための合計処理時間の見積もり値は`(0+1+2+3+4+5+6+7)/10. = 2.8`秒となります。実際の時間を計測するために以下のコードを実行します。

# COMMAND ----------

def print_runtime(input_data, start_time):
    print(f'実行時間: {time.time() - start_time:.2f} 秒、データ:')
    print(*input_data, sep="\n")


start = time.time()
data = [retrieve(item) for item in range(8)]
print_runtime(data, start)

# COMMAND ----------

# MAGIC %md
# MAGIC この例では、関数を実行する際の処理時間合計は2.88秒でしたが、お使いのコンピューターによっては時間が変動するかもしれません。基本的なPythonのバージョンでは関数を同時に実行できないことに注意してください。
# MAGIC
# MAGIC Pythonリストの解釈はもっと効率的であることを期待するかもしれません。計測した実行時間2.8秒は実際には最悪ケースのシナリオです。このプログラムは実行時間のほとんどを"sleep"していますが、これは、Global Interpreter Lock (GIL)のために遅くなっています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rayタスク
# MAGIC
# MAGIC このタスクは、並列化によってメリットを享受することができます。完璧に分散することができれば、実行時間は最も遅いサブタスクよりも長くなることはないはずです。すなわち、`7/10. = 0.7`秒となります。このサンプルをRayで並列実行するように拡張するには、`@ray.remote`デコレーターで始まる関数を記述します。

# COMMAND ----------

import ray 


@ray.remote
def retrieve_task(item):
    return retrieve(item)

# COMMAND ----------

# MAGIC %md
# MAGIC デコレーターを用いることで、関数`retrieve_task`は`ray-remote-functions<Ray task>`への参照となります。Rayタスクは、呼び出された場所から、場合によっては異なるマシンから様々なプロセスを実行する関数となります。
# MAGIC
# MAGIC Rayでは、アプローチやプログラミングスタイルを大きく変更することなしに、Pythonコードを書き続けることができるので便利です。この例では、retrieve関数に`ray.remote()<@ray.remote>`関数デコレーターを用いることは、デコレーターの意図した使い方であり、オリジナルのコードを変更していません。
# MAGIC
# MAGIC データベースのエントリーを収集し、パフォーマンスを計測するために、コードに対して多くの変更を行う必要はありません。こちらがプロセスのオーバービューとなります。

# COMMAND ----------

start = time.time()
object_references = [
    retrieve_task.remote(item) for item in range(8)
]
data = ray.get(object_references)
print_runtime(data, start)

# COMMAND ----------

# MAGIC %md
# MAGIC タスクを並列で実行するには、主要な2つのコード修正を必要としています。リモートでRayタスクを実行するには、`.remote()`コールを用います。Rayはローカルクラスター上であっても、リモートタスクを非同期的に実行します。コードスニペットにある`object_references`リストのアイテムは、結果を直接格納するものではありません。`type(object_references[0])`を用いて最初のアイテムのPythonタイプをチェックをすると、これが実際には`ObjectRef`であることがわかります。これらのオブジェクト参照は、結果をリクエストする*将来的な*値に対応します。関数`ray.get()<ray.get(...)>`の呼び出しは、結果をリクエストするためのものです。Rayタスクに対してリモートコールを行うと常に、1つ以上のオブジェクト参照を即座に返却します。Rayタスクはオブジェクト生成の主要な手段であると考えてください。以下のセクションでは、複数のタスクをまとめてリンクさせ、それらの間でRayにオブジェクトを引き渡し、解決させています。
# MAGIC
# MAGIC 前のステップをレビューしましょう。Python関数からスタートし、`@ray.remote`でデコレートし、関数をRayタスクにしています。コードでオリジナルの関数を直接呼び出すのではなく、Rayタスクに対して`.remote(...)`を呼び出しました。最後に、`.get(...)`を用いてRayクラスターから結果を取得しました。追加のエクササイズとして、ご自身の関数の一つからRayタスクを作ってみてください。
# MAGIC
# MAGIC Rayタスクを用いることによるパフォーマンスのゲインをレビューしましょう。ほとんどのラップトップでは、実行時間は約0.71秒となり、最も遅いサブタスクである0.7秒よりも若干時間を要しているものとなります。RayのAPIをさらに活用することでプログラムをさらに改善することができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## オブジェクトストア
# MAGIC
# MAGIC 取得処理の定義では、`database`からアイテムに直接アクセスしています。ローカルのRayクラスターであればこれは問題ありませんが、複数のコンピュータから構成される実際のクラスターでどのように動作するのかを考えてみましょう。Rayクラスターは、ドライバープロセスを持つヘッドノードと、タスクを実行するワーカープロセスを持つ複数のワーカーノードから構成されます。このシナリオでは、データベースはドライバーでのみ定義されますが、ワーカープロセスは取得タスクを実行するためにそれにアクセスする必要があります。ドライバーとワーカー間、ワーカー間でオブジェクトを共有するRayのソリューションは、データをRayの分散オブジェクトストアに配置するための`ray.put`関数を用いるというものです。`retrieve_task`の定義において、`db_object_ref`オブジェクトとしてあとで引き渡せるように、引数`db`を追加することができます。

# COMMAND ----------

db_object_ref = ray.put(database)


@ray.remote
def retrieve_task(item, db):
    time.sleep(item / 10.)
    return item, db[item]

# COMMAND ----------

# MAGIC %md
# MAGIC オブジェクトストアを用いることで、Rayはクラスター全体を通じたデータのアクセスの管理を行うことができます。オブジェクトストアにはある程度のオーバーヘッドが含まれますが、大規模なデータセットでのパフォーマンスが改善します。このステップは、真に分散された環境においては重要なものとなります。期待した通りに実行されることを確認するために、サンプルの`retrieve_task`関数を再実行しましょう。

# COMMAND ----------

start = time.time()
object_references = [
    retrieve_task.remote(item, db_object_ref) for item in range(8)
]
data = ray.get(object_references)
print_runtime(data, start)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ブロッキングなしのコール
# MAGIC
# MAGIC 前のセクションでは、結果を取得するために`ray.get(object_references)`を使用しました。このコールは、すべての結果が利用できるようになるまでドライバープロセスをブロックします。データベースのそれぞれのアイテムが処理に数分を要する場合には、この依存性が問題となる場合があります。結果を待っている際にドライバープロセスが他のタスクを実行できるようにし、すべてのアイテムの処理が完了するのを待つのではなしに結果を処理できるようにすることで、更なる効率性を得ることができます。さらに、データベースアイテムの一つがデータベース接続におけるデッドロックのような問題によって取得できなかった場合、ドライバーは永遠にハングすることになります。無限のハングを避けるために、`wait`関数を用いる際に適切な`timeout`の値を設定します。例えば、最も遅いデータ取得タスクの時間の十倍までは待ちたい場合には、その時間を経過した際にタスクをストップするように`wait`関数を活用します。

# COMMAND ----------

start = time.time()
object_references = [
    retrieve_task.remote(item, db_object_ref) for item in range(8)
]
all_data = []

while len(object_references) > 0:
    finished, object_references = ray.wait(
        object_references, timeout=7.0
    )
    data = ray.get(finished)
    print_runtime(data, start)
    all_data.extend(data)

print_runtime(all_data, start)

# COMMAND ----------

# MAGIC %md
# MAGIC 結果を出力する代わりに、他のワーカーで新たなタスクを起動するために、`while`ループの中で取得した値を使用することもできます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## タスクの依存関係
# MAGIC
# MAGIC 取得したデータに対して追加の処理タスクを実行したいと思うかもしれません。例えば、同じデータベースから取得した(おそらく他のテーブルの)他のデータをクエリーするために最初の取得タスクの結果を使うなどです。以下のコードでは、このフォローアップタスクをセットアップし、`retrieve_task`と`follow_up_task`の両方を順に実行しています。

# COMMAND ----------

@ray.remote
def follow_up_task(retrieve_result):
    original_item, _ = retrieve_result
    follow_up_result = retrieve(original_item + 1)
    return retrieve_result, follow_up_result


retrieve_refs = [retrieve_task.remote(item, db_object_ref) for item in [0, 2, 4, 6]]
follow_up_refs = [follow_up_task.remote(ref) for ref in retrieve_refs]

result = [print(data) for data in ray.get(follow_up_refs)]

# COMMAND ----------

# MAGIC %md
# MAGIC 非同期プログラミングに慣れ親しんでいない場合、このサンプルはあまり印象的なものではないかもしれません。しかし、よく見てみるとこのコードが実行されることは驚くべきことなのです。このコードはいくつかのリスト解釈を伴う通常のPythonプログラムのように見えます。
# MAGIC
# MAGIC `follow_up_task`の関数の本体は、入力引数の`retrieve_result`としてPythonのタプルを期待します。しかし、`[follow_up_task.remote(ref) for ref in retrieve_refs]`コマンドを使う際は、フォローアップ多数にはタプルを引き渡しません。代わりに、Rayオブジェクトの参照を渡すために`retrieve_refs`を使うことになります。
# MAGIC
# MAGIC 背後では、Rayは`follow_up_task`が実際の値を必要とすることを認識しているので、これらの将来的な値を解決するために*自動で*`ray.get`関数を使用します。さらに、Rayはすべてのタスクの依存関係グラフを生成し、これらの依存関係を考慮した方法で実行します。実行順序を推定するので、明示的にRayに前のタスクの完了を待つように指示する必要はありません。次のタスクにオブジェクト参照を渡し、Rayが残りを面倒見てくれるので、大規模な中間的な値をドライバーにコピーすることを回避できるので、Rayオブジェクトストアのこの機能は有用なものとなります。
# MAGIC
# MAGIC 情報を取得することに特化して設計されたタスクが完了すると、このプロセスの次のステップがスケジュールされます。実際、`retrieve_refs`を`retrieve_result`と呼んでいたら、この重要で意図的なネーミングのニュアンスに気づかなかったことでしょう。Rayによって、あたなたはクラスターコンピューティングの技術的なことではなく、自分の作業に集中できるようになります。2つのタスクの依存関係グラフは以下のようになります:
# MAGIC ![](https://raw.githubusercontent.com/maxpumperla/learning_ray/main/notebooks/images/chapter_02/task_dependency.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rayアクター
# MAGIC
# MAGIC このサンプルでは、Ray Coreのより重要な側面の一つをカバーします。このステップまでは、すべては基本的には関数でした。特定の関数をリモート実行させるためには`@ray.remote`デコレーターを使用しましたが、それ以外の部分に関しては標準的なPythonを使っただけです。
# MAGIC
# MAGIC データベースがどの程度の頻度でクエリーされるのかを追跡したい場合、取得タスクの結果をカウントすることができるでしょう。しかし、これを行うためのより効率的な方法は無いのでしょうか？理想的には、大規模なデータを取り扱えるように、この処理は分散されて欲しいと考えることでしょう。Rayでは、クラスター上でステートフルな処理を実行し、互いにコミュニケーションを行うこともできるアクターによるソリューションを提供します。デコレートした関数を用いたRayタスクの作成と同様に、デコレートしたPythonクラスを用いてRayアクターを作成します。これによって、データベース呼び出しの回数を追跡するために、Rayアクターを用いたシンプルなカウンターを作成することができます。

# COMMAND ----------

@ray.remote
class DataTracker:
    def __init__(self):
        self._counts = 0

    def increment(self):
        self._counts += 1

    def counts(self):
        return self._counts

# COMMAND ----------

# MAGIC %md
# MAGIC `ray.remote`デコレーターを追加すると、DataTrackerクラスはアクターになります。このアクターは、カウントのような状態を追跡することができ、そのメソッドは`.remote()`を用いた関数と同じように起動できるRayアクターのタスクとなります。このアクターと連携するようにretrieve_taskを変更します。

# COMMAND ----------

@ray.remote
def retrieve_tracker_task(item, tracker, db):
    time.sleep(item / 10.)
    tracker.increment.remote()
    return item, db[item]


tracker = DataTracker.remote()

object_references = [
    retrieve_tracker_task.remote(item, tracker, db_object_ref) for item in range(8)
]
data = ray.get(object_references)

print(data)
print(ray.get(tracker.counts.remote()))

# COMMAND ----------

# MAGIC %md
# MAGIC 期待したように、この計算による結果は8となります。この計算の実行にアクターを必要とはしませんが、このクラスターにおける状態を維持し、複数のタスクを伴う方法をデモンストレートしています。実際のところ、任意の関連タスクや、別のアクターのコンストラクタにすら、このアクターを引き渡す頃ができます。RayのAPIは柔軟であり、無制限の可能性を提供します。強化学習のような複雑な分散アルゴリズムを実行する際にはとくに有用な、ステートフルな計算処理を可能とする分散Pythonツールはレアです。

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## サマリー
# MAGIC
# MAGIC この例では、6つのAPIメソッドのみを使用しました。これらには、クラスターを起動するための`ray.init()`、関数やクラスをタスクやアクターに変換する`@ray.remote`、Rayのオブジェクトストアに値を転送する`ray.put()`、クラスターからオブジェクトを取得する`ray.get()`。さらに、クラスターでコードを実行するためにアクターメソッドやタスクの`.remote()`を使用し、ブロッキングコールを避けるために`ray.wait`を使用しました。
# MAGIC
# MAGIC RayのAPIはこれら6つの呼び出し以上のものから構成されていますが、あなたが使い始める際には、これらの6つはパワフルなものとなります。より一般的にまとめると、これらのメソッドは以下の通りとなります:
# MAGIC
# MAGIC - `ray.init()`: お使いのRayクラスターを初期化します。既存クラスターに接続するためにはアドレスを指定します。
# MAGIC - `@ray.remote`: 関数をタスクに、クラスをアクターに変換します。
# MAGIC - `ray.put()`: Rayのオブジェクトストアに値を保存します。
# MAGIC - `ray.get()`: オブジェクトストアから値を取得します。保存した値、あるいはタスクやアクターによって計算された値を返却します。
# MAGIC - `.remote()`: お使いのRayクラスターでアクターメソッドやタスクを実行し、アクターのインスタンスを作成する際に使用されます。
# MAGIC - `ray.wait()`: オブジェクト参照の2つのリストを返却し、一つは完了を待っていたタスクのうち完了したタスク、もう一つは未完了のタスクとなります。
