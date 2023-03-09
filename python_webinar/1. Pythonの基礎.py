# Databricks notebook source
# MAGIC %md
# MAGIC # Pythonの基礎
# MAGIC 
# MAGIC [Introduction to Data Analysis Workshop Series – Databricks](https://www.databricks.com/discover/introduction-to-data-analysis-workshop-series)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下のことを学びます:<br>
# MAGIC  - Pythonの基礎
# MAGIC    * 数値 & 文字列
# MAGIC    * 変数
# MAGIC    * Print文
# MAGIC    * リスト
# MAGIC    * Forループ
# MAGIC    * 関数
# MAGIC    * 条件文
# MAGIC    * 型と型チェック
# MAGIC    
# MAGIC [Pythonチートシート](https://atmarkit.itmedia.co.jp/ait/articles/2001/10/news018.html)をブックマークしておくと便利です。Pythonの使い方を学ぶ包括的なチュートリアルを必要としているのであれば、Pythonのウェブサイトにある[公式のチュートリアル](https://docs.python.org/3/tutorial/)をチェックしてください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) 数値: Pythonを計算機として使ってみましょう!

# COMMAND ----------

# 実行ボタン ▶︎ あるいは shift + enter を押します
1+1 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 文字列

# COMMAND ----------

'アイスクリーム'

# COMMAND ----------

'アイスクリーム' + 'は楽園です'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) 変数
# MAGIC 
# MAGIC シェイクスピアは言いました: 「バラはどんな名前で呼ばれても、同じように甘く香るのに」と。Pythonにおける変数はシンプルに値を保持します。(合理性のある範囲で)好きな名前で呼び出すことができます！

# COMMAND ----------

# 私はアイスクリームが好きです

# COMMAND ----------

best_food = 'アイスクリーム'
best_food

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) Print文
# MAGIC 
# MAGIC DatabricksやJupyterノートブックでは、評価したセルの最後の行が自動でプリントアウトされます。
# MAGIC 
# MAGIC しかし、`print`文を用いることで他のコンポーネントのプリントアウトを強制することができます。

# COMMAND ----------

print("やあ、一番好きな食べ物を教えてくれる？")
print(best_food)

# COMMAND ----------

# MAGIC %md
# MAGIC また、プリントする内容をより明示的なものにすることができます。print文の中に変数を追加することができます。

# COMMAND ----------

print(f"{best_food}が地球で一番の食べ物です。")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5) リスト
# MAGIC 
# MAGIC 皆さんが今朝食べた朝食のリストを作成してみましょう。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/yayoi/python_fundamentals/breakfast.jpg" width="20%" height="10%">

# COMMAND ----------

breakfast_list = ["パンケーキ", "卵", "ワッフル"]

# COMMAND ----------

breakfast_list.append("牛乳")
breakfast_list

# COMMAND ----------

# MAGIC %md
# MAGIC リストの最初にある朝食の要素を取り出しましょう。
# MAGIC 
# MAGIC **注意:** Pythonでは全てが0ベースのインデックスとなりますので、最初の要素は0の場所に存在します。

# COMMAND ----------

breakfast_list[0]

# COMMAND ----------

# MAGIC %md
# MAGIC リストから最後の朝食アイテムを取り出しましょう。

# COMMAND ----------

breakfast_list[-1]

# COMMAND ----------

# MAGIC %md
# MAGIC 2つ目以降のアイテムが必要な場合にはどうすればいいのでしょうか？

# COMMAND ----------

breakfast_list[1:]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6) 条件文
# MAGIC 
# MAGIC 時には、すべてのコードではなく、特定の条件に基づいて特定の行のみを実行したいと言うケースがあります。`if`、`elif`、`else`文を用いてこのような制御を行うことができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ここでは食べ物の複数形を表示したいものとします(`best_food`を他のものに変えてみてください)！ これを実現するは、単語の末尾に 's' がない場合にのみ、文字列の末尾に 's' を追加する必要があります。

# COMMAND ----------

best_food = "chocolate"

if best_food.endswith("s"):
  print(best_food)
else:
  print(best_food + "s")

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のセルの `best_food` を変更して、3つの異なるアウトプットを確認してみてください。

# COMMAND ----------

best_food = "アイスクリーム"
#best_food = ""
#best_food = "大福"
ice_cream_count = 1000

if best_food == "アイスクリーム":
  print(f"アイスクリームが {ice_cream_count} 個欲しいです。")
elif best_food == "":
  print("好きな食べ物を教えてください")
else:
  print("本当ですか？アイスクリームは好きではありませんか？")

# COMMAND ----------

# MAGIC %md
# MAGIC 変数が等しいかどうかは `==` (等しい) あるいは `!=` (異なる) でチェックすることができます。

# COMMAND ----------

best_food == "アイスクリーム"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7) Forループ
# MAGIC 
# MAGIC 今朝食べた朝食をすべてプリントしたい場合はどうしたらいいのでしょうか？
# MAGIC 
# MAGIC ループは順序に沿って、コードブロックを繰り返す手段です。("for-loop")

# COMMAND ----------

for food in breakfast_list:
  print(food)

# COMMAND ----------

# MAGIC %md
# MAGIC それぞれの単語に何文字含まれているのかをカウントしたい場合にはどうしたらいいのでしょうか？

# COMMAND ----------

for food in breakfast_list:
  print(f"{food} は {len(food)} 文字です。")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8) 関数
# MAGIC 
# MAGIC 上のコードは`breakfast_list`に対してのみ動作しますが、関数を作成することで別のリストに対応するように一般化することができます！`関数`は`def`キーワードの後に関数名、括弧無いに任意のパラメーター、そしてコロンを記述することで作成することができます。

# COMMAND ----------

# 一般的な構文
# def function_name(parameter_name):
#   関数が呼び出されるたびに実行されるコードブロック

# 関数の定義
def print_length(breakfast_list):
  for food in breakfast_list:
    print(f"{food} は {len(food)} 文字です。")

# print_lengthにリストを引き渡すことで関数を実行します
print_length(breakfast_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9) 複数の引数を持つ関数
# MAGIC 
# MAGIC 2つの値を加算する関数を定義してみましょう。

# COMMAND ----------

ice_cream_count = 1000
chocolate_count = 500

def count_fav_food(ice_cream, chocolate):
  return ice_cream + chocolate

count_fav_food(ice_cream_count, chocolate_count)

# COMMAND ----------

# MAGIC %md
# MAGIC 常に`chocolate`が500になることを知っている場合はどうでしょうか？デフォルト値を設定することができます！

# COMMAND ----------

def count_fav_food(ice_cream, chocolate=500):
  return ice_cream + chocolate

count_fav_food(ice_cream_count)

# COMMAND ----------

# MAGIC %md
# MAGIC どのくらいチョコレートが好きなのかを定量化するために、チョコレートのパーセンテージを計算したいとしたらどうでしょうか？

# COMMAND ----------

def chocolate_percentage(ice_cream, chocolate):
  percentage = chocolate / (ice_cream + chocolate) * 100
  return round(percentage, 2)

percent = chocolate_percentage(ice_cream_count, chocolate_count)
print(f"私は {percent}% の確率でチョコレートが好きです。")

# COMMAND ----------

# MAGIC %md
# MAGIC 関数のパラメーターを忘れた場合には、`help()`を呼び出すことができます。

# COMMAND ----------

help(count_fav_food)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10) 型と型チェック
# MAGIC 
# MAGIC ここまでで数多くの変数を定義してきました。変数の型を忘れてしまった場合にはどうすればいいでしょうか？心配しないでください。いつでもチェックすることができます！
# MAGIC 
# MAGIC 定義した変数は以下の通りです:
# MAGIC 
# MAGIC 1. `percent`
# MAGIC 1. `best_food`
# MAGIC 1. `breakfast_list`

# COMMAND ----------

type(percent)

# COMMAND ----------

type(best_food)

# COMMAND ----------

type(breakfast_list)

# COMMAND ----------

# MAGIC %md
# MAGIC 型のまとめです:
# MAGIC 1. `int` はPythonにおける数値の型です。整数値であり、小数を含まない数値全体となります。
# MAGIC 1. `float` はPythonにおける数値の型です。基本的に小数を持つ値となります。 
# MAGIC 1. `String` は、食べ物の`"チョコレート"`のように文字のシーケンスとなります。単語に限らず任意の文字列のシーケンスとなります。`"Hello123"` や `"123"` も文字列となります。これらは引用符で囲まれます。
# MAGIC 1. `Boolean` は Ture か False の値を取ります。
# MAGIC 
# MAGIC そして、変数エクスプローラ(Variable explorer)を用いることで、よりクイックに変数の型を確認することができます。
# MAGIC 
# MAGIC [Databricksノートブックで変数エクスプローラがサポートされます](https://qiita.com/taka_yayoi/items/b38fb466aeb0d23b805b)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
# MAGIC 
# MAGIC ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) おめでとうございます！Pythonの最初のレッスンを完了しました！
