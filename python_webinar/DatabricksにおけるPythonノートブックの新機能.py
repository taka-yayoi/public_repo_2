# Databricks notebook source
# MAGIC %md
# MAGIC ### Monacoによるコード編集の改善
# MAGIC 
# MAGIC - [新たなエディタの有効化](https://qiita.com/taka_yayoi/items/8dd3cabb54b08f4ed604#%E6%96%B0%E3%81%9F%E3%81%AA%E3%82%A8%E3%83%87%E3%82%A3%E3%82%BF%E3%81%AE%E6%9C%89%E5%8A%B9%E5%8C%96)に従って新エディタを有効化してください。
# MAGIC - 稼働中のクラスターにノートブックをアタッチしてください。

# COMMAND ----------

# MAGIC %md
# MAGIC #### タイプしながらオートコンプリート
# MAGIC ![autocomplete](https://github.com/RafiKurlansik/notebook2/blob/main/assets/autocomplete2.gif?raw=true)

# COMMAND ----------

import numpy as np

# 次の行に右の内容をタイプしてください: a = np.array([1,2,3])


# COMMAND ----------

# MAGIC %md 
# MAGIC #### マウスホバーによる変数の調査
# MAGIC ![var-inspect](https://github.com/RafiKurlansik/notebook2/blob/main/assets/variable_inspection.gif?raw=true)

# COMMAND ----------

# このコードを実行し3行目の'variable'にマウスカーソルを移動します
variable = np.array([1,2,3])
variable

# COMMAND ----------

# MAGIC %md
# MAGIC #### コードの折りたたみ & 括弧のマッチング
# MAGIC 
# MAGIC コードブロックの隣の矢印アイコンをクリックすることで折り畳むことができます:
# MAGIC 
# MAGIC ![code-folding](https://github.com/RafiKurlansik/notebook2/blob/main/assets/code_folding.gif?raw=true)
# MAGIC 
# MAGIC 括弧の隣をクリックすることで、対応する括弧をハイライトします:
# MAGIC 
# MAGIC ![bracket-matching](https://github.com/RafiKurlansik/notebook2/blob/main/assets/bracket_matching.gif?raw=true)

# COMMAND ----------

# 行9、17、24、31のコードブロックを折りたたみ、展開します
!pip install folium --quiet

import json
import folium
import requests

url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
vis1 = json.loads(requests.get(f"{url}/vis1.json").text)
vis2 = json.loads(requests.get(f"{url}/vis2.json").text)
vis3 = json.loads(requests.get(f"{url}/vis3.json").text)
m = folium.Map(location=[46.3014, -123.7390], zoom_start=7, tiles="Stamen Terrain")

folium.Marker(
    location=[47.3489, -124.708],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis1, width=450, height=250)
    ),
).add_to(m)

folium.Marker(
    location=[44.639, -124.5339],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis2, width=450, height=250)
    ),
).add_to(m)

folium.Marker(
    location=[46.216, -124.1280],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis3, width=450, height=250)
    ),
).add_to(m)

m

# COMMAND ----------

# MAGIC %md
# MAGIC #### マルチカーソルのサポート
# MAGIC 
# MAGIC ![multi-cursor-support](https://github.com/RafiKurlansik/notebook2/blob/main/assets/multi_cursor.gif?raw=true)
# MAGIC 
# MAGIC Monacoは高速に同時編集を行うためのマルチカーソルをサポートしています。Windowsであれば、`Alt+Click`で2つ目のカーソル(細く表示されます)を追加することができます。それぞれのカーソルは文脈に合わせて独立に動作します。さらに多くのカーソルを追加する一般的な方法は、上や下にカーソルを挿入するために`⌥⌘↓`や`⌥⌘↑`を使うというものです。([source](https://code.visualstudio.com/Docs/editor/codebasics#_multiple-selections-multicursor))

# COMMAND ----------

# 行2にカーソルを移動し、行3、行4に追加のカーソルを作成するために option+command+down ( ⌥⌘↓ ) を押します




# COMMAND ----------

# MAGIC %md 
# MAGIC #### カラム(ボックス)選択
# MAGIC 
# MAGIC ![boxselection](https://github.com/RafiKurlansik/notebook2/blob/main/assets/column_selection.gif?raw=true)
# MAGIC 
# MAGIC 一方の角にカーソルを移動し、反対側にドラッグする際に`Shift+Alt` (あるいはMacなら `Shift+option` )を押し続けます。 ([source](https://code.visualstudio.com/Docs/editor/codebasics#_column-box-selection))

# COMMAND ----------

print("""
PassengerId,Name,Sex,Age,SibSp
1,"Braund, Mr. Owen Harris",male,22.0,1
2,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38.0,1
3,"Heikkinen, Miss. Laina",female,26.0,0
4,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35.0,1
5,"Allen, Mr. William Henry",male,35.0,0
6,"Moran, Mr. James",male,,0
7,"McCarthy, Mr. Timothy J",male,54.0,0
8,"Palsson, Master. Gosta Leonard",male,2.0,3
9,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27.0,0
10,"Nasser, Mrs. Nicholas (Adele Achem)",female,14.0,1
11,"Sandstrom, Miss. Marguerite Rut",female,4.0,1
12,"Bonnell, Miss. Elizabeth",female,58.0,0
13,"Saundercock, Mr. William Henry",male,20.0,0
14,"Andersson, Mr. Anders Johan",male,39.0,1
15,"Vestrom, Miss. Hulda Amanda Adolfina",female,14.0,0
""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 選択テキストの実行
# MAGIC 
# MAGIC テキストをハイライトし `Shift + Control + Enter` で実行します:
# MAGIC 
# MAGIC ![run_selected](https://github.com/RafiKurlansik/notebook2/blob/main/assets/run_selected.gif?raw=true)

# COMMAND ----------

# 試してみましょう
print(1)
print(2)
print(3)
print(4)
print(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 行コメントの切り替え
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/notebook2/blob/main/assets/togglecomment.gif?raw=true" width=500/>
# MAGIC 
# MAGIC ユーザーがドキュメントされたコードを迅速に作成できるように、コメントの作成を容易にしました。コードのコメント行や複数行のコメントを一度に切り替えることができます。このためには、対象の行を選択しOSに応じたショートカットを押します。
# MAGIC 
# MAGIC Mac: `Cmd + /`
# MAGIC 
# MAGIC Windows: `Ctrl + /`

# COMMAND ----------

print(1)
print(2)
print(3)
print(4)
print(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ブロッククオート
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/notebook2/blob/main/assets/blockquote.gif?raw=true" width=600>
# MAGIC 
# MAGIC また、プログラミング言語がブロッククオートをサポートしている場合には、ノートブックでもそれをサポートしています。ユーザーは、コードブロック全体をクオートするためにコマンドを使用することができます。
# MAGIC 
# MAGIC Mac:		`Shift+Option+A`
# MAGIC 
# MAGIC Windows: 	`Shift+Option+A`

# COMMAND ----------

# print()の中身をクオートするために、テキストを選択して Shift+Option+A を押します
print(this is a 
multi-line text 
that you can put in quotes with
Shift + Option + A)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pythonのフォーマッティング/リンティング

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/notebook2/blob/main/assets/blackformatter2.gif?raw=true" width="500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC [Python Formatter Public Preview Docs](https://docs.databricks.com/notebooks/notebooks-use.html#format-code-cells)
# MAGIC 
# MAGIC DatabricksではPEP 8互換のコードフォーマッターであるBlackをサポートしました。Blackはすべてのコードを同じ方法でフォーマットするので、フォーマットに要する時間を削減し、すべきことに集中できるようになります。Blackによってフォーマットされたコードはレビューしているプロジェクトに関係なく、同じように見えるので、コードレビューも高速になります。Blackを使うには、DBR11.2以降のクラスターに接続します。DatabricksにはBlackとTokenize-rtがプレインストールされています。
# MAGIC 
# MAGIC **試すには、以下のセルをクリックし、ノートブックメニューバーから、 _Edit --> Format cell(s)_ をクリックします。** ([example source](https://www.freecodecamp.org/news/auto-format-your-python-code-with-black/)).

# COMMAND ----------

def add(a,        b):
    answer  =  a   +       b

    return    answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### デバッグ
# MAGIC 
# MAGIC Databricksランタイム11.2以降で動作するノートブックでは、[The Python Debugger](https://docs.python.org/3/library/pdb.html) (pdb)がサポートされています。
# MAGIC 
# MAGIC ノートブックでpdbを使うサンプルとしては:
# MAGIC 
# MAGIC * 最後の例外からデバッグするために`%debug`を使います。これは、予期しないエラーに遭遇し、原因をデバッグしようとする際に役立ちます(`pdb.pm()`と同じようなものです)。
# MAGIC * 例外の後に(しかしプログラムが停止する前に)インタラクティブなデバッガを自動で起動させるために`%pdb on`を使います。
# MAGIC 
# MAGIC これらのコマンドを使う際、他のセルを実行できるようにするためにはデバッガの使用を終了しなくてはいけないことに注意してください。デバッガーの終了方法はいくつかそんざいします:
# MAGIC 
# MAGIC * セルの実行を終了するために`c`か`continue`を入力。
# MAGIC * エラーをスローし、コード実行を停止するために`exit`を入力。
# MAGIC * 出力ボックスの隣の`Cancel`をクリックすることでコマンドをキャンセル。

# COMMAND ----------

# MAGIC %md 
# MAGIC ####` %debug` : 事後のデバッグ
# MAGIC Databricksノートブックで`%debug`を使うには:
# MAGIC 1. 例外が起きるまでノートブックのコマンドを実行します。
# MAGIC 2. 新規セルで `%debug` を実行します。セルの出力エリアでデバッガーが動作を始めます。
# MAGIC 3. 変数を調査するには、入力フィールドに変数名を入力し **Enter** を押します。  
# MAGIC 4. 以下のコマンドを用いることで、変数の調査の様に、コンテキストを切り替えたり他のデバッガータスクを行うことができます。デバッガーの完全なコマンドの一覧に関しては、[pdb documentation](https://docs.python.org/3/library/pdb.html)をご覧ください。文字列を入力し、 **Enter** を押します。  
# MAGIC    - `n`: 次の行
# MAGIC    - `u`: 現在のスタックフレームを抜けて1レベル上に移動
# MAGIC    - `d`: 現在のスタックフレームを抜けて1レベル下に移動
# MAGIC 5. このノートブックの最初のセルで説明した方法のいずれかでデバッガーを抜けます。
# MAGIC 
# MAGIC `%debug`を用いたこれらのステップのサンプルを以下に示します。

# COMMAND ----------

class ComplexSystem1:
  def getAccuracy(self, correct, total):
    # ...
    accuracy = correct / total
    # ...
    return accuracy
  
class UserTest:
  def __init__(self, system, correct, total):
    self.system = system
    self.correct = correct
    self.total = 0 # 間違った合計を設定しています！
    
  def printScore(self):
    print(f"You're score is: {self.system.getAccuracy(self.correct, self.total)}")
  
test = UserTest(
  system = ComplexSystem1(),
  correct = 10,
  total = 100
)
 
test.printScore()

# COMMAND ----------

# MAGIC %debug

# COMMAND ----------

# MAGIC %md
# MAGIC #### `%pdb on` : 事前のデバッグ
# MAGIC 
# MAGIC Databricksノートブックで`%pdb on`を使うには:
# MAGIC 1. ノートブックの最初のセルで`%pdb on`を実行して自動pdbをオンにします。
# MAGIC 1. 例外が起きるまでノートブックでコマンドを実行します。インタラクティブなデバッガーが起動します。
# MAGIC 1. 変数を調査するには、入力フィールドに変数名を入力し **Enter** を押します。
# MAGIC 1. 以下のコマンドを用いることで、変数の調査の様に、コンテキストを切り替えたり他のデバッガータスクを行うことができます。デバッガーの完全なコマンドの一覧に関しては、[pdb documentation](https://docs.python.org/3/library/pdb.html)をご覧ください。文字列を入力し、 **Enter** を押します。  
# MAGIC    - `n`: 次の行
# MAGIC    - `u`: 現在のスタックフレームを抜けて1レベル上に移動
# MAGIC    - `d`: 現在のスタックフレームを抜けて1レベル下に移動
# MAGIC 1. このノートブックの最初のセルで説明した方法のいずれかでデバッガーを抜けます。
# MAGIC 
# MAGIC `%pdb on`を用いたこれらのステップのサンプルを以下に示します。

# COMMAND ----------

# MAGIC %pdb on

# COMMAND ----------

class ComplexSystem2:
  def getAccuracy(self, correct, total):
    # ...
    accuracy = correct / total
    # ...
    return accuracy
 
system = ComplexSystem2()
 
## テストのカバレッジ
print("Tests")
print(system.getAccuracy(10, 100) == 0.1)
print(system.getAccuracy(10, 0), 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### バージョン履歴における隣り合わせのdiff
# MAGIC 
# MAGIC ![diffs](https://github.com/RafiKurlansik/notebook2/blob/main/assets/diffs.gif?raw=true)

# COMMAND ----------

# MAGIC %md
# MAGIC ### シンタックスのハイライト
# MAGIC 
# MAGIC 適切にコードがハイライトされることで、コーディング、編集、トラブルシュートをより迅速なものにします。今では、**ノートブックはPythonコードセルのSQLコードを認識し、ハイライトします。**これまでは、意図した通りに実行されるようにするために、手動でコードをパースしなくてはなりませんでした。パッと見るだけで、ユーザーは適切なSQL文が含まれ、適切な順序に並んでいることを確認することができます。

# COMMAND ----------

df = spark.sql('''
select carat, cut, color, price 
from default.diamonds
where price < 350
''')

# COMMAND ----------

# MAGIC %md
# MAGIC # END
