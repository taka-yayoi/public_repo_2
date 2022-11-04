# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksノートブックにおけるデバッグ
# MAGIC 
# MAGIC Databricksランタイム11.2以降で動作するノートブックでは[Python Debugger](https://docs.python.org/3/library/pdb.html) (pdb)がサポートされます。
# MAGIC 
# MAGIC ノートブックにおけるpdbの使い方のサンプルを示します。
# MAGIC - 最後の例外からデバッグするには`%debug`を使います。これは、予期しないエラーに遭遇し、原因をデバッグする際に役立ちます(`pdb.pm()`と似ています)。
# MAGIC - 例外後(しかし、プログラム終了前)にインタラクティブなデバッガーを自動で起動するには`%pdb on`を使います。
# MAGIC 
# MAGIC これらのコマンドを使う際には、他のセルを実行する前にデバッガーの使用を停止する必要があることに注意してください。デバッガーを終了する方法はいくつかあります。
# MAGIC - 実行中のセルを終了するために`c`あるいは`continue`を入力
# MAGIC - エラーをスローしコード実行を停止するために`exit`を入力
# MAGIC - 出力ボックスの隣にある`Cancel`をクリックしてコマンドをキャンセル

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## `%debug` : 事後のデバッグ
# MAGIC Databricksノートブックで`%debug`を使うには:
# MAGIC 1. 例外が起きるまでノートブックのコマンドを実行します。
# MAGIC 2. 新規セルで `%debug` を実行します。セルの出力エリアでデバッガーが動作を始めます。
# MAGIC 3. 変数を調査するには、入力フィールドに変数名を入力し **Enter** を押します。  
# MAGIC 4. 以下のコマンドを用いることで、変数の調査の様に、コンテキストを切り替えたり他のデバッガータスクを行うことができます。デバッガーの完全なコマンドの一覧に関しては、Y[pdb documentation](https://docs.python.org/3/library/pdb.html)をご覧ください。文字列を入力し、 **Enter** を押します。  
# MAGIC    - `n`: 次の行
# MAGIC    - `u`: 現在のスタックフレームを抜けて1レベル上に移動
# MAGIC    - `d`: 現在のスタックフレームを抜けて1レベル下に移動
# MAGIC 5. このノートブックの最初のセルで説明した方法のいずれかでデバッガーを抜けます。
# MAGIC 
# MAGIC 以下に、`%debug`を用いたこれらのステップを実行する例を示しています。
# MAGIC 
# MAGIC *デバッガーに以下を入力しています。*
# MAGIC - total
# MAGIC - correct
# MAGIC - u
# MAGIC - self.total
# MAGIC - c

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
# MAGIC ## `%pdb on` : 事前のデバッグ
# MAGIC 
# MAGIC Databricksノートブックで`%pdb on`を使うには:
# MAGIC 1. ノートブックの最初のセルで`%pdb on`を実行して自動pdbをオンにします。
# MAGIC 1. 例外が起きるまでノートブックでコマンドを実行します。インタラクティブなデバッガーが起動します。
# MAGIC 1. 変数を調査するには、入力フィールドに変数名を入力し **Enter** を押します。
# MAGIC 1. 以下のコマンドを用いることで、変数の調査の様に、コンテキストを切り替えたり他のデバッガータスクを行うことができます。デバッガーの完全なコマンドの一覧に関しては、Y[pdb documentation](https://docs.python.org/3/library/pdb.html)をご覧ください。文字列を入力し、 **Enter** を押します。  
# MAGIC    - `n`: 次の行
# MAGIC    - `u`: 現在のスタックフレームを抜けて1レベル上に移動
# MAGIC    - `d`: 現在のスタックフレームを抜けて1レベル下に移動
# MAGIC 1. このノートブックの最初のセルで説明した方法のいずれかでデバッガーを抜けます。
# MAGIC 
# MAGIC 以下に、`%debug on`を用いたこれらのステップを実行する例を示しています。
# MAGIC 
# MAGIC *デバッガーに以下を入力しています。*
# MAGIC - correct / total
# MAGIC - correct
# MAGIC - u
# MAGIC - c

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
# MAGIC # END
