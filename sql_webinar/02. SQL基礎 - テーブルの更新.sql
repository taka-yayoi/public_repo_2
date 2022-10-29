-- Databricks notebook source
-- MAGIC %md
-- MAGIC # SQL基礎 - テーブルの更新
-- MAGIC 
-- MAGIC こちらではSQLの以下のSQLのうち、データの取り出しに使用する**DML**、特にテーブル更新の操作にフォーカスして説明します。
-- MAGIC 
-- MAGIC - DDL(Data Definition Language)
-- MAGIC - DML(Data Manipulation Language)
-- MAGIC - DCL(Data Control Language)

-- COMMAND ----------

-- 事前に作成しておいたデータベースを選択します
USE japan_covid_takaakiyayoidatabrickscom;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## テーブルの作成
-- MAGIC 
-- MAGIC 更新処理を行うダミーのテーブルを作成します。なお、こちらの`DROP TABLE`や`CREATE TABLE`は**DDL**です。

-- COMMAND ----------

-- テーブルが存在する場合には削除
DROP TABLE IF EXISTS dummy_covid;

-- COMMAND ----------

-- covid_casesから取得したデータを使って新たにdummy_covidテーブルを作成します
CREATE TABLE dummy_covid COMMENT 'これは更新処理向けのダミーテーブルです。' AS
SELECT
  *
FROM
  covid_cases;

-- COMMAND ----------

-- テーブルのメタデータを確認します
DESCRIBE TABLE EXTENDED dummy_covid;

-- COMMAND ----------

-- 都道府県別の合計感染者数を合計します
SELECT
  Prefecture,
  sum(cases) AS total_cases
FROM
  dummy_covid
GROUP BY
  Prefecture;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## UPDATE
-- MAGIC 
-- MAGIC テーブルのデータを更新するには`UPDATE`を使用します。

-- COMMAND ----------

-- Hokkaidoの感染者数を100倍に更新します
UPDATE
  dummy_covid
SET
  cases = cases * 100
WHERE
  Prefecture = "Hokkaido";

-- COMMAND ----------

-- 都道府県別の合計感染者数を合計します
SELECT
  Prefecture,
  sum(cases) AS total_cases
FROM
  dummy_covid
GROUP BY
  Prefecture;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## DELETE
-- MAGIC 
-- MAGIC データを削除するには`DELETE`を使用します。削除対象を限定するためには`WHERE`を使用します。

-- COMMAND ----------

-- Hokkaidoのデータをすべて削除します
DELETE FROM
  dummy_covid
WHERE
  Prefecture = "Hokkaido";

-- COMMAND ----------

-- 都道府県別の合計感染者数を合計します
SELECT
  Prefecture,
  sum(cases) AS total_cases
FROM
  dummy_covid
GROUP BY
  Prefecture;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## INSERT
-- MAGIC 
-- MAGIC テーブルにデータを追加するには`INSERT`を使います。

-- COMMAND ----------

INSERT INTO
  dummy_covid(Prefecture, date_timestamp, cases, pref_no, Area)
VALUES("ぷりぷり県", "2022-10-28", 10, 99, "不明");

-- COMMAND ----------

SELECT
  *
FROM
  dummy_covid
ORDER BY
  pref_no DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 更新処理には注意が必要です
-- MAGIC 
-- MAGIC - 誤ってデータを消してしまった、上書きしてしまったということを避けることは正直困難です。
-- MAGIC - そんな場合に頼りになる機能があります。Delta Lakeの**タイムトラベル**です。
-- MAGIC - ここまで操作してきたテーブルの実態はDelta Lakeというソフトウェアによって実装されているものです。
-- MAGIC - Delta Lakeはトランザクション保証、インデックスなどの機能がサポートされていますが、上の様なケースで役立つのがデータのバージョン管理である**タイムトラベル**です。
-- MAGIC - Delta Lakeのテーブルに実行される更新処理はすべて記録され、任意のタイミングにロールバックすることができます。
-- MAGIC 
-- MAGIC [Delta Lakeにダイビング：トランザクションログを読み解く \- Qiita](https://qiita.com/taka_yayoi/items/b7f628c219463e055592)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### DESCRIBE HISTORY
-- MAGIC 
-- MAGIC [Delta Lakeにおけるテーブルユーティリティコマンド \- Qiita](https://qiita.com/taka_yayoi/items/152a31dc44bda51eeecd#delta%E3%83%86%E3%83%BC%E3%83%96%E3%83%AB%E5%B1%A5%E6%AD%B4%E3%82%92%E5%8F%96%E5%BE%97%E3%81%99%E3%82%8B)

-- COMMAND ----------

-- テーブルの更新履歴を確認します
DESCRIBE HISTORY dummy_covid;

-- COMMAND ----------

-- バージョン0、テーブル作成直後の状態を参照します
SELECT
  *
FROM
  dummy_covid VERSION AS OF 0
ORDER BY
  pref_no DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### SELECT AS OF

-- COMMAND ----------

-- 都道府県別の合計感染者数を合計します
SELECT
  Prefecture,
  sum(cases) AS total_cases
FROM
  dummy_covid VERSION AS OF 0
GROUP BY
  Prefecture;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### RESTORE
-- MAGIC 
-- MAGIC [RESTORE \(Delta Lake on Databricks\) \| Databricks on AWS](https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-restore.html)

-- COMMAND ----------

-- バージョン0の状態にレストアします
RESTORE TABLE dummy_covid TO VERSION AS OF 0;

-- COMMAND ----------

-- テーブルの更新履歴を確認します
DESCRIBE HISTORY dummy_covid;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
