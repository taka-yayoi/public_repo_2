-- Databricks notebook source
-- MAGIC %md
-- MAGIC # SQL基礎 - テーブルの参照
-- MAGIC 
-- MAGIC こちらではSQLの以下のSQLのうち、データの取り出しに使用する**DML**、特に参照系の操作にフォーカスして説明します。
-- MAGIC 
-- MAGIC - DDL(Data Definition Language)
-- MAGIC - DML(Data Manipulation Language)
-- MAGIC - DCL(Data Control Language)

-- COMMAND ----------

-- 事前に作成しておいたデータベースを選択します
USE japan_covid_takaakiyayoidatabrickscom;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## SELECT
-- MAGIC 
-- MAGIC テーブルからデータを取り出す際に必須になるのが`SELECT`です。

-- COMMAND ----------

-- テーブル「covid_cases」から全レコードを取得します
SELECT
  *
FROM
  covid_cases;

-- COMMAND ----------

-- レコード件数を取得します
SELECT
  count(*)
FROM
  covid_cases;

-- COMMAND ----------

-- ユニークな県名のみを取得します
SELECT
  DISTINCT Prefecture,
  pref_no
FROM
  covid_cases
ORDER BY -- 県番号の昇順でソートします
  pref_no ASC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## GROUP BY
-- MAGIC 
-- MAGIC 集計キーを指定してテーブルのデータをグルーピングします。

-- COMMAND ----------

SELECT
  Area,
  COUNT(*) AS count -- レコード数をカウントし、列名をcountにします
FROM
  covid_cases
GROUP BY -- Areaでグルーピングします
  Area
ORDER BY -- Areaごとの件数の降順でソートします
  count DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ビュー
-- MAGIC 
-- MAGIC ビューは保存済みのクエリーと言えます。

-- COMMAND ----------

CREATE
OR REPLACE VIEW tokyo_only AS
SELECT
  *
FROM
  covid_cases
WHERE -- Tokyoのレコードのみに限定します
  Prefecture = "Tokyo"

-- COMMAND ----------

SELECT * FROM tokyo_only;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 関数
-- MAGIC 
-- MAGIC 頻繁に使用する処理は関数によって実現できます。

-- COMMAND ----------

SELECT
  date(date_timestamp) || " : " || Cases AS date_case -- 日付と感染者数を結合して date_case という列を生成します
FROM
  tokyo_only;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 述語(predicate)
-- MAGIC 
-- MAGIC テーブルから取得するデータを絞り組むには述語を指定します。

-- COMMAND ----------

SELECT
  *
FROM
  covid_cases
WHERE
  date(date_timestamp) BETWEEN "2022-01-01"
  AND "2022-01-10"
  AND Lower(Prefecture) LIKE "%shima%"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## CASE
-- MAGIC 
-- MAGIC 取得するデータの内容に対して判定を行い、新たなデータを生成することができます。

-- COMMAND ----------

SELECT
  Prefecture,
  CASE -- Prefectureの文字列に条件を適用した結果に基づく新たな列を作成します
    WHEN Lower(Prefecture) LIKE "%shima%" THEN "島を含む"
    ELSE "島を含まない"
  END AS shima_included
FROM
  covid_cases
GROUP BY
  Prefecture

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## UNION
-- MAGIC 
-- MAGIC テーブルのデータを**縦方向**に連結するには`UNION`を使います。

-- COMMAND ----------

CREATE
OR REPLACE VIEW chiba_only AS
SELECT
  *
FROM
  japan_covid_takaakiyayoidatabrickscom.covid_cases
WHERE
  Prefecture = "Chiba"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ここまでで作成したテーブル・ビューを確認します。

-- COMMAND ----------

SHOW TABLES;

-- COMMAND ----------

SELECT
  *
FROM
  tokyo_only
UNION
SELECT
  *
FROM
  chiba_only;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## JOIN
-- MAGIC 
-- MAGIC テーブルのデータを**横方向**に連結するには`JOIN`を使います。
-- MAGIC 
-- MAGIC 以下のテーブルの元データは[こちら](https://github.com/taka-yayoi/public_repo/blob/main/covid-19%20analytics%20with%20lakehouse/prefectural_capital.csv)にあります。

-- COMMAND ----------

SELECT * FROM prefectural_capital;

-- COMMAND ----------

SELECT
  cc.*,
  pc.prefectural_capital
FROM
  covid_cases cc
  INNER JOIN prefectural_capital pc ON cc.Prefecture = pc.Prefecture;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 分析の文脈でのSQLの活用
-- MAGIC 
-- MAGIC 今では、機械学習やBIの文脈でSQLを活用するケースが増えています。

-- COMMAND ----------

SELECT
  Area,
  date(date_timestamp) AS date,
  SUM(Cases) AS total_cases
FROM
  covid_cases
GROUP BY
  Area,
  date
ORDER BY
  Area,
  date;

-- COMMAND ----------

SELECT
  Prefecture,
  SUM(Cases) AS totale_cases
FROM
  covid_cases
GROUP BY
  Prefecture

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
