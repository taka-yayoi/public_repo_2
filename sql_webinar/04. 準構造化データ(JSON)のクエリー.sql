-- Databricks notebook source
-- 事前に作成しておいたデータベースを選択します
USE japan_covid_takaakiyayoidatabrickscom;

-- COMMAND ----------

CREATE TABLE store_data AS SELECT
'{
   "store":{
      "fruit": [
        {"weight":8,"type":"apple"},
        {"weight":9,"type":"pear"}
      ],
      "basket":[
        [1,2,{"b":"y","a":"x"}],
        [3,4],
        [5,6]
      ],
      "book":[
        {
          "author":"Nigel Rees",
          "title":"Sayings of the Century",
          "category":"reference",
          "price":8.95
        },
        {
          "author":"Herman Melville",
          "title":"Moby Dick",
          "category":"fiction",
          "price":8.99,
          "isbn":"0-553-21311-3"
        },
        {
          "author":"J. R. R. Tolkien",
          "title":"The Lord of the Rings",
          "category":"fiction",
          "reader":[
            {"age":25,"name":"bob"},
            {"age":26,"name":"jack"}
          ],
          "price":22.99,
          "isbn":"0-395-19395-8"
        }
      ],
      "bicycle":{
        "price":19.95,
        "color":"red"
      }
    },
    "owner":"amy",
    "zip code":"94025",
    "fb:testid":"1234"
 }' AS raw

-- COMMAND ----------

SELECT raw FROM store_data;

-- COMMAND ----------

SELECT raw:owner, RAW:owner FROM store_data;

-- COMMAND ----------

SELECT raw:`zip code`, raw:`Zip Code`, raw:['fb:testid'] FROM store_data;

-- COMMAND ----------

SELECT raw:store.bicycle FROM store_data;

-- COMMAND ----------

-- 配列のインデックス
SELECT raw:store.fruit[0], raw:store.fruit[1] FROM store_data;

-- COMMAND ----------

-- 配列からサブフィールドを抽出
SELECT raw:store.book[*].isbn FROM store_data;

-- COMMAND ----------

-- 配列内の配列、配列内のstructへのアクセス
SELECT
    raw:store.basket[*],
    raw:store.basket[*][0] first_of_baskets,
    raw:store.basket[0][*] first_basket,
    raw:store.basket[*][*] all_elements_flattened,
    raw:store.basket[0][2].b subfield
FROM store_data;

-- COMMAND ----------

-- priceは文字列ではなくdoubleとして返却されます
SELECT raw:store.bicycle.price::double FROM store_data;

-- COMMAND ----------

-- より複雑な型にキャストするためにfrom_jsonを使用します
SELECT from_json(raw:store.bicycle, 'price double, color string') bicycle FROM store_data
-- 返却されるカラムはpriceとcolorを含むstructになります

-- COMMAND ----------


