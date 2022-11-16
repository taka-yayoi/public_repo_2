# Databricks notebook source
# MAGIC %md
# MAGIC # レコメンデーションシステムにおけるwide-and-deepモデルの構築、サービング
# MAGIC 
# MAGIC このノートブックでは、レコメンデーションシステムにおけるwide-and-deeopモデルの構築方法と、Databricksにおけるサービングの方法をデモンストレーションします。
# MAGIC 
# MAGIC ## 要件
# MAGIC * Databricks機械学習ランタイム8.2以降
# MAGIC * このノートブックはデータ生成ノートブックの出力を前提としています。([AWS](https://docs.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/reference-solutions/recommender-wide-n-deep)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html))

# COMMAND ----------

import mlflow
import pandas as pd
import platform
import tensorflow as tf
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from tensorflow.python.saved_model import tag_constants

# COMMAND ----------

# データ生成ノートブックの出力パス
DATA_DBFS_ROOT_DIR = '/tmp/recommender/data'

# 必要に応じて変更してください 
CACHE_PATH = 'file:///dbfs/tmp/recommender/converter_cache'
CKPT_PATH = '/dbfs/tmp/recommender/model_ckpt'
EXPORT_PATH = '/dbfs/tmp/recommender/model_export'

# メトリクスとモデルを記録するためにmlflow autologを有効化
mlflow.tensorflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. モデルの作成
# MAGIC 
# MAGIC このステップのアウトプットは `tf.estimator.DNNLinearCombinedClassifier` estimatorオブジェクトです。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1.1 入力カラムの定義
# MAGIC 
# MAGIC 関数`get_wide_and_deep_columns()`は、それぞれの要素が`tf.feature_column`のリストである`(wide_columns, deep_columns)`のタプルを返却します。
# MAGIC 
# MAGIC `LABEL_COLUMN`でラベルカラムの名前を指定します。
# MAGIC 
# MAGIC ご自分のデータにこのリファレンスソリューションを適用するには、お使いのデータセットに対応した適切なカラムを返却する`get_wide_and_deep_columns()`関数を実装する必要があります。

# COMMAND ----------

LABEL_COLUMN = 'label'
NUMERIC_COLUMNS = [
  'user_age',
  'item_age',
]
CATEGORICAL_COLUMNS = [
  'user_id',
  'item_id',
  'user_topic',
  'item_topic',
]
HASH_BUCKET_SIZES = {
  'user_id': 400,
  'item_id': 2000,
  'user_topic': 10,
  'item_topic': 10,
}
EMBEDDING_DIMENSIONS = {
    'user_id': 8,
    'item_id': 16,
    'user_topic': 3,
    'item_topic': 3,
}

def get_wide_and_deep_columns():
  wide_columns, deep_columns = [], []

  # エンべディングカラム
  for column_name in CATEGORICAL_COLUMNS:
      categorical_column = tf.feature_column.categorical_column_with_identity(
          column_name, num_buckets=HASH_BUCKET_SIZES[column_name])
      wrapped_column = tf.feature_column.embedding_column(
          categorical_column,
          dimension=EMBEDDING_DIMENSIONS[column_name],
          combiner='mean')

      wide_columns.append(categorical_column)
      deep_columns.append(wrapped_column)
  
  # ageカラムとcrossカラム
  user_age = tf.feature_column.numeric_column("user_age", shape=(1,), dtype=tf.float32)
  item_age = tf.feature_column.numeric_column("item_age", shape=(1,), dtype=tf.float32)       
  user_age_buckets = tf.feature_column.bucketized_column(user_age, boundaries=[18, 35])
  item_age_buckets = tf.feature_column.bucketized_column(item_age, boundaries=[18, 35])
  age_crossed = tf.feature_column.crossed_column([user_age_buckets, item_age_buckets], 9)
  wide_columns.extend([user_age_buckets, item_age_buckets, age_crossed])
  deep_columns.extend([user_age, item_age])
  
  # topicカラムとcrossカラム
  user_topic = tf.feature_column.categorical_column_with_identity(
          "user_topic", num_buckets=HASH_BUCKET_SIZES["user_topic"])
  item_topic = tf.feature_column.categorical_column_with_identity(
          "item_topic", num_buckets=HASH_BUCKET_SIZES["item_topic"])       
  topic_crossed = tf.feature_column.crossed_column([user_topic, item_topic], 30)
  wide_columns.append(topic_crossed)
  return wide_columns, deep_columns

# COMMAND ----------

wide_columns, deep_columns = get_wide_and_deep_columns()

# COMMAND ----------

wide_columns

# COMMAND ----------

deep_columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 wide-and-deepモデルの定義

# COMMAND ----------

# 以下の問題を回避するためにoptimizerオブジェクトではなくoptimizer callablesを渡します:
# https://stackoverflow.com/questions/58108945/cannot-do-incremental-training-with-dnnregressor
estimator = tf.estimator.DNNLinearCombinedClassifier(
    # wideの設定
    linear_feature_columns=wide_columns,
    linear_optimizer=tf.keras.optimizers.Ftrl,  # 意図的に中括弧を含めていません
    # deepの設定
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50],
    dnn_optimizer=tf.keras.optimizers.Adagrad,  # 意図的に中括弧を含めていません
    # warm-start設定
    model_dir=CKPT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. カスタムメトリックの作成
# MAGIC 
# MAGIC このノートブックでは評価メトリックとしてkにおける平均精度を使用しています。
# MAGIC 
# MAGIC TensorFlowはビルトインのメトリック`tf.compat.v1.metrics.average_precision_at_k`を提供しています。動作を理解するには、[`tf.compat.v1.metrics.average_precision_at_k`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/metrics/average_precision_at_k)や[this answer in stackoverflow](https://stackoverflow.com/a/52055189/12165968)をご覧ください。

# COMMAND ----------

# 右を修正: https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/WideAndDeep/utils/metrics.py
def map_custom_metric(features, labels, predictions):
  user_ids = tf.reshape(features['user_id'], [-1])
  predictions = predictions['probabilities'][:, 1]

  # ユニークなuser_id、インデックス、カウントを処理
  # 異なる二つの場所で同じuser_idがある場合にはソートが必要
  sorted_ids = tf.argsort(user_ids)
  user_ids = tf.gather(user_ids, indices=sorted_ids)
  predictions = tf.gather(predictions, indices=sorted_ids)
  labels = tf.gather(labels, indices=sorted_ids)

  _, user_ids_idx, user_ids_items_count = tf.unique_with_counts(
      user_ids, out_idx=tf.int64)
  pad_length = 30 - tf.reduce_max(user_ids_items_count)
  pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])

  preds = tf.RaggedTensor.from_value_rowids(
      predictions, user_ids_idx).to_tensor()
  labels = tf.RaggedTensor.from_value_rowids(
      labels, user_ids_idx).to_tensor()

  labels = tf.argmax(labels, axis=1)

  return {
      'map': tf.compat.v1.metrics.average_precision_at_k(
          predictions=pad_fn(preds),
          labels=labels,
          k=5,
          name="streaming_map")}

# COMMAND ----------

estimator = tf.estimator.add_metrics(estimator, map_custom_metric)

# COMMAND ----------

# MAGIC %md 
# MAGIC `tf.estimator.train_and_evaluate`の出力の'map'メトリックは、評価セットに対するモデルの結果である map@k となります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. データのロード
# MAGIC 
# MAGIC このステップのアウトプットは、エスティメーターの引数である`input_fn`、`input_fn_eval`、`input_fn_test`の作成に使用できる3つのSparkデータセットコンバーターとなります。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.1 トレーニングデータセットをSparkデータフレームにロード

# COMMAND ----------

def load_df(name):
  return spark.read.format("delta").load(f"{DATA_DBFS_ROOT_DIR}/{name}")

train_df = load_df("user_item_interaction_train")
val_df = load_df('user_item_interaction_val')
test_df = load_df('user_item_interaction_test')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 特徴量でトレーニングデータフレームを拡張

# COMMAND ----------

item_profile = load_df("item_profile")
user_profile = load_df("user_profile")

# COMMAND ----------

def join_features(df):
  return df.join(item_profile, on='item_id', how="left").join(user_profile, on='user_id', how="left")

# COMMAND ----------

train_df_w_features = join_features(train_df)
val_df_w_features = join_features(val_df)
test_df_w_features = join_features(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 `SparkDatasetConverter`を用いてSparkデータフレームを`tf.data.Dataset`に変換

# COMMAND ----------

# キャッシュディレクトリの指定
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_PATH)

# コンバーターの作成
# Section 4: Train and evaluate the model uses these converters
train_converter = make_spark_converter(train_df_w_features)    
val_converter = make_spark_converter(val_df_w_features)
test_converter = make_spark_converter(test_df_w_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. モデルのトレーニングと評価
# MAGIC 
# MAGIC このステップのアウトプットは、チェックポイントディレクトリに保存されたモデルです。
# MAGIC 
# MAGIC 関数`mlflow.tensorflow.autolog()`がそれぞれのランのメトリクスを記録します。これらの記録を参照するには、画面右上のビーカーアイコンをクリックします。

# COMMAND ----------

# 新たなデータセットに対してこのセルのコードスニペットを変更なしに使うことができます
def to_tuple(batch):
  """
  namedtupleタイプからtupleタイプにバッチを変換するユーティリティ関数
  """
  feature = {
    "user_id": batch.user_id,
    "user_age": batch.user_age,
    "user_topic": batch.user_topic,
    "item_id": batch.item_id,
    "item_age": batch.item_age,
    "item_topic": batch.item_topic,
  }
  if hasattr(batch, "label"):
    return feature, batch.label
  return feature, None

def get_input_fn(dataset_context_manager):
  """
  Sparkデータセットコンバーターによって返却されるtfデータセットから入力関数を作成するユーティリティ関数
  """
  def fn():
    return dataset_context_manager.__enter__().map(to_tuple)
  return fn

# COMMAND ----------

# モデルのチェックポイントを削除するにはこの行のコメントを解除してください
# dbutils.fs.rm('/tmp/recommender/model_ckpt', recurse=True)

# COMMAND ----------

train_tf_dataset = train_converter.make_tf_dataset()
val_tf_dataset = val_converter.make_tf_dataset()
test_tf_dataset = test_converter.make_tf_dataset()

train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(train_tf_dataset), max_steps=1250)
eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(val_tf_dataset))

with mlflow.start_run():
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  artifact_uri = mlflow.get_artifact_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. トレーニングしたモデルをエクスポートし、MLflowにモデルを記録
# MAGIC 
# MAGIC 以前のステップで保存したモデルのチェックポイントをエクスポートします。`tf.estimator.export_saved_model`を呼び出すことで、MLflowは自動でtensorflowのモデルを記録します。

# COMMAND ----------

feature_spec = tf.feature_column.make_parse_example_spec(
    wide_columns + deep_columns
)
fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    feature_spec
)
saved_model_path = estimator.export_saved_model(
    export_dir_base=EXPORT_PATH,
    serving_input_receiver_fn=fn
).decode("utf-8")

# COMMAND ----------

artifacts = {
  # 現在のアクティブなランに記録されたモデルのパスを取得
  "model": artifact_uri + '/model'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. オンラインレコメンダーの構築
# MAGIC 
# MAGIC これで、(`user_id`, `item_id`)ペアのインタラクションの確率を予測するwide-and-deepモデルを手に入れることができました。しかし、レコメンダーを構築するには、ユーザーとアイテムのリストが指定された際にアイテムのランキングを予測するモデルが必要です。
# MAGIC 
# MAGIC 特に、入力は以下のセルに示す様なフォーマットである必要があります。

# COMMAND ----------

input_pdf = pd.DataFrame(
  {
    "user_id": [1, 2],
    "item_id": [[1, 2, 3], [4, 5, 6]],
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC 期待するアウトプットは以下の様になります:
# MAGIC 
# MAGIC     user_id   ranking   probabilities                 
# MAGIC     1         [3, 2, 1]  [0.500410, 0.488329, 0.485111]
# MAGIC     2         [5, 4, 6]  [0.501141, 0.497792, 0.484209]
# MAGIC 
# MAGIC `mlflow.pyfunc.PythonModel`のサブクラスで[カスタム推論ロジックを実装](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel)することができます。

# COMMAND ----------

# お手元のデータセットの特徴量タイプにマッチさせるために少し変更することで、このセルのコードスニペットを活用することができます。
# https://www.tensorflow.org/tutorials/load_data/tfrecord をご覧ください。
def serialize_example(input_pdf):
  """
  取得した特徴量を用いてpandas DFをproto(tf.train.Example)にシリアライズ
  """
  proto_tensors = []
  for i in range(len(input_pdf)):
    feature = dict()
    for field in input_pdf:
      if field in NUMERIC_COLUMNS:
        feature[field] = tf.train.Feature(float_list=tf.train.FloatList(value=[input_pdf[field][i]]))
      else: 
        feature[field] = tf.train.Feature(int64_list=tf.train.Int64List(value=[input_pdf[field][i]]))

    # tf.train.Exampleを用いて特徴量のメッセージを作成します
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    proto_string = proto.SerializeToString()
    proto_tensors.append(tf.constant([proto_string]))
  return proto_tensors

def sort_by_group(input_pdf, results_list):
  """
  results_listの予測スコアを用いてinput_pdfの行をソート
  """
  result_pdf = input_pdf.copy()
  result_pdf['probabilities'] = [item['probabilities'][0, 1].numpy() for item in results_list]
  return result_pdf \
    .sort_values(by='probabilities', ascending=False) \
    .groupby("user_id") \
    .agg({'item_id': lambda x: x.to_list(), 'probabilities': lambda x: x.to_list()}) \
    .reset_index()

# カスタム推論ロジックを持つPythonModel
class Recommender(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    self.model = mlflow.tensorflow.load_model(context.artifacts["model"])    

  def predict(self, context, model_input):
    proto = serialize_example(model_input)
    results_list = []
    for p in proto:
      results_list.append(self.model(p))
    return sort_by_group(model_input, results_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 レコメンダーモデルの登録
# MAGIC 
# MAGIC 入力サンプルとして`input_features_pdf`を用い、モデルと一緒に登録することができます。

# COMMAND ----------

# この関数はユーザープロファイルの特徴量とアイテムプロファイルの特徴量を取得します
def retrieve_features(input_pdf):
  input_df = spark.createDataFrame(input_pdf.explode('item_id'))
  return join_features(input_df).toPandas().astype({'user_age': 'float', 'item_age': 'float'})

print("Retrieving features...")
input_features_pdf = retrieve_features(input_pdf)
input_features_pdf

# COMMAND ----------

# 必要なすべての依存関係を含む新たなMLflowモデルのためのConda環境を作成します。
# 使用している環境に合わせて以下のエントリーを修正して下さい。
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      f'python={platform.python_version()}',
      'pip',
      {
        'pip': [
          'mlflow==1.30.0',
          f'tensorflow-cpu=={tf.__version__}',
          'tensorflow-estimator',
        ],
      },
    ],
    'name': 'recommender_env'
}
# MLflow pyfunc PythonModelを記録します
mlflow_pyfunc_model_path = "recommender_mlflow_pyfunc"
mlflow.pyfunc.log_model(
  artifact_path=mlflow_pyfunc_model_path, 
  python_model=Recommender(), 
  artifacts=artifacts,
  conda_env=conda_env, 
  input_example=input_features_pdf, 
  registered_model_name="recommender")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 レコメンダーモデルのサービング

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ナビゲーションバーの「モデル」のモデル一覧から登録したモデルを特定します。登録モデルのモデルサービングを有効化します。詳細は([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/model-serving.html))をご覧ください。
# MAGIC 
# MAGIC サービングタブの**Request**ボックスに、以下のセルに表示されているコマンドの出力を入力します。

# COMMAND ----------

print("Request:")
print(input_features_pdf.to_json(orient='records'))

# COMMAND ----------

# MAGIC %md
# MAGIC サービングタブの**Response**ボックスに現れるアウトプットは以下の様になります:
# MAGIC ```
# MAGIC [{"user_id":1,"item_id":[3,2,1],"probabilities":[0.35572293400764465,0.3401849567890167,0.29400017857551575]},
# MAGIC {"user_id":2,"item_id":[5,6,4],"probabilities":[0.35679787397384644,0.3309633731842041,0.2681720554828644]}]
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 7. 制限
# MAGIC 
# MAGIC このノートブックではレコメンデーションシステムの構築におけるコアのステップのいくつかを説明しています。以下の重要なトピックはカバーしていません
# MAGIC 
# MAGIC ### 候補の選挙
# MAGIC 
# MAGIC 特定のユーザーにアイテムのリストをレコメンデーションすることに興味があるのであれば、候補の選挙(Candidate election)が特定ユーザーに対してアイテムのリストを返却する別のプロセスとなります。全アイテム数が大きくない場合には、候補としてすべてのアイテムのリストを使うことができます。このノートブックでは、外部の関数呼び出しによって選択された候補を取得できることを前提としています。
# MAGIC 
# MAGIC ### コールドスタート問題
# MAGIC 
# MAGIC コールドスタート問題は、新規ユーザーや新規アイテムに対してレコメンデーションを行う状況のことを示しています。このノートブックでは、コールドスタートをハンドリングする手段は含めていません。
