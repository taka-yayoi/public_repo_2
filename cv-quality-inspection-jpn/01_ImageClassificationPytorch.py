# Databricks notebook source
# MAGIC %md
# MAGIC # pytorchモデルの実装とデプロイ
# MAGIC
# MAGIC データサイエンティストとしての次のステップは、画像分類を実行するためにMLモデルを実装することです。
# MAGIC
# MAGIC トレーニングデータセットとして以前のデータパイプラインで構築したゴールドテーブルを再利用します。
# MAGIC
# MAGIC [torchvision](https://pytorch.org/vision/stable/index.html)を用いることで、このようなモデルの構築が非常にシンプルになります。
# MAGIC
# MAGIC ## MLOpsのステップ
# MAGIC
# MAGIC 画像分類モデルの構築は簡単に終わるかもしれませんが、プロダクション環境にモデルをデプロイすることはさらに難しいものとなります。
# MAGIC
# MAGIC Databricksにおいては、以下を提供するMLflowの助けを借りて、このプロセスをシンプルにし、価値創出に至るジャーニーを加速します。
# MAGIC
# MAGIC * 進捗を追跡し続ける自動エクスペリメント追跡
# MAGIC * ベストなモデルを得るためにhyperoptを用いたシンプルかつ分散されたハイパーパラメータチューニング
# MAGIC * MLフレームワークを抽象化し、MLflowにモデルをパッケージング
# MAGIC * ガバナンスのためのモデルレジストリ
# MAGIC * バッチやリアルタイムのサービング(1クリックでのデプロイメント)

# COMMAND ----------

model_name = "cv_pcb_classification_taka" # 適宜変更

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPUの有効化
# MAGIC
# MAGIC ディープラーニングでは、トレーニングでGPUを用いることが合理的です。

# COMMAND ----------

import torch

# GPUを利用できるかどうかをチェック
if not torch.cuda.is_available():  # gpuが利用できるかどうか
    raise Exception(
        "Please use a GPU-cluster for model training, CPU instances will be too slow"
    )

# COMMAND ----------

# Spark/Pythonバージョンの確認
import sys

print(
    "You are running a Databricks {0} cluster leveraging Python {1}".format(
        spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"),
        sys.version.split(" ")[0],
    )
)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

from PIL import Image

import torchvision
import torch

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import horovod.torch as hvd
from sparkdl import HorovodRunner

import mlflow

import pyspark.sql.functions as f

import numpy as np
from functools import partial
import io
import uuid

username = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
mlflow.set_experiment("/Users/{}/pcbqi".format(username))

petastorm_path = (
    f"file:///dbfs/tmp/petastorm/{str(uuid.uuid4())}/cache"  # petastormのキャッシュファイルの格納場所
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## データをトレーニング/テストデータセットとして分割
# MAGIC
# MAGIC 他のMLモデルと同様に、画像をトレーニング/テストデータセットに分割するところからスタートします。

# COMMAND ----------

# MAGIC %sql
# MAGIC USE takaakiyayoi_catalog.pcb;

# COMMAND ----------

# 対象画像の取得
images = spark.table("circuit_board_gold").select(
    "content", "label", "filename"
)  # 次のステップではユニークなIDとしてパスを使用します

# 階層化された画像のサンプルの取得
images_train = images.sampleBy(
    "label", fractions={0: 0.8, 1: 0.8}
)  # トレーニングにはそれぞれのクラスの80%をサンプリング
images_test = images.join(
    images_train, on="filename", how="leftanti"
)  # 残りはテストに使用

# 不要なフィールドを削除
images_train = images_train.drop("filename").repartition(
    sc.defaultParallelism
)  # パスのIDの削除
images_test = images_test.drop("filename").repartition(sc.defaultParallelism)

# サンプリングの検証
display(
    images_train.withColumn("eval_set", f.lit("train"))
    .union(images_test.withColumn("eval_set", f.lit("test")))
    .groupBy("eval_set", "label")
    .agg(f.count("*").alias("instances"))
    .orderBy("eval_set", "label")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## DLとpetastormのためのDeltaテーブル
# MAGIC
# MAGIC 我々のデータは現在Deltaテーブルに格納されており、Sparkデータフレームとして利用することができます。しかし、pytorchでは特定のデータタイプが期待されます。
# MAGIC
# MAGIC これを解決するために、テーブルからデータをモデルに自動で送信するように、PetastormとSparkコンバーターを活用します。このコンバーターは高速処理のために、ローカルキャッシュを用いてデータをインクリメンタルにロードします。詳細は[関連ドキュメント](https://docs.databricks.com/applications/machine-learning/load-data/petastorm.html)をご覧ください。

# COMMAND ----------

try:
    dbutils.fs.rm(petastorm_path, True)
except:
    pass

# COMMAND ----------

# petastoreキャッシュの宛先を設定
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, petastorm_path)

# データセットのおおよそのバイト数を特定
bytes_in_train = (
    images_train.withColumn("bytes", f.lit(4) + f.length("content"))
    .groupBy()
    .agg(f.sum("bytes").alias("bytes"))
    .collect()[0]["bytes"]
)
bytes_in_test = (
    images_test.withColumn("bytes", f.lit(4) + f.length("content"))
    .groupBy()
    .agg(f.sum("bytes").alias("bytes"))
    .collect()[0]["bytes"]
)

# 画像データのキャッシュ
converter_train = make_spark_converter(
    images_train,
    parquet_row_group_size_bytes=int(bytes_in_train / sc.defaultParallelism),
)
converter_test = make_spark_converter(
    images_test, parquet_row_group_size_bytes=int(bytes_in_test / sc.defaultParallelism)
)

# COMMAND ----------

NUM_CLASSES = 2  # ラベルは2クラス (0 あるいは 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Torchvision
# MAGIC
# MAGIC Torchvisionは再利用できる事前学習済みモデルを提供します。

# COMMAND ----------

from torchvision.models import (
    ViT_B_16_Weights,
    vit_b_16,
)


def get_model():
    # 事前学習済みモデルにアクセス
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # 転送学習のために新たな分類レイヤーを追加
    num_ftrs = model.heads.head.in_features

    # 新たに構成されたモジュールのパラメーターでは、デフォルトで requires_grad=True が設定されています
    model.heads.head = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    return model, weights

# COMMAND ----------

model, weights = get_model()
transforms = weights.transforms()
print(model.heads)
print(transforms)

# COMMAND ----------

# 画像変換のロジックを定義
def transform_row(is_train, batch_pd):

    # 画像にパイプラインを適用
    batch_pd["features"] = batch_pd["content"].map(
        lambda x: np.ascontiguousarray(
            transforms(Image.open(io.BytesIO(x)).convert("RGB")).numpy()
        )
    )

    # ラベルの変換 (我々の評価メトリックは値が float32 であることを期待します)
    # -----------------------------------------------------------
    batch_pd["label"] = batch_pd["label"].astype("float32")
    # -----------------------------------------------------------

    return batch_pd[["features", "label"]]


# 変換の仕様を取得する関数の定義
def get_transform_spec(is_train=True):

    spec = TransformSpec(
        partial(transform_row, is_train),  # 行を取得/変換するために呼び出す関数
        edit_fields=[  # 関数によって返却される行のスキーマ
            ("features", np.float32, (3, 224, 224), False),
            ("label", np.float32, (), False),
        ],
        selected_fields=["features", "label"],  # モデルに送信するスキーマのフィールド
    )

    return spec

# COMMAND ----------

# petastormのキャッシュにアクセスし、仕様を用いてデータを変換
with converter_train.make_torch_dataloader(
    transform_spec=get_transform_spec(is_train=True), batch_size=1
) as train_dataloader:

    # キャッシュからレコードを取得
    for i in iter(train_dataloader):
        print(i)
        break

# COMMAND ----------

BATCH_SIZE = 32  # 一度に32画像を処理
NUM_EPOCHS = 15  # すべての画像を5回繰り返し

# COMMAND ----------

from sklearn.metrics import f1_score


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    train_dataloader_iter,
    steps_per_epoch,
    epoch,
    device,
):

    model.train()  # モデルをトレーニングモードに設定

    # 統計情報
    running_loss = 0.0
    running_corrects = 0
    running_size = 0

    # 1つのエポックのデータに対するイテレーション
    for step in range(steps_per_epoch):

        # petastormから次のバッチを取得
        pd_batch = next(train_dataloader_iter)

        # 入力の特徴量とラベルを分離
        inputs, labels = pd_batch["features"].to(device), pd_batch["label"].to(device)

        # トレーニングの履歴を追跡
        with torch.set_grad_enabled(True):

            # パラメーターの勾配をゼロに
            optimizer.zero_grad()

            # フォワード
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=0)[:, 1]
            loss = criterion(probs, labels)

            # バックワード + 最適化
            loss.backward()
            optimizer.step()

        # 統計情報
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        running_size += inputs.size(0)

    scheduler.step()

    epoch_loss = running_loss / steps_per_epoch
    epoch_acc = running_corrects.double() / running_size

    print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(
    model, criterion, test_dataloader_iter, test_steps, device, metric_agg_fn=None
):

    model.eval()  # モデルを評価モードに設定

    # 統計情報
    running_loss = 0.0
    running_corrects = 0
    running_size = 0
    f1_scores = 0

    # すべての検証データに対してイテレーション
    for step in range(test_steps):

        pd_batch = next(test_dataloader_iter)
        inputs, labels = pd_batch["features"].to(device), pd_batch["label"].to(device)

        # メモリーを節約するために評価の際には履歴を追跡しない
        with torch.set_grad_enabled(False):

            # フォワード
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            loss = criterion(probs, labels)

        # 統計情報
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        running_size += inputs.size(0)
        f1_scores += f1_score(labels.cpu().data, preds.cpu())

    # それぞれのミニバッチにおける結果におけるロスを平均
    epoch_loss = running_loss / test_steps
    epoc_f1 = f1_scores / test_steps
    epoch_acc = running_corrects.double() / running_size

    # すべえのワーカーのメトリクスを集計するために分散トレーニングで metric_agg_fn を使用
    if metric_agg_fn is not None:
        epoch_loss = metric_agg_fn(epoch_loss, "avg_loss")
        epoch_acc = metric_agg_fn(epoch_acc, "avg_acc")
        epoc_f1 = metric_agg_fn(epoc_f1, "avg_f1")

    print(
        "Testing Loss: {:.4f} Acc: {:.4f} F1: {:.4f}".format(
            epoch_loss, epoch_acc, epoc_f1
        )
    )
    return epoch_loss, epoch_acc, epoc_f1

# COMMAND ----------

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def train_and_evaluate(lr=0.001):

    # 計算処理でGPUを使えるかどうかをチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの取得
    model, _ = get_model()

    # 特定されたプロセッサーデバイスのプロセスにモデルを割り当て
    model = model.to(device)

    # バイナリークロスエントロピーに最適化
    criterion = torch.nn.BCELoss()

    # 最終レイヤーのパラメーターのみを最適化
    filtered_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(filtered_params, lr=lr)

    # 7エポックごとに0.1の因数でLRを減衰
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    # petastormキャッシュのデータにアクセス
    with converter_train.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=True), batch_size=BATCH_SIZE
    ) as train_dataloader, converter_test.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=False), batch_size=BATCH_SIZE
    ) as val_dataloader:

        # データアクセスのためのイテレータと必要なサイクル数を定義
        train_dataloader_iter = iter(train_dataloader)
        steps_per_epoch = len(converter_train) // BATCH_SIZE

        val_dataloader_iter = iter(val_dataloader)
        validation_steps = max(1, len(converter_test) // BATCH_SIZE)

        # それぞれのエポックに対して
        for epoch in range(NUM_EPOCHS):

            print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("-" * 10)

            # トレーニング
            train_loss, train_acc = train_one_epoch(
                model,
                criterion,
                optimizer,
                exp_lr_scheduler,
                train_dataloader_iter,
                steps_per_epoch,
                epoch,
                device,
            )
            # 評価
            val_loss, val_acc, val_f1 = evaluate(
                model, criterion, val_dataloader_iter, validation_steps, device
            )

    # accで問題のあるタイプを訂正
    if type(val_acc) == torch.Tensor:
        val_acc = val_acc.item()

    return model, val_loss, val_acc, val_f1  # テンソルから値を抽出


# model, loss, acc, f1 = train_and_evaluate(**{'lr':0.00001})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperoptによるハイパーパラメータチューニング
# MAGIC
# MAGIC モデルの準備ができました。このようなモデルのチューニングは複雑なものです。アーキテクチャ、エンコーダー、学習率のようなハイパーパラメーターに対する選択肢が存在します。
# MAGIC
# MAGIC 我々のためにベストなハイパーパラメーターを探してもらえるように、Hyperoptを活用しましょう。Hyperoptは分散処理でも動作し、トレーニングプロセスをスピードアップするために、複数インスタンスで並列にトレーニングを実行できることにも注意してください。

# COMMAND ----------

# ハイパーパラメータの探索空間の定義
search_space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1.2e-5)),
}


# hyperoptが期待する形式の結果を返却するトレーニング関数の定義
def train_fn(params):

    # 指定されたハイパーパラメーターの設定を用いてモデルをトレーニング
    model, loss, acc, f1 = train_and_evaluate(**params)

    # 透明性を確保するためにこのイテレーションをmlflowに記録
    mlflow.log_metric("accuracy", acc)

    mlflow.log_metric("f1", f1)

    mlflow.pytorch.log_model(model, "model")
    # このイテレーションの結果を返却
    return {"loss": loss, "status": STATUS_OK}


# 適用する並列度を決定
if torch.cuda.is_available():  # GPUの場合
    nbrWorkers = sc.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers")
    if nbrWorkers is None:  # gcp
        nbrWorkers = sc.getConf().get(
            "spark.databricks.clusterUsageTags.clusterTargetWorkers"
        )
    parallelism = int(nbrWorkers)
    if parallelism == 0:  # シングルノードのクラスター
        parallelism = 1
else:  # CPUの場合
    parallelism = sc.defaultParallelism

# 分散ハイパーパラメーターチューニングの実行
with mlflow.start_run(run_name=model_name) as run:

    argmin = fmin(
        fn=train_fn,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,  # ハイパーパラメーターランの合計数 (通常この値はもっと大きなものにします)
        trials=SparkTrials(parallelism=parallelism),
    )  # 並列に実行されるハイパーパラメーターランの数

# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC ### Horovodによる分散ディープラーニング
# MAGIC
# MAGIC より多くのエポックでモデルをトレーニングできるようになりました。ランを加速させるには、Sparkクラスターの複数ノードでトレーニングを分散させることができます。
# MAGIC
# MAGIC 詳細は[Horovod](https://docs.databricks.com/machine-learning/train-model/distributed-training/horovod-runner.html)のドキュメントをご覧ください。

# COMMAND ----------

# モデル評価関数の定義
def metric_average_hvd(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


# 分散トレーニング & 評価の関数の定義
def train_and_evaluate_hvd(lr=0.001):

    # Step 1: Horovodの初期化
    hvd.init()

    # Step 2: 特定のCPUコア、あるいはGPUにHorovodプロセスを割り当て

    # トレーニングに使用するデバイスの特定
    if torch.cuda.is_available():  # gpu
        torch.cuda.set_device(hvd.local_rank())
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")  # cpu

    # モデルの取得及びデバイスへの割り当て
    model, _ = get_model()
    model = model.to(device)
    criterion = torch.nn.BCELoss()

    # Step 3: Horovodプロセスの数に基づいて学習率をスケール
    filtered_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(filtered_params, lr=lr * hvd.size())

    # Step 4: 分散処理のためにオプティマイザをラッピング
    optimizer_hvd = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_hvd, step_size=7, gamma=0.1
    )

    # Step 5: Horovodプロセスに関連づけられる状態変数の初期化
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # petastormキャッシュへのアクセスを解放
    with converter_train.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=True),
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        batch_size=BATCH_SIZE,
    ) as train_dataloader, converter_test.make_torch_dataloader(
        transform_spec=get_transform_spec(is_train=False),
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        batch_size=BATCH_SIZE,
    ) as test_dataloader:

        # それぞれのコア/GPUがバッチを処理します
        train_dataloader_iter = iter(train_dataloader)
        train_steps = len(converter_train) // (BATCH_SIZE * hvd.size())
        test_dataloader_iter = iter(test_dataloader)
        test_steps = max(1, len(converter_test) // (BATCH_SIZE * hvd.size()))

        # データセットに対するイテレーション
        for epoch in range(NUM_EPOCHS):

            # エポック情報の表示
            print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("-" * 10)

            # モデルのトレーニング
            train_loss, train_acc = train_one_epoch(
                model,
                criterion,
                optimizer_hvd,
                exp_lr_scheduler,
                train_dataloader_iter,
                train_steps,
                epoch,
                device,
            )

            # モデルの評価
            test_loss, test_acc, f1_acc = evaluate(
                model,
                criterion,
                test_dataloader_iter,
                test_steps,
                device,
                metric_agg_fn=metric_average_hvd,
            )

    return test_loss, test_acc, f1_acc, model

# COMMAND ----------

# horovodで利用できる並列度の特定
if torch.cuda.is_available():  # gpuの場合
    nbrWorkers = sc.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers")
    if nbrWorkers is None:  # gcp
        nbrWorkers = sc.getConf().get(
            "spark.databricks.clusterUsageTags.clusterTargetWorkers"
        )
    parallelism = int(nbrWorkers)
    if parallelism == 0:  # シングルノードのクラスター
        parallelism = 1
else:
    parallelism = 2  # 小規模なデータでは並列度を2と低く設定。それ以外の場合には、sc.defaultParallelismに設定することも可能

# horovodの実行環境の初期化
hr = HorovodRunner(np=parallelism)

# 分散トレーニングの実行
with mlflow.start_run(run_name=model_name) as run:

    # モデルのトレーニングと評価
    loss, acc, f1, model = hr.run(
        train_and_evaluate_hvd, **argmin
    )  # argminにはチューニングされたハイパーパラメーターが含まれます

    # mlflowにモデルを記録
    mlflow.log_params(argmin)
    mlflow.log_metrics({"loss": loss, "accuracy": acc, "f1": f1})
    mlflow.pytorch.log_model(model, "model")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## プロダクション環境にモデルをデプロイ
# MAGIC
# MAGIC モデルのトレーニングが完了しました。やらなくてはいけないことは、(`f1`メトリックに基づき)ベストなモデルを入手し、MLflowのレジストリにデプロイするということです。
# MAGIC
# MAGIC UI、あるいはいくつかのAPI呼び出しでこれを行うことができます:

# COMMAND ----------

# レジストリからベストモデルを取得
best_model = mlflow.search_runs(
    filter_string=f'attributes.status = "FINISHED"',
    order_by=["metrics.f1 DESC"],
    max_results=1,
).iloc[0]
model_registered = mlflow.register_model(
    "runs:/" + best_model.run_id + "/model", model_name
)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
print("registering model version " + model_registered.version + " as production model")
client.transition_model_version_stage(
    name=model_name,
    version=model_registered.version,
    stage="Production",
    archive_existing_versions=True,
)

# COMMAND ----------

try:
    dbutils.fs.rm(petastorm_path, True)
except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 我々のモデルはデプロイされ、プロダクションでの利用が可能とフラグが立てられました！
# MAGIC
# MAGIC モデルレジストリにモデルをデプロイしました。これによって、モデルのガバナンスが提供され、後段でのすべてのパイプライン開発をシンプルにし、加速させます。
# MAGIC
# MAGIC このモデルは、すべてのデータパイプライン(DLT、バッチ、あるいはDatabricksモデルサービングによるリアルタイム)で活用できるようになりました。
# MAGIC
# MAGIC それでは、大規模な[推論の実行]($./02_PredictionPCB)でこのモデルを活用しましょう。 
