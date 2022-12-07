# Databricks notebook source
# MAGIC %md # MLflowクイックスタート: トラッキング
# MAGIC 
# MAGIC このノートブックでは、シンプルなデータセットに対してランダムフォレストモデルを作成し、モデル、選択されたパラメーター、評価メトリクス、その他のアーティファクトを記録するためにMlflowトラッキングAPIを使用します。
# MAGIC 
# MAGIC ## 要件
# MAGIC このノートブックでは、Databricksランタイム6.4以降、あるいはDatabricks機械学習ランタイム6.4以降が必要です。

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ライブラリのインストール
# MAGIC 
# MAGIC このノートブックでは分散処理を行いませんので、ドライバーノードのみにパッケージをインストールするために、Rの`install.packages()`関数を使うことができます。
# MAGIC 
# MAGIC 分散処理の利点を活用するには、クラスターライブラリを作成することで、クラスターの全ノードにパッケージをインストールする必要があります。"Install a library on a cluster" ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster)|[GCP](https://docs.gcp.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster))をご覧ください。

# COMMAND ----------

install.packages("mlflow")

# COMMAND ----------

# MAGIC %md
# MAGIC このノートブックは、MLflow 2.0ワークフローとして実行できますが、MLflow 1.xを使用している場合には、適切なMLflowパッケージがインストールされる様に、`install_mlflow()`コマンドのコメントを解除してください。

# COMMAND ----------

library(mlflow)
# install_mlflow() 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ライブラリのインポート
# MAGIC 
# MAGIC 必要なライブラリをインポートします。
# MAGIC 
# MAGIC このノートブックでは、あとでモデルをロードし直せる様に、トレーニングしたモデルのpredictメソッドをシリアライズするために、Rライブラリの`carrier`を使用します。詳細に関しては、[`carrier` github repo](https://github.com/r-lib/carrier)をご覧ください。

# COMMAND ----------

install.packages("carrier")
install.packages("e1071")

library(MASS)
library(caret)
library(e1071)
library(randomForest)
library(SparkR)
library(carrier)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングおよびトラッキング

# COMMAND ----------

with(mlflow_start_run(), {
  
  # モデルパラメーターの設定
  ntree <- 100
  mtry <- 3
  
  # モデルの作成およびトレーニング
  rf <- randomForest(type ~ ., data=Pima.tr, ntree=ntree, mtry=mtry)
  
  # テストデータセットに対する予測にモデルを使用
  pred <- predict(rf, newdata=Pima.te[,1:7])
  
  # このランで使用されたモデルパラメーターを記録
  mlflow_log_param("ntree", ntree)
  mlflow_log_param("mtry", mtry)
  
  # モデルを評価するためのメトリクスの定義
  cm <- confusionMatrix(pred, reference = Pima.te[,8])
  sensitivity <- cm[["byClass"]]["Sensitivity"]
  specificity <- cm[["byClass"]]["Specificity"]
  
  # メトリクスの値を記録 
  mlflow_log_metric("sensitivity", sensitivity)
  mlflow_log_metric("specificity", specificity)
  
  # モデルの記録
  # 関数としてモデルを格納するRパッケージ "carrier" の crate() 関数  
  predictor <- crate(function(x) predict(rf,.x))
  mlflow_log_model(predictor, "model")     
  
  # コンフュージョンマトリクス(混同行列)の作成およびプロット
  png(filename="confusion_matrix_plot.png")
  barplot(as.matrix(cm), main="Results",
         xlab="Observed", ylim=c(0,200), col=c("green","blue"),
         legend=rownames(cm), beside=TRUE)
  dev.off()
  
  # アーティファクトとしてプロットを保存
  mlflow_log_artifact("confusion_matrix_plot.png") 
})

# COMMAND ----------

# MAGIC %md 
# MAGIC ## トラッキング結果の確認
# MAGIC 
# MAGIC 結果を参照するには、このページの右上にあるフラスコアイコンをクリックします。エクスペリメントのサイドバーが表示されます。エクスペリメントのサイドバーには、このノートブックにおけるそれぞれのランのパラメーターとメトリクスが表示されます。最新のランを表示する様にするには円形の矢印アイコンをクリックします。
# MAGIC 
# MAGIC ランダムに割り当てられたランの名称(`skillfull-bat-xxx`など)をクリックすると、ランのページが新規タブに表示されます。このページでは、ランとして記録されたすべての情報を確認することができます。記録されたモデルやプロットを参照するために、アーティファクトセクションまで下にスクロールします。
# MAGIC 
# MAGIC 詳細に関しては、"View notebook experiment" ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment))をご覧ください。
