# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksノートブックにおけるエンドツーエンドの分散トレーニング
# MAGIC 
# MAGIC PyTorchにおける分散トレーニングは多くの場合、ファイル(`train.py`)を作成し、そのファイルを用いた分散トレーニングを実行するために`torchrun` CLIを使用します。Databricksでは、Databricksノートブック上で直接分散トレーニングを実行するメソッドを提供します。ノートブック内で`train()`関数を定義し、複数のワーカーでモデルをトレーニングするために`TorchDistributor` APIを使用することができます。
# MAGIC 
# MAGIC このノートブックでは、どのようにノートブック内でインタラクティブな開発を行うのかを説明します。特に大規模なディープラーニングプロジェクトにおいては、ご自身のコードを管理可能なチャンクに分割するために`%run`コマンドを活用することを推奨します。
# MAGIC 
# MAGIC このノートブックでは:
# MAGIC - 古典的なMNISTデータセットに対してシンプルな単一GPUモデルをトレーニングします。
# MAGIC - 分散トレーニングのコードに変換します。
# MAGIC - 複数GPUあるいは複数ノードにモデルのトレーニングをスケールアップするために、どのようにTorchDistributorを活用できるのかを学びます。
# MAGIC 
# MAGIC ## 要件
# MAGIC - Databricks Runtime ML 13.0以降
# MAGIC - (推奨) GPUインスタンス [AWS](https://docs.databricks.com/clusters/gpu.html) | [Azure](https://learn.microsoft.com/en-gb/azure/databricks/clusters/gpu) | [GCP](https://docs.gcp.databricks.com/clusters/gpu.html)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### MLflowのセットアップ
# MAGIC 
# MAGIC MLflowは機械学習エクスペリメントとモデルのロギングをサポートするツールです。
# MAGIC 
# MAGIC **注意** MLflow PyTorch Autologging APIはPyTorch Lightning向けに設計されており、ネイティブなPyTorchでは動作しません。

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
username

experiment_path = f'/Users/{username}/pytorch-distributor'

# これらは後で必要となります
db_host = "https://<databricks host name>/"  # 変更してください!
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# IDを確認し、スケールする際にワーカーノードに送信できるように、手動でエクスペリメントを作成します
experiment = mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md ## トレーニング、テスト関数の定義
# MAGIC 
# MAGIC 以下のセルには、モデルを記述するコード、トレーニング関数、テスト関数が含まれています。これらすべてはローカルで実行するようにデザインされています。次に、このコードにはローカル環境から分散環境でのトレーニングへの移行に必要な変更が導入されます。
# MAGIC 
# MAGIC すべてのtorchコードは標準的なPyTorch APIを活用しており、カスタムライブラリやコードの記述方法の変更は不要です。このノートブックは`TorchDistributor`を用いたトレーニングのスケール方法にフォーカスしているので、モデルコードの説明はしません。

# COMMAND ----------

import torch
NUM_WORKERS = 2
NUM_GPUS_PER_NODE = torch.cuda.device_count()

# COMMAND ----------

PYTORCH_DIR = '/dbfs/ml/pytorch'

batch_size = 100
num_epochs = 3
momentum = 0.5
log_interval = 100
learning_rate = 0.001

import torch
import torch.nn as nn
import torch.nn.functional as F

# モデルの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                100. * batch_idx / len(data_loader), loss.item()))
            
            mlflow.log_metric('train_loss', loss.item())

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)
  
def load_checkpoint(log_dir, epoch=num_epochs):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)

def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()))
  os.makedirs(log_dir)
  return log_dir

import torch.optim as optim
from torchvision import datasets, transforms
from time import time
import os

base_log_dir = create_log_dir()
print("Log directory:", base_log_dir)

def train(log_dir):
  device = torch.device('cuda')

  train_parameters = {'batch_size': batch_size, 'epochs': num_epochs}
  mlflow.log_params(train_parameters)
  
  train_dataset = datasets.MNIST(
    'data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Net().to(device)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, data_loader, optimizer, epoch)
    save_checkpoint(log_dir, model, optimizer, epoch)
    
def test(log_dir):
  device = torch.device('cuda')
  loaded_model = Net().to(device)
  scripted_model = torch.jit.script(loaded_model)
  
  checkpoint = load_checkpoint(log_dir)
  loaded_model.load_state_dict(checkpoint['model'])
  loaded_model.eval()

  test_dataset = datasets.MNIST(
    'data', 
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(test_dataset)

  test_loss = 0
  for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = loaded_model(data)
      test_loss += F.nll_loss(output, target)
        
  test_loss /= len(data_loader.dataset)
  print("Average test loss: {}".format(test_loss.item()))
  
  mlflow.log_metric('test_loss', test_loss.item())
  
  mlflow.pytorch.log_model(scripted_model, "model")
  

# COMMAND ----------

# MAGIC %md ### ローカルでモデルをトレーニング
# MAGIC 
# MAGIC これが適切に動作することをテストするために、上で定義した関数を用いてトレーニングとテストのイテレーションを起動することができます。

# COMMAND ----------

with mlflow.start_run():
  
  mlflow.log_param('run_type', 'local')
  train(base_log_dir)
  test(base_log_dir)
  

# COMMAND ----------

# MAGIC %md ## 分散セットアップ
# MAGIC 
# MAGIC シングルノードのコードを`train()`関数でラッピングする際、ライブラリのpickleに関する問題を避けるために、すべてのimport文を`train()`関数に含めることを推奨します。
# MAGIC 
# MAGIC 他の全ては、PyTorch内で分散トレーニングが動作するようにするために通常必要となるものです。
# MAGIC 
# MAGIC - `train()`の最初で`dist.init_process_group("nccl")`の呼び出し
# MAGIC - `train()`の最後で`dist.destroy_process_group()`の呼び出し
# MAGIC - `local_rank = int(os.environ["LOCAL_RANK"])`の設定
# MAGIC - `DataLoader`に`DistributedSampler`を追加
# MAGIC - `DDP(model)`でモデルをラッピング
# MAGIC - 詳細は https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html をご覧ください

# COMMAND ----------

single_node_single_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_single_gpu_dir)

def train_one_epoch(model, device, data_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(data_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(data_loader) * len(data),
          100. * batch_idx / len(data_loader), loss.item()))
      
      if int(os.environ["RANK"]) == 0:
        mlflow.log_metric('train_loss', loss.item())

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.module.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)

# 分散トレーニングでは、1つのmain関数にトレーニングステップとテストステップをマージします
def main_fn(directory):
  
  #### ここにimport文を追加 ####
  import mlflow
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.utils.data.distributed import DistributedSampler
  
  ############################

  ##### MLflowのセットアップ ####
  # 別々のプロセスがMLflowを見つけられるようにするためにこれが必要です
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  # エクスペリメントの詳細をここで設定します
  experiment = mlflow.set_experiment(experiment_path)
  ############################
  
  print("Running distributed training")
  dist.init_process_group("nccl")
  
  local_rank = int(os.environ["LOCAL_RANK"])
  global_rank = int(os.environ["RANK"])
  
  if global_rank == 0:
    train_parameters = {'batch_size': batch_size, 'epochs': num_epochs, 'trainer': 'TorchDistributor'}
    mlflow.log_params(train_parameters)
  
  train_dataset = datasets.MNIST(
    'data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  
  #### 分散データローダーの追加 ####
  train_sampler = DistributedSampler(dataset=train_dataset)
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  ######################################
  
  model = Net().to(local_rank)
  #### 分散モデルの追加 ####
  ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
  #################################

  optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(1, num_epochs + 1):
    train_one_epoch(ddp_model, local_rank, data_loader, optimizer, epoch)
    
    if global_rank == 0: 
      save_checkpoint(directory, ddp_model, optimizer, epoch)
  
  # テスト用にモデルを保存
  if global_rank == 0:
    mlflow.pytorch.log_model(ddp_model, "model")
    
    ddp_model.eval()
    test_dataset = datasets.MNIST(
      'data', 
      train=False,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    data_loader = torch.utils.data.DataLoader(test_dataset)    

    test_loss = 0
    for data, target in data_loader:
      device = torch.device('cuda')
      data, target = data.to(device), target.to(device)
      output = ddp_model(data)
      test_loss += F.nll_loss(output, target)
          
    test_loss /= len(data_loader.dataset)
    print("Average test loss: {}".format(test_loss.item()))
    
    mlflow.log_metric('test_loss', test_loss.item())

    
  dist.destroy_process_group()
  
  return "finished" # 任意のpickle可能なオブジェクトを返却できます

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### TorchDistributorなしのテスト
# MAGIC 
# MAGIC 以下では、シングルGPUでトレーニングを実行することでトレーニングループを検証します。

# COMMAND ----------

# すべてのプロセスが動作していることをクイックにテストするためのシングルノードにおける分散実行
with mlflow.start_run():
  mlflow.log_param('run_type', 'test_dist_code')
  main_fn(single_node_single_gpu_dir)
  

# COMMAND ----------

# MAGIC %md ### マルチGPUシングルノードのトレーニング
# MAGIC 
# MAGIC PyTorchでは、マルチGPUシングルノードでトレーニングを行うための[roundabout way](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)を提供しています。Databricksでは、マルチGPUシングルノードをマルチノードにシームレスに移行できるより効率的なソリューションを提供しています。DatabricksでマルチGPUシングルノードのトレーニングを行うには、`TorchDistributor` APIを呼び出し、使用したいドライバーノードで利用できるGPUの数を`num_processes`に設定し、`local_mode=True`を設定します。

# COMMAND ----------

single_node_multi_gpu_dir = create_log_dir()
print("Data is located at: ", single_node_multi_gpu_dir)

from pyspark.ml.torch.distributor import TorchDistributor

output = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True).run(main_fn, single_node_multi_gpu_dir)
test(single_node_multi_gpu_dir)

# COMMAND ----------

# MAGIC %md ### マルチノードのトレーニング
# MAGIC 
# MAGIC マルチGPUシングルノードからマルチノードのトレーニングに移行するには、すべてのワーカーノードで利用したいGPUの数に`num_processes`を変更するだけです。このサンプルではすべてのGPU(`NUM_GPUS_PER_NODE * NUM_WORKERS`)を使用しています。また、`local_mode`を`False`に設定します。さらに、トレーニング関数を実行するそれぞれのSparkタスクでいくつのGPUを使用するのかを設定するには、クラスターを作成する前にクラスターページにあるSpark設定で`set spark.task.resource.gpu.amount <num_gpus_per_task>`を設定します。

# COMMAND ----------

multi_node_dir = create_log_dir()
print("Data is located at: ", multi_node_dir)

output_dist = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True).run(main_fn, multi_node_dir)
test(multi_node_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
