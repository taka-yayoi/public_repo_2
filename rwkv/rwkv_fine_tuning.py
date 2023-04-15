# Databricks notebook source
# MAGIC %md
# MAGIC # DatabricksでRWKVのファインチューニング
# MAGIC 
# MAGIC - [Google Colab で RWKV を試す｜npaka｜note](https://note.com/npaka/n/n59882803a92e)
# MAGIC - [rwkv · PyPI](https://pypi.org/project/rwkv/)
# MAGIC - [Google Colab で RWKV のファインチューニングを試す｜npaka｜note](https://note.com/npaka/n/n8f3c2c491901)

# COMMAND ----------

# MAGIC %md
# MAGIC ## git-lfsのインストール
# MAGIC 
# MAGIC [\_pickle\.UnpicklingError: invalid load key, 'v' \- 西尾泰和のScrapbox](https://scrapbox.io/nishio/_pickle.UnpicklingError:_invalid_load_key,_'v')

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install git-lfs

# COMMAND ----------

# MAGIC %pip install transformers pytorch-lightning==1.7 deepspeed wandb ninja rwkv

# COMMAND ----------

# 作業用ディレクトリ
rwkv_dir = "/tmp/takaaki.yayoi@databricks.com/rwkv"
rwkv_dir_local = "/dbfs/tmp/takaaki.yayoi@databricks.com/rwkv"

# COMMAND ----------

# MAGIC %md
# MAGIC [akane\-talk/dataset\.txt at main · npaka3/akane\-talk · GitHub](https://github.com/npaka3/akane-talk/blob/main/docs/dataset.txt)

# COMMAND ----------

# データセットを移動
#dbutils.fs.mv("dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/dataset.txt", rwkv_dir)

# COMMAND ----------

display(dbutils.fs.ls(rwkv_dir))

# COMMAND ----------

# MAGIC %sh
# MAGIC git lfs clone https://github.com/blinkdl/RWKV-LM

# COMMAND ----------

# MAGIC %sh
# MAGIC ls RWKV-LM

# COMMAND ----------

# MAGIC %md
# MAGIC カレントディレクトリを`/databricks/driver/RWKV-LM/RWKV-v4`に変更しておきます。

# COMMAND ----------

import os
os.chdir("/databricks/driver/RWKV-LM/RWKV-v4")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベースモデル
# MAGIC 
# MAGIC オプション:
# MAGIC - RWKV-4-Pile-1B5
# MAGIC - RWKV-4-Pile-430M
# MAGIC - RWKV-4-Pile-169M

# COMMAND ----------

base_model_name = "RWKV-4-Pile-169M"
base_model_url = f"https://huggingface.co/BlinkDL/{base_model_name.lower()}"

print(base_model_url)

os.environ['base_model_url'] = base_model_url

# COMMAND ----------

# MAGIC %sh
# MAGIC # This may take a while
# MAGIC git clone $base_model_url

# COMMAND ----------

from glob import glob
base_model_path = glob(f"{base_model_name.lower()}/{base_model_name}*.pth")[0]

print(f"Using {base_model_path} as base")

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータ

# COMMAND ----------

import numpy as np
from transformers import PreTrainedTokenizerFast

# 事前にトークンファイルをダウンロードしてください
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{rwkv_dir_local}/20B_tokenizer.json')

input_file = f"{rwkv_dir_local}/dataset.txt"
output_file = 'train.npy'

print(f'Tokenizing {input_file} (VERY slow. please wait)')

data_raw = open(input_file, encoding="utf-8").read()
print(f'Raw length = {len(data_raw)}')

data_code = tokenizer.encode(data_raw)
print(f'Tokenized length = {len(data_code)}')

out = np.array(data_code, dtype='uint16')
np.save(output_file, out, allow_pickle=False)

# COMMAND ----------

# MAGIC %sh ls /databricks/driver/RWKV-LM/RWKV-v4

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング

# COMMAND ----------

tuned_model_name = "tuned"
output_path = "rwkv-v4-rnn-pile-tuning"

# COMMAND ----------

os.mkdir(output_path)

# COMMAND ----------

#@title Training Options { display-mode: "form" }
from shutil import copy

def training_options():
    EXPRESS_PILE_MODE = True
    EXPRESS_PILE_MODEL_NAME = base_model_path.split(".")[0]
    EXPRESS_PILE_MODEL_TYPE = base_model_name
    n_epoch = 100 #@param {type:"integer"}
    epoch_save_frequency = 25 #@param {type:"integer"}
    batch_size =  11#@param {type:"integer"} 
    ctx_len = 384 #@param {type:"integer"}
    epoch_save_path = f"{output_path}/{tuned_model_name}"
    return locals()

def model_options():
    T_MAX = 384 #@param {type:"integer"}
    return locals()

def env_vars():
    RWKV_FLOAT_MODE = 'fp16' #@param ['fp16', 'bf16', 'bf32'] {type:"string"}
    RWKV_DEEPSPEED = '1' #@param ['0', '1'] {type:"string"}
    return {f"os.environ['{key}']": value for key, value in locals().items()}

def replace_lines(file_name, to_replace):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(f'{file_name}.tmp', 'w') as f:
        for line in lines:
            key = line.split(" =")[0]
            if key.strip() in to_replace:
                value = to_replace[key.strip()]
                if isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                else:
                    f.write(f'{key} = {value}\n')
            else:
                f.write(line)
    copy(f'{file_name}.tmp', file_name)
    os.remove(f'{file_name}.tmp')

values = training_options()
values.update(env_vars())
replace_lines('train.py', values)
replace_lines('src/model.py', model_options())

# COMMAND ----------

# MAGIC %md
# MAGIC n_epoch過ぎても止まらないので、十分学習できたら自分で停止します。

# COMMAND ----------

# MAGIC %sh
# MAGIC python train.py

# COMMAND ----------

# MAGIC %sh
# MAGIC ls rwkv-v4-rnn-pile-tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ## 推論

# COMMAND ----------

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

# COMMAND ----------

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# モデルとパイプラインの準備
model = RWKV(
    model="/databricks/driver/RWKV-LM/RWKV-v4/rwkv-v4-rnn-pile-tuning/tuned1", 
    strategy="cuda fp16")
pipeline = PIPELINE(model, "/dbfs/tmp/takaaki.yayoi@databricks.com/rwkv/20B_tokenizer.json")

# COMMAND ----------

# パイプライン引数の準備
args = PIPELINE_ARGS(
    temperature = 1.0,
    top_p = 0.7, 
    top_k = 100, 
    alpha_frequency = 0.25, 
    alpha_presence = 0.25, 
    token_ban = [],
    token_stop = [0],
    chunk_len = 256) 

# COMMAND ----------

# Instructプロンプトの生成
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""

# COMMAND ----------

# プロンプトの準備
prompt = "日本で一番人気のアニメは？"
print(prompt)

# Instructプロンプトの生成
prompt = generate_prompt(prompt)
print("--[prompt]--\n" + prompt + "----")

# パイプラインの実行
result = pipeline.generate(prompt, token_count=200, args=args)
print(result)

# COMMAND ----------

# プロンプトの準備
prompt = "こんにちは！"
print(prompt)

# Instructプロンプトの生成
prompt = generate_prompt(prompt)
print("--[prompt]--\n" + prompt + "----")

# パイプラインの実行
result = pipeline.generate(prompt, token_count=200, args=args)
print(result)

# COMMAND ----------

# プロンプトの準備
prompt = "セガサターンほしいです"
print(prompt)

# Instructプロンプトの生成
prompt = generate_prompt(prompt)
print("--[prompt]--\n" + prompt + "----")

# パイプラインの実行
result = pipeline.generate(prompt, token_count=200, args=args)
print(result)

# COMMAND ----------


