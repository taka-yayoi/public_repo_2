# Databricks notebook source
# MAGIC %md
# MAGIC ## Streamlit経由でのDriver Proxyへの接続

# COMMAND ----------

# MAGIC %pip install streamlit watchdog
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context

def front_url(port):
    """
    フロントエンドを実行するための URL を返す

    Returns
    -------
    proxy_url : str
        フロントエンドのURL
    """
    ctx = get_context()
    proxy_url = f"https://{ctx.browserHostName}/driver-proxy/o/{ctx.workspaceId}/{ctx.clusterId}/{port}/"

    return proxy_url

PORT = 1501

# Driver ProxyのURLを表示
print(front_url(PORT))

# 利便性のためにリンクをHTML表示
displayHTML(f"<a href='{front_url(PORT)}' target='_blank' rel='noopener noreferrer'>別ウインドウで開く</a>")

# COMMAND ----------

streamlit_file = "/Workspace/Users/takaaki.yayoi@databricks.com/20230908_llmbot_w_driver_proxy/client_streamlit.py"

!streamlit run {streamlit_file} --server.port {PORT}

# COMMAND ----------


