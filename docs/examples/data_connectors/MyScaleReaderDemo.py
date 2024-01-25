#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/MyScaleReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # MyScale Reader

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import clickhouse_connect

host = "YOUR_CLUSTER_HOST"
username = "YOUR_USERNAME"
password = "YOUR_CLUSTER_PASSWORD"
client = clickhouse_connect.get_client(
    host=host, port=8443, username=username, password=password
)

import random
from llama_index.readers.myscale import MyScaleReader

reader = MyScaleReader(myscale_host=host, username=username, password=password)
reader.load_data([random.random() for _ in range(1536)])

reader.load_data(
    [random.random() for _ in range(1536)],
    where_str="extra_info._dummy=0",
    limit=3,
)

