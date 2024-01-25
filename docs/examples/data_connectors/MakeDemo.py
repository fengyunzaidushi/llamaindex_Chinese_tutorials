#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/data_connectors/MakeDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Make Reader
# 
# We show how LlamaIndex can fit with your Make.com workflow by sending the GPT Index response to a scenario webhook.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers import MakeWrapper

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents=documents)

# set Logging to DEBUG for more detailed outputs
# query index
query_str = "What did the author do growing up?"
query_engine = index.as_query_engine()
response = query_engine.query(query_str)

# Send response to Make.com webhook
wrapper = MakeWrapper()
wrapper.pass_response_to_webhook(
    "<webhook_url>,
    response,
    query_str
)

