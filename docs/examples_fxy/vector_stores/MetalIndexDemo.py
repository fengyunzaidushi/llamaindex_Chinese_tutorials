#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/MetalIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Metal Vector Store

# ## Creating a Metal Vector Store

# 1. Register an account for [Metal](https://app.getmetal.io/)
# 2. Generate an API key in [Metal's Settings](https://app.getmetal.io/settings/organization). Save the `api_key` + `client_id`
# 3. Generate an Index in [Metal's Dashboard](https://app.getmetal.io/). Save the `index_id`

# ## Load data into your Index

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import MetalVectorStore
from IPython.#display import Markdown, #display

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# initialize Metal Vector Store
from llama_index.storage.storage_context import StorageContext

api_key = "api key"
client_id = "client id"
index_id = "index id"

vector_store = MetalVectorStore(
    api_key=api_key,
    client_id=client_id,
    index_id=index_id,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ## Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

