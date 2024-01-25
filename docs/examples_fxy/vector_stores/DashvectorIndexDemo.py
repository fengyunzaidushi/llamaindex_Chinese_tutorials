#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/DashvectorIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DashVector Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Creating a DashVector Collection

import dashvector

api_key = os.environ["DASHVECTOR_API_KEY"]
client = dashvector.Client(api_key=api_key)

# dimensions are for text-embedding-ada-002
client.create("llama-demo", dimension=1536)

dashvector_collection = client.get("quickstart")

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load documents, build the DashVectorStore and VectorStoreIndex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import DashVectorStore
from IPython.#display import Markdown, #display

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize without metadata filter
from llama_index.storage.storage_context import StorageContext

vector_store = DashVectorStore(dashvector_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

