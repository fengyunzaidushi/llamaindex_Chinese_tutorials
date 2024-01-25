#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/AwadbDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Awadb Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Creating an Awadb index

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Load documents, build the VectorStoreIndex

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from IPython.#display import Markdown, #display
import openai

openai.api_key = ""

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load Data

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

from llama_index import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import AwaDBVectorStore

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

vector_store = AwaDBVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)

#display(Markdown(f"<b>{response}</b>"))

