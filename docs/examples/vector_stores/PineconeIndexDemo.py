#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/PineconeIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Pinecone Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Creating a Pinecone Index

import pinecone

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="eu-west1-gcp")

# dimensions are for text-embedding-ada-002
pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)

pinecone_index = pinecone.Index("quickstart")

# #### Load documents, build the PineconeVectorStore and VectorStoreIndex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
from IPython.#display import Markdown, #display

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize without metadata filter
from llama_index.storage.storage_context import StorageContext

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

