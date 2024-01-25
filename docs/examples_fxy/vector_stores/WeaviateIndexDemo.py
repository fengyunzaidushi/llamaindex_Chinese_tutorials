#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/WeaviateIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Weaviate Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Creating a Weaviate Client

import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
openai.api_key = os.environ["OPENAI_API_KEY"]

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import weaviate

# cloud
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://llama-test-ezjahb4m.weaviate.network",
    auth_client_secret=resource_owner_config,
)

# local
# client = weaviate.Client("http://localhost:8080")

# #### Load documents, build the VectorStoreIndex

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import WeaviateVectorStore
from IPython.#display import Markdown, #display

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

from llama_index.storage.storage_context import StorageContext

# If you want to load the index later, be sure to give it a name!
vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# NOTE: you may also choose to define a index_name manually.
# index_name = "test_prefix"
# vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# ## Loading the index
# 
# Here, we use the same index name as when we created the initial index. This stops it from being auto-generated and allows us to easily connect back to it.

resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://llama-test-ezjahb4m.weaviate.network",
    auth_client_secret=resource_owner_config,
)

# local
# client = weaviate.Client("http://localhost:8080")

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)

loaded_index = VectorStoreIndex.from_vector_store(vector_store)

# set Logging to DEBUG for more detailed outputs
query_engine = loaded_index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
#display(Markdown(f"<b>{response}</b>"))

# ## Metadata Filtering
# 
# Let's insert a dummy document, and try to filter so that only that document is returned.

from llama_index import Document

doc = Document.example()
print(doc.metadata)
print("-----")
print(doc.text[:100])

loaded_index.insert(doc)

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="filename", value="README.md")]
)
query_engine = loaded_index.as_query_engine(filters=filters)
response = query_engine.query("What is the name of the file?")
#display(Markdown(f"<b>{response}</b>"))

