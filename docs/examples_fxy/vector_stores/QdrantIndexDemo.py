#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/QdrantIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Qdrant Vector Store

# #### Creating a Qdrant client

import logging
import sys
import os

import qdrant_client
from IPython.#display import Markdown, #display
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

# If running this for the first, time, install using this command: 
# 
# ```
# !pip install -U qdrant_client
# ```

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load the documents

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# #### Build the VectorStoreIndex

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    location=":memory:"
    # otherwise set Qdrant instance address with:
    # uri="http://<host>:<port>"
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

service_context = ServiceContext.from_defaults()
vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
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
    "What did the author do after his time at Viaweb?"
)

#display(Markdown(f"<b>{response}</b>"))

# #### Build the VectorStoreIndex asynchronously

# To connect to the same event-loop,
# allows async events to run on notebook

import nest_asyncio

nest_asyncio.apply()

client = qdrant_client.QdrantClient(
    # location=":memory:"
    # Async upsertion does not work
    # on 'memory' location and requires
    # Qdrant to be deployed somewhere.
    url="http://localhost:6334",
    prefer_grpc=True,
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

service_context = ServiceContext.from_defaults()
vector_store = QdrantVectorStore(
    client=client, collection_name="paul_graham", prefer_grpc=True
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    use_async=True,
)

# #### Async Query Index

query_engine = index.as_query_engine(use_async=True)
response = await query_engine.aquery("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(use_async=True)
response = await query_engine.aquery(
    "What did the author do after his time at Viaweb?"
)

#display(Markdown(f"<b>{response}</b>"))

