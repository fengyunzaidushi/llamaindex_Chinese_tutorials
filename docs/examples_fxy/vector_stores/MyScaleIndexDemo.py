#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/MyScaleIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # MyScale Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Creating a MyScale Client

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from os import environ
import clickhouse_connect

environ["OPENAI_API_KEY"] = "sk-*"

# initialize client
client = clickhouse_connect.get_client(
    host="YOUR_CLUSTER_HOST",
    port=8443,
    username="YOUR_USERNAME",
    password="YOUR_CLUSTER_PASSWORD",
)

# #### Load documents, build and store the VectorStoreIndex with MyScaleVectorStore
# 
# Here we will use a set of Paul Graham essays to provide the text to turn into embeddings, store in a ``MyScaleVectorStore`` and query to find context for our LLM QnA loop.

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import MyScaleVectorStore
from IPython.#display import Markdown, #display

# load documents
documents = SimpleDirectoryReader("../data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)
print("Number of Documents: ", len(documents))

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# You can process your files individually using [SimpleDirectoryReader](/examples/data_connectors/simple_directory_reader.ipynb):

loader = SimpleDirectoryReader("./data/paul_graham/")
documents = loader.load_data()
for file in loader.input_files:
    print(file)
    # Here is where you would do any preprocessing

# initialize with metadata filter and store indexes
from llama_index.storage.storage_context import StorageContext

for document in documents:
    document.metadata = {"user_id": "123", "favorite_color": "blue"}
vector_store = MyScaleVectorStore(myscale_client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# #### Query Index
# 
# Now MyScale vector store supports filter search and hybrid search
# 
# You can learn more about [query_engine](/module_guides/deploying/query_engine/root.md) and [retriever](/module_guides/querying/retriever/root.md).

import textwrap

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="user_id", value="123"),
        ]
    ),
    similarity_top_k=2,
    vector_store_query_mode="hybrid",
)
response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

# #### Clear All Indexes

for document in documents:
    index.delete_ref_doc(document.doc_id)

