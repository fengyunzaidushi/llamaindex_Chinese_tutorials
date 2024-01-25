#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/LanceDBIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LanceDB Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SimpleDirectoryReader, Document, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import LanceDBVectorStore
import textwrap

# ### Setup OpenAI
# The first step is to configure the openai key. It will be used to created embeddings for the documents loaded into the index

import openai

openai.api_key = ""

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print("Document ID:", documents[0].doc_id, "Document Hash:", documents[0].hash)

# ### Create the index
# Here we create an index backed by LanceDB using the documents loaded previously. LanceDBVectorStore takes a few arguments.
# - uri (str, required): Location where LanceDB will store its files.
# - table_name (str, optional): The table name where the embeddings will be stored. Defaults to "vectors".
# - nprobes (int, optional): The number of probes used. A higher number makes search more accurate but also slower. Defaults to 20.
# - refine_factor: (int, optional): Refine the results by reading extra elements and re-ranking them in memory. Defaults to None
# 
# - More details can be found at the [LanceDB docs](https://lancedb.github.io/lancedb/ann_indexes)

vector_store = LanceDBVectorStore(uri="/tmp/lancedb")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the index
# We can now ask questions using our index.

query_engine = index.as_query_engine()
response = query_engine.query("How much did Viaweb charge per month?")

print(textwrap.fill(str(response), 100))

response = query_engine.query("What did the author do growing up?")

print(textwrap.fill(str(response), 100))

# ### Appending data
# You can also add data to an existing index

del index

index = VectorStoreIndex.from_documents(
    [Document(text="The sky is purple in Portland, Maine")],
    uri="/tmp/new_dataset",
)

query_engine = index.as_query_engine()
response = query_engine.query("Where is the sky purple?")
print(textwrap.fill(str(response), 100))

index = VectorStoreIndex.from_documents(documents, uri="/tmp/new_dataset")

query_engine = index.as_query_engine()
response = query_engine.query("What companies did the author start?")
print(textwrap.fill(str(response), 100))

