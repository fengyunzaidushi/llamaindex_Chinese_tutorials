#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/DeepLakeIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DeepLake Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import textwrap

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import DeepLakeVectorStore

os.environ["OPENAI_API_KEY"] = "sk-********************************"
os.environ["ACTIVELOOP_TOKEN"] = "********************************"

#('pip install deeplake')

# if you don't export token in your environment alternativalay you can use deeplake CLI to loging to deeplake

# !activeloop login -t <TOKEN>

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

# dataset_path = "hub://adilkhan/paul_graham_essay" # if we comment this out and don't pass the path then GPTDeepLakeIndex will create dataset in memory
from llama_index.storage.storage_context import StorageContext

dataset_path = "paul_graham_essay"

# Create an index over the documnts
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# if we decide to not pass the path then GPTDeepLakeIndex will create dataset locally called llama_index

# Create an index over the documnts
# vector_store = DeepLakeVectorStore(overwrite=True)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author learn?",
)

print(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")

print(textwrap.fill(str(response), 100))

query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

# ## Deleting items from the database

import deeplake as dp

ds = dp.load("paul_graham_essay")

idx = ds.ids[0].numpy().tolist()

index.delete(idx[0])

