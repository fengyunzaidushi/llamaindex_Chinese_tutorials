#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/DocArrayHnswIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DocArray Hnsw Vector Store
# 
# [DocArrayHnswVectorStore](https://docs.docarray.org/user_guide/storing/index_hnswlib/) is a lightweight Document Index implementation provided by [DocArray](https://github.com/docarray/docarray) that runs fully locally and is best suited for small- to medium-sized datasets. It stores vectors on disk in hnswlib, and stores all other data in SQLite.
# 
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import sys
import logging
import textwrap

import warnings

warnings.filterwarnings("ignore")

# stop h|uggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import DocArrayHnswVectorStore
from IPython.#display import Markdown, #display

import os

os.environ["OPENAI_API_KEY"] = "<your openai key>"

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

# #

from llama_index.storage.storage_context import StorageContext

vector_store = DocArrayHnswVectorStore(work_dir="hnsw_index")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ## Querying

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

# ## Querying with filters

from llama_index.schema import TextNode

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

from llama_index.storage.storage_context import StorageContext

vector_store = DocArrayHnswVectorStore(work_dir="hnsw_filters")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = GPTVectorStoreIndex(nodes, storage_context=storage_context)

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

# remove created indices
import os, shutil

hnsw_dirs = ["hnsw_filters", "hnsw_index"]
for dir in hnsw_dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)

