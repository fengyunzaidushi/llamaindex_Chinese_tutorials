#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/ZepIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Zep Vector Store
# 
# ## A long-term memory store for LLM applications
# 
# This notebook demonstrates how to use the Zep Vector Store with LlamaIndex.
# 
# ## About Zep
# 
# Zep makes it easy for developers to add relevant documents, chat history memory & rich user data to their LLM app's prompts.
# 
# ## Note
# 
# Zep can automatically embed your documents. The LlamaIndex implementation of the Zep Vector Store utilizes LlamaIndex's embedders to do so.
# 
# ## Getting Started
# 
# **Quick Start Guide:** https://docs.getzep.com/deployment/quickstart/
# **GitHub:** https://github.com/getzep/zep
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# !pip install zep-python

import logging
import sys
from uuid import uuid4

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
import openai
from dotenv import load_dotenv

load_dotenv()

# os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.zep import ZepVectorStore

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("../data/paul_graham/").load_data()

# ## Create a Zep Vector Store and Index
# 
# You can use an existing Zep Collection, or create a new one.
# 

from llama_index.storage.storage_context import StorageContext

zep_api_url = "http://localhost:8000"
collection_name = f"graham{uuid4().hex}"

vector_store = ZepVectorStore(
    api_url=zep_api_url,
    collection_name=collection_name,
    embedding_dimensions=1536,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(str(response))

# ## Querying with Metadata filters
# 

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

collection_name = f"movies{uuid4().hex}"

vector_store = ZepVectorStore(
    api_url=zep_api_url,
    collection_name=collection_name,
    embedding_dimensions=1536,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("What is inception about?")

for r in result:
    print("\n", r.node)
    print("Score:", r.score)

