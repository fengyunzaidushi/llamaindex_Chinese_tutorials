#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/Lantern.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Lantern Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# import logging
# import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import LanternVectorStore
import textwrap
import openai

# ### Setup OpenAI
# The first step is to configure the openai key. It will be used to created embeddings for the documents loaded into the index

import os

os.environ["OPENAI_API_KEY"] = "<your key>"
openai.api_key = "<your key>"

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)

# ### Create the Database
# Using an existing postgres running at localhost, create the database we'll be using.

import psycopg2

connection_string = "postgresql://postgres:postgres@localhost:5432"
db_name = "vector_db"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext

# Setup global service context with embedding model
# So query strings will be transformed to embeddings and HNSW index will be used
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

from llama_index import set_global_service_context

set_global_service_context(service_context)

# ### Create the index
# Here we create an index backed by Postgres using the documents loaded previously. LanternVectorStore takes a few arguments.

from sqlalchemy import make_url

url = make_url(connection_string)
vector_store = LanternVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay",
    embed_dim=1536,  # openai embedding dimension
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

# ### Query the index
# We can now ask questions using our index.

response = query_engine.query("What did the author do?")

print(textwrap.fill(str(response), 100))

response = query_engine.query("What happened in the mid 1980s?")

print(textwrap.fill(str(response), 100))

# ### Querying existing index

vector_store = LanternVectorStore.from_params(
    database="vector_db",
    host="localhost",
    password="postgres",
    port=5432,
    user="postgres",
    table_name="paul_graham_essay",
    embed_dim=1536,  # openai embedding dimension
    m=16,  # HNSW M parameter
    ef_construction=128,  # HNSW ef construction parameter
    ef=64,  # HNSW ef search parameter
)

# Read more about HNSW parameters here: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do?")

print(textwrap.fill(str(response), 100))

# ### Hybrid Search  

# To enable hybrid search, you need to:
# 1. pass in `hybrid_search=True` when constructing the `LanternVectorStore` (and optionally configure `text_search_config` with the desired language)
# 2. pass in `vector_store_query_mode="hybrid"` when constructing the query engine (this config is passed to the retriever under the hood). You can also optionally set the `sparse_top_k` to configure how many results we should obtain from sparse text search (default is using the same value as `similarity_top_k`). 

from sqlalchemy import make_url

url = make_url(connection_string)
hybrid_vector_store = LanternVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay_hybrid_search",
    embed_dim=1536,  # openai embedding dimension
    hybrid_search=True,
    text_search_config="english",
)

storage_context = StorageContext.from_defaults(
    vector_store=hybrid_vector_store
)
hybrid_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

hybrid_query_engine = hybrid_index.as_query_engine(
    vector_store_query_mode="hybrid", sparse_top_k=2
)
hybrid_response = hybrid_query_engine.query(
    "Who does Paul Graham think of with the word schtick"
)

print(hybrid_response)

