#!/usr/bin/env python
# coding: utf-8

# # pgvecto.rs
# 
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/PGVectoRsDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Firstly, you will probably need to install dependencies :

get_ipython().run_line_magic('pip', 'install llama-index "pgvecto_rs[sdk]"')

# Then start the pgvecto.rs server as the [official document suggests](https://github.com/tensorchord/pgvecto.rs#installation):

#('docker run --name pgvecto-rs-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d tensorchord/pgvecto-rs:latest')

# Setup the logger.

import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# #### Creating a pgvecto_rs client

from pgvecto_rs.sdk import PGVectoRs

URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
    port=os.getenv("DB_PORT", "5432"),
    host=os.getenv("DB_HOST", "localhost"),
    username=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "mysecretpassword"),
    db_name=os.getenv("DB_NAME", "postgres"),
)

client = PGVectoRs(
    db_url=URL,
    collection_name="example",
    dimension=1536,  # Using OpenAI’s text-embedding-ada-002
)

# #### Setup OpenAI

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# #### Load documents, build the PGVectoRsStore and VectorStoreIndex

from IPython.#display import Markdown, #display

from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores import PGVectoRsStore

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize without metadata filter
from llama_index.storage.storage_context import StorageContext

vector_store = PGVectoRsStore(client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# #### Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

#display(Markdown(f"<b>{response}</b>"))

