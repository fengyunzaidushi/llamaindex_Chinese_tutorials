#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/SupabaseVectorIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Supabase Vector Store

# See [this guide](https://supabase.github.io/vecs/hosting/) for instructions on hosting a database on Supabase 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SimpleDirectoryReader, Document, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import SupabaseVectorStore
import textwrap

# ### Setup OpenAI
# The first step is to configure the OpenAI key. It will be used to created embeddings for the documents loaded into the index

import os

os.environ["OPENAI_API_KEY"] = "[your_openai_api_key]"

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# Load the documents stored in the `./data/paul_graham/` using the SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

# ### Create an index backed by Supabase's vector store. 
# This will work with all Postgres providers that support pgvector.
# If the collection does not exist, we will attempt to create a new collection 
# 
# > Note: you need to pass in the embedding dimension if not using OpenAI's text-embedding-ada-002, e.g. `vector_store = SupabaseVectorStore(..., dimension=...)`

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>"
    ),
    collection_name="base_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the index
# We can now ask questions using our index.

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")

print(textwrap.fill(str(response), 100))

response = query_engine.query("What did the author do growing up?")

print(textwrap.fill(str(response), 100))

# ## Using metadata filters

from llama_index.schema import TextNode

nodes = [
    TextNode(
        **{
            "text": "The Shawshank Redemption",
            "metadata": {
                "author": "Stephen King",
                "theme": "Friendship",
            },
        }
    ),
    TextNode(
        **{
            "text": "The Godfather",
            "metadata": {
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
        }
    ),
    TextNode(
        **{
            "text": "Inception",
            "metadata": {
                "director": "Christopher Nolan",
            },
        }
    ),
]

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>"
    ),
    collection_name="metadata_filters_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Define metadata filters

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

# Retrieve from vector store with filters

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

