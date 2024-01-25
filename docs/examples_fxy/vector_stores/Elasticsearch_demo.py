#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/Elasticsearch_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Elasticsearch
# 
# >[Elasticsearch](http://www.github.com/elastic/elasticsearch) is a search database, that supports full text and vector searches.  
# 

# ## Basic Example
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama-index elasticsearch --quiet
# !pip install sentence-transformers
# !pip install pydantic==1.10.11

# import
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ElasticsearchStore
from llama_index.storage.storage_context import StorageContext
from IPython.#display import Markdown, #display

# set up OpenAI
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# define embedding function
embed_model = "local/BAAI/bge-small-en-v1.5"

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

vector_store = ElasticsearchStore(
    index_name="paul_graham_essay", es_url="http://localhost:9200"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

