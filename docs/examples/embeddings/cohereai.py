#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/cohereai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # CohereAI Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os

cohere_api_key = "YOUR_API_KEY"
os.environ["COHERE_API_KEY"] = cohere_api_key

# #### With latest `embed-english-v3.0` embeddings.
# 
# - input_type="search_document": Use this for texts (documents) you want to store in your vector database
# 
# - input_type="search_query": Use this for search queries to find the most relevant documents in your vector database

from llama_index.embeddings.cohereai import CohereEmbedding

# with input_typ='search_query'
embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

print(len(embeddings))
print(embeddings[:5])

# with input_type = 'search_document'
embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

print(len(embeddings))
print(embeddings[:5])

# #### With old `embed-english-v2.0` embeddings.

embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key, model_name="embed-english-v2.0"
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

print(len(embeddings))
print(embeddings[:5])

# #### Now with latest `embed-english-v3.0` embeddings, 
# 
# let's use 
# 1. input_type=`search_document` to build index
# 2. input_type=`search_query` to retrive relevant context.

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)

from llama_index.llms import LiteLLM
from llama_index.response.notebook_utils import #display_source_node

from IPython.#display import Markdown, #display

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load Data

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# #### Build index with input_type = 'search_document'

llm = LiteLLM("command-nightly")
embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)
index = VectorStoreIndex.from_documents(
    documents=documents, service_context=service_context
)

# #### Build retriever with input_type = 'search_query'

embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

search_query_retriever = index.as_retriever(service_context=service_context)

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened in the summer of 1995?"
)

for n in search_query_retrieved_nodes:
    #display_source_node(n, source_length=2000)

