#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/AzureCosmosDBMongoDBvCoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Azure CosmosDB MongoDB Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import json
import openai
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# ### Setup Azure OpenAI
# The first step is to configure the models. They will be used to create embeddings for the documents loaded into the db and for llm completions. 

import os

# Set up the AzureOpenAI instance
llm = AzureOpenAI(
    model_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    deployment_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)

# Set up the OpenAIEmbedding instance
embed_model = OpenAIEmbedding(
    model=os.getenv("OPENAI_MODEL_EMBEDDING"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_EMBEDDING"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

from llama_index import set_global_service_context

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

set_global_service_context(service_context)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

print("Document ID:", documents[0].doc_id)

# ### Create the index
# Here we establish the connection to an Azure Cosmosdb mongodb vCore cluster and create an vector search index.

import pymongo
from llama_index.vector_stores.azurecosmosmongo import (
    AzureCosmosDBMongoDBVectorSearch,
)
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file.base import SimpleDirectoryReader

connection_string = os.environ.get("AZURE_COSMOSDB_MONGODB_URI")
mongodb_client = pymongo.MongoClient(connection_string)
store = AzureCosmosDBMongoDBVectorSearch(
    mongodb_client=mongodb_client,
    db_name="demo_vectordb",
    collection_name="paul_graham_essay",
)
storage_context = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the index
# We can now ask questions using our index.

query_engine = index.as_query_engine()
response = query_engine.query("What did the author love working on?")

import textwrap

print(textwrap.fill(str(response), 100))

response = query_engine.query("What did he/she do in summer of 2016?")

print(textwrap.fill(str(response), 100))

