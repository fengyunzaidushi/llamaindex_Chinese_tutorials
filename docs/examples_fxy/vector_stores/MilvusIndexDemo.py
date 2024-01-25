#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/MilvusIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Milvus Vector Store

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import MilvusVectorStore
from IPython.#display import Markdown, #display
import textwrap

# ### Setup OpenAI
# Lets first begin by adding the openai api key. This will allow us to access openai for embeddings and to use chatgpt.

import openai

openai.api_key = "sk-"

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Generate our data
# With our LLM set, lets start using the Milvus Index. As a first example, lets generate a document from the file found in the `data/paul_graham/` folder. In this folder there is a single essay from Paul Graham titled `What I Worked On`. To generate the documents we will use the SimpleDirectoryReader.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

print("Document ID:", documents[0].doc_id)

# ### Create an index across the data
# Now that we have a document, we can can create an index and insert the document. For the index we will use a GPTMilvusIndex. GPTMilvusIndex takes in a few arguments:
# 
# - collection_name (str, optional): The name of the collection where data will be stored. Defaults to "llamalection".
# - index_params (dict, optional): The index parameters for Milvus, if none are provided an HNSW index will be used. Defaults to None.
# - search_params (dict, optional): The search parameters for a Milvus query. If none are provided, default params will be generated. Defaults to None.
# - dim (int, optional): The dimension of the embeddings. If it is not provided, collection creation will be done on first insert. Defaults to None.
# - host (str, optional): The host address of Milvus. Defaults to "localhost".
# - port (int, optional): The port of Milvus. Defaults to 19530.
# - user (str, optional): The username for RBAC. Defaults to "".
# - password (str, optional): The password for RBAC. Defaults to "".
# - use_secure (bool, optional): Use https. Defaults to False.
# - overwrite (bool, optional): Whether to overwrite existing collection with same name. Defaults to False.
# 

# Create an index over the documnts
from llama_index.storage.storage_context import StorageContext

vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the data
# Now that we have our document stored in the index, we can ask questions against the index. The index will use the data stored in itself as the knowledge base for chatgpt.

query_engine = index.as_query_engine()
response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

# This next test shows that overwriting removes the previous data.

vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [Document(text="The number that is being searched for is ten.")],
    storage_context,
)
query_engine = index.as_query_engine()
res = query_engine.query("Who is the author?")
print("Res:", res)

# The next test shows adding additional data to an already existing  index.

del index, vector_store, storage_context, query_engine

vector_store = MilvusVectorStore(overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
query_engine = index.as_query_engine()
res = query_engine.query("What is the number?")
print("Res:", res)

res = query_engine.query("Who is the author?")
print("Res:", res)

