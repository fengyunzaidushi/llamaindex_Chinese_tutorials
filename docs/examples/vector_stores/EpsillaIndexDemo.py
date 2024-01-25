#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/EpsillaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Epsilla Vector Store

# As a prerequisite, you need to have a running Epsilla vector database (for example, through our docker image), and install the ``pyepsilla`` package.
# View full docs at [docs](https://epsilla-inc.gitbook.io/epsilladb/quick-start)

#('pip/pip3 install pyepsilla')

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SimpleDirectoryReader, Document, StorageContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import EpsillaVectorStore
import textwrap

# ### Setup OpenAI
# Lets first begin by adding the openai api key. It will be used to created embeddings for the documents loaded into the index.

import openai
import getpass

OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")
openai.api_key = OPENAI_API_KEY

# ### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Loading documents
# Load documents stored in the `/data/paul_graham` folder using the SimpleDirectoryReader.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(f"Total documents: {len(documents)}")
print(f"First document, id: {documents[0].doc_id}")
print(f"First document, hash: {documents[0].hash}")

# ### Create the index
# Here we create an index backed by Epsilla using the documents loaded previously. EpsillaVectorStore takes a few arguments.
# - client (Any): Epsilla client to connect to.
# 
# - collection_name (str, optional): Which collection to use. Defaults to "llama_collection".
# - db_path (str, optional): The path where the database will be persisted. Defaults to "/tmp/langchain-epsilla".
# - db_name (str, optional): Give a name to the loaded database. Defaults to "langchain_store".
# - dimension (int, optional): The dimension of the embeddings. If not provided, collection creation will be done on first insert. Defaults to None.
# - overwrite (bool, optional): Whether to overwrite existing collection with same name. Defaults to False.
# 
# Epsilla vectordb is running with default host "localhost" and port "8888".

# Create an index over the documnts
from pyepsilla import vectordb

client = vectordb.Client()
vector_store = EpsillaVectorStore(client=client, db_path="/tmp/llamastore")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# ### Query the data
# Now we have our document stored in the index, we can ask questions against the index.

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("How did the author learn about AI?")
print(textwrap.fill(str(response), 100))

# Next, let's try to overwrite the previous data.

vector_store = EpsillaVectorStore(client=client, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
single_doc = Document(text="Epsilla is the vector database we are using.")
index = VectorStoreIndex.from_documents(
    [single_doc],
    storage_context=storage_context,
)

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("What vector database is being used?")
print(textwrap.fill(str(response), 100))

# Next, let's add more data to existing collection.

vector_store = EpsillaVectorStore(client=client, overwrite=False)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
for doc in documents:
    index.insert(document=doc)

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("What vector database is being used?")
print(textwrap.fill(str(response), 100))

