#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/ChromaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chroma
# 
# >[Chroma](https://docs.trychroma.com/getting-started) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.
# 
# <a href="https://discord.gg/MMeYNTmh3x" target="_blank">
#       <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
#   </a>&nbsp;&nbsp;
#   <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">
#       <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=white" alt="License">
#   </a>&nbsp;&nbsp;
#   <img src="https://github.com/chroma-core/chroma/actions/workflows/chroma-integration-test.yml/badge.svg?branch=main" alt="Integration Tests">
# 
# - [Website](https://www.trychroma.com/)
# - [Documentation](https://docs.trychroma.com/)
# - [Twitter](https://twitter.com/trychroma)
# - [Discord](https://discord.gg/MMeYNTmh3x)
# 
# Chroma is fully-typed, fully-tested and fully-documented.
# 

# 
# ```sh
# pip install chromadb
# ```
# 
# Chroma runs in various modes. See below for examples of each integrated with LangChain.
# - `in-memory` - in a python script or jupyter notebook
# - `in-memory with persistance` - in a script or notebook and save/load to disk
# - `in a docker container` - as a server running your local machine or in the cloud
# 
# Like any other database, you can: 
# - `.add` 
# - `.get` 
# - `.update`
# - `.upsert`
# - `.delete`
# - `.peek`
# - and `.query` runs the similarity search.
# 
# View full docs at [docs](https://docs.trychroma.com/reference/Collection). 

# ## Basic Example
# 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Creating a Chroma Index

# !pip install llama-index chromadb --quiet
# !pip install chromadb
# !pip install sentence-transformers
# !pip install pydantic==1.10.11

# import
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.#display import Markdown, #display
import chromadb

# set up OpenAI
import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

# ## Basic Example (including saving to disk)
# 
# Extending the previous example, if you want to save to disk, simply initialize the Chroma client and pass the directory where you want the data to be saved to. 
# 
# `Caution`: Chroma makes a best-effort to automatically save data to disk, however multiple in-memory clients can stomp each other's work. As a best practice, only have one client per path running at any given time.

# save to disk

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)

# Query Data from the persisted index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

# ## Basic Example (using the Docker Container)
# 
# You can also run the Chroma Server in a Docker container separately, create a Client to connect to it, and then pass that to LlamaIndex. 
# 
# Here is how to clone, build, and run the Docker Image:
# ```
# git clone git@github.com:chroma-core/chroma.git
# docker-compose up -d --build
# ```

# create the chroma client and add our data
import chromadb

remote_db = chromadb.HttpClient()
chroma_collection = remote_db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Query Data from the Chroma Docker index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
#display(Markdown(f"<b>{response}</b>"))

# ## Update and Delete
# 
# While building toward a real application, you want to go beyond adding data, and also update and delete data. 
# 
# Chroma has users provide `ids` to simplify the bookkeeping here. `ids` can be the name of the file, or a combined has like `filename_paragraphNumber`, etc.
# 
# Here is a basic example showing how to do various operations:

doc_to_update = chroma_collection.get(limit=1)
doc_to_update["metadatas"][0] = {
    **doc_to_update["metadatas"][0],
    **{"author": "Paul Graham"},
}
chroma_collection.update(
    ids=[doc_to_update["ids"][0]], metadatas=[doc_to_update["metadatas"][0]]
)
updated_doc = chroma_collection.get(limit=1)
print(updated_doc["metadatas"][0])

# delete the last document
print("count before", chroma_collection.count())
chroma_collection.delete(ids=[doc_to_update["ids"][0]])
print("count after", chroma_collection.count())

