#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/MongoDBAtlasVectorSearch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## MongoDB Atlas

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# Provide URI to constructor, or use environment variable
import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file.base import SimpleDirectoryReader

# Download Data

#("mkdir -p 'data/10k/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'")

# mongo_uri = os.environ["MONGO_URI"]
mongo_uri = (
    "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority"
)
mongodb_client = pymongo.MongoClient(mongo_uri)
store = MongoDBAtlasVectorSearch(mongodb_client)
storage_context = StorageContext.from_defaults(vector_store=store)
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()
index = VectorStoreIndex.from_documents(
    uber_docs, storage_context=storage_context
)

response = index.as_query_engine().query("What was Uber's revenue?")
#display(Markdown(f"<b>{response}</b>"))

from llama_index.response.schema import Response

print(store._collection.count_documents({}))
# Get a ref_doc_id
typed_response = (
    response if isinstance(response, Response) else response.get_response()
)
ref_doc_id = typed_response.source_nodes[0].node.ref_doc_id
print(store._collection.count_documents({"metadata.ref_doc_id": ref_doc_id}))
# Test store delete
if ref_doc_id:
    store.delete(ref_doc_id)
    print(store._collection.count_documents({}))

# Note: For MongoDB Atlas, you have to additionally create an Atlas Search Index.
# 
# [MongoDB Docs | Create an Atlas Vector Search Index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/)
