#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/docstore/MongoDocstoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index import VectorStoreIndex, SummaryIndex, SimpleKeywordTableIndex
from llama_index.composability import ComposableGraph
from llama_index.llms import OpenAI
from llama_index.response.notebook_utils import #display_response

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# #### Load Documents

reader = SimpleDirectoryReader("./data/paul_graham/")
documents = reader.load_data()

# #### Parse into Nodes

from llama_index.node_parser import SentenceSplitter

nodes = SentenceSplitter().get_nodes_from_documents(documents)

# #### Add to Docstore

MONGO_URI = os.environ["MONGO_URI"]

from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store.mongo_index_store import MongoIndexStore

storage_context = StorageContext.from_defaults(
    docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
    index_store=MongoIndexStore.from_uri(uri=MONGO_URI),
)

storage_context.docstore.add_documents(nodes)

# #### Define Multiple Indexes
# 
# Each index uses the same underlying Node.

summary_index = SummaryIndex(nodes, storage_context=storage_context)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

# NOTE: the docstore still has the same nodes
len(storage_context.docstore.docs)

# #### Test out saving and loading

# NOTE: docstore and index_store is persisted in MongoDB by default
# NOTE: here only need to persist simple vector store to disk
storage_context.persist()

# note down index IDs
list_id = summary_index.index_id
vector_id = vector_index.index_id
keyword_id = keyword_table_index.index_id

from llama_index.indices.loading import load_index_from_storage

# re-create storage context
storage_context = StorageContext.from_defaults(
    docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
    index_store=MongoIndexStore.from_uri(uri=MONGO_URI),
)

# load indices
summary_index = load_index_from_storage(
    storage_context=storage_context, index_id=list_id
)
vector_index = load_index_from_storage(
    storage_context=storage_context, vector_id=vector_id
)
keyword_table_index = load_index_from_storage(
    storage_context=storage_context, keyword_id=keyword_id
)

# #### Test out some Queries

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_chatgpt = ServiceContext.from_defaults(
    llm=chatgpt, chunk_size=1024
)

query_engine = summary_index.as_query_engine()
list_response = query_engine.query("What is a summary of this document?")

#display_response(list_response)

query_engine = vector_index.as_query_engine()
vector_response = query_engine.query("What did the author do growing up?")

#display_response(vector_response)

query_engine = keyword_table_index.as_query_engine()
keyword_response = query_engine.query(
    "What did the author do after his time at YC?"
)

#display_response(keyword_response)

