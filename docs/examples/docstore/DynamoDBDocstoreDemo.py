#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/docstore/DynamoDBDocstoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]

from llama_index.storage.docstore.dynamodb_docstore import (
    DynamoDBDocumentStore,
)
from llama_index.storage.index_store.dynamodb_index_store import (
    DynamoDBIndexStore,
)
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore

storage_context = StorageContext.from_defaults(
    docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME),
    index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME),
    vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME),
)

storage_context.docstore.add_documents(nodes)

# #### Define & Add Multiple Indexes
# 
# Each index uses the same underlying Node.

# https://gpt-index.readthedocs.io/en/latest/api_reference/indices/list.html
summary_index = SummaryIndex(nodes, storage_context=storage_context)

# https://gpt-index.readthedocs.io/en/latest/api_reference/indices/vector_store.html
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# https://gpt-index.readthedocs.io/en/latest/api_reference/indices/table.html
keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

# NOTE: the docstore still has the same nodes
len(storage_context.docstore.docs)

# #### Test out saving and loading

# NOTE: docstore, index_store, and vector_index is persisted in DynamoDB by default when they are created
# NOTE: You can also persist simple vector store to disk by using the command below
storage_context.persist()

# note down index IDs
list_id = summary_index.index_id
vector_id = vector_index.index_id
keyword_id = keyword_table_index.index_id

from llama_index.indices.loading import load_index_from_storage

# re-create storage context
storage_context = StorageContext.from_defaults(
    docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME),
    index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME),
    vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME),
)

summary_index = load_index_from_storage(
    storage_context=storage_context, index_id=list_id
)
keyword_table_index = load_index_from_storage(
    storage_context=storage_context, index_id=keyword_id
)

# You need to add "vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME)" to StorageContext to load vector index from DynamoDB
vector_index = load_index_from_storage(
    storage_context=storage_context, index_id=vector_id
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

