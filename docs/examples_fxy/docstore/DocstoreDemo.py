#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/docstore/DocstoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index import VectorStoreIndex, SummaryIndex, SimpleKeywordTableIndex
from llama_index.composability import ComposableGraph
from llama_index.llms import OpenAI

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

from llama_index.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# #### Define Multiple Indexes
# 
# Each index uses the same underlying Node.

from llama_index.storage.storage_context import StorageContext

storage_context = StorageContext.from_defaults(docstore=docstore)
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

# NOTE: the docstore sitll has the same nodes
len(storage_context.docstore.docs)

# #### Test out some Queries

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_chatgpt = ServiceContext.from_defaults(
    llm=llm, chunk_size=1024
)

query_engine = summary_index.as_query_engine()
response = query_engine.query("What is a summary of this document?")

query_engine = vector_index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

query_engine = keyword_table_index.as_query_engine()
response = query_engine.query("What did the author do after his time at YC?")

print(response)

