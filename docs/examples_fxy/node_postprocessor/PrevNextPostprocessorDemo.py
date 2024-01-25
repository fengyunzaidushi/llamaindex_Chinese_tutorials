#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Forward/Backward Augmentation
# 
# Showcase capabilities of leveraging Node relationships on top of PG's essay

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.postprocessor import (
    PrevNextNodePostprocessor,
    AutoPrevNextNodePostprocessor,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.storage.docstore import SimpleDocumentStore

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Parse Documents into Nodes, add to Docstore

# load documents
from llama_index.storage.storage_context import StorageContext

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# define service context (wrapper container around current classes)
service_context = ServiceContext.from_defaults(chunk_size=512)

# use node parser in service context to parse into nodes
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# add to docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# ### Build Index

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# ### Add PrevNext Node Postprocessor

node_postprocessor = PrevNextNodePostprocessor(docstore=docstore, num_nodes=4)

query_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[node_postprocessor],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

print(response)

# Try querying index without node postprocessor
query_engine = index.as_query_engine(
    similarity_top_k=1, response_mode="tree_summarize"
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

print(response)

# Try querying index without node postprocessor and higher top-k
query_engine = index.as_query_engine(
    similarity_top_k=3, response_mode="tree_summarize"
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

print(response)

# ### Add Auto Prev/Next Node Postprocessor

node_postprocessor = AutoPrevNextNodePostprocessor(
    docstore=docstore,
    num_nodes=3,
    service_context=service_context,
    verbose=True,
)

query_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[node_postprocessor],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

print(response)

response = query_engine.query(
    "What did the author do during his time at Y Combinator?",
)

print(response)

response = query_engine.query(
    "What did the author do before handing off Y Combinator to Sam Altman?",
)

print(response)

response = query_engine.query(
    "What did the author do before handing off Y Combinator to Sam Altman?",
)

print(response)

