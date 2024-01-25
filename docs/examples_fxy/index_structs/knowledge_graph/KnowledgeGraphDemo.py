#!/usr/bin/env python
# coding: utf-8

# # Knowledge Graph Index
# 
# This tutorial gives a basic overview of how to use our `KnowledgeGraphIndex`, which handles
# automated knowledge graph construction from unstructured text as well as entity-based querying.
# 
# If you would like to query knowledge graphs in more flexible ways, including pre-existing ones, please
# check out our `KnowledgeGraphQueryEngine` and other constructs.

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ## Using Knowledge Graph

# #### Building the Knowledge Graph

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors

llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

from llama_index.storage.storage_context import StorageContext

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    service_context=service_context,
)

# #### [Optional] Try building the graph and manually add triplets!

# #### Querying the Knowledge Graph

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

#display(Markdown(f"<b>{response}</b>"))

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)

#display(Markdown(f"<b>{response}</b>"))

# #### Query with embeddings

# NOTE: can take a while!
new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    service_context=service_context,
    include_embeddings=True,
)

# query using top 3 triplets plus keywords (duplicate triplets are removed)
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)

#display(Markdown(f"<b>{response}</b>"))

# #### Visualizing the Graph

## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example.html")

# #### [Optional] Try building the graph and manually add triplets!

from llama_index.node_parser import SentenceSplitter

node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

# initialize an empty index for now
index = KnowledgeGraphIndex(
    [],
    service_context=service_context,
)

# add keyword mappings and nodes manually
# add triplets (subject, relationship, object)

# for node 0
node_0_tups = [
    ("author", "worked on", "writing"),
    ("author", "worked on", "programming"),
]
for tup in node_0_tups:
    index.upsert_triplet_and_node(tup, nodes[0])

# for node 1
node_1_tups = [
    ("Interleaf", "made software for", "creating documents"),
    ("Interleaf", "added", "scripting language"),
    ("software", "generate", "web sites"),
]
for tup in node_1_tups:
    index.upsert_triplet_and_node(tup, nodes[1])

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

str(response)

