#!/usr/bin/env python
# coding: utf-8

# # Kùzu Graph Store
# 
# This notebook walks through configuring `Kùzu` to be the backend for graph storage in LlamaIndex.

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ## Prepare for Kùzu

# Clean up all the directories used in this notebook
import shutil

shutil.rmtree("./test1", ignore_errors=True)
shutil.rmtree("./test2", ignore_errors=True)
shutil.rmtree("./test3", ignore_errors=True)

get_ipython().run_line_magic('pip', 'install kuzu')
import kuzu

db = kuzu.Database("test1")

# ## Using Knowledge Graph with KuzuGraphStore

from llama_index.graph_stores import KuzuGraphStore

graph_store = KuzuGraphStore(db)

# #### Building the Knowledge Graph

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    KnowledgeGraphIndex,
)

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display
import kuzu

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

# define LLM

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

from llama_index.storage.storage_context import StorageContext

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    service_context=service_context,
)

# #### Querying the Knowledge Graph
# 
# First, we can query and send only the triplets to the LLM.

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

#display(Markdown(f"<b>{response}</b>"))

# For more detailed answers, we can also send the text from where the retrieved tripets were extracted.

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

#display(Markdown(f"<b>{response}</b>"))

# #### Query with embeddings

# NOTE: can take a while!
db = kuzu.Database("test2")
graph_store = KuzuGraphStore(db)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    service_context=service_context,
    storage_context=storage_context,
    include_embeddings=True,
)

rel_map = graph_store.get_rel_map()

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

get_ipython().run_line_magic('pip', 'install pyvis')

## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("kuzugraph_draw.html")

# #### [Optional] Try building the graph and manually add triplets!

from llama_index.node_parser import SentenceSplitter

node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

# initialize an empty database
db = kuzu.Database("test3")
graph_store = KuzuGraphStore(db)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
index = KnowledgeGraphIndex(
    [],
    service_context=service_context,
    storage_context=storage_context,
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

