#!/usr/bin/env python
# coding: utf-8

# # Nebula Graph Store

# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"

import logging
import sys
from llama_index.llms import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# For Azure OpenAI
import os
import json
import openai
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

import logging
import sys

from IPython.#display import Markdown, #display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://<foo-bar>.openai.azure.com"
openai.api_version = "2022-12-01"
os.environ["OPENAI_API_KEY"] = "<your-openai-key>"
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = AzureOpenAI(
    model="<foo-bar-model>",
    engine="<foo-bar-deployment>",
    temperature=0,
    api_key=openai.api_key,
    api_type=openai.api_type,
    api_base=openai.api_base,
    api_version=openai.api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_model,
)

# ## Using Knowledge Graph with NebulaGraphStore

# #### Building the Knowledge Graph

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = OpenAI(temperature=0, model="text-davinci-002")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

# ## Prepare for NebulaGraph

get_ipython().run_line_magic('pip', 'install nebula3-python')

os.environ["NEBULA_USER"] = "root"
os.environ[
    "NEBULA_PASSWORD"
] = "<password>"  # replace with your password, by default it is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally

# Assume that the graph has already been created
# Create a NebulaGraph cluster with:
# Option 0: `curl -fsSL nebula-up.siwei.io/install.sh | bash`
# Option 1: NebulaGraph Docker Extension https://hub.docker.com/extensions/weygu/nebulagraph-dd-ext
# and that the graph space is called "paul_graham_essay"
# If not, create it with the following commands from NebulaGraph's console:
# CREATE SPACE paul_graham_essay(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# :sleep 10;
# USE paul_graham_essay;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# CREATE TAG INDEX entity_index ON entity(name(256));

space_name = "paul_graham_essay"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# #

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

# #### Querying the Knowledge Graph

query_engine = index.as_query_engine()

response = query_engine.query("Tell me more about Interleaf")

#display(Markdown(f"<b>{response}</b>"))

response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

#display(Markdown(f"<b>{response}</b>"))

# ## Visualizing the Graph RAG
# 
# If we visualize the Graph based RAG, starting from the term `['Interleaf', 'history', 'Software', 'Company'] `, we could see how those connected context looks like, and it's a different form of Info./Knowledge:
# 
# - Refined and Concise Form
# - Fine-grained Segmentation
# - Interconnected-sturcutred nature

get_ipython().run_line_magic('pip', 'install ipython-ngql networkx pyvis')
get_ipython().run_line_magic('load_ext', 'ngql')

get_ipython().run_line_magic('ngql', '--address 127.0.0.1 --port 9669 --user root --password <password>')

get_ipython().run_cell_magic('ngql', '', "USE paul_graham_essay;\nMATCH p=(n)-[*1..2]-()\n  WHERE id(n) IN ['Interleaf', 'history', 'Software', 'Company'] \nRETURN p LIMIT 100;\n")

get_ipython().run_line_magic('ng_draw', '')

# #### Query with embeddings

# NOTE: can take a while!

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

# query using top 3 triplets plus keywords (duplicate triplets are removed)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

#display(Markdown(f"<b>{response}</b>"))

# #### Query with more global(cross node) context

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
    explore_global_knowledge=True,
)

response = query_engine.query("Tell me more about what the author and Lisp")

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

# not yet implemented

# initialize an empty index for now
index = KnowledgeGraphIndex.from_documents([], storage_context=storage_context)

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

response = query_engine.query("Tell me more about Interleaf")

str(response)

