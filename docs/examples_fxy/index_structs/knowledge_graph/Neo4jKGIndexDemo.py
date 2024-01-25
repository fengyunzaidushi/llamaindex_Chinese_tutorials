#!/usr/bin/env python
# coding: utf-8

# # Neo4j Graph Store

# For OpenAI

import os

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

import logging
import sys
from llama_index.llms import OpenAI
from llama_index import ServiceContext

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# define LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
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
    deployment_name="<foo-bar-deployment>",
    temperature=0,
    openai_api_version=openai.api_version,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)

# You need to deploy your own embedding model as well as your own chat completion model
embedding_llm = OpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<foo-bar-deployment>",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

# ## Using Knowledge Graph with Neo4jGraphStore

# #### Building the Knowledge Graph

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import Neo4jGraphStore

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

# define LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

# ## Prepare for Neo4j

get_ipython().run_line_magic('pip', 'install neo4j')

username = "neo4j"
password = "retractor-knot-thermocouples"
url = "bolt://44.211.44.239:7687"
database = "neo4j"

# #

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    service_context=service_context,
)

# #### Querying the Knowledge Graph
# 
# First, we can query and send only the triplets to the LLM.

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("Tell me more about Interleaf")

#display(Markdown(f"<b>{response}</b>"))

# For more detailed answers, we can also send the text from where the retrieved tripets were extracted.

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

#display(Markdown(f"<b>{response}</b>"))

# #### Query with embeddings

# Clean dataset first
graph_store.query(
    """
MATCH (n) DETACH DELETE n
"""
)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    service_context=service_context,
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

# #### [Optional] Try building the graph and manually add triplets!

from llama_index.node_parser import SentenceSplitter

node_parser = SentenceSplitter()

nodes = node_parser.get_nodes_from_documents(documents)

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

#display(Markdown(f"<b>{response}</b>"))

