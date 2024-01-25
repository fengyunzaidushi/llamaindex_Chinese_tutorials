#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/index_structs/knowledge_graph/FalkorDBGraphDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # FalkorDB Graph Store
# 
# This notebook walks through configuring `FalkorDB` to be the backend for graph storage in LlamaIndex.

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# ## Using Knowledge Graph with FalkorDBGraphStore

# ### Start FalkorDB
# 
# The easiest way to start FalkorDB as a Graph database is using the [falkordb](https://hub.docker.com/r/falkordb/falkordb:edge) docker image.
# 
# To follow every step of this tutorial, launch the image as follows:
# 
# ```bash
# docker run -p 6379:6379 -it --rm falkordb/falkordb:edge
# ```

from llama_index.graph_stores import FalkorDBGraphStore

graph_store = FalkorDBGraphStore(
    "redis://localhost:6379", decode_responses=True
)

# #### Building the Knowledge Graph

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    KnowledgeGraphIndex,
)

from llama_index.llms import OpenAI
from IPython.#display import Markdown, #display

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

# #### Visualizing the Graph

get_ipython().run_line_magic('pip', 'install pyvis')

## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("falkordbgraph_draw.html")

