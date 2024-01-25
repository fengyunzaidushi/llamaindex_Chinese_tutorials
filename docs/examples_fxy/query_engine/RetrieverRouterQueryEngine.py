#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/query_engine/RetrieverRouterQueryEngine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Retriever Router Query Engine

# 
# We use our new `ToolRetrieverRouterQueryEngine` class for this! 

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize service context (set chunk size)
service_context = ServiceContext.from_defaults(chunk_size=1024)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# ### Define Summary Index and Vector Index over Same Data 

summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

# ### Define Query Engine and Tool for these Indices
# 
# We define a Query Engine for each Index. We then wrap these with our `QueryEngineTool`.

from llama_index.tools.query_engine import QueryEngineTool

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
vector_query_engine = vector_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description="Useful for questions asking for a biography of the author.",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific snippets from the author's life, like"
        " his time in college, his time in YC, or more."
    ),
)

# ### Define Retrieval-Augmented Router Query Engine
# 
# We define a router query engine that's augmented with a retrieval mechanism, to help deal with the case when the set of choices is too large. 
# 
# To do this, we first define an `ObjectIndex` over the set of query engine tools. The `ObjectIndex` is defined an underlying index data structure (e.g. a vector index, keyword index), and can serialize QueryEngineTool objects to/from our indices.
# 
# We then use our `ToolRetrieverRouterQueryEngine` class, and pass in an `ObjectRetriever` over `QueryEngineTool` objects.
# The `ObjectRetriever` corresponds to our `ObjectIndex`. 
# 
# This retriever can then dyamically retrieve the relevant query engines during query-time. This allows us to pass in an arbitrary number of query engine tools without worrying about prompt limitations. 

from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping

tool_mapping = SimpleToolNodeMapping.from_objects([list_tool, vector_tool])
obj_index = ObjectIndex.from_objects(
    [list_tool, vector_tool],
    tool_mapping,
    VectorStoreIndex,
)

from llama_index.query_engine import ToolRetrieverRouterQueryEngine

query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

response = query_engine.query("What is a biography of the author's life?")

print(str(response))

response

response = query_engine.query(
    "What did Paul Graham do during his time in college?"
)

print(str(response))

