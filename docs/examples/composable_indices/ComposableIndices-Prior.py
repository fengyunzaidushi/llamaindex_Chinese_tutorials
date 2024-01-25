#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/composable_indices/ComposableIndices-Prior.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Composable Graph Basic

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
    EmptyIndex,
    TreeIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Datasets
# 
# Load PG's essay

# load PG's essay
essay_documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ### Building the document indices
# - Build a vector index for PG's essay
# - Also build an empty index (to store prior knowledge)

# configure
service_context = ServiceContext.from_defaults(chunk_size=512)
storage_context = StorageContext.from_defaults()

# build essay index
essay_index = VectorStoreIndex.from_documents(
    essay_documents,
    service_context=service_context,
    storage_context=storage_context,
)
empty_index = EmptyIndex(
    service_context=service_context, storage_context=storage_context
)

# ### Query Indices
# See the response of querying each index

query_engine = essay_index.as_query_engine(
    similarity_top_k=3,
    response_mode="tree_summarize",
)
response = query_engine.query(
    "Tell me about what Sam Altman did during his time in YC",
)

print(str(response))

query_engine = empty_index.as_query_engine(response_mode="generation")
response = query_engine.query(
    "Tell me about what Sam Altman did during his time in YC",
)

print(str(response))

# Define summary for each index.

essay_index_summary = (
    "This document describes Paul Graham's life, from early adulthood to the"
    " present day."
)
empty_index_summary = "This can be used for general knowledge purposes."

# ### Define Graph (Summary Index as Parent Index)
# 
# This allows us to synthesize responses both using a knowledge corpus as well as prior knowledge.

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SummaryIndex,
    [essay_index, empty_index],
    index_summaries=[essay_index_summary, empty_index_summary],
    service_context=service_context,
    storage_context=storage_context,
)

# [optional] persist to disk
storage_context.persist()
root_id = graph.root_id

# [optional] load from disk
from llama_index.indices.loading import load_graph_from_storage

graph = load_graph_from_storage(storage_context, root_id=root_id)

# configure query engines
custom_query_engines = {
    essay_index.index_id: essay_index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
}

# set Logging to DEBUG for more detailed outputs
# ask it a question about Sam Altman
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
response = query_engine.query(
    "Tell me about what Sam Altman did during his time in YC",
)

print(str(response))

# Get source of response
print(response.get_formatted_sources())

# ### Define Graph (Tree Index as Parent Index)
# 
# This allows us to "route" a query to either a knowledge-augmented index, or to the LLM itself.

from llama_index.indices.composability import ComposableGraph

# configure retriever
custom_query_engines = {
    essay_index.index_id: essay_index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
}

graph2 = ComposableGraph.from_indices(
    TreeIndex,
    [essay_index, empty_index],
    index_summaries=[essay_index_summary, empty_index_summary],
)

# set Logging to DEBUG for more detailed outputs
# ask it a question about NYC
query_engine = graph2.as_query_engine(
    custom_query_engines=custom_query_engines
)
response = query_engine.query(
    "Tell me about what Paul Graham did growing up?",
)

str(response)

print(response.get_formatted_sources())

response = query_engine.query(
    "Tell me about Barack Obama",
)

str(response)

response.get_formatted_sources()

