#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/CustomRetrievers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Retriever Query Engine with Custom Retrievers - Simple Hybrid Search
# 

# 
# Combine keyword lookup retrieval with vector retrieval using "AND" and "OR" conditions.

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from IPython.#display import Markdown, #display

# ### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize service context (set chunk size)
service_context = ServiceContext.from_defaults(chunk_size=1024)
node_parser = service_context.node_parser

nodes = node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# ### Define Vector Index and Keyword Table Index over Same Data
# 
# We build a vector index and keyword index over the same DocumentStore

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

# ### Define Custom Retriever
# 
# We now define a custom retriever class that can implement basic hybrid search with both keyword lookup and semantic search.
# 
# - setting "AND" means we take the intersection of the two retrieved sets
# - setting "OR" means we take the union

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

# ### Plugin Retriever into Query Engine
# 
# Plugin retriever into a query engine, and run some queries

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# define custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# define response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# vector query engine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
# keyword query engine
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)

response = custom_query_engine.query(
    "What did the author do during his time at YC?"
)

print(response)

# hybrid search can allow us to not retrieve nodes that are irrelevant
# Yale is never mentioned in the essay
response = custom_query_engine.query(
    "What did the author do during his time at Yale?"
)

print(str(response))
len(response.source_nodes)

# in contrast, vector search will return an answer
response = vector_query_engine.query(
    "What did the author do during his time at Yale?"
)

print(str(response))
len(response.source_nodes)

