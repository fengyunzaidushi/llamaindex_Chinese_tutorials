#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/retrievers/bm25_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # BM25 Retriever

# 
# This notebook is very similar to the RouterQueryEngine notebook.

# ## Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes.
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.
import nest_asyncio

nest_asyncio.apply()

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.retrievers import BM25Retriever
from llama_index.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.llms import OpenAI

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ## Load Data
# 
# We first show how to convert a Document into a set of Nodes, and insert into a DocumentStore.

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# initialize service context (set chunk size)
llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    service_context=service_context,
)

# ## BM25 Retriever
# 
# We will search document with bm25 retriever.

# !pip install rank_bm25

# We can pass in the index, doctore, or list of nodes to create the retriever
retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

from llama_index.response.notebook_utils import #display_source_node

# will retrieve context from specific companies
nodes = retriever.retrieve("What happened at Viaweb and Interleaf?")
for node in nodes:
    #display_source_node(node)

nodes = retriever.retrieve("What did Paul Graham do after RISD?")
for node in nodes:
    #display_source_node(node)

# ## Router Retriever with bm25 method
# 
# Now we will combine bm25 retriever with vector index retriever.

from llama_index.tools import RetrieverTool

vector_retriever = VectorIndexRetriever(index)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

retriever_tools = [
    RetrieverTool.from_defaults(
        retriever=vector_retriever,
        description="Useful in most cases",
    ),
    RetrieverTool.from_defaults(
        retriever=bm25_retriever,
        description="Useful if searching about specific information",
    ),
]

from llama_index.retrievers import RouterRetriever

retriever = RouterRetriever.from_defaults(
    retriever_tools=retriever_tools,
    service_context=service_context,
    select_multi=True,
)

# will retrieve all context from the author's life
nodes = retriever.retrieve(
    "Can you give me all the context regarding the author's life?"
)
for node in nodes:
    #display_source_node(node)

# ## Advanced - Hybrid Retriever + Re-Ranking
# 
# Here we extend the base retriever class and create a custom retriever that always uses the vector retriever and BM25 retreiver.
# 
# Then, nodes can be re-ranked and filtered. This lets us keep intermediate top-k values large and letting the re-ranking filter out un-needed nodes.
# 
# To best demonstrate this, we will use a larger set of source documents -- Chapter 3 from the 2022 IPCC Climate Report.

# ### Setup data

#('curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf')

# !pip install pypdf

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI

# load documents
documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# initialize service context (set chunk size)
# -- here, we set a smaller chunk size, to allow for more effective re-ranking
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(chunk_size=256, llm=llm)
nodes = service_context.node_parser.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index = VectorStoreIndex(
    nodes, storage_context=storage_context, service_context=service_context
)

from llama_index.retrievers import BM25Retriever

# retireve the top 10 most similar nodes using embeddings
vector_retriever = index.as_retriever(similarity_top_k=10)

# retireve the top 10 most similar nodes using bm25
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

# ### Custom Retriever Implementation

from llama_index.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

index.as_retriever(similarity_top_k=5)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

# ### Re-Ranker Setup

# !pip install sentence_transformers

from llama_index.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

# ### Retrieve

from llama_index import QueryBundle

nodes = hybrid_retriever.retrieve(
    "What is the impact of climate change on the ocean?"
)
reranked_nodes = reranker.postprocess_nodes(
    nodes,
    query_bundle=QueryBundle(
        "What is the impact of climate change on the ocean?"
    ),
)

print("Initial retrieval: ", len(nodes), " nodes")
print("Re-ranked retrieval: ", len(reranked_nodes), " nodes")

from llama_index.response.notebook_utils import #display_source_node

for node in reranked_nodes:
    #display_source_node(node)

# ### Full Query Engine 

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    service_context=service_context,
)

response = query_engine.query(
    "What is the impact of climate change on the ocean?"
)

from llama_index.response.notebook_utils import #display_response

#display_response(response)

