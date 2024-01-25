#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/low_level/retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Building Retrieval from Scratch
# 

# 
# We use Pinecone as the vector database. We load in nodes using our high-level ingestion abstractions (to see how to build this from scratch, see our previous tutorial!).
# 
# We will show how to do the following:
# 1. How to generate a query embedding
# 2. How to query the vector database using different search modes (dense, sparse, hybrid)
# 3. How to parse results into a set of Nodes
# 4. How to put this in a custom retriever

# ## Setup
# 
# We build an empty Pinecone Index, and define the necessary LlamaIndex wrappers/abstractions so that we can start loading data into Pinecone. 

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Build Pinecone Index

import pinecone
import os

api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="us-west1-gcp")

# dimensions are for text-embedding-ada-002
pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)

pinecone_index = pinecone.Index("quickstart")

# [Optional] drop contents in index
pinecone_index.delete(deleteAll=True)

# #### Create PineconeVectorStore
# 
# Simple wrapper abstraction to use in LlamaIndex. Wrap in StorageContext so we can easily load in Nodes. 

from llama_index.vector_stores import PineconeVectorStore

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# #### Load Documents

#('mkdir data')
#('wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"')

from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

# #### Load into Vector Store
# 
# Load in documents into the PineconeVectorStore. 
# 
# **NOTE**: We use high-level ingestion abstractions here, with `VectorStoreIndex.from_documents.` We'll refrain from using `VectorStoreIndex` for the rest of this tutorial.

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.storage import StorageContext

service_context = ServiceContext.from_defaults(chunk_size=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context, storage_context=storage_context
)

# ## Define Vector Retriever
# 
# Now we're ready to define our retriever against this vector store to retrieve a set of nodes.
# 
# We'll show the processes step by step and then wrap it into a function.

query_str = "Can you tell me about the key concepts for safety finetuning"

# ### 1. Generate a Query Embedding

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

query_embedding = embed_model.get_query_embedding(query_str)

# ### 2. Query the Vector Database
# 
# We show how to query the vector database with different modes: default, sparse, and hybrid.
# 
# We first construct a `VectorStoreQuery` and then query the vector db.

# construct vector store query
from llama_index.vector_stores import VectorStoreQuery

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
query_result

# ### 3. Parse Result into a set of Nodes
# 
# The `VectorStoreQueryResult` returns the set of nodes and similarities. We construct a `NodeWithScore` object with this.

from llama_index.schema import NodeWithScore
from typing import Optional

nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

from llama_index.response.notebook_utils import #display_source_node

for node in nodes_with_scores:
    #display_source_node(node, source_length=1000)

# ### 4. Put this into a Retriever
# 
# Let's put this into a Retriever subclass that can plug into the rest of LlamaIndex workflows!

from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List

class PineconeRetriever(BaseRetriever):
    """Retriever over a pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

retriever = PineconeRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

retrieved_nodes = retriever.retrieve(query_str)
for node in retrieved_nodes:
    #display_source_node(node, source_length=1000)

# ## Plug this into our RetrieverQueryEngine to synthesize a response
# 
# **NOTE**: We'll cover more on how to build response synthesis from scratch in future tutorials! 

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query(query_str)

print(str(response))

