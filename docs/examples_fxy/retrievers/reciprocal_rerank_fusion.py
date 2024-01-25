#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/retrievers/reciprocal_rerank_fusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Reciprocal Rerank Fusion Retriever
# 

# 
# The retrieved nodes will be reranked according to the `Reciprocal Rerank Fusion` algorithm demonstrated in this [paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). It provides an effecient method for rerranking retrieval results without excessive computation or reliance on external models.
# 
# Full credits go to @Raduaschl on github for their [example implementation here](https://github.com/Raudaschl/rag-fusion).

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Setup

# 
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Next, we will setup a vector index over the documentation.

from llama_index import VectorStoreIndex, ServiceContext

service_context = ServiceContext.from_defaults(chunk_size=256)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# ## Create a Hybrid Fusion Retriever
# 

# 
# Since both of these retrievers calculate a score, we can use the reciprocal rerank algorithm to re-sort our nodes without using an additional models or excessive computation.
# 
# This setup will also query 4 times, once with your original query, and generate 3 more queries.
# 
# By default, it uses the following prompt to generate extra queries:
# 
# ```python
# QUERY_GEN_PROMPT = (
#     "You are a helpful assistant that generates multiple search queries based on a "
#     "single input query. Generate {num_queries} search queries, one on each line, "
#     "related to the following input query:\n"
#     "Query: {query}\n"
#     "Queries:\n"
# )
# ```

# First, we create our retrievers. Each will retrieve the top-2 most similar nodes:

from llama_index.retrievers import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

# Next, we can create our fusion retriever, which well return the top-2 most similar nodes from the 4 returned nodes from the retrievers:

from llama_index.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

# apply nested async to run in a notebook
import nest_asyncio

nest_asyncio.apply()

nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")

# As we can see, both retruned nodes correctly mention Viaweb and Interleaf!

# ## Use in a Query Engine!
# 
# Now, we can plug our retriever into a query engine to synthesize natural language responses.

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query("What happened at Interleafe and Viaweb?")

from llama_index.response.notebook_utils import #display_response

#display_response(response)

