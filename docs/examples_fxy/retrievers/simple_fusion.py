#!/usr/bin/env python
# coding: utf-8

# # Simple Fusion Retriever
# 

# 
# The retrieved nodes will be returned as the top-k across all queries and indexes, as well as handling de-duplication of any nodes.

import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

# ## Setup
# 
# For this notebook, we will use two very similar pages of our documentation, each stored in a separaete index.

from llama_index import SimpleDirectoryReader

documents_1 = SimpleDirectoryReader(
    input_files=["../../community/integrations/vector_stores.md"]
).load_data()
documents_2 = SimpleDirectoryReader(
    input_files=["../../core_modules/data_modules/storage/vector_stores.md"]
).load_data()

from llama_index import VectorStoreIndex

index_1 = VectorStoreIndex.from_documents(documents_1)
index_2 = VectorStoreIndex.from_documents(documents_2)

# ## Fuse the Indexes!
# 

# 
# This setup will query 4 times, once with your original query, and generate 3 more queries.
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

from llama_index.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [index_1.as_retriever(), index_2.as_retriever()],
    similarity_top_k=2,
    num_queries=4,  # set this to 1 to disable query generation
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

# apply nested async to run in a notebook
import nest_asyncio

nest_asyncio.apply()

nodes_with_scores = retriever.retrieve("How do I setup a chroma vector store?")

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text[:100]}...")

# ## Use in a Query Engine!
# 
# Now, we can plug our retriever into a query engine to synthesize natural language responses.

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query(
    "How do I setup a chroma vector store? Can you give an example?"
)

from llama_index.response.notebook_utils import #display_response

#display_response(response)

