#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/retrievers/you_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # You.com Retriever
# 
# This notebook walks you through how to setup a Retriever that can fetch from You.com

from llama_index.retrievers import YouRetriever

you_api_key = "" or os.environ["YOU_API_KEY"]

retriever = YouRetriever(api_key=you_api_key)

retrieved_results = retriever.retrieve("national parks in the US")

print(retrieved_results[0].get_content())

# from llama_index.response.notebook_utils import #display_source_node
# for n in retrieved_results:
#     #display_source_node(n)

# ## Use in Query Engine

from llama_index.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever,
)

response = query_engine.query("Tell me about national parks in the US")
print(str(response))

