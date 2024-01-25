#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/node_postprocessor/OptimizerDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Sentence Embedding Optimizer

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"

# ### Setup

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=["Berlin"])

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# Compare query with and without optimization for LLM token usage, Embedding Model usage on query, Embedding model usage for optimizer, and total time.

import time
from llama_index import VectorStoreIndex
from llama_index.postprocessor import SentenceEmbeddingOptimizer

print("Without optimization")
start_time = time.time()
query_engine = index.as_query_engine()
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

print("With optimization")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

print("Alternate optimization cutoff")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

