#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/vector_stores/AsyncIndexCreationDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Simple Vector Store - Async Index Creation

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import time

# Helps asyncio run within Jupyter
import nest_asyncio

nest_asyncio.apply()

# My OpenAI Key
import os

os.environ["OPENAI_API_KEY"] = "[YOUR_API_KEY]"

from llama_index import VectorStoreIndex, download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(
    pages=[
        "Berlin",
        "Santiago",
        "Moscow",
        "Tokyo",
        "Jakarta",
        "Cairo",
        "Bogota",
        "Shanghai",
        "Damascus",
    ]
)

len(documents)

# 9 Wikipedia articles downloaded as documents

start_time = time.perf_counter()
index = VectorStoreIndex.from_documents(documents)
duration = time.perf_counter() - start_time
print(duration)

# Standard index creation took 7.69 seconds

start_time = time.perf_counter()
index = VectorStoreIndex(documents, use_async=True)
duration = time.perf_counter() - start_time
print(duration)

# Async index creation took 2.37 seconds

query_engine = index.as_query_engine()
query_engine.query("What is the etymology of Jakarta?")

