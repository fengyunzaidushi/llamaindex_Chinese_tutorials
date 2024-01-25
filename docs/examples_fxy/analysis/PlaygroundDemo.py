#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/analysis/PlaygroundDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Playground

# My OpenAI Key
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-...."
openai.api_key = os.environ["OPENAI_API_KEY"]

# Hide logs
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# ## Setup
# 
# ### Generate some example Documents

from llama_index import download_loader
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.indices.tree.base import TreeIndex

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=["Berlin"])

# ### Create a list of any sort of indices (custom LLMs, custom embeddings, etc)

indices = [
    VectorStoreIndex.from_documents(documents),
    TreeIndex.from_documents(documents),
]

# ## Using the Playground
# 
# 
# ##

from llama_index.playground import Playground

playground = Playground(indices=indices)

result_df = playground.compare("What is the population of Berlin?")

result_df

# ##
# 
# Automatically construct the playground using a vector, tree, and summary index

# Uses documents in a preset list of indices
playground = Playground.from_docs(documents=documents)

