#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/voyageai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Voyage Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# imports

import os
from llama_index.embeddings import VoyageEmbedding

# get API key and create embeddings

model_name = "voyage-01"
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your-api-key")

embed_model = VoyageEmbedding(
    model_name=model_name, voyage_api_key=voyage_api_key
)

embeddings = embed_model.get_query_embedding("What is llamaindex?")

