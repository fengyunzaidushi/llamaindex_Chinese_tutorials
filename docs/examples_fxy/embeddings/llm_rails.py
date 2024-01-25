#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/llm_rails.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LLMRails Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# imports

from llama_index.embeddings.llm_rails import LLMRailsEmbedding

# get credentials and create embeddings

import os

api_key = os.environ.get("API_KEY", "your-api-key")
model_id = os.environ.get("MODEL_ID", "your-model-id")

embed_model = LLMRailsEmbedding(model_id=model_id, api_key=api_key)

embeddings = embed_model.get_text_embedding(
    "It is raining cats and dogs here!"
)

