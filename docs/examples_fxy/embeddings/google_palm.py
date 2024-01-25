#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/google_palm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Google PaLM Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# imports
from llama_index.embeddings import GooglePaLMEmbedding

# get API key and create embeddings

model_name = "models/embedding-gecko-001"
api_key = "YOUR API KEY"

embed_model = GooglePaLMEmbedding(model_name=model_name, api_key=api_key)

embeddings = embed_model.get_text_embedding("Google PaLM Embeddings.")

print(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

