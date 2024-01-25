#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/mistralai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # MistralAI Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# imports
from llama_index.embeddings import MistralAIEmbedding

# get API key and create embeddings
api_key = "YOUR API KEY"
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)

embeddings = embed_model.get_text_embedding("La Plateforme - The Platform")

print(f"Dimension of embeddings: {len(embeddings)}")

embeddings[:5]

