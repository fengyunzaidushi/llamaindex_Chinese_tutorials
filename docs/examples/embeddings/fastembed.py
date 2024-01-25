#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/clarifai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Qdrant FastEmbed Embeddings
# 
# LlamaIndex supports [FastEmbed](https://qdrant.github.io/fastembed/) for embeddings generation.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

get_ipython().run_line_magic('pip', 'install llama-index')

# To use this provider, the `fastembed` package needs to be installed.

get_ipython().run_line_magic('pip', 'install fastembed')

# The list of supported models can be found [here](https://qdrant.github.io/fastembed/examples/Supported_Models/).

from llama_index.embeddings import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

embeddings = embed_model.get_text_embedding("Some text to embed.")
print(len(embeddings))
print(embeddings[:5])

