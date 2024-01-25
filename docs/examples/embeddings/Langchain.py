#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/embeddings/Langchain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Langchain Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, set_global_service_context

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

# optionally set a global service context
set_global_service_context(service_context)

