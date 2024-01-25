#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/embeddings/OpenAI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI Embeddings

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext, set_global_service_context

embed_model = OpenAIEmbedding(embed_batch_size=10)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

# optionally set a global service context
set_global_service_context(service_context)

