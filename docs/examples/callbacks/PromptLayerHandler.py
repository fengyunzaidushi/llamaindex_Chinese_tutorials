#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/callbacks/LlamaDebugHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PromptLayer Handler
# [PromptLayer](https://promptlayer.com) is an LLMOps tool to help manage prompts, check out the [features](https://docs.promptlayer.com/introduction). Currently we only support OpenAI for this integration.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and PromptLayer.

#('pip install llama-index')
#('pip install promptlayer')

# ## Configure API keys

import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["PROMPTLAYER_API_KEY"] = "pl_..."

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

from llama_index import SimpleDirectoryReader

docs = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ## Callback Manager Setup

from llama_index import set_global_handler

# pl_tags are optional, to help you organize your prompts and apps
set_global_handler("promptlayer", pl_tags=["paul graham", "essay"])

# ## Trigger the callback with a query

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

# ## Access [promptlayer.com](https://promptlayer.com) to see stats

# ![image.png](attachment:image.png)
