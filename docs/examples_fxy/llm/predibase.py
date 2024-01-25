#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/predibase.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Predibase
# 
# This notebook shows how you can use Predibase-hosted LLM's within Llamaindex. You can add [Predibase](https://predibase.com) to your existing Llamaindex worklow to: 
# 1. Deploy and query pre-trained or custom open source LLM’s without the hassle
# 2. Operationalize an end-to-end Retrieval Augmented Generation (RAG) system
# 3. Fine-tune your own LLM in just a few lines of code
# 
# ## Getting Started
# 1. Sign up for a free Predibase account [here](https://predibase.com/free-trial)
# 2. Create an Account
# 3. Go to Settings > My profile and Generate a new API Token.

#('pip install llama-index --quiet')
#('pip install predibase --quiet')
#('pip install sentence-transformers --quiet')

import os

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"
from llama_index.llms import PredibaseLLM

# ## Flow 1: Query Predibase LLM directly

llm = PredibaseLLM(
    model_name="llama-2-13b", temperature=0.3, max_new_tokens=512
)
# You can query any HuggingFace or fine-tuned LLM that's hosted on Predibase

result = llm.complete("Can you recommend me a nice dry white wine?")
print(result)

# ## Flow 2: Retrieval Augmented Generation (RAG) with Predibase LLM

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Documents

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ### Configure Predibase LLM

llm = PredibaseLLM(
    model_name="llama-2-13b",
    temperature=0.3,
    max_new_tokens=400,
    context_window=1024,
)
service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)

# ### Setup and Query Index

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(response)

