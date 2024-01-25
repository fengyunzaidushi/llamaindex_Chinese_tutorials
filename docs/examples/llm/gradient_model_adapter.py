#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/gradient_model_adapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Gradient Model Adapter

get_ipython().run_line_magic('pip', 'install llama-index --quiet')
get_ipython().run_line_magic('pip', 'install gradientai --quiet')

import os

os.environ["GRADIENT_ACCESS_TOKEN"] = "{GRADIENT_ACCESS_TOKEN}"
os.environ["GRADIENT_WORKSPACE_ID"] = "{GRADIENT_WORKSPACE_ID}"

# ## Flow 1: Query Gradient LLM directly

from llama_index.llms import GradientModelAdapterLLM

llm = GradientModelAdapterLLM(
    model_adapter_id="{YOUR_MODEL_ADAPTER_ID}",
    max_tokens=400,
)

result = llm.complete("Can you tell me about large language models?")
print(result)

# ## Flow 2: Retrieval Augmented Generation (RAG) with Gradient LLM

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Documents

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# ### Configure Gradient LLM

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model=embed_model
)

# ### Setup and Query Index

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()

response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)
print(response)

