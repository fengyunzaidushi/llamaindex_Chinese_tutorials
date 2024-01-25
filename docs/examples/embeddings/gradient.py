#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/embeddings/gradient.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Gradient Embeddings
# 
# [Gradient](https://gradient.ai) offers embeddings model that can be easily integrated with LlamaIndex. Below is an example of how to use it with LlamaIndex.

get_ipython().run_line_magic('pip', 'install llama-index --quiet')
get_ipython().run_line_magic('pip', 'install gradientai --quiet')

# Gradient needs an access token and workspaces id for authorization. They can be obtained from:
# - [Gradient UI](https://auth.gradient.ai/login), or
# - [Gradient CLI](https://docs.gradient.ai/docs/cli-quickstart) with `gradient env` command.

import os

os.environ["GRADIENT_ACCESS_TOKEN"] = "{GRADIENT_ACCESS_TOKEN}"
os.environ["GRADIENT_WORKSPACE_ID"] = "{GRADIENT_WORKSPACE_ID}"

from llama_index.llms import GradientBaseModelLLM

# NOTE: we use a base model here, you can as well insert your fine-tuned model.
llm = GradientBaseModelLLM(
    base_model_slug="llama2-7b-chat",
    max_tokens=400,
)

# #### Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Load Documents

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print(f"Loaded {len(documents)} document(s).")

# ### Configure Gradient embeddings

from llama_index import ServiceContext
from llama_index.embeddings import GradientEmbedding

embed_model = GradientEmbedding(
    gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large",
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model=embed_model
)

# ### Setup and Query Index

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = index.as_query_engine()

response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)
print(response)

