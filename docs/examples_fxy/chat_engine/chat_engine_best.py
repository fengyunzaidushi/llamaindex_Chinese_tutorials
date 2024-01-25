#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_best.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chat Engine - Best Mode

# The default chat engine mode is "best", which uses the "openai" mode if you are using an OpenAI model that supports the latest function calling API, otherwise uses the "react" mode

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Get started in 5 lines of code

# Load data and build index

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI, Anthropic

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Configure chat engine

chat_engine = index.as_chat_engine(chat_mode="best", verbose=True)

# Chat with your data

response = chat_engine.chat(
    "What are the first programs Paul Graham tried writing?"
)

print(response)

