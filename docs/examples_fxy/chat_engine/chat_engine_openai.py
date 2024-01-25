#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chat Engine - OpenAI Agent Mode

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Download Data

#("mkdir -p 'data/paul_graham/'")
#("wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'")

# ### Get started in 5 lines of code

# Load data and build index

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI

# Necessary to use the latest OpenAI models that support function calling API
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo-0613")
)
data = SimpleDirectoryReader(input_dir="../data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Configure chat engine

chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

# Chat with your data

response = chat_engine.chat("Hi")
print(response)

response = chat_engine.chat(
    "Use the tool to answer: Who did Paul Graham hand over YC to?"
)
print(response)

response = chat_engine.stream_chat(
    "Use the tool to answer: Who did Paul Graham hand over YC to?"
)
print(response)

# ### Force chat engine to query the index

# NOTE: this is a feature unique to the "openai" chat mode (which uses the `OpenAIAgent` under the hood).

response = chat_engine.chat(
    "What did Paul Graham do growing up?", tool_choice="query_engine_tool"
)

print(response)

