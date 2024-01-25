#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/anyscale.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Anyscale

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import Anyscale
from llama_index.llms import ChatMessage

# ## Call `chat` with ChatMessage List
# You need to either set env var `ANYSCALE_API_KEY` or set api_key in the class constructor

# import os
# os.environ['ANYSCALE_API_KEY'] = '<your-api-key>'

llm = Anyscale(api_key="<your-api-key>")

message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)

# ### Streaming

message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")

# ## Call `complete` with Prompt

resp = llm.complete("Tell me a joke")
print(resp)

resp = llm.stream_complete("Tell me a story in 250 words")
for r in resp:
    print(r.delta, end="")

# ## Model Configuration

llm = Anyscale(model="codellama/CodeLlama-34b-Instruct-hf")

resp = llm.complete("Show me the c++ code to send requests to HTTP Server")
print(resp)

