#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/openrouter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenRouter

# OpenRouter provides a standardized API to access many LLMs at the best price offered. You can find out more on their [homepage](https://openrouter.ai).
# 
# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import OpenRouter
from llama_index.llms import ChatMessage

# ## Call `chat` with ChatMessage List
# You need to either set env var `OPENROUTER_API_KEY` or set api_key in the class constructor

# import os
# os.environ['OPENROUTER_API_KEY'] = '<your-api-key>'

llm = OpenRouter(
    api_key="<your-api-key>",
    max_tokens=256,
    context_window=4096,
    model="gryphe/mythomax-l2-13b",
)

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

# View options at https://openrouter.ai/models
# This example uses Mistral's MoE, Mixtral:
llm = OpenRouter(model="mistralai/mixtral-8x7b-instruct")

resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)

