#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/anthropic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Anthropic

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Call `complete` with a prompt

from llama_index.llms import Anthropic

# To customize your API key, do this
# otherwise it will lookup ANTHROPIC_API_KEY from your env variable
# llm = Anthropic(api_key="<api_key>")
llm = Anthropic()

resp = llm.complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, Anthropic

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = Anthropic().chat(messages)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint 

from llama_index.llms import Anthropic

llm = Anthropic()
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

from llama_index.llms import Anthropic

llm = Anthropic()
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

# ## Configure Model

from llama_index.llms import Anthropic

llm = Anthropic(model="claude-instant-1")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

# ## Async

from llama_index.llms import Anthropic

llm = Anthropic()
resp = await llm.acomplete("Paul Graham is ")

print(resp)

