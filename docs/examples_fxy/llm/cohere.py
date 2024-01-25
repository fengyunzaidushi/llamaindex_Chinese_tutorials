#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/cohere.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Cohere

# ## Basic Usage

# #### Call `complete` with a prompt

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import Cohere

api_key = "Your api key"
resp = Cohere(api_key=api_key).complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, Cohere

messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = Cohere(api_key=api_key).chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint 

from llama_index.llms import OpenAI

llm = Cohere(api_key=api_key)
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

# Using `stream_chat` endpoint

from llama_index.llms import OpenAI

llm = Cohere(api_key=api_key)
messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

for r in resp:
    print(r.delta, end="")

# ## Configure Model

from llama_index.llms import Cohere

llm = Cohere(model="command", api_key=api_key)

resp = llm.complete("Paul Graham is ")

print(resp)

# ## Async

from llama_index.llms import Cohere

llm = Cohere(model="command", api_key=api_key)

resp = await llm.acomplete("Paul Graham is ")

print(resp)

resp = await llm.astream_complete("Paul Graham is ")

async for delta in resp:
    print(delta.delta, end="")

# ## Set API Key at a per-instance level
# If desired, you can have separate LLM instances use separate API keys.

from llama_index.llms import Cohere

llm_good = Cohere(api_key=api_key)
llm_bad = Cohere(model="command", api_key="BAD_KEY")

resp = llm_good.complete("Paul Graham is ")
print(resp)

resp = llm_bad.complete("Paul Graham is ")
print(resp)

