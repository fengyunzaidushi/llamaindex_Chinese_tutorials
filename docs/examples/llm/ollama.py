#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/Ollama.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Ollama - Llama 2 7B

# ## Setup
# First, follow the [readme](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance.
# 
# When the Ollama app is running on your local machine:
# - All of your local models are automatically served on localhost:11434
# - Select your model when setting llm = Ollama(..., model="<model family>:<version>")
# - If you set llm = Ollama(..., model="<model family") without a version it will simply look for latest

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import Ollama

llm = Ollama(model="llama2")

resp = llm.complete("Who is Paul Graham?")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

print(resp)

# ### Streaming

# Using `stream_complete` endpoint 

response = llm.stream_complete("Who is Paul Graham?")

for r in response:
    print(r.delta, end="")

# Using `stream_chat` endpoint

from llama_index.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

