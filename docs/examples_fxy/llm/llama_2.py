#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Replicate - Llama 2 13B

# ## Setup

# Make sure you have the `REPLICATE_API_TOKEN` environment variable set.  
# If you don't have one yet, go to https://replicate.com/ to obtain one.  

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os

os.environ["REPLICATE_API_TOKEN"] = "<your API key>"

# ## Basic Usage

# We showcase the "llama13b-v2-chat" model, which you can play with directly at: https://replicate.com/a16z-infra/llama13b-v2-chat

from llama_index.llms import Replicate

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
)

# #### Call `complete` with a prompt

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

# ## Configure Model

from llama_index.llms import Replicate

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    temperature=0.9,
    context_window=32,
)

resp = llm.complete("Who is Paul Graham?")

print(resp)

