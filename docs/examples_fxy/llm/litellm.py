#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/litellm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # LiteLLM
# 
# ### LiteLLM supports 100+ LLM APIs (Anthropic, Replicate, Huggingface, TogetherAI, Cohere, etc.). [Complete List](https://docs.litellm.ai/docs/providers)

# #### Call `complete` with a prompt

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

import os
from llama_index.llms import LiteLLM, ChatMessage

# set env variable
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

message = ChatMessage(role="user", content="Hey! how's it going?")

# openai call
llm = LiteLLM("gpt-3.5-turbo")
chat_response = llm.chat([message])

# cohere call
llm = LiteLLM("command-nightly")
chat_response = llm.chat([message])

from llama_index.llms import ChatMessage, LiteLLM

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = LiteLLM("gpt-3.5-turbo").chat(messages)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint 

from llama_index.llms import LiteLLM

llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

from llama_index.llms import LiteLLM

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]

llm = LiteLLM("gpt-3.5-turbo")
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

# ## Async

from llama_index.llms import LiteLLM

llm = LiteLLM("gpt-3.5-turbo")
resp = await llm.acomplete("Paul Graham is ")

print(resp)

