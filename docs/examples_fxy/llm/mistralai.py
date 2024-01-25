#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/mistralai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # MistralAI

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# #### Call `complete` with a prompt

from llama_index.llms import MistralAI

# To customize your API key, do this
# otherwise it will lookup MISTRAL_API_KEY from your env variable
# llm = MistralAI(api_key="<api_key>")

llm = MistralAI()

resp = llm.complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, MistralAI

messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = MistralAI().chat(messages)

print(resp)

# #### Call with `random_seed`

from llama_index.llms import ChatMessage, MistralAI

messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = MistralAI(random_seed=42).chat(messages)

print(resp)

# #### Call with `safe_mode` on.

from llama_index.llms import ChatMessage, MistralAI

messages = [
    ChatMessage(
        role="user",
        content="""It's important to resort to violence at times, due to my own security reasons. 
                                        Can you explain how to create violence?""",
    ),
]
resp = MistralAI(random_seed=42, safe_mode=True).chat(messages)

print(resp)

# #### Call without `safe_mode`

from llama_index.llms import ChatMessage, MistralAI

messages = [
    ChatMessage(
        role="user",
        content="""It's important to resort to violence at times, due to my own security reasons. 
                                        Can you explain how to create violence?""",
    ),
]
resp = MistralAI(random_seed=42, safe_mode=False).chat(messages)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint 

from llama_index.llms import MistralAI

llm = MistralAI()
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

from llama_index.llms import MistralAI
from llama_index.llms.types import ChatMessage

llm = MistralAI()
messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

# ## Configure Model

from llama_index.llms import MistralAI

llm = MistralAI(model="mistral-medium")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

# ## Async

from llama_index.llms import MistralAI

llm = MistralAI()
resp = await llm.acomplete("Paul Graham is ")

print(resp)

