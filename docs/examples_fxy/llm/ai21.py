#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/ai21.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # AI21

# ## Basic Usage

# #### Call `complete` with a prompt

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import AI21

api_key = "Your api key"
resp = AI21(api_key=api_key).complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, AI21

messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = AI21(api_key=api_key).chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

print(resp)

# ## Configure Model

from llama_index.llms import AI21

llm = AI21(model="j2-mid", api_key=api_key)

resp = llm.complete("Paul Graham is ")

print(resp)

# ## Set API Key at a per-instance level
# If desired, you can have separate LLM instances use separate API keys.

from llama_index.llms import AI21

llm_good = AI21(api_key=api_key)
llm_bad = AI21(model="j2-mid", api_key="BAD_KEY")

resp = llm_good.complete("Paul Graham is ")
print(resp)

resp = llm_bad.complete("Paul Graham is ")
print(resp)

