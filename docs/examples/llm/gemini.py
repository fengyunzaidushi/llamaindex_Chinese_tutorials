#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Gemini

# 
# If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the Gemini Python SDK.

#('pip install -q llama-index google-generativeai')

# ## Basic Usage
# 
# You will need to get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey). Once you have one, you can either pass it explicity to the model, or use the `GOOGLE_API_KEY` environment variable.

get_ipython().run_line_magic('env', 'GOOGLE_API_KEY=...')

import os

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# #### Call `complete` with a prompt

from llama_index.llms import Gemini

resp = Gemini().complete("Write a poem about a magic backpack")
print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, Gemini

messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = Gemini().chat(messages)
print(resp)

# ## Streaming

# Using `stream_complete` endpoint

from llama_index.llms import Gemini

llm = Gemini()
resp = llm.stream_complete(
    "The story of Sourcrust, the bread creature, is really interesting. It all started when..."
)

for r in resp:
    print(r.text, end="")

# Using `stream_chat` endpoint

from llama_index.llms import Gemini, ChatMessage

llm = Gemini()
messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")

# ## Using other models
# 
# The [Gemini model site](https://ai.google.dev/models) lists the models that are currently available, along with their capabilities. You can also use the API to find suitable models.

import google.generativeai as genai

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

from llama_index.llms import Gemini

llm = Gemini(model="models/gemini-pro")

resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
print(resp)

# ## Asynchronous API

from llama_index.llms import Gemini

llm = Gemini()

resp = await llm.acomplete("Llamas are famous for ")
print(resp)

resp = await llm.astream_complete("Llamas are famous for ")
async for chunk in resp:
    print(chunk.text, end="")

