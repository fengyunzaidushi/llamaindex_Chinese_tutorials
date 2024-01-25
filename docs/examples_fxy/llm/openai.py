#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # OpenAI

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Basic Usage

# #### Call `complete` with a prompt

from llama_index.llms import OpenAI

resp = OpenAI().complete("Paul Graham is ")

print(resp)

# #### Call `chat` with a list of messages

from llama_index.llms import ChatMessage, OpenAI

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = OpenAI().chat(messages)

print(resp)

# ## Streaming

# Using `stream_complete` endpoint

from llama_index.llms import OpenAI

llm = OpenAI()
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    print(r.delta, end="")

# Using `stream_chat` endpoint

from llama_index.llms import OpenAI, ChatMessage

llm = OpenAI()
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

from llama_index.llms import OpenAI

llm = OpenAI(model="text-davinci-003")

resp = llm.complete("Paul Graham is ")

print(resp)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

print(resp)

# ## Function Calling

from pydantic import BaseModel
from llama_index.llms.openai_utils import to_openai_tool

class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str

song_fn = to_openai_tool(Song)

from llama_index.llms import OpenAI

response = OpenAI().complete("Generate a song", tools=[song_fn])
tool_calls = response.additional_kwargs["tool_calls"]
print(tool_calls)

# ## Async

from llama_index.llms import OpenAI

llm = OpenAI(model="text-davinci-003")

resp = await llm.acomplete("Paul Graham is ")

print(resp)

resp = await llm.astream_complete("Paul Graham is ")

async for delta in resp:
    print(delta.delta, end="")

# ## Set API Key at a per-instance level
# If desired, you can have separate LLM instances use separate API keys.

from llama_index.llms import OpenAI

llm = OpenAI(model="text-davinci-003", api_key="BAD_KEY")
resp = OpenAI().complete("Paul Graham is ")
print(resp)

