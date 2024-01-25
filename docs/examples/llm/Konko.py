#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/llm/Konko.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Konko

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

from llama_index.llms import Konko
from llama_index.llms import ChatMessage

# ## Call `chat` with ChatMessage List
# You need to either set env var `KONKO_API_KEY` or set konko_api_key in the class constructor

# import os
# os.environ['KONKO_API_KEY'] = '<your-api-key>'

llm = Konko(konko_api_key="<your-api-key>")

message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)

# ## Call `chat` with OpenAI Models
# You need to either set env var `OPENAI_API_KEY` or set openai_api_key in the class constructor

# import os
# os.environ['OPENAI_API_KEY'] = '<your-api-key>'

llm = Konko(model="gpt-3.5-turbo", openai_api_key="<your-api-key>")

message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)

# ### Streaming

message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message], max_tokens=1000)
for r in resp:
    print(r.delta, end="")

# ## Call `complete` with Prompt

resp = llm.complete("Tell me a joke")
print(resp)

resp = llm.stream_complete("Tell me a story in 250 words", max_tokens=1000)
for r in resp:
    print(r.delta, end="")

# ## Model Configuration

llm = Konko(model="meta-llama/Llama-2-13b-chat-hf")

resp = llm.stream_complete(
    "Show me the c++ code to send requests to HTTP Server", max_tokens=1000
)
for r in resp:
    print(r.delta, end="")

