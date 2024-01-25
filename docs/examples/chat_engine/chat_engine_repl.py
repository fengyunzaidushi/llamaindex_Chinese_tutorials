#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_repl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chat Engine - Simple Mode REPL

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ### Get started in 3 lines of code

# Using GPT3 ("text-davinci-003")

from llama_index.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults()
chat_engine.chat_repl()

# ### Customize LLM

# Use ChatGPT ("gpt-3.5-turbo")

from llama_index.llms import OpenAI
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0.0, model="gpt-3.5-turbo")
)

from llama_index.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults(service_context=service_context)
chat_engine.chat_repl()

# ## Streaming Support

from llama_index.llms import OpenAI
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0.0, model="gpt-3.5-turbo-0613")
)

from llama_index.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults(service_context=service_context)

response = chat_engine.stream_chat(
    "Write me a poem about raining cats and dogs."
)
for token in response.response_gen:
    print(token, end="")

