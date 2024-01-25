#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/chat_engine/chat_engine_personality.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Chat Engine with a Personality ✨

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

#('pip install llama-index')

# ## Default

from llama_index.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults()
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
print(response)

# ## Shakespeare

from llama_index.chat_engine import SimpleChatEngine
from llama_index.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
print(response)

# ## Marketing

from llama_index.chat_engine import SimpleChatEngine
from llama_index.prompts.system import MARKETING_WRITING_ASSISTANT

chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=MARKETING_WRITING_ASSISTANT
)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
print(response)

# ## IRS Tax

from llama_index.chat_engine import SimpleChatEngine
from llama_index.prompts.system import IRS_TAX_CHATBOT

chat_engine = SimpleChatEngine.from_defaults(system_prompt=IRS_TAX_CHATBOT)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
print(response)

